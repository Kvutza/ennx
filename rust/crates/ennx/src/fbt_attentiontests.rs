use super::*;
use crate::fbt::InputNorm;
use metal::objc::rc::autoreleasepool;

fn key(candidate: u64) -> CacheKey {
    CacheKey {
        candidate,
        sequence: 3,
        pass: 0,
    }
}
fn bf16(x: f32) -> u16 {
    let b = x.to_bits();
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}
fn rounded(x: f64) -> f64 {
    f32::from_bits(u32::from(bf16(x as f32)) << 16) as f64
}
fn read(b: &BufferRef, n: usize) -> Vec<f32> {
    unsafe { std::slice::from_raw_parts(b.contents().cast::<f32>(), n).to_vec() }
}
fn finish(command: &CommandBufferRef) {
    command.commit();
    command.wait_until_completed();
    assert_eq!(command.status(), Status::Completed);
}
fn config(window: Option<u32>) -> AttentionConfig {
    AttentionConfig {
        heads: 16,
        kv_heads: 8,
        head_dim: 96,
        capacity: 9,
        window,
        qk_norm: InputNorm::UnitRms { epsilon: 1e-5 },
        rope_base: 10000.0,
        score_scale: 1.0 / 96.0f32.sqrt(),
    }
}

// Independent f64 dense oracle. K/V are rounded to match the declared cache dtype.
fn reference(c: AttentionConfig, q: &[f32], k: &[f32], v: &[f32], gates: &[f32]) -> Vec<f64> {
    let d = c.head_dim as usize;
    let h = c.heads as usize;
    let kh = c.kv_heads as usize;
    let n = q.len() / (h * d);
    let rotate = |input: &[f32], heads: usize, quantize: bool| {
        let mut out = vec![0.0; input.len()];
        for t in 0..n {
            for head in 0..heads {
                let base = (t * heads + head) * d;
                let src = &input[base..base + d];
                let scale = match c.qk_norm {
                    InputNorm::None => 1.0,
                    InputNorm::UnitRms { epsilon } => {
                        (src.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() / d as f64
                            + epsilon as f64)
                            .sqrt()
                            .recip()
                    }
                };
                for i in 0..d / 2 {
                    let angle = t as f64 * (c.rope_base as f64).powf(-2.0 * i as f64 / d as f64);
                    let a = src[i] as f64 * scale;
                    let b = src[i + d / 2] as f64 * scale;
                    let x = a * angle.cos() - b * angle.sin();
                    let y = b * angle.cos() + a * angle.sin();
                    out[base + i] = if quantize { rounded(x) } else { x };
                    out[base + i + d / 2] = if quantize { rounded(y) } else { y };
                }
            }
        }
        out
    };
    let q = rotate(q, h, false);
    let k = rotate(k, kh, true);
    let mut output = vec![0.0; n * h * d];
    for t in 0..n {
        for head in 0..h {
            let kv = head / (h / kh);
            let start = c.window.map_or(0, |w| (t + 1).saturating_sub(w as usize));
            let scores: Vec<_> = (start..=t)
                .map(|s| {
                    (0..d)
                        .map(|i| q[(t * h + head) * d + i] * k[(s * kh + kv) * d + i])
                        .sum::<f64>()
                        * c.score_scale as f64
                })
                .collect();
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weights: Vec<_> = scores.iter().map(|s| (s - max).exp()).collect();
            let sum: f64 = weights.iter().sum();
            for i in 0..d {
                output[(t * h + head) * d + i] = (start..=t)
                    .enumerate()
                    .map(|(j, s)| weights[j] * rounded(v[(s * kh + kv) * d + i] as f64))
                    .sum::<f64>()
                    / sum
                    * gates[t * h + head] as f64;
            }
        }
    }
    output
}

#[test]
fn rms_normparity() {
    autoreleasepool(|| {
        let runtime = Runtime::shared().unwrap();
        for width in [1, 33, 1536] {
            let norm = RmsNorm::new(width, 1e-5).unwrap();
            let n = width as usize;
            let input: Vec<_> = (0..2 * n)
                .map(|i| {
                    if i < n {
                        0.0f32
                    } else {
                        (i % 17) as f32 / 8.0 - 1.0
                    }
                })
                .collect();
            let gamma: Vec<_> = (0..n).map(|i| bf16((i % 7) as f32 / 4.0 - 0.5)).collect();
            let x = runtime.buffer_with(&input);
            let g = runtime.buffer_with(&gamma);
            let out = runtime.buffer_with(&vec![123.0f32; 2 * n + 3]);
            let command = runtime.queue.new_command_buffer();
            norm.encode(command, 2, &x, &g, &out).unwrap();
            finish(command);
            let values = read(&out, 2 * n + 3);
            for row in 0..2 {
                let scale = (input[row * n..(row + 1) * n]
                    .iter()
                    .map(|&x| (x as f64).powi(2))
                    .sum::<f64>()
                    / n as f64
                    + 1e-5)
                    .sqrt()
                    .recip();
                for i in 0..n {
                    let expected = input[row * n + i] as f64
                        * scale
                        * f32::from_bits(u32::from(gamma[i]) << 16) as f64;
                    assert!((values[row * n + i] as f64 - expected).abs() < 1e-5);
                }
            }
            assert_eq!(&values[2 * n..], &[123.0; 3]);
            unsafe {
                std::ptr::write_bytes(g.contents(), 0, n * 2);
            }
            let command = runtime.queue.new_command_buffer();
            norm.encode(command, 2, &x, &g, &out).unwrap();
            finish(command);
            assert!(read(&out, 2 * n).iter().all(|&x| x == 0.0));
        }
        assert!(RmsNorm::new(0, 1e-5).is_err());
        assert!(RmsNorm::new(1, f32::NAN).is_err());
    });
}

#[test]
fn attention_chunking() {
    autoreleasepool(|| {
        let runtime = Runtime::shared().unwrap();
        for window in [None, Some(1), Some(3)] {
            for norm in [InputNorm::None, InputNorm::UnitRms { epsilon: 1e-5 }] {
                let c = AttentionConfig {
                    qk_norm: norm,
                    ..config(window)
                };
                let (h, kh, d, n) = (
                    c.heads as usize,
                    c.kv_heads as usize,
                    c.head_dim as usize,
                    7,
                );
                let q: Vec<_> = (0..n * h * d)
                    .map(|i| ((i * 7 % 31) as f32 - 15.0) / 16.0)
                    .collect();
                let k: Vec<_> = (0..n * kh * d)
                    .map(|i| ((i * 11 % 37) as f32 - 18.0) / 32.0)
                    .collect();
                let v: Vec<_> = (0..n * kh * d)
                    .map(|i| ((i * 3 % 41) as f32 - 20.0) / 8.0)
                    .collect();
                let gates: Vec<_> = (0..n * h).map(|i| (i % 3) as f32 / 2.0).collect();
                let expected = reference(c, &q, &k, &v, &gates);
                let mut sequential = None;
                for chunk in [1, 3, 7] {
                    let mut cache = Attention::new(c, chunk, key(0)).unwrap();
                    let mut actual = Vec::new();
                    for start in (0..n).step_by(chunk as usize) {
                        let end = (start + chunk as usize).min(n);
                        let rows = end - start;
                        let qb = runtime.buffer_with(&q[start * h * d..end * h * d]);
                        let kb = runtime.buffer_with(&k[start * kh * d..end * kh * d]);
                        let vb = runtime.buffer_with(&v[start * kh * d..end * kh * d]);
                        let gb = runtime.buffer_with(&gates[start * h..end * h]);
                        let out = runtime.buffer_with(&vec![456.0f32; rows * h * d + 3]);
                        let command = cache.command_buffer();
                        cache
                            .encode(&command, key(0), rows as u32, &qb, &kb, &vb, &gb, &out)
                            .unwrap();
                        finish(&command);
                        cache.check_completed().unwrap();
                        let values = read(&out, rows * h * d + 3);
                        assert_eq!(&values[rows * h * d..], &[456.0; 3]);
                        actual.extend_from_slice(&values[..rows * h * d]);
                    }
                    for (i, (&a, &e)) in actual.iter().zip(&expected).enumerate() {
                        assert!(
                            (a as f64 - e).abs() < 2e-3,
                            "chunk={chunk} window={window:?} norm={norm:?} i={i}: {a} vs {e}"
                        );
                    }
                    assert_eq!(cache.position(), n as u32);
                    if let Some(baseline) = &sequential {
                        assert_eq!(&actual, baseline, "chunking changed GPU attention");
                    } else {
                        sequential = Some(actual);
                    }
                }
                // Future K/V edits cannot alter any earlier output, including in one chunk.
                let mut cache = Attention::new(c, 7, key(0)).unwrap();
                let mut changed_k = k.clone();
                let mut changed_v = v.clone();
                changed_k[(n - 1) * kh * d..].fill(9.0);
                changed_v[(n - 1) * kh * d..].fill(-7.0);
                let qb = runtime.buffer_with(&q);
                let kb = runtime.buffer_with(&changed_k);
                let vb = runtime.buffer_with(&changed_v);
                let gb = runtime.buffer_with(&gates);
                let out = runtime.buffer::<f32>(n * h * d);
                let command = cache.command_buffer();
                cache
                    .encode(&command, key(0), 7, &qb, &kb, &vb, &gb, &out)
                    .unwrap();
                finish(&command);
                for (&a, &e) in read(&out, (n - 1) * h * d).iter().zip(&expected) {
                    assert!((a as f64 - e).abs() < 2e-3);
                }
            }
        }
    });
}

#[test]
fn attention_parity() {
    autoreleasepool(|| {
        let runtime = Runtime::shared().unwrap();
        let n = 67usize;
        for window in [None, Some(1), Some(3), Some(33)] {
            for norm in [InputNorm::None, InputNorm::UnitRms { epsilon: 1e-5 }] {
                let c = AttentionConfig {
                    heads: 4,
                    kv_heads: 2,
                    capacity: n as u32,
                    qk_norm: norm,
                    ..config(window)
                };
                let (h, kh, d) = (c.heads as usize, c.kv_heads as usize, c.head_dim as usize);
                let q: Vec<_> = (0..n * h * d)
                    .map(|i| ((i * 7 % 31) as f32 - 15.0) / 4.0)
                    .collect();
                let k: Vec<_> = (0..n * kh * d)
                    .map(|i| ((i * 11 % 37) as f32 - 18.0) / 8.0)
                    .collect();
                let v: Vec<_> = (0..n * kh * d)
                    .map(|i| ((i * 3 % 41) as f32 - 20.0) / 8.0)
                    .collect();
                let gates: Vec<_> = (0..n * h).map(|i| (i % 5) as f32 / 2.0 - 1.0).collect();
                let expected = reference(c, &q, &k, &v, &gates);
                for chunk in [1, 8, 17, 67] {
                    let mut cache = Attention::new(c, chunk, key(0)).unwrap();
                    let mut actual = Vec::new();
                    for start in (0..n).step_by(chunk as usize) {
                        let end = (start + chunk as usize).min(n);
                        let rows = end - start;
                        let qb = runtime.buffer_with(&q[start * h * d..end * h * d]);
                        let kb = runtime.buffer_with(&k[start * kh * d..end * kh * d]);
                        let vb = runtime.buffer_with(&v[start * kh * d..end * kh * d]);
                        let gb = runtime.buffer_with(&gates[start * h..end * h]);
                        let out = runtime.buffer_with(&vec![456.0f32; rows * h * d + 3]);
                        let command = cache.command_buffer();
                        cache
                            .encode_tiled(&command, key(0), rows as u32, &qb, &kb, &vb, &gb, &out)
                            .unwrap();
                        finish(&command);
                        cache.check_completed().unwrap();
                        let values = read(&out, rows * h * d + 3);
                        assert_eq!(&values[rows * h * d..], &[456.0; 3]);
                        actual.extend_from_slice(&values[..rows * h * d]);
                    }
                    for (i, (&a, &e)) in actual.iter().zip(&expected).enumerate() {
                        assert!(
                            (a as f64 - e).abs() < 2e-3,
                            "tiled chunk={chunk} window={window:?} norm={norm:?} i={i}: {a} vs {e}"
                        );
                    }
                }
            }
        }
    });
}

#[test]
fn state_reset() {
    autoreleasepool(|| {
        let runtime = Runtime::shared().unwrap();
        let c = AttentionConfig {
            heads: 2,
            kv_heads: 1,
            head_dim: 2,
            capacity: 2,
            window: None,
            qk_norm: InputNorm::None,
            rope_base: 10000.0,
            score_scale: 1.0,
        };
        let mut cache = Attention::new(c, 1, key(0)).unwrap();
        let q = runtime.buffer_with(&[1.0f32; 4]);
        let k = runtime.buffer_with(&[1.0f32; 2]);
        let v = runtime.buffer_with(&[2.0f32; 2]);
        let g = runtime.buffer_with(&[1.0f32; 2]);
        let out = runtime.buffer::<f32>(4);
        let command = cache.command_buffer();
        assert!(
            cache
                .encode(&command, key(1), 1, &q, &k, &v, &g, &out)
                .is_err()
        );
        for wrong_key in [
            CacheKey { pass: 1, ..key(0) },
            CacheKey {
                sequence: 4,
                ..key(0)
            },
        ] {
            assert!(
                cache
                    .encode(&command, wrong_key, 1, &q, &k, &v, &g, &out)
                    .is_err()
            );
        }
        assert!(
            cache
                .encode(&command, key(0), 0, &q, &k, &v, &g, &out)
                .is_err()
        );
        assert!(
            cache
                .encode(&command, key(0), 1, &q, &k, &v, &g, &q)
                .is_err()
        );
        let other_queue = runtime.device.new_command_queue();
        assert!(
            cache
                .encode(
                    other_queue.new_command_buffer(),
                    key(0),
                    1,
                    &q,
                    &k,
                    &v,
                    &g,
                    &out
                )
                .is_err()
        );
        cache
            .encode(&command, key(0), 1, &q, &k, &v, &g, &out)
            .unwrap();
        assert!(cache.reset(key(1)).is_err());
        assert!(cache.check_completed().is_err());
        assert!(
            cache
                .encode(&cache.command_buffer(), key(0), 1, &q, &k, &v, &g, &out)
                .is_err()
        );
        cache
            .encode(&command, key(0), 1, &q, &k, &v, &g, &out)
            .unwrap();
        assert!(
            cache
                .encode(&command, key(0), 1, &q, &k, &v, &g, &out)
                .is_err()
        );
        finish(&command);
        cache.check_completed().unwrap();
        assert_eq!(read(&out, 4), vec![2.0; 4]);
        cache.reset(key(1)).unwrap();
        assert_eq!(cache.position(), 0);
        unsafe {
            std::ptr::copy_nonoverlapping([5.0f32; 2].as_ptr(), v.contents().cast(), 2);
        }
        let command = cache.command_buffer();
        assert!(
            cache
                .encode(&command, key(0), 1, &q, &k, &v, &g, &out)
                .is_err()
        );
        cache
            .encode(&command, key(1), 1, &q, &k, &v, &g, &out)
            .unwrap();
        finish(&command);
        assert_eq!(read(&out, 4), vec![5.0; 4]);
        assert!(Attention::new(AttentionConfig { heads: 3, ..c }, 1, key(0)).is_ok());
        assert!(
            Attention::new(
                AttentionConfig {
                    heads: 3,
                    kv_heads: 2,
                    ..c
                },
                1,
                key(0)
            )
            .is_err()
        );
        assert!(
            Attention::new(
                AttentionConfig {
                    window: Some(0),
                    ..c
                },
                1,
                key(0)
            )
            .is_err()
        );
    });
}

#[test]
fn context_indexing() {
    autoreleasepool(|| {
        let runtime = Runtime::shared().unwrap();
        for capacity in [4096, 16384, 32768] {
            let c = AttentionConfig {
                heads: 2,
                kv_heads: 1,
                head_dim: 2,
                capacity,
                window: Some(1),
                qk_norm: InputNorm::UnitRms { epsilon: 1e-5 },
                rope_base: 10000.0,
                score_scale: 1.0,
            };
            let mut cache = Attention::new(c, capacity, key(0)).unwrap();
            let q = runtime.buffer_with(&vec![0.0f32; capacity as usize * 4]);
            let k = runtime.buffer_with(&vec![0.0f32; capacity as usize * 2]);
            let v = runtime.buffer_with(&vec![3.0f32; capacity as usize * 2]);
            let g = runtime.buffer_with(&vec![1.0f32; capacity as usize * 2]);
            let out = runtime.buffer_with(&vec![789.0f32; capacity as usize * 4 + 3]);
            let command = cache.command_buffer();
            cache
                .encode(&command, key(0), capacity, &q, &k, &v, &g, &out)
                .unwrap();
            finish(&command);
            cache.check_completed().unwrap();
            let values = read(&out, capacity as usize * 4 + 3);
            assert!(values[..capacity as usize * 4].iter().all(|&x| x == 3.0));
            assert_eq!(&values[capacity as usize * 4..], &[789.0; 3]);
            assert_eq!(cache.position(), capacity);
        }
    });
}
