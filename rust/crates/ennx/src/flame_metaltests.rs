use super::*;

fn tiny() -> FlameConfig {
    FlameConfig {
        layers: 3,
        width: 12,
        heads: 3,
        vocab: 19,
        dense_width: 15,
        expert_width: 9,
        shared_width: 11,
        experts: 5,
        top_k: 3,
        context: 140,
        epsilon: 1e-6,
        rope_base: 10000.0,
    }
}

fn fixture(c: FlameConfig) -> Vec<u16> {
    let layout = Layout::new(c).unwrap();
    let mut bits = vec![0u16; layout.len];
    let mut seed = 8123u32;
    for (name, &(offset, length)) in &layout.tensors {
        for v in &mut bits[offset..offset + length] {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let x = if name.contains("norm") {
                0.8 + (seed >> 8) as f32 / (1 << 24) as f32 * 0.4
            } else {
                ((seed >> 8) as f32 / (1 << 24) as f32 - 0.5) * 0.3
            };
            *v = ((x.to_bits() + 0x7fff + ((x.to_bits() >> 16) & 1)) >> 16) as u16;
        }
    }
    bits
}

fn close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && (a - b).abs() <= tolerance * (1.0 + b.abs()),
            "element {i}: Metal {a}, reference {b}, tolerance {tolerance}"
        );
    }
}

// Independent scalar FP32 forward, intentionally without GPU tiling or routing
// dispatch. It indexes QKV directly by head and evaluates each selected expert.
fn cpu_logits(c: FlameConfig, bits: &[u16], tokens: &[i32]) -> Vec<f32> {
    let layout = Layout::new(c).unwrap();
    let w: Vec<f32> = bits
        .iter()
        .map(|&x| f32::from_bits(u32::from(x) << 16))
        .collect();
    let h = c.width as usize;
    let n = tokens.len();
    let linear = |x: &[f32], offset: usize, inside: usize, outside: usize| -> Vec<f32> {
        x.chunks_exact(inside)
            .flat_map(|row| {
                (0..outside)
                    .map(|o| {
                        row.iter()
                            .zip(&w[offset + o * inside..offset + (o + 1) * inside])
                            .fold(0.0f32, |sum, (&a, &b)| a.mul_add(b, sum))
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    };
    let norm = |x: &[f32], offset: usize| -> Vec<f32> {
        x.chunks_exact(h)
            .flat_map(|row| {
                let inv = (row.iter().map(|&v| v * v).sum::<f32>() / h as f32 + c.epsilon)
                    .sqrt()
                    .recip();
                row.iter()
                    .enumerate()
                    .map(|(i, &v)| (v * inv) * w[offset + i])
                    .collect::<Vec<_>>()
            })
            .collect()
    };
    let mlp = |x: &[f32], first: usize, second: usize, hidden: usize| -> Vec<f32> {
        let gates = linear(x, first, h, 2 * hidden);
        let activation: Vec<f32> = gates
            .chunks_exact(2 * hidden)
            .flat_map(|row| {
                (0..hidden)
                    .map(|i| (row[i] / (1.0 + (-row[i]).exp())) * row[hidden + i])
                    .collect::<Vec<_>>()
            })
            .collect();
        linear(&activation, second, hidden, h)
    };
    let mut x: Vec<f32> = tokens
        .iter()
        .flat_map(|&t| {
            w[layout.embedding + t as usize * h..layout.embedding + (t as usize + 1) * h]
                .iter()
                .copied()
        })
        .collect();
    for (i, layer) in layout.layers.iter().enumerate() {
        let z = norm(&x, layer.attention_norm);
        let qkv = linear(&z, layer.qkv, h, 3 * h);
        let d = h / c.heads as usize;
        let mut q = vec![0.0f32; n * h];
        let mut k = q.clone();
        let mut v = q.clone();
        for pos in 0..n {
            for head in 0..c.heads as usize {
                for j in 0..d {
                    let src = pos * 3 * h + head * 3 * d;
                    let dst = pos * h + head * d + j;
                    let partner = (j + d / 2) % d;
                    let sign = if j < d / 2 { -1.0 } else { 1.0 };
                    let angle =
                        pos as f32 * c.rope_base.powf(-((2 * (j % (d / 2))) as f32) / d as f32);
                    q[dst] = qkv[src + j] * angle.cos() + sign * qkv[src + partner] * angle.sin();
                    k[dst] = qkv[src + d + j] * angle.cos()
                        + sign * qkv[src + d + partner] * angle.sin();
                    v[dst] = qkv[src + 2 * d + j];
                }
            }
        }
        let mut attended = vec![0.0f32; n * h];
        for pos in 0..n {
            for head in 0..c.heads as usize {
                let mut scores: Vec<f32> = (0..=pos)
                    .map(|past| {
                        (0..d).fold(0.0f32, |sum, j| {
                            q[pos * h + head * d + j].mul_add(k[past * h + head * d + j], sum)
                        }) / (d as f32).sqrt()
                    })
                    .collect();
                let peak = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                for s in &mut scores {
                    *s = (*s - peak).exp();
                }
                let sum = scores.iter().sum::<f32>();
                for s in &mut scores {
                    *s /= sum;
                }
                for j in 0..d {
                    attended[pos * h + head * d + j] = scores
                        .iter()
                        .enumerate()
                        .fold(0.0f32, |sum, (past, &score)| {
                            score.mul_add(v[past * h + head * d + j], sum)
                        });
                }
            }
        }
        let update = linear(&attended, layer.projection, h, h);
        for (x, a) in x.iter_mut().zip(update) {
            *x += a;
        }
        let z = norm(&x, layer.mlp_norm);
        let mut update = mlp(
            &z,
            layer.first,
            layer.second,
            if i == 0 {
                c.dense_width
            } else {
                c.shared_width
            } as usize,
        );
        if i != 0 {
            let logits = linear(&z, layer.router, h, c.experts as usize);
            for row in 0..n {
                let values = &logits[row * c.experts as usize..(row + 1) * c.experts as usize];
                let peak = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut probs: Vec<f32> = values.iter().map(|&x| (x - peak).exp()).collect();
                let sum = probs.iter().sum::<f32>();
                for p in &mut probs {
                    *p /= sum;
                }
                let mut ids: Vec<usize> = (0..c.experts as usize).collect();
                ids.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap().then(a.cmp(&b)));
                let mut routed = vec![0.0f32; h];
                for &e in &ids[..c.top_k as usize] {
                    let hidden = c.expert_width as usize;
                    let expert = mlp(
                        &z[row * h..(row + 1) * h],
                        layer.expert_first + e * 2 * hidden * h,
                        layer.expert_second + e * h * hidden,
                        hidden,
                    );
                    for j in 0..h {
                        routed[j] += probs[e] * expert[j];
                    }
                }
                for j in 0..h {
                    update[row * h + j] += routed[j];
                }
            }
        }
        for (x, update) in x.iter_mut().zip(update) {
            *x += update;
        }
    }
    linear(
        &norm(&x, layout.final_norm),
        layout.output,
        h,
        c.vocab as usize,
    )
}

#[test]
fn layout_workspace() {
    let c = FlameConfig::default();
    c.validate(c.context).unwrap();
    let layout = Layout::new(c).unwrap();
    assert_eq!(layout.tensors.len(), 81);
    assert_eq!(layout.len, 1_300_024_320);
    assert_eq!(weight_count(c).unwrap(), layout.len);
    assert_eq!(layout.final_norm, 0);
    let mut offset = 0;
    for &(start, len) in layout.tensors.values() {
        assert_eq!(start, offset);
        offset += len;
    }
    assert_eq!(offset, layout.len);
    let sizes = workspace_sizes(c, 2048).unwrap();
    assert!(sizes.iter().sum::<usize>() < 600 * 1024 * 1024);
    assert_eq!(sizes[W::Logits as usize], 128 * c.vocab as usize * 4);
    let c = FlameConfig {
        layers: 12,
        ..tiny()
    };
    let layout = Layout::new(c).unwrap();
    assert!(layout.layers[10].qkv < layout.layers[2].qkv);
    assert_eq!(weight_count(c).unwrap(), layout.len);
}

#[test]
fn overflow_guards() {
    for c in [
        FlameConfig { heads: 0, ..tiny() },
        FlameConfig {
            width: 11,
            ..tiny()
        },
        FlameConfig { heads: 4, ..tiny() },
        FlameConfig { top_k: 6, ..tiny() },
        FlameConfig {
            epsilon: f32::NAN,
            ..tiny()
        },
        FlameConfig {
            rope_base: 1.0,
            ..tiny()
        },
    ] {
        assert!(c.validate(2).is_err());
    }
    assert!(tiny().validate(0).is_err());
    assert!(tiny().validate(141).is_err());
    assert!(product(&[usize::MAX, 2]).is_err());
    assert!(workspace_sizes(FlameConfig::default(), u32::MAX).is_err());
    assert!(upload(&[]).is_err());
    for bits in [0x7f80u16, 0xff80, 0x7fc0, 0xffff] {
        assert!(upload(&[bits]).is_err());
    }
    assert_eq!(size_of::<FlameConfig>(), 48);
    assert_eq!(size_of::<Shape>(), 40);
    assert_eq!(size_of::<Matmul>(), 40);
}

#[test]
fn scalar_parity() {
    let c = tiny();
    let bits = fixture(c);
    let mut engine = FlameEvaluator::new(c, 140).unwrap();
    let weights = engine.upload(&bits).unwrap();
    for n in [1, 2, 7, 8, 33, 131] {
        let tokens: Vec<i32> = (0..n)
            .map(|i| ((i * 7 + 2) % c.vocab as usize) as i32)
            .collect();
        let reference = cpu_logits(c, &bits, &tokens);
        let logits = engine.logits(&weights, &tokens).unwrap();
        close(&logits, &reference, 3e-5);
        if n > 1 {
            let mask: Vec<bool> = (0..n).map(|i| i % 3 == 1 || i == n - 1).collect();
            let loss = engine
                .losses(&weights, &[tokens.clone()], &[mask.clone()])
                .unwrap()[0];
            let mut expected = 0.0f32;
            let mut scored = 0;
            for t in 1..n {
                if mask[t] {
                    let values = &reference[(t - 1) * c.vocab as usize..t * c.vocab as usize];
                    let peak = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    expected += values.iter().map(|&v| (v - peak).exp()).sum::<f32>().ln() + peak
                        - values[tokens[t] as usize];
                    scored += 1;
                }
            }
            close(&[loss], &[expected / scored as f32], 3e-5);
        }
    }
    let tokens = vec![1, 2, 3, 4];
    let first = engine.logits(&weights, &tokens).unwrap();
    let mut extended = tokens.clone();
    extended.extend([8, 9, 10, 11, 12]);
    close(
        &first,
        &engine.logits(&weights, &extended).unwrap()[..first.len()],
        3e-5,
    );
    let rows = vec![tokens, vec![5, 6]];
    let masks = vec![vec![false, true, false, true], vec![false, true]];
    let batch = engine.losses(&weights, &rows, &masks).unwrap();
    for i in 0..2 {
        close(
            &batch[i..i + 1],
            &engine
                .losses(&weights, &rows[i..i + 1], &masks[i..i + 1])
                .unwrap(),
            1e-6,
        );
    }
}

#[test]
fn logits_workspace() {
    autoreleasepool(|| {
        let c = FlameConfig {
            width: 80,
            heads: 2,
            vocab: 71,
            layers: 2,
            ..tiny()
        };
        let mut engine = FlameEvaluator::new(c, c.context).unwrap();
        let weights = engine.upload(&fixture(c)).unwrap();
        let workspace_bytes = engine.workspace_bytes();
        for n in [
            1, 2, 7, 8, 9, 31, 32, 33, 127, 128, 129, 131, 135, 136, 140, 1,
        ] {
            let mut tokens: Vec<i32> = (0..n).map(|i| (i * 13 % c.vocab) as i32).collect();
            tokens[n as usize - 1] = c.vocab as i32 - 1;
            let full = engine.logits(&weights, &tokens).unwrap();
            // Leave scored masks and losses in scratch before the next call.
            engine
                .losses(&weights, &[vec![0, 1]], &[vec![false, true]])
                .unwrap();
            let sentinel = vec![123.0f32; LOGIT_ROWS * c.vocab as usize];
            engine.write(W::Logits, &sentinel);
            let next = engine.next_logits(&weights, &tokens).unwrap();
            assert_eq!(next.len(), c.vocab as usize);
            let final_row = &full[full.len() - next.len()..];
            assert!(next.iter().all(|x| x.is_finite()));
            assert_eq!(
                next.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                final_row.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                "final-row bit parity at sequence length {n}"
            );
            let scratch = engine.read::<f32>(W::Logits, sentinel.len());
            assert_eq!(&scratch[next.len()..], &sentinel[next.len()..]);
            assert_eq!(engine.workspace_bytes(), workspace_bytes);
        }
    });
}

#[test]
fn logits_recovery() {
    autoreleasepool(|| {
        let c = tiny();
        let mut engine = FlameEvaluator::new(c, 8).unwrap();
        let bits = fixture(c);
        let weights = engine.upload(&bits).unwrap();
        let wrong = upload(&[0]).unwrap();
        assert!(engine.next_logits(&wrong, &[0]).is_err());
        for tokens in [vec![], vec![-1], vec![c.vocab as i32], vec![0; 9]] {
            assert!(engine.next_logits(&weights, &tokens).is_err());
        }
        let runtime = Runtime::shared().unwrap();
        for bad in [0x7fc0, 0x7f80, 0xff80] {
            let mut invalid_bits = bits.clone();
            // Bypass upload validation as a borrowed/search-produced buffer can.
            invalid_bits[engine.layout.output] = bad;
            let invalid = runtime.buffer_with(&invalid_bits);
            for tokens in [vec![0], vec![0; 8]] {
                assert!(
                    engine
                        .next_logits(&invalid, &tokens)
                        .unwrap_err()
                        .contains("nonfinite")
                );
                let full = engine.logits(&weights, &tokens).unwrap();
                assert_eq!(
                    engine.next_logits(&weights, &tokens).unwrap(),
                    full[full.len() - c.vocab as usize..]
                );
            }
        }
    });
}

#[test]
fn router_ties() {
    let c = tiny();
    let mut engine = FlameEvaluator::new(c, 8).unwrap();
    let mut bits = fixture(c);
    assert!(engine.upload(&bits[..bits.len() - 1]).is_err());
    let wrong = upload(&[0]).unwrap();
    assert!(engine.logits(&wrong, &[0]).is_err());
    for tokens in [vec![], vec![-1], vec![19], vec![0; 9]] {
        assert!(engine.check_tokens(&tokens).is_err());
    }
    for masks in [vec![], vec![true, true], vec![false, false], vec![false]] {
        assert!(engine.check_batch(&[vec![0, 1]], &[masks]).is_err());
    }
    assert!(engine.check_batch(&[], &[]).is_err());
    assert!(engine.check_batch(&[vec![0]], &[vec![false]]).is_err());
    for layer in &engine.layout.layers[1..] {
        bits[layer.router..layer.router + c.experts as usize * c.width as usize].fill(0);
    }
    let weights = engine.upload(&bits).unwrap();
    let tokens = vec![1, 2, 3, 4, 5, 6, 7, 8];
    close(
        &engine.logits(&weights, &tokens).unwrap(),
        &cpu_logits(c, &bits, &tokens),
        3e-5,
    );
    let indices = engine.read::<u32>(W::Indices, tokens.len() * c.top_k as usize);
    for row in indices.chunks_exact(3) {
        assert_eq!(row, &[0, 1, 2]);
    }
    close(
        &engine.read::<f32>(W::Probs, indices.len()),
        &vec![0.2; indices.len()],
        1e-6,
    );

    // A search-produced buffer is deliberately not scanned on the CPU. The
    // forward boundary must reject bad logits even on the final unscored row.
    let runtime = Runtime::shared().unwrap();
    bits[engine.layout.output] = 0x7fc0;
    let invalid = runtime.buffer_with(&bits);
    assert!(
        engine
            .logits(&invalid, &[0])
            .unwrap_err()
            .contains("nonfinite")
    );
    assert!(
        engine
            .losses(&invalid, &[vec![0, 1, 2]], &[vec![false, true, false]])
            .is_err()
    );
}

#[test]
fn tiles_bf16() {
    autoreleasepool(|| {
        let c = FlameConfig {
            width: 80,
            heads: 2,
            vocab: 71,
            dense_width: 93,
            expert_width: 37,
            shared_width: 61,
            layers: 2,
            ..tiny()
        };
        let mut engine = FlameEvaluator::new(c, 35).unwrap();
        let bits = fixture(c);
        let weights = engine.upload(&bits).unwrap();
        let tokens: Vec<i32> = (0..35).map(|i| (i * 13) % 71).collect();
        close(
            &engine.logits(&weights, &tokens).unwrap(),
            &cpu_logits(c, &bits, &tokens),
            5e-5,
        );

        let runtime = Runtime::shared().unwrap();
        let inside = 73;
        let outside = 67;
        let mut bits = vec![0u16; 3 + inside * outside];
        // These BF16 values cannot be represented in FP16 (overflow/underflow).
        for (i, b) in bits[3..].iter_mut().enumerate() {
            *b = if (i / inside) % 2 == 0 {
                0x5015
            } else {
                0x0da2
            };
        }
        let weights = runtime.buffer_with(&bits);
        for rows in [1, 7, 8, 35] {
            let x: Vec<f32> = (0..3 + rows * inside)
                .map(|i| (1.0 + (i % 17) as f32) * 1e20)
                .collect();
            let input = runtime.buffer_with(&x);
            let output = runtime.buffer::<f32>(rows * outside);
            let command = runtime.queue.new_command_buffer();
            engine.linear(
                command,
                &input,
                12,
                &weights,
                3,
                &output,
                rows as u32,
                inside as u32,
                outside as u32,
            );
            finish(command).unwrap();
            let expected: Vec<f32> = (0..rows)
                .flat_map(|r| {
                    let x = &x;
                    let bits = &bits;
                    (0..outside).map(move |o| {
                        (0..inside).fold(0.0f32, |s, k| {
                            x[3 + r * inside + k].mul_add(
                                f32::from_bits(u32::from(bits[3 + o * inside + k]) << 16),
                                s,
                            )
                        })
                    })
                })
                .collect();
            let actual = unsafe {
                std::slice::from_raw_parts(output.contents().cast::<f32>(), rows * outside)
            };
            for (&actual, &expected) in actual.iter().zip(&expected) {
                assert!(
                    actual.is_finite() && (actual - expected).abs() <= 1e-5 * expected.abs(),
                    "BF16 exponent-range result: Metal {actual}, reference {expected}"
                );
            }
        }
    });
}

#[test]
#[ignore = "Loads the original 2.6 GB BF16 checkpoint from ENNX_FLAME_CHECKPOINT or .cache/flame-290m"]
fn checkpoint_reg() {
    use std::io::{Read, Seek, SeekFrom};
    use std::path::PathBuf;

    autoreleasepool(|| {
        let c = FlameConfig::default();
        let mut engine = FlameEvaluator::new(c, 32).unwrap();
        let directory = std::env::var_os("ENNX_FLAME_CHECKPOINT")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(".cache/flame-290m"));
        memory_guard(
            &engine.runtime,
            engine.weights_len() as u64 * 2,
            engine.weights_len() as u64 * 2,
        )
        .unwrap();
        let weights = engine.runtime.buffer::<u16>(engine.weights_len());
        assert!(!weights.contents().is_null());
        assert!(cfg!(target_endian = "little"));
        // checkpoint.py writes one tensor per numbered file in sorted name
        // order. Read its length-prefixed binary payload, checking exact size.
        // This fixture loader is specific to that exporter, not a general
        // safetensors loader, and never interprets or edits checkpoint metadata.
        for (index, (_name, &(offset, count))) in engine.layout.tensors.iter().enumerate() {
            let mut file =
                std::fs::File::open(directory.join(format!("{index:03}.safetensors"))).unwrap();
            let mut prefix = [0u8; 8];
            file.read_exact(&mut prefix).unwrap();
            let header = u64::from_le_bytes(prefix);
            assert!(header < 1024 * 1024);
            assert_eq!(
                file.metadata().unwrap().len(),
                8 + header + count as u64 * 2
            );
            file.seek(SeekFrom::Start(8 + header)).unwrap();
            let target = unsafe {
                std::slice::from_raw_parts_mut(
                    weights.contents().cast::<u8>().add(offset * 2),
                    count * 2,
                )
            };
            file.read_exact(target).unwrap();
        }
        let rows = vec![
            vec![
                0, 100, 205, 37, 486, 903, 16, 2735, 421, 9821, 110, 14, 6, 508, 3042, 15, 121,
                4788, 91, 37, 2105, 367, 934, 22, 1789, 53, 922, 768, 199, 29, 4017, 17,
            ],
            vec![
                50256, 713, 8120, 309, 431, 521, 930, 371, 7482, 462, 87, 615, 1330, 942, 863, 109,
                206, 7319, 912, 630, 9013, 205, 68, 830, 534, 1012, 293, 991, 654, 324, 710, 13,
            ],
        ];
        // Independent PyTorch/JAX CPU and T4 reference in flame-*-batch-parity.json.
        let greedy = [
            [
                510, 209, 325, 1883, 13, 206, 37, 10019, 206, 206, 27, 325, 38, 84, 66, 25071, 14,
                466, 15, 486, 273, 5395, 90, 15, 187, 77, 89, 19, 199, 187, 66, 15,
            ],
            [
                18, 5849, 15, 452, 15, 4475, 1768, 301, 13, 13, 187, 15, 290, 13, 7250, 79, 85, 15,
                15, 70, 281, 281, 15, 15, 344, 15, 87, 264, 8, 11828, 31, 285,
            ],
        ];
        eprintln!(
            "FLAME device {:?}; BF16 bytes {}; workspace {}",
            memory_info().unwrap(),
            weights.length(),
            engine.workspace_bytes()
        );
        let start = std::time::Instant::now();
        for (row, expected) in rows.iter().zip(greedy) {
            let logits = engine.logits(&weights, row).unwrap();
            let actual: Vec<usize> = logits
                .chunks_exact(c.vocab as usize)
                .map(|row| {
                    row.iter()
                        .enumerate()
                        .max_by(|a, b| a.1.total_cmp(b.1))
                        .unwrap()
                        .0
                })
                .collect();
            assert_eq!(actual, expected);
        }
        let masks = vec![(0..32).map(|i| i != 0).collect::<Vec<_>>(); 2];
        let losses = engine.losses(&weights, &rows, &masks).unwrap();
        let mean = (losses[0] + losses[1]) / 2.0;
        close(&[mean], &[10.38037014], 2e-5);
        eprintln!(
            "FLAME full-model losses {losses:?}; mean {mean}; four forwards {:?}",
            start.elapsed()
        );
    });
}
