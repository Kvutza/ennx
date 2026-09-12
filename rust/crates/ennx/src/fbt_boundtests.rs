//! Optimistic, post-hoc local bounds, not an end-to-end or rounding certificate.
use super::*;

fn half(bits: u16) -> f64 {
    let exponent = (bits >> 10) & 31;
    assert_ne!(exponent, 31);
    let fraction = f64::from(bits & 1023);
    let value = if exponent == 0 {
        fraction * 2.0f64.powi(-24)
    } else {
        (1024.0 + fraction) * 2.0f64.powi(i32::from(exponent) - 25)
    };
    if bits & 0x8000 == 0 { value } else { -value }
}

fn norm(x: &[f64]) -> f64 {
    x.iter().map(|v| v * v).sum::<f64>().sqrt()
}

fn distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn rows(buffer: &BufferRef, half_precision: bool, width: usize) -> Vec<Vec<f64>> {
    [4095, 8191]
        .map(|row| {
            (0..width)
                .map(|col| unsafe {
                    let index = row * width + col;
                    if half_precision {
                        half(*buffer.contents().cast::<u16>().add(index))
                    } else {
                        f64::from(*buffer.contents().cast::<f32>().add(index))
                    }
                })
                .collect()
        })
        .to_vec()
}

fn rms_check(label: &str, a: &[Vec<f64>], b: &[Vec<f64>]) {
    for (sample, (a, b)) in a.iter().zip(b).enumerate() {
        let e = distance(a, b);
        let d = a.len() as f64;
        let denom = ((norm(a) - e).max(0.0).powi(2) / d + 1e-5).sqrt();
        let normalize = |x: &[f64]| {
            let s = (norm(x).powi(2) / d + 1e-5).sqrt();
            x.iter().map(|v| v / s).collect::<Vec<_>>()
        };
        let actual = distance(&normalize(a), &normalize(b));
        let bound = (e / denom).min(2.0 * d.sqrt());
        assert!(actual <= bound + 1e-10 * (1.0 + bound));
        eprintln!(
            "BOUND_RMS label={label} sample={sample} input_relative={:.6} actual={actual:.9} bound={bound:.9} ratio={:.6}",
            e / norm(a).max(1e-30),
            bound / actual.max(1e-30)
        );
    }
}

struct Snapshot {
    x: Vec<Vec<f64>>,
    q: Vec<u16>,
    k: Vec<u16>,
    v: Vec<u16>,
}

fn copy_half(buffer: &BufferRef, count: usize) -> Vec<u16> {
    unsafe { std::slice::from_raw_parts(buffer.contents().cast::<u16>(), count).to_vec() }
}

fn snapshot(p: &Prefill) -> Snapshot {
    let q = unsafe {
        std::slice::from_raw_parts(p.head_major[0].contents().cast::<u16>(), 2 * 16 * 4096 * 96)
    };
    Snapshot {
        x: rows(&p.x, false, 1536),
        q: (0..32)
            .flat_map(|head| {
                q[(head * 4096 + 4095) * 96..(head * 4096 + 4096) * 96]
                    .iter()
                    .copied()
            })
            .collect(),
        k: copy_half(&p.head_major[1], 2 * 8 * 4096 * 96),
        v: copy_half(&p.head_major[2], 2 * 8 * 4096 * 96),
    }
}

fn attention_check(a: &Snapshot, b: &Snapshot, pass: usize, layer: usize) {
    let start = if (layer + 1) % 6 == 0 { 0 } else { 2048 };
    for head in 0..32 {
        let kv = head / 16 * 8 + (head % 16) / 2;
        let qa: Vec<_> = a.q[head * 96..(head + 1) * 96]
            .iter()
            .map(|&x| half(x))
            .collect();
        let qb: Vec<_> = b.q[head * 96..(head + 1) * 96]
            .iter()
            .map(|&x| half(x))
            .collect();
        let (mut za, mut zb, mut value_error, mut value_radius) =
            (Vec::new(), Vec::new(), 0.0f64, 0.0f64);
        let mut value_center = vec![0.0; 96];
        for key in start..4096 {
            let offset = (kv * 4096 + key) * 96;
            for (d, center) in value_center.iter_mut().enumerate() {
                *center += half(a.v[offset + d]) / (4096 - start) as f64;
            }
        }
        for key in start..4096 {
            let offset = (kv * 4096 + key) * 96;
            za.push((0..96).map(|d| qa[d] * half(a.k[offset + d])).sum::<f64>() / 96.0f64.sqrt());
            zb.push((0..96).map(|d| qb[d] * half(b.k[offset + d])).sum::<f64>() / 96.0f64.sqrt());
            let va: Vec<_> = a.v[offset..offset + 96].iter().map(|&x| half(x)).collect();
            let vb: Vec<_> = b.v[offset..offset + 96].iter().map(|&x| half(x)).collect();
            value_error = value_error.max(distance(&va, &vb));
            value_radius = value_radius.max(distance(&va, &value_center));
        }
        let lo = za
            .iter()
            .zip(&zb)
            .map(|(a, b)| b - a)
            .fold(f64::INFINITY, f64::min);
        let hi = za
            .iter()
            .zip(&zb)
            .map(|(a, b)| b - a)
            .fold(f64::NEG_INFINITY, f64::max);
        let softmax = |z: &[f64]| {
            let max = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let exp: Vec<_> = z.iter().map(|v| (v - max).exp()).collect();
            let sum: f64 = exp.iter().sum();
            exp.iter().map(|v| v / sum).collect::<Vec<_>>()
        };
        let pa = softmax(&za);
        let pb = softmax(&zb);
        let tv = pa.iter().zip(&pb).map(|(a, b)| (a - b).abs()).sum::<f64>() * 0.5;
        let tv_bound = ((hi - lo) / 4.0).tanh();
        assert!(tv <= tv_bound + 1e-10);
        let (mut oa, mut ob) = (vec![0.0; 96], vec![0.0; 96]);
        for (key, (pa, pb)) in (start..4096).zip(pa.iter().zip(&pb)) {
            let offset = (kv * 4096 + key) * 96;
            for d in 0..96 {
                oa[d] += pa * half(a.v[offset + d]);
                ob[d] += pb * half(b.v[offset + d]);
            }
        }
        let actual = distance(&oa, &ob);
        let bound = value_error + 2.0 * value_radius * tv_bound;
        assert!(actual <= bound + 1e-10 * (1.0 + bound));
        eprintln!(
            "BOUND_ATTN pass={pass} layer={layer} head={head} logit_range={:.6} tv={tv:.6} tv_bound={tv_bound:.6} actual={actual:.9} bound={bound:.9} ratio={:.6}",
            hi - lo,
            bound / actual.max(1e-30)
        );

        // Exponential reweighting around the incumbent distribution. Retain
        // signed vector sums; use norms only for the Taylor remainder.
        let center: f64 = pa
            .iter()
            .zip(za.iter().zip(&zb))
            .map(|(p, (a, b))| p * (b - a))
            .sum();
        for order in [1, 2] {
            let mut numerator = vec![0.0; 96];
            let mut denominator = 0.0;
            let mut denominator_error = 0.0;
            let mut terms = Vec::new();
            let mut minimum = f64::INFINITY;
            for (i, &probability) in pa.iter().enumerate() {
                let t = zb[i] - za[i] - center;
                minimum = minimum.min(t);
                let polynomial = 1.0 + t + if order == 2 { t * t * 0.5 } else { 0.0 };
                let remainder =
                    t.max(0.0).exp() * t.abs().powi(order + 1) / if order == 1 { 2.0 } else { 6.0 };
                denominator += probability * polynomial;
                denominator_error += probability * remainder;
                let offset = (kv * 4096 + start + i) * 96;
                for d in 0..96 {
                    numerator[d] += probability * polynomial * (half(b.v[offset + d]) - oa[d]);
                }
                terms.push(probability * remainder);
            }
            assert!(denominator > 0.0);
            let estimate: Vec<_> = numerator.iter().map(|n| n / denominator).collect();
            let lower = minimum.exp().max(denominator - denominator_error);
            assert!(lower > 0.0);
            let mut remainder_bound = 0.0;
            for (i, &term) in terms.iter().enumerate() {
                let offset = (kv * 4096 + start + i) * 96;
                let residual_norm = (0..96)
                    .map(|d| (half(b.v[offset + d]) - oa[d] - estimate[d]).powi(2))
                    .sum::<f64>()
                    .sqrt();
                remainder_bound += term * residual_norm;
            }
            remainder_bound /= lower;
            let delta: Vec<_> = ob.iter().zip(&oa).map(|(b, a)| b - a).collect();
            let estimate_error = distance(&estimate, &delta);
            assert!(estimate_error <= remainder_bound + 1e-10 * (1.0 + remainder_bound));
            let total_bound = norm(&estimate) + remainder_bound;
            assert!(actual <= total_bound + 1e-10 * (1.0 + total_bound));
            eprintln!(
                "BOUND_RELATIONAL pass={pass} layer={layer} head={head} order={order} actual={actual:.9} estimate_error={estimate_error:.9} remainder={remainder_bound:.9} remainder_relative={:.6} total_ratio={:.6}",
                remainder_bound / actual.max(1e-30),
                total_bound / actual.max(1e-30)
            );
        }
    }
}

#[test]
#[ignore = "full 4K paired bound audit; post-hoc ideal arithmetic, not a certificate"]
fn paired_bound_audit() {
    metal::objc::rc::autoreleasepool(|| {
        let mut model = Model::new(super::super::bo::full_model_config(), 42).unwrap();
        model.set_optimized(true).unwrap();
        let config = crate::config::parse_turbo_enn_config("version=1\nstudy='end_to_end'\nacquisition='thompson'\nlength_init=0.01\nlength_min=0.0001\nlength_max=0.1\noutput='unused'\nrounds=3\ntarget_round_ms=1000").unwrap();
        let mut search = super::super::bo::model_search(&model, &config);
        search.observe_initial(-1.0, 0.0).unwrap();
        let proposal = search.test_candidate(123, 0).unwrap();
        let p = Prefill::new(&model, 2, 4096).unwrap();
        for sample in 0..2 {
            for i in 0..4096 {
                unsafe {
                    *p.tokens.contents().cast::<u32>().add(sample * 4096 + i) =
                        ((i * 7919 + sample * 29) % 100352) as u32;
                    *p.targets.contents().cast::<u32>().add(sample * 4096 + i) =
                        (((i + 1) * 7919 + sample * 29) % 100352) as u32;
                }
            }
        }
        let mut saved: Vec<Snapshot> = Vec::new();
        let mut starts: Vec<Vec<Vec<f64>>> = Vec::new();
        let mut feedback = Vec::new();
        let mut originals = None;
        for candidate in [false, true] {
            if candidate {
                originals = Some(model.bind_parameter_row(&proposal).unwrap());
            }
            model.ensure_transposed_weights().unwrap();
            for pass in 0..2 {
                let command = model.runtime.queue.new_command_buffer();
                p.start_pass(&model, command, pass).unwrap();
                command.commit();
                command.wait_until_completed();
                assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
                let x = rows(&p.x, false, 1536);
                if candidate {
                    rms_check(&format!("pass{pass}.start"), &starts[pass as usize], &x);
                } else {
                    starts.push(x);
                }
                if pass == 1 {
                    let a = rows(&p.qkv_half[0], true, 1536);
                    let b = rows(&p.embed_half, true, 1536);
                    let u: Vec<Vec<f64>> = a
                        .iter()
                        .zip(&b)
                        .map(|(a, b)| {
                            a.iter()
                                .zip(b)
                                .map(|(a, b)| a / (1.0 + (-b).exp()))
                                .collect()
                        })
                        .collect();
                    if candidate {
                        rms_check("feedback.gated", &feedback, &u);
                    } else {
                        feedback = u;
                    }
                }
                for layer in 0..24 {
                    let mut command = model.runtime.queue.new_command_buffer().to_owned();
                    p.layer(&model, &mut command, layer).unwrap();
                    command.commit();
                    command.wait_until_completed();
                    assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
                    let state = snapshot(&p);
                    if candidate {
                        let base = &saved[pass as usize * 24 + layer];
                        rms_check(
                            &format!("pass{pass}.layer{layer}.output"),
                            &base.x,
                            &state.x,
                        );
                        attention_check(base, &state, pass as usize, layer);
                    } else {
                        saved.push(state);
                    }
                }
                let command = model.runtime.queue.new_command_buffer();
                p.finish_pass(&model, command, pass == 1).unwrap();
                command.commit();
                command.wait_until_completed();
                assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
            }
            let loss =
                unsafe { std::slice::from_raw_parts(p.losses.contents().cast::<f32>(), 8192) };
            eprintln!(
                "BOUND_SCORE candidate={candidate} mean_nll={:.9}",
                loss.iter().map(|v| f64::from(*v)).sum::<f64>() / 8192.0
            );
        }
        drop(originals);
    });
}
