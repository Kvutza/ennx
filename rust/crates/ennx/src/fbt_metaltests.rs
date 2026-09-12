use super::*;
use crate::fbt::InputNorm;
use metal::MTLCommandBufferStatus;
use metal::objc::rc::autoreleasepool;

fn bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

fn decode(x: u16) -> f64 {
    f32::from_bits(u32::from(x) << 16) as f64
}

// Independent scalar f64 oracle: no SIMD reductions or shared kernel helpers.
fn reference(
    config: FeedbackConfig,
    weights: &[u16],
    previous: &[f32],
    tokens: &[f32],
    mask: &[u32],
) -> Vec<f32> {
    let d = config.width as usize;
    let normalize = |input: &[f64], mode| -> Vec<f64> {
        let scale = match mode {
            InputNorm::None => 1.0,
            InputNorm::UnitRms { epsilon } => (input.iter().map(|x| x * x).sum::<f64>() / d as f64
                + epsilon as f64)
                .sqrt()
                .recip(),
        };
        input.iter().map(|x| x * scale).collect()
    };
    let mut result = Vec::new();
    for (row, &enabled) in mask.iter().enumerate() {
        let token = &tokens[row * d..(row + 1) * d];
        if enabled == 0 {
            result.extend_from_slice(token);
            continue;
        }
        let token = normalize(
            &token.iter().map(|&x| x as f64).collect::<Vec<_>>(),
            config.token_norm,
        );
        let mut raw = vec![0.0; d];
        for out in 0..d {
            let mut u = 0.0;
            let mut g = 0.0;
            for i in 0..d {
                u += decode(weights[out * d + i]) * previous[row * d + i] as f64;
                g += decode(weights[d * d + out * d + i]) * token[i];
            }
            raw[out] = u / (1.0 + (-g).exp());
        }
        result.extend(
            normalize(&raw, config.fused_norm)
                .into_iter()
                .map(|x| x as f32),
        );
    }
    result
}

fn read(buffer: &BufferRef, count: usize) -> Vec<f32> {
    unsafe { std::slice::from_raw_parts(buffer.contents().cast::<f32>(), count).to_vec() }
}

fn grouped_case(width: u32, outputs: &[u32], rows: u32, measure: bool) {
    use crate::fbt::ProjectionActivation::{Identity, Sigmoid};
    let runtime = Runtime::shared().unwrap();
    let input = runtime.buffer_with(
        &(0..width * rows)
            .map(|i| (i % 17) as f32 * 0.03 - 0.2)
            .collect::<Vec<_>>(),
    );
    let linear: Vec<_> = outputs
        .iter()
        .enumerate()
        .map(|(i, &n)| {
            Linear::new(
                width,
                n,
                if i == 1 && !measure {
                    Sigmoid
                } else {
                    Identity
                },
            )
            .unwrap()
        })
        .collect();
    let weights: Vec<_> = outputs
        .iter()
        .map(|&n| {
            runtime.buffer_with(
                &(0..u64::from(width) * u64::from(n))
                    .map(|i| bf16((i % 23) as f32 * 0.02 - 0.2))
                    .collect::<Vec<_>>(),
            )
        })
        .collect();
    let separate: Vec<_> = outputs
        .iter()
        .map(|&n| runtime.buffer_with(&vec![-999.0f32; (rows * n) as usize + 7]))
        .collect();
    let grouped: Vec<_> = outputs
        .iter()
        .map(|&n| runtime.buffer_with(&vec![-999.0f32; (rows * n) as usize + 7]))
        .collect();
    let projections: Vec<_> = (0..outputs.len())
        .map(|i| (&linear[i], weights[i].as_ref(), grouped[i].as_ref()))
        .collect();
    let encode = |command: &CommandBufferRef, group: bool| {
        if group {
            Linear::encode_grouped(command, rows, &input, &projections).unwrap();
        } else {
            for i in 0..outputs.len() {
                linear[i]
                    .encode(command, rows, &weights[i], &input, &separate[i])
                    .unwrap();
            }
        }
    };
    let command = runtime.queue.new_command_buffer();
    assert!(Linear::encode_grouped(command, 0, &input, &projections).is_err());
    assert!(Linear::encode_grouped(command, rows, &input, &projections[..1]).is_err());
    let aliases = [projections[0], projections[0]];
    assert!(Linear::encode_grouped(command, rows, &input, &aliases).is_err());
    encode(command, false);
    encode(command, true);
    command.commit();
    command.wait_until_completed();
    assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
    for i in 0..outputs.len() {
        let count = (rows * outputs[i]) as usize;
        assert_eq!(read(&separate[i], count + 7), read(&grouped[i], count + 7));
        assert_eq!(&read(&grouped[i], count + 7)[count..], &[-999.0; 7]);
    }
    if measure {
        let mut samples = [Vec::new(), Vec::new()];
        // Alternate order; one projection set per command, allocation/compilation
        // excluded. Wall time includes encode, submit and completion wait.
        for round in 0..25 {
            for order in 0..2 {
                let mode = (round + order) % 2;
                let start = std::time::Instant::now();
                let command = runtime.queue.new_command_buffer();
                encode(command, mode == 1);
                command.commit();
                command.wait_until_completed();
                assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
                if round >= 5 {
                    samples[mode].push(start.elapsed().as_secs_f64() * 1e6);
                }
            }
        }
        for s in &mut samples {
            s.sort_by(f64::total_cmp);
        }
        eprintln!(
            "FBT grouped width={width} outputs={outputs:?} rows={rows}: separate median {:.1} us, grouped {:.1} us, ratio {:.3}",
            samples[0][10],
            samples[1][10],
            samples[0][10] / samples[1][10]
        );
    }
}

#[test]
fn grouped_projparity() {
    autoreleasepool(|| {
        for rows in [1, 3, 17] {
            grouped_case(33, &[31, 17, 9], rows, false);
            grouped_case(33, &[31, 17], rows, false);
        }
        grouped_case(1536, &[1536, 768, 768], 1, false);
    });
}

#[test]
#[ignore = "manual timing experiment; run without Metal validation"]
fn grouped_projtiming() {
    autoreleasepool(|| {
        for rows in [1, 16] {
            grouped_case(1536, &[1536, 768, 768], rows, true);
            grouped_case(1536, &[6656, 6656], rows, true);
        }
    });
}

#[test]
fn linear_liveweights() {
    use crate::fbt::ProjectionActivation::{Identity, Sigmoid};
    autoreleasepool(|| {
        let runtime = Runtime::shared().unwrap();
        for (input_width, output_width) in [(1, 1), (33, 17), (1536, 16), (96, 1536)] {
            let rows = 3;
            let weights: Vec<_> = (0..input_width * output_width)
                .map(|i| bf16((i % 29) as f32 / 29.0 - 0.4))
                .collect();
            let input: Vec<_> = (0..rows * input_width)
                .map(|i| (i % 13) as f32 / 7.0 - 0.6)
                .collect();
            let w = runtime.buffer_with(&weights);
            let x = runtime.buffer_with(&input);
            for activation in [Identity, Sigmoid] {
                let linear =
                    Linear::new(input_width as u32, output_width as u32, activation).unwrap();
                let y = runtime.buffer_with(&vec![-1234.0f32; rows * output_width + 7]);
                let command = runtime.queue.new_command_buffer();
                assert!(linear.encode(command, 0, &w, &x, &y).is_err());
                assert!(linear.encode(command, rows as u32, &w, &x, &x).is_err());
                let short = runtime.buffer::<u16>(1);
                if weights.len() > 1 {
                    assert!(linear.encode(command, rows as u32, &short, &x, &y).is_err());
                }
                linear.encode(command, rows as u32, &w, &x, &y).unwrap();
                command.commit();
                command.wait_until_completed();
                assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
                assert!(linear.encode(command, rows as u32, &w, &x, &y).is_err());
                let actual = read(&y, rows * output_width + 7);
                for row in 0..rows {
                    for out in 0..output_width {
                        let sum: f64 = (0..input_width)
                            .map(|i| {
                                decode(weights[out * input_width + i])
                                    * input[row * input_width + i] as f64
                            })
                            .sum();
                        let expected = match activation {
                            Identity => sum,
                            Sigmoid => 1.0 / (1.0 + (-sum).exp()),
                        } as f32;
                        assert!(
                            (actual[row * output_width + out] - expected).abs()
                                < 3e-5 * (1.0 + expected.abs())
                        );
                    }
                }
                assert_eq!(&actual[rows * output_width..], &[-1234.0; 7]);
            }
            let linear = Linear::new(input_width as u32, output_width as u32, Sigmoid).unwrap();
            unsafe {
                std::ptr::write_bytes(w.contents().cast::<u16>(), 0, weights.len());
            }
            let y = runtime.buffer::<f32>(rows * output_width);
            let command = runtime.queue.new_command_buffer();
            linear.encode(command, rows as u32, &w, &x, &y).unwrap();
            command.commit();
            command.wait_until_completed();
            assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
            assert!(read(&y, rows * output_width).iter().all(|&v| v == 0.5));
        }
        assert!(Linear::new(0, 1, Identity).is_err());
        assert!(Linear::new(u32::MAX, u32::MAX, Identity).is_err());
    });
}

#[test]
#[ignore = "full-context projection experiment, not complete model scoring"]
fn full_contexttiming() {
    projection_timing(false);
}

#[test]
#[ignore = "full-context tiled versus SIMD baseline experiment"]
fn full_tiledtiming() {
    projection_timing(true);
}

fn projection_timing(tiled: bool) {
    use crate::fbt::ProjectionActivation::Identity;
    autoreleasepool(|| {
        let runtime = Runtime::shared().unwrap();
        let width = 1536u32;
        let chunk = 256u32;
        for outputs in [vec![1536u32, 768, 768], vec![6656u32, 6656]] {
            let linear: Vec<_> = outputs
                .iter()
                .map(|&n| Linear::new(width, n, Identity).unwrap())
                .collect();
            let weights: Vec<_> = outputs
                .iter()
                .enumerate()
                .map(|(p, &n)| {
                    runtime.buffer_with(
                        &(0..u64::from(width) * u64::from(n))
                            .map(|i| bf16(((i + p as u64) % 37) as f32 / 37.0 - 0.5))
                            .collect::<Vec<_>>(),
                    )
                })
                .collect();
            let result: Vec<_> = outputs
                .iter()
                .map(|&n| runtime.buffer::<f32>((chunk * n) as usize))
                .collect();
            let projections: Vec<_> = (0..outputs.len())
                .map(|i| (&linear[i], weights[i].as_ref(), result[i].as_ref()))
                .collect();
            for context in [4096u32, 16384, 32768] {
                // Each token has distinct input; all input preparation is outside timing.
                let inputs: Vec<_> = (0..context / chunk)
                    .map(|part| {
                        runtime.buffer_with(
                            &(0..chunk * width)
                                .map(|i| {
                                    let index =
                                        u64::from(part * chunk) * u64::from(width) + u64::from(i);
                                    ((index.wrapping_mul(2654435761) % 65521) as f32 / 65521.0)
                                        - 0.5
                                })
                                .collect::<Vec<_>>(),
                        )
                    })
                    .collect();
                let mut wall = [Vec::new(), Vec::new()];
                let mut baseline: Option<Vec<Vec<f32>>> = None;
                for trial in 0..4 {
                    for order in 0..2 {
                        let mode = (trial + order) % 2;
                        let start = std::time::Instant::now();
                        let mut commands = Vec::with_capacity(inputs.len());
                        for input in &inputs {
                            let command = runtime.queue.new_command_buffer().to_owned();
                            if mode == 1 && tiled {
                                for i in 0..outputs.len() {
                                    linear[i]
                                        .encode_tiled(
                                            &command,
                                            chunk,
                                            &weights[i],
                                            input,
                                            &result[i],
                                        )
                                        .unwrap();
                                }
                            } else if mode == 1 {
                                Linear::encode_grouped(&command, chunk, input, &projections)
                                    .unwrap();
                            } else {
                                for i in 0..outputs.len() {
                                    linear[i]
                                        .encode(&command, chunk, &weights[i], input, &result[i])
                                        .unwrap();
                                }
                            }
                            command.commit();
                            commands.push(command);
                        }
                        commands.last().unwrap().wait_until_completed();
                        let elapsed = start.elapsed().as_secs_f64();
                        for command in &commands {
                            assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
                        }
                        if trial != 0 {
                            wall[mode].push(elapsed);
                        }
                        // Validate the final scratch contents after timing. Whole-context
                        // outputs are deliberately not retained by this projection benchmark.
                        let actual: Vec<_> = result
                            .iter()
                            .zip(&outputs)
                            .map(|(b, &n)| read(b, (chunk * n) as usize))
                            .collect();
                        assert!(actual.iter().flatten().all(|x| x.is_finite()));
                        if let Some(expected) = &baseline {
                            if tiled {
                                for (a, b) in actual.iter().flatten().zip(expected.iter().flatten())
                                {
                                    assert!(
                                        (a - b).abs() <= 2e-4 * (1.0 + b.abs()),
                                        "tiled parity: {a} vs {b}"
                                    );
                                }
                            } else {
                                assert_eq!(&actual, expected);
                            }
                        } else {
                            baseline = Some(actual);
                        }
                    }
                }
                for values in &mut wall {
                    values.sort_by(f64::total_cmp);
                }
                let flops = 2u64
                    * u64::from(context)
                    * u64::from(width)
                    * outputs.iter().map(|&n| u64::from(n)).sum::<u64>();
                eprintln!(
                    "FBT FULL-CONTEXT projections={outputs:?} context={context} chunk={chunk} candidates=1 layers=1 FLOPs={flops} input_MiB={:.1}: separate wall={:.6}s; {} wall={:.6}s",
                    f64::from(context) * f64::from(width) * 4.0 / 1048576.0,
                    wall[0][1],
                    if tiled { "tiled" } else { "grouped" },
                    wall[1][1]
                );
            }
        }
    });
}

#[test]
fn projection_parity() {
    use crate::fbt::ProjectionActivation::{Identity, Sigmoid};
    autoreleasepool(|| {
        let runtime = Runtime::shared().unwrap();
        for (k, n, rows) in [
            (1, 1, 1),
            (7, 9, 3),
            (31, 33, 63),
            (32, 32, 64),
            (33, 65, 65),
            (1536, 768, 65),
            (1536, 6656, 1),
        ] {
            let x: Vec<_> = (0..rows * k)
                .map(|i| ((i * 13 % 61) as f32 - 30.0) / 31.0)
                .collect();
            let w: Vec<_> = (0..n * k)
                .map(|i| bf16(((i * 7 % 43) as f32 - 21.0) / 23.0))
                .collect();
            let input = runtime.buffer_with(&x);
            for activation in [Identity, Sigmoid] {
                let weights = runtime.buffer_with(&w);
                let output = runtime.buffer_with(&vec![-999.0f32; rows * n + 7]);
                let linear = Linear::new(k as u32, n as u32, activation).unwrap();
                let command = runtime.queue.new_command_buffer();
                assert!(
                    linear
                        .encode_tiled(command, 0, &weights, &input, &output)
                        .is_err()
                );
                assert!(
                    linear
                        .encode_tiled(command, rows as u32, &weights, &input, &input)
                        .is_err()
                );
                let short = runtime.buffer::<f32>(1);
                if rows * n > 1 {
                    assert!(
                        linear
                            .encode_tiled(command, rows as u32, &weights, &input, &short)
                            .is_err()
                    );
                }
                linear
                    .encode_tiled(command, rows as u32, &weights, &input, &output)
                    .unwrap();
                command.commit();
                command.wait_until_completed();
                assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
                assert!(
                    linear
                        .encode_tiled(command, rows as u32, &weights, &input, &output)
                        .is_err()
                );
                let actual = read(&output, rows * n + 7);
                for row in 0..rows {
                    for col in 0..n {
                        let sum: f64 = (0..k)
                            .map(|i| decode(w[col * k + i]) * x[row * k + i] as f64)
                            .sum();
                        let expected = match activation {
                            Identity => sum,
                            Sigmoid => 1.0 / (1.0 + (-sum).exp()),
                        } as f32;
                        let a = actual[row * n + col];
                        assert!(
                            (a - expected).abs() <= 2e-4 * (1.0 + expected.abs()),
                            "shape={rows}/{k}/{n} ({row},{col}): {a} vs {expected}"
                        );
                    }
                }
                assert_eq!(&actual[rows * n..], &[-999.0; 7]);
                unsafe {
                    std::ptr::write_bytes(weights.contents().cast::<u16>(), 0, w.len());
                }
                let command = runtime.queue.new_command_buffer();
                linear
                    .encode_tiled(command, rows as u32, &weights, &input, &output)
                    .unwrap();
                command.commit();
                command.wait_until_completed();
                assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
                let expected = if activation == Identity { 0.0 } else { 0.5 };
                assert!(read(&output, rows * n).iter().all(|&a| a == expected));
            }
        }
    });
}

#[test]
fn feedback_parity() {
    autoreleasepool(|| {
        let runtime = Runtime::shared().expect("Metal GPU required for FBT tests");
        for width in [1, 31, 33, 96, 1536] {
            let d = width as usize;
            // Non-symmetric matrices catch projection orientation mistakes.
            let mut weights: Vec<_> = (0..2 * d * d)
                .map(|i| bf16(((i * 13 + i / d * 7) % 47) as f32 / 47.0 - 0.5))
                .collect();
            let mut previous: Vec<_> = (0..3 * d)
                .map(|i| ((i * 7) % 23) as f32 / 11.0 - 1.0)
                .collect();
            previous[..d].fill(f32::NAN); // Masked state must never contaminate plain rows.
            let tokens: Vec<_> = (0..3 * d)
                .map(|i| ((i * 11 + 3) % 29) as f32 / 13.0 - 1.0)
                .collect();
            let mask = [0u32, 1, 7];
            let w = runtime.buffer_with(&weights);
            let h = runtime.buffer_with(&previous);
            let e = runtime.buffer_with(&tokens);
            let m = runtime.buffer_with(&mask);
            let output = runtime.buffer_with(&vec![-1234.0f32; 3 * d + 7]);
            for token_norm in [InputNorm::None, InputNorm::UnitRms { epsilon: 1e-5 }] {
                for fused_norm in [InputNorm::None, InputNorm::UnitRms { epsilon: 1e-6 }] {
                    let config = FeedbackConfig {
                        width,
                        token_norm,
                        fused_norm,
                    };
                    let mut feedback = Feedback::new(config, 4).unwrap();
                    let command = runtime.queue.new_command_buffer();
                    feedback
                        .encode(command, 3, &w, &h, &e, &m, &output)
                        .unwrap();
                    command.commit();
                    command.wait_until_completed();
                    assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
                    let actual = read(&output, 3 * d + 7);
                    let expected = reference(config, &weights, &previous, &tokens, &mask);
                    for (index, (&a, &b)) in actual.iter().zip(&expected).enumerate() {
                        let tolerance = 3e-5 + 3e-5 * b.abs();
                        assert!(
                            (a - b).abs() <= tolerance,
                            "width={width} index={index} norms={token_norm:?}/{fused_norm:?}: {a} vs {b}"
                        );
                    }
                    assert_eq!(&actual[..d], &tokens[..d]);
                    assert_eq!(&actual[3 * d..], &[-1234.0; 7]);
                    // BO can change matrices without reconstructing the evaluator.
                    let saved = weights.clone();
                    weights[..d * d].fill(bf16(0.0));
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            weights.as_ptr(),
                            w.contents().cast(),
                            weights.len(),
                        );
                    }
                    let command = runtime.queue.new_command_buffer();
                    feedback
                        .encode(command, 3, &w, &h, &e, &m, &output)
                        .unwrap();
                    command.commit();
                    command.wait_until_completed();
                    assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
                    let zero = read(&output, 3 * d);
                    assert_eq!(&zero[..d], &tokens[..d]);
                    assert!(zero[d..].iter().all(|&x| x == 0.0));
                    weights = saved;
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            weights.as_ptr(),
                            w.contents().cast(),
                            weights.len(),
                        );
                    }
                }
            }
        }
    });
}

#[test]
fn buffer_guards() {
    autoreleasepool(|| {
        let runtime = Runtime::shared().unwrap();
        let config = FeedbackConfig {
            width: 33,
            token_norm: InputNorm::None,
            fused_norm: InputNorm::None,
        };
        assert!(Feedback::new(config, 0).is_err());
        let mut feedback = Feedback::new(config, 2).unwrap();
        let w = runtime.buffer_with(&vec![0u16; 2 * 33 * 33]);
        let h = runtime.buffer_with(&vec![1.0f32; 66]);
        let e = runtime.buffer_with(&vec![2.0f32; 66]);
        let m = runtime.buffer_with(&[1u32, 1]);
        let out = runtime.buffer::<f32>(66);
        let short = runtime.buffer::<f32>(1);
        let command = runtime.queue.new_command_buffer();
        for rows in [0, 3] {
            assert!(
                feedback
                    .encode(command, rows, &w, &h, &e, &m, &out)
                    .is_err()
            );
        }
        assert!(
            feedback
                .encode(command, 2, &short, &h, &e, &m, &out)
                .is_err()
        );
        assert!(feedback.encode(command, 2, &w, &h, &e, &m, &short).is_err());
        assert!(feedback.encode(command, 2, &w, &h, &e, &m, &e).is_err());
        assert!(feedback.encode(command, 2, &w, &h, &h, &m, &out).is_err());
    });
}

#[test]
fn gate_chain() {
    autoreleasepool(|| {
        let runtime = Runtime::shared().unwrap();
        let rows = 4096;
        let config = FeedbackConfig {
            width: 1,
            token_norm: InputNorm::None,
            fused_norm: InputNorm::None,
        };
        let mut feedback = Feedback::new(config, rows).unwrap();
        let w = runtime.buffer_with(&[bf16(2.0), bf16(1000.0)]);
        let h = runtime.buffer_with(&vec![3.0f32; rows as usize]);
        let tokens: Vec<_> = (0..rows)
            .map(|i| if i % 2 == 0 { 1.0f32 } else { -1.0 })
            .collect();
        let e = runtime.buffer_with(&tokens);
        let m = runtime.buffer_with(&vec![1u32; rows as usize]);
        let first = runtime.buffer::<f32>(rows as usize);
        let second = runtime.buffer::<f32>(rows as usize);
        let command = runtime.queue.new_command_buffer();
        feedback
            .encode(command, rows, &w, &h, &e, &m, &first)
            .unwrap();
        feedback
            .encode(command, rows, &w, &first, &e, &m, &second)
            .unwrap();
        command.commit();
        command.wait_until_completed();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        for (i, value) in read(&second, rows as usize).into_iter().enumerate() {
            assert_eq!(value, if i % 2 == 0 { 12.0 } else { 0.0 });
        }
        assert!(
            feedback
                .encode(command, rows, &w, &h, &e, &m, &first)
                .is_err()
        );
    });
}

fn ffn_reference(
    config: FeedForwardConfig,
    weights: &[u16],
    input: &[f32],
    residual: &[f32],
) -> Vec<f32> {
    let d = config.width as usize;
    let f = config.intermediate as usize;
    let matrix = d * f;
    let mut result = Vec::new();
    for (row, x) in input.chunks_exact(d).enumerate() {
        let mut hidden = vec![0.0; f];
        for j in 0..f {
            let mut gate = 0.0;
            let mut up = 0.0;
            for i in 0..d {
                gate += decode(weights[j * d + i]) * x[i] as f64;
                up += decode(weights[matrix + j * d + i]) * x[i] as f64;
            }
            hidden[j] = gate / (1.0 + (-gate).exp()) * up;
        }
        for j in 0..d {
            let down = (0..f)
                .map(|i| decode(weights[2 * matrix + j * f + i]) * hidden[i])
                .sum::<f64>();
            result
                .push((residual[row * d + j] as f64 + config.residual_scale as f64 * down) as f32);
        }
    }
    result
}

#[test]
fn feed_forwardparity() {
    autoreleasepool(|| {
        let runtime = Runtime::shared().unwrap();
        for (width, intermediate) in [(1, 1), (31, 65), (96, 193), (1536, 17), (33, 6656)] {
            let d = width as usize;
            let f = intermediate as usize;
            let mut weights: Vec<_> = (0..3 * d * f)
                .map(|i| bf16(((i * 7 + i / d) % 41) as f32 / 128.0 - 0.15))
                .collect();
            let input: Vec<_> = (0..3 * d)
                .map(|i| ((i * 3) % 17) as f32 / 32.0 - 0.2)
                .collect();
            let residual: Vec<_> = (0..3 * d)
                .map(|i| i as f32 / (3 * d) as f32 - 0.5)
                .collect();
            let w = runtime.buffer_with(&weights);
            let x = runtime.buffer_with(&input);
            let r = runtime.buffer_with(&residual);
            let y = runtime.buffer_with(&vec![-345.0f32; 3 * d + 5]);
            for scale in [0.0, -0.25, 1.0] {
                let config = FeedForwardConfig {
                    width,
                    intermediate,
                    residual_scale: scale,
                };
                let mut ffn = FeedForward::new(config, 4).unwrap();
                assert_eq!(ffn.layout().elements, weights.len());
                let command = runtime.queue.new_command_buffer();
                ffn.encode(command, 3, &w, &x, &r, &y).unwrap();
                command.commit();
                command.wait_until_completed();
                assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
                let expected = ffn_reference(config, &weights, &input, &residual);
                let actual = read(&y, 3 * d + 5);
                for (i, (&a, &b)) in actual.iter().zip(&expected).enumerate() {
                    assert!(
                        (a - b).abs() <= 5e-5 + 5e-5 * b.abs(),
                        "FFN {width}/{intermediate} scale={scale} index={i}: {a} vs {b}"
                    );
                }
                assert_eq!(&actual[3 * d..], &[-345.0; 5]);
                if scale == 0.0 {
                    assert_eq!(&actual[..3 * d], &residual);
                }
                assert!(ffn.encode(command, 3, &w, &x, &r, &y).is_err());
            }
            let mut ffn = FeedForward::new(
                FeedForwardConfig {
                    width,
                    intermediate,
                    residual_scale: 1.0,
                },
                3,
            )
            .unwrap();
            let command = runtime.queue.new_command_buffer();
            ffn.encode(command, 3, &w, &x, &r, &y).unwrap();
            command.commit();
            command.wait_until_completed();
            assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
            weights[2 * d * f..].fill(0);
            unsafe {
                std::ptr::copy_nonoverlapping(weights.as_ptr(), w.contents().cast(), weights.len());
            }
            let command = runtime.queue.new_command_buffer();
            assert!(ffn.encode(command, 4, &w, &x, &r, &y).is_err());
            assert!(ffn.encode(command, 0, &w, &x, &r, &y).is_err());
            assert!(ffn.encode(command, 3, &w, &x, &r, &r).is_err());
            let short = runtime.buffer::<u16>(1);
            assert!(ffn.encode(command, 3, &short, &x, &r, &y).is_err());
            ffn.encode(command, 3, &w, &x, &r, &y).unwrap();
            command.commit();
            command.wait_until_completed();
            assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
            assert_eq!(read(&y, 3 * d), residual);
        }
    });
}

#[test]
fn feedback_gpu() {
    autoreleasepool(|| {
        let runtime = Runtime::shared().unwrap();
        let config = FeedbackConfig {
            width: 1,
            token_norm: InputNorm::None,
            fused_norm: InputNorm::None,
        };
        let ffn_config = FeedForwardConfig {
            width: 1,
            intermediate: 1,
            residual_scale: 0.5,
        };
        let mut feedback = Feedback::new(config, 1).unwrap();
        let mut ffn = FeedForward::new(ffn_config, 1).unwrap();
        let w = runtime.buffer_with(&[bf16(2.0), bf16(0.0)]);
        let ffw = runtime.buffer_with(&[bf16(1.0), bf16(2.0), bf16(3.0)]);
        let previous = runtime.buffer_with(&[1.0f32]);
        let tokens = runtime.buffer_with(&[0.7f32]);
        let mask = runtime.buffer_with(&[1u32]);
        let residual = runtime.buffer_with(&[0.25f32]);
        let fused = runtime.buffer::<f32>(1);
        let output = runtime.buffer::<f32>(1);
        let command = runtime.queue.new_command_buffer();
        feedback
            .encode(command, 1, &w, &previous, &tokens, &mask, &fused)
            .unwrap();
        ffn.encode(command, 1, &ffw, &fused, &residual, &output)
            .unwrap();
        command.commit();
        command.wait_until_completed();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        let expected = 0.25 + 3.0 / (1.0 + (-1.0f32).exp());
        assert!((read(&output, 1)[0] - expected).abs() < 1e-5);
    });
}
