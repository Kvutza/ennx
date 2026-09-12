use super::*;
use metal::objc::rc::autoreleasepool;

// Synchronous measurement bridge: each parameter is copied once, including
// norms and feedback matrices. The tied readout uses the embedding allocation.
fn apply_weights(model: &mut Model, source: &BufferRef) {
    assert_eq!(source.length(), model.parameter_count() as u64 * 2);
    let revision = model.revision.checked_add(1).unwrap();
    autoreleasepool(|| {
        let command = model.runtime.queue.new_command_buffer();
        let blit = command.new_blit_command_encoder();
        let mut offset = 0;
        for p in &model.parameters {
            let bytes = p.elements as u64 * 2;
            blit.copy_from_buffer(source, offset, &p.buffer, 0, bytes);
            offset += bytes;
        }
        blit.end_encoding();
        command.commit();
        command.wait_until_completed();
    });
    model.revision = revision;
}

fn bind_weights(model: &mut Model, source: &BufferRef) -> Vec<Buffer> {
    model.bind_parameter_row(source).unwrap()
}

fn bo_config() -> crate::config::ConfigOverrides {
    crate::config::parse_turbo_enn_config(
        "version=1\nstudy='end_to_end'\nacquisition='thompson'\nlength_init=0.01\nlength_min=0.0001\nlength_max=0.1\noutput='unused'\nrounds=3\ntarget_round_ms=1000",
    )
    .unwrap()
}

fn model_search(model: &Model, value: f32) -> crate::bf16_metal::SearchState {
    let mut search = super::bo::model_search(model, &bo_config());
    search.observe_initial(value, 0.0).unwrap();
    assert_eq!(search.controller_info().unwrap().failure_tolerance, 4);
    search
}

#[test]
#[ignore = "full-size GPU proposal structure diagnostic; no forward passes"]
fn proposal_structure() {
    autoreleasepool(|| {
        let model = Model::new(super::bo::full_model_config(), 42).unwrap();
        let mut search = model_search(&model, -1.0);
        for candidate in 0..4 {
            let proposal = search.test_candidate(123, candidate).unwrap();
            let values = unsafe {
                std::slice::from_raw_parts(
                    proposal.contents().cast::<u16>(),
                    model.parameter_count(),
                )
            };
            let mut offset = 0;
            let mut total_changed = 0usize;
            for parameter in &model.parameters {
                let base = unsafe {
                    std::slice::from_raw_parts(
                        parameter.buffer.contents().cast::<u16>(),
                        parameter.elements,
                    )
                };
                let proposed = &values[offset..offset + parameter.elements];
                let changed = base.iter().zip(proposed).filter(|(a, b)| a != b).count();
                let width = *parameter.shape.last().unwrap() as usize;
                let unchanged_rows = base
                    .chunks_exact(width)
                    .zip(proposed.chunks_exact(width))
                    .filter(|(a, b)| a == b)
                    .count();
                let unchanged_blocks = base
                    .chunks_exact(64)
                    .zip(proposed.chunks_exact(64))
                    .filter(|(a, b)| a == b)
                    .count();
                assert!(
                    proposed
                        .iter()
                        .all(|bits| { f32::from_bits(u32::from(*bits) << 16).is_finite() })
                );
                eprintln!(
                    "PROPOSAL_STRUCTURE candidate={candidate} tensor={} shape={:?} elements={} changed={changed} unchanged_rows={unchanged_rows} rows={} unchanged_blocks64={unchanged_blocks} blocks64={}",
                    parameter.name,
                    parameter.shape,
                    parameter.elements,
                    parameter.elements / width,
                    parameter.elements / 64,
                );
                total_changed += changed;
                offset += parameter.elements;
            }
            assert_eq!(offset, model.parameter_count());
            eprintln!(
                "PROPOSAL_STRUCTURE_TOTAL candidate={candidate} elements={offset} changed={total_changed} changed_fraction={:.9}",
                total_changed as f64 / offset as f64,
            );
        }
    });
}

#[test]
fn bridge_restore() {
    autoreleasepool(|| {
        let mut model = Model::new(tiny(3), 42).unwrap();
        let mut reference = Model::new(tiny(3), 42).unwrap();
        let mut search = model_search(&model, -1.0);
        let ask = crate::trials::Ask {
            neighbors: 2,
            ..Default::default()
        };
        for accepted in [false, true] {
            let round = search.ask_round(1, 4, 123, ask).unwrap();
            let proposal = search.propose_buffer(&round).unwrap();
            apply_weights(&mut model, &proposal);
            let values = unsafe {
                std::slice::from_raw_parts(
                    proposal.contents().cast::<u16>(),
                    model.parameter_count(),
                )
            };
            let mut offset = 0;
            for p in &model.parameters {
                reference
                    .replace_parameter(&p.name, &values[offset..offset + p.elements])
                    .unwrap();
                offset += p.elements;
            }
            let actual = model
                .score(&[1, 2, 3], &[2, 3, 4], ScoreMode::Fused)
                .unwrap();
            let expected = reference
                .score(&[1, 2, 3], &[2, 3, 4], ScoreMode::Fused)
                .unwrap();
            assert!((actual.mean_nll - expected.mean_nll).abs() < 1e-6);
            let improvement = if accepted { 1.0 } else { -1.0 };
            search
                .tell_paired(&round, -1.0 + improvement, 0.0, -1.0, 0.0, accepted)
                .unwrap();
            assert_eq!(search.sync().unwrap(), vec![accepted]);
            let base = search.base_buffer();
            apply_weights(&mut model, &base);
            let values = search.read_best().unwrap();
            let mut offset = 0;
            for p in &model.parameters {
                let actual = unsafe {
                    std::slice::from_raw_parts(p.buffer.contents().cast::<u16>(), p.elements)
                };
                assert_eq!(actual, &values[offset..offset + p.elements]);
                offset += p.elements;
            }
        }
    });
}

#[test]
#[ignore = "GPU diagnostic; exercised by tools/fbt-bo --check"]
fn gpu_scorer() {
    gpu_scorer_case(false, 256, super::GateUpImplementation::Mps);
}

#[test]
#[ignore = "GPU diagnostic; exercised by tools/fbt-bo --check"]
fn gpu_scorer_optimized() {
    gpu_scorer_case(true, 256, super::GateUpImplementation::Mps);
}

#[test]
#[ignore = "GPU diagnostic; exercised by tools/fbt-bo --check"]
fn gpu_scorer_full_ffn() {
    gpu_scorer_case(true, 6656, super::GateUpImplementation::Mps);
}

#[test]
#[ignore = "full-width fused Metal gate/up parity; required before configured fused runs"]
fn gpu_gate_up_fused_metal() {
    gpu_scorer_case(true, 6656, super::GateUpImplementation::FusedMetal);
}

#[test]
#[ignore = "GPU diagnostic; exercised by tools/fbt-bo --check"]
fn gpu_scorer_readout_views() {
    use crate::fbt_mps::{Matmul, Matrix};
    autoreleasepool(|| {
        let runtime = crate::apple_gpu::Runtime::shared().unwrap();
        let mut gemm = Matmul::default();
        let values: Vec<f32> = (0..32 * 8).map(|i| i as f32 / 256.0).collect();
        let a = runtime.buffer_with(&values);
        let mut identity = vec![0.0f32; 64];
        for i in 0..8 {
            identity[i * 8 + i] = 1.0;
        }
        let b = runtime.buffer_with(&identity);
        let scratch = runtime.buffer_with(&vec![0.0f32; 8 * 8]);
        let out = runtime.buffer_with(&vec![0.0f32; 32 * 8]);
        let command = runtime.queue.new_command_buffer();
        for start in (0..32).step_by(8) {
            gemm.encode(
                &runtime.device,
                command,
                Matrix::new(&a, 8, 8).row_view(8, start * 8),
                Matrix::new(&b, 8, 8),
                Matrix::new(&scratch, 8, 8),
                true,
                1.0,
            )
            .unwrap();
            let blit = command.new_blit_command_encoder();
            blit.copy_from_buffer(&scratch, 0, &out, start * 8 * 4, 8 * 8 * 4);
            blit.end_encoding();
        }
        command.commit();
        command.wait_until_completed();
        assert_eq!(command.status(), metal::MTLCommandBufferStatus::Completed);
        let actual =
            unsafe { std::slice::from_raw_parts(out.contents().cast::<f32>(), values.len()) };
        assert_eq!(actual, values.as_slice());
        let padded_a =
            runtime.buffer_with(&[values.as_slice(), &vec![-999.0f32; values.len()]].concat());
        let padded_b = runtime.buffer_with(
            &[
                identity.as_slice(),
                identity.as_slice(),
                &vec![-999.0f32; 128],
            ]
            .concat(),
        );
        let padded_c = runtime.buffer_with(&vec![-999.0f32; values.len() * 2]);
        let command = runtime.queue.new_command_buffer();
        gemm.encode(
            &runtime.device,
            command,
            Matrix::new(&padded_a, 16, 8).layout(2, 128, 0),
            Matrix::new(&padded_b, 8, 8).layout(2, 64, 0),
            Matrix::new(&padded_c, 16, 8).layout(2, 128, 0),
            true,
            1.0,
        )
        .unwrap();
        command.commit();
        command.wait_until_completed();
        assert_eq!(command.status(), metal::MTLCommandBufferStatus::Completed);
        let actual = unsafe {
            std::slice::from_raw_parts(padded_c.contents().cast::<f32>(), values.len() * 2)
        };
        assert_eq!(&actual[..values.len()], values.as_slice());
        assert!(actual[values.len()..].iter().all(|v| *v == -999.0));
    });
}

fn gpu_scorer_case(optimized: bool, intermediate: u32, gate_up: super::GateUpImplementation) {
    autoreleasepool(|| {
        let mut c = tiny(16);
        c.width = 192;
        c.intermediate = intermediate;
        c.vocab = 97;
        c.capacity = 64;
        c.tiled_attention = true;
        if optimized {
            c.width = 1536;
            c.heads = 16;
            c.kv_heads = 8;
        }
        let mut reference = Model::new(c, 42).unwrap();
        let mut gpu = Model::new(c, 42).unwrap();
        gpu.set_optimized(optimized).unwrap();
        gpu.set_gate_up_implementation(gate_up).unwrap();
        reference.prepare_prefill(2, 64).unwrap();
        gpu.prepare_prefill(2, 64).unwrap();
        let x: Vec<_> = (0..64).map(|i| i * 7 % 97).collect();
        let y: Vec<_> = (0..64).map(|i| (i * 13 + 19) % 97).collect();
        let examples = [(&x[..], &y[..]), (&y[..], &x[..])];
        let compare = |reference: &mut Model, gpu: &mut Model, mode| {
            let ref_single = reference.score(&x, &y, mode).unwrap();
            let ref_single_tokens =
                unsafe { std::slice::from_raw_parts(reference.losses.contents().cast::<f32>(), 5) };
            eprintln!(
                "REF_SINGLE tokens[..5]={:?} mean={}",
                ref_single_tokens, ref_single.mean_nll
            );
            let expected = reference.score_batch(&examples, mode).unwrap();
            let actual = gpu.score_batch(&examples, mode).unwrap();
            let expected_tokens = reference.prefills[0].token_losses();
            let actual_tokens = gpu.prefills[0].token_losses();
            let max_error = expected_tokens
                .iter()
                .zip(&actual_tokens)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            for (idx, (a, b)) in expected_tokens.iter().zip(&actual_tokens).enumerate() {
                if (a - b).abs() > 0.02 {
                    eprintln!(
                        "DRIFT at idx={idx} expected={a} actual={b} diff={}",
                        (a - b).abs()
                    );
                }
            }
            eprintln!("EXPECTED={:?}", &expected_tokens[..5]);
            eprintln!("ACTUAL={:?}", &actual_tokens[..5]);
            eprintln!(
                "FBT_GPU_CHECK mode={mode:?} max_token_loss_error={max_error:.9} reference={:?} gpu={:?}",
                expected.mean_nll, actual.mean_nll
            );
            assert!(
                max_error < 0.02,
                "FP16 complete-scorer token NLL drift: {max_error}"
            );
            for (a, b) in expected.mean_nll.iter().zip(&actual.mean_nll) {
                assert!((a - b).abs() < 0.003);
            }
            (expected.mean_nll, actual.mean_nll)
        };
        if gate_up == super::GateUpImplementation::FusedMetal {
            super::prefill::reset_fused_gate_up_dispatches();
        }
        compare(&mut reference, &mut gpu, ScoreMode::Standard);
        if gate_up == super::GateUpImplementation::FusedMetal {
            assert!(
                super::prefill::fused_gate_up_dispatches() > 0,
                "normal score_batch did not dispatch the fused gate/up kernel"
            );
        }
        let mut search = model_search(&gpu, 0.0);
        for (step, force_accept) in [false, true].into_iter().enumerate() {
            let incumbent = compare(&mut reference, &mut gpu, ScoreMode::Fused);
            let round = search
                .ask_round(
                    1,
                    4,
                    123 + step as u64,
                    crate::trials::Ask {
                        neighbors: 2,
                        acquisition: crate::weights::AcquisitionKind::Thompson,
                        seed: 456,
                        ..Default::default()
                    },
                )
                .unwrap();
            let proposal = search.propose_buffer(&round).unwrap();
            apply_weights(&mut reference, &proposal);
            apply_weights(&mut gpu, &proposal);
            let candidate = compare(&mut reference, &mut gpu, ScoreMode::Fused);
            let decision = |old: &[f64], new: &[f64]| {
                let a = old[0] - new[0];
                let b = old[1] - new[1];
                (a + b) / 2.0 > (a - b).abs() && new.iter().sum::<f64>() < old.iter().sum::<f64>()
            };
            assert_eq!(
                decision(&incumbent.0, &candidate.0),
                decision(&incumbent.1, &candidate.1),
                "Precision changed the paired decision on this fixture"
            );
            // Separately exercise both controller branches; these synthetic tells
            // test state restoration, not the fixture's statistically chosen step.
            let improvement = if force_accept { 1.0 } else { -1.0 };
            search
                .tell_paired(&round, improvement, 0.0, 0.0, 0.0, force_accept)
                .unwrap();
            assert_eq!(search.sync().unwrap(), vec![force_accept]);
            let base = search.base_buffer();
            apply_weights(&mut reference, &base);
            apply_weights(&mut gpu, &base);
            let restored = compare(&mut reference, &mut gpu, ScoreMode::Fused);
            assert_eq!(
                restored.1,
                if force_accept {
                    candidate.1
                } else {
                    incumbent.1
                }
            );
            let expected = search.read_best().unwrap();
            let mut offset = 0;
            for parameter in &gpu.parameters {
                let bits = unsafe {
                    std::slice::from_raw_parts(
                        parameter.buffer.contents().cast::<u16>(),
                        parameter.elements,
                    )
                };
                assert_eq!(bits, &expected[offset..offset + parameter.elements]);
                offset += parameter.elements;
            }
        }
        if optimized && intermediate == 6656 {
            let ordinary = gpu.score_batch(&examples, ScoreMode::Fused).unwrap();
            apply_weights(&mut gpu, &search.base_buffer());
            let traced = gpu.score_batch_traced(&examples, ScoreMode::Fused).unwrap();
            assert_eq!(ordinary.mean_nll, traced.score.mean_nll);
            assert!(
                traced
                    .operations
                    .iter()
                    .any(|op| op.operation == "gate_up_weight_pack")
            );
            let operation = if gate_up == super::GateUpImplementation::FusedMetal {
                "gate_up_glu"
            } else {
                "gate_up_gemm"
            };
            assert_eq!(
                traced
                    .operations
                    .iter()
                    .filter(|op| op.operation == operation)
                    .count(),
                2 * c.layers as usize
            );
        }

        gpu.set_optimized(false).unwrap();
        if optimized {
            gpu.prefills.clear();
            gpu.prepare_prefill(2, 64).unwrap();
        }
        let a = reference.score_batch(&examples, ScoreMode::Fused).unwrap();
        let b = gpu.score_batch(&examples, ScoreMode::Fused).unwrap();
        assert_eq!(a.mean_nll, b.mean_nll);
    });
}

#[test]
fn full_model_study_preserves_model_geometry() {
    let config = super::bo::full_model_config();
    // memory() validates dimensions and counts parameters without allocating them.
    assert_eq!(config.memory().unwrap().parameters, 1_065_494_016);
    assert_eq!(config.width / config.heads, 96);
    assert_eq!(config.capacity, 4096);
    assert_eq!(config.chunk, 256);
    assert_eq!(config.local_window, 2048);
    assert_eq!(config.full_every, 6);
    assert_eq!(
        config.residual_scale.to_bits(),
        (1.0 / 48.0f32.sqrt()).to_bits()
    );
    for norm in [config.feedback_token_norm, config.feedback_fused_norm] {
        assert!(matches!(norm, InputNorm::UnitRms { epsilon } if epsilon == 1e-5));
    }
    assert!(config.tiled_attention);
}

fn tiny(chunk: u32) -> ModelConfig {
    ModelConfig {
        width: 8,
        intermediate: 13,
        layers: 2,
        vocab: 11,
        heads: 2,
        kv_heads: 1,
        capacity: 67,
        chunk,
        local_window: 3,
        full_every: 2,
        epsilon: 1e-5,
        rope_base: 10000.0,
        residual_scale: 0.5,
        feedback_token_norm: InputNorm::UnitRms { epsilon: 1e-5 },
        feedback_fused_norm: InputNorm::UnitRms { epsilon: 1e-5 },
        tiled_attention: false,
    }
}

#[test]
fn batch_parity() {
    autoreleasepool(|| {
        for length in [1, 67, 256, 257] {
            let mut c = tiny(17.min(length));
            c.capacity = length;
            c.width = 192;
            c.intermediate = 129;
            c.vocab = 97;
            c.tiled_attention = true;
            let mut model = Model::new(c, 42).unwrap();
            let x: Vec<_> = (0..length).map(|i| (i * 7) % 97).collect();
            let y: Vec<_> = (0..length)
                .map(|i| if i % 2 == 0 { 96 } else { 0 })
                .collect();
            let z: Vec<_> = (0..length).map(|i| (i * 13 + 19) % 97).collect();
            for mode in [ScoreMode::Standard, ScoreMode::Fused] {
                let a = model.score(&x, &y, mode).unwrap();
                let model_x_data = data(&model.x, 8);
                let expected_losses = data(&model.losses, length as usize);
                let b = model.score(&z, &x, mode).unwrap();
                let mut expected_losses = expected_losses;
                expected_losses.extend(data(&model.losses, length as usize));
                let single_batch = model.score_batch(&[(&x, &y)], mode).unwrap();
                let prefill_x_data = data(&model.prefills[0].x, 8);
                eprintln!("model.x: {:?}", model_x_data);
                eprintln!("prefill.x: {:?}", prefill_x_data);
                eprintln!("single_batch x->y: {:?}", single_batch.mean_nll);
                let batch = model.score_batch(&[(&x, &y), (&z, &x)], mode).unwrap();
                let actual_losses = model.prefills[0].token_losses();
                eprintln!(
                    "batch_parity length={length} mode={mode:?} expected={:?} actual={:?}",
                    expected_losses, actual_losses
                );
                let max_diff = actual_losses
                    .iter()
                    .zip(&expected_losses)
                    .map(|(a, b)| (f64::from(*a) - b).abs())
                    .fold(0.0f64, f64::max);
                assert!(
                    max_diff < 0.02,
                    "length={length} mode={mode:?} max_token_loss_diff={max_diff}"
                );
                for (actual, expected) in batch.mean_nll.iter().zip([a.mean_nll, b.mean_nll]) {
                    assert!(
                        (actual - expected).abs() < 0.003,
                        "length={length} mode={mode:?}: {actual} vs {expected}"
                    );
                }
                let reversed = model.score_batch(&[(&z, &x), (&x, &y)], mode).unwrap();
                assert!((batch.mean_nll[0] - reversed.mean_nll[1]).abs() < 1e-6);
                assert!((batch.mean_nll[1] - reversed.mean_nll[0]).abs() < 1e-6);
            }
            assert!(model.score_batch(&[], ScoreMode::Fused).is_err());
            assert!(
                model
                    .score_batch(&[(&x, &y)], ScoreMode::Sequential)
                    .is_err()
            );
            assert!(
                model
                    .score_batch(&[(&x, &y), (&[], &[])], ScoreMode::Fused)
                    .is_err()
            );
            assert!(
                model
                    .score_batch(&[(&[97], &[0])], ScoreMode::Standard)
                    .is_err()
            );
            assert!(model.prepare_prefill(u32::MAX, length).is_err());
        }
    });
}

#[test]
fn live_weights() {
    autoreleasepool(|| {
        let mut model = Model::new(tiny(3), 42).unwrap();
        let x = [1, 2, 3, 4, 5];
        let y = [2, 3, 4, 5, 6];
        let before = model
            .score_batch(&[(&x, &y), (&y, &x)], ScoreMode::Fused)
            .unwrap();
        let mut search = model_search(&model, -1.0);
        let ask = crate::trials::Ask {
            neighbors: 2,
            ..Default::default()
        };
        let round = search.ask_round(1, 4, 123, ask).unwrap();
        apply_weights(&mut model, &search.propose_buffer(&round).unwrap());
        let actual = model
            .score_batch(&[(&x, &y), (&y, &x)], ScoreMode::Fused)
            .unwrap();
        let expected = model.score(&x, &y, ScoreMode::Fused).unwrap();
        assert!((actual.mean_nll[0] - expected.mean_nll).abs() < 0.003);
        assert!((actual.mean_nll[0] - before.mean_nll[0]).abs() > 1e-7);
        search
            .tell_paired(&round, -2.0, 0.0, -1.0, 0.0, false)
            .unwrap();
        assert_eq!(search.sync().unwrap(), vec![false]);
        apply_weights(&mut model, &search.base_buffer());
        let restored = model
            .score_batch(&[(&x, &y), (&y, &x)], ScoreMode::Fused)
            .unwrap();
        assert_eq!(before.mean_nll, restored.mean_nll);
    });
}

#[test]
fn fused_graphparity() {
    autoreleasepool(|| {
        for chunk in [3, 64, 67] {
            let mut c = tiny(chunk);
            c.width = 192;
            c.intermediate = 129;
            c.vocab = 97;
            c.tiled_attention = true;
            let tokens: Vec<_> = (0..67).map(|i| i % c.vocab).collect();
            let targets: Vec<_> = (0..67).map(|i| if i % 2 == 0 { 96 } else { 0 }).collect();
            let mut model = Model::new(c, 42).unwrap();
            for mode in [ScoreMode::Standard, ScoreMode::Fused, ScoreMode::Sequential] {
                model.set_optimized(false).unwrap();
                let expected = model.score(&tokens, &targets, mode).unwrap();
                let expected_losses = data(&model.losses, 67);
                model.set_optimized(true).unwrap();
                let actual = model.score(&tokens, &targets, mode).unwrap();
                assert!((actual.mean_nll - expected.mean_nll).abs() < 0.003);
                for (a, b) in data(&model.losses, 67).iter().zip(&expected_losses) {
                    assert!(
                        (a - b).abs() < 0.004,
                        "mode={mode:?} chunk={chunk}: {a} vs {b}"
                    );
                }
                assert_eq!(model.logits.length(), u64::from(chunk) * 4 * 16);
            }
        }
    });
}

fn data(buffer: &BufferRef, count: usize) -> Vec<f64> {
    unsafe { std::slice::from_raw_parts(buffer.contents().cast::<f32>(), count) }
        .iter()
        .map(|&v| v as f64)
        .collect()
}

fn weights(m: &Model, index: usize) -> Vec<f64> {
    let p = &m.parameters[index];
    unsafe { std::slice::from_raw_parts(p.buffer.contents().cast::<u16>(), p.elements) }
        .iter()
        .map(|&v| f32::from_bits(u32::from(v) << 16) as f64)
        .collect()
}

fn norm(x: &[f64], gamma: &[f64], epsilon: f64) -> Vec<f64> {
    let inv = (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64 + epsilon)
        .sqrt()
        .recip();
    x.iter().zip(gamma).map(|(v, g)| v * inv * g).collect()
}

fn project(w: &[f64], x: &[f64]) -> Vec<f64> {
    w.chunks_exact(x.len())
        .map(|r| r.iter().zip(x).map(|(a, b)| a * b).sum())
        .collect()
}

fn rounded_bf16(v: f64) -> f64 {
    let bits = (v as f32).to_bits();
    f32::from_bits(((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) << 16) as f64
}

fn rotate(q: &mut [f64], heads: usize, position: usize, c: ModelConfig, round: bool) {
    let dim = q.len() / heads;
    for h in 0..heads {
        let values = norm(
            &q[h * dim..(h + 1) * dim],
            &vec![1.0; dim],
            c.epsilon as f64,
        );
        for i in 0..dim / 2 {
            let angle = position as f64 * (c.rope_base as f64).powf(-2.0 * i as f64 / dim as f64);
            let a = values[i] * angle.cos() - values[i + dim / 2] * angle.sin();
            let b = values[i] * angle.sin() + values[i + dim / 2] * angle.cos();
            q[h * dim + i] = if round { rounded_bf16(a) } else { a };
            q[h * dim + i + dim / 2] = if round { rounded_bf16(b) } else { b };
        }
    }
}

// Independent dense f64 graph. No Metal kernels or production graph orchestration.
fn reference_stack(m: &Model, input: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let c = m.config;
    let d = c.width as usize;
    let hd = d / c.heads as usize;
    let mut x = input.to_vec();
    for (index, layer) in m.layers.iter().enumerate() {
        let normalized: Vec<_> = x
            .iter()
            .map(|r| norm(r, &weights(m, layer.norm_attn), c.epsilon as f64))
            .collect();
        let q: Vec<_> = normalized
            .iter()
            .enumerate()
            .map(|(pos, r)| {
                let mut a = project(&weights(m, layer.q), r);
                rotate(&mut a, c.heads as usize, pos, c, false);
                a
            })
            .collect();
        let k: Vec<_> = normalized
            .iter()
            .enumerate()
            .map(|(pos, r)| {
                let mut a = project(&weights(m, layer.k), r);
                rotate(&mut a, c.kv_heads as usize, pos, c, true);
                a
            })
            .collect();
        let v: Vec<Vec<_>> = normalized
            .iter()
            .map(|r| {
                project(&weights(m, layer.v), r)
                    .into_iter()
                    .map(rounded_bf16)
                    .collect()
            })
            .collect();
        for pos in 0..x.len() {
            let gates = project(&weights(m, layer.head_gate), &normalized[pos]);
            let start = if (index + 1) % c.full_every as usize == 0 {
                0
            } else {
                (pos + 1).saturating_sub(c.local_window as usize)
            };
            let mut attended = vec![0.0; d];
            for head in 0..c.heads as usize {
                let kv_head = head / (c.heads / c.kv_heads) as usize;
                let scores: Vec<f64> = (start..=pos)
                    .map(|t| {
                        (0..hd)
                            .map(|i| q[pos][head * hd + i] * k[t][kv_head * hd + i])
                            .sum::<f64>()
                            / (hd as f64).sqrt()
                    })
                    .collect();
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let denominator: f64 = scores.iter().map(|s| (s - max).exp()).sum();
                for (offset, t) in (start..=pos).enumerate() {
                    let probability = (scores[offset] - max).exp() / denominator;
                    for i in 0..hd {
                        attended[head * hd + i] +=
                            probability * v[t][kv_head * hd + i] / (1.0 + (-gates[head]).exp());
                    }
                }
            }
            let branch = project(&weights(m, layer.out), &attended);
            for i in 0..d {
                x[pos][i] += c.residual_scale as f64 * branch[i];
            }
            let n = norm(&x[pos], &weights(m, layer.norm_ffn), c.epsilon as f64);
            let gate = project(&weights(m, layer.gate), &n);
            let up = project(&weights(m, layer.up), &n);
            let hidden: Vec<_> = gate
                .iter()
                .zip(up)
                .map(|(&g, u)| g / (1.0 + (-g).exp()) * u)
                .collect();
            let branch = project(&weights(m, layer.down), &hidden);
            for i in 0..d {
                x[pos][i] += c.residual_scale as f64 * branch[i];
            }
        }
    }
    let states: Vec<_> = x
        .iter()
        .map(|r| norm(r, &weights(m, m.final_norm), c.epsilon as f64))
        .collect();
    let logits = states
        .iter()
        .map(|r| project(&weights(m, m.embedding), r))
        .collect();
    (states, logits)
}

fn feedback(m: &Model, token: &[f64], previous: &[f64]) -> Vec<f64> {
    let normalize = |v: &[f64], mode| match mode {
        InputNorm::None => v.to_vec(),
        InputNorm::UnitRms { epsilon } => norm(v, &vec![1.0; v.len()], epsilon as f64),
    };
    let w = weights(m, m.feedback_weights);
    let matrix = (m.config.width * m.config.width) as usize;
    let u = project(&w[..matrix], previous);
    let gate = project(
        &w[matrix..],
        &normalize(token, m.config.feedback_token_norm),
    );
    let out: Vec<_> = u
        .iter()
        .zip(gate)
        .map(|(u, g)| u / (1.0 + (-g).exp()))
        .collect();
    normalize(&out, m.config.feedback_fused_norm)
}

fn reference(m: &Model, tokens: &[u32], mode: ScoreMode) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let w = weights(m, m.embedding);
    let d = m.config.width as usize;
    let mut input: Vec<_> = tokens
        .iter()
        .map(|&t| w[t as usize * d..(t as usize + 1) * d].to_vec())
        .collect();
    if mode == ScoreMode::Fused {
        let previous = reference_stack(m, &input).0;
        for i in 1..input.len() {
            input[i] = feedback(m, &input[i], &previous[i - 1]);
        }
    } else if mode == ScoreMode::Sequential {
        for i in 1..input.len() {
            let previous = reference_stack(m, &input[..i]).0;
            input[i] = feedback(m, &input[i], &previous[i - 1]);
        }
    }
    reference_stack(m, &input)
}

#[test]
fn whole_modelparity() {
    autoreleasepool(|| {
        for mode in [ScoreMode::Standard, ScoreMode::Fused, ScoreMode::Sequential] {
            let n = if mode == ScoreMode::Sequential { 7 } else { 67 };
            let tokens: Vec<_> = (0..n).map(|i| (i * 3 % 11) as u32).collect();
            let targets: Vec<_> = (0..n).map(|i| ((i * 3 + 2) % 11) as u32).collect();
            let mut score_by_chunk = Vec::new();
            for chunk in [1, 3, 64] {
                let mut m = Model::new(tiny(chunk), 42).unwrap();
                let (states, logits) = reference(&m, &tokens, mode);
                let expected: Vec<_> = logits
                    .iter()
                    .zip(&targets)
                    .map(|(r, &t)| {
                        let max = r.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                        max + r.iter().map(|x| (x - max).exp()).sum::<f64>().ln() - r[t as usize]
                    })
                    .collect();
                let result = m.score(&tokens, &targets, mode).unwrap();
                let actual = data(&m.losses, n);
                for (a, b) in actual.iter().zip(&expected) {
                    assert!(
                        (a - b).abs() < 0.004,
                        "mode={mode:?} chunk={chunk} loss {a} vs {b}"
                    );
                }
                let history = usize::from(mode == ScoreMode::Fused);
                for (i, (a, b)) in data(&m.histories[history], n * 8)
                    .iter()
                    .zip(states.iter().flatten())
                    .enumerate()
                {
                    assert!(
                        (a - b).abs() < 0.006,
                        "mode={mode:?} chunk={chunk} state {i}: {a} vs {b}"
                    );
                }
                assert!((result.mean_nll - expected.iter().sum::<f64>() / n as f64).abs() < 0.003);
                assert_eq!(result.tokens, n);
                assert_eq!(result.passes, if mode == ScoreMode::Fused { 2 } else { 1 });
                assert!(result.elapsed_seconds > 0.0);
                assert_eq!(
                    result.chunks.iter().map(|c| c.rows as usize).sum::<usize>(),
                    n * result.passes as usize
                );
                assert!(
                    result
                        .chunks
                        .iter()
                        .all(|c| c.gpu_seconds.is_some_and(|s| s >= 0.0 && s.is_finite()))
                );
                assert!(result.encode_submit_seconds > 0.0);
                assert!(result.completion_wait_seconds >= 0.0);
                assert!(
                    result.encode_submit_seconds + result.completion_wait_seconds
                        <= result.elapsed_seconds
                );
                let again = m.score(&tokens, &targets, mode).unwrap();
                assert_eq!(result.mean_nll, again.mean_nll);
                score_by_chunk.push(result.mean_nll);
            }
            assert!(
                score_by_chunk
                    .iter()
                    .all(|s| (s - score_by_chunk[0]).abs() < 0.003)
            );
        }
    });
}

#[test]
fn whole_modelupdates() {
    autoreleasepool(|| {
        let mut m = Model::new(tiny(3), 7).unwrap();
        let tokens = [1, 2, 3, 4];
        let targets = [2, 3, 4, 5];
        let before = m.score(&tokens, &targets, ScoreMode::Fused).unwrap();
        assert!(m.score(&[], &[], ScoreMode::Standard).is_err());
        assert!(m.score(&[11], &[0], ScoreMode::Standard).is_err());
        assert!(m.score(&[1], &[], ScoreMode::Standard).is_err());
        assert!(
            m.score(&vec![1; 68], &vec![2; 68], ScoreMode::Standard)
                .is_err()
        );
        assert!(m.replace_parameter("missing", &[]).is_err());
        assert!(m.replace_parameter("final_norm", &[0x7f80; 8]).is_err());
        assert!(m.replace_parameter("final_norm", &[0; 7]).is_err());
        assert_eq!(
            m.score(&tokens, &targets, ScoreMode::Fused)
                .unwrap()
                .mean_nll,
            before.mean_nll
        );
        m.replace_parameter("embedding_tied_head", &vec![0; 11 * 8])
            .unwrap();
        for mode in [ScoreMode::Standard, ScoreMode::Fused, ScoreMode::Sequential] {
            let result = m.score(&tokens, &targets, mode).unwrap();
            assert!((result.mean_nll - 11f64.ln()).abs() < 1e-6);
        }
        assert_eq!(m.parameters().len(), 3 + 10 * 2);
        let expected =
            11 * 8 + 2 * 8 * 8 + 8 + 2 * (2 * 8 + 8 * 8 * 2 + 4 * 8 * 2 + 2 * 8 + 3 * 13 * 8);
        assert_eq!(m.parameter_count(), expected);
        let memory = tiny(3).memory().unwrap();
        assert_eq!(memory.parameters, expected as u64);
        assert_eq!(memory.weight_bytes, expected as u64 * 2);
        assert_eq!(memory.kv_bytes, 2 * 67 * 4 * 4);
        assert_eq!(
            memory.total_bytes,
            memory.weight_bytes + memory.kv_bytes + memory.workspace_bytes
        );
        assert!(memory.largest_buffer_bytes <= memory.total_bytes);
        let mut bad = tiny(1);
        bad.heads = 0;
        assert!(bad.validate().is_err());
        bad = tiny(1);
        bad.kv_heads = 3;
        assert!(bad.validate().is_err());
    });
}

#[test]
fn model_resources() {
    // Intentionally no caller-provided autorelease pool.
    let mut model = Model::new(tiny(3), 13).unwrap();
    autoreleasepool(|| {
        let command = model.runtime.queue.new_command_buffer().to_owned();
        assert!(gpu_seconds(&command).is_none());
        command.commit();
        command.wait_until_completed();
    });
    let first = model
        .score(&[1, 2, 3, 4], &[2, 3, 4, 5], ScoreMode::Fused)
        .unwrap();
    for _ in 0..3 {
        let score = model
            .score(&[1, 2, 3, 4], &[2, 3, 4, 5], ScoreMode::Fused)
            .unwrap();
        assert_eq!(first.mean_nll, score.mean_nll);
        assert_eq!(score.chunks.len(), 4);
        assert_eq!(score.chunks[1].rows, 1);
        assert_eq!(score.chunks[2].pass, 1);
        assert_eq!(score.chunks[2].start, 0);
    }
}

#[test]
fn model_attention() {
    autoreleasepool(|| {
        let tokens: Vec<_> = (0..35).map(|i| (i * 3 % 11) as u32).collect();
        let targets: Vec<_> = (0..35).map(|i| ((i * 3 + 2) % 11) as u32).collect();
        for mode in [ScoreMode::Standard, ScoreMode::Fused] {
            let mut c = tiny(17);
            c.width = 192;
            let mut baseline = Model::new(c, 42).unwrap();
            let expected = baseline.score(&tokens, &targets, mode).unwrap();
            let (states, logits) = reference(&baseline, &tokens, mode);
            c.tiled_attention = true;
            let mut tiled = Model::new(c, 42).unwrap();
            let actual = tiled.score(&tokens, &targets, mode).unwrap();
            assert!((actual.mean_nll - expected.mean_nll).abs() < 0.003);
            let history = usize::from(mode == ScoreMode::Fused);
            for (a, b) in data(&tiled.histories[history], 35 * 192)
                .iter()
                .zip(states.iter().flatten())
            {
                assert!((a - b).abs() < 0.006, "tiled model state {a} vs {b}");
            }
            for ((a, r), &target) in data(&tiled.losses, 35).iter().zip(&logits).zip(&targets) {
                let max = r.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let loss =
                    max + r.iter().map(|v| (v - max).exp()).sum::<f64>().ln() - r[target as usize];
                assert!((a - loss).abs() < 0.004);
            }
        }
        let mut c = tiny(3);
        c.tiled_attention = true;
        assert!(c.validate().is_err());
    });
}

#[test]
#[ignore = "whole 1B LocalV1 model, full-context two-pass scoring, several GB and minutes"]
fn model_targettiming() {
    autoreleasepool(|| {
        let contexts: Vec<usize> = std::env::var("ENNX_FBT_CONTEXTS")
            .unwrap_or_else(|_| "4096,16384,32768".into())
            .split(',')
            .map(|s| s.parse().expect("context must be an integer"))
            .collect();
        assert!(!contexts.is_empty() && contexts.iter().all(|n| [4096, 16384, 32768].contains(n)));
        let tiled_attention = match std::env::var("ENNX_FBT_TILED_ATTENTION").as_deref() {
            Ok("0") => false,
            Ok("1") | Err(_) => true,
            _ => panic!("ENNX_FBT_TILED_ATTENTION must be 0 or 1"),
        };
        let c = ModelConfig {
            width: 1536,
            intermediate: 6656,
            layers: 24,
            vocab: 100352,
            heads: 16,
            kv_heads: 8,
            capacity: 32768,
            chunk: 256,
            local_window: 2048,
            full_every: 6,
            epsilon: 1e-5,
            rope_base: 10000.0,
            residual_scale: 1.0 / (48.0f32).sqrt(),
            feedback_token_norm: InputNorm::UnitRms { epsilon: 1e-5 },
            feedback_fused_norm: InputNorm::UnitRms { epsilon: 1e-5 },
            tiled_attention,
        };
        let start = std::time::Instant::now();
        let mut model = Model::new(c, 42).unwrap();
        eprintln!(
            "FBT LocalV1 initialized parameters={} BF16_GiB={:.4} Metal_allocated_GiB={:.4} seconds={:.3}",
            model.parameter_count(),
            model.parameter_count() as f64 * 2.0 / 1073741824.0,
            model.runtime.device.current_allocated_size() as f64 / 1073741824.0,
            start.elapsed().as_secs_f64()
        );
        // Warm all layer/projection/readout pipelines without timing initialization.
        model
            .score(&vec![1; 256], &vec![2; 256], ScoreMode::Fused)
            .unwrap();
        for context in contexts {
            let tokens: Vec<_> = (0..context)
                .map(|i| (i * 7919 % c.vocab as usize) as u32)
                .collect();
            let targets: Vec<_> = (0..context)
                .map(|i| ((i + 1) * 7919 % c.vocab as usize) as u32)
                .collect();
            eprintln!(
                "FBT LocalV1 START context={context} mode=Fused layers=24 candidates=1 tiled_attention={tiled_attention}"
            );
            let score = model.score(&tokens, &targets, ScoreMode::Fused).unwrap();
            assert!(score.mean_nll.is_finite());
            eprintln!(
                "FBT LocalV1 COMPLETE context={context} tokens_scored={} passes={} elapsed={:.6}s pass_seconds={:?} host_encode_submit={:.6}s host_completion_wait={:.6}s mean_nll={:.6}",
                score.tokens,
                score.passes,
                score.elapsed_seconds,
                score.pass_seconds,
                score.encode_submit_seconds,
                score.completion_wait_seconds,
                score.mean_nll
            );
            for pass in 0..score.passes {
                let chunks: Vec<_> = score.chunks.iter().filter(|c| c.pass == pass).collect();
                let sum: Option<f64> = chunks.iter().map(|c| c.gpu_seconds).sum();
                eprintln!(
                    "FBT GPU context={context} pass={pass} chunks={} interval_sum={sum:?} first={:?} last={:?}",
                    chunks.len(),
                    chunks.first().and_then(|c| c.gpu_seconds),
                    chunks.last().and_then(|c| c.gpu_seconds)
                );
            }
        }
    });
}
