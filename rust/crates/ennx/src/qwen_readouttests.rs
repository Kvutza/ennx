use super::*;

#[test]
fn readout_coverage() -> Result<()> {
    autoreleasepool(|| {
        let c = QwenConfig {
            layers: 1,
            hidden: 32,
            intermediate: 64,
            heads: 2,
            kv_heads: 1,
            vocab: 64,
            eos_token_id: 63,
            context: 260,
            epsilon: 1e-6,
            rope_theta: 10000.0,
        };
        let mut evaluator = QwenEvaluator::backend_options(c, 260, MpsMode::Off, false)?;
        let values = (0..evaluator.weights_len())
            .map(|index| ((((index * 7 % 29) as f32 - 14.0) / 128.0).to_bits() >> 16) as u16)
            .collect::<Vec<_>>();
        let weights = evaluator.runtime.buffer_with(&values);
        let lengths = [31usize, 35, 128, 259];
        let tokens = lengths
            .iter()
            .map(|&length| (0..length).map(|i| (i % 63) as i32).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let masks = lengths
            .iter()
            .map(|&length| {
                (0..length)
                    .map(|i| i >= 11 && i % 7 != 0)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let reference = evaluator.losses(&weights, &tokens, &masks)?;
        let prior_bytes = evaluator.workspace_bytes();
        evaluator.prepare_readout(&weights)?;
        assert_eq!(
            evaluator.workspace_bytes() - prior_bytes,
            u64::from(c.hidden) * u64::from(c.vocab) * 4
        );
        let actual = evaluator.losses(&weights, &tokens, &masks)?;
        for (actual, expected) in actual.iter().zip(&reference) {
            assert!((actual - expected).abs() <= 2e-5, "{actual} vs {expected}");
        }
        let other_values = values
            .iter()
            .map(|&value| value ^ 0x8000)
            .collect::<Vec<_>>();
        let other = evaluator.runtime.buffer_with(&other_values);
        let cached = evaluator.frozen_readout.take();
        let reference = evaluator.losses(&other, &tokens, &masks)?;
        evaluator.frozen_readout = cached;
        let actual = evaluator.losses(&other, &tokens, &masks)?;
        assert_eq!(
            actual, reference,
            "cache must not bind to another weight buffer"
        );
        assert!(evaluator.prepare_readout(&other).is_err());
        Ok(())
    })
}

#[test]
#[ignore = "full-width readout timing comparison; run explicitly"]
fn readout_timing() -> Result<()> {
    check_readout(true)
}

#[test]
#[ignore = "full-width numerical check without a performance claim"]
fn readout_numerics() -> Result<()> {
    check_readout(false)
}

fn readout_weights(
    evaluator: &mut QwenEvaluator,
    checkpoint: Option<&std::ffi::OsStr>,
) -> Result<Buffer> {
    if let Some(path) = checkpoint {
        return evaluator.load_weights(Path::new(path));
    }
    let weights = evaluator.runtime.buffer::<u16>(evaluator.weights_len());
    let elements = product(&[
        evaluator.config.hidden as usize,
        evaluator.config.vocab as usize,
    ])?;
    let values = unsafe {
        std::slice::from_raw_parts_mut(
            weights
                .contents()
                .cast::<u16>()
                .add(evaluator.layout.embedding),
            elements,
        )
    };
    for (index, value) in values.iter_mut().enumerate() {
        let bits = crate::hash::splitmix64(index as u64);
        let weight = ((bits >> 40) as f32 / (1u32 << 24) as f32 - 0.5) * 0.125;
        *value = (weight.to_bits() >> 16) as u16;
    }
    Ok(weights)
}

fn compare_outputs(
    actual: &[f32],
    reference: &[f32],
    losses: &[f32],
    reference_losses: &[f32],
) -> Result<(f32, f32)> {
    if actual.len() != reference.len() || losses.len() != reference_losses.len() {
        return Err("readout comparison has mismatched coverage".into());
    }
    let mut max_error = 0.0f32;
    let mut max_loss_error = 0.0f32;
    for (&actual, &reference) in actual.iter().zip(reference) {
        max_error = max_error.max((actual - reference).abs());
        if !actual.is_finite() || (actual - reference).abs() > 2e-4 * reference.abs().max(1.0) {
            return Err(format!("readout mismatch: {actual} vs {reference}"));
        }
    }
    for (actual, reference) in losses.iter().zip(reference_losses) {
        max_loss_error = max_loss_error.max((actual - reference).abs());
        if !actual.is_finite() || (actual - reference).abs() > 2e-5 {
            return Err(format!("readout loss mismatch: {actual} vs {reference}"));
        }
    }
    Ok((max_error, max_loss_error))
}

fn check_reference(input: &[f32], weights: &[u16], logits: &[f32], c: QwenConfig) -> Result<()> {
    if logits.iter().all(|&value| value == 0.0) {
        return Err("readout fixture unexpectedly produced only zero logits".into());
    }
    for (row, column) in [(0usize, 0usize), (63, 1729), (127, c.vocab as usize - 1)] {
        let expected = (0..c.hidden as usize)
            .map(|k| {
                f64::from(input[(row + 3) * c.hidden as usize + k])
                    * f64::from(f32::from_bits(
                        u32::from(weights[column * c.hidden as usize + k]) << 16,
                    ))
            })
            .sum::<f64>();
        let actual = f64::from(logits[row * c.vocab as usize + column]);
        if (actual - expected).abs() > 2e-4 * expected.abs().max(1.0) {
            return Err(format!(
                "reference readout disagrees with FP64 dot: {actual} vs {expected}"
            ));
        }
    }
    Ok(())
}

fn readout_input(evaluator: &QwenEvaluator, c: QwenConfig) -> Vec<f32> {
    let input = (0..256 * c.hidden as usize)
        .map(|index| {
            let bits = crate::hash::splitmix64(index as u64 ^ 0x696e_7075_74);
            (bits >> 40) as f32 / (1u32 << 24) as f32 - 0.5
        })
        .collect::<Vec<_>>();
    evaluator.write(W::Norm, &input);
    evaluator.write(W::Invalid, &[0u32]);
    evaluator.write(
        W::Tokens,
        &(0..132).map(|i| (i * 37) as i32).collect::<Vec<_>>(),
    );
    evaluator.write(W::Masks, &[1u8; 132]);
    evaluator.write(W::Losses, &[0.0f32; 132]);
    input
}

fn readout_pairs(
    evaluator: &mut QwenEvaluator,
    weights: &Buffer,
    measure: bool,
    reference: &[f32],
    reference_losses: &[f32],
) -> Result<(Option<readout::Readout>, Vec<[f64; 2]>, f32, f32)> {
    let mut cached = evaluator.frozen_readout.take();
    let mut times = Vec::new();
    let mut actual = vec![0.0f32; reference.len()];
    let mut max_error = 0.0f32;
    let mut max_loss_error = 0.0f32;
    for pair in 0..if measure { 3 } else { 1 } {
        if measure {
            crate::apple_gpu::require_power()?;
        }
        let mut elapsed = [0.0; 2];
        for branch in if pair % 2 == 0 { [0, 1] } else { [1, 0] } {
            if branch == 1 {
                evaluator.frozen_readout = cached.take();
            }
            let start = Instant::now();
            evaluator.project_rows(&weights, 131, 3, 0, 132, true, None)?;
            elapsed[branch] = start.elapsed().as_secs_f64() * 1000.0;
            let count = actual.len();
            actual.copy_from_slice(&evaluator.read::<f32>(W::Logits, count));
            let errors = compare_outputs(
                &actual,
                reference,
                &evaluator.read::<f32>(W::Losses, 132),
                reference_losses,
            )?;
            max_error = max_error.max(errors.0);
            max_loss_error = max_loss_error.max(errors.1);
            if branch == 1 {
                cached = evaluator.frozen_readout.take();
            }
        }
        times.push(elapsed);
        if measure {
            crate::apple_gpu::require_power()?;
        }
    }
    Ok((cached, times, max_error, max_loss_error))
}

fn check_readout(measure: bool) -> Result<()> {
    autoreleasepool(|| {
        if measure {
            crate::apple_gpu::require_power()?;
        }
        let checkpoint = std::env::var_os("ENNX_QWEN_CHECKPOINT");
        let mut c = QwenConfig::default();
        if checkpoint.is_none() {
            c.layers = 1;
        }
        let mut evaluator = QwenEvaluator::backend_options(c, 256, MpsMode::Off, false)?;
        let elements = product(&[c.hidden as usize, c.vocab as usize])?;
        let weights = readout_weights(&mut evaluator, checkpoint.as_deref())?;
        let values = unsafe {
            std::slice::from_raw_parts(
                weights
                    .contents()
                    .cast::<u16>()
                    .add(evaluator.layout.embedding),
                elements,
            )
        };
        let input = readout_input(&evaluator, c);
        let mut reference = vec![0.0f32; LOGIT_ROWS * c.vocab as usize];
        evaluator.project_rows(&weights, 131, 3, 0, 132, true, Some(&mut reference))?;
        check_reference(&input, values, &reference, c)?;
        let reference_losses = evaluator.read::<f32>(W::Losses, 132);
        let preparation = Instant::now();
        evaluator.prepare_readout(&weights)?;
        let preparation_ms = preparation.elapsed().as_secs_f64() * 1000.0;
        evaluator.project_rows(&weights, 131, 3, 0, 132, true, None)?;
        let (cached, times, max_error, max_loss_error) = readout_pairs(
            &mut evaluator,
            &weights,
            measure,
            &reference,
            &reference_losses,
        )?;
        if measure {
            eprintln!(
                "ENNX_READOUT_PROFILE {}",
                ennx_wire::json::to_string(&ennx_wire::json::json!({
                    "rows":LOGIT_ROWS,"hidden":c.hidden,"vocab":c.vocab,
                    "checkpoint_weights":checkpoint.is_some(),
                    "logical_matrix_operations":2u64 * LOGIT_ROWS as u64 * u64::from(c.hidden) * u64::from(c.vocab),
                    "expanded_bytes":cached.as_ref().unwrap().bytes(),"preparation_ms":preparation_ms,
                    "pairs_reference_mps_ms":times,"max_absolute_logit_error":max_error,
                    "max_absolute_token_nll_error":max_loss_error,
                })).map_err(|e| e.to_string())?
            );
        } else {
            eprintln!(
                "ENNX_READOUT_NUMERICS {}",
                ennx_wire::json::to_string(&ennx_wire::json::json!({
                    "rows":LOGIT_ROWS,"hidden":c.hidden,"vocab":c.vocab,
                    "checkpoint_weights":checkpoint.is_some(),
                    "max_absolute_logit_error":max_error,"max_absolute_token_nll_error":max_loss_error,
                    "performance_qualified":false,
                })).map_err(|e| e.to_string())?
            );
        }
        Ok(())
    })
}
