use super::*;
use crate::config::GenerationReward;
use metal::objc::rc::autoreleasepool;

#[test]
fn context_chunks() -> Result<(), String> {
    autoreleasepool(|| {
        let runtime = Runtime::shared()?;
        let regular = decode::Decoder::new(&runtime)?;
        let compact = decode::Decoder::with_cache(&runtime, CONTEXT, true)?;
        let verifier = BlockDecoder::new(&runtime)?;
        let model =
            CandidateWeights::seeded_for(&runtime, Some(17), ResidualArchitecture::LoopedMhc4);
        let row = runtime.buffer::<u16>(MHC_FULLPARAMS);
        let mut offset = 0usize;
        for (_, tensor, _) in model.tensors() {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    tensor.contents().cast::<u8>(),
                    row.contents().cast::<u8>().add(offset),
                    tensor.length() as usize,
                );
            }
            offset += tensor.length() as usize;
        }
        let weights = model.row(&row)?;
        let readout = scorer::ProposalReadout {
            output: &verifier.proposals,
            seeds: &verifier.seeds,
            temperature: 0.8,
            score_targets: true,
            feedback: crate::config::FeedbackTransition::Identity,
        };
        unsafe {
            verifier.seeds.contents().cast::<u64>().write(17);
        }
        let command = runtime.queue.new_command_buffer();
        scorer::encode_proposals(
            command,
            &verifier.pipelines,
            &verifier.tensorops,
            &verifier.pisa,
            &verifier.buffers,
            weights,
            readout,
            CONTEXT,
            Some(regular.cache()),
            None,
            None,
        )?;
        complete(command)?;
        let expected = unsafe {
            std::slice::from_raw_parts(
                verifier.proposals.contents().cast::<u32>(),
                CONTEXT as usize,
            )
        }
        .to_vec();
        let expected_hidden = unsafe {
            std::slice::from_raw_parts(
                verifier.buffers.normalized.contents().cast::<u16>(),
                (CONTEXT * WIDTH) as usize,
            )
        }
        .to_vec();
        let expected_losses = unsafe {
            std::slice::from_raw_parts(
                verifier.buffers.losses.contents().cast::<f32>(),
                CONTEXT as usize,
            )
        }
        .to_vec();
        let mut max_error = 0.0_f64;
        for start in (0..CONTEXT).step_by(128) {
            let command = runtime.queue.new_command_buffer();
            scorer::context_chunk(
                command,
                &verifier.pipelines,
                &verifier.tensorops,
                &verifier.pisa,
                &verifier.buffers,
                weights,
                readout,
                start,
                128,
                compact.cache(),
                false,
            )?;
            complete(command)?;
            let hidden = unsafe {
                std::slice::from_raw_parts(
                    verifier.buffers.normalized.contents().cast::<u16>(),
                    (128 * WIDTH) as usize,
                )
            };
            for (index, &actual) in hidden.iter().enumerate() {
                max_error = max_error.max(
                    (decode_half(actual)
                        - decode_half(expected_hidden[(start * WIDTH) as usize + index]))
                    .abs(),
                );
            }
        }
        let actual = unsafe {
            std::slice::from_raw_parts(
                verifier.proposals.contents().cast::<u32>(),
                CONTEXT as usize,
            )
        };
        let losses = unsafe {
            std::slice::from_raw_parts(
                verifier.buffers.losses.contents().cast::<f32>(),
                CONTEXT as usize,
            )
        };
        assert_eq!(actual, expected);
        assert!(max_error <= 0.003, "chunked hidden error {max_error}");
        assert!(
            losses
                .iter()
                .zip(&expected_losses)
                .all(|(actual, expected)| (actual - expected).abs() <= 0.003)
        );
        for (compact, regular) in compact
            .cache()
            .iter()
            .zip(regular.cache())
            .take(weights.architecture.layer_steps().len())
        {
            let nodes = (2 * CONTEXT / PISA_BLOCK - 1) * HEAD_DIM;
            let a = unsafe {
                std::slice::from_raw_parts(compact.pyramid.contents().cast::<u16>(), nodes as usize)
            };
            let b = unsafe {
                std::slice::from_raw_parts(regular.pyramid.contents().cast::<u16>(), nodes as usize)
            };
            assert_eq!(a, b);
        }
        Ok(())
    })
}

#[test]
fn target_windows() {
    let mut stats = TargetStats::new(128);
    for position in 0..256 {
        let loss = if position == 200 { 65.0 } else { 1.0 };
        stats.push(position, loss, position != 200);
    }
    let quality = stats.finish().unwrap();
    assert_eq!(quality.window_tokens, 128);
    assert_eq!(quality.maximum_nll, 65.0);
    assert_eq!(quality.worst_window_nll, 1.5);
    assert_eq!(quality.positional_mismatches, 1);
    assert_eq!(quality.first_positional_mismatch, Some(200));
    assert_eq!(quality.mismatch_target_nll, Some(65.0));
    assert_eq!(quality.positional_accuracy, 255.0 / 256.0);
}

#[test]
fn parallel_repairs() -> Result<(), String> {
    autoreleasepool(|| {
        let runtime = Runtime::shared()?;
        let decoder = decode::Decoder::new(&runtime)?;
        let verifier = BlockDecoder::new(&runtime)?;
        let model = CandidateWeights::new(&runtime);
        let row = runtime.buffer::<u16>(FULL_PARAMETERS);
        let mut offset = 0usize;
        for (_, tensor, _) in model.tensors() {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    tensor.contents().cast::<u8>(),
                    row.contents().cast::<u8>().add(offset),
                    tensor.length() as usize,
                );
            }
            offset += tensor.length() as usize;
        }
        let weights = model.row(&row)?;
        let config = GenerationConfig {
            purpose: crate::config::GenerationPurpose::SystemsProbe,
            initialization: crate::config::ModelInitialization::Patterned,
            feedback_transition: crate::config::FeedbackTransition::ProjectedSigmoid,
            max_tokens: 8,
            temperature: 0.0,
            temperature_bounds: None,
            temperature_step: None,
            eos_token: None,
            seed: None,
            checkpoint: None,
            qualification_manifest: None,
            save_checkpoint: None,
            save_final_checkpoint: true,
            record_tensor_updates: true,
            signal_gate: None,
            verify: Default::default(),
            draft: None,
            reward: GenerationReward::ExactMatch,
            corpus_prompt: Vec::new(),
            corpus_prompt_tokens: None,
            episode_dataset: None,
            tasks: vec![GenerationTask {
                prompt: vec![1, 2, 3, 4],
                expected: vec![5],
                decoys: Vec::new(),
            }],
        };
        let sampling_seed = rand::random();
        let drafts = config
            .tasks
            .iter()
            .enumerate()
            .map(|(index, task)| {
                decoder.generate(
                    &runtime,
                    weights,
                    task,
                    &config,
                    crate::hash::splitmix64(sampling_seed ^ index as u64),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut corrupted = drafts.clone();
        corrupted[0].tokens[6] = (corrupted[0].tokens[6] + 1) % VOCAB;
        let verified = verifier.verify(
            &runtime,
            &decoder,
            weights,
            &config.tasks,
            &config,
            sampling_seed,
            &corrupted,
        )?;
        assert_eq!(verified[0].tokens, drafts[0].tokens);
        assert_eq!(verified[0].broad_passes, 2);
        assert_eq!(verified[0].evaluated_positions, 2 * CONTEXT as usize);
        let fixed = verifier.verify(
            &runtime,
            &decoder,
            weights,
            &config.tasks,
            &config,
            sampling_seed,
            &verified,
        )?;
        assert_eq!(
            verified
                .iter()
                .map(|rollout| &rollout.tokens)
                .collect::<Vec<_>>(),
            fixed
                .iter()
                .map(|rollout| &rollout.tokens)
                .collect::<Vec<_>>()
        );
        assert!(
            fixed
                .iter()
                .all(|rollout| rollout.evaluated_positions == CONTEXT as usize)
        );
        eprintln!(
            "ENNX_BLOCK_VERIFY rows={} passes={} gpu_ms={:.3} wall_ms={:.3}",
            CONTEXT,
            fixed[0].broad_passes,
            fixed[0].gpu_seconds * 1000.0,
            fixed[0].wall_seconds * 1000.0,
        );
        let expected = unsafe {
            std::slice::from_raw_parts(
                verifier.proposals.contents().cast::<u32>(),
                CONTEXT as usize,
            )
        }
        .to_vec();
        let suffix_start = CONTEXT / 2;
        let command = runtime.queue.new_command_buffer();
        scorer::suffix_proposals(
            command,
            &verifier.pipelines,
            &verifier.tensorops,
            &verifier.pisa,
            &verifier.buffers,
            weights,
            scorer::ProposalReadout {
                output: &verifier.proposals,
                seeds: &verifier.seeds,
                temperature: config.temperature,
                score_targets: false,
                feedback: config.feedback_transition,
            },
            suffix_start,
            CONTEXT - suffix_start,
            decoder.cache(),
        )?;
        let suffix_gpu_seconds = complete(command)?;
        let actual = unsafe {
            std::slice::from_raw_parts(
                verifier.proposals.contents().cast::<u32>(),
                CONTEXT as usize,
            )
        };
        assert_eq!(
            &actual[suffix_start as usize..],
            &expected[suffix_start as usize..]
        );
        eprintln!(
            "ENNX_BLOCK_SUFFIX rows={} gpu_ms={:.3}",
            CONTEXT - suffix_start,
            suffix_gpu_seconds * 1000.0,
        );
        let trace = scorer::ScorerStageTrace::new(&runtime)?;
        let command = runtime.queue.new_command_buffer();
        scorer::encode_proposals(
            command,
            &verifier.pipelines,
            &verifier.tensorops,
            &verifier.pisa,
            &verifier.buffers,
            weights,
            scorer::ProposalReadout {
                output: &verifier.proposals,
                seeds: &verifier.seeds,
                temperature: config.temperature,
                score_targets: false,
                feedback: config.feedback_transition,
            },
            CONTEXT,
            Some(decoder.cache()),
            None,
            Some(&trace),
        )?;
        trace.resolve(command);
        let (mut cpu_start, mut gpu_start) = (0, 0);
        runtime
            .device
            .sample_timestamps(&mut cpu_start, &mut gpu_start);
        complete(command)?;
        let (mut cpu_end, mut gpu_end) = (0, 0);
        runtime.device.sample_timestamps(&mut cpu_end, &mut gpu_end);
        let scale = cpu_end
            .checked_sub(cpu_start)
            .ok_or("CPU timestamp reversed")? as f64
            / gpu_end
                .checked_sub(gpu_start)
                .ok_or("GPU timestamp reversed")? as f64;
        eprintln!("ENNX_BLOCK_STAGES {:?}", trace.durations_ms(scale)?);
        Ok(())
    })
}

#[test]
fn generation_fixedpoint() -> Result<(), String> {
    autoreleasepool(|| {
        let runtime = Runtime::shared()?;
        let decoder = decode::Decoder::new(&runtime)?;
        let verifier = BlockDecoder::new(&runtime)?;
        let model = CandidateWeights::new(&runtime);
        let row = runtime.buffer::<u16>(FULL_PARAMETERS);
        let mut offset = 0usize;
        for (_, tensor, _) in model.tensors() {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    tensor.contents().cast::<u8>(),
                    row.contents().cast::<u8>().add(offset),
                    tensor.length() as usize,
                );
            }
            offset += tensor.length() as usize;
        }
        let weights = model.row(&row)?;
        let config = GenerationConfig {
            purpose: crate::config::GenerationPurpose::SystemsProbe,
            initialization: crate::config::ModelInitialization::Patterned,
            feedback_transition: crate::config::FeedbackTransition::ProjectedSigmoid,
            max_tokens: 8,
            temperature: 0.0,
            temperature_bounds: None,
            temperature_step: None,
            eos_token: None,
            seed: None,
            checkpoint: None,
            qualification_manifest: None,
            save_checkpoint: None,
            save_final_checkpoint: true,
            record_tensor_updates: true,
            signal_gate: None,
            verify: Default::default(),
            draft: None,
            reward: GenerationReward::ExactMatch,
            corpus_prompt: Vec::new(),
            corpus_prompt_tokens: None,
            episode_dataset: None,
            tasks: vec![GenerationTask {
                prompt: vec![1],
                expected: vec![2],
                decoys: Vec::new(),
            }],
        };
        let sampling_seed = rand::random();
        let generated = verifier.generate(
            &runtime,
            &decoder,
            weights,
            &config.tasks,
            &config,
            sampling_seed,
        )?;
        let verified = verifier.verify(
            &runtime,
            &decoder,
            weights,
            &config.tasks,
            &config,
            sampling_seed,
            &generated,
        )?;
        assert_eq!(verified[0].tokens, generated[0].tokens);
        assert_eq!(verified[0].broad_passes, 1);
        assert_eq!(verified[0].evaluated_positions, CONTEXT as usize);
        Ok(())
    })
}
