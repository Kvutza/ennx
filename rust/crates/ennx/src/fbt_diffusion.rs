//! Learned block proposals followed by causal target verification.
use super::*;
use crate::config::{DiffusionConfig, GenerationConfig, GenerationTask};
use crate::forward_program::diffusion::{DiffusionMetrics, MASK};

#[derive(Clone, Copy)]
pub(super) struct Input<'a> {
    pub confidence: &'a BufferRef,
    pub moments: &'a BufferRef,
    pub config: DiffusionConfig,
    pub visits: u32,
    pub fresh: bool,
    pub trace: Option<&'a scorer::ScorerStageTrace>,
}

struct Session<'a> {
    engine: &'a block_decode::BlockDecoder,
    runtime: &'a Runtime,
    decoder: &'a decode::Decoder,
    weights: CandidateRow<'a>,
    generation: &'a GenerationConfig,
    config: DiffusionConfig,
    confidence: Buffer,
    moments: Buffer,
    metrics: DiffusionMetrics,
    gpu_seconds: f64,
    evaluated: usize,
    visits: u32,
    seed: u64,
    trace: Option<scorer::ScorerStageTrace>,
}

fn words(buffer: &BufferRef) -> &mut [u32] {
    // SAFETY: Session submits and completes GPU work before host access. The
    // mutable view is scoped to that access; no other CPU view may be retained.
    unsafe {
        std::slice::from_raw_parts_mut(
            buffer.contents().cast::<u32>(),
            buffer.length() as usize / 4,
        )
    }
}

fn floats(buffer: &BufferRef) -> &mut [f32] {
    unsafe {
        std::slice::from_raw_parts_mut(
            buffer.contents().cast::<f32>(),
            buffer.length() as usize / 4,
        )
    }
}

impl Session<'_> {
    fn forward(
        &mut self,
        start: u32,
        rows: u32,
        step: u32,
        hidden: bool,
        scored: bool,
    ) -> Result<(), String> {
        unsafe {
            self.engine
                .seeds
                .contents()
                .cast::<u64>()
                .write(crate::hash::splitmix64(
                    if !hidden && step + 1 == self.config.steps {
                        self.seed
                    } else {
                        self.seed ^ u64::from(step) ^ 0x6469_6666
                    },
                ));
        }
        let command = self.runtime.queue.new_command_buffer();
        scorer::denoise_chunk(
            command,
            &self.engine.pipelines,
            &self.engine.tensorops,
            &self.engine.pisa,
            &self.engine.buffers,
            self.weights,
            scorer::ProposalReadout {
                output: &self.engine.proposals,
                seeds: &self.engine.seeds,
                temperature: self.generation.temperature,
                score_targets: scored,
                feedback: self.generation.feedback_transition,
            },
            start,
            rows,
            self.decoder.cache(),
            hidden,
            Input {
                confidence: &self.confidence,
                moments: &self.moments,
                config: self.config,
                visits: self.visits,
                fresh: step == 0,
                trace: self.trace.as_ref(),
            },
        )?;
        let (mut cpu_start, mut gpu_start) = (0, 0);
        if let Some(trace) = &self.trace {
            trace.resolve(command);
            self.runtime
                .device
                .sample_timestamps(&mut cpu_start, &mut gpu_start);
        }
        let seconds = complete(command)?;
        if let Some(trace) = &self.trace {
            let (mut cpu_end, mut gpu_end) = (0, 0);
            self.runtime
                .device
                .sample_timestamps(&mut cpu_end, &mut gpu_end);
            let cpu_span = cpu_end
                .checked_sub(cpu_start)
                .ok_or("nonmonotonic CPU timestamp")?;
            let gpu_span = gpu_end
                .checked_sub(gpu_start)
                .filter(|span| *span > 0)
                .ok_or("invalid GPU timestamp calibration")?;
            let stages = trace.durations_ms(cpu_span as f64 / gpu_span as f64)?;
            let record = ennx_wire::json::json!({
                "start": start, "rows": rows, "step": step, "hidden": hidden,
                "visits": self.visits,
                "gpu_ms": seconds * 1000.0, "stage_ms": stages
            });
            eprintln!(
                "ENNX_DIFFUSION_STAGES {}",
                ennx_wire::json::to_string(&record).map_err(|error| error.to_string())?
            );
        }
        self.gpu_seconds += seconds;
        if hidden {
            self.metrics.cache_positions += rows as usize;
            self.metrics.cache_gpu_seconds += seconds;
        } else {
            self.metrics.generation_positions += rows as usize;
            self.metrics.generation_gpu_seconds += seconds;
        }
        self.evaluated += rows as usize;
        Ok(())
    }

    fn prefix(&mut self, task: &GenerationTask) -> Result<(), String> {
        words(&self.engine.buffers.tokens).fill(MASK);
        words(&self.engine.buffers.tokens)[..task.prompt.len()].copy_from_slice(&task.prompt);
        floats(&self.confidence).fill(0.0);
        floats(&self.confidence)[..task.prompt.len()].fill(1.0);
        for start in (0..task.prompt.len()).step_by(4096) {
            self.forward(
                start as u32,
                (task.prompt.len() - start).min(4096) as u32,
                0,
                true,
                false,
            )?;
        }
        Ok(())
    }

    fn rollout(&mut self, task: &GenerationTask) -> Result<Vec<u32>, String> {
        self.prefix(task)?;
        let prompt = task.prompt.len();
        let maximum = self.generation.max_tokens as usize;
        let mut tokens = Vec::with_capacity(maximum);
        // Multiple blocks share a submission but each query can see only its
        // own block and the preceding blocks. Future chunks are unreadable.
        for offset in (0..maximum).step_by(4096) {
            let start = prompt + offset;
            let rows = (maximum - offset).min(4096);
            for step in 0..self.config.steps {
                self.forward(start as u32, rows as u32, step, false, false)?;
                for position in start..start + rows {
                    let proposal = words(&self.engine.proposals)[position];
                    if proposal >= VOCAB {
                        return Err("diffusion sampled outside the vocabulary".into());
                    }
                    if words(&self.engine.buffers.tokens)[position] != proposal {
                        self.metrics.changed_tokens += 1;
                    }
                    words(&self.engine.buffers.tokens)[position] = proposal;
                    let confidence = floats(&self.confidence)[position];
                    if !confidence.is_finite() {
                        return Err("nonfinite diffusion confidence".into());
                    }
                    floats(&self.confidence)[position] =
                        crate::forward_program::diffusion::confidence(
                            confidence,
                            step,
                            self.config.steps,
                            self.config.soft,
                        );
                }
                self.metrics.steps += 1;
            }
            // Final token changes invalidate their KV. Re-encode those discrete
            // tokens before a later chunk reads them. This pass is measured.
            if offset + rows < maximum {
                self.forward(start as u32, rows as u32, self.config.steps, true, false)?;
            }
            tokens.extend_from_slice(&words(&self.engine.buffers.tokens)[start..start + rows]);
            self.metrics.blocks += rows.div_ceil(self.config.block as usize);
            if maximum > 4096 && tokens.len() % 65_536 == 0 {
                eprintln!(
                    "ENNX_DIFFUSION_PROGRESS generated={} target={} evaluated={} gpu_ms={:.3}",
                    tokens.len(),
                    maximum,
                    self.evaluated,
                    self.gpu_seconds * 1000.0
                );
            }
        }
        Ok(tokens)
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn verify(
    engine: &block_decode::BlockDecoder,
    runtime: &Runtime,
    decoder: &decode::Decoder,
    weights: CandidateRow<'_>,
    tasks: &[GenerationTask],
    config: &GenerationConfig,
    seed: u64,
) -> Result<Vec<decode::Rollout>, String> {
    let started = Instant::now();
    let drafts = propose(engine, runtime, decoder, weights, tasks, config, seed)?;
    let mut target = config.clone();
    target.draft = None;
    // The first target pass starts at zero and rebuilds every readable cache
    // entry under causal visibility. Block-visible draft KV is never reused.
    let mut rollouts = engine.verify(runtime, decoder, weights, tasks, &target, seed, &drafts)?;
    for (rollout, draft) in rollouts.iter_mut().zip(drafts) {
        let mut metrics = draft.draft.ok_or("diffusion did not report draft work")?;
        metrics.accepted_prefix = rollout.accepted_lengths.first().copied().unwrap_or(0);
        metrics.matching_tokens = rollout
            .tokens
            .iter()
            .zip(&draft.tokens)
            .filter(|(target, proposal)| target == proposal)
            .count();
        rollout.evaluated_positions += draft.evaluated_positions;
        rollout.gpu_seconds += draft.gpu_seconds;
        rollout.draft = Some(metrics);
        rollout.wall_seconds = started.elapsed().as_secs_f64();
    }
    Ok(rollouts)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn propose(
    engine: &block_decode::BlockDecoder,
    runtime: &Runtime,
    decoder: &decode::Decoder,
    weights: CandidateRow<'_>,
    tasks: &[GenerationTask],
    config: &GenerationConfig,
    seed: u64,
) -> Result<Vec<decode::Rollout>, String> {
    let diffusion = config.draft.ok_or("missing diffusion configuration")?;
    if weights.architecture != ResidualArchitecture::DiffusionMhc4 || tasks.len() != 1 {
        return Err("diffusion requires its own model and exactly one task".into());
    }
    let task = &tasks[0];
    if !task.prompt.len().is_multiple_of(diffusion.block as usize)
        || config.max_tokens % diffusion.block != 0
    {
        return Err("diffusion prompt and output must align to its block".into());
    }
    if task.prompt.len() + config.max_tokens as usize > engine.context as usize {
        return Err("diffusion prompt plus generated tokens exceeds the KV capacity".into());
    }
    let started = Instant::now();
    let counts = engine.pisa.index_counts();
    let visits = crate::forward_program::diffusion::visits(diffusion, seed);
    let mut session = Session {
        engine,
        runtime,
        decoder,
        weights,
        generation: config,
        config: diffusion,
        confidence: runtime.buffer_with(&vec![0.0f32; engine.context as usize]),
        moments: runtime.buffer::<[f32; 2]>(4096 * 128),
        metrics: DiffusionMetrics {
            visits,
            ..Default::default()
        },
        gpu_seconds: 0.0,
        evaluated: 0,
        visits,
        seed,
        trace: (std::env::var("ENNX_DIFFUSION_TRACE").as_deref() == Ok("1"))
            .then(|| scorer::ScorerStageTrace::new(runtime))
            .transpose()?,
    };
    let tokens = session.rollout(task)?;
    let next = engine.pisa.index_counts();
    session.metrics.index_reused = next[0] - counts[0];
    session.metrics.index_refreshed = next[1] - counts[1];
    session.metrics.wall_seconds = started.elapsed().as_secs_f64();
    session.metrics.generated_tokens = tokens.len();
    session.metrics.evaluated_positions = session.evaluated;
    eprintln!(
        "ENNX_DIFFUSION_DRAFT generated={} evaluated={} visits={} steps={} index_reused={} index_refreshed={}",
        tokens.len(),
        session.evaluated,
        visits,
        diffusion.steps,
        session.metrics.index_reused,
        session.metrics.index_refreshed
    );
    Ok(vec![decode::Rollout {
        tokens,
        draft: Some(session.metrics),
        finish_reason: "length",
        wall_seconds: started.elapsed().as_secs_f64(),
        gpu_seconds: session.gpu_seconds,
        evaluated_positions: session.evaluated,
        committed_tokens: 0,
        broad_passes: 0,
        correction_waves: 0,
        repair_batches: 0,
        accepted_tokens: 0,
        first_mismatch: None,
        evaluated_lengths: Vec::new(),
        accepted_lengths: Vec::new(),
        committed_lengths: Vec::new(),
        route_samples: Vec::new(),
        free_running_target_nll: None,
        target_quality: None,
    }])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn draft_target() -> Result<(), String> {
        metal::objc::rc::autoreleasepool(|| {
            let runtime = Runtime::shared()?;
            let model = CandidateWeights::seeded_for(
                &runtime,
                Some(17),
                ResidualArchitecture::DiffusionMhc4,
            );
            let row = runtime.buffer::<u16>(model.architecture.parameter_count());
            let mut offset = 0;
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
            let mut config: GenerationConfig = ennx_wire::toml::from_str(
                "purpose='systems-probe'\nmax-tokens=128\ntemperature=0.8\n[reward]\nkind='token-accuracy'\n[draft]\nsteps=1\nvisits=[1,1]",
            ).map_err(|error| error.to_string())?;
            config.tasks = vec![GenerationTask {
                prompt: (0..128).map(|token| token % VOCAB).collect(),
                expected: vec![7; 128],
                decoys: Vec::new(),
            }];
            config.validate()?;
            let compact = decode::Decoder::with_visits(&runtime, CONTEXT, true, 7)?;
            let direct = decode::Decoder::with_visits(&runtime, CONTEXT, false, 7)?;
            let engine = block_decode::BlockDecoder::new(&runtime)?;
            let result = verify(
                &engine,
                &runtime,
                &compact,
                weights,
                &config.tasks,
                &config,
                17,
            )?;
            let mut target = config.clone();
            target.draft = None;
            let expected = direct.generate(
                &runtime,
                weights,
                &target.tasks[0],
                &target,
                crate::hash::splitmix64(17),
            )?;
            assert_eq!(result[0].tokens, expected.tokens);
            assert_eq!(result[0].committed_tokens, 128);
            let draft = result[0].draft.as_ref().ok_or("missing draft metrics")?;
            assert_eq!(draft.generated_tokens, 128);
            assert!(result[0].evaluated_positions > draft.evaluated_positions);
            assert!(result[0].wall_seconds >= draft.wall_seconds);
            // Reference labels can affect scoring but cannot enter generation.
            config.tasks[0].expected.fill(31);
            let changed = verify(
                &engine,
                &runtime,
                &compact,
                weights,
                &config.tasks,
                &config,
                17,
            )?;
            assert_eq!(result[0].tokens, changed[0].tokens);
            // Equal zero-logit distributions must share every stochastic draw.
            unsafe {
                std::ptr::write_bytes(
                    row.contents().cast::<u8>().add(weights.readout as usize),
                    0,
                    (WIDTH * VOCAB * 2) as usize,
                );
            }
            let coupled = verify(
                &engine,
                &runtime,
                &compact,
                weights,
                &config.tasks,
                &config,
                17,
            )?;
            assert_eq!(coupled[0].draft.as_ref().unwrap().accepted_prefix, 128);
            assert_eq!(coupled[0].draft.as_ref().unwrap().matching_tokens, 128);
            assert_eq!(coupled[0].correction_waves, 0);
            Ok(())
        })
    }

    #[test]
    fn inputs() {
        metal::objc::rc::autoreleasepool(|| {
            let runtime = Runtime::shared().unwrap();
            let shader = include_str!("fbt_denoise.metal");
            let kernel = runtime
                .pipeline(shader, "diffusion test", "denoise_embed")
                .unwrap();
            let weights = runtime.buffer_with(&vec![0x3800u16; 512 * 8192]);
            let mask = runtime.buffer_with(&vec![0x3400u16; 512]);
            let tokens = runtime.buffer_with(&[MASK, 1, 1]);
            let confidence = runtime.buffer_with(&[0.0f32, 0.5, 1.0]);
            let output = runtime.buffer::<u16>(3 * 512);
            let command = runtime.queue.new_command_buffer();
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&kernel);
            for (index, buffer) in [&weights, &tokens, &output, &mask, &confidence]
                .into_iter()
                .enumerate()
            {
                encoder.set_buffer(index as u64, Some(buffer), 0);
            }
            encoder.dispatch_thread_groups(thread_group(3), thread_group(128));
            encoder.end_encoding();
            complete(command).unwrap();
            let values =
                unsafe { std::slice::from_raw_parts(output.contents().cast::<u16>(), 3 * 512) };
            for (row, expected) in [0.25f64, 0.02, 0.5].into_iter().enumerate() {
                assert!(
                    values[row * 512..(row + 1) * 512]
                        .iter()
                        .all(|value| (decode_half(*value) - expected).abs() < 0.00002)
                );
            }
        });
    }

    #[test]
    fn probability() {
        metal::objc::rc::autoreleasepool(|| {
            let runtime = Runtime::shared().unwrap();
            let kernel = runtime
                .pipeline(
                    include_str!("fbt_denoise.metal"),
                    "diffusion test",
                    "denoise_reduce",
                )
                .unwrap();
            let mut partials = vec![[0.0f32; 4]; 128];
            let mut moments = vec![[0.5f32, 64.0]; 128];
            for (tile, partial) in partials.iter_mut().enumerate() {
                *partial = [0.0, f32::from_bits((tile * 64) as u32), 1.0, 0.0];
            }
            partials[7][0] = 10.0;
            partials[7][2] = 3.0;
            moments[7] = [1.5, 1.0 + 63.0 * (-1.0f32).exp()];
            let partials = runtime.buffer_with(&partials);
            let moments = runtime.buffer_with(&moments);
            let tokens = runtime.buffer::<u32>(1);
            let confidence = runtime.buffer::<f32>(1);
            let temperature = 2.0f32;
            let command = runtime.queue.new_command_buffer();
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&kernel);
            for (index, buffer) in [&partials, &tokens, &confidence, &moments]
                .into_iter()
                .enumerate()
            {
                encoder.set_buffer(index as u64, Some(buffer), 0);
            }
            encoder.set_bytes(4, 4, std::ptr::from_ref(&temperature).cast());
            encoder.dispatch_thread_groups(thread_group(1), thread_group(128));
            encoder.end_encoding();
            complete(command).unwrap();
            let expected = 1.0f32.exp() / (8191.0 + 1.0f32.exp());
            assert_eq!(words(&tokens)[0], 448);
            assert!((floats(&confidence)[0] - expected).abs() < 1e-7);
        });
    }
}
