//! Exact speculative verification using the incumbent rollout as the draft.
//!
//! A full candidate forward proposes every next token in parallel. The longest
//! causally valid prefix is committed; the first mismatch is replaced by the
//! candidate's exact token and the remaining suffix is verified again.

use super::*;
use crate::config::{GenerationConfig, GenerationTask};

use super::target::TargetStats;

#[cfg(test)]
#[path = "fbt_block/tests.rs"]
mod tests;

pub(super) struct BlockDecoder {
    pub(super) context: u32,
    pub(super) pipelines: Pipelines,
    pub(super) tensorops: TensorOpsPipelines,
    pub(super) pisa: Pisa1,
    pub(super) buffers: Buffers,
    pub(super) proposals: Buffer,
    pub(super) seeds: Buffer,
}

#[derive(Default)]
struct VerificationProgress {
    cursor: usize,
    broad_passes: usize,
    correction_waves: usize,
    repair_batches: usize,
    parallel_positions: usize,
    repair_positions: usize,
    first_mismatch: Option<usize>,
    evaluated_lengths: Vec<usize>,
    accepted_lengths: Vec<usize>,
    committed_lengths: Vec<usize>,
    route_samples: Vec<decode::RouteSample>,
    gpu_seconds: f64,
    target: TargetStats,
}

struct CommitStats {
    accepted: usize,
    committed: usize,
    mismatch: Option<usize>,
}

#[derive(Default)]
struct RepairStats {
    gpu_seconds: f64,
    evaluated_positions: usize,
    correction_waves: usize,
    repair_batches: usize,
    first_mismatch: Option<usize>,
    evaluated_lengths: Vec<usize>,
    accepted_lengths: Vec<usize>,
    committed_lengths: Vec<usize>,
    route_samples: Vec<decode::RouteSample>,
}

impl RepairStats {
    fn report(&self, cursor: usize, count: usize, next: &mut usize) {
        if count > 4096 && cursor >= *next {
            eprintln!(
                "ENNX_GENERATION_PROGRESS committed={} target={} correction_waves={} gpu_ms={:.3}",
                cursor,
                count,
                self.correction_waves,
                self.gpu_seconds * 1000.0
            );
            *next = (cursor / 16_384 + 1) * 16_384;
        }
    }
}

fn route_sample(stats: routing::RouteStats) -> decode::RouteSample {
    decode::RouteSample {
        active_experts: stats.active_experts,
        routed_rows: stats.routed_rows,
        routed_tiles: stats.routed_tiles,
    }
}

fn repair_bounds(
    row: usize,
    context: u32,
    window: u32,
    prompt: usize,
    limit: usize,
) -> (u32, usize) {
    let start = ((row as u32 / 4) * 4).min(context - window);
    let end = (start as usize + window as usize)
        .saturating_sub(prompt - 1)
        .min(limit);
    (start, end)
}

fn adapt_window(
    window: u32,
    config: &GenerationConfig,
    mismatch: bool,
    accepted: usize,
    span: usize,
) -> u32 {
    if !mismatch {
        (window * 2).min(config.verify.max_window)
    } else if accepted * 4 < span {
        (window / 2).max(config.verify.window)
    } else {
        window
    }
}

fn next_window(
    window: u32,
    config: &GenerationConfig,
    mismatch: bool,
    accepted: usize,
    span: usize,
) -> (bool, u32) {
    (
        mismatch && accepted * 4 < span,
        adapt_window(window, config, mismatch, accepted, span),
    )
}

impl BlockDecoder {
    fn score_targets(config: &GenerationConfig) -> bool {
        matches!(
            config.reward,
            crate::config::GenerationReward::FreeRunningCrossEntropy
                | crate::config::GenerationReward::CodeObjectives { .. }
        )
    }

    fn loss_window(config: &GenerationConfig) -> usize {
        match &config.reward {
            crate::config::GenerationReward::CodeObjectives { critical_window } => {
                *critical_window as usize
            }
            _ => config.max_tokens as usize,
        }
    }

    fn load_targets(&self, task: &GenerationTask, maximum: usize) -> Result<(), String> {
        if task.expected.len() < maximum {
            return Err("free-running target has fewer tokens than max_tokens".into());
        }
        let labels = unsafe {
            std::slice::from_raw_parts_mut(
                self.buffers.labels.contents().cast::<u32>(),
                self.buffers.labels.length() as usize / size_of::<u32>(),
            )
        };
        for (position, &target) in task.expected[..maximum].iter().enumerate() {
            labels[task.prompt.len() - 1 + position] = target;
        }
        Ok(())
    }

    fn target_nll(
        &self,
        task: &GenerationTask,
        tokens: &[u32],
        start: usize,
        end: usize,
        stats: &mut TargetStats,
    ) -> Result<(), String> {
        let losses = unsafe {
            std::slice::from_raw_parts(
                self.buffers.losses.contents().cast::<f32>(),
                self.context as usize,
            )
        };
        for position in start..end {
            let loss = losses[task.prompt.len() - 1 + position];
            if !loss.is_finite() {
                return Err(format!(
                    "nonfinite free-running target loss at generated token {position}"
                ));
            }
            stats.push(position, loss, tokens[position] == task.expected[position]);
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn repair_waves(
        &self,
        runtime: &Runtime,
        decoder: &decode::Decoder,
        weights: CandidateRow<'_>,
        config: &GenerationConfig,
        score_targets: bool,
        row_start: u32,
        window: u32,
        cursor: usize,
        prompt: usize,
        waves: u32,
    ) -> Result<f64, String> {
        let command = runtime.queue.new_command_buffer();
        for wave in 0..waves {
            let readout = scorer::ProposalReadout {
                output: &self.proposals,
                seeds: &self.seeds,
                temperature: config.temperature,
                score_targets,
                feedback: config.feedback_transition,
            };
            if decoder.cache()[0].kv.is_some() {
                scorer::context_chunk(
                    command,
                    &self.pipelines,
                    &self.tensorops,
                    &self.pisa,
                    &self.buffers,
                    weights,
                    readout,
                    row_start,
                    window,
                    decoder.cache(),
                    false,
                )?;
            } else {
                scorer::suffix_proposals(
                    command,
                    &self.pipelines,
                    &self.tensorops,
                    &self.pisa,
                    &self.buffers,
                    weights,
                    scorer::ProposalReadout {
                        output: &self.proposals,
                        seeds: &self.seeds,
                        temperature: config.temperature,
                        score_targets,
                        feedback: config.feedback_transition,
                    },
                    row_start,
                    window,
                    decoder.cache(),
                )?;
            }
            if wave + 1 < waves {
                self.shift_draft(command, row_start, window, cursor, prompt);
            }
        }
        complete(command)
    }

    fn shift_draft(
        &self,
        command: &CommandBufferRef,
        row_start: u32,
        window: u32,
        cursor: usize,
        prompt: usize,
    ) {
        let exact_row = prompt - 1 + cursor;
        let row_end = row_start as usize + window as usize;
        let count = row_end.saturating_sub(exact_row + 1);
        if count == 0 {
            return;
        }
        let blit = command.new_blit_command_encoder();
        blit.copy_from_buffer(
            &self.proposals,
            (exact_row * size_of::<u32>()) as u64,
            &self.buffers.tokens,
            ((exact_row + 1) * size_of::<u32>()) as u64,
            (count * size_of::<u32>()) as u64,
        );
        blit.end_encoding();
    }

    fn refresh_draft(
        &self,
        tokens: &mut [u32],
        input: &[u32],
        cursor: usize,
        tile_end: usize,
        prompt: usize,
    ) {
        for position in cursor..tile_end.saturating_sub(1) {
            tokens[position] = input[prompt + position];
        }
    }

    fn proposed(&self) -> &[u32] {
        unsafe {
            std::slice::from_raw_parts(
                self.proposals.contents().cast::<u32>(),
                self.context as usize,
            )
        }
    }

    fn record_waves(stats: &mut RepairStats, window: u32, waves: u32) {
        stats.evaluated_positions += window as usize * waves as usize;
        stats.correction_waves += waves as usize;
        stats.repair_batches += 1;
        stats
            .evaluated_lengths
            .extend(std::iter::repeat_n(window as usize, waves as usize));
    }

    fn record_route(&self, stats: &mut RepairStats) {
        stats
            .route_samples
            .push(route_sample(self.buffers.fine_grained.route_stats()));
    }

    fn repair_target(
        &self,
        score: bool,
        task: &GenerationTask,
        tokens: &[u32],
        start: usize,
        end: usize,
        stats: &mut TargetStats,
    ) -> Result<(), String> {
        if score {
            self.target_nll(task, tokens, start, end, stats)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn repair_mismatch(
        &self,
        runtime: &Runtime,
        decoder: &decode::Decoder,
        weights: CandidateRow<'_>,
        task: &GenerationTask,
        config: &GenerationConfig,
        tokens: &mut [u32],
        cursor: &mut usize,
        score_targets: bool,
        target: &mut TargetStats,
    ) -> Result<RepairStats, String> {
        if *cursor >= tokens.len() {
            return Ok(RepairStats::default());
        }
        let input = unsafe {
            std::slice::from_raw_parts_mut(
                self.buffers.tokens.contents().cast::<u32>(),
                self.context as usize,
            )
        };
        let proposed = self.proposed();
        let mut stats = RepairStats::default();
        let mut window = config.verify.window;
        let mut stalled = false;
        let mut report_at = 16_384;
        while *cursor < tokens.len() {
            let (row_start, tile_end) = repair_bounds(
                task.prompt.len() - 1 + *cursor,
                self.context,
                window,
                task.prompt.len(),
                tokens.len(),
            );
            let waves = if stalled || config.verify.unroll > 1 {
                config.verify.unroll
            } else {
                1
            };
            stats.gpu_seconds += self.repair_waves(
                runtime,
                decoder,
                weights,
                config,
                score_targets,
                row_start,
                window,
                *cursor,
                task.prompt.len(),
                waves,
            )?;
            self.record_route(&mut stats);
            Self::record_waves(&mut stats, window, waves);

            if waves > 1 {
                self.refresh_draft(tokens, input, *cursor, tile_end, task.prompt.len());
            }

            // Every row through the first mismatch is causally exact. Commit
            // that mismatch as well, and retain the rest of this tile as the
            // next fixed-point draft instead of throwing away 127 results.
            let exact_start = *cursor;
            let mut position = exact_start;
            let mut committed = None;
            let mut accepted = 0usize;
            while position < tile_end {
                let prediction = proposed[task.prompt.len() - 1 + position];
                if prediction >= VOCAB {
                    return Err("block repair produced an invalid token".into());
                }
                let exact_prediction = committed.is_none();
                if exact_prediction {
                    if tokens[position] == prediction {
                        accepted += 1;
                    } else {
                        stats.first_mismatch.get_or_insert(position);
                        committed = Some(position + 1);
                    }
                }
                tokens[position] = prediction;
                if task.prompt.len() + position < self.context as usize {
                    input[task.prompt.len() + position] = prediction;
                }
                position += 1;
                if exact_prediction && Some(prediction) == config.eos_token {
                    tokens[position..].fill(prediction);
                    *cursor = tokens.len();
                    stats.accepted_lengths.push(accepted);
                    stats.committed_lengths.push(position - exact_start);
                    return Ok(stats);
                }
            }
            *cursor = committed.unwrap_or(tile_end);
            stats.report(*cursor, tokens.len(), &mut report_at);
            stats.accepted_lengths.push(accepted);
            stats.committed_lengths.push(*cursor - exact_start);
            self.repair_target(score_targets, task, tokens, exact_start, *cursor, target)?;
            (stalled, window) = next_window(
                window,
                config,
                committed.is_some(),
                accepted,
                tile_end - exact_start,
            );
        }
        Ok(stats)
    }

    pub fn new(runtime: &Runtime) -> Result<Self, String> {
        Self::with_context(runtime, CONTEXT)
    }

    pub(super) fn with_context(runtime: &Runtime, context: u32) -> Result<Self, String> {
        Ok(Self {
            context,
            pipelines: Pipelines::new(runtime)?,
            tensorops: TensorOpsPipelines::new(runtime)?,
            pisa: Pisa1::with_context(runtime, context)?,
            buffers: Buffers::for_context(runtime, context),
            proposals: runtime.buffer::<u32>(context as usize),
            seeds: runtime.buffer::<u64>(1),
        })
    }

    fn parallel_pass(
        &self,
        runtime: &Runtime,
        decoder: &decode::Decoder,
        weights: CandidateRow<'_>,
        temperature: f32,
        feedback: crate::config::FeedbackTransition,
        pass: usize,
        cursor: usize,
        total_rows: u32,
        score_targets: bool,
        prompt: usize,
    ) -> Result<(f64, u32, decode::RouteSample), String> {
        if decoder.cache()[0].kv.is_some() {
            let start = if pass == 0 {
                0
            } else {
                ((prompt - 1 + cursor) as u32 / 4) * 4
            };
            let mut position = start;
            let mut seconds = 0.0;
            let mut evaluated = 0;
            let mut in_flight = Vec::new();
            while position < total_rows {
                let rows = (total_rows - position).min(4096);
                // Later broad passes need 128-row tiles; the end can extend into
                // initialized scratch padding but never beyond the cache.
                let rows = rows.div_ceil(128) * 128;
                let row_start = position.min(self.context - rows);
                let command = runtime.queue.new_command_buffer().to_owned();
                scorer::context_chunk(
                    &command,
                    &self.pipelines,
                    &self.tensorops,
                    &self.pisa,
                    &self.buffers,
                    weights,
                    scorer::ProposalReadout {
                        output: &self.proposals,
                        seeds: &self.seeds,
                        temperature,
                        score_targets,
                        feedback,
                    },
                    row_start,
                    rows,
                    decoder.cache(),
                    row_start as usize + rows as usize <= prompt - 1,
                )?;
                command.commit();
                in_flight.push(command);
                if in_flight.len() >= 8 {
                    seconds += complete_committed(&in_flight.remove(0))?;
                }
                evaluated += rows;
                position += rows;
            }
            for command in &in_flight {
                seconds += complete_committed(command)?;
            }
            return Ok((
                seconds,
                evaluated,
                route_sample(self.buffers.fine_grained.route_stats()),
            ));
        }
        let command = runtime.queue.new_command_buffer();
        let active_rows = if pass == 0 {
            scorer::encode_proposals(
                command,
                &self.pipelines,
                &self.tensorops,
                &self.pisa,
                &self.buffers,
                weights,
                scorer::ProposalReadout {
                    output: &self.proposals,
                    seeds: &self.seeds,
                    temperature,
                    score_targets,
                    feedback,
                },
                total_rows,
                Some(decoder.cache()),
                None,
                None,
            )?;
            total_rows
        } else {
            let row_start = (cursor as u32 / 128) * 128;
            let rows = total_rows - row_start;
            scorer::suffix_proposals(
                command,
                &self.pipelines,
                &self.tensorops,
                &self.pisa,
                &self.buffers,
                weights,
                scorer::ProposalReadout {
                    output: &self.proposals,
                    seeds: &self.seeds,
                    temperature,
                    score_targets,
                    feedback,
                },
                row_start,
                rows,
                decoder.cache(),
            )?;
            rows
        };
        let seconds = complete(command)?;
        let route = route_sample(self.buffers.fine_grained.route_stats());
        Ok((seconds, active_rows, route))
    }

    #[allow(clippy::too_many_arguments)]
    fn commit_prefix(
        &self,
        task: &GenerationTask,
        tokens: &mut [u32],
        input: &mut [u32],
        cursor: &mut usize,
        maximum: usize,
        eos: Option<u32>,
    ) -> Result<CommitStats, String> {
        if *cursor == maximum {
            return Ok(CommitStats {
                accepted: 0,
                committed: 0,
                mismatch: None,
            });
        }
        let proposed = unsafe {
            std::slice::from_raw_parts(
                self.proposals.contents().cast::<u32>(),
                self.context as usize,
            )
        };
        let prompt = task.prompt.len();
        let exact_start = *cursor;
        let mut position = *cursor;
        let mut committed = None;
        let mut accepted = 0usize;
        while position < maximum {
            let prediction = proposed[prompt - 1 + position];
            if prediction >= VOCAB {
                return Err("block verifier produced an invalid token".into());
            }
            let exact_prediction = committed.is_none();
            if exact_prediction {
                let matched = tokens[position] == prediction;
                if matched {
                    accepted += 1;
                } else {
                    // This token is exact because every preceding token was
                    // already committed. Later predictions are only the next
                    // parallel fixed-point draft.
                    committed = Some(position + 1);
                }
            }
            tokens[position] = prediction;
            if position + 1 < maximum {
                input[prompt + position] = prediction;
            }
            position += 1;
            if exact_prediction && Some(prediction) == eos {
                tokens[position..].fill(prediction);
                *cursor = maximum;
                return Ok(CommitStats {
                    accepted,
                    committed: position - exact_start,
                    mismatch: committed.map(|after| after - 1),
                });
            }
        }
        *cursor = committed.unwrap_or(maximum);
        Ok(CommitStats {
            accepted,
            committed: *cursor - exact_start,
            mismatch: committed.map(|after| after - 1),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn run_verification(
        &self,
        runtime: &Runtime,
        decoder: &decode::Decoder,
        weights: CandidateRow<'_>,
        task: &GenerationTask,
        config: &GenerationConfig,
        tokens: &mut [u32],
        input: &mut [u32],
        total_rows: u32,
        score_targets: bool,
    ) -> Result<VerificationProgress, String> {
        let maximum = config.max_tokens as usize;
        let eos = config.eos_token;
        let mut progress = VerificationProgress {
            target: TargetStats::new(Self::loss_window(config)),
            ..Default::default()
        };
        while progress.cursor < maximum && progress.broad_passes < config.verify.passes as usize {
            let exact_start = progress.cursor;
            let (pass_seconds, active_rows, route) = self.parallel_pass(
                runtime,
                decoder,
                weights,
                config.temperature,
                config.feedback_transition,
                progress.broad_passes,
                progress.cursor,
                total_rows,
                score_targets,
                task.prompt.len(),
            )?;
            progress.gpu_seconds += pass_seconds;
            progress.broad_passes += 1;
            progress.parallel_positions += active_rows as usize;
            progress.evaluated_lengths.push(active_rows as usize);
            progress.route_samples.push(route);
            let stats =
                self.commit_prefix(task, tokens, input, &mut progress.cursor, maximum, eos)?;
            progress.accepted_lengths.push(stats.accepted);
            progress.committed_lengths.push(stats.committed);
            if progress.first_mismatch.is_none() {
                progress.first_mismatch = stats.mismatch;
            }
            if score_targets {
                self.target_nll(
                    task,
                    tokens,
                    exact_start,
                    progress.cursor,
                    &mut progress.target,
                )?;
            }
            if stats.committed == 0 {
                break;
            }
        }
        if progress.cursor < maximum {
            let stats = self.repair_mismatch(
                runtime,
                decoder,
                weights,
                task,
                config,
                tokens,
                &mut progress.cursor,
                score_targets,
                &mut progress.target,
            )?;
            progress.gpu_seconds += stats.gpu_seconds;
            progress.repair_positions += stats.evaluated_positions;
            progress.correction_waves += stats.correction_waves;
            progress.repair_batches += stats.repair_batches;
            if progress.first_mismatch.is_none() {
                progress.first_mismatch = stats.first_mismatch;
            }
            progress.evaluated_lengths.extend(stats.evaluated_lengths);
            progress.accepted_lengths.extend(stats.accepted_lengths);
            progress.committed_lengths.extend(stats.committed_lengths);
            progress.route_samples.extend(stats.route_samples);
        }
        Ok(progress)
    }

    pub fn verify(
        &self,
        runtime: &Runtime,
        decoder: &decode::Decoder,
        weights: CandidateRow<'_>,
        tasks: &[GenerationTask],
        config: &GenerationConfig,
        sampling_seed: u64,
        drafts: &[decode::Rollout],
    ) -> Result<Vec<decode::Rollout>, String> {
        if config.draft.is_some() {
            return diffusion::verify(
                self,
                runtime,
                decoder,
                weights,
                tasks,
                config,
                sampling_seed,
            );
        }
        if tasks.len() != 1 || drafts.len() != 1 {
            return Err("block verification requires exactly one trajectory and draft".into());
        }
        let maximum = config.max_tokens as usize;
        let total_rows =
            ((tasks[0].prompt.len() + maximum - 1).div_ceil(128) as u32 * 128).max(CONTEXT);
        if total_rows > self.context {
            return Err("prompt plus generation exceeds block cache capacity".into());
        }
        let score_targets = Self::score_targets(config);
        if score_targets {
            self.load_targets(&tasks[0], maximum)?;
        }
        let eos = config.eos_token;
        let mut tokens = expand_draft(&drafts[0], maximum, eos)?;
        let input = unsafe {
            std::slice::from_raw_parts_mut(
                self.buffers.tokens.contents().cast::<u32>(),
                self.context as usize,
            )
        };
        input.fill(0);
        let task = &tasks[0];
        input[..task.prompt.len()].copy_from_slice(&task.prompt);
        let input_tokens = maximum.saturating_sub(1);
        input[task.prompt.len()..task.prompt.len() + input_tokens]
            .copy_from_slice(&tokens[..input_tokens]);
        unsafe {
            self.seeds
                .contents()
                .cast::<u64>()
                .write(crate::hash::splitmix64(sampling_seed))
        };

        let started = Instant::now();
        let progress = self.run_verification(
            runtime,
            decoder,
            weights,
            task,
            config,
            &mut tokens,
            input,
            total_rows,
            score_targets,
        )?;
        let wall_seconds = started.elapsed().as_secs_f64();
        if score_targets && progress.target.count != maximum {
            return Err(format!(
                "free-running target loss covered {} of {maximum} tokens",
                progress.target.count
            ));
        }
        let target_quality = score_targets
            .then(|| progress.target.finish())
            .transpose()?;
        let end = tokens
            .iter()
            .position(|token| Some(*token) == eos)
            .map_or(tokens.len(), |index| index + 1);
        Ok(vec![decode::Rollout {
            draft: None,
            tokens: tokens[..end].to_vec(),
            finish_reason: if end < maximum { "eos" } else { "length" },
            wall_seconds,
            gpu_seconds: progress.gpu_seconds,
            evaluated_positions: progress.parallel_positions + progress.repair_positions,
            committed_tokens: end,
            broad_passes: progress.broad_passes,
            correction_waves: progress.correction_waves,
            repair_batches: progress.repair_batches,
            accepted_tokens: progress.accepted_lengths.iter().sum(),
            first_mismatch: progress.first_mismatch,
            evaluated_lengths: progress.evaluated_lengths,
            accepted_lengths: progress.accepted_lengths,
            committed_lengths: progress.committed_lengths,
            route_samples: progress.route_samples,
            free_running_target_nll: target_quality.map(|quality| quality.mean_nll),
            target_quality,
        }])
    }

    pub fn generate(
        &self,
        runtime: &Runtime,
        decoder: &decode::Decoder,
        weights: CandidateRow<'_>,
        tasks: &[GenerationTask],
        config: &GenerationConfig,
        sampling_seed: u64,
    ) -> Result<Vec<decode::Rollout>, String> {
        if config.draft.is_some() {
            return diffusion::verify(
                self,
                runtime,
                decoder,
                weights,
                tasks,
                config,
                sampling_seed,
            );
        }
        let draft = decode::Rollout {
            draft: None,
            tokens: vec![0; config.max_tokens as usize],
            finish_reason: "length",
            wall_seconds: 0.0,
            gpu_seconds: 0.0,
            evaluated_positions: 0,
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
        };
        self.verify(
            runtime,
            decoder,
            weights,
            tasks,
            config,
            sampling_seed,
            &[draft],
        )
    }
}

fn expand_draft(
    draft: &decode::Rollout,
    maximum: usize,
    eos: Option<u32>,
) -> Result<Vec<u32>, String> {
    if draft.tokens.len() == maximum {
        return Ok(draft.tokens.clone());
    }
    let Some(eos) = eos else {
        return Err("short incumbent rollout without EOS cannot seed block verification".into());
    };
    if draft.tokens.last() != Some(&eos) || draft.tokens.len() > maximum {
        return Err("incumbent rollout has an invalid generated length".into());
    }
    let mut tokens = draft.tokens.clone();
    tokens.resize(maximum, eos);
    Ok(tokens)
}
