use super::*;

pub(super) struct RoundLog {
    pub wall: Vec<f64>,
    pub gpu: Vec<f64>,
    pub accepted: u32,
    pub controller_seconds: Vec<f64>,
    pub controller_records: Vec<ennx_wire::json::Value>,
    last_family_metric: Option<[f32; 4]>,
}
impl RoundLog {
    pub(super) fn new(rounds: u32) -> Self {
        Self {
            wall: Vec::with_capacity(rounds as usize),
            gpu: Vec::with_capacity(rounds as usize),
            accepted: 0,
            controller_seconds: Vec::with_capacity(rounds as usize),
            controller_records: Vec::with_capacity(rounds as usize),
            last_family_metric: None,
        }
    }
}
type PairedScorer = Option<(metal::CommandQueue, Buffers, Pisa1)>;
pub(super) struct RoundContext<'a, 'b> {
    pub weights: &'a CandidateWeights,
    pub paired_scorer: &'a PairedScorer,
    pub frozen_anchors: &'a Option<Vec<ObjectiveStats>>,
    pub dataset: Option<&'a crate::pretrain_data::PretrainDataset>,
    pub control: BoControl<'a>,
    pub heldout: &'a heldout::Scorer<'b>,
    pub learning_start: Instant,
    pub initial_calls: u32,
    pub rounds: u32,
}
pub(super) fn execute_round(
    context: RoundContext<'_, '_>,
    search: &mut SearchState,
    updates: &mut UpdateLog,
    validation: &mut Vec<ennx_wire::json::Value>,
    step: u32,
    log: &mut RoundLog,
) -> Result<(), String> {
    let start = Instant::now();
    if context.control.kernel_trial.is_some() {
        crate::apple_gpu::require_power()?;
    }
    if let Some(selected_dataset) = context.dataset {
        pretrain_batch(
            context.heldout.buffers,
            selected_dataset,
            step % selected_dataset.batches(),
        )?;
        if let Some((_, paired_buffers, _)) = &context.paired_scorer {
            pretrain_batch(
                paired_buffers,
                selected_dataset,
                step % selected_dataset.batches(),
            )?;
        }
    }
    let PreparedRound {
        command,
        incumbent_command,
        proposal,
        ask_seconds,
        initializing,
    } = prepare_round(
        context.heldout.runtime,
        context.heldout.pipelines,
        context.heldout.tensorops,
        context.heldout.pisa1,
        context.heldout.buffers,
        context.weights,
        search,
        context.paired_scorer,
        context.control,
        step,
        log,
    )?;
    let changes = search.describe(&proposal)?.remove(0).3;
    let changed = changes.iter().map(|block| block.0).sum::<u64>();
    let measurement = measure_round(
        &command,
        incumbent_command.as_deref(),
        context.heldout.buffers,
        search,
        context.paired_scorer,
        context.frozen_anchors,
        context.dataset,
        context.control,
        step,
    )?;
    let tell_start = Instant::now();
    let decision = if initializing {
        search.tell_initial(
            &proposal,
            measurement.optimizer_value,
            measurement.optimizer_variance,
        )?
    } else if context.control.paired_objective {
        search.paired_modeled(
            &proposal,
            measurement.optimizer_value,
            measurement.optimizer_variance,
            measurement.incumbent_value,
            measurement.incumbent_variance,
            measurement.improvement,
            measurement.improvement_variance,
        )?
    } else {
        search.tell_modeled(
            &proposal,
            measurement.optimizer_value,
            measurement.optimizer_variance,
        )?
    };
    let synced = search.sync()?;
    if synced != vec![decision.accepted] {
        return Err("actual BO synchronization changed".into());
    }
    let tell_seconds = tell_start.elapsed().as_secs_f64();
    log.controller_seconds.push(ask_seconds + tell_seconds);
    log.accepted += u32::from(decision.accepted);
    updates.push_scaled(
        step + 1,
        proposal.seed,
        proposal.length,
        decision.accepted,
        changes,
        proposal.block_scales.clone(),
    )?;
    let wall_seconds = start.elapsed().as_secs_f64();
    log.gpu.push(measurement.gpu_seconds);
    log.wall.push(wall_seconds);
    let fitted = search.fitted_enn();
    let reliability = search.reliability_info()?;
    log.controller_records.push(controller_record(
        step + 1,
        initializing,
        &proposal,
        decision,
        search.controller_info()?,
        reliability,
        wall_seconds,
        ask_seconds + tell_seconds,
    ));
    let record = log
        .controller_records
        .last_mut()
        .ok_or("missing training record")?;
    record["training"] = ennx_wire::json::json!({
        "candidate_nll":-measurement.value, "sequence_nlls":measurement.scores,
        "candidate_evaluations":step + 1,
        "objective_calls":context.initial_calls + (step + 1) * if context.control.paired_objective { 2 } else { 1 },
        "selection":if context.control.random_selection { "random" } else { "enn" },
    });
    capture_round(
        context.control,
        context.heldout.buffers,
        log,
        measurement.scores,
        measurement.gpu_seconds,
    )?;
    eprintln!(
        "TURBO_ENN_ACTUAL_ROUND round={} total={rounds} phase={} logical_history={} resident_history={HISTORY_CAPACITY} wall_seconds={wall_seconds:.6} scorer_gpu_seconds={gpu_seconds:.6} ask_seconds={ask_seconds:.6} tell_seconds={tell_seconds:.6} changed_weights={changed} parameters={parameters} proposal_radius={:.9} reward={value:.9} variance={variance:.9} sequence_nlls={scores:?} incumbent_sequence_nlls={incumbent_scores:?} optimizer_value={optimizer_value:.9} paired_improvement={:.9} paired_variance={improvement_variance:.9} accepted={} radius={:.6} fitted_enn={fitted:?}",
        step + 1,
        if initializing {
            "initialization"
        } else {
            "turbo_enn"
        },
        search.history_len()?,
        proposal.length,
        decision.improvement,
        decision.accepted,
        search.length()?,
        rounds = context.rounds,
        gpu_seconds = measurement.gpu_seconds,
        value = measurement.value,
        variance = measurement.variance,
        scores = measurement.scores,
        optimizer_value = measurement.optimizer_value,
        improvement_variance = measurement.improvement_variance,
        incumbent_scores = measurement.incumbent_scores,
        parameters = context.weights.architecture.parameter_count(),
    );
    if !initializing {
        eprintln!(
            "TURBO_ENN_TRUST round={} candidate_mean={:.9} incumbent_mean={:.9} predicted_improvement={:.9} observed_improvement={:.9} threshold={:.9} agreement_ratio={} outcome={:?}",
            step + 1,
            proposal.predicted_mean,
            proposal.incumbent_mean,
            decision.predicted_improvement.unwrap_or(f64::NAN),
            decision.improvement,
            decision.threshold,
            decision
                .agreement_ratio
                .map_or_else(|| "none".to_string(), |value| format!("{value:.6}")),
            decision.trust_outcome,
        );
        if let Some(state) = reliability {
            log_reliability(step + 1, state);
        }
    }
    if context.control.kernel_trial.is_some() {
        let record = log
            .controller_records
            .last_mut()
            .ok_or("missing kernel round record")?;
        record["kernel_check"]["complete_round_seconds"] =
            ennx_wire::json::json!(start.elapsed().as_secs_f64());
    }
    if let Some(validation_dataset) = context.control.validation {
        if (step + 1) % context.control.validation_interval == 0 || step + 1 == context.rounds {
            let base = search.base_buffer();
            let calls = context.initial_calls
                + (step + 1)
                    * if context.control.paired_objective {
                        2
                    } else {
                        1
                    };
            validation.push(context.heldout.observe(
                validation_dataset,
                context.weights.row(&base)?,
                step + 1,
                calls,
                context.learning_start,
            )?);
        }
    }
    Ok(())
}

struct PreparedRound {
    command: metal::CommandBuffer,
    incumbent_command: Option<metal::CommandBuffer>,
    proposal: Proposals,
    ask_seconds: f64,
    initializing: bool,
}
fn prepare_round(
    runtime: &Runtime,
    pipelines: &Pipelines,
    tensorops: &TensorOpsPipelines,
    pisa1: &Pisa1,
    buffers: &Buffers,
    weights: &CandidateWeights,
    search: &mut SearchState,
    paired_scorer: &PairedScorer,
    control: BoControl<'_>,
    step: u32,
    log: &mut RoundLog,
) -> Result<PreparedRound, String> {
    let ask_start = Instant::now();
    search.compact_history()?;
    let history = search.history_len()?;
    let initializing = history < control.enn.ask.neighbors;
    if !initializing {
        if let Some((metric, scales)) = search.family_shape() {
            if log.last_family_metric != Some(metric) {
                eprintln!(
                    "TURBO_ENN_FAMILY round={} groups=experts,projections,routers_feedback,norms metric={metric:?} proposal={scales:?}",
                    step + 1
                );
                log.last_family_metric = Some(metric);
            }
        }
    }
    let proposal_seed = control
        .proposal_seed
        .checked_add(u64::from(step))
        .ok_or("proposal seed overflow")?;
    let row = if initializing {
        search.begin_initial(proposal_seed, step as usize % 4)?
    } else {
        let mut ask = control.enn.ask;
        ask.neighbors = ask.neighbors.min(history);
        ask.seed = control
            .acquisition_seed
            .checked_add(u64::from(step))
            .ok_or("acquisition seed overflow")?;
        if control.random_selection {
            let index = crate::hash::splitmix64(ask.seed) as usize % 4;
            search.begin_forced(proposal_seed, ask, index)?
        } else {
            search.begin_ask(1, 4, proposal_seed, ask)?
        }
    };
    let candidate_weights = weights.row(&row)?;
    let command = runtime.queue.new_command_buffer().to_owned();
    objective_fused(
        &command,
        pipelines,
        tensorops,
        pisa1,
        buffers,
        candidate_weights,
    )?;
    command.commit();
    let incumbent_command = if let Some((queue, paired_buffers, paired_pisa)) = &paired_scorer {
        let base = search.base_buffer();
        let incumbent_weights = weights.row(&base)?;
        let incumbent_command = queue.new_command_buffer().to_owned();
        objective_fused(
            &incumbent_command,
            pipelines,
            tensorops,
            paired_pisa,
            paired_buffers,
            incumbent_weights,
        )?;
        incumbent_command.commit();
        Some(incumbent_command)
    } else {
        None
    };
    let proposal = search.finish_ask()?;
    let ask_seconds = ask_start.elapsed().as_secs_f64();
    if let Some(profile) = search.last_profile() {
        eprintln!(
            "TURBO_ENN_CTRL round={} pool_gpu_ms={:.6} select_gpu_ms={:.6} materialize_gpu_ms={:.6} envelope_gpu_ms={:.6}",
            step + 1,
            profile.score_ms,
            profile.pick_ms,
            profile.materialize_ms,
            profile.total_ms,
        );
    }
    Ok(PreparedRound {
        command,
        incumbent_command,
        proposal,
        ask_seconds,
        initializing,
    })
}

struct Measurement {
    gpu_seconds: f64,
    value: f32,
    variance: f32,
    scores: [f32; 2],
    incumbent_value: f32,
    incumbent_variance: f32,
    optimizer_value: f32,
    optimizer_variance: f32,
    improvement: f32,
    improvement_variance: f32,
    incumbent_scores: Option<[f32; 2]>,
}

fn measure_round(
    command: &CommandBufferRef,
    incumbent_command: Option<&CommandBufferRef>,
    buffers: &Buffers,
    search: &SearchState,
    paired_scorer: &PairedScorer,
    frozen_anchors: &Option<Vec<ObjectiveStats>>,
    dataset: Option<&crate::pretrain_data::PretrainDataset>,
    control: BoControl<'_>,
    step: u32,
) -> Result<Measurement, String> {
    let mut gpu_seconds = complete_committed(command)?;
    let candidate_interval = gpu_interval(command);
    let candidate_objective = sequence_objective(buffers)?;
    let value = candidate_objective.reward;
    let variance = candidate_objective.variance;
    let scores = candidate_objective.sequence_nlls;
    let incumbent_value = search.best()?;
    let incumbent_variance = search.best_variance()?;
    let (optimizer_value, optimizer_variance, improvement, improvement_variance, incumbent_scores) =
        if control.paired_objective {
            let incumbent_command =
                incumbent_command.ok_or("paired objective did not encode its incumbent")?;
            let incumbent_gpu_seconds = complete_committed(incumbent_command)?;
            gpu_seconds = match (candidate_interval, gpu_interval(incumbent_command)) {
                (Some(candidate), Some(incumbent)) => {
                    candidate.1.max(incumbent.1) - candidate.0.min(incumbent.0)
                }
                _ => gpu_seconds + incumbent_gpu_seconds,
            };
            let (_, paired_buffers, _) = paired_scorer
                .as_ref()
                .ok_or("paired objective scorer is unavailable")?;
            let incumbent_objective = sequence_objective(paired_buffers)?;
            let incumbent_scores = incumbent_objective.sequence_nlls;
            let (improvement, improvement_variance) =
                paired_improvement(&candidate_objective, &incumbent_objective);
            (
                incumbent_value + improvement,
                incumbent_variance + improvement_variance,
                improvement,
                improvement_variance,
                Some(incumbent_scores),
            )
        } else if let Some(anchors) = &frozen_anchors {
            let batch = dataset
                .map(|dataset| step % dataset.batches())
                .ok_or("frozen objective reference lost its dataset")?;
            let anchor = anchors
                .get(batch as usize)
                .ok_or("frozen objective reference lost its batch")?;
            let (improvement, improvement_variance) =
                paired_improvement(&candidate_objective, anchor);
            (
                improvement,
                improvement_variance,
                improvement,
                improvement_variance,
                Some(anchor.sequence_nlls),
            )
        } else {
            (value, variance, value - incumbent_value, variance, None)
        };
    Ok(Measurement {
        gpu_seconds,
        value,
        variance,
        scores,
        incumbent_value,
        incumbent_variance,
        optimizer_value,
        optimizer_variance,
        improvement,
        improvement_variance,
        incumbent_scores,
    })
}

fn capture_round(
    control: BoControl<'_>,
    buffers: &Buffers,
    log: &mut RoundLog,
    scores: [f32; 2],
    gpu_seconds: f64,
) -> Result<(), String> {
    if control.kernel_trial.is_some() {
        crate::apple_gpu::require_power()?;
        // Capture after the round timer. Complete-loop timing still includes
        // the capture, symmetrically for production and candidate trials.
        let losses = unsafe {
            std::slice::from_raw_parts(buffers.losses.contents().cast::<f32>(), ROWS as usize)
        };
        if losses.iter().any(|loss| !loss.is_finite()) {
            return Err("kernel trial produced a nonfinite token loss".into());
        }
        let record = log
            .controller_records
            .last_mut()
            .ok_or("missing kernel round record")?;
        record["kernel_check"] = ennx_wire::json::json!({
            "sequence_nlls": scores,
            "token_nlls": losses,
            "scorer_gpu_seconds": gpu_seconds,
        });
    }
    Ok(())
}
