//! Configured end-to-end TuRBO-ENN round experiment.
use super::*;
use metal::objc::rc::autoreleasepool;

#[derive(Clone, Debug)]
pub struct RoundLatencyRecord {
    pub round_ms: f64,
    pub incumbent_nll: f64,
    pub candidate_nll: f64,
    pub incumbent_variance: f64,
    pub candidate_variance: f64,
    pub acceptance_threshold: f64,
    pub acceptance_margin: f64,
    pub accepted: bool,
    pub radius: f32,
    pub trust_length: f64,
    pub success_counter: i32,
    pub failure_counter: i32,
    pub restarts: usize,
    pub allocated_bytes: u64,
}

#[derive(Clone, Debug)]
pub struct RoundLatencyExperiment {
    pub initial_objective_ms: f64,
    pub mean_round_ms: f64,
    pub rounds: Vec<RoundLatencyRecord>,
}

impl RoundLatencyExperiment {
    pub fn meets_target(&self, target_round_ms: u32) -> bool {
        self.rounds
            .iter()
            .all(|round| round.round_ms <= f64::from(target_round_ms))
    }

    pub fn max_bytes(&self) -> u64 {
        self.rounds
            .iter()
            .map(|round| round.allocated_bytes)
            .max()
            .unwrap_or(0)
    }

    pub fn max_ms(&self) -> f64 {
        self.rounds
            .iter()
            .map(|round| round.round_ms)
            .fold(0.0, f64::max)
    }
}

pub(super) fn full_config() -> ModelConfig {
    ModelConfig {
        width: 1536,
        intermediate: 6656,
        layers: 24,
        vocab: 100352,
        heads: 16,
        kv_heads: 8,
        capacity: 4096,
        chunk: 256,
        local_window: 2048,
        full_every: 6,
        epsilon: 1e-5,
        rope_base: 10000.0,
        residual_scale: 1.0 / 48.0f32.sqrt(),
        feedback_token_norm: InputNorm::UnitRms { epsilon: 1e-5 },
        feedback_fused_norm: InputNorm::UnitRms { epsilon: 1e-5 },
        tiled_attention: true,
    }
}

fn synthetic_token(index: u32, step: u32, sample: u32, vocab: u32) -> u32 {
    ((u64::from(index) * 7919 + u64::from(step) * 137 + u64::from(sample) * 29) % u64::from(vocab))
        as u32
}

fn synthetic_examples(context: u32, step: u32, vocab: u32) -> Vec<(Vec<u32>, Vec<u32>)> {
    (0..2)
        .map(|sample| {
            let tokens = (0..context)
                .map(|i| synthetic_token(i, step, sample, vocab))
                .collect();
            let targets = (0..context)
                .map(|i| synthetic_token(i + 1, step, sample, vocab))
                .collect();
            (tokens, targets)
        })
        .collect()
}

pub(super) fn model_search(
    model: &Model,
    run: &crate::config::ConfigOverrides,
) -> crate::bf16_metal::SearchState {
    use crate::bf16_metal::{ParamBlock, SearchState};
    let mut base = Vec::with_capacity(model.parameter_count());
    let mut blocks = Vec::new();
    for (index, p) in model.parameters.iter().enumerate() {
        let values =
            unsafe { std::slice::from_raw_parts(p.buffer.contents().cast::<u16>(), p.elements) };
        let rms = (values
            .iter()
            .map(|v| f64::from(f32::from_bits(u32::from(*v) << 16)).powi(2))
            .sum::<f64>()
            / p.elements as f64)
            .sqrt() as f32;
        let scale = rms.max(1e-6);
        blocks.push(
            ParamBlock::new(
                index as u64,
                base.len(),
                p.elements,
                scale,
                1.0 / (p.elements as f32 * scale * scale),
            )
            .unwrap(),
        );
        base.extend_from_slice(values);
    }
    let length = run.length();
    let mut search = SearchState::new_unscored(&base, blocks, 2, 1, length).unwrap();
    search.configure_pool(run.resident_pool().unwrap()).unwrap();
    search.set_tolerance(4).unwrap();
    search.correlate(run.reference_seed()).unwrap();
    search
}
pub fn round_experiment(
    run: &crate::config::ConfigOverrides,
) -> Result<RoundLatencyExperiment, String> {
    run.validate_experiment()?;
    use std::time::Instant;
    let mean_round_ms = autoreleasepool(|| {
        let setup = Instant::now();
        let rounds = run.rounds();
        let context_tokens = 4096;
        let score_mode = ScoreMode::Fused;
        let c = full_config();
        let mut model = Model::new(c, run.model_seed()).unwrap();
        let optimized = true;
        model.set_optimized(optimized).unwrap();
        model.set_gateup(super::GateUpImplementation::Mps).unwrap();
        let trace = run.trace();
        if let Err(e) = model.prepare_prefill(2, context_tokens) {
            panic!("prepare_prefill error: {}", e);
        }
        let mut search = model_search(&model, run);
        search.set_profiling(trace);
        search.prepare_reference().unwrap();
        let length = run.length();
        eprintln!(
            "TURBO_ENN setup_seconds={:.6} parameters={} context={context_tokens} batch=2 pool=4 history=2 mode={score_mode:?} random_init=true synthetic=true radius={} optimized={optimized} prefill=true target_round_ms={}",
            setup.elapsed().as_secs_f64(),
            model.parameter_count(),
            length.length_init,
            run.target_ms()
        );
        eprintln!(
            "FBT_MODEL experiment=round-latency config={c:?} device={}",
            model.runtime.device.name()
        );
        let controller = search.controller_info().unwrap();
        eprintln!(
            "TURBO_ENN_CONTROLLER success_tolerance={} failure_tolerance={}",
            controller.success_tolerance, controller.failure_tolerance
        );
        let initial_objective_ms = initialize_objective(&mut model, &mut search, trace, score_mode);
        let all = Instant::now();
        let mut round_seconds = Vec::with_capacity(rounds as usize);
        let mut records = Vec::with_capacity(rounds as usize);
        for step in 0..rounds {
            let (record, seconds) = execute_round(&mut model, &mut search, run, step);
            round_seconds.push(seconds);
            records.push(record);
        }
        let loop_seconds = all.elapsed().as_secs_f64();
        let mean_round_ms = round_seconds.iter().sum::<f64>() * 1000.0 / f64::from(rounds);
        let experiment = RoundLatencyExperiment {
            initial_objective_ms,
            mean_round_ms,
            rounds: records,
        };
        eprintln!(
            "TURBO_ENN complete rounds={rounds} loop_seconds={loop_seconds:.6} mean_round_ms={mean_round_ms:.3} target_round_ms={} goal_met={}",
            run.target_ms(),
            experiment.meets_target(run.target_ms())
        );
        experiment
    });
    Ok(mean_round_ms)
}

struct RoundIncumbent {
    incumbent_reward: f32,
    incumbent_variance: f32,
}

fn initialize_objective(
    model: &mut Model,
    search: &mut crate::bf16_metal::SearchState,
    trace: bool,
    mode: ScoreMode,
) -> f64 {
    let start = std::time::Instant::now();
    let examples = synthetic_examples(4096, 0, model.config.vocab);
    let losses = score_examples(model, &examples, trace, mode, "initial.incumbent");
    let nll = (losses[0] + losses[1]) / 2.0;
    let variance = (losses[0] - losses[1]).powi(2) / 4.0;
    search
        .observe_initial(-nll as f32, variance as f32)
        .unwrap();
    let milliseconds = start.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "TURBO_ENN_INITIAL objective_seconds={:.6} nll={nll:.9} variance={variance:.9} sequence_scores=2 transformer_passes=4",
        milliseconds / 1000.0
    );
    milliseconds
}

fn ask_round(
    search: &mut crate::bf16_metal::SearchState,
    run: &crate::config::ConfigOverrides,
    step: u32,
    trace: bool,
) -> (crate::bf16_metal::Proposals, f64) {
    let config = run
        .resident_ask(
            search.history_len().unwrap(),
            run.acquisition_seed() + step as u64,
        )
        .unwrap();
    let start = std::time::Instant::now();
    search
        .begin_ask(1, 4, run.proposal_seed() + step as u64, config)
        .unwrap();
    let round = search.finish_ask().unwrap();
    let seconds = start.elapsed().as_secs_f64();
    if trace {
        let profile = search.last_profile().unwrap();
        eprintln!(
            "FBT_CTRL round={} operation=ask pool_gpu_seconds={:.9} select_gpu_seconds={:.9} materialize_gpu_seconds={:.9} gpu_envelope_seconds={:.9}",
            step + 1,
            f64::from(profile.score_ms) / 1000.0,
            f64::from(profile.pick_ms) / 1000.0,
            f64::from(profile.materialize_ms) / 1000.0,
            f64::from(profile.total_ms) / 1000.0
        );
    }
    (round, seconds)
}
struct DecisionPhases {
    decision: crate::bf16_metal::NoisyDecision,
    restore_seconds: f64,
    tell_seconds: f64,
    sync_seconds: f64,
    rebind_seconds: f64,
    decision_restore_seconds: f64,
}

fn execute_round(
    model: &mut Model,
    search: &mut crate::bf16_metal::SearchState,
    run: &crate::config::ConfigOverrides,
    step: u32,
) -> (RoundLatencyRecord, f64) {
    use std::time::Instant;
    let trace = run.trace();
    let start = Instant::now();
    let examples = synthetic_examples(4096, step + 1, model.config.vocab);
    let state = RoundIncumbent {
        incumbent_reward: search.best().unwrap(),
        incumbent_variance: search.best_variance().unwrap(),
    };
    let (round, ask_seconds) = ask_round(search, run, step, trace);
    let apply_start = Instant::now();
    let proposal = search.propose_buffer(&round).unwrap();
    let original_buffers = model.bind_row(&proposal).unwrap();
    let apply_seconds = apply_start.elapsed().as_secs_f64();
    let score_start = Instant::now();
    let candidate = score_examples(
        model,
        &examples,
        trace,
        ScoreMode::Fused,
        &format!("round{}.candidate", step + 1),
    );
    let score_seconds = score_start.elapsed().as_secs_f64();
    let decision_start = Instant::now();
    assert!(candidate.iter().all(|v| v.is_finite()));
    let incumbent_nll = -f64::from(state.incumbent_reward);
    let candidate_nll = (candidate[0] + candidate[1]) / 2.0;
    let candidate_variance = (candidate[0] - candidate[1]).powi(2) / 4.0;
    let phases = decide_round(
        model,
        search,
        &round,
        original_buffers,
        decision_start,
        candidate_nll,
        candidate_variance,
        &state,
        trace,
        step,
    );
    let decision = &phases.decision;
    let accepted = decision.accepted;
    let controller = search.controller_info().unwrap();
    let compute_seconds = start.elapsed().as_secs_f64();
    let record = RoundLatencyRecord {
        round_ms: compute_seconds * 1000.0,
        incumbent_nll,
        candidate_nll,
        incumbent_variance: f64::from(state.incumbent_variance),
        candidate_variance,
        acceptance_threshold: decision.threshold,
        acceptance_margin: decision.improvement - decision.threshold,
        accepted,
        radius: round.length,
        trust_length: controller.length,
        success_counter: controller.success_counter,
        failure_counter: controller.failure_counter,
        restarts: controller.restarts,
        allocated_bytes: model.runtime.device.current_allocated_size() as u64,
    };
    report_round(
        &record,
        &phases,
        step,
        ask_seconds,
        apply_seconds,
        score_seconds,
        compute_seconds,
        start,
    );
    (record, compute_seconds)
}

fn decide_round(
    model: &mut Model,
    search: &mut crate::bf16_metal::SearchState,
    round: &crate::bf16_metal::Proposals,
    original_buffers: Vec<metal::Buffer>,
    decision_start: std::time::Instant,
    candidate_nll: f64,
    candidate_variance: f64,
    state: &RoundIncumbent,
    trace: bool,
    step: u32,
) -> DecisionPhases {
    use std::time::Instant;
    let tell_start = Instant::now();
    let decision = search
        .tell_noisy(&round, -candidate_nll as f32, candidate_variance as f32)
        .unwrap();
    let tell_seconds = tell_start.elapsed().as_secs_f64();
    assert_eq!(decision.incumbent_value, state.incumbent_reward);
    assert_eq!(decision.incumbent_variance, state.incumbent_variance);
    let accepted = decision.accepted;
    let restore_start = Instant::now();
    if accepted {
        drop(original_buffers);
    } else {
        model.restore_buffers(original_buffers).unwrap();
    }
    let restore_seconds = restore_start.elapsed().as_secs_f64();
    if trace {
        let profile = search.tell_profile().unwrap();
        eprintln!(
            "FBT_CTRL round={} operation=tell reference_gpu_seconds={:.9} history_copy_gpu_seconds={:.9} gpu_envelope_seconds={:.9}",
            step + 1,
            f64::from(profile.reference_ms) / 1000.0,
            f64::from(profile.history_copy_ms) / 1000.0,
            f64::from(profile.total_ms) / 1000.0,
        );
    }
    let sync_start = Instant::now();
    assert_eq!(search.sync().unwrap(), vec![accepted]);
    let sync_seconds = sync_start.elapsed().as_secs_f64();
    let rebind_start = Instant::now();
    if accepted {
        let base = search.base_buffer();
        let stale = model.bind_row(&base).unwrap();
        drop(stale);
    }
    let rebind_seconds = rebind_start.elapsed().as_secs_f64();
    let decision_restore_seconds = decision_start.elapsed().as_secs_f64();
    DecisionPhases {
        decision,
        restore_seconds,
        tell_seconds,
        sync_seconds,
        rebind_seconds,
        decision_restore_seconds,
    }
}

fn report_round(
    record: &RoundLatencyRecord,
    phases: &DecisionPhases,
    step: u32,
    ask_seconds: f64,
    apply_seconds: f64,
    score_seconds: f64,
    compute_seconds: f64,
    start: std::time::Instant,
) {
    let decision_restore_seconds = phases.decision_restore_seconds;
    let restore_seconds = phases.restore_seconds;
    let tell_seconds = phases.tell_seconds;
    let sync_seconds = phases.sync_seconds;
    let rebind_seconds = phases.rebind_seconds;
    let incumbent_nll = record.incumbent_nll;
    let candidate_nll = record.candidate_nll;
    let incumbent_variance = record.incumbent_variance;
    let candidate_variance = record.candidate_variance;
    let accepted = record.accepted;
    eprintln!(
        "TURBO_ENN round={} ask_seconds={ask_seconds:.6} apply_seconds={apply_seconds:.6} candidate_seconds={score_seconds:.6} decision_restore_seconds={decision_restore_seconds:.6} compute_seconds={compute_seconds:.6} incumbent_nll={incumbent_nll:.9} candidate_nll={candidate_nll:.9} incumbent_variance={incumbent_variance:.9} candidate_variance={candidate_variance:.9} acceptance_threshold={:.9} acceptance_margin={:.9} accepted={accepted} objective_calls=1 sequence_scores=2 transformer_passes=4 radius={} allocated_GiB={:.4}",
        step + 1,
        record.acceptance_threshold,
        record.acceptance_margin,
        record.radius,
        record.allocated_bytes as f64 / 1073741824.0
    );
    eprintln!(
        "TURBO_ENN_CONTROLLER round={} length={} successes={} failures={} restarts={}",
        step + 1,
        record.trust_length,
        record.success_counter,
        record.failure_counter,
        record.restarts,
    );
    eprintln!(
        "TURBO_ENN_PHASE round={} restore_seconds={restore_seconds:.9} tell_seconds={tell_seconds:.9} sync_seconds={sync_seconds:.9} accepted_rebind_seconds={rebind_seconds:.9}",
        step + 1,
    );
    eprintln!(
        "TURBO_ENN round={} wall_including_result_logging_seconds={:.6}",
        step + 1,
        start.elapsed().as_secs_f64()
    );
}

fn score_examples(
    model: &mut Model,
    examples: &[(Vec<u32>, Vec<u32>)],
    trace: bool,
    mode: ScoreMode,
    call: &str,
) -> Vec<f64> {
    let batch: Vec<_> = examples
        .iter()
        .map(|(x, y)| (x.as_slice(), y.as_slice()))
        .collect();
    if trace {
        let traced = model.score_traced(&batch, mode).unwrap();
        let reconciliation_error = (traced.gpu_sum_seconds + traced.gpu_gap_seconds
            - traced.gpu_overlap_seconds
            - traced.gpu_envelope_seconds)
            .abs();
        assert!(
            reconciliation_error <= 1e-6,
            "GPU trace does not reconcile: {reconciliation_error}"
        );
        assert!(traced.scorer_operations > 0);
        assert_eq!(
            traced.operations.len(),
            traced.preparation_operations + traced.scorer_operations
        );
        eprintln!(
            "FBT_TRACE call={call} operations={} preparation_operations={} scorer_operations={} wall_seconds={:.9} gpu_sum_seconds={:.9} gpu_envelope_seconds={:.9} gpu_gap_seconds={:.9} gpu_overlap_seconds={:.9} reconciliation_error_seconds={reconciliation_error:.12}",
            traced.operations.len(),
            traced.preparation_operations,
            traced.scorer_operations,
            traced.wall_seconds,
            traced.gpu_sum_seconds,
            traced.gpu_envelope_seconds,
            traced.gpu_gap_seconds,
            traced.gpu_overlap_seconds,
        );
        for operation in &traced.operations {
            eprintln!(
                "FBT_OP call={call} sequence={} domain={} pass={} layer={} operation={} shape={:?} cpu_start_seconds={:.9} cpu_seconds={:.9} gpu_start_seconds={:.9} gpu_end_seconds={:.9} gpu_seconds={:.9}",
                operation.sequence,
                operation.domain,
                operation.pass.unwrap_or(0),
                operation.layer.map_or(-1, i64::from),
                operation.operation,
                operation.shape,
                operation.cpu_start_seconds,
                operation.cpu_seconds,
                operation.gpu_start_seconds.unwrap_or(-1.0),
                operation.gpu_end_seconds.unwrap_or(-1.0),
                operation.gpu_seconds.unwrap_or(-1.0),
            );
        }
        let mut groups = std::collections::BTreeMap::new();
        for operation in &traced.operations {
            let entry = groups
                .entry((operation.domain, operation.operation))
                .or_insert((0usize, 0.0f64, 0.0f64));
            entry.0 += 1;
            entry.1 += operation.cpu_seconds;
            entry.2 += operation.gpu_seconds.unwrap_or(0.0);
        }
        for ((domain, operation), (count, cpu_seconds, gpu_seconds)) in groups {
            eprintln!(
                "FBT_OP_GROUP call={call} domain={domain} operation={operation} count={count} cpu_seconds={cpu_seconds:.9} gpu_seconds={gpu_seconds:.9}"
            );
        }
        return traced.score.mean_nll;
    }
    let score = model.score_batch(&batch, mode).unwrap();

    eprintln!(
        "FBT_PREFILL batch={} tokens={} elapsed={:.6} passes={:?} encode_submit={:.6} completion_wait={:.6}",
        batch.len(),
        score.tokens_per_example,
        score.elapsed_seconds,
        score.pass_seconds,
        score.encode_submit_seconds,
        score.completion_wait_seconds
    );
    score.mean_nll
}

#[cfg(test)]
mod tests {
    use super::{RoundLatencyExperiment, RoundLatencyRecord};

    fn experiment(old_nll: f64, allocated_bytes: u64) -> RoundLatencyExperiment {
        RoundLatencyExperiment {
            initial_objective_ms: 2.0,
            mean_round_ms: 1.0,
            rounds: vec![RoundLatencyRecord {
                round_ms: 1.0,
                incumbent_nll: old_nll,
                candidate_nll: 2.0,
                incumbent_variance: 0.0,
                candidate_variance: 0.0,
                acceptance_threshold: 0.0,
                acceptance_margin: -1.0,
                accepted: false,
                radius: 0.01,
                trust_length: 0.01,
                success_counter: 0,
                failure_counter: 1,
                restarts: 0,
                allocated_bytes,
            }],
        }
    }

    #[test]
    fn experiment_limits() {
        let control = experiment(1.000_000_000_1, 20);
        assert_eq!(control.max_bytes(), 20);
        assert!(control.meets_target(1));
        let mut slow_round = control.clone();
        slow_round.rounds[0].round_ms = 1.001;
        assert!(!slow_round.meets_target(1));
    }
}
