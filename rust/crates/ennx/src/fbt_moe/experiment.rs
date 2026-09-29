use super::*;
use crate::fbt_pisa1::pisa_probe;

pub fn run_pretrain(
    run: &crate::config::ConfigOverrides,
    dataset_path: &std::path::Path,
) -> Result<ActualBoResult, String> {
    run.validate_experiment()?;
    let rounds = run.rounds();
    if rounds == 0 {
        return Err("pretraining requires positive rounds".into());
    }
    let objective_reference = run.objective_reference();
    let validation = run
        .validation_dataset
        .as_deref()
        .map(crate::pretrain_data::PretrainDataset::load)
        .transpose()?;
    let control = BoControl {
        length: run.length(),
        enn: run.resident_enn(run.acquisition_seed())?,
        proposal_seed: run.proposal_seed(),
        acquisition_seed: run.acquisition_seed(),
        perturbation: run.perturbation(),
        shape: run
            .trust_region_shape
            .unwrap_or(crate::config::TrustRegionShape::TensorFamilyStatic),
        paired_objective: objective_reference == crate::config::ObjectiveReference::MovingIncumbent,
        objective_reference,
        reliability: run.reliability_controller(),
        kernel_trial: run.kernel_trial.as_ref(),
        validation: validation.as_ref(),
        validation_interval: run.validation_interval.unwrap_or(rounds),
        random_selection: run.selection == Some(crate::config::PretrainSelection::Random),
    };
    let started = Instant::now();
    eprintln!("[tune] load dataset and compile pipelines");
    let dataset = crate::pretrain_data::PretrainDataset::load(dataset_path)?;
    metal::objc::rc::autoreleasepool(|| {
        if run.kernel_trial.is_some() {
            crate::apple_gpu::require_power()?;
            eprintln!("ENNX_KERNEL_STAGE compile");
        }
        let runtime = Runtime::shared()?;
        let pipelines = Pipelines::for_trial(&runtime, run.kernel_trial.as_ref())?;
        let tensorops = TensorOpsPipelines::new(&runtime)?;
        let buffers = Buffers::new(&runtime);
        let pisa1 = Pisa1::for_trial(&runtime, run.kernel_trial.as_ref())?;
        if run.kernel_trial.is_some() {
            eprintln!("ENNX_KERNEL_STAGE execute");
        }
        eprintln!(
            "[tune] initialize model and controller; score initial model | elapsed {:.1}s",
            started.elapsed().as_secs_f64()
        );
        let architecture =
            ResidualArchitecture::from_model(run.model.ok_or("pretrain model is required")?);
        let weights = CandidateWeights::seeded_for(&runtime, Some(run.model_seed()), architecture);
        let result = benchmark_bo(
            &runtime,
            &pipelines,
            &tensorops,
            &pisa1,
            &buffers,
            &weights,
            rounds,
            Some(&dataset),
            false,
            run.trace(),
            run.scorer_stage_samples,
            control,
        )?;
        eprintln!(
            "[tune] BO rounds complete; writing results | elapsed {:.1}s",
            started.elapsed().as_secs_f64()
        );
        Ok(result)
    })
}

pub fn moe_probe(rounds: u32, target_ms: u32) -> Result<GroupedMoeProbe, String> {
    moe_dataset(rounds, target_ms, None)
}

fn controller_summary(mut seconds: Vec<f64>) -> [f64; 3] {
    seconds.sort_by(f64::total_cmp);
    [
        seconds[seconds.len() / 2],
        seconds[0],
        seconds[seconds.len() - 1],
    ]
}

pub fn moe_dataset(
    rounds: u32,
    target_ms: u32,
    dataset_path: Option<&std::path::Path>,
) -> Result<GroupedMoeProbe, String> {
    if rounds == 0 || target_ms == 0 {
        return Err("grouped MoE probe requires positive rounds and target_ms".into());
    }
    let started = Instant::now();
    let progress = |stage: &str| {
        eprintln!(
            "[tune] {stage} | elapsed {:.1}s",
            started.elapsed().as_secs_f64()
        );
    };
    progress("load dataset and compile pipelines");
    let dataset = dataset_path
        .map(crate::pretrain_data::PretrainDataset::load)
        .transpose()?;
    metal::objc::rc::autoreleasepool(|| {
        let runtime = Runtime::shared()?;
        let pipelines = Pipelines::new(&runtime)?;
        let tensorops = TensorOpsPipelines::new(&runtime)?;
        let buffers = Buffers::new(&runtime);
        progress("setup complete; materialization and parity checks");
        let materialization = materialization(&runtime, &pipelines, &tensorops, &buffers)?;
        let mps_reference_gpu_seconds = mps_reference(&runtime, &buffers)?;
        let (projections_gpu_seconds, projections_wall_seconds) =
            benchmark_projections(&runtime, &tensorops, &buffers, rounds)?;
        let mps_projection_reference_gpu_seconds = mps_projections(&runtime, &buffers)?;
        progress("projection checks complete; PISA parity and timing");
        let pisa1 = pisa_probe(&runtime, &buffers.qkv, rounds)?;
        progress("PISA checks complete; FFN component timing");
        let sustained_pisa1 = Pisa1::new(&runtime)?;
        let parity = parity(
            &buffers,
            mps_reference_gpu_seconds,
            mps_projection_reference_gpu_seconds,
        )?;
        let components = components(&runtime, &pipelines, &tensorops, &buffers, rounds)?;
        let estimate = estimate(
            components.prematerialized_layer_wall_seconds,
            materialization.materialize_gate_up_gpu_seconds,
            materialization.materialize_down_gpu_seconds,
            projections_wall_seconds,
            &pisa1,
        )?;
        progress("component timing complete; sustained model diagnostic");
        let sustained = sustained(
            &runtime,
            &pipelines,
            &tensorops,
            &sustained_pisa1,
            &buffers,
            rounds,
            &progress,
        )?;
        progress("diagnostics complete; allocate candidate weights and check initial objective");
        let candidate_weights = CandidateWeights::new(&runtime);
        let actual_bo = benchmark_bo(
            &runtime,
            &pipelines,
            &tensorops,
            &sustained_pisa1,
            &buffers,
            &candidate_weights,
            rounds,
            dataset.as_ref(),
            true,
            false,
            None,
            BoControl::diagnostic(crate::Perturbation::Gaussian)?,
        )?;
        let controller_summary = controller_summary(actual_bo.controller_seconds);
        progress("BO rounds complete; writing results");
        let target_seconds = f64::from(target_ms) / 1000.0;
        let result = GroupedMoeProbe {
            updates: actual_bo.updates,
            parameters: FULL_PARAMETERS,
            routing_gpu_seconds: components.stage_gpu[0] + components.stage_gpu[1],
            activation_gpu_seconds: components.stage_gpu[2],
            residual_gpu_seconds: components.stage_gpu[3],
            materialize_gate_up_gpu_seconds: materialization.materialize_gate_up_gpu_seconds,
            materialize_down_gpu_seconds: materialization.materialize_down_gpu_seconds,
            projections_gpu_seconds,
            projections_wall_seconds,
            pisa1_pyramid_gpu_seconds: pisa1.pyramid_gpu_seconds,
            pisa1_selection_gpu_seconds: pisa1.selection_gpu_seconds,
            pisa1_attention_gpu_seconds: pisa1.attention_gpu_seconds,
            pisa1_layer_gpu_seconds: pisa1.layer_gpu_seconds,
            pisa1_layer_wall_seconds: pisa1.layer_wall_seconds,
            layer_gpu_seconds: components.prematerialized_layer_gpu_seconds,
            layer_wall_seconds: components.prematerialized_layer_wall_seconds,
            projected_ffn_seconds: estimate.projected_ffn_seconds,
            projected_ffn_and_projections_seconds: estimate.projected_ffn_and_projections_seconds,
            projected_measured_model_seconds: estimate.projected_measured_model_seconds,
            sustained_model_gpu_seconds: sustained.sustained_model_gpu_seconds,
            sustained_model_wall_seconds: sustained.sustained_model_wall_seconds,
            sustained_model_min_wall_seconds: sustained.sustained_model_min_wall_seconds,
            sustained_model_max_wall_seconds: sustained.sustained_model_max_wall_seconds,
            sustained_materialization_gpu_seconds: sustained.sustained_materialization_gpu_seconds,
            sustained_projections_gpu_seconds: sustained.sustained_projections_gpu_seconds,
            sustained_pisa1_gpu_seconds: sustained.sustained_pisa1_gpu_seconds,
            sustained_ffn_gpu_seconds: sustained.sustained_ffn_gpu_seconds,
            sustained_mps_projections_gpu_seconds: sustained.sustained_mps_projections_gpu_seconds,
            sustained_mps_projections_wall_seconds: sustained
                .sustained_mps_projections_wall_seconds,
            sustained_mps_ffn_gpu_seconds: sustained.sustained_mps_ffn_gpu_seconds,
            sustained_mps_ffn_wall_seconds: sustained.sustained_mps_ffn_wall_seconds,
            tail_gpu_seconds: sustained.tail_gpu_seconds,
            tail_wall_seconds: sustained.tail_wall_seconds,
            complete_envelope_gpu_seconds: sustained.complete_envelope_gpu_seconds,
            complete_envelope_wall_seconds: sustained.complete_envelope_wall_seconds,
            complete_envelope_min_wall_seconds: sustained.complete_envelope_min_wall_seconds,
            complete_envelope_max_wall_seconds: sustained.complete_envelope_max_wall_seconds,
            controller_median_wall_seconds: controller_summary[0],
            controller_min_wall_seconds: controller_summary[1],
            controller_max_wall_seconds: controller_summary[2],
            actual_bo_median_wall_seconds: actual_bo.median_wall_seconds,
            actual_bo_min_wall_seconds: actual_bo.min_wall_seconds,
            actual_bo_max_wall_seconds: actual_bo.max_wall_seconds,
            actual_bo_median_gpu_seconds: actual_bo.median_gpu_seconds,
            actual_bo_accepted: actual_bo.accepted,
            target_seconds,
            objective_flops: estimate.objective_flops,
            tail_flops: estimate.tail_flops,
            complete_objective_flops: estimate.complete_objective_flops,
            projection_objective_flops: estimate.projection_objective_flops,
            pisa1_objective_flops: estimate.pisa1_objective_flops,
            effective_tflops: estimate.objective_flops as f64
                / estimate.projected_measured_model_seconds
                / 1e12,
            gate_up_max_abs_error: parity.prematerialized_gate_up_max_abs_error,
            down_max_abs_error: parity.prematerialized_down_max_abs_error,
            qkv_max_abs_error: parity.qkv_max_abs_error,
            output_projection_max_abs_error: parity.output_projection_max_abs_error,
            pisa1_max_abs_error: pisa1.max_abs_error,
            tail_max_abs_error: sustained.tail_max_abs_error,
            meets_target: actual_bo.max_wall_seconds <= target_seconds,
        };
        report_probe(&result, rounds);
        Ok(result)
    })
}

fn report_probe(result: &GroupedMoeProbe, rounds: u32) {
    eprintln!(
        "TURBO_ENN_MODEL_FLOOR rows={ROWS} width={WIDTH} query_heads={QUERY_HEADS} kv_heads={KV_HEADS} head_dim={HEAD_DIM} qkv_width={QKV_WIDTH} experts={EXPERTS} rows_per_expert={EXPERT_ROWS} expert_width={EXPERT_WIDTH} routing=balanced_hash_top1 proposal=prematerialized_exact_full_rank_kronecker kernel=metal4_tensorops_m128_n64 data=deterministic_noncompressible_fp16 objective_flops={} projection_objective_flops={} pisa1_objective_flops={} routing_gpu_seconds={:.6} activation_gpu_seconds={:.6} residual_gpu_seconds={:.6} materialize_gate_up_gpu_seconds={:.6} materialize_down_gpu_seconds={:.6} projections_gpu_seconds={:.6} projections_wall_seconds={:.6} pisa1_pyramid_gpu_seconds={:.6} pisa1_selection_gpu_seconds={:.6} pisa1_attention_gpu_seconds={:.6} pisa1_layer_gpu_seconds={:.6} pisa1_layer_wall_seconds={:.6} layer_gpu_seconds={:.6} layer_wall_seconds={:.6} projected_ffn_seconds={:.6} projected_ffn_and_projections_seconds={:.6} projected_measured_model_seconds={:.6} effective_tflops={:.3} gate_up_max_abs_error={:.9} down_max_abs_error={:.9} qkv_max_abs_error={:.9} output_projection_max_abs_error={:.9} pisa1_max_abs_error={:.9} target_seconds={:.6} component_target_met={}",
        result.objective_flops,
        result.projection_objective_flops,
        result.pisa1_objective_flops,
        result.routing_gpu_seconds,
        result.activation_gpu_seconds,
        result.residual_gpu_seconds,
        result.materialize_gate_up_gpu_seconds,
        result.materialize_down_gpu_seconds,
        result.projections_gpu_seconds,
        result.projections_wall_seconds,
        result.pisa1_pyramid_gpu_seconds,
        result.pisa1_selection_gpu_seconds,
        result.pisa1_attention_gpu_seconds,
        result.pisa1_layer_gpu_seconds,
        result.pisa1_layer_wall_seconds,
        result.layer_gpu_seconds,
        result.layer_wall_seconds,
        result.projected_ffn_seconds,
        result.projected_ffn_and_projections_seconds,
        result.projected_measured_model_seconds,
        result.effective_tflops,
        result.gate_up_max_abs_error,
        result.down_max_abs_error,
        result.qkv_max_abs_error,
        result.output_projection_max_abs_error,
        result.pisa1_max_abs_error,
        result.target_seconds,
        result.meets_target,
    );
    eprintln!(
        "TURBO_ENN_MODEL_SUSTAINED scope=single_command_24_layers_2_passes_same_layer_weights samples={} gpu_median_seconds={:.6} wall_median_seconds={:.6} wall_min_seconds={:.6} wall_max_seconds={:.6} objective_flops={} effective_tflops={:.3}",
        rounds,
        result.sustained_model_gpu_seconds,
        result.sustained_model_wall_seconds,
        result.sustained_model_min_wall_seconds,
        result.sustained_model_max_wall_seconds,
        result.objective_flops,
        result.objective_flops as f64 / result.sustained_model_wall_seconds / 1e12,
    );
    eprintln!(
        "TURBO_ENN_MODEL_SUSTAINED_BREAKDOWN materialization_gpu_seconds={:.6} projections_gpu_seconds={:.6} pisa1_gpu_seconds={:.6} ffn_gpu_seconds={:.6} sum_gpu_seconds={:.6}",
        result.sustained_materialization_gpu_seconds,
        result.sustained_projections_gpu_seconds,
        result.sustained_pisa1_gpu_seconds,
        result.sustained_ffn_gpu_seconds,
        result.sustained_materialization_gpu_seconds
            + result.sustained_projections_gpu_seconds
            + result.sustained_pisa1_gpu_seconds
            + result.sustained_ffn_gpu_seconds,
    );
    eprintln!(
        "TURBO_ENN_MODEL_SUSTAINED_MPS projections_gpu_seconds={:.6} projections_wall_seconds={:.6} ffn_gpu_seconds={:.6} ffn_wall_seconds={:.6}",
        result.sustained_mps_projections_gpu_seconds,
        result.sustained_mps_projections_wall_seconds,
        result.sustained_mps_ffn_gpu_seconds,
        result.sustained_mps_ffn_wall_seconds,
    );
    eprintln!(
        "TURBO_ENN_MODEL_TAIL feedback_readout_loss_gpu_seconds={:.6} feedback_readout_loss_wall_seconds={:.6} tail_flops={}",
        result.tail_gpu_seconds, result.tail_wall_seconds, result.tail_flops,
    );
    eprintln!(
        "TURBO_ENN_COMPLETE_ENVELOPE scope=single_command_24_layers_2_passes_feedback_readout_loss samples={} gpu_median_seconds={:.6} wall_median_seconds={:.6} wall_min_seconds={:.6} wall_max_seconds={:.6} complete_objective_flops={} effective_tflops={:.3}",
        rounds,
        result.complete_envelope_gpu_seconds,
        result.complete_envelope_wall_seconds,
        result.complete_envelope_min_wall_seconds,
        result.complete_envelope_max_wall_seconds,
        result.complete_objective_flops,
        result.complete_objective_flops as f64 / result.complete_envelope_wall_seconds / 1e12,
    );
    eprintln!(
        "TURBO_ENN_FULL_CONTROLLER dimensions={} history_capacity={HISTORY_CAPACITY} samples={} policy=actual_objective wall_median_seconds={:.6} wall_min_seconds={:.6} wall_max_seconds={:.6}",
        FULL_PARAMETERS,
        rounds,
        result.controller_median_wall_seconds,
        result.controller_min_wall_seconds,
        result.controller_max_wall_seconds,
    );
    eprintln!(
        "TURBO_ENN_ACTUAL_BO scope=one_observation_noisy_distinct_24_layers_two_feedback_passes_pisa1_shared1_routed128_top3_moe rounds={} parameters={} history_capacity={HISTORY_CAPACITY} noise=independent_gaussian_ziggurat256_v1 distance=full_realized_weights gpu_median_seconds={:.6} wall_median_seconds={:.6} wall_min_seconds={:.6} wall_max_seconds={:.6} accepted={} target_seconds={:.6} target_met={}",
        rounds,
        FULL_PARAMETERS,
        result.actual_bo_median_gpu_seconds,
        result.actual_bo_median_wall_seconds,
        result.actual_bo_min_wall_seconds,
        result.actual_bo_max_wall_seconds,
        result.actual_bo_accepted,
        result.target_seconds,
        result.meets_target,
    );
}

struct SustainedTimings {
    sustained_model_gpu_seconds: f64,
    sustained_model_wall_seconds: f64,
    sustained_model_min_wall_seconds: f64,
    sustained_model_max_wall_seconds: f64,
    sustained_materialization_gpu_seconds: f64,
    sustained_projections_gpu_seconds: f64,
    sustained_pisa1_gpu_seconds: f64,
    sustained_ffn_gpu_seconds: f64,
    sustained_mps_projections_gpu_seconds: f64,
    sustained_mps_projections_wall_seconds: f64,
    sustained_mps_ffn_gpu_seconds: f64,
    sustained_mps_ffn_wall_seconds: f64,
    tail_gpu_seconds: f64,
    tail_wall_seconds: f64,
    complete_envelope_gpu_seconds: f64,
    complete_envelope_wall_seconds: f64,
    complete_envelope_min_wall_seconds: f64,
    complete_envelope_max_wall_seconds: f64,
    tail_max_abs_error: f64,
}

fn sustained(
    runtime: &Runtime,
    pipelines: &Pipelines,
    tensorops: &TensorOpsPipelines,
    sustained_pisa1: &Pisa1,
    buffers: &Buffers,
    rounds: u32,
    progress: &impl Fn(&str),
) -> Result<SustainedTimings, String> {
    let (
        sustained_model_gpu_seconds,
        sustained_model_wall_seconds,
        sustained_model_min_wall_seconds,
        sustained_model_max_wall_seconds,
    ) = benchmark_model(
        runtime,
        pipelines,
        tensorops,
        sustained_pisa1,
        buffers,
        rounds,
    )?;
    progress("sustained model complete; component breakdown and MPS references");
    let [
        sustained_materialization_gpu_seconds,
        sustained_projections_gpu_seconds,
        sustained_pisa1_gpu_seconds,
        sustained_ffn_gpu_seconds,
    ] = benchmark_stages(runtime, pipelines, tensorops, sustained_pisa1, buffers)?;
    let [
        (sustained_mps_projections_gpu_seconds, sustained_mps_projections_wall_seconds),
        (sustained_mps_ffn_gpu_seconds, sustained_mps_ffn_wall_seconds),
    ] = benchmark_mps(runtime, pipelines, &buffers)?;
    let (tail_gpu_seconds, tail_wall_seconds) =
        benchmark_tail(runtime, pipelines, tensorops, &buffers)?;
    progress("reference checks complete; complete envelope diagnostic");
    let (
        complete_envelope_gpu_seconds,
        complete_envelope_wall_seconds,
        complete_envelope_min_wall_seconds,
        complete_envelope_max_wall_seconds,
    ) = benchmark_envelope(
        runtime,
        pipelines,
        tensorops,
        sustained_pisa1,
        buffers,
        rounds,
    )?;
    let tail_max_abs_error = validate_tail(&buffers)?;
    Ok(SustainedTimings {
        sustained_model_gpu_seconds,
        sustained_model_wall_seconds,
        sustained_model_min_wall_seconds,
        sustained_model_max_wall_seconds,
        sustained_materialization_gpu_seconds,
        sustained_projections_gpu_seconds,
        sustained_pisa1_gpu_seconds,
        sustained_ffn_gpu_seconds,
        sustained_mps_projections_gpu_seconds,
        sustained_mps_projections_wall_seconds,
        sustained_mps_ffn_gpu_seconds,
        sustained_mps_ffn_wall_seconds,
        tail_gpu_seconds,
        tail_wall_seconds,
        complete_envelope_gpu_seconds,
        complete_envelope_wall_seconds,
        complete_envelope_min_wall_seconds,
        complete_envelope_max_wall_seconds,
        tail_max_abs_error,
    })
}

struct WorkEstimate {
    projected_ffn_seconds: f64,
    projected_ffn_and_projections_seconds: f64,
    projected_measured_model_seconds: f64,
    objective_flops: u64,
    tail_flops: u64,
    complete_objective_flops: u64,
    projection_objective_flops: u64,
    pisa1_objective_flops: u64,
}

fn estimate(
    prematerialized_layer_wall_seconds: f64,
    materialize_gate_up_gpu_seconds: f64,
    materialize_down_gpu_seconds: f64,
    projections_wall_seconds: f64,
    pisa1: &crate::fbt_pisa1::Pisa1Probe,
) -> Result<WorkEstimate, String> {
    let dense = 2u64 * u64::from(ROWS) * u64::from(WIDTH) * u64::from(GATE_UP)
        + 2u64 * u64::from(ROWS) * u64::from(EXPERT_WIDTH) * u64::from(WIDTH);
    let gate = 2u64 * u64::from(ROWS) * u64::from(WIDTH);
    let materialization_flops = 2u64
        * u64::from(EXPERTS)
        * (u64::from(WIDTH) * u64::from(GATE_UP) + u64::from(EXPERT_WIDTH) * u64::from(WIDTH));
    let projected_ffn_seconds = prematerialized_layer_wall_seconds
        * f64::from(MODEL_LAYERS * FEEDBACK_PASSES)
        + (materialize_gate_up_gpu_seconds + materialize_down_gpu_seconds)
            * f64::from(MODEL_LAYERS);
    let ffn_objective_flops = (dense + gate) * u64::from(MODEL_LAYERS * FEEDBACK_PASSES)
        + materialization_flops * u64::from(MODEL_LAYERS);
    let projection_pair_flops =
        2u64 * u64::from(ROWS) * u64::from(WIDTH) * u64::from(QKV_WIDTH + WIDTH);
    let projection_objective_flops =
        projection_pair_flops * u64::from(MODEL_LAYERS * FEEDBACK_PASSES);
    let pyramid_flops = u64::from(BATCH * PISA_LEAVES * HEAD_DIM * (PISA_BLOCK - 1))
        + 2 * u64::from(BATCH * (PISA_NODES - PISA_LEAVES) * HEAD_DIM);
    let routing_flops =
        u64::from(ROWS * HEAD_DIM * (QUERY_HEADS - 1)) + 2 * u64::from(ROWS * 48 * HEAD_DIM);
    let sparse_attention_flops =
        4 * u64::from(ROWS * QUERY_HEADS * PISA_SELECTED * PISA_BLOCK * HEAD_DIM);
    let pisa1_layer_flops = pyramid_flops + routing_flops + sparse_attention_flops;
    let pisa1_objective_flops = pisa1_layer_flops * u64::from(MODEL_LAYERS * FEEDBACK_PASSES);
    let objective_flops = ffn_objective_flops + projection_objective_flops + pisa1_objective_flops;
    let feedback_flops = 4u64 * u64::from(ROWS) * u64::from(WIDTH) * u64::from(WIDTH);
    let readout_flops = 2u64 * u64::from(ROWS) * u64::from(WIDTH) * u64::from(VOCAB);
    let tail_flops = feedback_flops + readout_flops;
    let complete_objective_flops = objective_flops + tail_flops;
    let projected_ffn_and_projections_seconds = projected_ffn_seconds
        + projections_wall_seconds * f64::from(MODEL_LAYERS * FEEDBACK_PASSES);
    let projected_measured_model_seconds = projected_ffn_and_projections_seconds
        + pisa1.layer_wall_seconds * f64::from(MODEL_LAYERS * FEEDBACK_PASSES);
    Ok(WorkEstimate {
        projected_ffn_seconds,
        projected_ffn_and_projections_seconds,
        projected_measured_model_seconds,
        objective_flops,
        tail_flops,
        complete_objective_flops,
        projection_objective_flops,
        pisa1_objective_flops,
    })
}

struct ComponentTimings {
    stage_gpu: [f64; 4],
    prematerialized_layer_gpu_seconds: f64,
    prematerialized_layer_wall_seconds: f64,
}

fn components(
    runtime: &Runtime,
    pipelines: &Pipelines,
    tensorops: &TensorOpsPipelines,
    buffers: &Buffers,
    rounds: u32,
) -> Result<ComponentTimings, String> {
    let mut stage_gpu = [0.0; 4];
    for (stage_index, (stage, name)) in Stage::ALL.into_iter().enumerate() {
        let mut seconds = Vec::with_capacity(3);
        for _ in 0..3 {
            let command = runtime.queue.new_command_buffer();
            encode(pipelines, buffers, command, Some(stage));
            seconds.push(complete(command)?);
        }
        seconds.sort_by(f64::total_cmp);
        stage_gpu[stage_index] = seconds[1];
        eprintln!(
            "TURBO_ENN_MOE_STAGE operation={name} median_gpu_seconds={:.9}",
            seconds[1]
        );
    }
    for _ in 0..3 {
        materialized_layer(runtime, pipelines, tensorops, &buffers)?;
    }
    let mut prematerialized_layer_gpu = Vec::with_capacity(rounds as usize);
    let mut prematerialized_layer_wall = Vec::with_capacity(rounds as usize);
    for _ in 0..rounds {
        let start = Instant::now();
        prematerialized_layer_gpu.push(materialized_layer(runtime, pipelines, tensorops, buffers)?);
        prematerialized_layer_wall.push(start.elapsed().as_secs_f64());
    }
    prematerialized_layer_gpu.sort_by(f64::total_cmp);
    prematerialized_layer_wall.sort_by(f64::total_cmp);
    let prematerialized_layer_gpu_seconds =
        prematerialized_layer_gpu[prematerialized_layer_gpu.len() / 2];
    let prematerialized_layer_wall_seconds =
        prematerialized_layer_wall[prematerialized_layer_wall.len() / 2];
    let values = unsafe {
        std::slice::from_raw_parts(
            buffers.output.contents().cast::<u16>(),
            (ROWS * WIDTH) as usize + 64,
        )
    };
    if values[..(ROWS * WIDTH) as usize]
        .iter()
        .any(|bits| bits & 0x7c00 == 0x7c00)
        || values[(ROWS * WIDTH) as usize..]
            .iter()
            .any(|&bits| bits != 0x7e00)
    {
        return Err("grouped MoE produced nonfinite output or overwrote its canary".into());
    }
    Ok(ComponentTimings {
        stage_gpu,
        prematerialized_layer_gpu_seconds,
        prematerialized_layer_wall_seconds,
    })
}

struct ProbeParity {
    prematerialized_gate_up_max_abs_error: f64,
    prematerialized_down_max_abs_error: f64,
    qkv_max_abs_error: f64,
    output_projection_max_abs_error: f64,
}

fn parity(
    buffers: &Buffers,
    mps_reference_gpu_seconds: f64,
    mps_projection_reference_gpu_seconds: f64,
) -> Result<ProbeParity, String> {
    eprintln!(
        "TURBO_ENN_MOE_PARITY_SAMPLE operation=gate_up row=0 column=0 cpu={:.9} mps={:.9} tensorops={:.9}",
        matmul_element(
            &buffers.grouped,
            &buffers.materialized_gate_up,
            0,
            0,
            WIDTH,
            GATE_UP,
        ),
        buffer_element(&buffers.mps_gate_up, 0, 0, GATE_UP),
        buffer_element(&buffers.gate_up, 0, 0, GATE_UP),
    );
    eprintln!(
        "TURBO_ENN_MOE_PARITY_SAMPLE operation=down row=0 column=0 cpu={:.9} mps={:.9} tensorops={:.9}",
        matmul_element(
            &buffers.activation,
            &buffers.materialized_down,
            0,
            0,
            EXPERT_WIDTH,
            WIDTH,
        ),
        buffer_element(&buffers.mps_down, 0, 0, WIDTH),
        buffer_element(&buffers.down, 0, 0, WIDTH),
    );
    eprintln!(
        "TURBO_ENN_MOE_MPS_REFERENCE scope=materialized_exact_batched_gemms gpu_seconds={mps_reference_gpu_seconds:.9}"
    );
    eprintln!(
        "TURBO_ENN_PROJECTION_PARITY_SAMPLE operation=qkv row=0 column=0 cpu={:.9} mps={:.9} tensorops={:.9}",
        matmul_dense(&buffers.input, &buffers.qkv_weights, 0, 0, WIDTH, QKV_WIDTH),
        buffer_element(&buffers.mps_qkv, 0, 0, QKV_WIDTH),
        buffer_element(&buffers.qkv, 0, 0, QKV_WIDTH),
    );
    eprintln!(
        "TURBO_ENN_PROJECTION_PARITY_SAMPLE operation=output row=0 column=0 cpu={:.9} mps={:.9} tensorops={:.9}",
        matmul_dense(
            &buffers.input,
            &buffers.output_projection_weights,
            0,
            0,
            WIDTH,
            WIDTH,
        ),
        buffer_element(&buffers.mps_projected, 0, 0, WIDTH),
        buffer_element(&buffers.projected, 0, 0, WIDTH),
    );
    eprintln!(
        "TURBO_ENN_PROJECTION_MPS_REFERENCE scope=fused_qkv_and_output_projection gpu_seconds={mps_projection_reference_gpu_seconds:.9}"
    );
    let prematerialized_gate_up_max_abs_error = compare_half(
        &buffers.mps_gate_up,
        &buffers.gate_up,
        (ROWS * GATE_UP) as usize,
        "MPS versus prematerialized TensorOps gate/up",
    )?;
    let prematerialized_down_max_abs_error = compare_half(
        &buffers.mps_down,
        &buffers.down,
        (ROWS * WIDTH) as usize,
        "MPS versus prematerialized TensorOps down",
    )?;
    let qkv_max_abs_error = compare_half(
        &buffers.mps_qkv,
        &buffers.qkv,
        (ROWS * QKV_WIDTH) as usize,
        "MPS versus TensorOps fused QKV projection",
    )?;
    let output_projection_max_abs_error = compare_half(
        &buffers.mps_projected,
        &buffers.projected,
        (ROWS * WIDTH) as usize,
        "MPS versus TensorOps output projection",
    )?;
    Ok(ProbeParity {
        prematerialized_gate_up_max_abs_error,
        prematerialized_down_max_abs_error,
        qkv_max_abs_error,
        output_projection_max_abs_error,
    })
}

struct MaterializationTimings {
    materialize_gate_up_gpu_seconds: f64,
    materialize_down_gpu_seconds: f64,
}

fn materialization(
    runtime: &Runtime,
    pipelines: &Pipelines,
    tensorops: &TensorOpsPipelines,
    buffers: &Buffers,
) -> Result<MaterializationTimings, String> {
    let materialize_gate_up_gpu_seconds = benchmark_materialize(
        runtime,
        &tensorops.materialize_gate_up,
        &[
            &buffers.gate_up_base,
            &buffers.gate_inner,
            &buffers.gate_outer,
            &buffers.materialized_gate_up,
        ],
        GATE_UP,
        WIDTH,
    )?;
    let materialize_down_gpu_seconds = benchmark_materialize(
        runtime,
        &tensorops.materialize_down,
        &[
            &buffers.down_base,
            &buffers.down_inner,
            &buffers.down_outer,
            &buffers.materialized_down,
        ],
        WIDTH,
        EXPERT_WIDTH,
    )?;
    materialized_layer(runtime, pipelines, tensorops, &buffers)?;
    let activation = runtime.buffer_with(&vec![0x7e00u16; (ROWS * EXPERT_WIDTH) as usize + 64]);
    let command = runtime.queue.new_command_buffer();
    tensor_groups(
        command,
        &tensorops.gate_activation,
        &[&buffers.grouped, &buffers.materialized_gate_up, &activation],
        MTLSize {
            width: u64::from(EXPERT_WIDTH.div_ceil(64)),
            height: u64::from(EXPERT_ROWS / 128),
            depth: u64::from(EXPERTS),
        },
    )?;
    complete(command)?;
    let activation_error = compare_half(
        &buffers.activation,
        &activation,
        (ROWS * EXPERT_WIDTH) as usize,
        "fused gate/up activation",
    )?;
    if activation_error > 0.002 {
        return Err(format!(
            "fused activation error {activation_error:.9} exceeds 0.002"
        ));
    }
    Ok(MaterializationTimings {
        materialize_gate_up_gpu_seconds,
        materialize_down_gpu_seconds,
    })
}
