use super::*;

pub(super) fn benchmark_bo(
    runtime: &Runtime,
    pipelines: &Pipelines,
    tensorops: &TensorOpsPipelines,
    pisa1: &Pisa1,
    buffers: &Buffers,
    weights: &CandidateWeights,
    rounds: u32,
    dataset: Option<&crate::pretrain_data::PretrainDataset>,
    check_fusion: bool,
    trace: bool,
    scorer_stage_samples: Option<u32>,
    control: BoControl<'_>,
) -> Result<ActualBoResult, String> {
    let learning_start = Instant::now();
    if let Some(dataset) = dataset {
        if dataset.batches() == 0 {
            return Err("pretraining dataset has no batches".into());
        }
        pretrain_batch(buffers, dataset, 0)?;
    }
    let (mut search, mut updates) =
        weights.search_shaped(control.perturbation, control.length, control.shape)?;
    let paired_scorer = if control.paired_objective {
        Some((
            runtime.device.new_command_queue(),
            Buffers::new(runtime),
            Pisa1::for_trial(runtime, control.kernel_trial)?,
        ))
    } else {
        None
    };
    if let (Some(dataset), Some((_, paired_buffers, _))) = (dataset, &paired_scorer) {
        pretrain_batch(paired_buffers, dataset, 0)?;
    }
    search.configure_enn(control.enn)?;
    if let Some(config) = control.reliability {
        search.configure_controller(config)?;
    }
    search.set_profiling(trace);
    let info = search.controller_info()?;
    let blocks = weights
        .tensors()
        .iter()
        .map(|(_, buffer, len)| buffer.length() as usize / 2 / len)
        .sum::<usize>();
    eprintln!(
        "TURBO_ENN_FULL_SPACE parameters={} blocks={blocks} context={CONTEXT} batch={BATCH} history_capacity={HISTORY_CAPACITY} storage=fp16 noise={} shape={} distance=exact_seeded_replay",
        info.dimensions,
        control.perturbation.name(),
        control.shape.name(),
    );
    eprintln!(
        "TURBO_ENN_PARAMETERS acquisition={:?} neighbors_max={} fit_neighbors={} epistemic_scale={} aleatoric_scale={} y_scale={} beta={} controller={} length_init={} length_min={} length_max={} proposal_seed={} acquisition_seed={}",
        control.enn.ask.acquisition,
        control.enn.ask.neighbors,
        control.enn.fit_neighbors,
        control.enn.ask.epistemic_scale,
        control.enn.ask.aleatoric_scale,
        control.enn.ask.y_scale,
        control.enn.ask.beta,
        if control.reliability.is_some() {
            "reliability"
        } else {
            "turbo"
        },
        control.length.length_init,
        control.length.length_min,
        control.length.length_max,
        control.proposal_seed,
        control.acquisition_seed,
    );

    let row = search.base_buffer();
    let initial_weights = weights.row(&row)?;
    let (initial, frozen_anchors) = initial_objective(
        runtime,
        pipelines,
        tensorops,
        pisa1,
        buffers,
        initial_weights,
        &mut search,
        dataset,
        check_fusion,
        scorer_stage_samples,
        control,
    )?;
    let initial_value = initial.reward;
    let initial_variance = initial.variance;
    let initial_scores = initial.sequence_nlls;
    eprintln!(
        "TURBO_ENN_ACTUAL_INITIAL reward={initial_value:.9} variance={initial_variance:.9} sequence_nlls={initial_scores:?}"
    );
    let heldout = heldout::Scorer {
        runtime,
        pipelines,
        tensorops,
        pisa1,
        buffers,
    };
    let mut validation = Vec::new();
    let initial_calls = if frozen_anchors.is_some() {
        dataset.map_or(1, |data| data.batches())
    } else {
        1
    };
    if let Some(dataset) = control.validation {
        validation.push(heldout.observe(
            dataset,
            initial_weights,
            0,
            initial_calls,
            learning_start,
        )?);
    }

    let mut log = RoundLog::new(rounds);
    let loop_start = Instant::now();
    for step in 0..rounds {
        execute_round(
            RoundContext {
                weights,
                paired_scorer: &paired_scorer,
                frozen_anchors: &frozen_anchors,
                dataset,
                control,
                heldout: &heldout,
                learning_start,
                initial_calls,
                rounds,
            },
            &mut search,
            &mut updates,
            &mut validation,
            step,
            &mut log,
        )?;
    }
    let loop_seconds = loop_start.elapsed().as_secs_f64();
    eprintln!(
        "TURBO_ENN_LOOP rounds={rounds} elapsed_seconds={loop_seconds:.9} rounds_per_second={:.6}",
        f64::from(rounds) / loop_seconds,
    );
    let RoundLog {
        mut wall,
        mut gpu,
        accepted,
        controller_seconds,
        controller_records,
        ..
    } = log;
    wall.sort_by(f64::total_cmp);
    gpu.sort_by(f64::total_cmp);
    Ok(ActualBoResult {
        parameters: weights.architecture.parameter_count(),
        loop_seconds,
        median_wall_seconds: wall[wall.len() / 2],
        min_wall_seconds: wall[0],
        max_wall_seconds: wall[wall.len() - 1],
        median_gpu_seconds: gpu[gpu.len() / 2],
        accepted,
        learning_seconds: learning_start.elapsed().as_secs_f64(),
        validation,
        controller_seconds,
        controller_records,
        updates,
    })
}

fn initial_objective(
    runtime: &Runtime,
    pipelines: &Pipelines,
    tensorops: &TensorOpsPipelines,
    pisa1: &Pisa1,
    buffers: &Buffers,
    initial_weights: CandidateRow<'_>,
    search: &mut SearchState,
    dataset: Option<&crate::pretrain_data::PretrainDataset>,
    check_fusion: bool,
    scorer_stage_samples: Option<u32>,
    control: BoControl<'_>,
) -> Result<(ObjectiveStats, Option<Vec<ObjectiveStats>>), String> {
    let unfused_scores = if check_fusion {
        let command = runtime.queue.new_command_buffer();
        objective_unfused(
            command,
            pipelines,
            tensorops,
            pisa1,
            buffers,
            initial_weights,
        )?;
        complete(command)?;
        Some(sequence_objective(buffers)?.sequence_nlls)
    } else {
        None
    };
    let command = runtime.queue.new_command_buffer();
    objective_fused(
        command,
        pipelines,
        tensorops,
        pisa1,
        buffers,
        initial_weights,
    )?;
    complete(command)?;
    let initial = sequence_objective(buffers)?;
    let initial_value = initial.reward;
    let initial_variance = initial.variance;
    let initial_scores = initial.sequence_nlls;
    if let Some(samples) = scorer_stage_samples {
        profile_scorer(
            runtime,
            pipelines,
            tensorops,
            pisa1,
            buffers,
            initial_weights,
            samples,
            initial_scores,
        )?;
    }
    if let Some(unfused_scores) = unfused_scores {
        let fusion_error = unfused_scores
            .into_iter()
            .zip(initial_scores)
            .map(|(unfused, fused)| (unfused - fused).abs())
            .fold(0.0f32, f32::max);
        if fusion_error > 1.0e-5 {
            return Err(format!(
                "fused candidate objective changed sequence NLL by {fusion_error:.9}"
            ));
        }
        eprintln!("TURBO_ENN_FUSION_PARITY fused_unfused_max_abs_error={fusion_error:.9}");
    }
    search.observe_initial(
        if control.paired_objective
            || control.objective_reference == crate::config::ObjectiveReference::FrozenInitial
        {
            0.0
        } else {
            initial_value
        },
        if control.paired_objective
            || control.objective_reference == crate::config::ObjectiveReference::FrozenInitial
        {
            0.0
        } else {
            initial_variance
        },
    )?;
    let frozen_anchors = if control.objective_reference
        == crate::config::ObjectiveReference::FrozenInitial
    {
        let dataset = dataset.ok_or("frozen objective reference requires a pretraining dataset")?;
        let mut anchors = Vec::with_capacity(dataset.batches() as usize);
        anchors.push(initial);
        for batch in 1..dataset.batches() {
            pretrain_batch(buffers, dataset, batch)?;
            let command = runtime.queue.new_command_buffer();
            objective_fused(
                command,
                pipelines,
                tensorops,
                pisa1,
                buffers,
                initial_weights,
            )?;
            complete(command)?;
            anchors.push(sequence_objective(buffers)?);
        }
        Some(anchors)
    } else {
        None
    };
    Ok((initial, frozen_anchors))
}
