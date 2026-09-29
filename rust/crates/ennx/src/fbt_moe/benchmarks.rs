use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn profile_scorer(
    runtime: &Runtime,
    pipelines: &Pipelines,
    tensorops: &TensorOpsPipelines,
    pisa1: &Pisa1,
    buffers: &Buffers,
    weights: CandidateRow<'_>,
    samples: u32,
    reference: [f32; BATCH as usize],
) -> Result<(), String> {
    crate::apple_gpu::require_power()?;
    let trace = ScorerStageTrace::new(runtime)?;
    let mut stage_times = std::collections::BTreeMap::<&str, Vec<f64>>::new();
    let mut scorer_times = Vec::with_capacity(samples as usize);
    let mut max_error = 0.0f32;
    for _ in 0..samples {
        crate::apple_gpu::require_power()?;
        let command = runtime.queue.new_command_buffer();
        objective_trace(
            command,
            pipelines,
            tensorops,
            pisa1,
            buffers,
            weights,
            Some(&trace),
        )?;
        trace.resolve(command);
        let (mut cpu_start, mut gpu_start) = (0, 0);
        runtime
            .device
            .sample_timestamps(&mut cpu_start, &mut gpu_start);
        scorer_times.push(complete(command)? * 1000.0);
        crate::apple_gpu::require_power()?;
        let (mut cpu_end, mut gpu_end) = (0, 0);
        runtime.device.sample_timestamps(&mut cpu_end, &mut gpu_end);
        let cpu_span = cpu_end
            .checked_sub(cpu_start)
            .ok_or("nonmonotonic CPU timestamp")?;
        let gpu_span = gpu_end
            .checked_sub(gpu_start)
            .ok_or("nonmonotonic GPU timestamp")?;
        if gpu_span == 0 {
            return Err("zero GPU timestamp calibration span".into());
        }
        let scale_ns_per_tick = cpu_span as f64 / gpu_span as f64;
        for (stage, duration_ms) in trace.durations_ms(scale_ns_per_tick)? {
            stage_times.entry(stage).or_default().push(duration_ms);
        }
        for (actual, expected) in sequence_objective(buffers)?
            .sequence_nlls
            .into_iter()
            .zip(reference)
        {
            max_error = max_error.max((actual - expected).abs());
        }
        if max_error > 1.0e-5 {
            return Err(format!(
                "stage timing changed sequence NLL by {max_error:.9}"
            ));
        }
    }
    let mut medians = std::collections::BTreeMap::new();
    for (stage, mut durations) in stage_times {
        durations.sort_by(f64::total_cmp);
        medians.insert(stage, durations[durations.len() / 2]);
    }
    scorer_times.sort_by(f64::total_cmp);
    eprintln!(
        "TURBO_ENN_SCORER_STAGES samples={samples} profiled_gpu_ms={:.3} stage_ms={medians:?} stage_sum_ms={:.3} max_nll_error={max_error:.9}",
        scorer_times[scorer_times.len() / 2],
        medians.values().sum::<f64>(),
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn benchmark_model(
    runtime: &Runtime,
    pipelines: &Pipelines,
    tensorops: &TensorOpsPipelines,
    pisa1: &Pisa1,
    buffers: &Buffers,
    rounds: u32,
) -> Result<(f64, f64, f64, f64), String> {
    let mut gpu = Vec::with_capacity(rounds as usize);
    let mut wall = Vec::with_capacity(rounds as usize);
    for iteration in 0..rounds + 1 {
        let start = Instant::now();
        let command = runtime.queue.new_command_buffer();
        sustained_model(command, pipelines, tensorops, pisa1, buffers)?;
        let gpu_seconds = complete(command)?;
        let elapsed = start.elapsed().as_secs_f64();
        if iteration > 0 {
            gpu.push(gpu_seconds);
            wall.push(elapsed);
        }
        eprintln!(
            "[diagnostic] sustained model | sample {iteration}/{rounds} (0=warmup) | wall {elapsed:.3}s"
        );
    }
    gpu.sort_by(f64::total_cmp);
    wall.sort_by(f64::total_cmp);
    Ok((
        gpu[gpu.len() / 2],
        wall[wall.len() / 2],
        wall[0],
        wall[wall.len() - 1],
    ))
}

pub(super) fn benchmark_mps(
    runtime: &Runtime,
    pipelines: &Pipelines,
    buffers: &Buffers,
) -> Result<[(f64, f64); 2], String> {
    let mut projection_gpu = Vec::with_capacity(3);
    let mut projection_wall = Vec::with_capacity(3);
    let mut ffn_gpu = Vec::with_capacity(3);
    let mut ffn_wall = Vec::with_capacity(3);
    let mut matmul = Matmul::default();
    for iteration in 0..4 {
        let start = Instant::now();
        let command = runtime.queue.new_command_buffer();
        for _ in 0..MODEL_LAYERS * FEEDBACK_PASSES {
            matmul.encode(
                &runtime.device,
                command,
                Matrix::half(&buffers.input, ROWS, WIDTH),
                Matrix::half(&buffers.qkv_weights, WIDTH, QKV_WIDTH),
                Matrix::half(&buffers.qkv, ROWS, QKV_WIDTH),
                false,
                1.0,
            )?;
            matmul.encode(
                &runtime.device,
                command,
                Matrix::half(&buffers.input, ROWS, WIDTH),
                Matrix::half(&buffers.output_projection_weights, WIDTH, WIDTH),
                Matrix::half(&buffers.projected, ROWS, WIDTH),
                false,
                1.0,
            )?;
        }
        let gpu = complete(command)?;
        let wall = start.elapsed().as_secs_f64();
        if iteration > 0 {
            projection_gpu.push(gpu);
            projection_wall.push(wall);
        }

        let start = Instant::now();
        let command = runtime.queue.new_command_buffer();
        for _ in 0..MODEL_LAYERS * FEEDBACK_PASSES {
            encode(pipelines, buffers, command, Some(Stage::Gate));
            encode(pipelines, buffers, command, Some(Stage::Group));
            matmul.encode(
                &runtime.device,
                command,
                Matrix::half(&buffers.grouped, EXPERT_ROWS, WIDTH).layout(
                    EXPERTS,
                    u64::from(EXPERT_ROWS) * u64::from(WIDTH),
                    0,
                ),
                Matrix::half(&buffers.materialized_gate_up, WIDTH, GATE_UP).layout(
                    EXPERTS,
                    u64::from(WIDTH) * u64::from(GATE_UP),
                    0,
                ),
                Matrix::half(&buffers.gate_up, EXPERT_ROWS, GATE_UP).layout(
                    EXPERTS,
                    u64::from(EXPERT_ROWS) * u64::from(GATE_UP),
                    0,
                ),
                false,
                1.0,
            )?;
            encode(pipelines, buffers, command, Some(Stage::Swiglu));
            matmul.encode(
                &runtime.device,
                command,
                Matrix::half(&buffers.activation, EXPERT_ROWS, EXPERT_WIDTH).layout(
                    EXPERTS,
                    u64::from(EXPERT_ROWS) * u64::from(EXPERT_WIDTH),
                    0,
                ),
                Matrix::half(&buffers.materialized_down, EXPERT_WIDTH, WIDTH).layout(
                    EXPERTS,
                    u64::from(EXPERT_WIDTH) * u64::from(WIDTH),
                    0,
                ),
                Matrix::half(&buffers.down, EXPERT_ROWS, WIDTH).layout(
                    EXPERTS,
                    u64::from(EXPERT_ROWS) * u64::from(WIDTH),
                    0,
                ),
                false,
                1.0,
            )?;
            encode(pipelines, buffers, command, Some(Stage::Ungroup));
        }
        let gpu = complete(command)?;
        let wall = start.elapsed().as_secs_f64();
        if iteration > 0 {
            ffn_gpu.push(gpu);
            ffn_wall.push(wall);
        }
    }
    for values in [
        &mut projection_gpu,
        &mut projection_wall,
        &mut ffn_gpu,
        &mut ffn_wall,
    ] {
        values.sort_by(f64::total_cmp);
    }
    Ok([
        (projection_gpu[1], projection_wall[1]),
        (ffn_gpu[1], ffn_wall[1]),
    ])
}

pub(super) fn benchmark_stages(
    runtime: &Runtime,
    pipelines: &Pipelines,
    tensorops: &TensorOpsPipelines,
    pisa1: &Pisa1,
    buffers: &Buffers,
) -> Result<[f64; 4], String> {
    let mut samples = [
        Vec::with_capacity(3),
        Vec::with_capacity(3),
        Vec::with_capacity(3),
        Vec::with_capacity(3),
    ];
    for iteration in 0..4 {
        let command = runtime.queue.new_command_buffer();
        sustained_materialize(command, tensorops, buffers);
        let materialization = complete(command)?;

        let command = runtime.queue.new_command_buffer();
        for _ in 0..MODEL_LAYERS * FEEDBACK_PASSES {
            sustained_qkv(command, tensorops, &buffers.input, buffers)?;
            sustained_output(command, tensorops, &buffers.input, buffers)?;
        }
        let projections = complete(command)?;

        let command = runtime.queue.new_command_buffer();
        for _ in 0..MODEL_LAYERS * FEEDBACK_PASSES {
            pisa1.encode_layer(command, &buffers.qkv);
        }
        let attention = complete(command)?;

        let command = runtime.queue.new_command_buffer();
        for _ in 0..MODEL_LAYERS * FEEDBACK_PASSES {
            sustained_ffn(command, pipelines, tensorops, buffers)?;
        }
        let ffn = complete(command)?;

        if iteration > 0 {
            for (slot, value) in [materialization, projections, attention, ffn]
                .into_iter()
                .enumerate()
            {
                samples[slot].push(value);
            }
        }
    }
    let mut medians = [0.0; 4];
    for (index, values) in samples.iter_mut().enumerate() {
        values.sort_by(f64::total_cmp);
        medians[index] = values[values.len() / 2];
    }
    Ok(medians)
}

pub(super) fn benchmark_projections(
    runtime: &Runtime,
    tensorops: &TensorOpsPipelines,
    buffers: &Buffers,
    rounds: u32,
) -> Result<(f64, f64), String> {
    let mut gpu = Vec::with_capacity(rounds as usize);
    let mut wall = Vec::with_capacity(rounds as usize);
    for iteration in 0..rounds + 3 {
        let command = runtime.queue.new_command_buffer();
        projection_pair(command, tensorops, buffers)?;
        let start = Instant::now();
        let gpu_seconds = complete(command)?;
        let wall_seconds = start.elapsed().as_secs_f64();
        if iteration >= 3 {
            gpu.push(gpu_seconds);
            wall.push(wall_seconds);
        }
    }
    gpu.sort_by(f64::total_cmp);
    wall.sort_by(f64::total_cmp);
    Ok((gpu[gpu.len() / 2], wall[wall.len() / 2]))
}

pub(super) fn mps_projections(runtime: &Runtime, buffers: &Buffers) -> Result<f64, String> {
    let mut matmul = Matmul::default();
    let command = runtime.queue.new_command_buffer();
    matmul.encode(
        &runtime.device,
        command,
        Matrix::half(&buffers.input, ROWS, WIDTH),
        Matrix::half(&buffers.qkv_weights, WIDTH, QKV_WIDTH),
        Matrix::half(&buffers.mps_qkv, ROWS, QKV_WIDTH),
        false,
        1.0,
    )?;
    matmul.encode(
        &runtime.device,
        command,
        Matrix::half(&buffers.input, ROWS, WIDTH),
        Matrix::half(&buffers.output_projection_weights, WIDTH, WIDTH),
        Matrix::half(&buffers.mps_projected, ROWS, WIDTH),
        false,
        1.0,
    )?;
    complete(command)
}
