use super::*;

pub(super) fn projection_pair(
    command: &CommandBufferRef,
    tensorops: &TensorOpsPipelines,
    buffers: &Buffers,
) -> Result<(), String> {
    tensor_groups(
        command,
        &tensorops.qkv,
        &[&buffers.input, &buffers.qkv_weights, &buffers.qkv],
        MTLSize {
            width: u64::from(QKV_WIDTH / 64),
            height: u64::from(ROWS / 128),
            depth: 1,
        },
    )?;
    tensor_groups(
        command,
        &tensorops.output_projection,
        &[
            &buffers.input,
            &buffers.output_projection_weights,
            &buffers.projected,
        ],
        MTLSize {
            width: u64::from(WIDTH / 64),
            height: u64::from(ROWS / 128),
            depth: 1,
        },
    )
}

pub(super) fn sustained_model(
    command: &CommandBufferRef,
    pipelines: &Pipelines,
    tensorops: &TensorOpsPipelines,
    pisa1: &Pisa1,
    buffers: &Buffers,
) -> Result<(), String> {
    sustained_materialize(command, tensorops, buffers);
    for _ in 0..MODEL_LAYERS * FEEDBACK_PASSES {
        sustained_qkv(command, tensorops, &buffers.input, buffers)?;
        pisa1.encode_layer(command, &buffers.qkv);
        sustained_output(command, tensorops, pisa1.output(), buffers)?;
        sustained_ffn(command, pipelines, tensorops, buffers)?;
    }
    Ok(())
}

pub(super) fn sustained_materialize(
    command: &CommandBufferRef,
    tensorops: &TensorOpsPipelines,
    buffers: &Buffers,
) {
    for _ in 0..MODEL_LAYERS {
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&tensorops.materialize_gate_up);
        for (index, buffer) in [
            &buffers.gate_up_base,
            &buffers.gate_inner,
            &buffers.gate_outer,
            &buffers.materialized_gate_up,
        ]
        .into_iter()
        .enumerate()
        {
            encoder.set_buffer(index as u64, Some(buffer), 0);
        }
        encoder.dispatch_threads(
            MTLSize {
                width: u64::from(GATE_UP / 4),
                height: u64::from(WIDTH),
                depth: u64::from(EXPERTS),
            },
            MTLSize {
                width: 64,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();

        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&tensorops.materialize_down);
        for (index, buffer) in [
            &buffers.down_base,
            &buffers.down_inner,
            &buffers.down_outer,
            &buffers.materialized_down,
        ]
        .into_iter()
        .enumerate()
        {
            encoder.set_buffer(index as u64, Some(buffer), 0);
        }
        encoder.dispatch_threads(
            MTLSize {
                width: u64::from(WIDTH / 4),
                height: u64::from(EXPERT_WIDTH),
                depth: u64::from(EXPERTS),
            },
            MTLSize {
                width: 64,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
    }
}

pub(super) fn sustained_qkv(
    command: &CommandBufferRef,
    tensorops: &TensorOpsPipelines,
    input: &BufferRef,
    buffers: &Buffers,
) -> Result<(), String> {
    tensor_groups(
        command,
        &tensorops.qkv,
        &[input, &buffers.qkv_weights, &buffers.qkv],
        MTLSize {
            width: u64::from(QKV_WIDTH / 64),
            height: u64::from(ROWS / 128),
            depth: 1,
        },
    )
}

pub(super) fn sustained_output(
    command: &CommandBufferRef,
    tensorops: &TensorOpsPipelines,
    input: &BufferRef,
    buffers: &Buffers,
) -> Result<(), String> {
    tensor_groups(
        command,
        &tensorops.output_projection,
        &[
            input,
            &buffers.output_projection_weights,
            &buffers.projected,
        ],
        MTLSize {
            width: u64::from(WIDTH / 64),
            height: u64::from(ROWS / 128),
            depth: 1,
        },
    )
}

pub(super) fn sustained_ffn(
    command: &CommandBufferRef,
    pipelines: &Pipelines,
    tensorops: &TensorOpsPipelines,
    buffers: &Buffers,
) -> Result<(), String> {
    ffn_input(
        command,
        pipelines,
        tensorops,
        &buffers.input,
        &buffers.input,
        buffers,
    )
}

pub(super) fn model_shape() -> MoeShape {
    model_rows(ROWS)
}

pub(super) fn model_rows(rows: u32) -> MoeShape {
    MoeShape {
        rows,
        width: WIDTH,
        experts: EXPERTS,
        rows_per_expert: rows / EXPERTS,
        expert_width: EXPERT_WIDTH,
    }
}

pub(super) fn encode_rms(
    command: &CommandBufferRef,
    pipelines: &Pipelines,
    input: &BufferRef,
    weight: &BufferRef,
    output: &BufferRef,
) {
    let shape = model_shape();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipelines.rms);
    encoder.set_buffer(0, Some(input), 0);
    encoder.set_buffer(1, Some(weight), 0);
    encoder.set_buffer(2, Some(output), 0);
    encoder.set_bytes(
        3,
        std::mem::size_of::<MoeShape>() as u64,
        (&shape as *const MoeShape).cast(),
    );
    encoder.dispatch_thread_groups(
        thread_group(u64::from(ROWS)),
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();
}

pub(super) fn encode_residual(
    command: &CommandBufferRef,
    pipelines: &Pipelines,
    input: &BufferRef,
    branch: &BufferRef,
    output: &BufferRef,
) {
    let shape = model_shape();
    dispatch(
        command,
        &pipelines.residual,
        &[input, branch, output],
        (&shape as *const MoeShape).cast(),
        std::mem::size_of::<MoeShape>() as u64,
        u64::from(ROWS) * u64::from(WIDTH),
    );
}

pub(super) fn ffn_input(
    command: &CommandBufferRef,
    pipelines: &Pipelines,
    tensorops: &TensorOpsPipelines,
    normalized: &BufferRef,
    residual: &BufferRef,
    buffers: &Buffers,
) -> Result<(), String> {
    let shape = model_shape();
    let shape_ptr = (&shape as *const MoeShape).cast();
    let shape_bytes = std::mem::size_of::<MoeShape>() as u64;
    dispatch(
        command,
        &pipelines.gate,
        &[normalized, &buffers.router, &buffers.gates],
        shape_ptr,
        shape_bytes,
        u64::from(ROWS) * 32,
    );
    dispatch(
        command,
        &pipelines.group,
        &[normalized, &buffers.grouped],
        shape_ptr,
        shape_bytes,
        u64::from(ROWS) * u64::from(WIDTH),
    );
    encode_tensorops(
        command,
        &tensorops.gate_up,
        &[
            &buffers.grouped,
            &buffers.materialized_gate_up,
            &buffers.gate_up,
        ],
        GATE_UP,
    )?;
    encode(pipelines, buffers, command, Some(Stage::Swiglu));
    encode_tensorops(
        command,
        &tensorops.down,
        &[
            &buffers.activation,
            &buffers.materialized_down,
            &buffers.down,
        ],
        WIDTH,
    )?;
    dispatch(
        command,
        &pipelines.ungroup,
        &[residual, &buffers.down, &buffers.gates, &buffers.output],
        shape_ptr,
        shape_bytes,
        u64::from(ROWS) * u64::from(WIDTH),
    );
    Ok(())
}

pub(super) fn encode_feedback(
    command: &CommandBufferRef,
    pipelines: &Pipelines,
    tensorops: &TensorOpsPipelines,
    input: &BufferRef,
    buffers: &Buffers,
) -> Result<(), String> {
    encode_rms(
        command,
        pipelines,
        input,
        &buffers.unit_norm,
        &buffers.normalized,
    );
    width_projection(
        command,
        tensorops,
        input,
        FeedbackProjection {
            weights: &buffers.feedback_state_weights,
            output: &buffers.feedback_state,
        },
    )?;
    width_projection(
        command,
        tensorops,
        &buffers.normalized,
        FeedbackProjection {
            weights: &buffers.feedback_gate_weights,
            output: &buffers.feedback_gate,
        },
    )?;
    let shape = model_shape();
    dispatch(
        command,
        &pipelines.feedback_fuse,
        &[
            &buffers.feedback_state,
            &buffers.feedback_gate,
            &buffers.feedback,
        ],
        (&shape as *const MoeShape).cast(),
        std::mem::size_of::<MoeShape>() as u64,
        u64::from(ROWS) * u64::from(WIDTH),
    );
    Ok(())
}

struct FeedbackProjection<'a> {
    weights: &'a BufferRef,
    output: &'a BufferRef,
}

fn width_projection(
    command: &CommandBufferRef,
    tensorops: &TensorOpsPipelines,
    input: &BufferRef,
    projection: FeedbackProjection<'_>,
) -> Result<(), String> {
    tensor_groups(
        command,
        &tensorops.output_projection,
        &[input, projection.weights, projection.output],
        MTLSize {
            width: u64::from(WIDTH / 64),
            height: u64::from(ROWS / 128),
            depth: 1,
        },
    )
}

pub(super) fn readout_loss(
    command: &CommandBufferRef,
    pipelines: &Pipelines,
    tensorops: &TensorOpsPipelines,
    input: &BufferRef,
    buffers: &Buffers,
) -> Result<(), String> {
    let logits = buffers.logits()?;
    tensor_groups(
        command,
        &tensorops.readout,
        &[input, &buffers.readout_weights, logits],
        MTLSize {
            width: u64::from(VOCAB / 64),
            height: u64::from(ROWS / 128),
            depth: 1,
        },
    )?;
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipelines.cross_entropy);
    encoder.set_buffer(0, Some(logits), 0);
    encoder.set_buffer(1, Some(&buffers.labels), 0);
    encoder.set_buffer(2, Some(&buffers.losses), 0);
    encoder.dispatch_thread_groups(
        thread_group(u64::from(ROWS)),
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();
    Ok(())
}

pub(super) fn complete_envelope(
    command: &CommandBufferRef,
    pipelines: &Pipelines,
    tensorops: &TensorOpsPipelines,
    pisa1: &Pisa1,
    buffers: &Buffers,
) -> Result<(), String> {
    sustained_materialize(command, tensorops, buffers);
    let mut input: &BufferRef = &buffers.input;
    for pass in 0..FEEDBACK_PASSES {
        for _ in 0..MODEL_LAYERS {
            encode_rms(
                command,
                pipelines,
                input,
                &buffers.unit_norm,
                &buffers.normalized,
            );
            sustained_qkv(command, tensorops, &buffers.normalized, buffers)?;
            pisa1.encode_layer(command, &buffers.qkv);
            sustained_output(command, tensorops, pisa1.output(), buffers)?;
            encode_residual(
                command,
                pipelines,
                input,
                &buffers.projected,
                &buffers.attention_state,
            );
            encode_rms(
                command,
                pipelines,
                &buffers.attention_state,
                &buffers.unit_norm,
                &buffers.normalized,
            );
            ffn_input(
                command,
                pipelines,
                tensorops,
                &buffers.normalized,
                &buffers.attention_state,
                buffers,
            )?;
            input = &buffers.output;
        }
        if pass + 1 < FEEDBACK_PASSES {
            encode_feedback(command, pipelines, tensorops, input, buffers)?;
            input = &buffers.feedback;
        }
    }
    encode_rms(
        command,
        pipelines,
        input,
        &buffers.unit_norm,
        &buffers.normalized,
    );
    readout_loss(command, pipelines, tensorops, &buffers.normalized, buffers)
}

pub(super) fn benchmark_envelope(
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
        complete_envelope(command, pipelines, tensorops, pisa1, buffers)?;
        let gpu_seconds = complete(command)?;
        let elapsed = start.elapsed().as_secs_f64();
        if iteration > 0 {
            gpu.push(gpu_seconds);
            wall.push(elapsed);
        }
        eprintln!(
            "[diagnostic] complete envelope | sample {iteration}/{rounds} (0=warmup) | wall {elapsed:.3}s"
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

pub(super) fn benchmark_tail(
    runtime: &Runtime,
    pipelines: &Pipelines,
    tensorops: &TensorOpsPipelines,
    buffers: &Buffers,
) -> Result<(f64, f64), String> {
    let mut gpu = Vec::with_capacity(3);
    let mut wall = Vec::with_capacity(3);
    for iteration in 0..4 {
        let start = Instant::now();
        let command = runtime.queue.new_command_buffer();
        encode_feedback(command, pipelines, tensorops, &buffers.output, buffers)?;
        readout_loss(command, pipelines, tensorops, &buffers.output, buffers)?;
        let gpu_seconds = complete(command)?;
        if iteration > 0 {
            gpu.push(gpu_seconds);
            wall.push(start.elapsed().as_secs_f64());
        }
    }
    gpu.sort_by(f64::total_cmp);
    wall.sort_by(f64::total_cmp);
    Ok((gpu[1], wall[1]))
}

pub(super) fn half_bytes(elements: u64) -> u64 {
    elements * std::mem::size_of::<u16>() as u64
}
