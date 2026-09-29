use super::*;

pub(super) fn rms_at(
    command: &CommandBufferRef,
    pipelines: &Pipelines,
    input: &BufferRef,
    weight: &BufferRef,
    weight_offset: u64,
    output: &BufferRef,
) {
    let encoder = command.new_compute_command_encoder();
    rms_command(&encoder, pipelines, input, weight, weight_offset, output);
    encoder.end_encoding();
}

pub(super) fn rms_command(
    encoder: &ComputeCommandEncoderRef,
    pipelines: &Pipelines,
    input: &BufferRef,
    weight: &BufferRef,
    weight_offset: u64,
    output: &BufferRef,
) {
    rms_rows(
        encoder,
        pipelines,
        input,
        weight,
        weight_offset,
        output,
        ROWS,
    );
}

#[allow(clippy::too_many_arguments)]
pub(super) fn rms_rows(
    encoder: &ComputeCommandEncoderRef,
    pipelines: &Pipelines,
    input: &BufferRef,
    weight: &BufferRef,
    weight_offset: u64,
    output: &BufferRef,
    rows: u32,
) {
    rms_offsets(
        encoder,
        pipelines,
        (input, 0),
        (weight, weight_offset),
        (output, 0),
        rows,
    );
}

pub(super) fn rms_offsets(
    encoder: &ComputeCommandEncoderRef,
    pipelines: &Pipelines,
    input: (&BufferRef, u64),
    weight: (&BufferRef, u64),
    output: (&BufferRef, u64),
    rows: u32,
) {
    let shape = model_rows(rows);
    encoder.set_compute_pipeline_state(&pipelines.rms);
    encoder.set_buffer(0, Some(input.0), input.1);
    encoder.set_buffer(1, Some(weight.0), weight.1);
    encoder.set_buffer(2, Some(output.0), output.1);
    encoder.set_bytes(
        3,
        std::mem::size_of::<MoeShape>() as u64,
        (&shape as *const MoeShape).cast(),
    );
    encoder.dispatch_thread_groups(
        thread_group(u64::from(rows)),
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
}

pub(super) fn candidate_ffn(
    command: &CommandBufferRef,
    pipelines: &Pipelines,
    normalized: &BufferRef,
    residual: &BufferRef,
    layer: u32,
    buffers: &Buffers,
    weights: CandidateRow<'_>,
    next_norm: Option<(&BufferRef, u64)>,
) -> Result<(), String> {
    let shape = model_shape();
    let shape_ptr = (&shape as *const MoeShape).cast();
    let shape_bytes = std::mem::size_of::<MoeShape>() as u64;
    let router_offset =
        half_bytes(u64::from(layer) * u64::from(WIDTH) * u64::from(routing::ROUTED_EXPERTS));
    let gate_up_offset = half_bytes(
        u64::from(layer)
            * u64::from(routing::MOE_EXPERTS)
            * u64::from(WIDTH)
            * u64::from(routing::MOE_GATEUP),
    );
    let down_offset = half_bytes(
        u64::from(layer)
            * u64::from(routing::MOE_EXPERTS)
            * u64::from(routing::ROUTED_WIDTH)
            * u64::from(WIDTH),
    );
    pipelines.fine_grained.encode(
        command,
        &buffers.fine_grained,
        normalized,
        0,
        weights.buffer,
        weights.router + router_offset,
        weights.gate_up + gate_up_offset,
        weights.down + down_offset,
        &buffers.down,
    )?;
    if let Some((weight, weight_offset)) = next_norm {
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipelines.residual_rms);
        encoder.set_buffer(0, Some(residual), 0);
        encoder.set_buffer(1, Some(&buffers.down), 0);
        encoder.set_buffer(2, Some(weight), weight_offset);
        encoder.set_buffer(3, Some(&buffers.output), 0);
        encoder.set_buffer(4, Some(&buffers.normalized), 0);
        encoder.set_bytes(5, shape_bytes, shape_ptr);
        encoder.dispatch_thread_groups(
            thread_group(u64::from(ROWS)),
            MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
    } else {
        encode_residual(command, pipelines, residual, &buffers.down, &buffers.output);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn ffn_rows(
    encoder: &ComputeCommandEncoderRef,
    pipelines: &Pipelines,
    normalized: &BufferRef,
    residual: &BufferRef,
    layer: u32,
    buffers: &Buffers,
    weights: CandidateRow<'_>,
    next_norm: (&BufferRef, u64),
    row_start: u32,
    rows: u32,
) -> Result<(), String> {
    let (router_offset, gate_up_offset, down_offset) = ffn_offsets(layer);
    let activation_offset = half_bytes(u64::from(row_start) * u64::from(WIDTH));
    if pipelines.fine_grained.fused_combine() {
        let fine = &pipelines.fine_grained;
        let fine_buffers = &buffers.fine_grained;
        fine.route_rows(
            encoder,
            fine_buffers,
            normalized,
            activation_offset,
            weights.buffer,
            weights.router + router_offset,
            rows,
        )?;
        let int8_gate = fine.int8_gate();
        let gate_weights = if int8_gate {
            buffers.quantized_gateup()?
        } else if fine.interleaved_gate() {
            buffers.interleaved_gateup()?
        } else {
            weights.buffer
        };
        let gate_offset = if int8_gate {
            gate_up_offset / 2
        } else if fine.interleaved_gate() {
            gate_up_offset
        } else {
            weights.gate_up + gate_up_offset
        };
        fine.gate_rows(
            encoder,
            fine_buffers,
            normalized,
            activation_offset,
            gate_weights,
            gate_offset,
            rows,
        )?;
        fine.down_rows(
            encoder,
            fine_buffers,
            weights.buffer,
            weights.down + down_offset,
            rows,
        )?;
        fine.combine_offsets(
            encoder,
            fine_buffers,
            (residual, activation_offset),
            next_norm,
            (&buffers.output, activation_offset),
            (&buffers.normalized, activation_offset),
            rows,
        );
        return Ok(());
    }
    if row_start != 0 || rows != ROWS {
        return Err("active-row scoring requires fused MoE combine/RMS".into());
    }
    pipelines.fine_grained.encode_on(
        encoder,
        &buffers.fine_grained,
        normalized,
        0,
        weights.buffer,
        weights.router + router_offset,
        weights.gate_up + gate_up_offset,
        weights.down + down_offset,
        &buffers.down,
    )?;

    ffn_finish(encoder, pipelines, residual, buffers, next_norm);
    Ok(())
}

pub(super) fn ffn_offsets(layer: u32) -> (u64, u64, u64) {
    let router_offset =
        half_bytes(u64::from(layer) * u64::from(WIDTH) * u64::from(routing::ROUTED_EXPERTS));
    let gate_up_offset = half_bytes(
        u64::from(layer)
            * u64::from(routing::MOE_EXPERTS)
            * u64::from(WIDTH)
            * u64::from(routing::MOE_GATEUP),
    );
    let down_offset = half_bytes(
        u64::from(layer)
            * u64::from(routing::MOE_EXPERTS)
            * u64::from(routing::ROUTED_WIDTH)
            * u64::from(WIDTH),
    );
    (router_offset, gate_up_offset, down_offset)
}

pub(super) fn ffn_finish(
    encoder: &ComputeCommandEncoderRef,
    pipelines: &Pipelines,
    residual: &BufferRef,
    buffers: &Buffers,
    next_norm: (&BufferRef, u64),
) {
    let shape = model_shape();
    encoder.set_compute_pipeline_state(&pipelines.residual_rms);
    encoder.set_buffer(0, Some(residual), 0);
    encoder.set_buffer(1, Some(&buffers.down), 0);
    encoder.set_buffer(2, Some(next_norm.0), next_norm.1);
    encoder.set_buffer(3, Some(&buffers.output), 0);
    encoder.set_buffer(4, Some(&buffers.normalized), 0);
    encoder.set_bytes(
        5,
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
    encoder.memory_barrier_with_resources(&[&buffers.output, &buffers.normalized]);
}

pub(super) fn objective_unfused(
    command: &CommandBufferRef,
    pipelines: &Pipelines,
    tensorops: &TensorOpsPipelines,
    pisa1: &Pisa1,
    buffers: &Buffers,
    weights: CandidateRow<'_>,
) -> Result<(), String> {
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipelines.embed);
    encoder.set_buffer(0, Some(weights.buffer), weights.readout);
    encoder.set_buffer(1, Some(&buffers.tokens), 0);
    encoder.set_buffer(2, Some(&buffers.input), 0);
    encoder.dispatch_threads(
        thread_group(u64::from(ROWS) * u64::from(WIDTH)),
        thread_group(pipelines.embed.max_total_threads_per_threadgroup().min(256)),
    );
    encoder.end_encoding();

    let mut input: &BufferRef = &buffers.input;
    for pass in 0..FEEDBACK_PASSES {
        for layer in 0..MODEL_LAYERS {
            let norm_offset = half_bytes(u64::from(layer) * u64::from(WIDTH));
            let qkv_offset = half_bytes(u64::from(layer) * u64::from(WIDTH) * u64::from(QKV_WIDTH));
            let output_offset = half_bytes(u64::from(layer) * u64::from(WIDTH) * u64::from(WIDTH));
            rms_at(
                command,
                pipelines,
                input,
                weights.buffer,
                weights.attention_norm + norm_offset,
                &buffers.normalized,
            );
            tensor_at(
                command,
                &tensorops.qkv,
                &[
                    (&buffers.normalized, 0),
                    (weights.buffer, weights.qkv + qkv_offset),
                    (&buffers.qkv, 0),
                ],
                MTLSize {
                    width: u64::from(QKV_WIDTH / 64),
                    height: u64::from(ROWS / 128),
                    depth: 1,
                },
            )?;
            pisa1.encode_layer(command, &buffers.qkv);
            tensor_at(
                command,
                &tensorops.output_projection,
                &[
                    (pisa1.output(), 0),
                    (weights.buffer, weights.output + output_offset),
                    (&buffers.projected, 0),
                ],
                MTLSize {
                    width: u64::from(WIDTH / 64),
                    height: u64::from(ROWS / 128),
                    depth: 1,
                },
            )?;
            encode_residual(
                command,
                pipelines,
                input,
                &buffers.projected,
                &buffers.attention_state,
            );
            rms_at(
                command,
                pipelines,
                &buffers.attention_state,
                weights.buffer,
                weights.ffn_norm + norm_offset,
                &buffers.normalized,
            );
            candidate_ffn(
                command,
                pipelines,
                &buffers.normalized,
                &buffers.attention_state,
                layer,
                buffers,
                weights,
                None,
            )?;
            input = &buffers.output;
        }
        if pass + 1 < FEEDBACK_PASSES {
            encode_rms(
                command,
                pipelines,
                input,
                &buffers.unit_norm,
                &buffers.normalized,
            );
            tensor_at(
                command,
                &tensorops.output_projection,
                &[
                    (input, 0),
                    (weights.buffer, weights.feedback_state),
                    (&buffers.feedback_state, 0),
                ],
                MTLSize {
                    width: u64::from(WIDTH / 64),
                    height: u64::from(ROWS / 128),
                    depth: 1,
                },
            )?;
            tensor_at(
                command,
                &tensorops.output_projection,
                &[
                    (&buffers.normalized, 0),
                    (weights.buffer, weights.feedback_gate),
                    (&buffers.feedback_gate, 0),
                ],
                MTLSize {
                    width: u64::from(WIDTH / 64),
                    height: u64::from(ROWS / 128),
                    depth: 1,
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
            input = &buffers.feedback;
        }
    }
    rms_at(
        command,
        pipelines,
        input,
        weights.buffer,
        weights.final_norm,
        &buffers.normalized,
    );
    tensor_at(
        command,
        &tensorops.readout,
        &[
            (&buffers.normalized, 0),
            (weights.buffer, weights.readout),
            (buffers.logits()?, 0),
        ],
        MTLSize {
            width: u64::from(VOCAB / 64),
            height: u64::from(ROWS / 128),
            depth: 1,
        },
    )?;
    candidate_losses(command, pipelines, buffers)?;
    Ok(())
}

fn candidate_losses(
    command: &CommandBufferRef,
    pipelines: &Pipelines,
    buffers: &Buffers,
) -> Result<(), String> {
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipelines.cross_entropy);
    encoder.set_buffer(0, Some(buffers.logits()?), 0);
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
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipelines.sequence_loss);
    encoder.set_buffer(0, Some(&buffers.losses), 0);
    encoder.set_buffer(1, Some(&buffers.score_mask), 0);
    encoder.set_buffer(2, Some(&buffers.sequence_scores), 0);
    encoder.dispatch_thread_groups(
        thread_group(u64::from(BATCH)),
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();
    Ok(())
}
