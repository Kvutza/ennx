//! Encodable feedback primitive; no command submission or host readback.

use std::sync::Arc;

use metal::{
    Buffer, BufferRef, CommandBufferRef, ComputePipelineState, MTLCommandBufferStatus, MTLSize,
};

use crate::apple_gpu::{Runtime, thread_group};
use crate::fbt::{FeedForwardConfig, FeedForwardLayout, FeedbackConfig, FeedbackLayout};

const SOURCE: &str = include_str!("fbt.metal");

#[repr(C)]
struct LinearParams {
    input: u32,
    output: u32,
    sigmoid: u32,
    rows: u32,
}

#[repr(C)]
struct GemmParams {
    input: u32,
    output: u32,
    sigmoid: u32,
    rows: u32,
    mode: u32,
    start: u32,
    scale: f32,
    padding: u32,
}

pub(crate) enum GemmEpilogue<'a> {
    Projection,
    Gated(&'a BufferRef),
    Residual(f32),
    Loss { targets: &'a BufferRef, start: u32 },
}

#[repr(C)]
struct GroupedParams {
    input: u32,
    output: [u32; 3],
    sigmoid: [u32; 3],
}

/// BF16 row-major [output, input] weights and FP32 activations. No converted
/// weight cache: changes made between completed evaluations are immediately used.
pub struct Linear {
    runtime: Arc<Runtime>,
    params: LinearParams,
    weight_bytes: u64,
    pipeline: ComputePipelineState,
    grouped: ComputePipelineState,
    tiled: ComputePipelineState,
    gemm: ComputePipelineState,
}

impl Linear {
    pub fn new(
        input: u32,
        output: u32,
        activation: crate::fbt::ProjectionActivation,
    ) -> Result<Self, String> {
        if input == 0 || output == 0 {
            return Err("FBT projection dimensions must be nonzero".into());
        }
        let weight_bytes = u64::from(input)
            .checked_mul(u64::from(output))
            .and_then(|n| n.checked_mul(2))
            .ok_or("FBT projection size overflow")?;
        let runtime = Runtime::shared()?;
        if weight_bytes > runtime.device.max_buffer_length() {
            return Err("FBT projection exceeds Metal buffer limit".into());
        }
        let pipeline = pipeline(&runtime, "fbt_linear")?;
        let grouped = self::pipeline(&runtime, "fbt_linear_grouped")?;
        let tiled = self::pipeline(&runtime, "fbt_linear_tiled")?;
        let gemm = self::pipeline(&runtime, "fbt_gemm")?;
        if gemm.max_total_threads_per_threadgroup() < 128
            || gemm.static_threadgroup_memory_length()
                > runtime.device.max_threadgroup_memory_length()
        {
            return Err("FBT GEMM exceeds device threadgroup capabilities".into());
        }
        if tiled.max_total_threads_per_threadgroup() < 128 {
            return Err("FBT tiled projection requires 128-thread groups".into());
        }
        Ok(Self {
            runtime,
            params: LinearParams {
                input,
                output,
                sigmoid: u32::from(activation == crate::fbt::ProjectionActivation::Sigmoid),
                rows: 0,
            },
            weight_bytes,
            pipeline,
            grouped,
            tiled,
            gemm,
        })
    }

    /// Encodes live-weight GEMM with an output epilogue and no host synchronization.
    pub(crate) fn encode_fused(
        &self,
        command: &CommandBufferRef,
        rows: u32,
        weights: &BufferRef,
        input: &BufferRef,
        output: &BufferRef,
        epilogue: GemmEpilogue<'_>,
    ) -> Result<(), String> {
        self.fused_offset(command, rows, weights, input, 0, output, epilogue)
    }

    pub(crate) fn fused_offset(
        &self,
        command: &CommandBufferRef,
        rows: u32,
        weights: &BufferRef,
        input: &BufferRef,
        input_offset: u64,
        output: &BufferRef,
        epilogue: GemmEpilogue<'_>,
    ) -> Result<(), String> {
        if rows == 0 {
            return Err("FBT GEMM rows must be nonzero".into());
        }
        check_command(command)?;
        let bytes = |width: u32| {
            (u64::from(rows) * u64::from(width))
                .checked_mul(4)
                .ok_or("FBT GEMM activation size overflow")
        };
        let mut params = GemmParams {
            input: self.params.input,
            output: self.params.output,
            sigmoid: self.params.sigmoid,
            rows,
            mode: 0,
            start: 0,
            scale: 1.0,
            padding: 0,
        };
        let mut checked = vec![
            (weights, self.weight_bytes),
            (input, input_offset + bytes(self.params.input)?),
        ];
        let mut aux = input;
        let mut targets = input;
        let output_bytes = match epilogue {
            GemmEpilogue::Projection => bytes(self.params.output)?,
            GemmEpilogue::Gated(gate) => {
                params.mode = 1;
                aux = gate;
                checked.push((gate, bytes(self.params.output)?));
                bytes(self.params.output)?
            }
            GemmEpilogue::Residual(scale) => {
                if !scale.is_finite() {
                    return Err("FBT residual scale must be finite".into());
                }
                params.mode = 2;
                params.scale = scale;
                bytes(self.params.output)?
            }
            GemmEpilogue::Loss {
                targets: labels,
                start,
            } => {
                params.mode = 3;
                params.start = start;
                targets = labels;
                let end = start.checked_add(rows).ok_or("FBT label range overflow")?;
                checked.push((labels, u64::from(end) * 4));
                u64::from(rows) * u64::from(self.params.output).div_ceil(32) * 16
            }
        };
        checked.push((output, output_bytes));
        check_buffers(&self.runtime, &checked)?;
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.gemm);
        for (index, buffer, offset) in [
            (0, weights, 0),
            (1, input, input_offset),
            (2, output, 0),
            (4, aux, 0),
            (5, targets, 0),
        ] {
            encoder.set_buffer(index, Some(buffer), offset);
        }
        encoder.set_bytes(
            3,
            size_of::<GemmParams>() as u64,
            (&params as *const GemmParams).cast(),
        );
        encoder.dispatch_thread_groups(
            MTLSize {
                width: u64::from(self.params.output).div_ceil(32),
                height: u64::from(rows).div_ceil(64),
                depth: 1,
            },
            thread_group(128),
        );
        encoder.end_encoding();
        Ok(())
    }

    /// Two or three independent projections of the same input in one dispatch.
    /// Weights remain in their original allocations; no packing or BO-stale copy.
    /// All input, weight and output allocations must be distinct.
    pub fn encode_grouped(
        command: &CommandBufferRef,
        rows: u32,
        input: &BufferRef,
        projections: &[(&Self, &BufferRef, &BufferRef)],
    ) -> Result<(), String> {
        if rows == 0 || !(2..=3).contains(&projections.len()) {
            return Err(
                "FBT grouped projection requires nonzero rows and 2 or 3 projections".into(),
            );
        }
        check_command(command)?;
        let first = projections[0].0;
        let mut params = GroupedParams {
            input: first.params.input,
            output: [0; 3],
            sigmoid: [0; 3],
        };
        let bytes = |width: u32| {
            u64::from(rows)
                .checked_mul(u64::from(width))
                .and_then(|n| n.checked_mul(4))
                .ok_or("FBT grouped activation size overflow")
        };
        let mut checked = vec![(input, bytes(params.input)?)];
        let mut total = 0u32;
        for (i, &(linear, weights, output)) in projections.iter().enumerate() {
            if linear.params.input != params.input {
                return Err("FBT grouped projections require the same input width".into());
            }
            total = total
                .checked_add(linear.params.output)
                .ok_or("FBT grouped output size overflow")?;
            params.output[i] = linear.params.output;
            params.sigmoid[i] = linear.params.sigmoid;
            checked.push((weights, linear.weight_bytes));
            checked.push((output, bytes(linear.params.output)?));
        }
        check_buffers(&first.runtime, &checked)?;
        let third = projections.get(2).unwrap_or(&projections[1]);
        dispatch(
            command,
            &first.grouped,
            &[
                projections[0].1,
                projections[1].1,
                third.1,
                input,
                projections[0].2,
                projections[1].2,
                third.2,
            ],
            &params,
            MTLSize {
                width: u64::from(total),
                height: u64::from(rows),
                depth: 1,
            },
        );
        Ok(())
    }

    /// Encode only. Buffers must be distinct, finite and unmodified until GPU
    /// completion. The caller owns submission, ordering and completion checks.
    pub fn encode(
        &self,
        command: &CommandBufferRef,
        rows: u32,
        weights: &BufferRef,
        input: &BufferRef,
        output: &BufferRef,
    ) -> Result<(), String> {
        self.encode_impl(command, rows, weights, input, output, false)
    }

    /// Explicit tiled prefill path. Same storage and ownership contract as encode;
    /// no packed weight copy, scratch allocation, submission or host readback.
    pub fn encode_tiled(
        &self,
        command: &CommandBufferRef,
        rows: u32,
        weights: &BufferRef,
        input: &BufferRef,
        output: &BufferRef,
    ) -> Result<(), String> {
        self.encode_impl(command, rows, weights, input, output, true)
    }

    fn encode_impl(
        &self,
        command: &CommandBufferRef,
        rows: u32,
        weights: &BufferRef,
        input: &BufferRef,
        output: &BufferRef,
        tiled: bool,
    ) -> Result<(), String> {
        if rows == 0 {
            return Err("FBT projection rows must be nonzero".into());
        }
        check_command(command)?;
        let bytes = |width: u32| {
            u64::from(rows)
                .checked_mul(u64::from(width))
                .and_then(|n| n.checked_mul(4))
                .ok_or("FBT projection activation size overflow")
        };
        check_buffers(
            &self.runtime,
            &[
                (weights, self.weight_bytes),
                (input, bytes(self.params.input)?),
                (output, bytes(self.params.output)?),
            ],
        )?;
        let params = LinearParams {
            rows,
            ..self.params
        };
        if tiled {
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.tiled);
            for (index, buffer) in [weights, input, output].into_iter().enumerate() {
                encoder.set_buffer(index as u64, Some(buffer), 0);
            }
            encoder.set_bytes(
                3,
                size_of::<LinearParams>() as u64,
                (&params as *const LinearParams).cast(),
            );
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: u64::from(self.params.output).div_ceil(32),
                    height: u64::from(rows).div_ceil(64),
                    depth: 1,
                },
                thread_group(128),
            );
            encoder.end_encoding();
        } else {
            dispatch(
                command,
                &self.pipeline,
                &[weights, input, output],
                &params,
                MTLSize {
                    width: u64::from(self.params.output),
                    height: u64::from(rows),
                    depth: 1,
                },
            );
        }
        Ok(())
    }
}

pub(crate) fn check_command(command: &CommandBufferRef) -> Result<(), String> {
    if !matches!(
        command.status(),
        MTLCommandBufferStatus::NotEnqueued | MTLCommandBufferStatus::Enqueued
    ) {
        return Err("FBT cannot encode into a submitted command buffer".into());
    }
    Ok(())
}

pub(crate) fn check_buffers(
    runtime: &Runtime,
    buffers: &[(&BufferRef, u64)],
) -> Result<(), String> {
    for (i, &(buffer, size)) in buffers.iter().enumerate() {
        if buffer.length() < size {
            return Err(format!(
                "FBT buffer {i} has {} bytes, needs {size}",
                buffer.length()
            ));
        }
        if buffer.device().registry_id() != runtime.device.registry_id() {
            return Err("FBT buffer is on a different Metal device".into());
        }
        if buffers[..i]
            .iter()
            .any(|(other, _)| std::ptr::eq(*other, buffer))
        {
            return Err("FBT buffers must not alias".into());
        }
    }
    Ok(())
}

pub(crate) fn check_memory(runtime: &Runtime, total: u64, largest: u64) -> Result<(), String> {
    if largest > runtime.device.max_buffer_length()
        || runtime
            .device
            .current_allocated_size()
            .checked_add(total)
            .is_none_or(|n| n > runtime.device.recommended_max_working_set_size())
    {
        return Err("FBT scratch exceeds Metal memory budget".into());
    }
    Ok(())
}

fn pipeline(runtime: &Runtime, name: &str) -> Result<ComputePipelineState, String> {
    let pipeline = runtime.precise(SOURCE, "FBT", name)?;
    if pipeline.thread_execution_width() != 32 || pipeline.max_total_threads_per_threadgroup() < 32
    {
        return Err("FBT primitives require 32-lane SIMD groups".into());
    }
    Ok(pipeline)
}

pub(crate) fn dispatch<T>(
    command: &CommandBufferRef,
    pipeline: &ComputePipelineState,
    inputs: &[&BufferRef],
    params: &T,
    groups: MTLSize,
) {
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    for (index, buffer) in inputs.iter().enumerate() {
        encoder.set_buffer(index as u64, Some(buffer), 0);
    }
    encoder.set_bytes(
        inputs.len() as u64,
        size_of::<T>() as u64,
        (params as *const T).cast(),
    );
    encoder.dispatch_thread_groups(groups, thread_group(32));
    encoder.end_encoding();
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Params {
    width: u32,
    rows: u32,
    token_epsilon: f32,
    fused_epsilon: f32,
}

/// Persistent scratch and pipelines for a fixed-width feedback operation.
///
/// `encode` borrows current weight buffers rather than caching converted weights,
/// so BO updates are visible on the next dispatch. Calls sharing this workspace
/// must execute serially. The caller owns submission and completion handling.
pub struct Feedback {
    runtime: Arc<Runtime>,
    layout: FeedbackLayout,
    capacity: u32,
    params: Params,
    scales: Buffer,
    scratch: Buffer,
    token_scale: ComputePipelineState,
    project: ComputePipelineState,
    fused_norm: ComputePipelineState,
}

impl Feedback {
    pub fn new(config: FeedbackConfig, capacity: u32) -> Result<Self, String> {
        let layout = config.validate()?;
        if capacity == 0 {
            return Err("FBT row capacity must be nonzero".into());
        }
        let elements = (capacity as usize)
            .checked_mul(config.width as usize)
            .ok_or("FBT scratch size overflow")?;
        let bytes = elements
            .checked_mul(4)
            .ok_or("FBT scratch byte size overflow")? as u64;
        let total = bytes
            .checked_add(u64::from(capacity) * 4)
            .ok_or("FBT memory overflow")?;
        let runtime = Runtime::shared()?;
        check_memory(&runtime, total, bytes.max(u64::from(capacity) * 4))?;
        let token_scale = pipeline(&runtime, "fbt_token_scale")?;
        let project = pipeline(&runtime, "fbt_project")?;
        let fused_norm = pipeline(&runtime, "fbt_fused_norm")?;
        let scales = runtime.buffer::<f32>(capacity as usize);
        let scratch = runtime.buffer::<f32>(elements);
        if scales.contents().is_null() || scratch.contents().is_null() {
            return Err("FBT scratch allocation failed".into());
        }
        Ok(Self {
            runtime,
            layout,
            capacity,
            params: Params {
                width: config.width,
                rows: 0,
                token_epsilon: config.token_norm.epsilon()?,
                fused_epsilon: config.fused_norm.epsilon()?,
            },
            scales,
            scratch,
            token_scale,
            project,
            fused_norm,
        })
    }

    pub fn layout(&self) -> FeedbackLayout {
        self.layout
    }

    /// Encode three ordered GPU dispatches, without submitting or synchronizing.
    ///
    /// Weights are BF16 in `layout()` order. States, tokens and output are FP32
    /// `[rows, width]`; the u32 mask is zero for plain inputs, nonzero for fusion.
    /// Plain rows copy token inputs exactly and do not consume previous states.
    /// Previous states must already be aligned to their consuming token rows;
    /// this primitive does not shift positions or manage per-sequence state.
    /// Buffers must belong to the same device, have distinct allocations, and
    /// contain finite values in used coordinates. Buffer views with offsets are
    /// not supported. The caller must not mutate buffers while GPU work uses them,
    /// and must serialize all commands using this workspace on one command queue
    /// belonging to the same device.
    /// Read output only after successful command completion.
    pub fn encode(
        &mut self,
        command: &CommandBufferRef,
        rows: u32,
        weights: &BufferRef,
        previous: &BufferRef,
        tokens: &BufferRef,
        fused: &BufferRef,
        output: &BufferRef,
    ) -> Result<(), String> {
        if rows == 0 || rows > self.capacity {
            return Err("FBT row count is outside workspace capacity".into());
        }
        check_command(command)?;
        let state_bytes = u64::from(rows) * u64::from(self.layout.width) * 4;
        check_buffers(
            &self.runtime,
            &[
                (weights, self.layout.elements as u64 * 2),
                (previous, state_bytes),
                (tokens, state_bytes),
                (fused, u64::from(rows) * 4),
                (output, state_bytes),
            ],
        )?;
        let params = Params {
            rows,
            ..self.params
        };
        dispatch(
            command,
            &self.token_scale,
            &[tokens, &self.scales],
            &params,
            thread_group(u64::from(rows)),
        );
        dispatch(
            command,
            &self.project,
            &[
                weights,
                previous,
                tokens,
                fused,
                &self.scales,
                &self.scratch,
            ],
            &params,
            MTLSize {
                width: u64::from(self.layout.width),
                height: u64::from(rows),
                depth: 1,
            },
        );
        dispatch(
            command,
            &self.fused_norm,
            &[&self.scratch, fused, output],
            &params,
            thread_group(u64::from(rows)),
        );
        Ok(())
    }
}

#[repr(C)]
struct FfnParams {
    width: u32,
    intermediate: u32,
    residual_scale: f32,
}

/// Native SiLU-GLU plus residual addition, with no implicit input normalization.
/// The graph owner supplies the normalized input and the original residual.
pub struct FeedForward {
    runtime: Arc<Runtime>,
    layout: FeedForwardLayout,
    capacity: u32,
    params: FfnParams,
    activation: Buffer,
    up: ComputePipelineState,
    down: ComputePipelineState,
}

impl FeedForward {
    pub fn new(config: FeedForwardConfig, capacity: u32) -> Result<Self, String> {
        let layout = config.validate()?;
        if capacity == 0 {
            return Err("FBT row capacity must be nonzero".into());
        }
        let elements = (capacity as usize)
            .checked_mul(config.intermediate as usize)
            .ok_or("FBT activation size overflow")?;
        let bytes = elements
            .checked_mul(4)
            .ok_or("FBT activation byte size overflow")? as u64;
        let runtime = Runtime::shared()?;
        check_memory(&runtime, bytes, bytes)?;
        let up = pipeline(&runtime, "fbt_ffn_up")?;
        let down = pipeline(&runtime, "fbt_ffn_down_residual")?;
        let activation = runtime.buffer::<f32>(elements);
        if activation.contents().is_null() {
            return Err("FBT activation allocation failed".into());
        }
        Ok(Self {
            runtime,
            layout,
            capacity,
            params: FfnParams {
                width: config.width,
                intermediate: config.intermediate,
                residual_scale: config.residual_scale,
            },
            activation,
            up,
            down,
        })
    }

    pub fn layout(&self) -> FeedForwardLayout {
        self.layout
    }

    /// Encode `residual + scale * down(silu(gate(input)) * up(input))`.
    /// BF16 weights follow `layout()`; all other buffers are FP32 [rows, width].
    /// No submission, readback, or synchronization is performed. The distinct,
    /// same-device buffer and serial-workspace ownership rules of Feedback::encode
    /// also apply here. Finite inputs/weights and successful completion are the
    /// caller's responsibility. This primitive neither applies nor learns norms.
    pub fn encode(
        &mut self,
        command: &CommandBufferRef,
        rows: u32,
        weights: &BufferRef,
        input: &BufferRef,
        residual: &BufferRef,
        output: &BufferRef,
    ) -> Result<(), String> {
        if rows == 0 || rows > self.capacity {
            return Err("FBT row count is outside workspace capacity".into());
        }
        check_command(command)?;
        let bytes = u64::from(rows)
            .checked_mul(u64::from(self.layout.width))
            .and_then(|n| n.checked_mul(4))
            .ok_or("FBT input size overflow")?;
        check_buffers(
            &self.runtime,
            &[
                (weights, self.layout.elements as u64 * 2),
                (input, bytes),
                (residual, bytes),
                (output, bytes),
            ],
        )?;
        dispatch(
            command,
            &self.up,
            &[weights, input, &self.activation],
            &self.params,
            MTLSize {
                width: u64::from(self.layout.intermediate),
                height: u64::from(rows),
                depth: 1,
            },
        );
        dispatch(
            command,
            &self.down,
            &[weights, &self.activation, residual, output],
            &self.params,
            MTLSize {
                width: u64::from(self.layout.width),
                height: u64::from(rows),
                depth: 1,
            },
        );
        Ok(())
    }
}

#[cfg(test)]
#[path = "fbt_metaltests.rs"]
mod tests;
