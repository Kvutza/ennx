//! Deterministic token-choice routing and exact dropless packing for the
//! fine-grained MoE architecture. The width-216 expert consumer is a separate
//! integration boundary from this route/pack stage.

#![allow(dead_code)]

use super::{EXPERT_WIDTH, ROWS, WIDTH, dispatch};
use crate::apple_gpu::{Runtime, thread_group};
use metal::{Buffer, BufferRef, CommandBufferRef, ComputePipelineState, MTLSize};

pub(super) const SHARED_EXPERTS: u32 = 1;
pub(super) const SHARED_EXPERT_WIDTH: u32 = 216;
pub(super) const ROUTED_EXPERTS: u32 = 128;
pub(super) const ROUTED_TOP_K: u32 = 3;
pub(super) const ROUTED_EXPERT_WIDTH: u32 = 216;
pub(super) const MOE_EXPERTS: u32 = SHARED_EXPERTS + ROUTED_EXPERTS;
pub(super) const MOE_GATE_UP: u32 = 2 * ROUTED_EXPERT_WIDTH;
const ROUTE_BLOCK_TOKENS: u32 = 64;
const ROUTED_ACTIVE_WIDTH: u32 = SHARED_EXPERT_WIDTH + ROUTED_TOP_K * ROUTED_EXPERT_WIDTH;
pub(super) const ROUTED_GATE_UP_PARAMETERS_PER_LAYER: usize =
    MOE_EXPERTS as usize * 2 * ROUTED_EXPERT_WIDTH as usize * WIDTH as usize;
pub(super) const ROUTED_DOWN_PARAMETERS_PER_LAYER: usize =
    MOE_EXPERTS as usize * ROUTED_EXPERT_WIDTH as usize * WIDTH as usize;
pub(super) const ROUTED_FFN_PARAMETERS_PER_LAYER: usize =
    ROUTED_GATE_UP_PARAMETERS_PER_LAYER + ROUTED_DOWN_PARAMETERS_PER_LAYER;
const _: () = assert!(ROUTED_ACTIVE_WIDTH == EXPERT_WIDTH);

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct Top3RouteShape {
    rows: u32,
    width: u32,
    experts: u32,
    top_k: u32,
    block_tokens: u32,
    blocks: u32,
}

impl Top3RouteShape {
    const fn production() -> Self {
        Self::new(ROWS, WIDTH)
    }

    const fn new(rows: u32, width: u32) -> Self {
        Self {
            rows,
            width,
            experts: ROUTED_EXPERTS,
            top_k: ROUTED_TOP_K,
            block_tokens: ROUTE_BLOCK_TOKENS,
            blocks: rows.div_ceil(ROUTE_BLOCK_TOKENS),
        }
    }

    const fn assignments(self) -> u32 {
        self.rows * self.top_k
    }
}

pub(super) struct FineGrainedMoePipelines {
    router_logits: ComputePipelineState,
    select: ComputePipelineState,
    histogram: ComputePipelineState,
    prefix: ComputePipelineState,
    rank: ComputePipelineState,
    pack: ComputePipelineState,
    routed_gate: ComputePipelineState,
    routed_down: ComputePipelineState,
    shared_gate: ComputePipelineState,
    shared_down: ComputePipelineState,
    combine: ComputePipelineState,
}

impl FineGrainedMoePipelines {
    pub(super) fn new(runtime: &Runtime) -> Result<Self, String> {
        let source = include_str!("fbt_moe_routing.metal");
        let utility = |name| runtime.precise(source, "deterministic top-3 MoE routing", name);
        let tensorops_source = include_str!("fbt_moe_routing_tensorops.metal");
        let tensorops =
            |description, name| runtime.precise_metal4(tensorops_source, description, name, &[]);
        Ok(Self {
            router_logits: tensorops(
                "Metal 4 fine-grained MoE router projection",
                "fbt_moe_router_logits",
            )?,
            select: utility("fbt_moe_select_top3")?,
            histogram: utility("fbt_moe_route_histogram")?,
            prefix: utility("fbt_moe_route_prefix")?,
            rank: utility("fbt_moe_route_rank")?,
            pack: utility("fbt_moe_route_pack")?,
            routed_gate: tensorops(
                "Metal 4 variable-load routed MoE gate/up",
                "fbt_moe_routed_gate",
            )?,
            routed_down: tensorops(
                "Metal 4 variable-load routed MoE down",
                "fbt_moe_routed_down",
            )?,
            shared_gate: tensorops("Metal 4 shared MoE gate/up", "fbt_moe_shared_gate")?,
            shared_down: tensorops("Metal 4 shared MoE down", "fbt_moe_shared_down")?,
            combine: utility("fbt_moe_combine")?,
        })
    }
}

/// Exact-size buffers for three assignments per token. Expert segments are
/// represented by offsets and loads, not padded per-expert capacities.
pub(super) struct FineGrainedMoeBuffers {
    scores: Buffer,
    route_experts: Buffer,
    route_weights: Buffer,
    route_margin: Buffer,
    block_histogram: Buffer,
    block_offsets: Buffer,
    expert_offsets: Buffer,
    expert_loads: Buffer,
    packed_rows: Buffer,
    packed_input: Buffer,
    packed_tokens: Buffer,
    packed_experts: Buffer,
    packed_weights: Buffer,
    routed_activation: Buffer,
    routed_output: Buffer,
    shared_activation: Buffer,
    shared_output: Buffer,
}

impl FineGrainedMoeBuffers {
    pub(super) fn new(runtime: &Runtime) -> Self {
        Self::with_shape(runtime, Top3RouteShape::production())
    }

    fn with_shape(runtime: &Runtime, shape: Top3RouteShape) -> Self {
        let rows = shape.rows as usize;
        let width = shape.width as usize;
        let assignments = shape.assignments() as usize;
        let experts = shape.experts as usize;
        let blocks = shape.blocks as usize;
        Self {
            scores: runtime.buffer::<u16>(rows * experts),
            route_experts: runtime.buffer::<u32>(assignments),
            route_weights: runtime.buffer::<u16>(assignments),
            route_margin: runtime.buffer::<f32>(rows),
            block_histogram: runtime.buffer::<u32>(blocks * experts),
            block_offsets: runtime.buffer::<u32>(blocks * experts),
            expert_offsets: runtime.buffer::<u32>(experts),
            expert_loads: runtime.buffer::<u32>(experts),
            packed_rows: runtime.buffer::<u32>(assignments),
            packed_input: runtime.buffer::<u16>(assignments * width),
            packed_tokens: runtime.buffer::<u32>(assignments),
            packed_experts: runtime.buffer::<u32>(assignments),
            packed_weights: runtime.buffer::<u16>(assignments),
            routed_activation: runtime.buffer::<u16>(assignments * ROUTED_EXPERT_WIDTH as usize),
            routed_output: runtime.buffer::<u16>(assignments * width),
            shared_activation: runtime.buffer::<u16>(rows * SHARED_EXPERT_WIDTH as usize),
            shared_output: runtime.buffer::<u16>(rows * width),
        }
    }
}

/// Encode token-choice top-3 selection and stable variable-count packing. The
/// consumer receives contiguous expert segments plus exact offsets and counts.
fn encode_top3_routing(
    command: &CommandBufferRef,
    pipelines: &FineGrainedMoePipelines,
    buffers: &FineGrainedMoeBuffers,
    input: &BufferRef,
    input_offset: u64,
    router: &BufferRef,
    router_offset: u64,
    shape: Top3RouteShape,
) -> Result<(), String> {
    let shape_ptr = (&shape as *const Top3RouteShape).cast();
    let shape_bytes = std::mem::size_of::<Top3RouteShape>() as u64;
    let simd_width = pipelines.router_logits.thread_execution_width();
    let tensorops_threads = simd_width * 4;
    if simd_width != 32
        || pipelines.router_logits.max_total_threads_per_threadgroup() < tensorops_threads
    {
        return Err(format!(
            "router TensorOps kernel requires four 32-lane SIMD groups; pipeline reports SIMD width {simd_width} and max threads {}",
            pipelines.router_logits.max_total_threads_per_threadgroup()
        ));
    }
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipelines.router_logits);
    encoder.set_buffer(0, Some(input), input_offset);
    encoder.set_buffer(1, Some(router), router_offset);
    encoder.set_buffer(2, Some(&buffers.scores), 0);
    encoder.set_bytes(3, shape_bytes, shape_ptr);
    encoder.dispatch_thread_groups(
        MTLSize {
            width: u64::from(shape.experts.div_ceil(64)),
            height: u64::from(shape.rows.div_ceil(128)),
            depth: 1,
        },
        MTLSize {
            width: tensorops_threads,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();

    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipelines.select);
    for (index, buffer) in [
        &buffers.scores,
        &buffers.route_experts,
        &buffers.route_weights,
        &buffers.route_margin,
    ]
    .into_iter()
    .enumerate()
    {
        encoder.set_buffer(index as u64, Some(buffer), 0);
    }
    encoder.set_bytes(4, shape_bytes, shape_ptr);
    encoder.dispatch_thread_groups(
        MTLSize {
            width: 1,
            height: u64::from(shape.rows),
            depth: 1,
        },
        MTLSize {
            width: u64::from(shape.experts),
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();

    dispatch(
        command,
        &pipelines.histogram,
        &[&buffers.route_experts, &buffers.block_histogram],
        shape_ptr,
        shape_bytes,
        u64::from(shape.blocks * shape.experts),
    );
    dispatch(
        command,
        &pipelines.prefix,
        &[
            &buffers.block_histogram,
            &buffers.block_offsets,
            &buffers.expert_offsets,
            &buffers.expert_loads,
        ],
        shape_ptr,
        shape_bytes,
        u64::from(shape.experts),
    );
    dispatch(
        command,
        &pipelines.rank,
        &[
            &buffers.route_experts,
            &buffers.block_offsets,
            &buffers.packed_rows,
        ],
        shape_ptr,
        shape_bytes,
        u64::from(shape.assignments()),
    );
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipelines.pack);
    for (index, (buffer, offset)) in [
        (input, input_offset),
        (&buffers.route_experts, 0),
        (&buffers.route_weights, 0),
        (&buffers.packed_rows, 0),
        (&buffers.packed_input, 0),
        (&buffers.packed_tokens, 0),
        (&buffers.packed_experts, 0),
        (&buffers.packed_weights, 0),
    ]
    .into_iter()
    .enumerate()
    {
        encoder.set_buffer(index as u64, Some(buffer), offset);
    }
    encoder.set_bytes(8, shape_bytes, shape_ptr);
    let width = pipelines.pack.max_total_threads_per_threadgroup().min(256);
    encoder.dispatch_threads(
        thread_group(u64::from(shape.assignments()) * u64::from(shape.width)),
        thread_group(width),
    );
    encoder.end_encoding();
    Ok(())
}

impl FineGrainedMoePipelines {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode(
        &self,
        command: &CommandBufferRef,
        buffers: &FineGrainedMoeBuffers,
        input: &BufferRef,
        input_offset: u64,
        weights: &BufferRef,
        router_offset: u64,
        gate_up_offset: u64,
        down_offset: u64,
        output: &BufferRef,
    ) -> Result<(), String> {
        self.encode_shape(
            command,
            buffers,
            input,
            input_offset,
            weights,
            router_offset,
            gate_up_offset,
            down_offset,
            output,
            Top3RouteShape::production(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_shape(
        &self,
        command: &CommandBufferRef,
        buffers: &FineGrainedMoeBuffers,
        input: &BufferRef,
        input_offset: u64,
        weights: &BufferRef,
        router_offset: u64,
        gate_up_offset: u64,
        down_offset: u64,
        output: &BufferRef,
        shape: Top3RouteShape,
    ) -> Result<(), String> {
        encode_top3_routing(
            command,
            self,
            buffers,
            input,
            input_offset,
            weights,
            router_offset,
            shape,
        )?;
        let shape_ptr = (&shape as *const Top3RouteShape).cast();
        let shape_bytes = std::mem::size_of::<Top3RouteShape>() as u64;
        let max_routed_tiles = shape.assignments().div_ceil(128) + shape.experts - 1;
        encode_tensorops(
            command,
            &self.routed_gate,
            &[
                (&buffers.packed_input, 0),
                (weights, gate_up_offset),
                (&buffers.expert_offsets, 0),
                (&buffers.expert_loads, 0),
                (&buffers.routed_activation, 0),
            ],
            shape_ptr,
            shape_bytes,
            MTLSize {
                width: u64::from(ROUTED_EXPERT_WIDTH.div_ceil(64)),
                height: u64::from(max_routed_tiles),
                depth: 1,
            },
        )?;
        encode_tensorops(
            command,
            &self.routed_down,
            &[
                (&buffers.routed_activation, 0),
                (weights, down_offset),
                (&buffers.expert_offsets, 0),
                (&buffers.expert_loads, 0),
                (&buffers.routed_output, 0),
            ],
            shape_ptr,
            shape_bytes,
            MTLSize {
                width: u64::from(shape.width.div_ceil(64)),
                height: u64::from(max_routed_tiles),
                depth: 1,
            },
        )?;
        encode_tensorops(
            command,
            &self.shared_gate,
            &[
                (input, input_offset),
                (weights, gate_up_offset),
                (&buffers.shared_activation, 0),
            ],
            shape_ptr,
            shape_bytes,
            MTLSize {
                width: u64::from(SHARED_EXPERT_WIDTH.div_ceil(64)),
                height: u64::from(shape.rows.div_ceil(128)),
                depth: 1,
            },
        )?;
        encode_tensorops(
            command,
            &self.shared_down,
            &[
                (&buffers.shared_activation, 0),
                (weights, down_offset),
                (&buffers.shared_output, 0),
            ],
            shape_ptr,
            shape_bytes,
            MTLSize {
                width: u64::from(shape.width.div_ceil(64)),
                height: u64::from(shape.rows.div_ceil(128)),
                depth: 1,
            },
        )?;
        dispatch(
            command,
            &self.combine,
            &[
                &buffers.route_weights,
                &buffers.packed_rows,
                &buffers.routed_output,
                &buffers.shared_output,
                output,
            ],
            shape_ptr,
            shape_bytes,
            u64::from(shape.rows) * u64::from(shape.width),
        );
        Ok(())
    }
}

fn encode_tensorops(
    command: &CommandBufferRef,
    pipeline: &ComputePipelineState,
    buffers: &[(&BufferRef, u64)],
    parameters: *const std::ffi::c_void,
    parameter_bytes: u64,
    groups: MTLSize,
) -> Result<(), String> {
    let simd_width = pipeline.thread_execution_width();
    let threads = simd_width * 4;
    if simd_width != 32 || pipeline.max_total_threads_per_threadgroup() < threads {
        return Err(format!(
            "MoE TensorOps kernel requires four 32-lane SIMD groups; pipeline reports SIMD width {simd_width} and max threads {}",
            pipeline.max_total_threads_per_threadgroup()
        ));
    }
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    for (index, (buffer, offset)) in buffers.iter().enumerate() {
        encoder.set_buffer(index as u64, Some(buffer), *offset);
    }
    encoder.set_bytes(buffers.len() as u64, parameter_bytes, parameters);
    encoder.dispatch_thread_groups(
        groups,
        MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use metal::objc::rc::autoreleasepool;

    unsafe fn contents<T>(buffer: &BufferRef, elements: usize) -> &[T] {
        unsafe { std::slice::from_raw_parts(buffer.contents().cast::<T>(), elements) }
    }

    #[test]
    fn top3_route_pack_is_deterministic_dropless_and_exact() -> Result<(), String> {
        autoreleasepool(|| {
            let runtime = Runtime::shared()?;
            let shape = Top3RouteShape::new(4, WIDTH);
            let pipelines = FineGrainedMoePipelines::new(&runtime)?;
            let buffers = FineGrainedMoeBuffers::with_shape(&runtime, shape);
            let input = runtime.buffer_with(&vec![0x3c00u16; (shape.rows * shape.width) as usize]);
            let router =
                runtime.buffer_with(&vec![0x3c00u16; (shape.width * shape.experts) as usize]);
            let command = runtime.queue.new_command_buffer();
            encode_top3_routing(command, &pipelines, &buffers, &input, 0, &router, 0, shape)?;
            super::super::complete(command)?;

            let assignments = shape.assignments() as usize;
            let route_experts = unsafe { contents::<u32>(&buffers.route_experts, assignments) };
            let route_weights = unsafe { contents::<u16>(&buffers.route_weights, assignments) };
            let margins = unsafe { contents::<f32>(&buffers.route_margin, shape.rows as usize) };
            for token in 0..shape.rows as usize {
                assert_eq!(
                    &route_experts
                        [token * ROUTED_TOP_K as usize..(token + 1) * ROUTED_TOP_K as usize],
                    &[0, 1, 2]
                );
                let weights = &route_weights
                    [token * ROUTED_TOP_K as usize..(token + 1) * ROUTED_TOP_K as usize];
                assert!(weights[0] != 0 && weights.iter().all(|weight| *weight == weights[0]));
            }
            assert!(margins.iter().all(|margin| *margin == 0.0));

            let loads = unsafe { contents::<u32>(&buffers.expert_loads, shape.experts as usize) };
            assert_eq!(&loads[..3], &[shape.rows, shape.rows, shape.rows]);
            assert!(loads[3..].iter().all(|load| *load == 0));
            assert_eq!(loads.iter().sum::<u32>(), shape.assignments());

            let offsets =
                unsafe { contents::<u32>(&buffers.expert_offsets, shape.experts as usize) };
            assert_eq!(
                &offsets[..4],
                &[0, shape.rows, 2 * shape.rows, 3 * shape.rows]
            );
            assert!(offsets[3..].iter().all(|offset| *offset == 3 * shape.rows));

            let packed_rows = unsafe { contents::<u32>(&buffers.packed_rows, assignments) };
            let mut permutation = packed_rows.to_vec();
            permutation.sort_unstable();
            assert_eq!(permutation, (0..shape.assignments()).collect::<Vec<_>>());

            let packed_tokens = unsafe { contents::<u32>(&buffers.packed_tokens, assignments) };
            let packed_experts = unsafe { contents::<u32>(&buffers.packed_experts, assignments) };
            let packed_weights = unsafe { contents::<u16>(&buffers.packed_weights, assignments) };
            for expert in 0..ROUTED_TOP_K {
                let start = (expert * shape.rows) as usize;
                let end = start + shape.rows as usize;
                assert_eq!(
                    &packed_tokens[start..end],
                    &(0..shape.rows).collect::<Vec<_>>()
                );
                assert!(
                    packed_experts[start..end]
                        .iter()
                        .all(|value| *value == expert)
                );
                assert!(
                    packed_weights[start..end]
                        .iter()
                        .all(|weight| *weight == route_weights[0])
                );
            }
            let packed_input = unsafe {
                contents::<u16>(&buffers.packed_input, assignments * shape.width as usize)
            };
            assert!(packed_input.iter().all(|value| *value == 0x3c00));
            Ok(())
        })
    }

    #[test]
    fn production_contract_preserves_active_width() {
        let shape = Top3RouteShape::production();
        assert_eq!(shape.assignments(), ROWS * ROUTED_TOP_K);
        assert_eq!(ROUTED_ACTIVE_WIDTH, EXPERT_WIDTH);
        assert_eq!(ROUTED_GATE_UP_PARAMETERS_PER_LAYER, 28_532_736);
        assert_eq!(ROUTED_DOWN_PARAMETERS_PER_LAYER, 14_266_368);
        assert_eq!(ROUTED_FFN_PARAMETERS_PER_LAYER, 42_799_104);
    }

    #[test]
    fn shared_and_routed_experts_recombine_without_padding_or_atomics() -> Result<(), String> {
        autoreleasepool(|| {
            let runtime = Runtime::shared()?;
            let shape = Top3RouteShape::new(4, WIDTH);
            let pipelines = FineGrainedMoePipelines::new(&runtime)?;
            let buffers = FineGrainedMoeBuffers::with_shape(&runtime, shape);
            let input = runtime.buffer::<u16>((shape.rows * shape.width) as usize);
            let input_values = unsafe {
                std::slice::from_raw_parts_mut(
                    input.contents().cast::<u16>(),
                    (shape.rows * shape.width) as usize,
                )
            };
            input_values.fill(0);
            for row in 0..shape.rows as usize {
                input_values[row * WIDTH as usize] = 0x3c00;
            }
            let router_elements = (WIDTH * ROUTED_EXPERTS) as usize;
            let gate_elements = (MOE_EXPERTS * WIDTH * MOE_GATE_UP) as usize;
            let down_elements = (MOE_EXPERTS * ROUTED_EXPERT_WIDTH * WIDTH) as usize;
            let router = runtime.buffer_with(&vec![0x3c00u16; router_elements]);
            let gate_up = runtime.buffer::<u16>(gate_elements);
            let down = runtime.buffer::<u16>(down_elements);
            let gate_values = unsafe {
                std::slice::from_raw_parts_mut(gate_up.contents().cast::<u16>(), gate_elements)
            };
            let down_values = unsafe {
                std::slice::from_raw_parts_mut(down.contents().cast::<u16>(), down_elements)
            };
            gate_values.fill(0);
            down_values.fill(0);
            for expert in 0..=ROUTED_TOP_K as usize {
                let gate_base = expert * (WIDTH * MOE_GATE_UP) as usize;
                gate_values[gate_base] = 0x3c00;
                gate_values[gate_base + ROUTED_EXPERT_WIDTH as usize] = 0x3c00;
                let down_base = expert * (ROUTED_EXPERT_WIDTH * WIDTH) as usize;
                down_values[down_base] = 0x3c00;
            }
            let total_elements = router_elements + gate_elements + down_elements;
            let weights = runtime.buffer::<u16>(total_elements);
            let weight_values = unsafe {
                std::slice::from_raw_parts_mut(weights.contents().cast::<u16>(), total_elements)
            };
            weight_values[..router_elements]
                .copy_from_slice(unsafe { contents::<u16>(&router, router_elements) });
            weight_values[router_elements..router_elements + gate_elements]
                .copy_from_slice(gate_values);
            weight_values[router_elements + gate_elements..].copy_from_slice(down_values);
            let output = runtime.buffer::<u16>((shape.rows * shape.width) as usize);
            let command = runtime.queue.new_command_buffer();
            pipelines.encode_shape(
                command,
                &buffers,
                &input,
                0,
                &weights,
                0,
                (router_elements * 2) as u64,
                ((router_elements + gate_elements) * 2) as u64,
                &output,
                shape,
            )?;
            super::super::complete(command)?;
            let values = unsafe { contents::<u16>(&output, (shape.rows * shape.width) as usize) };
            for row in 0..shape.rows as usize {
                let first = super::super::decode_half(values[row * WIDTH as usize]) as f32;
                assert!((first - 1.462).abs() < 0.01, "row {row}: {first}");
                assert!(
                    values[row * WIDTH as usize + 1..(row + 1) * WIDTH as usize]
                        .iter()
                        .all(|value| *value == 0)
                );
            }
            Ok(())
        })
    }
}
