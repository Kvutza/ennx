//! Deterministic fine-grained MoE token routing and exact dropless packing.
//! The width-216 expert consumer remains a separate integration boundary.

use super::{EXPERT_WIDTH, ROWS, WIDTH, dispatch_on};
use crate::apple_gpu::{Runtime, thread_group};
use metal::{
    Buffer, BufferRef, CommandBufferRef, ComputeCommandEncoderRef, ComputePipelineState, MTLSize,
};

#[path = "fbt_routing/branch.rs"]
mod branch;

pub(super) const SHARED_EXPERTS: u32 = 1;
pub(super) const SHARED_WIDTH: u32 = 216;
pub(super) const ROUTED_EXPERTS: u32 = 625;
pub(super) const ROUTED_TOPK: u32 = 3;
pub(super) const ROUTED_WIDTH: u32 = 216;
const ACTIVATION_STRIDE: u32 = 224;
pub(super) const MOE_EXPERTS: u32 = SHARED_EXPERTS + ROUTED_EXPERTS;
pub(super) const MOE_GATEUP: u32 = 2 * ROUTED_WIDTH;
const ROUTE_BLOCK: u32 = 64;
const ROUTE_THREADS: u64 = 640;
const TILE_ROWS: u32 = 64;
const TILE_WORDS: usize = 4;
const DISPATCH_WORDS: usize = 6;
const ACTIVE_WIDTH: u32 = SHARED_WIDTH + ROUTED_TOPK * ROUTED_WIDTH;
pub(super) const GATE_PARAMETERS: usize =
    MOE_EXPERTS as usize * 2 * ROUTED_WIDTH as usize * WIDTH as usize;
pub(super) const DOWN_PARAMETERS: usize =
    MOE_EXPERTS as usize * ROUTED_WIDTH as usize * WIDTH as usize;
pub(super) const FFN_PARAMETERS: usize = GATE_PARAMETERS + DOWN_PARAMETERS;
const _: () = assert!(ACTIVE_WIDTH == EXPERT_WIDTH);

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
    const fn new(rows: u32, width: u32) -> Self {
        Self {
            rows,
            width,
            experts: ROUTED_EXPERTS,
            top_k: ROUTED_TOPK,
            block_tokens: ROUTE_BLOCK,
            blocks: rows.div_ceil(ROUTE_BLOCK),
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
    joint_activation: Option<ComputePipelineState>,
    routed_down: ComputePipelineState,
    shared_gate: ComputePipelineState,
    shared_down: ComputePipelineState,
    combine: ComputePipelineState,
    combine_residual_rms: ComputePipelineState,
    gate_simdgroups: u64,
    down_simdgroups: u64,
    shared_column_step: u32,
    fused_route_pack: bool,
    fused_combine_rms: bool,
    int8_gate: bool,
    interleaved_gate: bool,
}

impl FineGrainedMoePipelines {
    pub(super) fn new(runtime: &Runtime) -> Result<Self, String> {
        Self::with_source(runtime, None)
    }

    pub(super) fn with_source(
        runtime: &Runtime,
        replacement: Option<&str>,
    ) -> Result<Self, String> {
        let source = include_str!("fbt_routing.metal");
        let production_source = format!(
            "#define FUSED_ROUTED_GATE_COLUMNS\n#define FUSED_ROUTED_DOWN_COLUMNS\n{source}"
        );
        let utility = |name| runtime.pipeline(source, "deterministic top-3 MoE routing", name);
        let tensorops_source =
            replacement.unwrap_or(include_str!("fbt_moe_routing_tensorops.metal"));
        let tensorops =
            |description, name| runtime.pipeline_metal4(tensorops_source, description, name, &[]);
        Ok(Self {
            router_logits: tensorops(
                "Metal 4 fine-grained MoE router projection",
                "fbt_moe_router_logits",
            )?,
            select: utility("fbt_moe_select_top3")?,
            histogram: utility("fbt_moe_route_histogram")?,
            prefix: runtime.pipeline(
                &production_source,
                "deterministic top-3 MoE routing",
                "fbt_moe_route_prefix",
            )?,
            rank: utility("fbt_moe_route_rank")?,
            pack: utility("fbt_moe_route_pack_rows")?,
            routed_gate: tensorops(
                "Metal 4 routed MoE gate/up",
                "fbt_moe_routed_gate_fused_columns",
            )?,
            joint_activation: None,
            routed_down: tensorops(
                "Metal 4 routed MoE down",
                "fbt_moe_routed_down_fused_columns",
            )?,
            shared_gate: tensorops(
                "Metal 4 shared MoE gate/up",
                "fbt_moe_shared_gate_fused_columns",
            )?,
            shared_down: tensorops(
                "Metal 4 shared MoE down",
                "fbt_moe_shared_down_fused_columns",
            )?,
            combine: utility("fbt_moe_combine_rows")?,
            combine_residual_rms: utility("fbt_moe_combine_residual_rms")?,
            gate_simdgroups: 4,
            down_simdgroups: 4,
            shared_column_step: 128,
            fused_route_pack: true,
            fused_combine_rms: true,
            int8_gate: false,
            interleaved_gate: false,
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
    routed_tiles: Buffer,
    routed_dispatch: Buffer,
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

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct RouteStats {
    pub active_experts: usize,
    pub routed_rows: usize,
    pub routed_tiles: usize,
}

impl FineGrainedMoeBuffers {
    pub(super) fn new(runtime: &Runtime) -> Self {
        Self::with_shape(runtime, Top3RouteShape::new(ROWS, WIDTH))
    }

    fn with_shape(runtime: &Runtime, shape: Top3RouteShape) -> Self {
        let rows = shape.rows as usize;
        let width = shape.width as usize;
        let assignments = shape.assignments() as usize;
        let experts = shape.experts as usize;
        let blocks = shape.blocks as usize;
        let max_tiles = shape.assignments().div_ceil(TILE_ROWS) as usize + experts - 1;
        Self {
            scores: runtime.buffer::<u16>(rows * experts),
            route_experts: runtime.buffer::<u32>(assignments),
            route_weights: runtime.buffer::<u16>(assignments),
            route_margin: runtime.buffer::<f32>(rows),
            block_histogram: runtime.buffer::<u32>(blocks * experts),
            block_offsets: runtime.buffer::<u32>(blocks * experts),
            expert_offsets: runtime.buffer::<u32>(experts),
            expert_loads: runtime.buffer::<u32>(experts),
            routed_tiles: runtime.buffer::<u32>(max_tiles * TILE_WORDS),
            routed_dispatch: runtime.buffer::<u32>(DISPATCH_WORDS),
            packed_rows: runtime.buffer::<u32>(assignments),
            packed_input: runtime.buffer::<u16>(assignments * width),
            packed_tokens: runtime.buffer::<u32>(assignments),
            packed_experts: runtime.buffer::<u32>(assignments),
            packed_weights: runtime.buffer::<u16>(assignments),
            routed_activation: runtime
                .buffer_with(&vec![0u16; assignments * ACTIVATION_STRIDE as usize]),
            routed_output: runtime.buffer::<u16>(assignments * width),
            shared_activation: runtime.buffer_with(&vec![0u16; rows * ACTIVATION_STRIDE as usize]),
            shared_output: runtime.buffer::<u16>(rows * width),
        }
    }

    pub(super) fn validate_metadata(&self) -> Result<(), String> {
        let shape = Top3RouteShape::new(ROWS, WIDTH);
        let assignments = shape.assignments() as usize;
        let read = |buffer: &BufferRef, elements: usize| unsafe {
            std::slice::from_raw_parts(buffer.contents().cast::<u32>(), elements)
        };
        let experts = read(&self.route_experts, assignments);
        let packed_rows = read(&self.packed_rows, assignments);
        let tokens = read(&self.packed_tokens, assignments);
        let packed_experts = read(&self.packed_experts, assignments);
        let mut seen = vec![false; assignments];
        for assignment in 0..assignments {
            let expert = experts[assignment] as usize;
            let packed = packed_rows[assignment] as usize;
            if packed >= assignments || seen[packed] {
                return Err(format!(
                    "fused route pack produced invalid or duplicate row {packed} for assignment {assignment}"
                ));
            }
            seen[packed] = true;
            if tokens[packed] as usize != assignment / shape.top_k as usize
                || packed_experts[packed] as usize != expert
            {
                return Err(format!(
                    "fused route pack metadata mismatch at assignment {assignment}, packed row {packed}"
                ));
            }
        }
        Ok(())
    }

    /// Read the final routed layer after its command buffer has completed.
    /// These are logical routing counts, not estimates of physical DRAM traffic.
    pub(super) fn route_stats(&self) -> RouteStats {
        let loads = unsafe {
            std::slice::from_raw_parts(
                self.expert_loads.contents().cast::<u32>(),
                ROUTED_EXPERTS as usize,
            )
        };
        let dispatch = unsafe {
            std::slice::from_raw_parts(
                self.routed_dispatch.contents().cast::<u32>(),
                DISPATCH_WORDS,
            )
        };
        RouteStats {
            active_experts: loads.iter().filter(|&&load| load != 0).count(),
            routed_rows: loads.iter().map(|&load| load as usize).sum(),
            routed_tiles: dispatch[1] as usize,
        }
    }
}

/// Encode token-choice top-3 selection and stable variable-count packing. The
/// consumer receives contiguous expert segments plus exact offsets and counts.
fn route_top3(
    command: &CommandBufferRef,
    pipelines: &FineGrainedMoePipelines,
    buffers: &FineGrainedMoeBuffers,
    input: &BufferRef,
    input_offset: u64,
    router: &BufferRef,
    router_offset: u64,
    shape: Top3RouteShape,
) -> Result<(), String> {
    let encoder = command.new_compute_command_encoder();
    let result = route_command(
        &encoder,
        pipelines,
        buffers,
        input,
        input_offset,
        router,
        router_offset,
        shape,
    );
    encoder.end_encoding();
    result
}

fn route_pack(
    encoder: &ComputeCommandEncoderRef,
    pipelines: &FineGrainedMoePipelines,
    buffers: &FineGrainedMoeBuffers,
    input: &BufferRef,
    input_offset: u64,
    shape: Top3RouteShape,
) -> Result<(), String> {
    let shape_ptr = (&shape as *const Top3RouteShape).cast();
    let shape_bytes = std::mem::size_of::<Top3RouteShape>() as u64;
    encoder.set_compute_pipeline_state(&pipelines.pack);
    let common: [(&BufferRef, u64); 4] = [
        (input, input_offset),
        (&buffers.route_experts, 0),
        (&buffers.route_weights, 0),
        (&buffers.packed_rows, 0),
    ];
    let outputs: [(&BufferRef, u64); 4] = [
        (&buffers.packed_input, 0),
        (&buffers.packed_tokens, 0),
        (&buffers.packed_experts, 0),
        (&buffers.packed_weights, 0),
    ];
    if pipelines.fused_route_pack {
        for (index, (buffer, offset)) in common
            .into_iter()
            .chain(std::iter::once((buffers.block_offsets.as_ref(), 0)))
            .chain(outputs)
            .enumerate()
        {
            encoder.set_buffer(index as u64, Some(buffer), offset);
        }
        encoder.set_bytes(9, shape_bytes, shape_ptr);
        let threads = 128;
        if pipelines.pack.max_total_threads_per_threadgroup() < threads {
            return Err("row-oriented MoE pack requires 128 threads per threadgroup".into());
        }
        encoder.dispatch_thread_groups(
            thread_group(u64::from(shape.assignments())),
            thread_group(threads),
        );
    } else {
        for (index, (buffer, offset)) in common.into_iter().chain(outputs).enumerate() {
            encoder.set_buffer(index as u64, Some(buffer), offset);
        }
        encoder.set_bytes(8, shape_bytes, shape_ptr);
        let width = pipelines.pack.max_total_threads_per_threadgroup().min(256);
        encoder.dispatch_threads(
            thread_group(u64::from(shape.assignments()) * u64::from(shape.width)),
            thread_group(width),
        );
    }
    encoder.memory_barrier_with_resources(&[
        &buffers.packed_input,
        &buffers.packed_tokens,
        &buffers.packed_experts,
        &buffers.packed_weights,
    ]);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn route_command(
    encoder: &ComputeCommandEncoderRef,
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
    encoder.memory_barrier_with_resources(&[&buffers.scores]);

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
            width: simd_width,
            height: 1,
            depth: 1,
        },
    );
    encoder.memory_barrier_with_resources(&[
        &buffers.route_experts,
        &buffers.route_weights,
        &buffers.route_margin,
    ]);

    encoder.set_compute_pipeline_state(&pipelines.histogram);
    encoder.set_buffer(0, Some(&buffers.route_experts), 0);
    encoder.set_buffer(1, Some(&buffers.block_histogram), 0);
    encoder.set_buffer(2, Some(&buffers.packed_rows), 0);
    encoder.set_bytes(3, shape_bytes, shape_ptr);
    if pipelines.histogram.max_total_threads_per_threadgroup() < ROUTE_THREADS
        || pipelines.prefix.max_total_threads_per_threadgroup() < ROUTE_THREADS
    {
        return Err("wide MoE routing requires 640-thread groups".into());
    }
    encoder.dispatch_thread_groups(
        thread_group(u64::from(shape.blocks)),
        thread_group(ROUTE_THREADS),
    );
    encoder.memory_barrier_with_resources(&[&buffers.block_histogram, &buffers.packed_rows]);
    encoder.set_compute_pipeline_state(&pipelines.prefix);
    for (index, buffer) in [
        &buffers.block_histogram,
        &buffers.block_offsets,
        &buffers.expert_offsets,
        &buffers.expert_loads,
        &buffers.routed_tiles,
        &buffers.routed_dispatch,
    ]
    .into_iter()
    .enumerate()
    {
        encoder.set_buffer(index as u64, Some(buffer), 0);
    }
    encoder.set_bytes(6, shape_bytes, shape_ptr);
    encoder.dispatch_thread_groups(thread_group(1), thread_group(ROUTE_THREADS));
    encoder.memory_barrier_with_resources(&[
        &buffers.block_offsets,
        &buffers.expert_offsets,
        &buffers.expert_loads,
        &buffers.routed_tiles,
        &buffers.routed_dispatch,
    ]);
    if !pipelines.fused_route_pack {
        dispatch_on(
            encoder,
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
        encoder.memory_barrier_with_resources(&[&buffers.packed_rows]);
    }

    route_pack(encoder, pipelines, buffers, input, input_offset, shape)
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
            Top3RouteShape::new(ROWS, WIDTH),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_on(
        &self,
        encoder: &ComputeCommandEncoderRef,
        buffers: &FineGrainedMoeBuffers,
        input: &BufferRef,
        input_offset: u64,
        weights: &BufferRef,
        router_offset: u64,
        gate_up_offset: u64,
        down_offset: u64,
        output: &BufferRef,
    ) -> Result<(), String> {
        self.shape_command(
            encoder,
            buffers,
            input,
            input_offset,
            weights,
            router_offset,
            gate_up_offset,
            down_offset,
            output,
            Top3RouteShape::new(ROWS, WIDTH),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn shape_command(
        &self,
        encoder: &ComputeCommandEncoderRef,
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
        route_command(
            encoder,
            self,
            buffers,
            input,
            input_offset,
            weights,
            router_offset,
            shape,
        )?;
        self.gate_shape(
            encoder,
            buffers,
            input,
            input_offset,
            weights,
            gate_up_offset,
            shape,
        )?;
        self.down_shape(encoder, buffers, weights, down_offset, shape)?;
        self.combine_shape(encoder, buffers, output, shape);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn route_rows(
        &self,
        encoder: &ComputeCommandEncoderRef,
        buffers: &FineGrainedMoeBuffers,
        input: &BufferRef,
        input_offset: u64,
        weights: &BufferRef,
        router_offset: u64,
        rows: u32,
    ) -> Result<(), String> {
        route_command(
            encoder,
            self,
            buffers,
            input,
            input_offset,
            weights,
            router_offset,
            Top3RouteShape::new(rows, WIDTH),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn gate_rows(
        &self,
        encoder: &ComputeCommandEncoderRef,
        buffers: &FineGrainedMoeBuffers,
        input: &BufferRef,
        input_offset: u64,
        weights: &BufferRef,
        gate_up_offset: u64,
        rows: u32,
    ) -> Result<(), String> {
        self.gate_shape(
            encoder,
            buffers,
            input,
            input_offset,
            weights,
            gate_up_offset,
            Top3RouteShape::new(rows, WIDTH),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn gate_shape(
        &self,
        encoder: &ComputeCommandEncoderRef,
        buffers: &FineGrainedMoeBuffers,
        input: &BufferRef,
        input_offset: u64,
        weights: &BufferRef,
        gate_up_offset: u64,
        shape: Top3RouteShape,
    ) -> Result<(), String> {
        self.routed_gate(encoder, buffers, weights, gate_up_offset, shape)?;
        self.shared_gate(
            encoder,
            buffers,
            input,
            input_offset,
            weights,
            gate_up_offset,
            shape,
        )?;
        encoder.memory_barrier_with_resources(&[
            &buffers.routed_activation,
            &buffers.shared_activation,
        ]);
        Ok(())
    }

    fn routed_gate(
        &self,
        encoder: &ComputeCommandEncoderRef,
        buffers: &FineGrainedMoeBuffers,
        weights: &BufferRef,
        gate_up_offset: u64,
        shape: Top3RouteShape,
    ) -> Result<(), String> {
        let shape_ptr = (&shape as *const Top3RouteShape).cast();
        let shape_bytes = std::mem::size_of::<Top3RouteShape>() as u64;
        let gate_output = if self.joint_activation.is_some() {
            &buffers.routed_output
        } else {
            &buffers.routed_activation
        };
        tensor_indirect(
            encoder,
            &self.routed_gate,
            &[
                (&buffers.packed_input, 0),
                (weights, gate_up_offset),
                (&buffers.routed_tiles, 0),
                (gate_output, 0),
            ],
            shape_ptr,
            shape_bytes,
            &buffers.routed_dispatch,
            0,
            self.gate_simdgroups,
        )?;
        if let Some(activation) = &self.joint_activation {
            encoder.memory_barrier_with_resources(&[&buffers.routed_output]);
            encoder.set_compute_pipeline_state(activation);
            encoder.set_buffer(0, Some(&buffers.routed_output), 0);
            encoder.set_buffer(1, Some(&buffers.routed_activation), 0);
            encoder.dispatch_threads(
                thread_group(u64::from(shape.assignments()) * 54),
                thread_group(128),
            );
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn shared_gate(
        &self,
        encoder: &ComputeCommandEncoderRef,
        buffers: &FineGrainedMoeBuffers,
        input: &BufferRef,
        input_offset: u64,
        weights: &BufferRef,
        gate_up_offset: u64,
        shape: Top3RouteShape,
    ) -> Result<(), String> {
        let shape_ptr = (&shape as *const Top3RouteShape).cast();
        let shape_bytes = std::mem::size_of::<Top3RouteShape>() as u64;
        tensor_encode(
            encoder,
            &self.shared_gate,
            &[
                (input, input_offset),
                (weights, gate_up_offset),
                (&buffers.shared_activation, 0),
            ],
            shape_ptr,
            shape_bytes,
            MTLSize {
                width: u64::from(SHARED_WIDTH.div_ceil(self.shared_column_step)),
                height: u64::from(shape.rows.div_ceil(128)),
                depth: 1,
            },
        )?;
        Ok(())
    }

    pub(super) fn down_rows(
        &self,
        encoder: &ComputeCommandEncoderRef,
        buffers: &FineGrainedMoeBuffers,
        weights: &BufferRef,
        down_offset: u64,
        rows: u32,
    ) -> Result<(), String> {
        self.down_shape(
            encoder,
            buffers,
            weights,
            down_offset,
            Top3RouteShape::new(rows, WIDTH),
        )
    }

    fn down_shape(
        &self,
        encoder: &ComputeCommandEncoderRef,
        buffers: &FineGrainedMoeBuffers,
        weights: &BufferRef,
        down_offset: u64,
        shape: Top3RouteShape,
    ) -> Result<(), String> {
        let shape_ptr = (&shape as *const Top3RouteShape).cast();
        let shape_bytes = std::mem::size_of::<Top3RouteShape>() as u64;
        tensor_indirect(
            encoder,
            &self.routed_down,
            &[
                (&buffers.routed_activation, 0),
                (weights, down_offset),
                (&buffers.routed_tiles, 0),
                (&buffers.routed_output, 0),
            ],
            shape_ptr,
            shape_bytes,
            &buffers.routed_dispatch,
            3 * std::mem::size_of::<u32>() as u64,
            self.down_simdgroups,
        )?;
        tensor_encode(
            encoder,
            &self.shared_down,
            &[
                (&buffers.shared_activation, 0),
                (weights, down_offset),
                (&buffers.shared_output, 0),
            ],
            shape_ptr,
            shape_bytes,
            MTLSize {
                width: u64::from(shape.width.div_ceil(self.shared_column_step)),
                height: u64::from(shape.rows.div_ceil(128)),
                depth: 1,
            },
        )?;
        encoder.memory_barrier_with_resources(&[&buffers.routed_output, &buffers.shared_output]);
        Ok(())
    }

    pub(super) const fn fused_combine(&self) -> bool {
        self.fused_combine_rms
    }

    pub(super) const fn int8_gate(&self) -> bool {
        self.int8_gate
    }

    pub(super) const fn interleaved_gate(&self) -> bool {
        self.interleaved_gate
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn combine_rows(
        &self,
        encoder: &ComputeCommandEncoderRef,
        buffers: &FineGrainedMoeBuffers,
        residual: &BufferRef,
        norm: (&BufferRef, u64),
        output: &BufferRef,
        normalized: &BufferRef,
        rows: u32,
    ) {
        self.combine_offsets(
            encoder,
            buffers,
            (residual, 0),
            norm,
            (output, 0),
            (normalized, 0),
            rows,
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn combine_offsets(
        &self,
        encoder: &ComputeCommandEncoderRef,
        buffers: &FineGrainedMoeBuffers,
        residual: (&BufferRef, u64),
        norm: (&BufferRef, u64),
        output: (&BufferRef, u64),
        normalized: (&BufferRef, u64),
        rows: u32,
    ) {
        let shape = Top3RouteShape::new(rows, WIDTH);
        encoder.set_compute_pipeline_state(&self.combine_residual_rms);
        let bindings: [(&BufferRef, u64); 8] = [
            (&buffers.route_weights, 0),
            (&buffers.packed_rows, 0),
            (&buffers.routed_output, 0),
            (&buffers.shared_output, 0),
            residual,
            norm,
            output,
            normalized,
        ];
        for (index, (buffer, offset)) in bindings.into_iter().enumerate() {
            encoder.set_buffer(index as u64, Some(buffer), offset);
        }
        encoder.set_bytes(
            8,
            std::mem::size_of::<Top3RouteShape>() as u64,
            (&shape as *const Top3RouteShape).cast(),
        );
        encoder.dispatch_thread_groups(
            thread_group(u64::from(shape.rows)),
            MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );
        encoder.memory_barrier_with_resources(&[output.0, normalized.0]);
    }

    fn combine_shape(
        &self,
        encoder: &ComputeCommandEncoderRef,
        buffers: &FineGrainedMoeBuffers,
        output: &BufferRef,
        shape: Top3RouteShape,
    ) {
        let shape_ptr = (&shape as *const Top3RouteShape).cast();
        let shape_bytes = std::mem::size_of::<Top3RouteShape>() as u64;
        encoder.set_compute_pipeline_state(&self.combine);
        for (index, buffer) in [
            &buffers.route_weights,
            &buffers.packed_rows,
            &buffers.routed_output,
            &buffers.shared_output,
            output,
        ]
        .into_iter()
        .enumerate()
        {
            encoder.set_buffer(index as u64, Some(buffer), 0);
        }
        encoder.set_bytes(5, shape_bytes, shape_ptr);
        encoder.dispatch_thread_groups(
            thread_group(u64::from(shape.rows)),
            thread_group(u64::from(shape.width / 4)),
        );
        encoder.memory_barrier_with_resources(&[output]);
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
        let encoder = command.new_compute_command_encoder();
        let result = self.shape_command(
            &encoder,
            buffers,
            input,
            input_offset,
            weights,
            router_offset,
            gate_up_offset,
            down_offset,
            output,
            shape,
        );
        encoder.end_encoding();
        result
    }
}

fn tensor_encode(
    encoder: &ComputeCommandEncoderRef,
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
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn tensor_indirect(
    encoder: &ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    buffers: &[(&BufferRef, u64)],
    parameters: *const std::ffi::c_void,
    parameter_bytes: u64,
    dispatch: &BufferRef,
    dispatch_offset: u64,
    simdgroups: u64,
) -> Result<(), String> {
    let simd_width = pipeline.thread_execution_width();
    let threads = simd_width * simdgroups;
    if simd_width != 32 || pipeline.max_total_threads_per_threadgroup() < threads {
        return Err(format!(
            "MoE TensorOps kernel requires {simdgroups} 32-lane SIMD groups; pipeline reports SIMD width {simd_width} and max threads {}",
            pipeline.max_total_threads_per_threadgroup()
        ));
    }
    encoder.set_compute_pipeline_state(pipeline);
    for (index, (buffer, offset)) in buffers.iter().enumerate() {
        encoder.set_buffer(index as u64, Some(buffer), *offset);
    }
    encoder.set_bytes(buffers.len() as u64, parameter_bytes, parameters);
    encoder.dispatch_thread_groups_indirect(
        dispatch,
        dispatch_offset,
        MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        },
    );
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
    fn route_contract() -> Result<(), String> {
        autoreleasepool(|| {
            let runtime = Runtime::shared()?;
            let shape = Top3RouteShape::new(257, WIDTH);
            let pipelines = FineGrainedMoePipelines::new(&runtime)?;
            let buffers = FineGrainedMoeBuffers::with_shape(&runtime, shape);
            let input = runtime.buffer_with(&vec![0x3c00u16; (shape.rows * shape.width) as usize]);
            let router =
                runtime.buffer_with(&vec![0x3c00u16; (shape.width * shape.experts) as usize]);
            let command = runtime.queue.new_command_buffer();
            route_top3(command, &pipelines, &buffers, &input, 0, &router, 0, shape)?;
            super::super::complete(command)?;

            let assignments = shape.assignments() as usize;
            let route_experts = unsafe { contents::<u32>(&buffers.route_experts, assignments) };
            let route_weights = unsafe { contents::<u16>(&buffers.route_weights, assignments) };
            let margins = unsafe { contents::<f32>(&buffers.route_margin, shape.rows as usize) };
            for token in 0..shape.rows as usize {
                assert_eq!(
                    &route_experts
                        [token * ROUTED_TOPK as usize..(token + 1) * ROUTED_TOPK as usize],
                    &[0, 1, 2]
                );
                let weights = &route_weights
                    [token * ROUTED_TOPK as usize..(token + 1) * ROUTED_TOPK as usize];
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

            let expected_tiles: Vec<_> = (0..ROUTED_TOPK)
                .flat_map(|expert| {
                    (0..shape.rows)
                        .step_by(TILE_ROWS as usize)
                        .flat_map(move |first| {
                            [
                                expert,
                                expert * shape.rows + first,
                                TILE_ROWS.min(shape.rows - first),
                                0,
                            ]
                        })
                })
                .collect();
            let routed_tiles =
                unsafe { contents::<u32>(&buffers.routed_tiles, expected_tiles.len()) };
            assert_eq!(routed_tiles, expected_tiles);
            let tile_count = ROUTED_TOPK * shape.rows.div_ceil(TILE_ROWS);
            let dispatch = unsafe { contents::<u32>(&buffers.routed_dispatch, DISPATCH_WORDS) };
            assert_eq!(
                dispatch,
                &[
                    ROUTED_WIDTH.div_ceil(128),
                    tile_count,
                    1,
                    WIDTH.div_ceil(128),
                    tile_count,
                    1,
                ]
            );

            let packed_rows = unsafe { contents::<u32>(&buffers.packed_rows, assignments) };
            let mut permutation = packed_rows.to_vec();
            permutation.sort_unstable();
            assert_eq!(permutation, (0..shape.assignments()).collect::<Vec<_>>());

            let packed_tokens = unsafe { contents::<u32>(&buffers.packed_tokens, assignments) };
            let packed_experts = unsafe { contents::<u32>(&buffers.packed_experts, assignments) };
            let packed_weights = unsafe { contents::<u16>(&buffers.packed_weights, assignments) };
            for expert in 0..ROUTED_TOPK {
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
    fn varied_routes() -> Result<(), String> {
        autoreleasepool(|| {
            let runtime = Runtime::shared()?;
            let shape = Top3RouteShape::new(257, WIDTH);
            let pipelines = FineGrainedMoePipelines::new(&runtime)?;
            let buffers = FineGrainedMoeBuffers::with_shape(&runtime, shape);
            let routes: Vec<_> = (0..shape.assignments())
                .map(|i| (i / ROUTED_TOPK * 17 + i % ROUTED_TOPK * 43) % ROUTED_EXPERTS)
                .collect();
            unsafe {
                std::ptr::copy_nonoverlapping(
                    routes.as_ptr(),
                    buffers.route_experts.contents().cast::<u32>(),
                    routes.len(),
                );
            }
            let command = runtime.queue.new_command_buffer();
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipelines.histogram);
            for (index, buffer) in [
                &buffers.route_experts,
                &buffers.block_histogram,
                &buffers.packed_rows,
            ]
            .into_iter()
            .enumerate()
            {
                encoder.set_buffer(index as u64, Some(buffer), 0);
            }
            let shape_ptr = (&shape as *const Top3RouteShape).cast();
            let shape_bytes = std::mem::size_of::<Top3RouteShape>() as u64;
            encoder.set_bytes(3, shape_bytes, shape_ptr);
            encoder.dispatch_thread_groups(
                thread_group(u64::from(shape.blocks)),
                thread_group(ROUTE_THREADS),
            );
            encoder
                .memory_barrier_with_resources(&[&buffers.block_histogram, &buffers.packed_rows]);
            prefix_routes(encoder, &pipelines, &buffers, &shape);
            encoder.memory_barrier_with_resources(&[&buffers.block_offsets]);
            dispatch_on(
                encoder,
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
            encoder.end_encoding();
            super::super::complete(command)?;
            check_prefix(&buffers, &routes);
            Ok(())
        })
    }

    fn prefix_routes(
        encoder: &ComputeCommandEncoderRef,
        pipelines: &FineGrainedMoePipelines,
        buffers: &FineGrainedMoeBuffers,
        shape: &Top3RouteShape,
    ) {
        encoder.set_compute_pipeline_state(&pipelines.prefix);
        for (index, buffer) in [
            &buffers.block_histogram,
            &buffers.block_offsets,
            &buffers.expert_offsets,
            &buffers.expert_loads,
            &buffers.routed_tiles,
            &buffers.routed_dispatch,
        ]
        .into_iter()
        .enumerate()
        {
            encoder.set_buffer(index as u64, Some(buffer), 0);
        }
        encoder.set_bytes(
            6,
            std::mem::size_of::<Top3RouteShape>() as u64,
            (shape as *const Top3RouteShape).cast(),
        );
        encoder.dispatch_thread_groups(thread_group(1), thread_group(ROUTE_THREADS));
    }

    fn check_prefix(buffers: &FineGrainedMoeBuffers, routes: &[u32]) {
        let mut loads = [0u32; ROUTED_EXPERTS as usize];
        for &expert in routes {
            loads[expert as usize] += 1;
        }
        assert_eq!(
            unsafe { contents::<u32>(&buffers.expert_loads, loads.len()) },
            loads
        );
        let mut offsets = loads;
        let mut sum = 0;
        for (offset, count) in offsets.iter_mut().zip(loads) {
            *offset = sum;
            sum += count;
        }
        assert_eq!(
            unsafe { contents::<u32>(&buffers.expert_offsets, offsets.len()) },
            offsets
        );
        let packed = unsafe { contents::<u32>(&buffers.packed_rows, routes.len()) };
        for (&expert, &row) in routes.iter().zip(packed) {
            assert_eq!(row, offsets[expert as usize]);
            offsets[expert as usize] += 1;
        }
    }

    #[test]
    fn active_contract() {
        let shape = Top3RouteShape::new(ROWS, WIDTH);
        assert_eq!(shape.assignments(), ROWS * ROUTED_TOPK);
        assert_eq!(ACTIVE_WIDTH, EXPERT_WIDTH);
        assert_eq!(shape.experts, 625);
        assert_eq!(MOE_EXPERTS, 626);
        assert_eq!(GATE_PARAMETERS, 138_461_184);
        assert_eq!(DOWN_PARAMETERS, 69_230_592);
        assert_eq!(FFN_PARAMETERS, 207_691_776);
    }

    #[test]
    fn expert_recombine() -> Result<(), String> {
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
            let gate_elements = (MOE_EXPERTS * WIDTH * MOE_GATEUP) as usize;
            let down_elements = (MOE_EXPERTS * ROUTED_WIDTH * WIDTH) as usize;
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
            for expert in 0..=ROUTED_TOPK as usize {
                let gate_base = expert * (WIDTH * MOE_GATEUP) as usize;
                gate_values[gate_base] = 0x3c00;
                gate_values[gate_base + ROUTED_WIDTH as usize] = 0x3c00;
                let down_base = expert * (ROUTED_WIDTH * WIDTH) as usize;
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
