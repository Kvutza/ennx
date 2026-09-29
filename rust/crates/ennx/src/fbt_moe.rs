//! Exact-shape grouped-MoE feasibility gate for the proposed subsecond scorer.

use crate::apple_gpu::{Runtime, gpu_seconds, thread_group};
use crate::bf16_metal::{ControllerInfo, NoisyDecision, ParamBlock, Proposals, SearchState};
use crate::fbt_mps::{Matmul, Matrix};
use crate::fbt_pisa1::{Pisa1, run_pisa1_probe};
use crate::trust_region::TRLengthConfig;
use metal::{
    Buffer, BufferRef, CommandBufferRef, ComputePipelineState, MTLCommandBufferStatus, MTLSize,
};
use std::io::Write;
use std::time::Instant;

#[path = "fbt_updates.rs"]
mod updates;
use updates::{Tensor, UpdateLog};

#[cfg(test)]
#[path = "fbt_ablation.rs"]
mod ablation;

#[path = "fbt_moe_routing.rs"]
mod routing;

const ROWS: u32 = 8192;
const BATCH: u32 = 2;
const CONTEXT: u32 = ROWS / BATCH;
const WIDTH: u32 = 512;
const EXPERTS: u32 = 32;
const ROWS_PER_EXPERT: u32 = ROWS / EXPERTS;
const EXPERT_WIDTH: u32 = 864;
const GATE_UP: u32 = 2 * EXPERT_WIDTH;
const QUERY_HEADS: u32 = 8;
const KV_HEADS: u32 = 1;
const HEAD_DIM: u32 = WIDTH / QUERY_HEADS;
const QKV_WIDTH: u32 = (QUERY_HEADS + 2 * KV_HEADS) * HEAD_DIM;
const PISA_BLOCK: u32 = 64;
const PISA_LEAVES: u32 = CONTEXT / PISA_BLOCK;
const PISA_NODES: u32 = 2 * PISA_LEAVES - 1;
const PISA_SELECTED: u32 = 8;
const MODEL_LAYERS: u32 = 24;
const FEEDBACK_PASSES: u32 = 2;
const VOCAB: u32 = 8192;
const LAYER_PARAMETERS: usize = (WIDTH * routing::ROUTED_EXPERTS) as usize
    + (WIDTH * QKV_WIDTH) as usize
    + (WIDTH * WIDTH) as usize
    + routing::ROUTED_FFN_PARAMETERS_PER_LAYER
    + (2 * WIDTH) as usize;
const GLOBAL_PARAMETERS: usize =
    (WIDTH * VOCAB) as usize + (2 * WIDTH * WIDTH) as usize + WIDTH as usize;
const FULL_PARAMETERS: usize = MODEL_LAYERS as usize * LAYER_PARAMETERS + GLOBAL_PARAMETERS;
const _: () = assert!(FULL_PARAMETERS == 1_047_650_816);
const HISTORY_CAPACITY: usize = 2;

#[repr(C)]
#[derive(Clone, Copy)]
struct MoeShape {
    rows: u32,
    width: u32,
    experts: u32,
    rows_per_expert: u32,
    expert_width: u32,
}

#[derive(Debug, Clone)]
pub struct GroupedMoeProbe {
    updates: UpdateLog,
    pub parameters: usize,
    pub routing_gpu_seconds: f64,
    pub activation_gpu_seconds: f64,
    pub residual_gpu_seconds: f64,
    pub materialize_gate_up_gpu_seconds: f64,
    pub materialize_down_gpu_seconds: f64,
    pub projections_gpu_seconds: f64,
    pub projections_wall_seconds: f64,
    pub pisa1_pyramid_gpu_seconds: f64,
    pub pisa1_selection_gpu_seconds: f64,
    pub pisa1_attention_gpu_seconds: f64,
    pub pisa1_layer_gpu_seconds: f64,
    pub pisa1_layer_wall_seconds: f64,
    pub layer_gpu_seconds: f64,
    pub layer_wall_seconds: f64,
    pub projected_ffn_seconds: f64,
    pub projected_ffn_and_projections_seconds: f64,
    pub projected_measured_model_seconds: f64,
    pub sustained_model_gpu_seconds: f64,
    pub sustained_model_wall_seconds: f64,
    pub sustained_model_min_wall_seconds: f64,
    pub sustained_model_max_wall_seconds: f64,
    pub sustained_materialization_gpu_seconds: f64,
    pub sustained_projections_gpu_seconds: f64,
    pub sustained_pisa1_gpu_seconds: f64,
    pub sustained_ffn_gpu_seconds: f64,
    pub sustained_mps_projections_gpu_seconds: f64,
    pub sustained_mps_projections_wall_seconds: f64,
    pub sustained_mps_ffn_gpu_seconds: f64,
    pub sustained_mps_ffn_wall_seconds: f64,
    pub tail_gpu_seconds: f64,
    pub tail_wall_seconds: f64,
    pub complete_envelope_gpu_seconds: f64,
    pub complete_envelope_wall_seconds: f64,
    pub complete_envelope_min_wall_seconds: f64,
    pub complete_envelope_max_wall_seconds: f64,
    pub controller_median_wall_seconds: f64,
    pub controller_min_wall_seconds: f64,
    pub controller_max_wall_seconds: f64,
    pub actual_bo_median_wall_seconds: f64,
    pub actual_bo_min_wall_seconds: f64,
    pub actual_bo_max_wall_seconds: f64,
    pub actual_bo_median_gpu_seconds: f64,
    pub actual_bo_accepted: u32,
    pub target_seconds: f64,
    pub objective_flops: u64,
    pub tail_flops: u64,
    pub complete_objective_flops: u64,
    pub projection_objective_flops: u64,
    pub pisa1_objective_flops: u64,
    pub effective_tflops: f64,
    pub gate_up_max_abs_error: f64,
    pub down_max_abs_error: f64,
    pub qkv_max_abs_error: f64,
    pub output_projection_max_abs_error: f64,
    pub pisa1_max_abs_error: f64,
    pub tail_max_abs_error: f64,
    pub meets_target: bool,
}

impl GroupedMoeProbe {
    pub fn write_updates(&self, path: &std::path::Path) -> Result<(), String> {
        self.updates.write(path)
    }
}

struct Pipelines {
    fine_grained: routing::FineGrainedMoePipelines,
    gate: ComputePipelineState,
    group: ComputePipelineState,
    swiglu: ComputePipelineState,
    ungroup: ComputePipelineState,
    feedback_fuse: ComputePipelineState,
    cross_entropy: ComputePipelineState,
    sequence_loss: ComputePipelineState,
    embed: ComputePipelineState,
    rms: ComputePipelineState,
    residual: ComputePipelineState,
    residual_rms: ComputePipelineState,
    feedback_fuse_rms: ComputePipelineState,
}

struct TensorOpsPipelines {
    gate_up: ComputePipelineState,
    gate_activation: ComputePipelineState,
    down: ComputePipelineState,
    materialize_gate_up: ComputePipelineState,
    materialize_down: ComputePipelineState,
    qkv: ComputePipelineState,
    output_projection: ComputePipelineState,
    readout: ComputePipelineState,
}

struct Buffers {
    fine_grained: routing::FineGrainedMoeBuffers,
    input: Buffer,
    router: Buffer,
    gates: Buffer,
    grouped: Buffer,
    gate_up_base: Buffer,
    gate_up: Buffer,
    gate_inner: Buffer,
    gate_outer: Buffer,
    activation: Buffer,
    down_base: Buffer,
    down: Buffer,
    down_inner: Buffer,
    down_outer: Buffer,
    output: Buffer,
    mps_gate_up: Buffer,
    mps_down: Buffer,
    materialized_gate_up: Buffer,
    materialized_down: Buffer,
    qkv_weights: Buffer,
    qkv: Buffer,
    output_projection_weights: Buffer,
    projected: Buffer,
    mps_qkv: Buffer,
    mps_projected: Buffer,
    feedback_state_weights: Buffer,
    feedback_gate_weights: Buffer,
    feedback_state: Buffer,
    feedback_gate: Buffer,
    feedback: Buffer,
    readout_weights: Buffer,
    logits: Buffer,
    labels: Buffer,
    losses: Buffer,
    score_mask: Buffer,
    sequence_scores: Buffer,
    tokens: Buffer,
    unit_norm: Buffer,
    normalized: Buffer,
    attention_state: Buffer,
}

struct CandidateWeights {
    router: Buffer,
    qkv: Buffer,
    output: Buffer,
    gate_up: Buffer,
    down: Buffer,
    attention_norm: Buffer,
    ffn_norm: Buffer,
    readout: Buffer,
    feedback_state: Buffer,
    feedback_gate: Buffer,
    final_norm: Buffer,
}

#[derive(Clone, Copy)]
struct CandidateRow<'a> {
    buffer: &'a BufferRef,
    router: u64,
    qkv: u64,
    output: u64,
    gate_up: u64,
    down: u64,
    attention_norm: u64,
    ffn_norm: u64,
    readout: u64,
    feedback_state: u64,
    feedback_gate: u64,
    final_norm: u64,
}

pub struct ActualBoResult {
    pub parameters: usize,
    pub loop_seconds: f64,
    pub median_wall_seconds: f64,
    pub min_wall_seconds: f64,
    pub max_wall_seconds: f64,
    pub median_gpu_seconds: f64,
    pub accepted: u32,
    controller_seconds: Vec<f64>,
    controller_records: Vec<serde_json::Value>,
    updates: UpdateLog,
}

impl ActualBoResult {
    pub fn write_updates(&self, path: &std::path::Path) -> Result<(), String> {
        self.updates.write(path)
    }

    pub fn write_controller(&self, path: &std::path::Path) -> Result<(), String> {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(|error| error.to_string())?;
        for record in &self.controller_records {
            writeln!(file, "{record}").map_err(|error| error.to_string())?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Stage {
    Gate,
    Group,
    Swiglu,
    Ungroup,
}

impl Stage {
    const ALL: [(Self, &'static str); 4] = [
        (Self::Gate, "router_gate"),
        (Self::Group, "group"),
        (Self::Swiglu, "swiglu"),
        (Self::Ungroup, "ungroup_residual"),
    ];
}

fn filled(runtime: &Runtime, elements: usize, bits: u16) -> Buffer {
    let buffer = runtime.buffer::<u16>(elements);
    let values =
        unsafe { std::slice::from_raw_parts_mut(buffer.contents().cast::<u16>(), elements) };
    values.fill(bits);
    buffer
}

fn patterned(runtime: &Runtime, elements: usize, seed: u32, exponent: u16) -> Buffer {
    let buffer = runtime.buffer::<u16>(elements);
    let values =
        unsafe { std::slice::from_raw_parts_mut(buffer.contents().cast::<u16>(), elements) };
    let mut state = seed;
    for value in values {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        let sign = ((state >> 31) as u16) << 15;
        *value = sign | exponent | ((state >> 6) as u16 & 0x03ff);
    }
    buffer
}

fn factor(runtime: &Runtime, experts: u32, input: u32, output: u32) -> Buffer {
    let elements = (experts * input * output) as usize;
    let buffer = patterned(
        runtime,
        elements,
        0x9e37_79b9 ^ experts ^ (input << 8) ^ (output << 16),
        0x0800,
    );
    let values =
        unsafe { std::slice::from_raw_parts_mut(buffer.contents().cast::<u16>(), elements) };
    for expert in 0..experts as usize {
        for diagonal in 0..input.min(output) as usize {
            values[(expert * input as usize + diagonal) * output as usize + diagonal] = 0x2400;
        }
    }
    buffer
}

fn dispatch(
    command: &CommandBufferRef,
    pipeline: &ComputePipelineState,
    buffers: &[&BufferRef],
    parameters: *const std::ffi::c_void,
    parameter_bytes: u64,
    threads: u64,
) {
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    for (index, buffer) in buffers.iter().enumerate() {
        encoder.set_buffer(index as u64, Some(buffer), 0);
    }
    encoder.set_bytes(buffers.len() as u64, parameter_bytes, parameters);
    let width = pipeline.max_total_threads_per_threadgroup().min(256);
    encoder.dispatch_threads(thread_group(threads), thread_group(width));
    encoder.end_encoding();
}

impl Pipelines {
    fn new(runtime: &Runtime) -> Result<Self, String> {
        let utility_source = include_str!("fbt_moe.metal");
        let utility = |name| runtime.precise(utility_source, "grouped MoE layer probe", name);
        Ok(Self {
            fine_grained: routing::FineGrainedMoePipelines::new(runtime)?,
            gate: utility("fbt_moe_balanced_gate")?,
            group: utility("fbt_moe_group_balanced")?,
            swiglu: utility("fbt_moe_swiglu")?,
            ungroup: utility("fbt_moe_ungroup_residual")?,
            feedback_fuse: utility("fbt_moe_feedback_fuse")?,
            cross_entropy: utility("fbt_moe_cross_entropy")?,
            sequence_loss: utility("fbt_moe_sequence_loss")?,
            embed: utility("fbt_moe_embed")?,
            rms: utility("fbt_moe_rms")?,
            residual: utility("fbt_moe_residual")?,
            residual_rms: utility("fbt_moe_residual_rms")?,
            feedback_fuse_rms: utility("fbt_moe_feedback_fuse_rms")?,
        })
    }
}

impl TensorOpsPipelines {
    fn new(runtime: &Runtime) -> Result<Self, String> {
        let source = include_str!("fbt_moe_tensorops.metal");
        Ok(Self {
            gate_up: runtime.precise_metal4(
                source,
                "Metal 4 TensorOps grouped MoE gate/up",
                "fbt_moe_tensorops_gate_up",
                &[],
            )?,
            gate_activation: runtime.precise_metal4(
                source,
                "Metal 4 gate/up activation",
                "gate_activation",
                &[],
            )?,
            down: runtime.precise_metal4(
                source,
                "Metal 4 TensorOps grouped MoE down",
                "fbt_moe_tensorops_down",
                &[],
            )?,
            materialize_gate_up: runtime.precise_metal4(
                source,
                "Metal 4 materialized Kronecker grouped MoE gate/up",
                "fbt_moe_materialize_kronecker_gate_up",
                &[],
            )?,
            materialize_down: runtime.precise_metal4(
                source,
                "Metal 4 materialized Kronecker grouped MoE down",
                "fbt_moe_materialize_kronecker_down",
                &[],
            )?,
            qkv: runtime.precise_metal4(
                source,
                "Metal 4 TensorOps fused QKV projection",
                "fbt_model_tensorops_qkv",
                &[],
            )?,
            output_projection: runtime.precise_metal4(
                source,
                "Metal 4 TensorOps attention output projection",
                "fbt_model_tensorops_output_projection",
                &[],
            )?,
            readout: runtime.precise_metal4(
                source,
                "Metal 4 TensorOps vocabulary readout",
                "fbt_model_tensorops_readout",
                &[],
            )?,
        })
    }
}

impl Buffers {
    fn new(runtime: &Runtime) -> Self {
        let half = |n: u32| n as usize;
        Self {
            fine_grained: routing::FineGrainedMoeBuffers::new(runtime),
            input: patterned(runtime, half(ROWS * WIDTH), 0x243f_6a88, 0x2c00),
            router: patterned(runtime, half(WIDTH * EXPERTS), 0x85a3_08d3, 0x1800),
            gates: runtime.buffer::<u16>(half(ROWS)),
            grouped: runtime.buffer::<u16>(half(ROWS * WIDTH)),
            gate_up_base: patterned(
                runtime,
                half(EXPERTS * WIDTH * GATE_UP),
                0x1319_8a2e,
                0x2000,
            ),
            gate_up: runtime.buffer_with(&vec![0x7e00u16; half(ROWS * GATE_UP) + 64]),
            gate_inner: factor(runtime, EXPERTS, 32, 32),
            gate_outer: factor(runtime, EXPERTS, 16, 54),
            activation: runtime.buffer_with(&vec![0x7e00u16; half(ROWS * EXPERT_WIDTH) + 64]),
            down_base: patterned(
                runtime,
                half(EXPERTS * EXPERT_WIDTH * WIDTH),
                0x0370_7344,
                0x2000,
            ),
            down: runtime.buffer_with(&vec![0x7e00u16; half(ROWS * WIDTH) + 64]),
            down_inner: factor(runtime, EXPERTS, 27, 16),
            down_outer: factor(runtime, EXPERTS, 32, 32),
            output: runtime.buffer_with(&vec![0x7e00u16; half(ROWS * WIDTH) + 64]),
            mps_gate_up: runtime.buffer_with(&vec![0x7e00u16; half(ROWS * GATE_UP) + 64]),
            mps_down: runtime.buffer_with(&vec![0x7e00u16; half(ROWS * WIDTH) + 64]),
            materialized_gate_up: runtime
                .buffer_with(&vec![0x7e00u16; half(EXPERTS * WIDTH * GATE_UP) + 64]),
            materialized_down: runtime
                .buffer_with(&vec![0x7e00u16; half(EXPERTS * EXPERT_WIDTH * WIDTH) + 64]),
            qkv_weights: patterned(runtime, half(WIDTH * QKV_WIDTH), 0xa409_3822, 0x2000),
            qkv: runtime.buffer_with(&vec![0x7e00u16; half(ROWS * QKV_WIDTH) + 64]),
            output_projection_weights: patterned(runtime, half(WIDTH * WIDTH), 0x299f_31d0, 0x2000),
            projected: runtime.buffer_with(&vec![0x7e00u16; half(ROWS * WIDTH) + 64]),
            mps_qkv: runtime.buffer_with(&vec![0x7e00u16; half(ROWS * QKV_WIDTH) + 64]),
            mps_projected: runtime.buffer_with(&vec![0x7e00u16; half(ROWS * WIDTH) + 64]),
            feedback_state_weights: patterned(runtime, half(WIDTH * WIDTH), 0x082e_fa98, 0x2000),
            feedback_gate_weights: patterned(runtime, half(WIDTH * WIDTH), 0xec4e_6c89, 0x2000),
            feedback_state: runtime.buffer::<u16>(half(ROWS * WIDTH)),
            feedback_gate: runtime.buffer::<u16>(half(ROWS * WIDTH)),
            feedback: runtime.buffer::<u16>(half(ROWS * WIDTH)),
            readout_weights: patterned(runtime, half(WIDTH * VOCAB), 0x4528_21e6, 0x1800),
            logits: runtime.buffer::<u16>(half(ROWS * VOCAB)),
            labels: runtime.buffer_with(
                &(0..ROWS)
                    .map(|row| (row.wrapping_mul(17) + row / CONTEXT * 97 + 1) % VOCAB)
                    .collect::<Vec<_>>(),
            ),
            losses: runtime.buffer_with(&vec![f32::NAN; ROWS as usize + 64]),
            score_mask: runtime.buffer_with(&vec![1u8; ROWS as usize]),
            sequence_scores: runtime.buffer_with(&[f32::NAN; BATCH as usize]),
            tokens: runtime.buffer_with(
                &(0..ROWS)
                    .map(|row| (row.wrapping_mul(17) + row / CONTEXT * 97) % VOCAB)
                    .collect::<Vec<_>>(),
            ),
            unit_norm: filled(runtime, half(WIDTH), 0x3c00),
            normalized: runtime.buffer::<u16>(half(ROWS * WIDTH)),
            attention_state: runtime.buffer::<u16>(half(ROWS * WIDTH)),
        }
    }
}

fn load_pretraining_batch(
    buffers: &Buffers,
    dataset: &crate::pretrain_data::PretrainDataset,
    batch: u32,
) -> Result<(), String> {
    let tokens = dataset.batch(batch)?;
    let input = unsafe {
        std::slice::from_raw_parts_mut(buffers.tokens.contents().cast::<u32>(), ROWS as usize)
    };
    let labels = unsafe {
        std::slice::from_raw_parts_mut(buffers.labels.contents().cast::<u32>(), ROWS as usize)
    };
    let score_mask = unsafe {
        std::slice::from_raw_parts_mut(buffers.score_mask.contents().cast::<u8>(), ROWS as usize)
    };
    for sequence in 0..BATCH as usize {
        let start = sequence * CONTEXT as usize;
        let end = start + CONTEXT as usize;
        input[start..end]
            .iter_mut()
            .zip(&tokens[start..end])
            .for_each(|(output, &token)| *output = u32::from(token));
        for position in 0..CONTEXT as usize - 1 {
            labels[start + position] = u32::from(tokens[start + position + 1]);
            score_mask[start + position] = 1;
        }
        labels[end - 1] = 0;
        score_mask[end - 1] = 0;
    }
    Ok(())
}

impl CandidateWeights {
    fn new(runtime: &Runtime) -> Self {
        let layers = MODEL_LAYERS as usize;
        let experts = routing::MOE_EXPERTS as usize;
        let width = WIDTH as usize;
        let expert_width = routing::ROUTED_EXPERT_WIDTH as usize;
        let gate_up = routing::MOE_GATE_UP as usize;
        let qkv_width = QKV_WIDTH as usize;
        let vocab = VOCAB as usize;
        let router_elements = layers * width * routing::ROUTED_EXPERTS as usize;
        let qkv_elements = layers * width * qkv_width;
        let output_elements = layers * width * width;
        let gate_up_elements = layers * experts * width * gate_up;
        let down_elements = layers * experts * expert_width * width;
        let norm_elements = layers * width;
        let readout_elements = width * vocab;
        let feedback_elements = width * width;
        Self {
            router: patterned(runtime, router_elements, 0x6a09_e667, 0x1800),
            qkv: patterned(runtime, qkv_elements, 0xbb67_ae85, 0x2000),
            output: patterned(runtime, output_elements, 0x3c6e_f372, 0x2000),
            gate_up: patterned(runtime, gate_up_elements, 0xa54f_f53a, 0x2000),
            down: patterned(runtime, down_elements, 0x510e_527f, 0x2000),
            attention_norm: filled(runtime, norm_elements, 0x3c00),
            ffn_norm: filled(runtime, norm_elements, 0x3c00),
            readout: patterned(runtime, readout_elements, 0x9b05_688c, 0x1800),
            feedback_state: patterned(runtime, feedback_elements, 0x1f83_d9ab, 0x2000),
            feedback_gate: patterned(runtime, feedback_elements, 0x5be0_cd19, 0x2000),
            final_norm: filled(runtime, width, 0x3c00),
        }
    }
    fn tensors(&self) -> [(&'static str, &BufferRef, usize); 11] {
        [
            (
                "router",
                &self.router,
                (WIDTH * routing::ROUTED_EXPERTS) as usize,
            ),
            ("qkv", &self.qkv, (WIDTH * QKV_WIDTH) as usize),
            ("attention_output", &self.output, (WIDTH * WIDTH) as usize),
            (
                "expert_gate_up",
                &self.gate_up,
                (WIDTH * routing::MOE_GATE_UP) as usize,
            ),
            (
                "expert_down",
                &self.down,
                (routing::ROUTED_EXPERT_WIDTH * WIDTH) as usize,
            ),
            ("attention_norm", &self.attention_norm, WIDTH as usize),
            ("ffn_norm", &self.ffn_norm, WIDTH as usize),
            ("embedding_readout", &self.readout, (WIDTH * VOCAB) as usize),
            (
                "feedback_state",
                &self.feedback_state,
                (WIDTH * WIDTH) as usize,
            ),
            (
                "feedback_gate",
                &self.feedback_gate,
                (WIDTH * WIDTH) as usize,
            ),
            ("final_norm", &self.final_norm, WIDTH as usize),
        ]
    }

    fn row<'a>(&self, buffer: &'a BufferRef) -> Result<CandidateRow<'a>, String> {
        let mut offsets = [0; 11];
        let mut end = 0;
        for (index, (_, tensor, _)) in self.tensors().iter().enumerate() {
            offsets[index] = end;
            end += tensor.length();
        }
        if end != buffer.length() {
            return Err(format!(
                "candidate row has {} bytes, expected {end}",
                buffer.length()
            ));
        }
        Ok(CandidateRow {
            buffer,
            router: offsets[0],
            qkv: offsets[1],
            output: offsets[2],
            gate_up: offsets[3],
            down: offsets[4],
            attention_norm: offsets[5],
            ffn_norm: offsets[6],
            readout: offsets[7],
            feedback_state: offsets[8],
            feedback_gate: offsets[9],
            final_norm: offsets[10],
        })
    }

    fn search(
        &self,
        perturbation: crate::Perturbation,
        length: TRLengthConfig,
    ) -> Result<(SearchState, UpdateLog), String> {
        self.search_shaped(
            perturbation,
            length,
            crate::config::TrustRegionShape::TensorFamilyStatic,
        )
    }

    fn search_shaped(
        &self,
        perturbation: crate::Perturbation,
        length: TRLengthConfig,
        shape: crate::config::TrustRegionShape,
    ) -> Result<(SearchState, UpdateLog), String> {
        let mut base = Vec::with_capacity(FULL_PARAMETERS);
        let mut blocks = Vec::new();
        let mut groups = Vec::new();
        let mut updates = UpdateLog::default();
        for (family, buffer, block_len) in self.tensors() {
            let values = unsafe {
                std::slice::from_raw_parts(
                    buffer.contents().cast::<u16>(),
                    buffer.length() as usize / 2,
                )
            };
            for (index, tensor) in values.chunks_exact(block_len).enumerate() {
                let initial_rms = (tensor
                    .iter()
                    .map(|&bits| decode_half(bits).powi(2))
                    .sum::<f64>()
                    / tensor.len() as f64)
                    .sqrt();
                let trust_multiplier =
                    if shape == crate::config::TrustRegionShape::TensorFamilyStatic {
                        trust_multiplier(family)
                    } else {
                        1.0
                    };
                let scale = initial_rms.max(1e-6) as f32 * trust_multiplier;
                let (layer, expert) = match family {
                    "expert_gate_up" | "expert_down" => (
                        Some(index / routing::MOE_EXPERTS as usize),
                        Some(index % routing::MOE_EXPERTS as usize),
                    ),
                    "embedding_readout" | "feedback_state" | "feedback_gate" | "final_norm" => {
                        (None, None)
                    }
                    _ => (Some(index), None),
                };
                let name = match (layer, expert) {
                    (Some(layer), Some(expert)) => {
                        format!("layer.{layer}.expert.{expert}.{family}")
                    }
                    (Some(layer), None) => format!("layer.{layer}.{family}"),
                    _ => family.to_string(),
                };
                updates.tensors.push(Tensor {
                    name,
                    family,
                    layer,
                    expert,
                    elements: tensor.len(),
                    initial_rms,
                    proposal_scale: scale,
                    trust_multiplier,
                });
                blocks.push(ParamBlock::new(
                    blocks.len() as u64,
                    base.len(),
                    tensor.len(),
                    scale,
                    1.0 / (tensor.len() as f32 * scale * scale),
                )?);
                groups.push(family_group(family));
                base.extend_from_slice(tensor);
            }
        }
        if base.len() != FULL_PARAMETERS {
            return Err(format!(
                "Full-weight inventory has {} parameters, expected {FULL_PARAMETERS}",
                base.len()
            ));
        }
        let mut search =
            SearchState::new_fp16_implicit(&base, blocks, HISTORY_CAPACITY, length, perturbation)?;
        search.set_failure_tolerance(4)?;
        if shape == crate::config::TrustRegionShape::TensorFamilyLearned {
            search.enable_family_shape(groups)?;
        }
        Ok((search, updates))
    }
}

fn trust_multiplier(family: &str) -> f32 {
    match family {
        "expert_gate_up" | "expert_down" => 1.0,
        "qkv" | "attention_output" | "embedding_readout" => 0.75,
        "router" | "feedback_state" | "feedback_gate" => 0.5,
        "attention_norm" | "ffn_norm" | "final_norm" => 0.25,
        _ => 1.0,
    }
}

fn family_group(family: &str) -> usize {
    match family {
        "expert_gate_up" | "expert_down" => 0,
        "qkv" | "attention_output" | "embedding_readout" => 1,
        "router" | "feedback_state" | "feedback_gate" => 2,
        "attention_norm" | "ffn_norm" | "final_norm" => 3,
        _ => unreachable!("unmapped model tensor family"),
    }
}

#[derive(Clone, Copy)]
struct BoControl {
    length: TRLengthConfig,
    enn: crate::config::ResidentEnnConfig,
    proposal_seed: u64,
    acquisition_seed: u64,
    perturbation: crate::Perturbation,
    shape: crate::config::TrustRegionShape,
    paired_objective: bool,
    reliability: Option<crate::ReliabilityControllerConfig>,
}

impl BoControl {
    fn diagnostic(perturbation: crate::Perturbation) -> Result<Self, String> {
        let acquisition_seed = 0xbb67_ae85_84ca_a73b;
        Ok(Self {
            length: TRLengthConfig::new(0.01, 0.0001, 0.1),
            enn: crate::config::ConfigOverrides {
                acquisition: Some(crate::config::AcquisitionConfig::Thompson),
                ..Default::default()
            }
            .resident_enn(acquisition_seed)?,
            proposal_seed: 0x6a09_e667_f3bc_c909,
            acquisition_seed,
            perturbation,
            shape: crate::config::TrustRegionShape::TensorFamilyStatic,
            paired_objective: false,
            reliability: None,
        })
    }
}

fn encode(
    pipelines: &Pipelines,
    buffers: &Buffers,
    command: &CommandBufferRef,
    selected: Option<Stage>,
) {
    let shape = MoeShape {
        rows: ROWS,
        width: WIDTH,
        experts: EXPERTS,
        rows_per_expert: ROWS_PER_EXPERT,
        expert_width: EXPERT_WIDTH,
    };
    let shape_ptr = (&shape as *const MoeShape).cast();
    let shape_bytes = std::mem::size_of::<MoeShape>() as u64;
    if selected.is_none_or(|stage| stage == Stage::Gate) {
        dispatch(
            command,
            &pipelines.gate,
            &[&buffers.input, &buffers.router, &buffers.gates],
            shape_ptr,
            shape_bytes,
            u64::from(ROWS) * 32,
        );
    }
    if selected.is_none_or(|stage| stage == Stage::Group) {
        dispatch(
            command,
            &pipelines.group,
            &[&buffers.input, &buffers.grouped],
            shape_ptr,
            shape_bytes,
            u64::from(ROWS) * u64::from(WIDTH),
        );
    }
    if selected.is_none_or(|stage| stage == Stage::Swiglu) {
        dispatch(
            command,
            &pipelines.swiglu,
            &[&buffers.gate_up, &buffers.activation],
            shape_ptr,
            shape_bytes,
            u64::from(ROWS) * u64::from(EXPERT_WIDTH),
        );
    }
    if selected.is_none_or(|stage| stage == Stage::Ungroup) {
        dispatch(
            command,
            &pipelines.ungroup,
            &[
                &buffers.input,
                &buffers.down,
                &buffers.gates,
                &buffers.output,
            ],
            shape_ptr,
            shape_bytes,
            u64::from(ROWS) * u64::from(WIDTH),
        );
    }
}

fn complete(command: &CommandBufferRef) -> Result<f64, String> {
    command.commit();
    command.wait_until_completed();
    if command.status() != MTLCommandBufferStatus::Completed {
        return Err(format!(
            "grouped MoE command failed: {:?}",
            command.status()
        ));
    }
    gpu_seconds(command).ok_or("Metal did not report grouped MoE GPU timing".into())
}

fn encode_tensorops(
    command: &CommandBufferRef,
    pipeline: &ComputePipelineState,
    buffers: &[&BufferRef],
    output_width: u32,
) -> Result<(), String> {
    encode_tensorops_groups(
        command,
        pipeline,
        buffers,
        MTLSize {
            width: u64::from(output_width / 64),
            height: u64::from(ROWS_PER_EXPERT / 128),
            depth: u64::from(EXPERTS),
        },
    )
}

fn encode_tensorops_groups(
    command: &CommandBufferRef,
    pipeline: &ComputePipelineState,
    buffers: &[&BufferRef],
    groups: MTLSize,
) -> Result<(), String> {
    let simd_width = pipeline.thread_execution_width();
    let threads = simd_width * 4;
    if simd_width != 32 || pipeline.max_total_threads_per_threadgroup() < threads {
        return Err(format!(
            "TensorOps kernel requires four 32-lane SIMD groups; pipeline reports SIMD width {simd_width} and max threads {}",
            pipeline.max_total_threads_per_threadgroup()
        ));
    }
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    for (index, buffer) in buffers.iter().enumerate() {
        encoder.set_buffer(index as u64, Some(buffer), 0);
    }
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

fn encode_tensorops_groups_at(
    command: &CommandBufferRef,
    pipeline: &ComputePipelineState,
    buffers: &[(&BufferRef, u64)],
    groups: MTLSize,
) -> Result<(), String> {
    let simd_width = pipeline.thread_execution_width();
    let threads = simd_width * 4;
    if simd_width != 32 || pipeline.max_total_threads_per_threadgroup() < threads {
        return Err(format!(
            "TensorOps kernel requires four 32-lane SIMD groups; pipeline reports SIMD width {simd_width} and max threads {}",
            pipeline.max_total_threads_per_threadgroup()
        ));
    }
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    for (index, (buffer, offset)) in buffers.iter().enumerate() {
        encoder.set_buffer(index as u64, Some(buffer), *offset);
    }
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

fn benchmark_materialization(
    runtime: &Runtime,
    pipeline: &ComputePipelineState,
    buffers: &[&BufferRef],
    output_width: u32,
    input_width: u32,
) -> Result<f64, String> {
    let mut samples = Vec::with_capacity(3);
    for iteration in 0..6 {
        let command = runtime.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        for (index, buffer) in buffers.iter().enumerate() {
            encoder.set_buffer(index as u64, Some(buffer), 0);
        }
        encoder.dispatch_threads(
            MTLSize {
                width: u64::from(output_width / 4),
                height: u64::from(input_width),
                depth: u64::from(EXPERTS),
            },
            MTLSize {
                width: 64,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        let seconds = complete(command)?;
        if iteration >= 3 {
            samples.push(seconds);
        }
    }
    samples.sort_by(f64::total_cmp);
    Ok(samples[1])
}

fn mps_reference(runtime: &Runtime, buffers: &Buffers) -> Result<f64, String> {
    let mut matmul = Matmul::default();
    let command = runtime.queue.new_command_buffer();
    matmul.encode(
        &runtime.device,
        command,
        Matrix::half(&buffers.grouped, ROWS_PER_EXPERT, WIDTH).layout(
            EXPERTS,
            u64::from(ROWS_PER_EXPERT) * u64::from(WIDTH),
            0,
        ),
        Matrix::half(&buffers.materialized_gate_up, WIDTH, GATE_UP).layout(
            EXPERTS,
            u64::from(WIDTH) * u64::from(GATE_UP),
            0,
        ),
        Matrix::half(&buffers.mps_gate_up, ROWS_PER_EXPERT, GATE_UP).layout(
            EXPERTS,
            u64::from(ROWS_PER_EXPERT) * u64::from(GATE_UP),
            0,
        ),
        false,
        1.0,
    )?;
    matmul.encode(
        &runtime.device,
        command,
        Matrix::half(&buffers.activation, ROWS_PER_EXPERT, EXPERT_WIDTH).layout(
            EXPERTS,
            u64::from(ROWS_PER_EXPERT) * u64::from(EXPERT_WIDTH),
            0,
        ),
        Matrix::half(&buffers.materialized_down, EXPERT_WIDTH, WIDTH).layout(
            EXPERTS,
            u64::from(EXPERT_WIDTH) * u64::from(WIDTH),
            0,
        ),
        Matrix::half(&buffers.mps_down, ROWS_PER_EXPERT, WIDTH).layout(
            EXPERTS,
            u64::from(ROWS_PER_EXPERT) * u64::from(WIDTH),
            0,
        ),
        false,
        1.0,
    )?;
    complete(command)
}

fn encode_projection_pair(
    command: &CommandBufferRef,
    tensorops: &TensorOpsPipelines,
    buffers: &Buffers,
) -> Result<(), String> {
    encode_tensorops_groups(
        command,
        &tensorops.qkv,
        &[&buffers.input, &buffers.qkv_weights, &buffers.qkv],
        MTLSize {
            width: u64::from(QKV_WIDTH / 64),
            height: u64::from(ROWS / 128),
            depth: 1,
        },
    )?;
    encode_tensorops_groups(
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

fn encode_sustained_model(
    command: &CommandBufferRef,
    pipelines: &Pipelines,
    tensorops: &TensorOpsPipelines,
    pisa1: &Pisa1,
    buffers: &Buffers,
) -> Result<(), String> {
    encode_sustained_materialization(command, tensorops, buffers);
    for _ in 0..MODEL_LAYERS * FEEDBACK_PASSES {
        encode_sustained_qkv(command, tensorops, &buffers.input, buffers)?;
        pisa1.encode_layer(command, &buffers.qkv);
        encode_sustained_output_projection(command, tensorops, pisa1.output(), buffers)?;
        encode_sustained_ffn(command, pipelines, tensorops, buffers)?;
    }
    Ok(())
}

fn encode_sustained_materialization(
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

fn encode_sustained_qkv(
    command: &CommandBufferRef,
    tensorops: &TensorOpsPipelines,
    input: &BufferRef,
    buffers: &Buffers,
) -> Result<(), String> {
    encode_tensorops_groups(
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

fn encode_sustained_output_projection(
    command: &CommandBufferRef,
    tensorops: &TensorOpsPipelines,
    input: &BufferRef,
    buffers: &Buffers,
) -> Result<(), String> {
    encode_tensorops_groups(
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

fn encode_sustained_ffn(
    command: &CommandBufferRef,
    pipelines: &Pipelines,
    tensorops: &TensorOpsPipelines,
    buffers: &Buffers,
) -> Result<(), String> {
    encode_sustained_ffn_from(
        command,
        pipelines,
        tensorops,
        &buffers.input,
        &buffers.input,
        buffers,
    )
}

fn model_shape() -> MoeShape {
    MoeShape {
        rows: ROWS,
        width: WIDTH,
        experts: EXPERTS,
        rows_per_expert: ROWS_PER_EXPERT,
        expert_width: EXPERT_WIDTH,
    }
}

fn encode_rms(
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

fn encode_residual(
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

fn encode_sustained_ffn_from(
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

fn encode_feedback(
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
    encode_width_projection(
        command,
        tensorops,
        input,
        FeedbackProjection {
            weights: &buffers.feedback_state_weights,
            output: &buffers.feedback_state,
        },
    )?;
    encode_width_projection(
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

fn encode_width_projection(
    command: &CommandBufferRef,
    tensorops: &TensorOpsPipelines,
    input: &BufferRef,
    projection: FeedbackProjection<'_>,
) -> Result<(), String> {
    encode_tensorops_groups(
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

fn encode_readout_loss(
    command: &CommandBufferRef,
    pipelines: &Pipelines,
    tensorops: &TensorOpsPipelines,
    input: &BufferRef,
    buffers: &Buffers,
) -> Result<(), String> {
    encode_tensorops_groups(
        command,
        &tensorops.readout,
        &[input, &buffers.readout_weights, &buffers.logits],
        MTLSize {
            width: u64::from(VOCAB / 64),
            height: u64::from(ROWS / 128),
            depth: 1,
        },
    )?;
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipelines.cross_entropy);
    encoder.set_buffer(0, Some(&buffers.logits), 0);
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

fn encode_complete_envelope(
    command: &CommandBufferRef,
    pipelines: &Pipelines,
    tensorops: &TensorOpsPipelines,
    pisa1: &Pisa1,
    buffers: &Buffers,
) -> Result<(), String> {
    encode_sustained_materialization(command, tensorops, buffers);
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
            encode_sustained_qkv(command, tensorops, &buffers.normalized, buffers)?;
            pisa1.encode_layer(command, &buffers.qkv);
            encode_sustained_output_projection(command, tensorops, pisa1.output(), buffers)?;
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
            encode_sustained_ffn_from(
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
    encode_readout_loss(command, pipelines, tensorops, &buffers.normalized, buffers)
}

fn benchmark_complete_envelope(
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
        encode_complete_envelope(command, pipelines, tensorops, pisa1, buffers)?;
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

fn benchmark_tail(
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
        encode_readout_loss(command, pipelines, tensorops, &buffers.output, buffers)?;
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

fn half_bytes(elements: u64) -> u64 {
    elements * std::mem::size_of::<u16>() as u64
}

fn encode_rms_at(
    command: &CommandBufferRef,
    pipelines: &Pipelines,
    input: &BufferRef,
    weight: &BufferRef,
    weight_offset: u64,
    output: &BufferRef,
) {
    let shape = model_shape();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipelines.rms);
    encoder.set_buffer(0, Some(input), 0);
    encoder.set_buffer(1, Some(weight), weight_offset);
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

fn encode_candidate_ffn(
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
            * u64::from(routing::MOE_GATE_UP),
    );
    let down_offset = half_bytes(
        u64::from(layer)
            * u64::from(routing::MOE_EXPERTS)
            * u64::from(routing::ROUTED_EXPERT_WIDTH)
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

fn encode_candidate_objective_unfused(
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
            encode_rms_at(
                command,
                pipelines,
                input,
                weights.buffer,
                weights.attention_norm + norm_offset,
                &buffers.normalized,
            );
            encode_tensorops_groups_at(
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
            encode_tensorops_groups_at(
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
            encode_rms_at(
                command,
                pipelines,
                &buffers.attention_state,
                weights.buffer,
                weights.ffn_norm + norm_offset,
                &buffers.normalized,
            );
            encode_candidate_ffn(
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
            encode_tensorops_groups_at(
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
            encode_tensorops_groups_at(
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
    encode_rms_at(
        command,
        pipelines,
        input,
        weights.buffer,
        weights.final_norm,
        &buffers.normalized,
    );
    encode_tensorops_groups_at(
        command,
        &tensorops.readout,
        &[
            (&buffers.normalized, 0),
            (weights.buffer, weights.readout),
            (&buffers.logits, 0),
        ],
        MTLSize {
            width: u64::from(VOCAB / 64),
            height: u64::from(ROWS / 128),
            depth: 1,
        },
    )?;
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipelines.cross_entropy);
    encoder.set_buffer(0, Some(&buffers.logits), 0);
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

fn encode_candidate_objective_fused(
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

    let layer_norm_offset = |layer: u32| half_bytes(u64::from(layer) * u64::from(WIDTH));
    encode_rms_at(
        command,
        pipelines,
        &buffers.input,
        weights.buffer,
        weights.attention_norm,
        &buffers.normalized,
    );
    let mut input: &BufferRef = &buffers.input;
    for pass in 0..FEEDBACK_PASSES {
        for layer in 0..MODEL_LAYERS {
            let qkv_offset = half_bytes(u64::from(layer) * u64::from(WIDTH) * u64::from(QKV_WIDTH));
            let output_offset = half_bytes(u64::from(layer) * u64::from(WIDTH) * u64::from(WIDTH));
            encode_tensorops_groups_at(
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
            encode_tensorops_groups_at(
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
            let shape = model_shape();
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipelines.residual_rms);
            encoder.set_buffer(0, Some(input), 0);
            encoder.set_buffer(1, Some(&buffers.projected), 0);
            encoder.set_buffer(
                2,
                Some(weights.buffer),
                weights.ffn_norm + layer_norm_offset(layer),
            );
            encoder.set_buffer(3, Some(&buffers.attention_state), 0);
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
            encoder.end_encoding();
            let next_norm: (&BufferRef, u64) = if layer + 1 < MODEL_LAYERS {
                (
                    weights.buffer,
                    weights.attention_norm + layer_norm_offset(layer + 1),
                )
            } else if pass + 1 < FEEDBACK_PASSES {
                (&buffers.unit_norm, 0)
            } else {
                (weights.buffer, weights.final_norm)
            };
            encode_candidate_ffn(
                command,
                pipelines,
                &buffers.normalized,
                &buffers.attention_state,
                layer,
                buffers,
                weights,
                Some(next_norm),
            )?;
            input = &buffers.output;
        }
        if pass + 1 < FEEDBACK_PASSES {
            encode_tensorops_groups_at(
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
            encode_tensorops_groups_at(
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
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipelines.feedback_fuse_rms);
            encoder.set_buffer(0, Some(&buffers.feedback_state), 0);
            encoder.set_buffer(1, Some(&buffers.feedback_gate), 0);
            encoder.set_buffer(2, Some(weights.buffer), weights.attention_norm);
            encoder.set_buffer(3, Some(&buffers.feedback), 0);
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
            encoder.end_encoding();
            input = &buffers.feedback;
        }
    }
    encode_tensorops_groups_at(
        command,
        &tensorops.readout,
        &[
            (&buffers.normalized, 0),
            (weights.buffer, weights.readout),
            (&buffers.logits, 0),
        ],
        MTLSize {
            width: u64::from(VOCAB / 64),
            height: u64::from(ROWS / 128),
            depth: 1,
        },
    )?;
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipelines.cross_entropy);
    encoder.set_buffer(0, Some(&buffers.logits), 0);
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

fn complete_committed(command: &CommandBufferRef) -> Result<f64, String> {
    command.wait_until_completed();
    if command.status() != MTLCommandBufferStatus::Completed {
        return Err(format!(
            "candidate objective command failed: {:?}",
            command.status()
        ));
    }
    gpu_seconds(command).ok_or("Metal did not report candidate objective GPU timing".into())
}

const OBJECTIVE_BLOCK_TOKENS: usize = 128;
const OBJECTIVE_BLOCKS: usize = ROWS as usize / OBJECTIVE_BLOCK_TOKENS;

struct ObjectiveStats {
    reward: f32,
    variance: f32,
    sequence_nlls: [f32; 2],
    block_nlls: [f32; OBJECTIVE_BLOCKS],
}

fn mean_variance(samples: &[f32]) -> (f32, f32) {
    let mean = samples.iter().sum::<f32>() / samples.len() as f32;
    let centered = samples
        .iter()
        .map(|sample| (sample - mean).powi(2))
        .sum::<f32>();
    let variance = centered / (samples.len() * (samples.len() - 1)) as f32;
    (mean, variance)
}

fn sequence_objective(buffers: &Buffers) -> Result<ObjectiveStats, String> {
    let values = unsafe {
        std::slice::from_raw_parts(
            buffers.sequence_scores.contents().cast::<f32>(),
            BATCH as usize,
        )
    };
    let scores = [values[0], values[1]];
    if scores.iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "candidate objective produced invalid sequence scores: {scores:?}"
        ));
    }
    let losses = unsafe {
        std::slice::from_raw_parts(buffers.losses.contents().cast::<f32>(), ROWS as usize)
    };
    let mask = unsafe {
        std::slice::from_raw_parts(buffers.score_mask.contents().cast::<u8>(), ROWS as usize)
    };
    let mut block_nlls = [0.0f32; OBJECTIVE_BLOCKS];
    for (block, output) in block_nlls.iter_mut().enumerate() {
        let start = block * OBJECTIVE_BLOCK_TOKENS;
        let end = start + OBJECTIVE_BLOCK_TOKENS;
        let mut sum = 0.0f32;
        let mut count = 0usize;
        for index in start..end {
            if mask[index] != 0 {
                let loss = losses[index];
                if !loss.is_finite() {
                    return Err(format!(
                        "candidate objective produced invalid token loss at {index}"
                    ));
                }
                sum += loss;
                count += 1;
            }
        }
        if count == 0 {
            return Err(format!(
                "candidate objective block {block} has no scored tokens"
            ));
        }
        *output = sum / count as f32;
    }
    let mean = (scores[0] + scores[1]) * 0.5;
    let (_, variance) = mean_variance(&block_nlls);
    Ok(ObjectiveStats {
        reward: -mean,
        variance,
        sequence_nlls: scores,
        block_nlls,
    })
}

fn paired_improvement(candidate: &ObjectiveStats, incumbent: &ObjectiveStats) -> (f32, f32) {
    let samples = std::array::from_fn::<_, OBJECTIVE_BLOCKS, _>(|index| {
        incumbent.block_nlls[index] - candidate.block_nlls[index]
    });
    let mean = ((incumbent.sequence_nlls[0] - candidate.sequence_nlls[0])
        + (incumbent.sequence_nlls[1] - candidate.sequence_nlls[1]))
        * 0.5;
    let (_, variance) = mean_variance(&samples);
    (mean, variance)
}

#[cfg(test)]
mod objective_tests {
    use super::*;

    #[test]
    fn block_variance_is_variance_of_the_mean() {
        let (mean, variance) = mean_variance(&[1.0, 3.0]);
        assert_eq!(mean, 2.0);
        assert_eq!(variance, 1.0);
    }

    #[test]
    fn paired_improvement_uses_exact_sequence_mean_and_block_noise() {
        let candidate = ObjectiveStats {
            reward: -1.0,
            variance: 0.0,
            sequence_nlls: [1.0, 1.0],
            block_nlls: [1.0; OBJECTIVE_BLOCKS],
        };
        let incumbent = ObjectiveStats {
            reward: -3.0,
            variance: 0.0,
            sequence_nlls: [2.0, 4.0],
            block_nlls: [2.0; OBJECTIVE_BLOCKS],
        };
        let (improvement, variance) = paired_improvement(&candidate, &incumbent);
        assert_eq!(improvement, 2.0);
        assert_eq!(variance, 0.0);
    }
}

fn reliability_json(state: crate::ReliabilityTelemetry) -> serde_json::Value {
    serde_json::json!({
        "action": format!("{:?}", state.action),
        "concordance": state.concordance,
        "coverage": state.coverage,
        "posterior_mean": state.reliability_mean,
        "posterior_lower": state.reliability_lower,
        "evidence": state.evidence,
        "progress": state.progress,
        "center_radius": state.center_radius,
        "normalized_step": state.normalized_step,
        "conversion": state.conversion,
        "escape_remaining": state.escape_remaining,
        "escapes": state.escapes,
    })
}

#[allow(clippy::too_many_arguments)]
fn controller_record(
    round: u32,
    initializing: bool,
    proposal: &Proposals,
    decision: NoisyDecision,
    controller: ControllerInfo,
    reliability: Option<crate::ReliabilityTelemetry>,
    wall_seconds: f64,
    controller_seconds: f64,
) -> serde_json::Value {
    serde_json::json!({
        "schema": "ennx.reliability_controller.v1",
        "round": round,
        "phase": if initializing { "initialization" } else { "guided" },
        "candidate_index": proposal.index,
        "candidate_seed": proposal.seed,
        "nominal_radius": proposal.length,
        "predicted_mean": proposal.predicted_mean,
        "predicted_standard_error": proposal.predicted_standard_error,
        "incumbent_mean": proposal.incumbent_mean,
        "observed_improvement": decision.improvement,
        "acceptance_threshold": decision.threshold,
        "agreement_ratio": decision.agreement_ratio,
        "trust_outcome": format!("{:?}", decision.trust_outcome),
        "accepted": decision.accepted,
        "next_length": controller.length,
        "reliability": reliability.map(reliability_json),
        "wall_seconds": wall_seconds,
        "controller_seconds": controller_seconds,
    })
}

fn log_reliability(round: u32, state: crate::ReliabilityTelemetry) {
    eprintln!(
        "TURBO_ENN_RELIABILITY round={round} action={:?} concordance={} coverage={:.6} posterior_mean={:.6} posterior_lower={:.6} evidence={:.6} progress={:.6} center_radius={} normalized_step={} conversion={} escape_remaining={} escapes={} next_length={:.9}",
        state.action,
        state
            .concordance
            .map_or_else(|| "none".into(), |value| format!("{value:.6}")),
        state.coverage,
        state.reliability_mean,
        state.reliability_lower,
        state.evidence,
        state.progress,
        state
            .center_radius
            .map_or_else(|| "none".into(), |value| format!("{value:.9}")),
        state
            .normalized_step
            .map_or_else(|| "none".into(), |value| format!("{value:.6}")),
        state
            .conversion
            .map_or_else(|| "none".into(), |value| format!("{value:.6}")),
        state.escape_remaining,
        state.escapes,
        state.length,
    );
}

fn benchmark_actual_bo(
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
    control: BoControl,
) -> Result<ActualBoResult, String> {
    if let Some(dataset) = dataset {
        if dataset.batches() == 0 {
            return Err("pretraining dataset has no batches".into());
        }
        load_pretraining_batch(buffers, dataset, 0)?;
    }
    let (mut search, mut updates) =
        weights.search_shaped(control.perturbation, control.length, control.shape)?;
    search.configure_implicit_enn(control.enn)?;
    if let Some(config) = control.reliability {
        search.configure_reliability_controller(config)?;
    }
    search.set_profiling(trace);
    let info = search.controller_info()?;
    let blocks = weights
        .tensors()
        .iter()
        .map(|(_, buffer, len)| buffer.length() as usize / 2 / len)
        .sum::<usize>();
    eprintln!(
        "TURBO_ENN_FULL_SPACE parameters={} blocks={blocks} context={CONTEXT} batch={BATCH} history_capacity={HISTORY_CAPACITY} storage=fp16 noise={} shape={} distance=resident_exact_nonresident_approximate",
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
    let unfused_scores = if check_fusion {
        let command = runtime.queue.new_command_buffer();
        encode_candidate_objective_unfused(
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
    encode_candidate_objective_fused(
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
        if control.paired_objective {
            0.0
        } else {
            initial_value
        },
        if control.paired_objective {
            0.0
        } else {
            initial_variance
        },
    )?;
    eprintln!(
        "TURBO_ENN_ACTUAL_INITIAL reward={initial_value:.9} variance={initial_variance:.9} sequence_nlls={initial_scores:?}"
    );

    let mut wall = Vec::with_capacity(rounds as usize);
    let mut gpu = Vec::with_capacity(rounds as usize);
    let mut accepted = 0u32;
    let mut controller_seconds = Vec::with_capacity(rounds as usize);
    let mut controller_records = Vec::with_capacity(rounds as usize);
    let mut last_family_metric = None;
    let loop_start = Instant::now();
    for step in 0..rounds {
        let start = Instant::now();
        if let Some(dataset) = dataset {
            load_pretraining_batch(buffers, dataset, step % dataset.batches())?;
        }
        let ask_start = Instant::now();
        search.compact_implicit_history()?;
        let history = search.history_len()?;
        let initializing = history < control.enn.ask.neighbors;
        if !initializing {
            if let Some((metric, scales)) = search.family_shape() {
                if last_family_metric != Some(metric) {
                    eprintln!(
                        "TURBO_ENN_FAMILY round={} groups=experts,projections,routers_feedback,norms metric={metric:?} proposal={scales:?}",
                        step + 1
                    );
                    last_family_metric = Some(metric);
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
            search.begin_ask(1, 4, proposal_seed, ask)?
        };
        let candidate_weights = weights.row(&row)?;
        let command = runtime.queue.new_command_buffer();
        encode_candidate_objective_fused(
            command,
            pipelines,
            tensorops,
            pisa1,
            buffers,
            candidate_weights,
        )?;
        command.commit();
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
        let changes = search.describe(&proposal)?.remove(0).3;
        let changed = changes.iter().map(|block| block.0).sum::<u64>();
        let mut gpu_seconds = complete_committed(command)?;
        let candidate_objective = sequence_objective(buffers)?;
        let value = candidate_objective.reward;
        let variance = candidate_objective.variance;
        let scores = candidate_objective.sequence_nlls;
        let incumbent_value = search.best()?;
        let incumbent_variance = search.best_variance()?;
        let (optimizer_value, optimizer_variance, improvement_variance, incumbent_scores) =
            if control.paired_objective {
                let base = search.base_buffer();
                let incumbent_weights = weights.row(&base)?;
                let command = runtime.queue.new_command_buffer();
                encode_candidate_objective_fused(
                    command,
                    pipelines,
                    tensorops,
                    pisa1,
                    buffers,
                    incumbent_weights,
                )?;
                let incumbent_gpu_seconds = complete(command)?;
                gpu_seconds += incumbent_gpu_seconds;
                let incumbent_objective = sequence_objective(buffers)?;
                let incumbent_scores = incumbent_objective.sequence_nlls;
                let (improvement, improvement_variance) =
                    paired_improvement(&candidate_objective, &incumbent_objective);
                (
                    incumbent_value + improvement,
                    incumbent_variance + improvement_variance,
                    improvement_variance,
                    Some(incumbent_scores),
                )
            } else {
                (value, variance, variance, None)
            };
        let tell_start = Instant::now();
        let decision = if initializing {
            search.tell_initial(&proposal, optimizer_value, optimizer_variance)?
        } else if control.paired_objective {
            search.tell_paired_model_aware(
                &proposal,
                optimizer_value,
                optimizer_variance,
                incumbent_value,
                incumbent_variance,
                improvement_variance,
            )?
        } else {
            search.tell_model_aware(&proposal, value, variance)?
        };
        let synced = search.sync()?;
        if synced != vec![decision.accepted] {
            return Err("actual BO synchronization changed".into());
        }
        let tell_seconds = tell_start.elapsed().as_secs_f64();
        controller_seconds.push(ask_seconds + tell_seconds);
        accepted += u32::from(decision.accepted);
        updates.push_scaled(
            step + 1,
            proposal.seed,
            proposal.length,
            decision.accepted,
            changes,
            proposal.block_scales.clone(),
        )?;
        let wall_seconds = start.elapsed().as_secs_f64();
        gpu.push(gpu_seconds);
        wall.push(wall_seconds);
        let fitted = search.fitted_enn();
        let reliability = search.reliability_info()?;
        let controller = search.controller_info()?;
        controller_records.push(controller_record(
            step + 1,
            initializing,
            &proposal,
            decision,
            controller,
            reliability,
            wall_seconds,
            ask_seconds + tell_seconds,
        ));
        eprintln!(
            "TURBO_ENN_ACTUAL_ROUND round={} total={rounds} phase={} logical_history={} resident_history={HISTORY_CAPACITY} wall_seconds={wall_seconds:.6} scorer_gpu_seconds={gpu_seconds:.6} ask_seconds={ask_seconds:.6} tell_seconds={tell_seconds:.6} changed_weights={changed} parameters={FULL_PARAMETERS} proposal_radius={:.9} reward={value:.9} variance={variance:.9} sequence_nlls={scores:?} incumbent_sequence_nlls={incumbent_scores:?} optimizer_value={optimizer_value:.9} paired_improvement={:.9} paired_variance={improvement_variance:.9} accepted={} radius={:.6} fitted_enn={fitted:?}",
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
    }
    let loop_seconds = loop_start.elapsed().as_secs_f64();
    eprintln!(
        "TURBO_ENN_LOOP rounds={rounds} elapsed_seconds={loop_seconds:.9} rounds_per_second={:.6}",
        f64::from(rounds) / loop_seconds,
    );
    wall.sort_by(f64::total_cmp);
    gpu.sort_by(f64::total_cmp);
    Ok(ActualBoResult {
        parameters: FULL_PARAMETERS,
        loop_seconds,
        median_wall_seconds: wall[wall.len() / 2],
        min_wall_seconds: wall[0],
        max_wall_seconds: wall[wall.len() - 1],
        median_gpu_seconds: gpu[gpu.len() / 2],
        accepted,
        controller_seconds,
        controller_records,
        updates,
    })
}

fn benchmark_sustained_model(
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
        encode_sustained_model(command, pipelines, tensorops, pisa1, buffers)?;
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

fn benchmark_sustained_mps(
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
                Matrix::half(&buffers.grouped, ROWS_PER_EXPERT, WIDTH).layout(
                    EXPERTS,
                    u64::from(ROWS_PER_EXPERT) * u64::from(WIDTH),
                    0,
                ),
                Matrix::half(&buffers.materialized_gate_up, WIDTH, GATE_UP).layout(
                    EXPERTS,
                    u64::from(WIDTH) * u64::from(GATE_UP),
                    0,
                ),
                Matrix::half(&buffers.gate_up, ROWS_PER_EXPERT, GATE_UP).layout(
                    EXPERTS,
                    u64::from(ROWS_PER_EXPERT) * u64::from(GATE_UP),
                    0,
                ),
                false,
                1.0,
            )?;
            encode(pipelines, buffers, command, Some(Stage::Swiglu));
            matmul.encode(
                &runtime.device,
                command,
                Matrix::half(&buffers.activation, ROWS_PER_EXPERT, EXPERT_WIDTH).layout(
                    EXPERTS,
                    u64::from(ROWS_PER_EXPERT) * u64::from(EXPERT_WIDTH),
                    0,
                ),
                Matrix::half(&buffers.materialized_down, EXPERT_WIDTH, WIDTH).layout(
                    EXPERTS,
                    u64::from(EXPERT_WIDTH) * u64::from(WIDTH),
                    0,
                ),
                Matrix::half(&buffers.down, ROWS_PER_EXPERT, WIDTH).layout(
                    EXPERTS,
                    u64::from(ROWS_PER_EXPERT) * u64::from(WIDTH),
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

fn benchmark_sustained_breakdown(
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
        encode_sustained_materialization(command, tensorops, buffers);
        let materialization = complete(command)?;

        let command = runtime.queue.new_command_buffer();
        for _ in 0..MODEL_LAYERS * FEEDBACK_PASSES {
            encode_sustained_qkv(command, tensorops, &buffers.input, buffers)?;
            encode_sustained_output_projection(command, tensorops, &buffers.input, buffers)?;
        }
        let projections = complete(command)?;

        let command = runtime.queue.new_command_buffer();
        for _ in 0..MODEL_LAYERS * FEEDBACK_PASSES {
            pisa1.encode_layer(command, &buffers.qkv);
        }
        let attention = complete(command)?;

        let command = runtime.queue.new_command_buffer();
        for _ in 0..MODEL_LAYERS * FEEDBACK_PASSES {
            encode_sustained_ffn(command, pipelines, tensorops, buffers)?;
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

fn benchmark_projection_pair(
    runtime: &Runtime,
    tensorops: &TensorOpsPipelines,
    buffers: &Buffers,
    rounds: u32,
) -> Result<(f64, f64), String> {
    let mut gpu = Vec::with_capacity(rounds as usize);
    let mut wall = Vec::with_capacity(rounds as usize);
    for iteration in 0..rounds + 3 {
        let command = runtime.queue.new_command_buffer();
        encode_projection_pair(command, tensorops, buffers)?;
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

fn mps_projection_reference(runtime: &Runtime, buffers: &Buffers) -> Result<f64, String> {
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

fn decode_half(bits: u16) -> f64 {
    let exponent = (bits >> 10) & 31;
    let fraction = f64::from(bits & 1023);
    let magnitude = match exponent {
        0 => fraction * 2.0f64.powi(-24),
        31 => f64::NAN,
        _ => (1024.0 + fraction) * 2.0f64.powi(i32::from(exponent) - 25),
    };
    if bits & 0x8000 == 0 {
        magnitude
    } else {
        -magnitude
    }
}

fn compare_half(
    reference: &BufferRef,
    candidate: &BufferRef,
    elements: usize,
    label: &str,
) -> Result<f64, String> {
    let read = |buffer: &BufferRef| unsafe {
        std::slice::from_raw_parts(buffer.contents().cast::<u16>(), elements + 64)
    };
    let reference = read(reference);
    let candidate = read(candidate);
    if !reference[elements..].iter().all(|&bits| bits == 0x7e00)
        || !candidate[elements..].iter().all(|&bits| bits == 0x7e00)
    {
        return Err(format!("{label} overwrote its output canary"));
    }
    let mut maximum = 0.0f64;
    let mut maximum_index = 0usize;
    let mut maximum_reference = 0.0f64;
    let mut maximum_candidate = 0.0f64;
    for index in 0..elements {
        let left = decode_half(reference[index]);
        let right = decode_half(candidate[index]);
        if !left.is_finite() || !right.is_finite() {
            return Err(format!("{label} produced a non-finite output at {index}"));
        }
        let error = (left - right).abs();
        if error > maximum {
            maximum = error;
            maximum_index = index;
            maximum_reference = left;
            maximum_candidate = right;
        }
    }
    eprintln!(
        "TURBO_ENN_MOE_PARITY operation={label} max_abs_error={maximum:.9} index={maximum_index} reference={maximum_reference:.9} candidate={maximum_candidate:.9}"
    );
    Ok(maximum)
}

fn matmul_element(
    input: &BufferRef,
    weights: &BufferRef,
    row: u32,
    column: u32,
    input_width: u32,
    output_width: u32,
) -> f64 {
    let expert = row / ROWS_PER_EXPERT;
    let input = unsafe {
        std::slice::from_raw_parts(
            input.contents().cast::<u16>(),
            (ROWS * input_width) as usize,
        )
    };
    let weights = unsafe {
        std::slice::from_raw_parts(
            weights.contents().cast::<u16>(),
            (EXPERTS * input_width * output_width) as usize,
        )
    };
    let mut sum = 0.0f32;
    for k in 0..input_width {
        let left = decode_half(input[(row * input_width + k) as usize]) as f32;
        let weight = ((expert * input_width + k) * output_width + column) as usize;
        sum += left * decode_half(weights[weight]) as f32;
    }
    f64::from(sum)
}

fn dense_matmul_element(
    input: &BufferRef,
    weights: &BufferRef,
    row: u32,
    column: u32,
    input_width: u32,
    output_width: u32,
) -> f64 {
    let input = unsafe {
        std::slice::from_raw_parts(
            input.contents().cast::<u16>(),
            (ROWS * input_width) as usize,
        )
    };
    let weights = unsafe {
        std::slice::from_raw_parts(
            weights.contents().cast::<u16>(),
            (input_width * output_width) as usize,
        )
    };
    let mut sum = 0.0f32;
    for k in 0..input_width {
        let left = decode_half(input[(row * input_width + k) as usize]) as f32;
        sum += left * decode_half(weights[(k * output_width + column) as usize]) as f32;
    }
    f64::from(sum)
}

fn buffer_element(buffer: &BufferRef, row: u32, column: u32, width: u32) -> f64 {
    let values = unsafe {
        std::slice::from_raw_parts(buffer.contents().cast::<u16>(), (ROWS * width) as usize)
    };
    decode_half(values[(row * width + column) as usize])
}

fn validate_tail(buffers: &Buffers) -> Result<f64, String> {
    let state = buffer_element(&buffers.feedback_state, 0, 0, WIDTH);
    let gate = buffer_element(&buffers.feedback_gate, 0, 0, WIDTH);
    let feedback = buffer_element(&buffers.feedback, 0, 0, WIDTH);
    let expected_feedback = state / (1.0 + (-gate).exp());
    let feedback_error = (feedback - expected_feedback).abs();

    let logits = unsafe {
        std::slice::from_raw_parts(buffers.logits.contents().cast::<u16>(), VOCAB as usize)
    };
    let maximum = logits
        .iter()
        .map(|&bits| decode_half(bits))
        .fold(f64::NEG_INFINITY, f64::max);
    let total = logits
        .iter()
        .map(|&bits| (decode_half(bits) - maximum).exp())
        .sum::<f64>();
    let labels = unsafe {
        std::slice::from_raw_parts(buffers.labels.contents().cast::<u32>(), ROWS as usize)
    };
    let expected_loss = total.ln() + maximum - decode_half(logits[labels[0] as usize]);
    let losses = unsafe {
        std::slice::from_raw_parts(buffers.losses.contents().cast::<f32>(), ROWS as usize + 64)
    };
    if losses[..ROWS as usize]
        .iter()
        .any(|value| !value.is_finite())
        || losses[ROWS as usize..].iter().any(|value| !value.is_nan())
    {
        return Err(
            "feedback/readout tail produced non-finite loss or overwrote its canary".into(),
        );
    }
    let loss_error = (f64::from(losses[0]) - expected_loss).abs();
    let maximum_error = feedback_error.max(loss_error);
    if feedback_error > 0.002 || loss_error > 0.002 {
        return Err(format!(
            "feedback/readout parity failed: feedback={feedback_error:.9}, loss={loss_error:.9}"
        ));
    }
    eprintln!(
        "TURBO_ENN_TAIL_PARITY feedback_max_abs_error={feedback_error:.9} loss_max_abs_error={loss_error:.9}"
    );
    Ok(maximum_error)
}

fn complete_prematerialized_layer(
    runtime: &Runtime,
    pipelines: &Pipelines,
    tensorops: &TensorOpsPipelines,
    buffers: &Buffers,
) -> Result<f64, String> {
    let command = runtime.queue.new_command_buffer();
    encode(pipelines, buffers, command, Some(Stage::Gate));
    encode(pipelines, buffers, command, Some(Stage::Group));
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
    encode(pipelines, buffers, command, Some(Stage::Ungroup));
    complete(command)
}

pub fn run_pretrain(
    run: &crate::config::ConfigOverrides,
    dataset_path: &std::path::Path,
    rep: u32,
) -> Result<ActualBoResult, String> {
    run.validate_round_study()?;
    if rep >= run.reps() {
        return Err(format!(
            "pretraining repetition {rep} is outside configured reps {}",
            run.reps()
        ));
    }
    let rounds = run.rounds();
    if rounds == 0 {
        return Err("pretraining requires positive rounds".into());
    }
    let proposal_seed = run.proposal_seed_for_rep(rep);
    let acquisition_seed = run.acquisition_seed_for_rep(rep);
    let control = BoControl {
        length: run.length(),
        enn: run.resident_enn(acquisition_seed)?,
        proposal_seed,
        acquisition_seed,
        perturbation: run.perturbation(),
        shape: run
            .trust_region_shape
            .unwrap_or(crate::config::TrustRegionShape::TensorFamilyStatic),
        paired_objective: true,
        reliability: run.reliability_controller(),
    };
    let started = Instant::now();
    eprintln!(
        "TURBO_ENN_REP rep={} reps={} proposal_seed={proposal_seed} acquisition_seed={acquisition_seed}",
        rep + 1,
        run.reps()
    );
    eprintln!("[tune] load dataset and compile pipelines");
    let dataset = crate::pretrain_data::PretrainDataset::load(dataset_path)?;
    metal::objc::rc::autoreleasepool(|| {
        let runtime = Runtime::shared()?;
        let pipelines = Pipelines::new(&runtime)?;
        let tensorops = TensorOpsPipelines::new(&runtime)?;
        let buffers = Buffers::new(&runtime);
        let pisa1 = Pisa1::new(&runtime)?;
        eprintln!(
            "[tune] initialize model and controller; score initial model | elapsed {:.1}s",
            started.elapsed().as_secs_f64()
        );
        let weights = CandidateWeights::new(&runtime);
        let result = benchmark_actual_bo(
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
            control,
        )?;
        eprintln!(
            "[tune] BO rounds complete; writing results | elapsed {:.1}s",
            started.elapsed().as_secs_f64()
        );
        Ok(result)
    })
}

pub fn run_grouped_moe_probe(rounds: u32, target_ms: u32) -> Result<GroupedMoeProbe, String> {
    run_grouped_moe_probe_with_dataset(rounds, target_ms, None)
}

pub fn run_grouped_moe_probe_with_dataset(
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
        let materialize_gate_up_gpu_seconds = benchmark_materialization(
            &runtime,
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
        let materialize_down_gpu_seconds = benchmark_materialization(
            &runtime,
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
        complete_prematerialized_layer(&runtime, &pipelines, &tensorops, &buffers)?;
        let activation = runtime.buffer_with(&vec![0x7e00u16; (ROWS * EXPERT_WIDTH) as usize + 64]);
        let command = runtime.queue.new_command_buffer();
        encode_tensorops_groups(
            command,
            &tensorops.gate_activation,
            &[&buffers.grouped, &buffers.materialized_gate_up, &activation],
            MTLSize {
                width: u64::from(EXPERT_WIDTH.div_ceil(64)),
                height: u64::from(ROWS_PER_EXPERT / 128),
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
        let mps_reference_gpu_seconds = mps_reference(&runtime, &buffers)?;
        let (projections_gpu_seconds, projections_wall_seconds) =
            benchmark_projection_pair(&runtime, &tensorops, &buffers, rounds)?;
        let mps_projection_reference_gpu_seconds = mps_projection_reference(&runtime, &buffers)?;
        progress("projection checks complete; PISA parity and timing");
        let pisa1 = run_pisa1_probe(&runtime, &buffers.qkv, rounds)?;
        progress("PISA checks complete; FFN component timing");
        let sustained_pisa1 = Pisa1::new(&runtime)?;
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
            dense_matmul_element(&buffers.input, &buffers.qkv_weights, 0, 0, WIDTH, QKV_WIDTH),
            buffer_element(&buffers.mps_qkv, 0, 0, QKV_WIDTH),
            buffer_element(&buffers.qkv, 0, 0, QKV_WIDTH),
        );
        eprintln!(
            "TURBO_ENN_PROJECTION_PARITY_SAMPLE operation=output row=0 column=0 cpu={:.9} mps={:.9} tensorops={:.9}",
            dense_matmul_element(
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
        let mut stage_gpu = [0.0; 4];
        for (stage_index, (stage, name)) in Stage::ALL.into_iter().enumerate() {
            let mut seconds = Vec::with_capacity(3);
            for _ in 0..3 {
                let command = runtime.queue.new_command_buffer();
                encode(&pipelines, &buffers, command, Some(stage));
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
            complete_prematerialized_layer(&runtime, &pipelines, &tensorops, &buffers)?;
        }
        let mut prematerialized_layer_gpu = Vec::with_capacity(rounds as usize);
        let mut prematerialized_layer_wall = Vec::with_capacity(rounds as usize);
        for _ in 0..rounds {
            let start = Instant::now();
            prematerialized_layer_gpu.push(complete_prematerialized_layer(
                &runtime, &pipelines, &tensorops, &buffers,
            )?);
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
        let objective_flops =
            ffn_objective_flops + projection_objective_flops + pisa1_objective_flops;
        let feedback_flops = 4u64 * u64::from(ROWS) * u64::from(WIDTH) * u64::from(WIDTH);
        let readout_flops = 2u64 * u64::from(ROWS) * u64::from(WIDTH) * u64::from(VOCAB);
        let tail_flops = feedback_flops + readout_flops;
        let complete_objective_flops = objective_flops + tail_flops;
        let projected_ffn_and_projections_seconds = projected_ffn_seconds
            + projections_wall_seconds * f64::from(MODEL_LAYERS * FEEDBACK_PASSES);
        let projected_measured_model_seconds = projected_ffn_and_projections_seconds
            + pisa1.layer_wall_seconds * f64::from(MODEL_LAYERS * FEEDBACK_PASSES);
        progress("component timing complete; sustained model diagnostic");
        let (
            sustained_model_gpu_seconds,
            sustained_model_wall_seconds,
            sustained_model_min_wall_seconds,
            sustained_model_max_wall_seconds,
        ) = benchmark_sustained_model(
            &runtime,
            &pipelines,
            &tensorops,
            &sustained_pisa1,
            &buffers,
            rounds,
        )?;
        progress("sustained model complete; component breakdown and MPS references");
        let [
            sustained_materialization_gpu_seconds,
            sustained_projections_gpu_seconds,
            sustained_pisa1_gpu_seconds,
            sustained_ffn_gpu_seconds,
        ] = benchmark_sustained_breakdown(
            &runtime,
            &pipelines,
            &tensorops,
            &sustained_pisa1,
            &buffers,
        )?;
        let [
            (sustained_mps_projections_gpu_seconds, sustained_mps_projections_wall_seconds),
            (sustained_mps_ffn_gpu_seconds, sustained_mps_ffn_wall_seconds),
        ] = benchmark_sustained_mps(&runtime, &pipelines, &buffers)?;
        let (tail_gpu_seconds, tail_wall_seconds) =
            benchmark_tail(&runtime, &pipelines, &tensorops, &buffers)?;
        progress("reference checks complete; complete envelope diagnostic");
        let (
            complete_envelope_gpu_seconds,
            complete_envelope_wall_seconds,
            complete_envelope_min_wall_seconds,
            complete_envelope_max_wall_seconds,
        ) = benchmark_complete_envelope(
            &runtime,
            &pipelines,
            &tensorops,
            &sustained_pisa1,
            &buffers,
            rounds,
        )?;
        let tail_max_abs_error = validate_tail(&buffers)?;
        progress("diagnostics complete; allocate candidate weights and check initial objective");
        let candidate_weights = CandidateWeights::new(&runtime);
        let actual_bo = benchmark_actual_bo(
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
            BoControl::diagnostic(crate::Perturbation::Gaussian)?,
        )?;
        let mut controller_seconds = actual_bo.controller_seconds;
        controller_seconds.sort_by(f64::total_cmp);
        let controller_median_wall_seconds = controller_seconds[controller_seconds.len() / 2];
        let controller_min_wall_seconds = controller_seconds[0];
        let controller_max_wall_seconds = controller_seconds[controller_seconds.len() - 1];
        progress("BO rounds complete; writing results");
        let target_seconds = f64::from(target_ms) / 1000.0;
        let result = GroupedMoeProbe {
            updates: actual_bo.updates,
            parameters: FULL_PARAMETERS,
            routing_gpu_seconds: stage_gpu[0] + stage_gpu[1],
            activation_gpu_seconds: stage_gpu[2],
            residual_gpu_seconds: stage_gpu[3],
            materialize_gate_up_gpu_seconds,
            materialize_down_gpu_seconds,
            projections_gpu_seconds,
            projections_wall_seconds,
            pisa1_pyramid_gpu_seconds: pisa1.pyramid_gpu_seconds,
            pisa1_selection_gpu_seconds: pisa1.selection_gpu_seconds,
            pisa1_attention_gpu_seconds: pisa1.attention_gpu_seconds,
            pisa1_layer_gpu_seconds: pisa1.layer_gpu_seconds,
            pisa1_layer_wall_seconds: pisa1.layer_wall_seconds,
            layer_gpu_seconds: prematerialized_layer_gpu_seconds,
            layer_wall_seconds: prematerialized_layer_wall_seconds,
            projected_ffn_seconds,
            projected_ffn_and_projections_seconds,
            projected_measured_model_seconds,
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
            controller_median_wall_seconds,
            controller_min_wall_seconds,
            controller_max_wall_seconds,
            actual_bo_median_wall_seconds: actual_bo.median_wall_seconds,
            actual_bo_min_wall_seconds: actual_bo.min_wall_seconds,
            actual_bo_max_wall_seconds: actual_bo.max_wall_seconds,
            actual_bo_median_gpu_seconds: actual_bo.median_gpu_seconds,
            actual_bo_accepted: actual_bo.accepted,
            target_seconds,
            objective_flops,
            tail_flops,
            complete_objective_flops,
            projection_objective_flops,
            pisa1_objective_flops,
            effective_tflops: objective_flops as f64 / projected_measured_model_seconds / 1e12,
            gate_up_max_abs_error: prematerialized_gate_up_max_abs_error,
            down_max_abs_error: prematerialized_down_max_abs_error,
            qkv_max_abs_error,
            output_projection_max_abs_error,
            pisa1_max_abs_error: pisa1.max_abs_error,
            tail_max_abs_error,
            meets_target: actual_bo.max_wall_seconds <= target_seconds,
        };
        eprintln!(
            "TURBO_ENN_MODEL_FLOOR rows={ROWS} width={WIDTH} query_heads={QUERY_HEADS} kv_heads={KV_HEADS} head_dim={HEAD_DIM} qkv_width={QKV_WIDTH} experts={EXPERTS} rows_per_expert={ROWS_PER_EXPERT} expert_width={EXPERT_WIDTH} routing=balanced_hash_top1 proposal=prematerialized_exact_full_rank_kronecker kernel=metal4_tensorops_m128_n64 data=deterministic_noncompressible_fp16 objective_flops={} projection_objective_flops={} pisa1_objective_flops={} routing_gpu_seconds={:.6} activation_gpu_seconds={:.6} residual_gpu_seconds={:.6} materialize_gate_up_gpu_seconds={:.6} materialize_down_gpu_seconds={:.6} projections_gpu_seconds={:.6} projections_wall_seconds={:.6} pisa1_pyramid_gpu_seconds={:.6} pisa1_selection_gpu_seconds={:.6} pisa1_attention_gpu_seconds={:.6} pisa1_layer_gpu_seconds={:.6} pisa1_layer_wall_seconds={:.6} layer_gpu_seconds={:.6} layer_wall_seconds={:.6} projected_ffn_seconds={:.6} projected_ffn_and_projections_seconds={:.6} projected_measured_model_seconds={:.6} effective_tflops={:.3} gate_up_max_abs_error={:.9} down_max_abs_error={:.9} qkv_max_abs_error={:.9} output_projection_max_abs_error={:.9} pisa1_max_abs_error={:.9} target_seconds={:.6} component_target_met={}",
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
            "TURBO_ENN_ACTUAL_BO scope=one_observation_noisy_distinct_24_layers_two_feedback_passes_pisa1_shared1_routed128_top3_moe rounds={} parameters={} history_capacity={HISTORY_CAPACITY} noise=independent_gaussian distance=full_realized_weights gpu_median_seconds={:.6} wall_median_seconds={:.6} wall_min_seconds={:.6} wall_max_seconds={:.6} accepted={} target_seconds={:.6} target_met={}",
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
        Ok(result)
    })
}
