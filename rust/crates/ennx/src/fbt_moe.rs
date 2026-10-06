//! Exact-shape grouped-MoE feasibility gate for the proposed subsecond scorer.

use crate::apple_gpu::Runtime;
use crate::fbt_pisa1::Pisa1;
pub(super) use metal::{
    Buffer, BufferRef, CommandBuffer, CommandBufferRef, ComputeCommandEncoderRef,
    ComputePipelineState, MTLCommandBufferStatus,
};
use std::time::Instant;

#[path = "fbt_moe/resources.rs"]
mod resources;
use resources::*;

#[path = "fbt_moe/buffers.rs"]
mod buffers;

#[path = "fbt_moe/dispatch.rs"]
mod dispatch;
use dispatch::*;

#[path = "fbt_moe/envelope.rs"]
mod envelope;
use envelope::*;

#[path = "fbt_moe/candidate.rs"]
mod candidate;
use candidate::*;

mod causal;

#[path = "fbt_moe/objective.rs"]
mod objective;
use objective::*;

#[path = "fbt_moe/optimization.rs"]
mod optimization;
use optimization::*;
#[path = "fbt_moe/round.rs"]
mod round;
use round::*;

mod architecture;
#[path = "fbt_moe/benchmarks.rs"]
mod benchmarks;
use benchmarks::*;

#[path = "fbt_moe/validation.rs"]
mod validation;

use architecture::ResidualArchitecture;
use validation::*;

#[path = "fbt_moe/experiment.rs"]
mod experiment;
pub use experiment::*;

#[path = "fbt_updates.rs"]
mod updates;
use updates::{Tensor, UpdateLog};

#[cfg(test)]
#[path = "fbt_ablation.rs"]
mod ablation;

#[cfg(test)]
#[path = "fbt_losstests.rs"]
mod loss_tests;

#[cfg(test)]
#[path = "fbt_moe/mhc_reference.rs"]
mod mhc_reference;

#[path = "fbt_routing.rs"]
mod routing;

#[path = "fbt_block.rs"]
mod block_decode;
#[path = "fbt_contrast.rs"]
mod code_contrast;
#[path = "code_environment.rs"]
mod code_environment;
#[path = "fbt_decode.rs"]
pub(crate) mod decode;
#[path = "fbt_diffusion.rs"]
mod diffusion;
#[path = "fbt_draft.rs"]
mod draft;
#[path = "fbt_generation/checkpoint.rs"]
mod gen_checkpoint;
#[path = "fbt_generation/finish.rs"]
mod gen_finish;
#[path = "fbt_journal.rs"]
mod gen_journal;
#[path = "fbt_generation/metrics.rs"]
mod gen_metrics;
#[path = "fbt_generation/oracle.rs"]
mod gen_oracle;
#[path = "fbt_generation/protocol.rs"]
mod gen_protocol;
#[path = "fbt_generation/reward.rs"]
mod gen_reward;
#[path = "fbt_generation/rounds.rs"]
mod gen_rounds;
#[path = "fbt_generation/setup.rs"]
mod gen_setup;
#[path = "fbt_generation.rs"]
mod generation;
#[path = "fbt_reward.rs"]
mod generation_reward;
#[path = "fbt_heldout.rs"]
mod heldout;
#[path = "fbt_readout.rs"]
mod readout;
#[path = "fbt_reconstruction.rs"]
mod reconstruction;
#[path = "fbt_scorer.rs"]
mod scorer;
#[path = "fbt_scorer/trace.rs"]
mod scorer_trace;
#[path = "fbt_block/target.rs"]
mod target;
#[path = "fbt_block/window.rs"]
mod window;
pub use generation::{context_loop, run_generated, run_generation};
use scorer::{ScorerStageTrace, objective_fused, objective_trace};

const ROWS: u32 = 8192;
const BATCH: u32 = 2;
const CONTEXT: u32 = ROWS / BATCH;
const WIDTH: u32 = 512;
const EXPERTS: u32 = 32;
const EXPERT_ROWS: u32 = ROWS / EXPERTS;
const EXPERT_WIDTH: u32 = 864;
const GATE_UP: u32 = 2 * EXPERT_WIDTH;
const QUERY_HEADS: u32 = 8;
const KV_HEADS: u32 = 1;
const HEAD_DIM: u32 = WIDTH / QUERY_HEADS;
const QKV_WIDTH: u32 = (QUERY_HEADS + 2 * KV_HEADS) * HEAD_DIM;
const ROPE_PAIRS: u32 = HEAD_DIM / 2;
const ROPE_HEADS: u32 = QUERY_HEADS + KV_HEADS;
const ROPE_BASE: f64 = 10_000.0;
const MAX_CONTEXT: u32 = 8192;
const PISA_BLOCK: u32 = 64;
const PISA_LEAVES: u32 = CONTEXT / PISA_BLOCK;
const PISA_NODES: u32 = 2 * PISA_LEAVES - 1;
const PISA_SELECTED: u32 = 8;
const MODEL_LAYERS: u32 = 5;
const FEEDBACK_PASSES: u32 = 2;
const VOCAB: u32 = 8192;
const LAYER_PARAMETERS: usize = (WIDTH * routing::ROUTED_EXPERTS) as usize
    + (WIDTH * QKV_WIDTH) as usize
    + (WIDTH * WIDTH) as usize
    + routing::FFN_PARAMETERS
    + (2 * WIDTH) as usize;
const GLOBAL_PARAMETERS: usize =
    (WIDTH * VOCAB) as usize + (2 * WIDTH * WIDTH) as usize + WIDTH as usize;
const FULL_PARAMETERS: usize = MODEL_LAYERS as usize * LAYER_PARAMETERS + GLOBAL_PARAMETERS;
const _: () = assert!(FULL_PARAMETERS == 1_047_732_224);
const MHC_STREAMS: u32 = 4;
const MHC_COEFFICIENTS: u32 = MHC_STREAMS + MHC_STREAMS * MHC_STREAMS + MHC_STREAMS;
const MHC_INPUT: u32 = MHC_STREAMS * WIDTH;
const MHC_SITEPARAMS: usize = (MHC_INPUT * MHC_COEFFICIENTS + MHC_COEFFICIENTS + 4) as usize;
const MHC_PARAMETERS: usize = MODEL_LAYERS as usize * 2 * MHC_SITEPARAMS;
const MHC_FULLPARAMS: usize = FULL_PARAMETERS - 2 * (WIDTH * WIDTH) as usize + MHC_PARAMETERS;
const _: () = assert!(MHC_SITEPARAMS == 49_180);
const _: () = assert!(MHC_FULLPARAMS == 1_047_699_736);
// Exact seeded replay needs one recyclable model-sized scratch row.  The second
// historical row was only required by the former approximate-distance path;
// its allocation is now the immutable replay-window origin instead.
const HISTORY_CAPACITY: usize = 1;

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
    quantize_gate_up_int8: ComputePipelineState,
    interleave_gate_up: ComputePipelineState,
    gate: ComputePipelineState,
    group: ComputePipelineState,
    swiglu: ComputePipelineState,
    ungroup: ComputePipelineState,
    feedback_fuse: ComputePipelineState,
    cross_entropy: ComputePipelineState,
    readout_loss_reduce: ComputePipelineState,
    readout_proposal_reduce: ComputePipelineState,
    sequence_loss: ComputePipelineState,
    embed: ComputePipelineState,
    denoise_embed: ComputePipelineState,
    denoise_reduce: ComputePipelineState,
    rms: ComputePipelineState,
    residual: ComputePipelineState,
    residual_rms: ComputePipelineState,
    feedback_fuse_rms: ComputePipelineState,
    mhc_replicate: ComputePipelineState,
    mhc_predict: ComputePipelineState,
    mhc_scale: ComputePipelineState,
    mhc_predict_rows: ComputePipelineState,
    mhc_mix_rms: ComputePipelineState,
    mhc_update: ComputePipelineState,
    mhc_mean_rms: ComputePipelineState,
    rope: ComputePipelineState,
}

struct TensorOpsPipelines {
    gate_up: ComputePipelineState,
    gate_activation: ComputePipelineState,
    down: ComputePipelineState,
    materialize_gate_up: ComputePipelineState,
    materialize_down: ComputePipelineState,
    qkv: ComputePipelineState,
    output_projection: ComputePipelineState,
    qkv_wide: ComputePipelineState,
    output_projection_wide: ComputePipelineState,
    readout: ComputePipelineState,
    readout_loss_tiles: ComputePipelineState,
    readout_proposal_tiles: ComputePipelineState,
    denoise_tiles: ComputePipelineState,
}

struct Buffers {
    fine_grained: routing::FineGrainedMoeBuffers,
    quantized_gateup: std::cell::OnceCell<Result<Buffer, String>>,
    interleaved_gateup: std::cell::OnceCell<Result<Buffer, String>>,
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
    logits: std::cell::OnceCell<Result<Buffer, String>>,
    loss_partials: Buffer,
    proposal_loss_partials: Buffer,
    labels: Buffer,
    losses: Buffer,
    score_mask: Buffer,
    sequence_scores: Buffer,
    tokens: Buffer,
    unit_norm: Buffer,
    normalized: Buffer,
    attention_state: Buffer,
    mhc_streams: [Buffer; 2],
    mhc_coefficients: Buffer,
    rope: Buffer,
}

struct CandidateWeights {
    architecture: ResidualArchitecture,
    router: Buffer,
    qkv: Buffer,
    output: Buffer,
    gate_up: Buffer,
    down: Buffer,
    attention_norm: Buffer,
    ffn_norm: Buffer,
    readout: Buffer,
    mask_embed: Buffer,
    index_query: Buffer,
    feedback_state: Buffer,
    feedback_gate: Buffer,
    mhc_attention_predictor: Buffer,
    mhc_attention_bias: Buffer,
    mhc_attention_control: Buffer,
    mhc_moe_predictor: Buffer,
    mhc_moe_bias: Buffer,
    mhc_moe_control: Buffer,
    final_norm: Buffer,
}

#[derive(Clone, Copy)]
struct CandidateRow<'a> {
    buffer: &'a BufferRef,
    architecture: ResidualArchitecture,
    router: u64,
    qkv: u64,
    output: u64,
    gate_up: u64,
    down: u64,
    attention_norm: u64,
    ffn_norm: u64,
    readout: u64,
    mask_embed: u64,
    index_query: u64,
    feedback_state: u64,
    feedback_gate: u64,
    mhc_attention_predictor: u64,
    mhc_attention_bias: u64,
    mhc_attention_control: u64,
    mhc_moe_predictor: u64,
    mhc_moe_bias: u64,
    mhc_moe_control: u64,
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
    pub learning_seconds: f64,
    pub validation: Vec<ennx_wire::json::Value>,
    controller_seconds: Vec<f64>,
    controller_records: Vec<ennx_wire::json::Value>,
    updates: UpdateLog,
}

impl ActualBoResult {
    pub fn write_validation(&self, path: &std::path::Path) -> Result<(), String> {
        ennx_wire::json::pretty_writer(
            std::fs::File::create(path).map_err(|error| error.to_string())?,
            &ennx_wire::json::json!({
                "schema":"ennx.pretrain_validation.v1", "learning_seconds":self.learning_seconds,
                "used_for_acceptance":false,
                "measurements":self.validation,
            }),
        )
        .map_err(|error| error.to_string())
    }

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
            ennx_wire::json::write_line(&mut file, record).map_err(|error| error.to_string())?;
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

fn trust_multiplier(family: &str) -> f32 {
    match family {
        "expert_gate_up" | "expert_down" => 1.0,
        "qkv" | "attention_output" | "embedding_readout" | "mask_embed" => 0.75,
        "index_query"
        | "router"
        | "feedback_state"
        | "feedback_gate"
        | "mhc_attention_predictor"
        | "mhc_moe_predictor" => 0.5,
        "mhc_attention_bias" | "mhc_attention_control" | "mhc_moe_bias" | "mhc_moe_control" => 0.25,
        "attention_norm" | "ffn_norm" | "final_norm" => 0.25,
        _ => 1.0,
    }
}

fn family_group(family: &str) -> usize {
    match family {
        "expert_gate_up" | "expert_down" => 0,
        "qkv" | "attention_output" | "embedding_readout" | "mask_embed" => 1,
        "index_query"
        | "router"
        | "feedback_state"
        | "feedback_gate"
        | "mhc_attention_predictor"
        | "mhc_moe_predictor" => 2,
        "attention_norm"
        | "ffn_norm"
        | "final_norm"
        | "mhc_attention_bias"
        | "mhc_attention_control"
        | "mhc_moe_bias"
        | "mhc_moe_control" => 3,
        _ => unreachable!("unmapped model tensor family"),
    }
}

#[derive(Clone, Copy)]
struct BoControl<'a> {
    length: crate::trust_region::TRLengthConfig,
    enn: crate::config::ResidentEnnConfig,
    proposal_seed: u64,
    acquisition_seed: u64,
    perturbation: crate::Perturbation,
    shape: crate::config::TrustRegionShape,
    paired_objective: bool,
    objective_reference: crate::config::ObjectiveReference,
    reliability: Option<crate::ReliabilityPolicy>,
    kernel_trial: Option<&'a crate::config::KernelTrial>,
    validation: Option<&'a crate::pretrain_data::PretrainDataset>,
    validation_interval: u32,
    random_selection: bool,
}

impl BoControl<'_> {
    fn diagnostic(perturbation: crate::Perturbation) -> Result<Self, String> {
        let acquisition_seed = 0xbb67_ae85_84ca_a73b;
        Ok(Self {
            length: crate::trust_region::TRLengthConfig::new(0.01, 0.0001, 0.1),
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
            objective_reference: crate::config::ObjectiveReference::MovingIncumbent,
            reliability: None,
            kernel_trial: None,
            validation: None,
            validation_interval: 1,
            random_selection: false,
        })
    }
}

pub(super) fn complete_committed(command: &CommandBufferRef) -> Result<f64, String> {
    command.wait_until_completed();
    if command.status() != MTLCommandBufferStatus::Completed {
        return Err(format!(
            "candidate objective command failed: {:?}",
            command.status()
        ));
    }
    gpu_seconds(command).ok_or("Metal did not report candidate objective GPU timing".into())
}
