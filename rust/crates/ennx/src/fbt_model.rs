//! Complete token-ID scoring graph for an explicitly local FBT architecture.
//! This is not a pretrained model or a claim of exact paper reproduction.

use crate::apple_gpu::{Runtime, thread_group};
use crate::fbt::{
    Attention, AttentionConfig, CacheKey, Feedback, FeedbackConfig, InputNorm, Linear,
    ProjectionActivation, RmsNorm,
};
use crate::fbt_metal::{GemmEpilogue, check_memory, dispatch};
use metal::objc::{
    __send_message as send_message,
    runtime::{Object, Sel},
};
use metal::{Buffer, BufferRef, CommandBufferRef, ComputePipelineState, MTLCommandBufferStatus};
use std::sync::{Arc, OnceLock};

#[path = "fbt_round.rs"]
mod bo;
pub use crate::fbt_mps::{GateUpProbe, gateup_probe};
pub use bo::{RoundLatencyExperiment, RoundLatencyRecord, round_experiment};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum GateUpImplementation {
    #[default]
    Mps,
    FusedMetal,
}

const SOURCE: &str = include_str!("fbt_model.metal");
#[path = "fbt_model/execution.rs"]
mod execution;

#[path = "fbt_prefill.rs"]
mod prefill;
#[path = "fbt_prefill/allocation.rs"]
mod prefill_allocation;
#[path = "fbt_prefill/operations.rs"]
mod prefill_operations;
#[path = "fbt_prefill/passes.rs"]
mod prefill_passes;
#[path = "fbt_prefill/scoring.rs"]
mod prefill_scoring;
#[path = "fbt_prefill/traced.rs"]
mod prefill_traced;
#[path = "fbt_prefill/transpose.rs"]
mod prefill_transpose;
pub use prefill::BatchScore;

/// LocalV1: learned pre-attention/pre-FFN/final RMS scales, unit QK RMS,
/// sigmoid head gates projected from normalized attention input, post-final-norm
/// feedback states, explicit residual multiplier, no biases, tied readout.
#[derive(Clone, Copy, Debug)]
pub struct ModelConfig {
    pub width: u32,
    pub intermediate: u32,
    pub layers: u32,
    pub vocab: u32,
    pub heads: u32,
    pub kv_heads: u32,
    pub capacity: u32,
    pub chunk: u32,
    pub local_window: u32,
    pub full_every: u32,
    pub epsilon: f32,
    pub rope_base: f32,
    pub residual_scale: f32,
    pub feedback_token_norm: InputNorm,
    pub feedback_fused_norm: InputNorm,
    /// Use the validated 8-query/16-key tiled prefill kernel for 96-wide heads.
    /// This changes execution/reduction order, not the model's attention mask.
    pub tiled_attention: bool,
}

/// Reference scorer's requested buffer bytes, excluding driver/pipeline storage
/// and the separately allocated opt-in batched prefill workspace.
#[derive(Clone, Copy, Debug)]
pub struct ModelMemory {
    pub parameters: u64,
    pub weight_bytes: u64,
    pub kv_bytes: u64,
    pub workspace_bytes: u64,
    pub total_bytes: u64,
    pub largest_buffer_bytes: u64,
}

impl ModelConfig {
    pub fn memory(self) -> Result<ModelMemory, String> {
        self.validate()?;
        let d = u128::from(self.width);
        let f = u128::from(self.intermediate);
        let v = u128::from(self.vocab);
        let l = u128::from(self.layers);
        let h = u128::from(self.heads);
        let kv = u128::from(self.kv_heads) * (d / h);
        let t = u128::from(self.capacity);
        let b = u128::from(self.chunk);
        let parameters =
            v * d + 2 * d * d + d + l * (2 * d + 2 * d * d + 2 * kv * d + h * d + 3 * f * d);
        let weight_bytes = parameters * 2;
        let kv_bytes = l * t * kv * 4;
        // Per-layer rotated-Q scratch; shared graph/feedback scratch; two full
        // feedback histories; token IDs, labels, losses and feedback masks.
        let workspace_bytes = l * b * d * 4
            + b * (8 * d + 2 * kv + h + 3 * f + v + 2) * 4
            + 2 * t * d * 4
            + 3 * t * 4;
        let largest = [
            v * d * 2,
            4 * d * d,
            f * d * 2,
            t * kv * 2,
            t * d * 4,
            b * v * 4,
            b * f * 4,
            b * d * 4,
            b * h * 4,
            t * 4,
        ]
        .into_iter()
        .max()
        .unwrap();
        let checked = |n| u64::try_from(n).map_err(|_| "FBT memory estimate overflow".to_owned());
        Ok(ModelMemory {
            parameters: checked(parameters)?,
            weight_bytes: checked(weight_bytes)?,
            kv_bytes: checked(kv_bytes)?,
            workspace_bytes: checked(workspace_bytes)?,
            total_bytes: checked(weight_bytes + kv_bytes + workspace_bytes)?,
            largest_buffer_bytes: checked(largest)?,
        })
    }

    pub fn validate(self) -> Result<(), String> {
        if self.width == 0
            || self.intermediate == 0
            || self.layers == 0
            || self.vocab < 2
            || self.heads == 0
            || self.width % self.heads != 0
            || self.chunk == 0
            || self.chunk > self.capacity
            || self.full_every == 0
            || self.local_window == 0
            || !self.epsilon.is_finite()
            || self.epsilon <= 0.0
            || !self.residual_scale.is_finite()
        {
            return Err("Invalid FBT LocalV1 model configuration".into());
        }
        self.attention(0).validate()?;
        if self.tiled_attention && self.width / self.heads != 96 {
            return Err("FBT tiled attention requires 96-wide heads".into());
        }
        self.feedback_token_norm.epsilon()?;
        self.feedback_fused_norm.epsilon()?;
        // Glue kernels use u32 coordinates; byte sizes below use checked u64.
        self.chunk
            .checked_mul(self.intermediate)
            .ok_or("FBT FFN index overflow")?;
        self.chunk
            .checked_mul(self.vocab)
            .ok_or("FBT logits index overflow")?;
        Ok(())
    }

    fn attention(self, layer: u32) -> AttentionConfig {
        let dim = self.width / self.heads;
        AttentionConfig {
            heads: self.heads,
            kv_heads: self.kv_heads,
            head_dim: dim,
            capacity: self.capacity,
            window: if (layer + 1) % self.full_every == 0 {
                None
            } else {
                Some(self.local_window)
            },
            qk_norm: InputNorm::UnitRms {
                epsilon: self.epsilon,
            },
            rope_base: self.rope_base,
            score_scale: 1.0 / (dim as f32).sqrt(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScoreMode {
    /// Ordinary causal inputs, also the prompt stage used by Soft generation.
    Standard,
    /// Plain pass followed by a fresh-cache, shifted-feedback pass.
    Fused,
    /// Teacher-supplied tokens with exact preceding-token feedback recurrence.
    Sequential,
}

#[derive(Debug)]
pub struct ChunkTiming {
    pub pass: u32,
    pub start: u32,
    pub rows: u32,
    /// Completed command-buffer GPU interval; None if timestamps unavailable.
    /// No per-kernel attribution is implied by a whole-chunk interval.
    pub gpu_seconds: Option<f64>,
}

fn gpu_seconds(command: &CommandBufferRef) -> Option<f64> {
    if command.status() != MTLCommandBufferStatus::Completed {
        return None;
    }
    static SELECTORS: OnceLock<[Sel; 2]> = OnceLock::new();
    let selectors =
        SELECTORS.get_or_init(|| [Sel::register("GPUStartTime"), Sel::register("GPUEndTime")]);
    // Public MTLCommandBuffer properties, available on macOS 10.15+. metal-rs
    // lacks wrappers. Read only after successful completion, as Apple requires.
    let read = |selector| unsafe {
        send_message::<Object, (), f64>(
            command as *const CommandBufferRef as *const Object,
            selector,
            (),
        )
        .ok()
    };
    let start = read(selectors[0])?;
    let end = read(selectors[1])?;
    if start.is_finite() && end.is_finite() && start > 0.0 && end >= start {
        Some(end - start)
    } else {
        None
    }
}

#[derive(Debug)]
pub struct Score {
    pub mean_nll: f64,
    pub tokens: usize,
    pub passes: u32,
    /// Entire encode/submit/wait/read-loss call, excluding model initialization.
    pub elapsed_seconds: f64,
    /// Completed wall time per pass, including command encoding and waiting.
    pub pass_seconds: Vec<f64>,
    /// Host command construction/submission, including any queue backpressure.
    /// GPU execution may overlap this interval; this is not CPU-only kernel time.
    pub encode_submit_seconds: f64,
    /// Host time waiting after the final command of each pass was submitted.
    /// This is not total GPU time, since the GPU runs during encoding as well.
    pub completion_wait_seconds: f64,
    pub chunks: Vec<ChunkTiming>,
}

/// Each entry is one independent BF16 allocation. The tied head has no second entry.
pub struct Parameter {
    pub name: String,
    pub shape: Vec<u32>,
    buffer: Buffer,
    elements: usize,
}

struct Layer {
    norm_attn: usize,
    norm_ffn: usize,
    q: usize,
    k: usize,
    v: usize,
    head_gate: usize,
    out: usize,
    gate: usize,
    up: usize,
    down: usize,
    attention: Attention,
}

#[repr(C)]
struct GraphParams {
    width: u32,
    rows: u32,
    start: u32,
    vocab: u32,
    scale: f32,
}

pub struct Model {
    runtime: Arc<Runtime>,
    config: ModelConfig,
    parameters: Vec<Parameter>,
    layers: Vec<Layer>,
    embedding: usize,
    feedback_weights: usize,
    final_norm: usize,
    norm: RmsNorm,
    feedback: Feedback,
    square: Linear,
    kv: Linear,
    head_gate: Linear,
    up: Linear,
    down: Linear,
    readout: Linear,
    lookup: ComputePipelineState,
    residual: ComputePipelineState,
    glu: ComputePipelineState,
    shift: ComputePipelineState,
    capture: ComputePipelineState,
    cross_entropy: ComputePipelineState,
    cross_entropy_partials: ComputePipelineState,
    gate_up_implementation: GateUpImplementation,
    optimized: bool,
    tokens: Buffer,
    targets: Buffer,
    mask: Buffer,
    losses: Buffer,
    x: Buffer,
    embed: Buffer,
    normalized: Buffer,
    branch: Buffer,
    previous: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    gates: Buffer,
    attended: Buffer,
    ff_gate: Buffer,
    ff_up: Buffer,
    ff_hidden: Buffer,
    logits: Buffer,
    histories: [Buffer; 2],
    pub(crate) revision: u64,
    pub(crate) sequence: u64,
    prefills: Vec<prefill::Prefill>,
    pub(crate) transposed_weights: Vec<Option<Buffer>>,
    pub(crate) transposed_feedback: Option<[Buffer; 2]>,
    pub(crate) transposed_qkvg: Vec<Option<Buffer>>,
    pub(crate) transposed_gate_up: Vec<Option<Buffer>>,
    pub(crate) transposed_revision: u64,
}

fn allocate<T>(runtime: &Runtime, elements: u64) -> Result<Buffer, String> {
    let bytes = elements
        .checked_mul(size_of::<T>() as u64)
        .ok_or("FBT model allocation overflow")?;
    check_memory(runtime, bytes, bytes)?;
    let buffer =
        runtime.buffer::<T>(usize::try_from(elements).map_err(|_| "FBT allocation too large")?);
    if buffer.contents().is_null() {
        return Err("FBT model allocation failed".into());
    }
    Ok(buffer)
}

fn add_parameter(
    runtime: &Runtime,
    parameters: &mut Vec<Parameter>,
    seed: &mut u64,
    name: String,
    shape: Vec<u32>,
    scale: f32,
    norm: bool,
) -> Result<usize, String> {
    let elements = shape
        .iter()
        .try_fold(1u64, |n, &d| n.checked_mul(u64::from(d)))
        .ok_or("FBT parameter overflow")?;
    let buffer = allocate::<u16>(runtime, elements)?;
    // LocalV1 initialization: SplitMix64 uniform [-scale, scale], BF16 RNE.
    // Norm vectors are exactly one. No claim of the paper's initialization.
    let data = unsafe {
        std::slice::from_raw_parts_mut(buffer.contents().cast::<u16>(), elements as usize)
    };
    for value in data {
        *seed = seed.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = *seed;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^= z >> 31;
        let x = if norm {
            1.0
        } else {
            (((z >> 40) as f32 / 16777216.0) * 2.0 - 1.0) * scale
        };
        let bits = x.to_bits();
        *value = ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16;
    }
    let index = parameters.len();
    parameters.push(Parameter {
        name,
        shape,
        buffer,
        elements: elements as usize,
    });
    Ok(index)
}

fn build_layers(
    runtime: &Runtime,
    parameters: &mut Vec<Parameter>,
    seed: &mut u64,
    config: ModelConfig,
    key: CacheKey,
) -> Result<Vec<Layer>, String> {
    let kv_width = config.kv_heads * (config.width / config.heads);
    let mut layers = Vec::with_capacity(config.layers as usize);
    for layer in 0..config.layers {
        let mut parameter = |name: &str, shape: Vec<u32>, norm: bool| {
            let fan_in = *shape.last().expect("FBT parameters have dimensions");
            add_parameter(
                runtime,
                parameters,
                seed,
                format!("layers.{layer}.{name}"),
                shape,
                (3.0 / fan_in as f32).sqrt(),
                norm,
            )
        };
        layers.push(Layer {
            norm_attn: parameter("norm_attn", vec![config.width], true)?,
            norm_ffn: parameter("norm_ffn", vec![config.width], true)?,
            q: parameter("q", vec![config.width, config.width], false)?,
            k: parameter("k", vec![kv_width, config.width], false)?,
            v: parameter("v", vec![kv_width, config.width], false)?,
            head_gate: parameter("head_gate", vec![config.heads, config.width], false)?,
            out: parameter("out", vec![config.width, config.width], false)?,
            gate: parameter("ff_gate", vec![config.intermediate, config.width], false)?,
            up: parameter("ff_up", vec![config.intermediate, config.width], false)?,
            down: parameter("ff_down", vec![config.width, config.intermediate], false)?,
            attention: Attention::new(config.attention(layer), config.chunk, key)?,
        });
    }
    Ok(layers)
}

fn validate_inputs(config: ModelConfig, tokens: &[u32], targets: &[u32]) -> Result<(), String> {
    let invalid = tokens.is_empty()
        || tokens.len() != targets.len()
        || tokens.len() > config.capacity as usize
        || tokens
            .iter()
            .chain(targets)
            .any(|&token| token >= config.vocab);
    if invalid {
        return Err("Invalid FBT token/target IDs or sequence length".into());
    }
    Ok(())
}

impl Model {
    pub fn new(config: ModelConfig, seed: u64) -> Result<Self, String> {
        metal::objc::rc::autoreleasepool(|| Self::new_inner(config, seed))
    }

    fn new_inner(config: ModelConfig, mut seed: u64) -> Result<Self, String> {
        config.validate()?;
        let runtime = Runtime::shared()?;
        let memory = config.memory()?;
        check_memory(&runtime, memory.total_bytes, memory.largest_buffer_bytes)?;
        let c = config;
        let mut parameters = Vec::new();
        let scale = (3.0 / c.width as f32).sqrt();
        let embedding = add_parameter(
            &runtime,
            &mut parameters,
            &mut seed,
            "embedding_tied_head".into(),
            vec![c.vocab, c.width],
            scale,
            false,
        )?;
        let feedback_weights = add_parameter(
            &runtime,
            &mut parameters,
            &mut seed,
            "feedback_state_gate".into(),
            vec![2, c.width, c.width],
            scale,
            false,
        )?;
        let final_norm = add_parameter(
            &runtime,
            &mut parameters,
            &mut seed,
            "final_norm".into(),
            vec![c.width],
            1.0,
            true,
        )?;
        let key = CacheKey {
            candidate: 0,
            sequence: 0,
            pass: 0,
        };
        let kv_width = c.kv_heads * (c.width / c.heads);
        let layers = build_layers(&runtime, &mut parameters, &mut seed, c, key)?;
        let projection = |input, output| Linear::new(input, output, ProjectionActivation::Identity);
        let pipe = |name| runtime.precise(SOURCE, "FBT LocalV1 model", name);
        let b = |width| allocate::<f32>(&runtime, u64::from(c.chunk) * u64::from(width));
        Ok(Self {
            config: c,
            parameters,
            layers,
            embedding,
            feedback_weights,
            final_norm,
            norm: RmsNorm::new(c.width, c.epsilon)?,
            feedback: model_feedback(c)?,
            square: projection(c.width, c.width)?,
            kv: projection(c.width, kv_width)?,
            head_gate: Linear::new(c.width, c.heads, ProjectionActivation::Sigmoid)?,
            up: projection(c.width, c.intermediate)?,
            down: projection(c.intermediate, c.width)?,
            readout: projection(c.width, c.vocab)?,
            lookup: pipe("fbt_lookup")?,
            residual: pipe("fbt_residual")?,
            glu: pipe("fbt_glu")?,
            shift: pipe("fbt_shift_state")?,
            capture: pipe("fbt_capture_state")?,
            cross_entropy: pipe("fbt_cross_entropy")?,
            cross_entropy_partials: pipe("fbt_cross_entropy_partials")?,
            optimized: false,
            gate_up_implementation: GateUpImplementation::default(),
            tokens: allocate::<u32>(&runtime, u64::from(c.capacity))?,
            targets: allocate::<u32>(&runtime, u64::from(c.capacity))?,
            mask: allocate::<u32>(&runtime, u64::from(c.chunk))?,
            losses: allocate::<f32>(&runtime, u64::from(c.capacity))?,
            x: b(c.width)?,
            embed: b(c.width)?,
            normalized: b(c.width)?,
            branch: b(c.width)?,
            previous: b(c.width)?,
            q: b(c.width)?,
            k: b(kv_width)?,
            v: b(kv_width)?,
            gates: b(c.heads)?,
            attended: b(c.width)?,
            ff_gate: b(c.intermediate)?,
            ff_up: b(c.intermediate)?,
            ff_hidden: b(c.intermediate)?,
            logits: b(c.vocab)?,
            histories: feedback_histories(&runtime, c)?,
            runtime,
            revision: 0,
            sequence: 0,
            prefills: Vec::new(),
            transposed_weights: Vec::new(),
            transposed_feedback: None,
            transposed_qkvg: Vec::new(),
            transposed_gate_up: Vec::new(),
            transposed_revision: u64::MAX,
        })
    }

    pub fn parameters(&self) -> &[Parameter] {
        &self.parameters
    }

    pub fn parameter_count(&self) -> usize {
        self.parameters.iter().map(|p| p.elements).sum()
    }

    /// Opt into the experimental fused graph. The original graph remains the
    /// default until whole-loop speedups are established. Switching allocates
    /// the appropriate output workspace between synchronous score calls.
    pub fn set_optimized(&mut self, enabled: bool) -> Result<(), String> {
        if enabled != self.optimized {
            let width = if enabled {
                u64::from(self.config.vocab).div_ceil(32) * 4
            } else {
                u64::from(self.config.vocab)
            };
            self.logits = allocate::<f32>(&self.runtime, u64::from(self.config.chunk) * width)?;
            self.optimized = enabled;
        }
        Ok(())
    }

    pub(crate) fn set_gateup(
        &mut self,
        implementation: GateUpImplementation,
    ) -> Result<(), String> {
        if implementation == GateUpImplementation::FusedMetal
            && (!self.optimized || self.config.width != 1536 || self.config.intermediate != 6656)
        {
            return Err("fused Metal gate/up requires the optimized 1536x6656 scorer".into());
        }
        self.gate_up_implementation = implementation;
        Ok(())
    }

    /// Replace a full tensor between synchronous evaluations. No stale packed
    /// copies; every score call starts fresh caches under the new revision.
    pub fn replace_parameter(&mut self, name: &str, bf16: &[u16]) -> Result<(), String> {
        let p = self
            .parameters
            .iter()
            .find(|p| p.name == name)
            .ok_or("Unknown FBT parameter")?;
        if bf16.len() != p.elements || bf16.iter().any(|&v| v & 0x7f80 == 0x7f80) {
            return Err("FBT parameter length mismatch or non-finite value".into());
        }
        let revision = self
            .revision
            .checked_add(1)
            .ok_or("FBT revision exhausted")?;
        unsafe {
            std::ptr::copy_nonoverlapping(bf16.as_ptr(), p.buffer.contents().cast(), bf16.len());
        }
        self.revision = revision;
        Ok(())
    }

    /// Bind parameters to zero-copy views into a resident contiguous BF16 row.
    /// The caller must keep `source` alive until the model is restored or rebound.
    pub(crate) fn bind_row(&mut self, source: &BufferRef) -> Result<Vec<Buffer>, String> {
        let required = self.parameter_count() as u64 * 2;
        if source.length() < required {
            return Err("FBT resident parameter row is too short".into());
        }
        let options = metal::MTLResourceOptions::StorageModeShared
            | metal::MTLResourceOptions::HazardTrackingModeTracked;
        let mut originals = Vec::with_capacity(self.parameters.len());
        let mut offset = 0usize;
        for parameter in &mut self.parameters {
            let bytes = parameter.elements * 2;
            let pointer = unsafe { source.contents().cast::<u8>().add(offset) };
            let view = self.runtime.device.new_buffer_with_bytes_no_copy(
                pointer.cast(),
                bytes as u64,
                options,
                None,
            );
            if view.length() != bytes as u64 {
                return Err(format!(
                    "FBT resident parameter view has {} bytes, needs {bytes}",
                    view.length()
                ));
            }
            originals.push(std::mem::replace(&mut parameter.buffer, view));
            offset += bytes;
        }
        self.revision = self
            .revision
            .checked_add(1)
            .ok_or("FBT revision exhausted")?;
        Ok(originals)
    }

    pub(crate) fn restore_buffers(&mut self, originals: Vec<Buffer>) -> Result<(), String> {
        if originals.len() != self.parameters.len() {
            return Err("FBT parameter buffer restore length mismatch".into());
        }
        for (parameter, original) in self.parameters.iter_mut().zip(originals) {
            parameter.buffer = original;
        }
        self.revision = self
            .revision
            .checked_add(1)
            .ok_or("FBT revision exhausted")?;
        Ok(())
    }

    /// Explicit teacher-forced labels: targets[i] is predicted after tokens[i].
    /// No implicit shifting, tokenizer, sampling, masking or hidden validation set.
    /// Upload, encode, GPU completion and loss readback are included in elapsed.
    pub fn score(
        &mut self,
        tokens: &[u32],
        targets: &[u32],
        mode: ScoreMode,
    ) -> Result<Score, String> {
        let start = std::time::Instant::now();
        let mut score =
            metal::objc::rc::autoreleasepool(|| self.score_inner(tokens, targets, mode))?;
        score.elapsed_seconds = start.elapsed().as_secs_f64();
        Ok(score)
    }
}

fn model_feedback(c: ModelConfig) -> Result<Feedback, String> {
    Feedback::new(
        FeedbackConfig {
            width: c.width,
            token_norm: c.feedback_token_norm,
            fused_norm: c.feedback_fused_norm,
        },
        c.chunk,
    )
}

fn feedback_histories(runtime: &Runtime, c: ModelConfig) -> Result<[Buffer; 2], String> {
    Ok([
        allocate::<f32>(runtime, u64::from(c.capacity) * u64::from(c.width))?,
        allocate::<f32>(runtime, u64::from(c.capacity) * u64::from(c.width))?,
    ])
}

#[cfg(test)]
#[path = "fbt_modeltests.rs"]
mod tests;
