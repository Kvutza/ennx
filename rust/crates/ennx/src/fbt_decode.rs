//! Resident incremental PISA/MoE generation. One cache per physical layer/visit.
//! Each rollout starts at position zero under one immutable candidate row.
use super::*;
use crate::config::{FeedbackTransition, GenerationConfig, GenerationTask};
use deser::Serialize;
use metal::ComputeCommandEncoderRef;

const PADDED_EXPERTS: u32 = 628;

#[path = "fbt_decode/mhc.rs"]
mod mhc;
#[cfg(test)]
#[path = "fbt_decodetests.rs"]
mod tests;

pub(super) struct Cache {
    pub(super) qkv: Buffer,
    pub(super) pyramid: Buffer,
    pub(super) kv: Option<Buffer>,
}

struct DecodeKernels {
    gemv: ComputePipelineState,
    gemv_vector: ComputePipelineState,
    pad_router: ComputePipelineState,
    embed: ComputePipelineState,
    leaf: ComputePipelineState,
    attention: ComputePipelineState,
    rms: ComputePipelineState,
    residual: ComputePipelineState,
    route: ComputePipelineState,
    activation: ComputePipelineState,
    combine_residual: ComputePipelineState,
    combine_branch: ComputePipelineState,
    feedback: ComputePipelineState,
    mhc_replicate: ComputePipelineState,
    mhc_predict: ComputePipelineState,
    mhc_mix_rms: ComputePipelineState,
    mhc_update: ComputePipelineState,
    mhc_mean_rms: ComputePipelineState,
    rope: ComputePipelineState,
    sample: ComputePipelineState,
}

pub(crate) struct Decoder {
    context: u32,
    kernels: DecodeKernels,
    cache: Vec<Cache>,
    state: Buffer,
    normalized: Buffer,
    attention: Buffer,
    projected: Buffer,
    residual: Buffer,
    logits: Buffer,
    route_scores: Buffer,
    router: Buffer,
    experts: Buffer,
    route_weights: Buffer,
    margin: Buffer,
    gate: Buffer,
    activation: Buffer,
    down: Buffer,
    feedback_state: Buffer,
    feedback_gate: Buffer,
    mhc_streams: [Buffer; 2],
    mhc_coefficients: Buffer,
    unit_norm: Buffer,
    blocks: Buffer,
    tokens: Buffer,
    invalid: Buffer,
    rope: Buffer,
}

#[derive(Clone, Serialize)]
pub(super) struct RouteSample {
    /// Active routed experts in the final MoE layer of one GPU submission.
    pub active_experts: usize,
    /// Top-k row assignments in that same final layer.
    pub routed_rows: usize,
    /// Physical expert tiles dispatched for that same final layer.
    pub routed_tiles: usize,
}

/// Localized target-loss and realized token outcomes from one free-running
/// trajectory. These values reuse losses produced by verification; collecting
/// them does not execute another model forward pass.
#[derive(Clone, Copy, Debug, Serialize)]
pub(super) struct TargetQuality {
    pub mean_nll: f32,
    pub maximum_nll: f32,
    pub worst_window_nll: f32,
    pub window_tokens: usize,
    pub positional_accuracy: f32,
    pub positional_mismatches: usize,
    #[deser(skip_serializing_if = Option::is_none)]
    pub first_positional_mismatch: Option<usize>,
    #[deser(skip_serializing_if = Option::is_none)]
    pub mismatch_target_nll: Option<f32>,
}

#[derive(Clone, Serialize)]
pub(super) struct Rollout {
    #[deser(skip_serializing_if = Option::is_none)]
    pub draft: Option<crate::forward_program::diffusion::DiffusionMetrics>,
    pub tokens: Vec<u32>,
    pub finish_reason: &'static str,
    pub wall_seconds: f64,
    pub gpu_seconds: f64,
    pub evaluated_positions: usize,
    /// Exact output tokens committed through EOS or the configured limit.
    pub committed_tokens: usize,
    /// Whole- or remaining-sequence verification submissions.
    pub broad_passes: usize,
    /// Fixed-point correction waves after broad verification stops.
    pub correction_waves: usize,
    /// Metal submissions containing one or more fixed-point correction waves.
    pub repair_batches: usize,
    /// Draft tokens accepted before mismatches, summed across submissions.
    pub accepted_tokens: usize,
    /// Zero-based generated-token position in the incoming draft.
    #[deser(skip_serializing_if = Option::is_none)]
    pub first_mismatch: Option<usize>,
    /// Per verification submission, in dispatch order.
    pub evaluated_lengths: Vec<usize>,
    /// Draft-matching prefix lengths, excluding each corrective token.
    pub accepted_lengths: Vec<usize>,
    /// Exact output lengths committed, including each corrective token.
    pub committed_lengths: Vec<usize>,
    /// Final-layer routing samples, one per verifier GPU submission.
    pub route_samples: Vec<RouteSample>,
    #[deser(skip_serializing_if = Option::is_none)]
    pub free_running_target_nll: Option<f32>,
    #[deser(skip_serializing_if = Option::is_none)]
    pub target_quality: Option<TargetQuality>,
}

fn bytes<T>(encoder: &ComputeCommandEncoderRef, slot: u64, value: &T) {
    encoder.set_bytes(
        slot,
        size_of::<T>() as u64,
        std::ptr::from_ref(value).cast(),
    );
}

#[repr(C)]
struct Sample {
    seed: u64,
    position: u32,
    eos: u32,
    first: u32,
    temperature: f32,
}

fn launch(
    encoder: &ComputeCommandEncoderRef,
    kernel: &ComputePipelineState,
    buffers: &[(&BufferRef, u64)],
    groups: MTLSize,
    threads: u64,
) {
    encoder.set_compute_pipeline_state(kernel);
    for (index, (buffer, offset)) in buffers.iter().enumerate() {
        encoder.set_buffer(index as u64, Some(buffer), *offset);
    }
    encoder.dispatch_thread_groups(groups, thread_group(threads));
    let mut resources: [&metal::ResourceRef; 6] = [buffers[0].0; 6];
    for (target, (buffer, _)) in resources.iter_mut().zip(buffers) {
        *target = buffer;
    }
    encoder.memory_barrier_with_resources(&resources[..buffers.len()]);
}

fn decode_pipelines(
    runtime: &Runtime,
    context: u32,
    trial: Option<&crate::config::KernelTrial>,
) -> Result<DecodeKernels, String> {
    let custom_decode = trial
        .and_then(|t| t.decode.as_ref())
        .map(std::fs::read_to_string)
        .transpose()
        .map_err(|error| format!("read candidate decode: {error}"))?;
    let base_decode = custom_decode
        .as_deref()
        .unwrap_or(include_str!("fbt_decode.metal"));
    let source_text = format!("#define PISA_CONTEXT {context}\n{base_decode}");
    let source = source_text.as_str();
    let kernel = |name| runtime.precise_metal4(source, "resident decode", name, &[]);
    let utility = |name| runtime.pipeline(include_str!("fbt_moe.metal"), "decode utility", name);
    let custom_pisa = trial
        .and_then(|t| t.pisa.as_ref())
        .map(std::fs::read_to_string)
        .transpose()
        .map_err(|error| format!("read candidate pisa for decode: {error}"))?;
    let base_pisa = custom_pisa
        .as_deref()
        .unwrap_or(include_str!("fbt_pisa1.metal"));
    let attention_source = format!(
        "#define PISA_CONTEXT {context}\n#define PISA_DECODE\n#define PISA_QUERY_TILE 1\n#define PISA_SKIP_IDENTITY_RESCALE\n{base_pisa}"
    );
    let kernels = DecodeKernels {
        gemv: kernel("decode_gemv")?,
        gemv_vector: kernel("decode_gemv_vector")?,
        pad_router: kernel("decode_pad_router")?,
        embed: kernel("decode_embed")?,
        leaf: kernel("decode_leaf")?,
        combine_residual: kernel("decode_combine_residual_rms")?,
        combine_branch: kernel("decode_combine_branch")?,
        sample: kernel("decode_sample")?,
        attention: runtime.pipeline_metal4(
            &attention_source,
            "cached PISA",
            "fbt_pisa1_select_attention_q4",
            &[],
        )?,
        rms: utility("fbt_moe_rms")?,
        residual: utility("fbt_moe_residual_rms")?,
        activation: utility("fbt_moe_swiglu")?,
        feedback: utility("fbt_moe_feedback_fuse_rms")?,
        mhc_replicate: utility("fbt_mhc_replicate")?,
        mhc_predict: utility("fbt_mhc_predict")?,
        mhc_mix_rms: utility("fbt_mhc_mix_rms")?,
        mhc_update: utility("fbt_mhc_update")?,
        mhc_mean_rms: utility("fbt_mhc_mean_rms")?,
        rope: utility("fbt_moe_rope")?,
        route: runtime.pipeline(
            include_str!("fbt_routing.metal"),
            "decode route",
            "fbt_moe_select_top3",
        )?,
    };
    if kernels.gemv.thread_execution_width() != 32
        || kernels.gemv.max_total_threads_per_threadgroup() < 128
        || kernels.sample.max_total_threads_per_threadgroup() < 256
    {
        return Err("decode requires 32-lane SIMD and 256-thread groups".into());
    }
    Ok(kernels)
}

impl Decoder {
    pub(super) fn cache(&self) -> &[Cache] {
        &self.cache
    }

    pub fn new(runtime: &Runtime) -> Result<Self, String> {
        Self::with_context(runtime, CONTEXT)
    }

    pub(super) fn with_context(runtime: &Runtime, context: u32) -> Result<Self, String> {
        Self::with_cache(runtime, context, context > ROWS)
    }

    pub(super) fn with_cache(
        runtime: &Runtime,
        context: u32,
        compact: bool,
    ) -> Result<Self, String> {
        Self::with_visits(
            runtime,
            context,
            compact,
            (MODEL_LAYERS * FEEDBACK_PASSES) as usize,
        )
    }

    pub(super) fn with_visits(
        runtime: &Runtime,
        context: u32,
        compact: bool,
        visits: usize,
    ) -> Result<Self, String> {
        Self::for_trial(runtime, context, compact, visits, None)
    }

    pub(crate) fn for_trial(
        runtime: &Runtime,
        context: u32,
        compact: bool,
        visits: usize,
        trial: Option<&crate::config::KernelTrial>,
    ) -> Result<Self, String> {
        if !(4096..=crate::context::MAX_CONTEXT).contains(&context)
            || !context.is_power_of_two()
            || visits == 0
            || visits > 11
        {
            return Err(
                "decode requires a power-of-two context in 4096..2097152 and 1..11 visits".into(),
            );
        }
        let work = if compact { 4096 } else { context };
        let kernels = decode_pipelines(runtime, context, trial)?;
        Ok(Self {
            context,
            kernels,
            cache: (0..visits)
                .map(|_| Cache {
                    // PV's final eight-key tile can load masked future lanes.
                    // Zero initialization prevents 0 * uninitialized NaN there.
                    qkv: filled(runtime, (work * QKV_WIDTH) as usize, 0),
                    pyramid: filled(
                        runtime,
                        ((2 * context / PISA_BLOCK - 1) * HEAD_DIM) as usize,
                        0,
                    ),
                    kv: compact.then(|| filled(runtime, (context * 2 * HEAD_DIM) as usize, 0)),
                })
                .collect(),
            state: runtime.buffer::<u16>(512),
            normalized: runtime.buffer::<u16>(512),
            attention: runtime.buffer::<u16>((work * WIDTH) as usize),
            projected: runtime.buffer::<u16>(512),
            residual: runtime.buffer::<u16>(512),
            logits: runtime.buffer::<u16>(8192),
            route_scores: runtime.buffer::<u16>(PADDED_EXPERTS as usize),
            router: runtime.buffer::<u16>((MODEL_LAYERS * WIDTH * PADDED_EXPERTS) as usize),
            experts: runtime.buffer::<u32>(3),
            route_weights: runtime.buffer::<u16>(3),
            margin: runtime.buffer::<f32>(1),
            gate: runtime.buffer::<u16>(4 * 432),
            activation: runtime.buffer::<u16>(4 * 216),
            down: runtime.buffer::<u16>(4 * 512),
            feedback_state: runtime.buffer::<u16>(512),
            feedback_gate: runtime.buffer::<u16>(512),
            mhc_streams: [
                runtime.buffer::<u16>(4 * 512),
                runtime.buffer::<u16>(4 * 512),
            ],
            mhc_coefficients: runtime.buffer::<f32>(24),
            unit_norm: runtime.buffer_with(&[0x3c00u16; 512]),
            blocks: runtime.buffer::<u32>((work * PISA_SELECTED) as usize),
            tokens: runtime.buffer::<u32>(context as usize + 1),
            invalid: runtime.buffer::<u32>(1),
            rope: rope_table(runtime, context.min(MAX_CONTEXT)),
        })
    }

    pub(crate) fn gemv(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input: (&BufferRef, u64),
        weight: (&BufferRef, u64),
        output: (&BufferRef, u64),
        k: u32,
        n: u32,
        mode: u32,
    ) {
        // Metal uint3 is 16-byte aligned.
        bytes(encoder, 4, &[k, n, mode, 0]);
        let kernel = if n != VOCAB && n.is_multiple_of(4) {
            &self.kernels.gemv_vector
        } else {
            &self.kernels.gemv
        };
        launch(
            encoder,
            kernel,
            &[input, weight, output, (&self.experts, 0)],
            MTLSize {
                width: u64::from(n.div_ceil(32)),
                height: if mode == 0 { 1 } else { 4 },
                depth: 1,
            },
            128,
        );
    }

    fn prepare_router(&self, encoder: &ComputeCommandEncoderRef, weights: CandidateRow<'_>) {
        let elements = MODEL_LAYERS * WIDTH * PADDED_EXPERTS;
        launch(
            encoder,
            &self.kernels.pad_router,
            &[(weights.buffer, weights.router), (&self.router, 0)],
            thread_group(u64::from(elements.div_ceil(256))),
            256,
        );
    }

    fn norm(&self, encoder: &ComputeCommandEncoderRef, weight: (&BufferRef, u64)) {
        bytes(
            encoder,
            3,
            &MoeShape {
                rows: 1,
                width: WIDTH,
                experts: 1,
                rows_per_expert: 1,
                expert_width: 216,
            },
        );
        launch(
            encoder,
            &self.kernels.rms,
            &[(&self.state, 0), weight, (&self.normalized, 0)],
            thread_group(1),
            128,
        );
    }

    fn residual(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input: &BufferRef,
        branch: &BufferRef,
        weight: (&BufferRef, u64),
        output: &BufferRef,
    ) {
        bytes(
            encoder,
            5,
            &MoeShape {
                rows: 1,
                width: WIDTH,
                experts: 1,
                rows_per_expert: 1,
                expert_width: 216,
            },
        );
        launch(
            encoder,
            &self.kernels.residual,
            &[
                (input, 0),
                (branch, 0),
                weight,
                (output, 0),
                (&self.normalized, 0),
            ],
            thread_group(1),
            128,
        );
    }

    fn layer(
        &self,
        encoder: &ComputeCommandEncoderRef,
        weights: CandidateRow<'_>,
        position: u32,
        layer: u32,
        pass: u32,
        execution: u32,
    ) {
        if weights.architecture.is_multistream() {
            self.mhc_prepare(encoder, weights, layer, execution, true);
        }
        let cache = &self.cache[execution as usize];
        self.gemv(
            encoder,
            (&self.normalized, 0),
            (
                weights.buffer,
                weights.qkv + half_bytes(u64::from(layer * WIDTH * QKV_WIDTH)),
            ),
            (&cache.qkv, half_bytes(u64::from(position * QKV_WIDTH))),
            WIDTH,
            QKV_WIDTH,
            0,
        );
        let rope_shape = [1, position, self.context, ROPE_PAIRS];
        bytes(encoder, 2, &rope_shape);
        launch(
            encoder,
            &self.kernels.rope,
            &[
                (&cache.qkv, half_bytes(u64::from(position * QKV_WIDTH))),
                (&self.rope, 0),
            ],
            thread_group(u64::from(ROPE_HEADS * ROPE_PAIRS)),
            256,
        );
        if position % PISA_BLOCK == PISA_BLOCK - 1 {
            bytes(encoder, 2, &position);
            launch(
                encoder,
                &self.kernels.leaf,
                &[(&cache.qkv, 0), (&cache.pyramid, 0)],
                thread_group(1),
                64,
            );
        }
        bytes(encoder, 4, &position);
        launch(
            encoder,
            &self.kernels.attention,
            &[
                (&cache.qkv, 0),
                (&cache.pyramid, 0),
                (&self.blocks, 0),
                (&self.attention, 0),
            ],
            thread_group(1),
            32,
        );
        self.gemv(
            encoder,
            (&self.attention, half_bytes(u64::from(position * WIDTH))),
            (
                weights.buffer,
                weights.output + half_bytes(u64::from(layer * WIDTH * WIDTH)),
            ),
            (&self.projected, 0),
            WIDTH,
            WIDTH,
            0,
        );
        if weights.architecture.is_multistream() {
            self.finish_mhc(encoder, weights, layer, execution);
        } else {
            self.finish_layer(encoder, weights, layer, pass);
        }
    }

    fn finish_layer(
        &self,
        encoder: &ComputeCommandEncoderRef,
        weights: CandidateRow<'_>,
        layer: u32,
        pass: u32,
    ) {
        self.residual(
            encoder,
            &self.state,
            &self.projected,
            (
                weights.buffer,
                weights.ffn_norm + half_bytes(u64::from(layer * WIDTH)),
            ),
            &self.residual,
        );
        let (_, gate, down) = ffn_offsets(layer);
        self.gemv(
            encoder,
            (&self.normalized, 0),
            (
                &self.router,
                half_bytes(u64::from(layer * WIDTH * PADDED_EXPERTS)),
            ),
            (&self.route_scores, 0),
            WIDTH,
            PADDED_EXPERTS,
            0,
        );
        bytes(
            encoder,
            4,
            &[1u32, WIDTH, routing::ROUTED_EXPERTS, 3, 64, 1],
        );
        launch(
            encoder,
            &self.kernels.route,
            &[
                (&self.route_scores, 0),
                (&self.experts, 0),
                (&self.route_weights, 0),
                (&self.margin, 0),
            ],
            thread_group(1),
            32,
        );
        self.gemv(
            encoder,
            (&self.normalized, 0),
            (weights.buffer, weights.gate_up + gate),
            (&self.gate, 0),
            WIDTH,
            432,
            1,
        );
        bytes(
            encoder,
            2,
            &MoeShape {
                rows: 4,
                width: WIDTH,
                experts: 4,
                rows_per_expert: 1,
                expert_width: 216,
            },
        );
        launch(
            encoder,
            &self.kernels.activation,
            &[(&self.gate, 0), (&self.activation, 0)],
            thread_group(7),
            128,
        );
        self.gemv(
            encoder,
            (&self.activation, 0),
            (weights.buffer, weights.down + down),
            (&self.down, 0),
            216,
            WIDTH,
            2,
        );
        let norm = if layer + 1 < MODEL_LAYERS {
            (
                weights.buffer,
                weights.attention_norm + half_bytes(u64::from((layer + 1) * WIDTH)),
            )
        } else if pass + 1 < FEEDBACK_PASSES {
            if weights
                .architecture
                .feedback_transition(crate::config::FeedbackTransition::ProjectedSigmoid)
                == crate::config::FeedbackTransition::ProjectedSigmoid
            {
                (&*self.unit_norm, 0)
            } else {
                (weights.buffer, weights.attention_norm)
            }
        } else {
            (weights.buffer, weights.final_norm)
        };
        launch(
            encoder,
            &self.kernels.combine_residual,
            &[
                (&self.down, 0),
                (&self.route_weights, 0),
                (&self.residual, 0),
                norm,
                (&self.state, 0),
                (&self.normalized, 0),
            ],
            thread_group(1),
            128,
        );
    }

    fn token(
        &self,
        encoder: &ComputeCommandEncoderRef,
        weights: CandidateRow<'_>,
        position: u32,
        feedback: FeedbackTransition,
    ) {
        let feedback = weights.architecture.feedback_transition(feedback);
        bytes(encoder, 3, &position);
        launch(
            encoder,
            &self.kernels.embed,
            &[
                (weights.buffer, weights.readout),
                (&self.tokens, 0),
                (&self.state, 0),
            ],
            thread_group(4),
            128,
        );
        if weights.architecture.is_multistream() {
            let shape = [1u32, WIDTH, weights.architecture.kernel_code(), 0];
            bytes(encoder, 2, &shape);
            launch(
                encoder,
                &self.kernels.mhc_replicate,
                &[(&self.state, 0), (&self.mhc_streams[0], 0)],
                thread_group(4),
                128,
            );
        } else {
            self.norm(encoder, (weights.buffer, weights.attention_norm));
        }
        for step in weights.architecture.layer_steps() {
            self.layer(
                encoder,
                weights,
                position,
                step.layer,
                step.pass,
                step.execution,
            );
            if !weights.architecture.is_multistream()
                && step.layer + 1 == MODEL_LAYERS
                && step.pass + 1 < FEEDBACK_PASSES
            {
                if feedback == FeedbackTransition::ProjectedSigmoid {
                    self.gemv(
                        encoder,
                        (&self.state, 0),
                        (weights.buffer, weights.feedback_state),
                        (&self.feedback_state, 0),
                        WIDTH,
                        WIDTH,
                        0,
                    );
                    self.gemv(
                        encoder,
                        (&self.normalized, 0),
                        (weights.buffer, weights.feedback_gate),
                        (&self.feedback_gate, 0),
                        WIDTH,
                        WIDTH,
                        0,
                    );
                    bytes(
                        encoder,
                        5,
                        &MoeShape {
                            rows: 1,
                            width: WIDTH,
                            experts: 1,
                            rows_per_expert: 1,
                            expert_width: 216,
                        },
                    );
                    launch(
                        encoder,
                        &self.kernels.feedback,
                        &[
                            (&self.feedback_state, 0),
                            (&self.feedback_gate, 0),
                            (weights.buffer, weights.attention_norm),
                            (&self.state, 0),
                            (&self.normalized, 0),
                        ],
                        thread_group(1),
                        128,
                    );
                }
            }
        }
        if weights.architecture.is_multistream() {
            self.mhc_finalize(encoder, weights);
        }
    }

    pub fn generate(
        &self,
        runtime: &Runtime,
        weights: CandidateRow<'_>,
        task: &GenerationTask,
        config: &GenerationConfig,
        seed: u64,
    ) -> Result<Rollout, String> {
        if self.cache[0].kv.is_some() {
            return Err("compact context requires the chunked block executor".into());
        }
        let start = Instant::now();
        let positions = task.prompt.len() + config.max_tokens as usize - 1;
        if task.prompt.is_empty() || positions > self.context as usize {
            return Err("rollout exceeds cache capacity".into());
        }
        // No persistent cache is carried across candidate revisions or tasks.
        // Every readable prefix entry is overwritten in causal order below.
        unsafe {
            std::ptr::copy_nonoverlapping(
                task.prompt.as_ptr(),
                self.tokens.contents().cast::<u32>(),
                task.prompt.len(),
            );
            *self.invalid.contents().cast::<u32>() = 0;
        }
        let command = runtime.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        self.prepare_router(&encoder, weights);
        for position in 0..positions as u32 {
            self.token(&encoder, weights, position, config.feedback_transition);
            if position as usize + 1 >= task.prompt.len() {
                self.gemv(
                    &encoder,
                    (&self.normalized, 0),
                    (weights.buffer, weights.readout),
                    (&self.logits, 0),
                    WIDTH,
                    VOCAB,
                    0,
                );
                bytes(
                    &encoder,
                    3,
                    &Sample {
                        seed,
                        position,
                        eos: config.eos_token.unwrap_or(u32::MAX),
                        first: task.prompt.len() as u32,
                        temperature: config.temperature,
                    },
                );
                launch(
                    &encoder,
                    &self.kernels.sample,
                    &[(&self.logits, 0), (&self.tokens, 0), (&self.invalid, 0)],
                    thread_group(1),
                    256,
                );
            }
        }
        encoder.end_encoding();
        let gpu_seconds = complete(command)?;
        if unsafe { *self.invalid.contents().cast::<u32>() } != 0 {
            return Err("nonfinite generation logits".into());
        }
        let generated = unsafe {
            std::slice::from_raw_parts(
                self.tokens.contents().cast::<u32>().add(task.prompt.len()),
                config.max_tokens as usize,
            )
        };
        if generated.iter().any(|&token| token >= VOCAB) {
            return Err("invalid generated token".into());
        }
        let end = generated
            .iter()
            .position(|token| Some(*token) == config.eos_token);
        Ok(Rollout {
            draft: None,
            tokens: generated[..end.map_or(generated.len(), |i| i + 1)].to_vec(),
            finish_reason: if end.is_some() { "eos" } else { "length" },
            wall_seconds: start.elapsed().as_secs_f64(),
            gpu_seconds,
            evaluated_positions: positions,
            committed_tokens: end.map_or(generated.len(), |i| i + 1),
            broad_passes: 0,
            correction_waves: 0,
            repair_batches: 0,
            accepted_tokens: 0,
            first_mismatch: None,
            evaluated_lengths: Vec::new(),
            accepted_lengths: Vec::new(),
            committed_lengths: Vec::new(),
            route_samples: Vec::new(),
            free_running_target_nll: None,
            target_quality: None,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn repair_suffix(
        &self,
        runtime: &Runtime,
        weights: CandidateRow<'_>,
        task: &GenerationTask,
        config: &GenerationConfig,
        seed: u64,
        generated: &mut [u32],
        fixed: usize,
    ) -> Result<(f64, usize), String> {
        if fixed >= generated.len() {
            return Ok((0.0, 0));
        }
        let positions = task.prompt.len() + generated.len() - 1;
        let start_position = task.prompt.len() + fixed - 1;
        unsafe {
            let tokens = std::slice::from_raw_parts_mut(
                self.tokens.contents().cast::<u32>(),
                self.context as usize + 1,
            );
            tokens[..task.prompt.len()].copy_from_slice(&task.prompt);
            tokens[task.prompt.len()..task.prompt.len() + generated.len()]
                .copy_from_slice(generated);
            *self.invalid.contents().cast::<u32>() = 0;
        }
        let command = runtime.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        self.prepare_router(&encoder, weights);
        for position in start_position as u32..positions as u32 {
            self.token(&encoder, weights, position, config.feedback_transition);
            self.gemv(
                &encoder,
                (&self.normalized, 0),
                (weights.buffer, weights.readout),
                (&self.logits, 0),
                WIDTH,
                VOCAB,
                0,
            );
            bytes(
                &encoder,
                3,
                &Sample {
                    seed,
                    position,
                    eos: config.eos_token.unwrap_or(u32::MAX),
                    first: task.prompt.len() as u32,
                    temperature: config.temperature,
                },
            );
            launch(
                &encoder,
                &self.kernels.sample,
                &[(&self.logits, 0), (&self.tokens, 0), (&self.invalid, 0)],
                thread_group(1),
                256,
            );
        }
        encoder.end_encoding();
        let gpu_seconds = complete(command)?;
        if unsafe { *self.invalid.contents().cast::<u32>() } != 0 {
            return Err("nonfinite suffix-repair logits".into());
        }
        let repaired = unsafe {
            std::slice::from_raw_parts(
                self.tokens.contents().cast::<u32>().add(task.prompt.len()),
                generated.len(),
            )
        };
        generated[fixed..].copy_from_slice(&repaired[fixed..]);
        Ok((gpu_seconds, positions - start_position))
    }
}
