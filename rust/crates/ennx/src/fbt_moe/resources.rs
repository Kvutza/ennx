use super::*;
pub(super) use crate::apple_gpu::{gpu_interval, gpu_seconds, thread_group};
pub(super) use crate::bf16_metal::{
    ControllerInfo, NoisyDecision, ParamBlock, ProgramAddress, Proposals, SearchState,
};
pub(super) use crate::fbt_mps::{Matmul, Matrix};
pub(super) use metal::MTLSize;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};

pub(super) fn filled(runtime: &Runtime, elements: usize, bits: u16) -> Buffer {
    let buffer = runtime.buffer::<u16>(elements);
    let values =
        unsafe { std::slice::from_raw_parts_mut(buffer.contents().cast::<u16>(), elements) };
    values.fill(bits);
    buffer
}

pub(super) fn patterned(runtime: &Runtime, elements: usize, seed: u32, exponent: u16) -> Buffer {
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

fn half_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let fraction = bits & 0x7f_ffff;
    if exponent == 0xff {
        return sign | 0x7c00 | u16::from(fraction != 0);
    }
    let mut half_exponent = exponent - 127 + 15;
    if half_exponent >= 31 {
        return sign | 0x7c00;
    }
    if half_exponent <= 0 {
        if half_exponent < -10 {
            return sign;
        }
        let mantissa = fraction | 0x80_0000;
        let shift = (14 - half_exponent) as u32;
        let mut rounded = mantissa >> shift;
        let remainder = mantissa & ((1u32 << shift) - 1);
        let halfway = 1u32 << (shift - 1);
        if remainder > halfway || (remainder == halfway && rounded & 1 != 0) {
            rounded += 1;
        }
        return sign | rounded as u16;
    }
    let mut rounded = fraction >> 13;
    let remainder = fraction & 0x1fff;
    if remainder > 0x1000 || (remainder == 0x1000 && rounded & 1 != 0) {
        rounded += 1;
        if rounded == 0x400 {
            rounded = 0;
            half_exponent += 1;
            if half_exponent == 31 {
                return sign | 0x7c00;
            }
        }
    }
    sign | ((half_exponent as u16) << 10) | rounded as u16
}

fn mhc_bias(runtime: &Runtime, layers: usize) -> Buffer {
    let mut values = vec![0u16; layers * MHC_COEFFICIENTS as usize];
    let input_bias = half_bits(-3.0f32.ln());
    for layer in 0..layers {
        let base = layer * MHC_COEFFICIENTS as usize;
        values[base..base + MHC_STREAMS as usize].fill(input_bias);
    }
    runtime.buffer_with(&values)
}

fn normal_buffer(runtime: &Runtime, elements: usize, seed: u64, scale: f32) -> Buffer {
    let buffer = runtime.buffer::<u16>(elements);
    let values =
        unsafe { std::slice::from_raw_parts_mut(buffer.contents().cast::<u16>(), elements) };
    let mut random = rand::rngs::StdRng::seed_from_u64(seed);
    for value in values {
        let sample: f32 = StandardNormal.sample(&mut random);
        *value = half_bits(sample * scale);
    }
    buffer
}

fn uniform_buffer(runtime: &Runtime, elements: usize, seed: u64, bound: f32) -> Buffer {
    let buffer = runtime.buffer::<u16>(elements);
    let values =
        unsafe { std::slice::from_raw_parts_mut(buffer.contents().cast::<u16>(), elements) };
    let mut random = rand::rngs::StdRng::seed_from_u64(seed);
    for value in values {
        *value = half_bits(random.gen_range(-bound..=bound));
    }
    buffer
}

fn index_weights(runtime: &Runtime) -> Buffer {
    let mut values = vec![0u16; 64 * 64];
    for dim in 0..64 {
        values[dim * 64 + dim] = 0x3c00;
    }
    runtime.buffer_with(&values)
}

pub(super) fn factor(runtime: &Runtime, experts: u32, input: u32, output: u32) -> Buffer {
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

pub(super) fn rope_table(runtime: &Runtime, context: u32) -> Buffer {
    let mut values = Vec::with_capacity((context * ROPE_PAIRS) as usize);
    for position in 0..context {
        for pair in 0..ROPE_PAIRS {
            let exponent = -2.0 * f64::from(pair) / f64::from(HEAD_DIM);
            let angle = f64::from(position) * ROPE_BASE.powf(exponent);
            values.push([angle.cos() as f32, angle.sin() as f32]);
        }
    }
    runtime.buffer_with(&values)
}

pub(super) fn dispatch(
    command: &CommandBufferRef,
    pipeline: &ComputePipelineState,
    buffers: &[&BufferRef],
    parameters: *const std::ffi::c_void,
    parameter_bytes: u64,
    threads: u64,
) {
    let encoder = command.new_compute_command_encoder();
    dispatch_on(
        &encoder,
        pipeline,
        buffers,
        parameters,
        parameter_bytes,
        threads,
    );
    encoder.end_encoding();
}

pub(super) fn dispatch_on(
    encoder: &ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    buffers: &[&BufferRef],
    parameters: *const std::ffi::c_void,
    parameter_bytes: u64,
    threads: u64,
) {
    encoder.set_compute_pipeline_state(pipeline);
    for (index, buffer) in buffers.iter().enumerate() {
        encoder.set_buffer(index as u64, Some(buffer), 0);
    }
    encoder.set_bytes(buffers.len() as u64, parameter_bytes, parameters);
    let width = pipeline.max_total_threads_per_threadgroup().min(256);
    encoder.dispatch_threads(thread_group(threads), thread_group(width));
}

impl Pipelines {
    pub(super) fn for_trial(
        runtime: &Runtime,
        trial: Option<&crate::config::KernelTrial>,
    ) -> Result<Self, String> {
        let mut result = Self::new(runtime)?;
        if let Some(path) = trial.and_then(|trial| trial.moe.as_ref()) {
            let source = std::fs::read_to_string(path)
                .map_err(|error| format!("read MoE candidate {}: {error}", path.display()))?;
            result.fine_grained =
                routing::FineGrainedMoePipelines::with_source(runtime, Some(&source))?;
        }
        Ok(result)
    }

    pub(super) fn new(runtime: &Runtime) -> Result<Self, String> {
        let utility_source = include_str!("../fbt_moe.metal");
        let utility = |name| runtime.precise(utility_source, "grouped MoE layer probe", name);
        Ok(Self {
            fine_grained: routing::FineGrainedMoePipelines::new(runtime)?,
            quantize_gate_up_int8: utility("fbt_quantize_gate_up_int8")?,
            interleave_gate_up: utility("fbt_interleave_gate_up")?,
            gate: utility("fbt_moe_balanced_gate")?,
            group: utility("fbt_moe_group_balanced")?,
            swiglu: utility("fbt_moe_swiglu")?,
            ungroup: utility("fbt_moe_ungroup_residual")?,
            feedback_fuse: utility("fbt_moe_feedback_fuse")?,
            cross_entropy: utility("fbt_moe_cross_entropy")?,
            readout_loss_reduce: utility("fbt_readout_loss_reduce")?,
            readout_proposal_reduce: utility("fbt_readout_proposal_reduce")?,
            sequence_loss: utility("fbt_moe_sequence_loss")?,
            embed: utility("fbt_moe_embed")?,
            denoise_embed: runtime.precise(
                include_str!("../fbt_denoise.metal"),
                "diffusion input",
                "denoise_embed",
            )?,
            denoise_reduce: runtime.precise(
                include_str!("../fbt_denoise.metal"),
                "diffusion sampling",
                "denoise_reduce",
            )?,
            rms: utility("fbt_moe_rms")?,
            residual: utility("fbt_moe_residual")?,
            residual_rms: utility("fbt_moe_residual_rms")?,
            feedback_fuse_rms: utility("fbt_moe_feedback_fuse_rms")?,
            mhc_replicate: utility("fbt_mhc_replicate")?,
            mhc_predict: utility("fbt_mhc_predict")?,
            mhc_scale: utility("fbt_mhc_scale")?,
            mhc_predict_rows: utility("fbt_mhc_predict_rows")?,
            mhc_mix_rms: utility("fbt_mhc_mix_rms")?,
            mhc_update: utility("fbt_mhc_update_rows")?,
            mhc_mean_rms: utility("fbt_mhc_mean_rms")?,
            rope: utility("fbt_moe_rope")?,
        })
    }
}

impl TensorOpsPipelines {
    pub(super) fn new(runtime: &Runtime) -> Result<Self, String> {
        let source = include_str!("../fbt_moe_tensorops.metal");
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
            qkv_wide: runtime.precise_metal4(
                source,
                "Metal 4 TensorOps wide QKV projection",
                "fbt_model_tensorops_qkv_wide",
                &[],
            )?,
            output_projection_wide: runtime.precise_metal4(
                source,
                "Metal 4 TensorOps wide attention output projection",
                "fbt_model_tensorops_output_projection_wide",
                &[],
            )?,
            readout: runtime.precise_metal4(
                source,
                "Metal 4 TensorOps vocabulary readout",
                "fbt_model_tensorops_readout",
                &[],
            )?,
            readout_loss_tiles: runtime.precise_metal4(
                include_str!("../fbt_loss_tensorops.metal"),
                "Metal 4 exact full-vocabulary loss tiles",
                "fbt_readout_loss_tiles",
                &[],
            )?,
            readout_proposal_tiles: runtime.precise_metal4(
                include_str!("../fbt_loss_tensorops.metal"),
                "Metal 4 full-vocabulary proposal tiles",
                "fbt_readout_proposal_tiles",
                &[],
            )?,
            denoise_tiles: runtime.precise_metal4(
                &format!(
                    "#define DENOISE\n{}",
                    include_str!("../fbt_loss_tensorops.metal")
                ),
                "diffusion readout",
                "fbt_readout_proposal_tiles",
                &[],
            )?,
        })
    }
}

pub(super) fn synthetic_buffer(runtime: &Runtime, shift: u32) -> Buffer {
    runtime.buffer_with(
        &(0..ROWS)
            .map(|row| (row.wrapping_mul(17) + row / CONTEXT * 97 + shift) % VOCAB)
            .collect::<Vec<_>>(),
    )
}

pub(super) fn pretrain_batch(
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
    pub(super) fn new(runtime: &Runtime) -> Self {
        Self::seeded(runtime, None)
    }

    pub(super) fn seeded(runtime: &Runtime, seed: Option<u64>) -> Self {
        Self::seeded_for(runtime, seed, ResidualArchitecture::Legacy)
    }

    pub(super) fn seeded_for(
        runtime: &Runtime,
        seed: Option<u64>,
        architecture: ResidualArchitecture,
    ) -> Self {
        let family = |domain: u32| {
            seed.map_or(domain, |seed| {
                crate::hash::splitmix64(seed ^ u64::from(domain)) as u32
            })
        };
        let layers = MODEL_LAYERS as usize;
        let experts = routing::MOE_EXPERTS as usize;
        let width = WIDTH as usize;
        let expert_width = routing::ROUTED_WIDTH as usize;
        let gate_up = routing::MOE_GATEUP as usize;
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
        let mhc_predictor_elements = layers * MHC_INPUT as usize * MHC_COEFFICIENTS as usize;
        let mhc_control_elements = layers * 4;
        Self {
            architecture,
            router: patterned(runtime, router_elements, family(0x6a09_e667), 0x1800),
            qkv: patterned(runtime, qkv_elements, family(0xbb67_ae85), 0x2000),
            output: patterned(runtime, output_elements, family(0x3c6e_f372), 0x2000),
            gate_up: patterned(runtime, gate_up_elements, family(0xa54f_f53a), 0x2000),
            down: patterned(runtime, down_elements, family(0x510e_527f), 0x2000),
            attention_norm: filled(runtime, norm_elements, 0x3c00),
            ffn_norm: filled(runtime, norm_elements, 0x3c00),
            readout: patterned(runtime, readout_elements, family(0x9b05_688c), 0x1800),
            mask_embed: normal_buffer(runtime, width, u64::from(family(0x4d41_534b)), 0.02),
            index_query: index_weights(runtime),
            feedback_state: patterned(runtime, feedback_elements, family(0x1f83_d9ab), 0x2000),
            feedback_gate: patterned(runtime, feedback_elements, family(0x5be0_cd19), 0x2000),
            mhc_attention_predictor: patterned(
                runtime,
                mhc_predictor_elements,
                family(0x428a_2f98),
                0x1400,
            ),
            mhc_attention_bias: mhc_bias(runtime, layers),
            mhc_attention_control: filled(runtime, mhc_control_elements, 0),
            mhc_moe_predictor: patterned(
                runtime,
                mhc_predictor_elements,
                family(0x7137_4491),
                0x1400,
            ),
            mhc_moe_bias: mhc_bias(runtime, layers),
            mhc_moe_control: filled(runtime, mhc_control_elements, 0),
            final_norm: filled(runtime, width, 0x3c00),
        }
    }

    pub(super) fn initialized(
        runtime: &Runtime,
        seed: u64,
        initialization: crate::config::ModelInitialization,
        architecture: ResidualArchitecture,
    ) -> Self {
        use crate::config::ModelInitialization;
        match initialization {
            ModelInitialization::Patterned => Self::seeded_for(runtime, Some(seed), architecture),
            ModelInitialization::Gpt2 => Self::normal(runtime, seed, false, architecture),
            ModelInitialization::Megatron => Self::normal(runtime, seed, true, architecture),
            ModelInitialization::XavierUniform => Self::xavier(runtime, seed, architecture),
        }
    }

    fn normal(
        runtime: &Runtime,
        seed: u64,
        scaled: bool,
        architecture: ResidualArchitecture,
    ) -> Self {
        let layers = MODEL_LAYERS as usize;
        let experts = routing::MOE_EXPERTS as usize;
        let width = WIDTH as usize;
        let expert_width = routing::ROUTED_WIDTH as usize;
        let gate_up = routing::MOE_GATEUP as usize;
        let qkv_width = QKV_WIDTH as usize;
        let vocab = VOCAB as usize;
        let residual = if scaled {
            0.02 / (2.0 * MODEL_LAYERS as f32).sqrt()
        } else {
            0.02
        };
        let draw = |elements, domain, standard_deviation| {
            normal_buffer(
                runtime,
                elements,
                crate::hash::splitmix64(seed ^ domain),
                standard_deviation,
            )
        };
        Self {
            architecture,
            router: draw(
                layers * width * routing::ROUTED_EXPERTS as usize,
                0x6a09_e667,
                0.02,
            ),
            qkv: draw(layers * width * qkv_width, 0xbb67_ae85, 0.02),
            output: draw(layers * width * width, 0x3c6e_f372, residual),
            gate_up: draw(layers * experts * width * gate_up, 0xa54f_f53a, 0.02),
            down: draw(
                layers * experts * expert_width * width,
                0x510e_527f,
                residual,
            ),
            attention_norm: filled(runtime, layers * width, 0x3c00),
            ffn_norm: filled(runtime, layers * width, 0x3c00),
            readout: draw(width * vocab, 0x9b05_688c, 0.02),
            mask_embed: draw(width, 0x4d41_534b, 0.02),
            index_query: index_weights(runtime),
            feedback_state: draw(width * width, 0x1f83_d9ab, 0.02),
            feedback_gate: draw(width * width, 0x5be0_cd19, 0.02),
            mhc_attention_predictor: draw(
                layers * MHC_INPUT as usize * MHC_COEFFICIENTS as usize,
                0x428a_2f98,
                0.02,
            ),
            mhc_attention_bias: mhc_bias(runtime, layers),
            mhc_attention_control: filled(runtime, layers * 4, 0),
            mhc_moe_predictor: draw(
                layers * MHC_INPUT as usize * MHC_COEFFICIENTS as usize,
                0x7137_4491,
                0.02,
            ),
            mhc_moe_bias: mhc_bias(runtime, layers),
            mhc_moe_control: filled(runtime, layers * 4, 0),
            final_norm: filled(runtime, width, 0x3c00),
        }
    }

    fn xavier(runtime: &Runtime, seed: u64, architecture: ResidualArchitecture) -> Self {
        let layers = MODEL_LAYERS as usize;
        let experts = routing::MOE_EXPERTS as usize;
        let width = WIDTH as usize;
        let expert_width = routing::ROUTED_WIDTH as usize;
        let gate_up = routing::MOE_GATEUP as usize;
        let qkv_width = QKV_WIDTH as usize;
        let vocab = VOCAB as usize;
        let draw = |elements, domain, input, output| {
            let bound = (6.0f32 / (input + output) as f32).sqrt();
            uniform_buffer(
                runtime,
                elements,
                crate::hash::splitmix64(seed ^ domain),
                bound,
            )
        };
        Self {
            architecture,
            router: draw(
                layers * width * routing::ROUTED_EXPERTS as usize,
                0x6a09_e667,
                width,
                routing::ROUTED_EXPERTS as usize,
            ),
            qkv: draw(layers * width * qkv_width, 0xbb67_ae85, width, qkv_width),
            output: draw(layers * width * width, 0x3c6e_f372, width, width),
            gate_up: draw(
                layers * experts * width * gate_up,
                0xa54f_f53a,
                width,
                gate_up,
            ),
            down: draw(
                layers * experts * expert_width * width,
                0x510e_527f,
                expert_width,
                width,
            ),
            attention_norm: filled(runtime, layers * width, 0x3c00),
            ffn_norm: filled(runtime, layers * width, 0x3c00),
            readout: draw(width * vocab, 0x9b05_688c, width, vocab),
            mask_embed: draw(width, 0x4d41_534b, width, width),
            index_query: index_weights(runtime),
            feedback_state: draw(width * width, 0x1f83_d9ab, width, width),
            feedback_gate: draw(width * width, 0x5be0_cd19, width, width),
            mhc_attention_predictor: draw(
                layers * MHC_INPUT as usize * MHC_COEFFICIENTS as usize,
                0x428a_2f98,
                MHC_INPUT as usize,
                MHC_COEFFICIENTS as usize,
            ),
            mhc_attention_bias: mhc_bias(runtime, layers),
            mhc_attention_control: filled(runtime, layers * 4, 0),
            mhc_moe_predictor: draw(
                layers * MHC_INPUT as usize * MHC_COEFFICIENTS as usize,
                0x7137_4491,
                MHC_INPUT as usize,
                MHC_COEFFICIENTS as usize,
            ),
            mhc_moe_bias: mhc_bias(runtime, layers),
            mhc_moe_control: filled(runtime, layers * 4, 0),
            final_norm: filled(runtime, width, 0x3c00),
        }
    }
    pub(super) fn tensors(&self) -> Vec<(&'static str, &BufferRef, usize)> {
        let mut tensors: Vec<(&'static str, &BufferRef, usize)> = vec![
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
                (WIDTH * routing::MOE_GATEUP) as usize,
            ),
            (
                "expert_down",
                &self.down,
                (routing::ROUTED_WIDTH * WIDTH) as usize,
            ),
            ("attention_norm", &self.attention_norm, WIDTH as usize),
            ("ffn_norm", &self.ffn_norm, WIDTH as usize),
            ("embedding_readout", &self.readout, (WIDTH * VOCAB) as usize),
        ];
        if self.architecture.is_multistream() {
            tensors.extend([
                (
                    "mhc_attention_predictor",
                    self.mhc_attention_predictor.as_ref(),
                    (MHC_INPUT * MHC_COEFFICIENTS) as usize,
                ),
                (
                    "mhc_attention_bias",
                    self.mhc_attention_bias.as_ref(),
                    MHC_COEFFICIENTS as usize,
                ),
                (
                    "mhc_attention_control",
                    self.mhc_attention_control.as_ref(),
                    4,
                ),
                (
                    "mhc_moe_predictor",
                    self.mhc_moe_predictor.as_ref(),
                    (MHC_INPUT * MHC_COEFFICIENTS) as usize,
                ),
                (
                    "mhc_moe_bias",
                    self.mhc_moe_bias.as_ref(),
                    MHC_COEFFICIENTS as usize,
                ),
                ("mhc_moe_control", self.mhc_moe_control.as_ref(), 4),
            ]);
        } else {
            tensors.extend([
                (
                    "feedback_state",
                    self.feedback_state.as_ref(),
                    (WIDTH * WIDTH) as usize,
                ),
                (
                    "feedback_gate",
                    self.feedback_gate.as_ref(),
                    (WIDTH * WIDTH) as usize,
                ),
            ]);
        }
        tensors.push(("final_norm", &self.final_norm, WIDTH as usize));
        if self.architecture == ResidualArchitecture::DiffusionMhc4 {
            tensors.push(("mask_embed", &self.mask_embed, WIDTH as usize));
            tensors.push(("index_query", &self.index_query, 64 * 64));
        }
        tensors
    }

    pub(super) fn row<'a>(&self, buffer: &'a BufferRef) -> Result<CandidateRow<'a>, String> {
        self.row_view(buffer, 0, buffer.length() as usize)
    }

    pub(super) fn row_view<'a>(
        &self,
        buffer: &'a BufferRef,
        base: usize,
        row_bytes: usize,
    ) -> Result<CandidateRow<'a>, String> {
        let mut offsets = std::collections::BTreeMap::new();
        let mut end = 0u64;
        for (name, tensor, _) in self.tensors() {
            offsets.insert(name, base as u64 + end);
            end += tensor.length();
        }
        if end as usize != row_bytes
            || base
                .checked_add(row_bytes)
                .is_none_or(|required| required > buffer.length() as usize)
        {
            return Err(format!(
                "candidate row has {row_bytes} bytes at offset {base}, expected {end} within {} bytes",
                buffer.length(),
            ));
        }
        let offset = |name| offsets.get(name).copied().unwrap_or(0);
        Ok(CandidateRow {
            buffer,
            architecture: self.architecture,
            router: offset("router"),
            qkv: offset("qkv"),
            output: offset("attention_output"),
            gate_up: offset("expert_gate_up"),
            down: offset("expert_down"),
            attention_norm: offset("attention_norm"),
            ffn_norm: offset("ffn_norm"),
            readout: offset("embedding_readout"),
            mask_embed: offset("mask_embed"),
            index_query: offset("index_query"),
            feedback_state: offset("feedback_state"),
            feedback_gate: offset("feedback_gate"),
            mhc_attention_predictor: offset("mhc_attention_predictor"),
            mhc_attention_bias: offset("mhc_attention_bias"),
            mhc_attention_control: offset("mhc_attention_control"),
            mhc_moe_predictor: offset("mhc_moe_predictor"),
            mhc_moe_bias: offset("mhc_moe_bias"),
            mhc_moe_control: offset("mhc_moe_control"),
            final_norm: offset("final_norm"),
        })
    }

    pub(super) fn search(
        &self,
        perturbation: crate::Perturbation,
        length: crate::trust_region::TRLengthConfig,
    ) -> Result<(SearchState, UpdateLog), String> {
        self.search_shaped(
            perturbation,
            length,
            crate::config::TrustRegionShape::TensorFamilyStatic,
        )
    }

    pub(super) fn search_shaped(
        &self,
        perturbation: crate::Perturbation,
        length: crate::trust_region::TRLengthConfig,
        shape: crate::config::TrustRegionShape,
    ) -> Result<(SearchState, UpdateLog), String> {
        let expected_parameters = self.architecture.parameter_count();
        let mut base = Vec::with_capacity(expected_parameters);
        let mut blocks = Vec::new();
        let mut groups = Vec::new();
        let mut updates = UpdateLog::default();
        for (family_code, (family, buffer, block_len)) in self.tensors().into_iter().enumerate() {
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
                let rms_floor = if matches!(family, "mhc_attention_control" | "mhc_moe_control") {
                    0.02
                } else {
                    1e-6
                };
                let scale = initial_rms.max(rms_floor) as f32 * trust_multiplier;
                let (layer, expert) = match family {
                    "expert_gate_up" | "expert_down" => (
                        Some(index / routing::MOE_EXPERTS as usize),
                        Some(index % routing::MOE_EXPERTS as usize),
                    ),
                    "embedding_readout" | "feedback_state" | "feedback_gate" | "final_norm"
                    | "mask_embed" | "index_query" => (None, None),
                    _ => (Some(index), None),
                };
                let name = match (layer, expert) {
                    (Some(layer), Some(expert)) => {
                        format!("layer.{layer}.expert.{expert}.{family}")
                    }
                    (Some(layer), None) => format!("layer.{layer}.{family}"),
                    _ => family.to_string(),
                };
                let logical_start = index
                    .checked_mul(block_len)
                    .ok_or("tensor block logical offset overflow")?;
                let tensor_id = crate::hash::tensor_key(family);
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
                let address = ProgramAddress::new(family_code, layer, expert)?;
                blocks.push(
                    ParamBlock::new_logical(
                        tensor_id,
                        logical_start as u64,
                        base.len(),
                        tensor.len(),
                        scale,
                        1.0 / (tensor.len() as f32 * scale * scale),
                    )?
                    .with_address(address),
                );
                groups.push(family_group(family));
                base.extend_from_slice(tensor);
            }
        }
        if base.len() != expected_parameters {
            return Err(format!(
                "Full-weight inventory has {} parameters, expected {expected_parameters}",
                base.len()
            ));
        }
        let mut search =
            SearchState::new_implicit(&base, blocks, HISTORY_CAPACITY, length, perturbation)?;
        search.set_tolerance(4)?;
        if shape == crate::config::TrustRegionShape::TensorFamilyLearned {
            search.enable_family(groups)?;
        }
        Ok((search, updates))
    }
}

pub(super) fn canary(runtime: &Runtime, elements: usize) -> Buffer {
    runtime.buffer_with(&vec![0x7e00u16; elements + 64])
}
