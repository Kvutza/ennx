//! Resident Metal resources for the experimental KDA-MoE forward path.
//!
//! This module deliberately owns no Python objects. Callers upload packed
//! weights once, then reuse the same buffers for every seeded BO round.

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::sync::Arc;

use metal::{Buffer, ComputePipelineState};

use crate::apple_gpu::{thread_group, Runtime};
use crate::forward_program::{KdaControlRequest, KdaMoeLayerRequest, KdaPackedLinear};
use crate::trials::{Search, Trial};
use crate::weights::ComputeDevice;

const KERNELS: &[&str] = &[
    "decoder_rms_norm",
    "kda_project_packed_16k",
    "kda_split_qkv_16k",
    "kda_split_control_16k",
    "kda_make_gate_beta_16k",
    "kda_recurrence_16k",
    "kda_prefill_16k",
    "kda_decode_step",
    "kda_postprocess_16k",
    "decoder_project_packed",
    "decoder_project_packed_simd",
    "packed_dequantize_row_half",
    "packed_dequantize_row_float",
    "kda_short_conv_decode",
    "kda_normalize_qk",
    "decoder_residual",
    "moe_router_topk",
    "moe_router_topk_simd",
    "moe_gate_up",
    "moe_gate_up_simd",
    "moe_down",
    "moe_down_simd",
    "packed_embedding_lookup",
    "decoder_argmax",
];

const KDA_MOE_SOURCE: &str = include_str!("kda_moe.metal");

/// Immutable packed model arenas uploaded once per resident evaluator.
pub struct KdaMoeMetalWeights<'a> {
    pub packed: &'a [u8],
    pub scales: &'a [f32],
    pub biases: &'a [f32],
}

/// Shared immutable model arenas. Cloning a Metal handle is cheap; all layer
/// executors refer to these three allocations instead of copying the model.
pub struct KdaMoeMetalArena {
    packed: Buffer,
    scales: Buffer,
    biases: Buffer,
    packed_bytes: usize,
}

impl KdaMoeMetalArena {
    pub fn new(weights: KdaMoeMetalWeights<'_>) -> Result<Self, String> {
        if weights.packed.is_empty() || weights.scales.is_empty() || weights.biases.is_empty() {
            return Err("KDA-MoE model arenas cannot be empty".to_string());
        }
        let runtime = Runtime::shared()?;
        let packed = runtime.buffer_with(weights.packed);
        let scales = runtime.buffer_with(weights.scales);
        let biases = runtime.buffer_with(weights.biases);
        if packed.contents().is_null() || scales.contents().is_null() || biases.contents().is_null()
        {
            return Err("Metal could not allocate KDA-MoE model arenas".to_string());
        }
        Ok(Self {
            packed,
            scales,
            biases,
            packed_bytes: weights.packed.len(),
        })
    }

    pub fn packed_buffer(&self) -> Buffer {
        self.packed.to_owned()
    }

    pub fn packed_bytes(&self) -> usize {
        self.packed_bytes
    }
}

/// Dequantized KDA vectors. They are tiny compared with model weights and are
/// uploaded once so the recurrent path can remain entirely on the GPU.
pub struct KdaMoeMetalKdaVectors<'a> {
    pub decay: &'a [f32],
    pub time_bias: &'a [f32],
    pub output_norm: &'a [f32],
}

/// Sizes of the device-resident forward buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KdaMoeMetalMemory {
    pub hidden_elements: usize,
    pub qkv_elements: usize,
    pub recurrence_state_elements: usize,
    pub route_elements: usize,
    pub expert_activation_elements: usize,
}

/// Compiled fused pipeline set and all per-layer resident storage.
pub struct KdaMoeMetalExecutor {
    #[allow(dead_code)] // Retained for command-buffer encoding methods.
    runtime: Arc<Runtime>,
    pipelines: BTreeMap<&'static str, ComputePipelineState>,
    pub packed: Buffer,
    packed_bytes: usize,
    packed_offset: u64,
    resident_row: bool,
    pub scales: Buffer,
    pub biases: Buffer,
    pub hidden: Buffer,
    pub normalized: Buffer,
    pub qkv: Buffer,
    pub query: Buffer,
    pub key: Buffer,
    pub value: Buffer,
    pub recurrence_state: Buffer,
    pub route_indices: Buffer,
    pub route_weights: Buffer,
    pub expert_activation: Buffer,
    pub update: Buffer,
    attention_norm: Buffer,
    moe_norm: Buffer,
    attention_norm_linear: Buffer,
    moe_norm_linear: Buffer,
    router: Buffer,
    expert_gate: Buffer,
    expert_up: Buffer,
    expert_down: Buffer,
    qkv_linear: Buffer,
    qkv_conv_linear: Buffer,
    qkv_conv_history: Buffer,
    control_linear: Buffer,
    forget_linear: Buffer,
    output_gate_linear: Buffer,
    output_linear: Buffer,
    decay_linear: Buffer,
    time_bias_linear: Buffer,
    output_norm_linear: Buffer,
    control: Buffer,
    forget_state: Buffer,
    output_state: Buffer,
    raw_beta: Buffer,
    raw_gate: Buffer,
    output_gate: Buffer,
    post_kda: Buffer,
    kda_gated: Buffer,
    gate: Buffer,
    beta: Buffer,
    decay: Buffer,
    time_bias: Buffer,
    output_norm: Buffer,
    params: MetalDecoderParams,
    kda_params: MetalKdaParams,
    kda_control_params: MetalKdaControlParams,
    memory: KdaMoeMetalMemory,
}

impl KdaMoeMetalExecutor {
    /// Compile or load every layer pipeline and allocate its resident buffers.
    /// `ComputeDevice::Agx` uses ENNX's Metal binary archive cache.
    pub fn new(
        request: &KdaMoeLayerRequest,
        control: KdaControlRequest,
        vectors: KdaMoeMetalKdaVectors<'_>,
        device: ComputeDevice,
        weights: KdaMoeMetalWeights<'_>,
    ) -> Result<Self, String> {
        let arena = KdaMoeMetalArena::new(weights)?;
        Self::new_with_arena(request, control, vectors, device, &arena)
    }

    pub fn new_with_arena(
        request: &KdaMoeLayerRequest,
        control: KdaControlRequest,
        vectors: KdaMoeMetalKdaVectors<'_>,
        device: ComputeDevice,
        arena: &KdaMoeMetalArena,
    ) -> Result<Self, String> {
        request.validate()?;
        control.validate(request.kda.tensor, request.kda.qkv.input_width)?;
        let agx = match device {
            ComputeDevice::Metal => false,
            ComputeDevice::Agx => true,
            _ => return Err("KDA-MoE Metal executor requires device 'metal' or 'agx'".to_string()),
        };
        let runtime = Runtime::shared()?;
        let mut pipelines = BTreeMap::new();
        for &name in KERNELS {
            let pipeline = if agx {
                runtime.agx_pipeline(KDA_MOE_SOURCE, "kda-moe", name)?
            } else {
                runtime.pipeline(KDA_MOE_SOURCE, "kda-moe", name)?
            };
            pipelines.insert(name, pipeline);
        }

        let tensor = request.kda.tensor;
        let hidden_elements = request.hidden_elements();
        let qkv_elements = hidden_elements
            .checked_mul(3)
            .ok_or("KDA-MoE QKV buffer size overflow")?;
        let recurrence_state_elements = tensor.state_elements();
        let route_elements = tensor
            .batch
            .checked_mul(tensor.sequence_length)
            .and_then(|n| n.checked_mul(request.top_k))
            .ok_or("KDA-MoE route buffer size overflow")?;
        let expert_activation_elements = request.expert_activation_elements();
        let memory = KdaMoeMetalMemory {
            hidden_elements,
            qkv_elements,
            recurrence_state_elements,
            route_elements,
            expert_activation_elements,
        };
        let params = MetalDecoderParams {
            batch: u32::try_from(tensor.batch).map_err(|_| "batch exceeds u32")?,
            length: u32::try_from(tensor.sequence_length)
                .map_err(|_| "sequence length exceeds u32")?,
            hidden_width: u32::try_from(request.kda.qkv.input_width)
                .map_err(|_| "hidden width exceeds u32")?,
            experts: u32::try_from(request.expert_gate.len())
                .map_err(|_| "expert count exceeds u32")?,
            top_k: u32::try_from(request.top_k).map_err(|_| "top_k exceeds u32")?,
            expert_width: u32::try_from(request.expert_gate[0].output_width)
                .map_err(|_| "expert width exceeds u32")?,
            residual_scale: request.residual_scale,
            rms_epsilon: request.rms_epsilon,
        };
        let kda_params = MetalKdaParams {
            batch: u32::try_from(tensor.batch).map_err(|_| "batch exceeds u32")?,
            length: u32::try_from(tensor.sequence_length)
                .map_err(|_| "sequence length exceeds u32")?,
            heads: u32::try_from(tensor.heads).map_err(|_| "head count exceeds u32")?,
            key_width: u32::try_from(tensor.key_width).map_err(|_| "key width exceeds u32")?,
            value_width: u32::try_from(tensor.value_width)
                .map_err(|_| "value width exceeds u32")?,
        };
        let kda_control_params = MetalKdaControlParams {
            batch: kda_params.batch,
            length: kda_params.length,
            heads: kda_params.heads,
            head_width: kda_params.value_width,
            gate_rank: u32::try_from(control.gate_rank).map_err(|_| "KDA gate rank exceeds u32")?,
            rms_epsilon: params.rms_epsilon,
        };
        let tokens = tensor.batch * tensor.sequence_length;
        let hidden = request.kda.qkv.input_width;
        if vectors.decay.len() != tensor.heads
            || vectors.time_bias.len() != hidden
            || vectors.output_norm.len() != tensor.value_width
        {
            return Err("KDA vector dimensions do not match the tensor layout".to_string());
        }
        Ok(Self {
            packed: arena.packed.to_owned(),
            packed_bytes: arena.packed_bytes,
            packed_offset: 0,
            resident_row: false,
            scales: arena.scales.to_owned(),
            biases: arena.biases.to_owned(),
            hidden: runtime.buffer::<u16>(hidden_elements),
            normalized: runtime.buffer::<u16>(hidden_elements),
            qkv: runtime.buffer::<u16>(qkv_elements),
            query: runtime.buffer::<u16>(hidden_elements),
            key: runtime.buffer::<u16>(hidden_elements),
            value: runtime.buffer::<u16>(hidden_elements),
            recurrence_state: runtime.buffer::<f32>(recurrence_state_elements),
            route_indices: runtime.buffer::<u32>(route_elements),
            route_weights: runtime.buffer::<u16>(route_elements),
            expert_activation: runtime.buffer::<u16>(expert_activation_elements),
            update: runtime.buffer::<u16>(hidden_elements),
            attention_norm: runtime.buffer::<u16>(request.kda.qkv.input_width),
            moe_norm: runtime.buffer::<u16>(request.kda.qkv.input_width),
            attention_norm_linear: runtime
                .buffer_with(&[MetalPackedLinear::from(request.attention_norm)]),
            moe_norm_linear: runtime.buffer_with(&[MetalPackedLinear::from(request.moe_norm)]),
            router: runtime.buffer_with(&[MetalPackedLinear::from(request.router)]),
            expert_gate: runtime.buffer_with(&metal_linears(&request.expert_gate)),
            expert_up: runtime.buffer_with(&metal_linears(&request.expert_up)),
            expert_down: runtime.buffer_with(&metal_linears(&request.expert_down)),
            qkv_linear: runtime.buffer_with(&[MetalPackedLinear::from(request.kda.qkv)]),
            qkv_conv_linear: runtime.buffer_with(&[MetalPackedLinear::from(control.qkv_conv)]),
            qkv_conv_history: runtime
                .buffer::<u16>(tensor.batch * 3 * control.qkv_conv.output_width),
            control_linear: runtime.buffer_with(&[MetalPackedLinear::from(control.control)]),
            forget_linear: runtime.buffer_with(&[MetalPackedLinear::from(control.forget)]),
            output_gate_linear: runtime
                .buffer_with(&[MetalPackedLinear::from(control.output_gate)]),
            output_linear: runtime.buffer_with(&[MetalPackedLinear::from(control.output)]),
            decay_linear: runtime.buffer_with(&[MetalPackedLinear::from(control.decay)]),
            time_bias_linear: runtime.buffer_with(&[MetalPackedLinear::from(control.time_bias)]),
            output_norm_linear: runtime
                .buffer_with(&[MetalPackedLinear::from(control.output_norm)]),
            control: runtime.buffer::<u16>(tokens * (2 * control.gate_rank + tensor.heads)),
            forget_state: runtime.buffer::<u16>(tokens * control.gate_rank),
            output_state: runtime.buffer::<u16>(tokens * control.gate_rank),
            raw_beta: runtime.buffer::<u16>(tokens * tensor.heads),
            raw_gate: runtime.buffer::<u16>(hidden_elements),
            output_gate: runtime.buffer::<u16>(hidden_elements),
            post_kda: runtime.buffer::<u16>(hidden_elements),
            kda_gated: runtime.buffer::<u16>(hidden_elements),
            gate: runtime.buffer::<f32>(tensor.qkv_elements()),
            beta: runtime.buffer::<f32>(tensor.batch * tensor.sequence_length * tensor.heads),
            decay: runtime.buffer_with(vectors.decay),
            time_bias: runtime.buffer_with(vectors.time_bias),
            output_norm: runtime.buffer_with(vectors.output_norm),
            params,
            kda_params,
            kda_control_params,
            runtime,
            pipelines,
            memory,
        })
    }

    pub fn memory(&self) -> KdaMoeMetalMemory {
        self.memory
    }

    pub fn pipeline(&self, name: &str) -> Result<&ComputePipelineState, String> {
        self.pipelines
            .get(name)
            .ok_or_else(|| format!("KDA-MoE pipeline {name:?} is not loaded"))
    }

    /// Bind the packed row selected by the resident ENN search.
    ///
    /// The search owns the allocation and materializes the candidate into a
    /// slot. This executor keeps another Metal handle to that allocation; it
    /// does not copy the model row. Kernels then read the row as-is.
    pub fn bind_resident_row(
        &mut self,
        rows: &Buffer,
        offset: usize,
        row_bytes: usize,
    ) -> Result<(), String> {
        if row_bytes < self.packed_bytes {
            return Err(format!(
                "resident row has {row_bytes} bytes, needs at least {}",
                self.packed_bytes
            ));
        }
        if offset % 4 != 0 {
            return Err("resident row offset must be four-byte aligned".to_string());
        }
        let end = offset
            .checked_add(self.packed_bytes)
            .ok_or("resident row offset overflow")?;
        if end > rows.length() as usize {
            return Err("resident row exceeds the search model arena".to_string());
        }
        self.packed = rows.to_owned();
        self.packed_offset = offset as u64;
        self.resident_row = true;
        Ok(())
    }

    pub fn bind_pending_search(&mut self, search: &Search, trial: Trial) -> Result<(), String> {
        let (rows, offset) = search.pending_metal_row(trial)?;
        self.bind_resident_row(&rows, offset, search.row_bytes())
    }

    fn set_packed(&self, encoder: &metal::ComputeCommandEncoderRef, index: u64) {
        encoder.set_buffer(index, Some(&self.packed), self.packed_offset);
    }

    fn seed(&self, value: u64) -> MetalSeed {
        if self.resident_row {
            MetalSeed::disabled()
        } else {
            MetalSeed::from(value)
        }
    }

    pub fn upload_hidden(&mut self, hidden: &[u16]) -> Result<(), String> {
        if hidden.len() != self.memory.hidden_elements {
            return Err(format!(
                "hidden input has {} elements, expected {}",
                hidden.len(),
                self.memory.hidden_elements
            ));
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                hidden.as_ptr(),
                self.hidden.contents().cast::<u16>(),
                hidden.len(),
            );
        }
        Ok(())
    }

    pub fn upload_norms(&mut self, attention: &[u16], moe: &[u16]) -> Result<(), String> {
        let width = usize::try_from(self.params.hidden_width).unwrap();
        if attention.len() != width || moe.len() != width {
            return Err(format!("decoder norm vectors need {width} elements"));
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                attention.as_ptr(),
                self.attention_norm.contents().cast(),
                width,
            );
            std::ptr::copy_nonoverlapping(moe.as_ptr(), self.moe_norm.contents().cast(), width);
        }
        Ok(())
    }

    /// Clear the persistent KDA memory before prefill or a new decode stream.
    pub fn reset_recurrence_state(&mut self) {
        unsafe {
            std::ptr::write_bytes(
                self.recurrence_state.contents().cast::<f32>(),
                0,
                self.memory.recurrence_state_elements,
            );
        }
    }

    pub fn reset_decode_state(&mut self) {
        self.reset_recurrence_state();
        let elements = usize::try_from(self.kda_params.batch).unwrap()
            * 3
            * usize::try_from(self.kda_params.heads).unwrap()
            * usize::try_from(self.kda_params.key_width).unwrap()
            * 3;
        unsafe {
            std::ptr::write_bytes(self.qkv_conv_history.contents().cast::<u16>(), 0, elements);
        }
    }

    /// Upload a single token's already-projected KDA inputs.
    ///
    /// This narrow entry point is also the numerical seam used to validate
    /// the persistent recurrence independently of projection and MoE kernels.
    pub fn upload_kda_decode_inputs(
        &mut self,
        query: &[u16],
        key: &[u16],
        value: &[u16],
        gate: &[f32],
        beta: &[f32],
    ) -> Result<(), String> {
        let qkv = usize::try_from(self.kda_params.batch).unwrap()
            * usize::try_from(self.kda_params.heads).unwrap()
            * usize::try_from(self.kda_params.key_width).unwrap();
        let values = usize::try_from(self.kda_params.batch).unwrap()
            * usize::try_from(self.kda_params.heads).unwrap()
            * usize::try_from(self.kda_params.value_width).unwrap();
        let rates = usize::try_from(self.kda_params.batch).unwrap()
            * usize::try_from(self.kda_params.heads).unwrap();
        if query.len() != qkv || key.len() != qkv || gate.len() != qkv {
            return Err(format!(
                "single-token query, key, and gate need {qkv} elements"
            ));
        }
        if value.len() != values {
            return Err(format!("single-token value needs {values} elements"));
        }
        if beta.len() != rates {
            return Err(format!("single-token beta needs {rates} elements"));
        }
        unsafe {
            std::ptr::copy_nonoverlapping(query.as_ptr(), self.query.contents().cast(), qkv);
            std::ptr::copy_nonoverlapping(key.as_ptr(), self.key.contents().cast(), qkv);
            std::ptr::copy_nonoverlapping(value.as_ptr(), self.value.contents().cast(), values);
            std::ptr::copy_nonoverlapping(gate.as_ptr(), self.gate.contents().cast(), qkv);
            std::ptr::copy_nonoverlapping(beta.as_ptr(), self.beta.contents().cast(), rates);
        }
        Ok(())
    }

    /// Advance persistent KDA memory by one token and leave the output on GPU.
    pub fn kda_decode_step(&mut self) -> Result<(), String> {
        let params = MetalKdaParams {
            length: 1,
            ..self.kda_params
        };
        let columns =
            u64::from(params.batch) * u64::from(params.heads) * u64::from(params.value_width);
        let command = self.runtime.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(self.pipeline("kda_decode_step")?);
        encoder.set_buffer(0, Some(&self.query), 0);
        encoder.set_buffer(1, Some(&self.key), 0);
        encoder.set_buffer(2, Some(&self.value), 0);
        encoder.set_buffer(3, Some(&self.gate), 0);
        encoder.set_buffer(4, Some(&self.beta), 0);
        encoder.set_buffer(5, Some(&self.recurrence_state), 0);
        encoder.set_buffer(6, Some(&self.post_kda), 0);
        set_bytes(encoder, 7, &params);
        encoder.dispatch_threads(thread_group(columns), thread_group(256));
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        Ok(())
    }

    fn encode_packed_projection(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        input: &Buffer,
        output: &Buffer,
        linear: &Buffer,
        seed: &MetalSeed,
        threads: u64,
    ) -> Result<(), String> {
        encoder.set_compute_pipeline_state(self.pipeline("decoder_project_packed_simd")?);
        encoder.set_buffer(0, Some(input), 0);
        self.set_packed(encoder, 1);
        encoder.set_buffer(2, Some(&self.scales), 0);
        encoder.set_buffer(3, Some(&self.biases), 0);
        encoder.set_buffer(4, Some(output), 0);
        encoder.set_buffer(5, Some(linear), 0);
        set_bytes(encoder, 6, seed);
        set_bytes(encoder, 7, &self.params);
        encoder.dispatch_threads(thread_group(threads * 32), thread_group(256));
        Ok(())
    }

    fn encode_dequantized_vector(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        pipeline: &str,
        output: &Buffer,
        linear: &Buffer,
        seed: &MetalSeed,
        elements: u64,
    ) -> Result<(), String> {
        encoder.set_compute_pipeline_state(self.pipeline(pipeline)?);
        self.set_packed(encoder, 0);
        encoder.set_buffer(1, Some(&self.scales), 0);
        encoder.set_buffer(2, Some(&self.biases), 0);
        encoder.set_buffer(3, Some(output), 0);
        encoder.set_buffer(4, Some(linear), 0);
        set_bytes(encoder, 5, seed);
        encoder.dispatch_threads(thread_group(elements), thread_group(256));
        Ok(())
    }

    fn encode_prepare_candidate(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        seed: u64,
    ) -> Result<(), String> {
        let seed = self.seed(seed);
        self.encode_dequantized_vector(
            encoder,
            "packed_dequantize_row_half",
            &self.attention_norm,
            &self.attention_norm_linear,
            &seed,
            u64::from(self.params.hidden_width),
        )?;
        self.encode_dequantized_vector(
            encoder,
            "packed_dequantize_row_half",
            &self.moe_norm,
            &self.moe_norm_linear,
            &seed,
            u64::from(self.params.hidden_width),
        )?;
        self.encode_dequantized_vector(
            encoder,
            "packed_dequantize_row_float",
            &self.decay,
            &self.decay_linear,
            &seed,
            u64::from(self.kda_params.heads),
        )?;
        self.encode_dequantized_vector(
            encoder,
            "packed_dequantize_row_float",
            &self.time_bias,
            &self.time_bias_linear,
            &seed,
            u64::from(self.params.hidden_width),
        )?;
        self.encode_dequantized_vector(
            encoder,
            "packed_dequantize_row_float",
            &self.output_norm,
            &self.output_norm_linear,
            &seed,
            u64::from(self.kda_params.value_width),
        )
    }

    pub fn prepare_candidate(&mut self, seed: u64) -> Result<(), String> {
        let command = self.runtime.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        self.encode_prepare_candidate(encoder, seed)?;
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        Ok(())
    }

    /// Execute one exact KDA-MoE decoder token in a single command buffer.
    /// Projection weights, convolution history, recurrence memory, routing,
    /// and selected-expert activations remain resident throughout the call.
    fn encode_decode_layer(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        seed: u64,
    ) -> Result<(), String> {
        if self.params.batch != 1 || self.params.length != 1 {
            return Err(
                "resident layer decode currently requires shape [1, 1, hidden]".to_string(),
            );
        }
        let seed = self.seed(seed);
        let tokens = u64::from(self.params.batch) * u64::from(self.params.length);
        let hidden = tokens * u64::from(self.params.hidden_width);
        let qkv_width = u64::from(self.kda_params.heads) * u64::from(self.kda_params.key_width) * 3;
        let qkv_elements = qkv_width / 3;
        let control_width =
            2 * u64::from(self.kda_control_params.gate_rank) + u64::from(self.kda_params.heads);
        let recurrence_columns = u64::from(self.kda_params.batch)
            * u64::from(self.kda_params.heads)
            * u64::from(self.kda_params.value_width);
        let expert_activations =
            tokens * u64::from(self.params.top_k) * u64::from(self.params.expert_width);
        encoder.set_compute_pipeline_state(self.pipeline("decoder_rms_norm")?);
        encoder.set_buffer(0, Some(&self.hidden), 0);
        encoder.set_buffer(1, Some(&self.attention_norm), 0);
        encoder.set_buffer(2, Some(&self.normalized), 0);
        set_bytes(encoder, 3, &self.params);
        encoder.dispatch_threads(thread_group(tokens), thread_group(1));

        self.encode_packed_projection(
            encoder,
            &self.normalized,
            &self.qkv,
            &self.qkv_linear,
            &seed,
            tokens * qkv_width,
        )?;

        encoder.set_compute_pipeline_state(self.pipeline("kda_short_conv_decode")?);
        encoder.set_buffer(0, Some(&self.qkv), 0);
        encoder.set_buffer(1, Some(&self.qkv_conv_history), 0);
        self.set_packed(encoder, 2);
        encoder.set_buffer(3, Some(&self.scales), 0);
        encoder.set_buffer(4, Some(&self.biases), 0);
        encoder.set_buffer(5, Some(&self.qkv_conv_linear), 0);
        set_bytes(encoder, 6, &seed);
        encoder.dispatch_threads(thread_group(qkv_width), thread_group(256));

        encoder.set_compute_pipeline_state(self.pipeline("kda_split_qkv_16k")?);
        encoder.set_buffer(0, Some(&self.qkv), 0);
        encoder.set_buffer(1, Some(&self.query), 0);
        encoder.set_buffer(2, Some(&self.key), 0);
        encoder.set_buffer(3, Some(&self.value), 0);
        set_bytes(encoder, 4, &self.kda_params);
        encoder.dispatch_threads(thread_group(qkv_elements), thread_group(256));

        encoder.set_compute_pipeline_state(self.pipeline("kda_normalize_qk")?);
        encoder.set_buffer(0, Some(&self.query), 0);
        encoder.set_buffer(1, Some(&self.key), 0);
        set_bytes(encoder, 2, &self.kda_params);
        encoder.dispatch_threads(
            thread_group(u64::from(self.kda_params.batch) * u64::from(self.kda_params.heads)),
            thread_group(32),
        );

        self.encode_packed_projection(
            encoder,
            &self.normalized,
            &self.control,
            &self.control_linear,
            &seed,
            tokens * control_width,
        )?;

        encoder.set_compute_pipeline_state(self.pipeline("kda_split_control_16k")?);
        encoder.set_buffer(0, Some(&self.control), 0);
        encoder.set_buffer(1, Some(&self.forget_state), 0);
        encoder.set_buffer(2, Some(&self.output_state), 0);
        encoder.set_buffer(3, Some(&self.raw_beta), 0);
        set_bytes(encoder, 4, &self.kda_control_params);
        encoder.dispatch_threads(thread_group(tokens * control_width), thread_group(256));

        self.encode_packed_projection(
            encoder,
            &self.forget_state,
            &self.raw_gate,
            &self.forget_linear,
            &seed,
            hidden,
        )?;
        self.encode_packed_projection(
            encoder,
            &self.output_state,
            &self.output_gate,
            &self.output_gate_linear,
            &seed,
            hidden,
        )?;

        encoder.set_compute_pipeline_state(self.pipeline("kda_make_gate_beta_16k")?);
        encoder.set_buffer(0, Some(&self.raw_gate), 0);
        encoder.set_buffer(1, Some(&self.raw_beta), 0);
        encoder.set_buffer(2, Some(&self.decay), 0);
        encoder.set_buffer(3, Some(&self.time_bias), 0);
        encoder.set_buffer(4, Some(&self.gate), 0);
        encoder.set_buffer(5, Some(&self.beta), 0);
        set_bytes(encoder, 6, &self.kda_control_params);
        encoder.dispatch_threads(thread_group(hidden), thread_group(256));

        encoder.set_compute_pipeline_state(self.pipeline("kda_decode_step")?);
        encoder.set_buffer(0, Some(&self.query), 0);
        encoder.set_buffer(1, Some(&self.key), 0);
        encoder.set_buffer(2, Some(&self.value), 0);
        encoder.set_buffer(3, Some(&self.gate), 0);
        encoder.set_buffer(4, Some(&self.beta), 0);
        encoder.set_buffer(5, Some(&self.recurrence_state), 0);
        encoder.set_buffer(6, Some(&self.post_kda), 0);
        set_bytes(encoder, 7, &self.kda_params);
        encoder.dispatch_threads(thread_group(recurrence_columns), thread_group(256));

        encoder.set_compute_pipeline_state(self.pipeline("kda_postprocess_16k")?);
        encoder.set_buffer(0, Some(&self.post_kda), 0);
        encoder.set_buffer(1, Some(&self.output_gate), 0);
        encoder.set_buffer(2, Some(&self.output_norm), 0);
        encoder.set_buffer(3, Some(&self.kda_gated), 0);
        set_bytes(encoder, 4, &self.kda_control_params);
        encoder.dispatch_threads(thread_group(hidden), thread_group(256));

        self.encode_packed_projection(
            encoder,
            &self.kda_gated,
            &self.update,
            &self.output_linear,
            &seed,
            hidden,
        )?;

        encoder.set_compute_pipeline_state(self.pipeline("decoder_residual")?);
        encoder.set_buffer(0, Some(&self.hidden), 0);
        encoder.set_buffer(1, Some(&self.update), 0);
        encoder.set_buffer(2, Some(&self.hidden), 0);
        set_bytes(encoder, 3, &self.params);
        encoder.dispatch_threads(thread_group(hidden), thread_group(256));

        encoder.set_compute_pipeline_state(self.pipeline("decoder_rms_norm")?);
        encoder.set_buffer(0, Some(&self.hidden), 0);
        encoder.set_buffer(1, Some(&self.moe_norm), 0);
        encoder.set_buffer(2, Some(&self.normalized), 0);
        set_bytes(encoder, 3, &self.params);
        encoder.dispatch_threads(thread_group(tokens), thread_group(1));

        encoder.set_compute_pipeline_state(self.pipeline("moe_router_topk_simd")?);
        encoder.set_buffer(0, Some(&self.normalized), 0);
        self.set_packed(encoder, 1);
        encoder.set_buffer(2, Some(&self.scales), 0);
        encoder.set_buffer(3, Some(&self.biases), 0);
        encoder.set_buffer(4, Some(&self.route_indices), 0);
        encoder.set_buffer(5, Some(&self.route_weights), 0);
        encoder.set_buffer(6, Some(&self.router), 0);
        set_bytes(encoder, 7, &seed);
        set_bytes(encoder, 8, &self.params);
        encoder.dispatch_thread_groups(thread_group(tokens), thread_group(512));

        encoder.set_compute_pipeline_state(self.pipeline("moe_gate_up_simd")?);
        encoder.set_buffer(0, Some(&self.normalized), 0);
        self.set_packed(encoder, 1);
        encoder.set_buffer(2, Some(&self.scales), 0);
        encoder.set_buffer(3, Some(&self.biases), 0);
        encoder.set_buffer(4, Some(&self.route_indices), 0);
        encoder.set_buffer(5, Some(&self.expert_activation), 0);
        encoder.set_buffer(6, Some(&self.expert_gate), 0);
        encoder.set_buffer(7, Some(&self.expert_up), 0);
        set_bytes(encoder, 8, &seed);
        set_bytes(encoder, 9, &self.params);
        encoder.dispatch_threads(thread_group(expert_activations * 32), thread_group(256));

        encoder.set_compute_pipeline_state(self.pipeline("moe_down_simd")?);
        encoder.set_buffer(0, Some(&self.expert_activation), 0);
        self.set_packed(encoder, 1);
        encoder.set_buffer(2, Some(&self.scales), 0);
        encoder.set_buffer(3, Some(&self.biases), 0);
        encoder.set_buffer(4, Some(&self.route_weights), 0);
        encoder.set_buffer(5, Some(&self.expert_down), 0);
        encoder.set_buffer(6, Some(&self.route_indices), 0);
        encoder.set_buffer(7, Some(&self.update), 0);
        set_bytes(encoder, 8, &seed);
        set_bytes(encoder, 9, &self.params);
        encoder.dispatch_threads(thread_group(hidden * 32), thread_group(256));

        encoder.set_compute_pipeline_state(self.pipeline("decoder_residual")?);
        encoder.set_buffer(0, Some(&self.hidden), 0);
        encoder.set_buffer(1, Some(&self.update), 0);
        encoder.set_buffer(2, Some(&self.hidden), 0);
        set_bytes(encoder, 3, &self.params);
        encoder.dispatch_threads(thread_group(hidden), thread_group(256));
        Ok(())
    }

    pub fn decode_layer(&mut self, seed: u64) -> Result<(), String> {
        let command = self.runtime.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        self.encode_decode_layer(encoder, seed)?;
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        Ok(())
    }

    pub fn recurrence_state(&self) -> Vec<f32> {
        unsafe {
            std::slice::from_raw_parts(
                self.recurrence_state.contents().cast::<f32>(),
                self.memory.recurrence_state_elements,
            )
            .to_vec()
        }
    }

    pub fn kda_decode_output(&self) -> Vec<u16> {
        let elements = usize::try_from(self.kda_params.batch).unwrap()
            * usize::try_from(self.kda_params.heads).unwrap()
            * usize::try_from(self.kda_params.value_width).unwrap();
        unsafe {
            std::slice::from_raw_parts(self.post_kda.contents().cast::<u16>(), elements).to_vec()
        }
    }

    /// Run the first resident operation in a decoder layer. The normalization
    /// vector is uploaded once per layer; hidden state never leaves Metal.
    pub fn attention_rms_norm(
        &mut self,
        norm: &[u16],
        batch: usize,
        sequence_length: usize,
        epsilon: f32,
    ) -> Result<(), String> {
        let norm_buffer = self.attention_norm.to_owned();
        self.encode_rms_norm(
            "decoder_rms_norm",
            norm,
            &norm_buffer,
            batch,
            sequence_length,
            epsilon,
        )
    }

    pub fn moe_rms_norm(
        &mut self,
        norm: &[u16],
        batch: usize,
        sequence_length: usize,
        epsilon: f32,
    ) -> Result<(), String> {
        let norm_buffer = self.moe_norm.to_owned();
        self.encode_rms_norm(
            "decoder_rms_norm",
            norm,
            &norm_buffer,
            batch,
            sequence_length,
            epsilon,
        )
    }

    fn encode_rms_norm(
        &mut self,
        pipeline_name: &str,
        norm: &[u16],
        norm_buffer: &Buffer,
        batch: usize,
        sequence_length: usize,
        epsilon: f32,
    ) -> Result<(), String> {
        let hidden_width = self.memory.hidden_elements / (batch * sequence_length);
        if norm.len() != hidden_width {
            return Err(format!(
                "RMSNorm has {} elements, expected {hidden_width}",
                norm.len()
            ));
        }
        if batch == 0
            || sequence_length == 0
            || batch * sequence_length * hidden_width != self.memory.hidden_elements
        {
            return Err("RMSNorm shape does not match resident hidden buffer".to_string());
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                norm.as_ptr(),
                norm_buffer.contents().cast::<u16>(),
                norm.len(),
            );
        }
        let params = MetalDecoderParams {
            batch: u32::try_from(batch).map_err(|_| "batch exceeds u32")?,
            length: u32::try_from(sequence_length).map_err(|_| "sequence length exceeds u32")?,
            hidden_width: u32::try_from(hidden_width).map_err(|_| "hidden width exceeds u32")?,
            rms_epsilon: epsilon,
            ..self.params
        };
        let command = self.runtime.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(self.pipeline(pipeline_name)?);
        encoder.set_buffer(0, Some(&self.hidden), 0);
        encoder.set_buffer(1, Some(norm_buffer), 0);
        encoder.set_buffer(2, Some(&self.normalized), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<MetalDecoderParams>() as u64,
            (&params as *const MetalDecoderParams).cast::<c_void>(),
        );
        encoder.dispatch_threads(
            thread_group((batch * sequence_length) as u64),
            thread_group(256),
        );
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        Ok(())
    }

    /// Route and evaluate only the selected MoE experts. All operations are
    /// encoded in one command buffer; neither indices nor activations return
    /// to the host.
    pub fn moe(&mut self, seed: u64) -> Result<(), String> {
        let seed = self.seed(seed);
        let tokens = u64::from(self.params.batch) * u64::from(self.params.length);
        let expert_activations =
            tokens * u64::from(self.params.top_k) * u64::from(self.params.expert_width);
        let hidden = tokens * u64::from(self.params.hidden_width);
        let command = self.runtime.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(self.pipeline("moe_router_topk_simd")?);
        encoder.set_buffer(0, Some(&self.normalized), 0);
        self.set_packed(encoder, 1);
        encoder.set_buffer(2, Some(&self.scales), 0);
        encoder.set_buffer(3, Some(&self.biases), 0);
        encoder.set_buffer(4, Some(&self.route_indices), 0);
        encoder.set_buffer(5, Some(&self.route_weights), 0);
        encoder.set_buffer(6, Some(&self.router), 0);
        set_bytes(encoder, 7, &seed);
        set_bytes(encoder, 8, &self.params);
        encoder.dispatch_thread_groups(thread_group(tokens), thread_group(512));

        encoder.set_compute_pipeline_state(self.pipeline("moe_gate_up_simd")?);
        encoder.set_buffer(0, Some(&self.normalized), 0);
        self.set_packed(encoder, 1);
        encoder.set_buffer(2, Some(&self.scales), 0);
        encoder.set_buffer(3, Some(&self.biases), 0);
        encoder.set_buffer(4, Some(&self.route_indices), 0);
        encoder.set_buffer(5, Some(&self.expert_activation), 0);
        encoder.set_buffer(6, Some(&self.expert_gate), 0);
        encoder.set_buffer(7, Some(&self.expert_up), 0);
        set_bytes(encoder, 8, &seed);
        set_bytes(encoder, 9, &self.params);
        encoder.dispatch_threads(thread_group(expert_activations * 32), thread_group(256));

        encoder.set_compute_pipeline_state(self.pipeline("moe_down_simd")?);
        encoder.set_buffer(0, Some(&self.expert_activation), 0);
        self.set_packed(encoder, 1);
        encoder.set_buffer(2, Some(&self.scales), 0);
        encoder.set_buffer(3, Some(&self.biases), 0);
        encoder.set_buffer(4, Some(&self.route_weights), 0);
        encoder.set_buffer(5, Some(&self.expert_down), 0);
        encoder.set_buffer(6, Some(&self.route_indices), 0);
        encoder.set_buffer(7, Some(&self.update), 0);
        set_bytes(encoder, 8, &seed);
        set_bytes(encoder, 9, &self.params);
        encoder.dispatch_threads(thread_group(hidden * 32), thread_group(256));

        encoder.set_compute_pipeline_state(self.pipeline("decoder_residual")?);
        encoder.set_buffer(0, Some(&self.hidden), 0);
        encoder.set_buffer(1, Some(&self.update), 0);
        encoder.set_buffer(2, Some(&self.hidden), 0);
        set_bytes(encoder, 3, &self.params);
        encoder.dispatch_threads(thread_group(hidden), thread_group(256));
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        Ok(())
    }

    /// Execute a complete KDA attention update without returning control,
    /// gates, state, or activations to the host.
    pub fn kda(&mut self, seed: u64) -> Result<(), String> {
        if self.kda_params.length != 16_384 {
            return Err(
                "the current fused KDA projection kernel requires sequence length 16384"
                    .to_string(),
            );
        }
        let seed = self.seed(seed);
        let tokens = u64::from(self.kda_params.batch) * u64::from(self.kda_params.length);
        let qkv_threads = u64::from(self.kda_params.batch)
            * u64::from(self.kda_params.length)
            * u64::from(self.kda_params.heads)
            * u64::from(self.kda_params.key_width)
            * 3;
        let qkv_elements = qkv_threads / 3;
        let recurrence_threads = u64::from(self.kda_params.batch)
            * u64::from(self.kda_params.heads)
            * u64::from(self.kda_params.value_width);
        let hidden = tokens * u64::from(self.params.hidden_width);
        let control_width =
            2 * u64::from(self.kda_control_params.gate_rank) + u64::from(self.kda_params.heads);
        let command = self.runtime.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(self.pipeline("kda_project_packed_16k")?);
        encoder.set_buffer(0, Some(&self.normalized), 0);
        self.set_packed(encoder, 1);
        encoder.set_buffer(2, Some(&self.scales), 0);
        encoder.set_buffer(3, Some(&self.biases), 0);
        encoder.set_buffer(4, Some(&self.qkv), 0);
        encoder.set_buffer(5, Some(&self.qkv_linear), 0);
        set_bytes(encoder, 6, &seed);
        encoder.dispatch_threads(thread_group(qkv_threads), thread_group(256));

        encoder.set_compute_pipeline_state(self.pipeline("kda_project_packed_16k")?);
        encoder.set_buffer(0, Some(&self.normalized), 0);
        self.set_packed(encoder, 1);
        encoder.set_buffer(2, Some(&self.scales), 0);
        encoder.set_buffer(3, Some(&self.biases), 0);
        encoder.set_buffer(4, Some(&self.control), 0);
        encoder.set_buffer(5, Some(&self.control_linear), 0);
        set_bytes(encoder, 6, &seed);
        encoder.dispatch_threads(thread_group(tokens * control_width), thread_group(256));

        encoder.set_compute_pipeline_state(self.pipeline("kda_split_qkv_16k")?);
        encoder.set_buffer(0, Some(&self.qkv), 0);
        encoder.set_buffer(1, Some(&self.query), 0);
        encoder.set_buffer(2, Some(&self.key), 0);
        encoder.set_buffer(3, Some(&self.value), 0);
        set_bytes(encoder, 4, &self.kda_params);
        encoder.dispatch_threads(thread_group(qkv_elements), thread_group(256));

        encoder.set_compute_pipeline_state(self.pipeline("kda_split_control_16k")?);
        encoder.set_buffer(0, Some(&self.control), 0);
        encoder.set_buffer(1, Some(&self.forget_state), 0);
        encoder.set_buffer(2, Some(&self.output_state), 0);
        encoder.set_buffer(3, Some(&self.raw_beta), 0);
        set_bytes(encoder, 4, &self.kda_control_params);
        encoder.dispatch_threads(thread_group(tokens * control_width), thread_group(256));

        encoder.set_compute_pipeline_state(self.pipeline("kda_project_packed_16k")?);
        encoder.set_buffer(0, Some(&self.forget_state), 0);
        self.set_packed(encoder, 1);
        encoder.set_buffer(2, Some(&self.scales), 0);
        encoder.set_buffer(3, Some(&self.biases), 0);
        encoder.set_buffer(4, Some(&self.raw_gate), 0);
        encoder.set_buffer(5, Some(&self.forget_linear), 0);
        set_bytes(encoder, 6, &seed);
        encoder.dispatch_threads(thread_group(hidden), thread_group(256));

        encoder.set_compute_pipeline_state(self.pipeline("kda_project_packed_16k")?);
        encoder.set_buffer(0, Some(&self.output_state), 0);
        self.set_packed(encoder, 1);
        encoder.set_buffer(2, Some(&self.scales), 0);
        encoder.set_buffer(3, Some(&self.biases), 0);
        encoder.set_buffer(4, Some(&self.output_gate), 0);
        encoder.set_buffer(5, Some(&self.output_gate_linear), 0);
        set_bytes(encoder, 6, &seed);
        encoder.dispatch_threads(thread_group(hidden), thread_group(256));

        encoder.set_compute_pipeline_state(self.pipeline("kda_make_gate_beta_16k")?);
        encoder.set_buffer(0, Some(&self.raw_gate), 0);
        encoder.set_buffer(1, Some(&self.raw_beta), 0);
        encoder.set_buffer(2, Some(&self.decay), 0);
        encoder.set_buffer(3, Some(&self.time_bias), 0);
        encoder.set_buffer(4, Some(&self.gate), 0);
        encoder.set_buffer(5, Some(&self.beta), 0);
        set_bytes(encoder, 6, &self.kda_control_params);
        encoder.dispatch_threads(thread_group(hidden), thread_group(256));

        encoder.set_compute_pipeline_state(self.pipeline("kda_recurrence_16k")?);
        encoder.set_buffer(0, Some(&self.query), 0);
        encoder.set_buffer(1, Some(&self.key), 0);
        encoder.set_buffer(2, Some(&self.value), 0);
        encoder.set_buffer(3, Some(&self.gate), 0);
        encoder.set_buffer(4, Some(&self.beta), 0);
        encoder.set_buffer(5, Some(&self.post_kda), 0);
        encoder.set_buffer(6, Some(&self.recurrence_state), 0);
        set_bytes(encoder, 7, &self.kda_params);
        encoder.dispatch_threads(thread_group(recurrence_threads), thread_group(256));

        encoder.set_compute_pipeline_state(self.pipeline("kda_postprocess_16k")?);
        encoder.set_buffer(0, Some(&self.post_kda), 0);
        encoder.set_buffer(1, Some(&self.output_gate), 0);
        encoder.set_buffer(2, Some(&self.output_norm), 0);
        encoder.set_buffer(3, Some(&self.kda_gated), 0);
        set_bytes(encoder, 4, &self.kda_control_params);
        encoder.dispatch_threads(thread_group(hidden), thread_group(256));

        encoder.set_compute_pipeline_state(self.pipeline("kda_project_packed_16k")?);
        encoder.set_buffer(0, Some(&self.kda_gated), 0);
        self.set_packed(encoder, 1);
        encoder.set_buffer(2, Some(&self.scales), 0);
        encoder.set_buffer(3, Some(&self.biases), 0);
        encoder.set_buffer(4, Some(&self.update), 0);
        encoder.set_buffer(5, Some(&self.output_linear), 0);
        set_bytes(encoder, 6, &seed);
        encoder.dispatch_threads(thread_group(hidden), thread_group(256));

        encoder.set_compute_pipeline_state(self.pipeline("decoder_residual")?);
        encoder.set_buffer(0, Some(&self.hidden), 0);
        encoder.set_buffer(1, Some(&self.update), 0);
        encoder.set_buffer(2, Some(&self.hidden), 0);
        set_bytes(encoder, 3, &self.params);
        encoder.dispatch_threads(thread_group(hidden), thread_group(256));
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        Ok(())
    }
}

/// A complete decoder stack sharing one packed model arena and one hidden
/// buffer. All layers are encoded into one command buffer per generated token.
struct KdaMoeMetalHead {
    embedding_linear: Buffer,
    final_norm_linear: Buffer,
    final_norm: Buffer,
    logits: Buffer,
    token: Buffer,
    next_token: Buffer,
    params: MetalEmbeddingParams,
}

pub struct KdaMoeMetalModel {
    runtime: Arc<Runtime>,
    layers: Vec<KdaMoeMetalExecutor>,
    hidden: Buffer,
    hidden_elements: usize,
    head: Option<KdaMoeMetalHead>,
}

impl KdaMoeMetalModel {
    pub fn new(mut layers: Vec<KdaMoeMetalExecutor>) -> Result<Self, String> {
        let first = layers
            .first()
            .ok_or("resident KDA-MoE model has no layers")?;
        let hidden_elements = first.memory.hidden_elements;
        if first.params.batch != 1 || first.params.length != 1 {
            return Err("resident KDA-MoE model requires single-token layers".to_string());
        }
        if layers.iter().any(|layer| {
            layer.memory.hidden_elements != hidden_elements
                || layer.params.batch != 1
                || layer.params.length != 1
        }) {
            return Err("resident KDA-MoE layer hidden shapes do not match".to_string());
        }
        let runtime = Arc::clone(&first.runtime);
        let hidden = first.hidden.to_owned();
        for layer in &mut layers {
            layer.hidden = hidden.to_owned();
        }
        Ok(Self {
            runtime,
            layers,
            hidden,
            hidden_elements,
            head: None,
        })
    }

    pub fn attach_causal_head(
        &mut self,
        embedding: KdaPackedLinear,
        final_norm: KdaPackedLinear,
        embedding_scale: f32,
    ) -> Result<(), String> {
        embedding.validate()?;
        final_norm.validate()?;
        if embedding.input_width != self.hidden_elements {
            return Err("embedding hidden width does not match decoder".to_string());
        }
        if final_norm.input_width != self.hidden_elements || final_norm.output_width != 1 {
            return Err("final norm must be a hidden-width vector".to_string());
        }
        if !embedding_scale.is_finite() {
            return Err("embedding scale must be finite".to_string());
        }
        self.head = Some(KdaMoeMetalHead {
            embedding_linear: self
                .runtime
                .buffer_with(&[MetalPackedLinear::from(embedding)]),
            final_norm_linear: self
                .runtime
                .buffer_with(&[MetalPackedLinear::from(final_norm)]),
            final_norm: self.runtime.buffer::<u16>(self.hidden_elements),
            logits: self.runtime.buffer::<u16>(embedding.output_width),
            token: self.runtime.buffer_with(&[0_u32]),
            next_token: self.runtime.buffer_with(&[0_u32]),
            params: MetalEmbeddingParams {
                vocab_size: u32::try_from(embedding.output_width)
                    .map_err(|_| "vocabulary exceeds u32")?,
                hidden_width: u32::try_from(self.hidden_elements)
                    .map_err(|_| "hidden width exceeds u32")?,
                embedding_scale,
            },
        });
        Ok(())
    }

    pub fn upload_hidden(&mut self, hidden: &[u16]) -> Result<(), String> {
        if hidden.len() != self.hidden_elements {
            return Err(format!(
                "model hidden input has {} elements, expected {}",
                hidden.len(),
                self.hidden_elements
            ));
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                hidden.as_ptr(),
                self.hidden.contents().cast::<u16>(),
                hidden.len(),
            );
        }
        Ok(())
    }

    pub fn bind_resident_row(
        &mut self,
        rows: &Buffer,
        offset: usize,
        row_bytes: usize,
    ) -> Result<(), String> {
        for layer in &mut self.layers {
            layer.bind_resident_row(rows, offset, row_bytes)?;
        }
        Ok(())
    }

    pub fn bind_pending_search(&mut self, search: &Search, trial: Trial) -> Result<(), String> {
        let (rows, offset) = search.pending_metal_row(trial)?;
        self.bind_resident_row(&rows, offset, search.row_bytes())
    }

    pub fn reset_decode_state(&mut self) {
        for layer in &mut self.layers {
            layer.reset_decode_state();
        }
    }

    pub fn prepare_candidate(&mut self, seed: u64) -> Result<(), String> {
        let command = self.runtime.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        for layer in &self.layers {
            layer.encode_prepare_candidate(encoder, seed)?;
        }
        if let Some(head) = &self.head {
            let first = &self.layers[0];
            let seed = first.seed(seed);
            first.encode_dequantized_vector(
                encoder,
                "packed_dequantize_row_half",
                &head.final_norm,
                &head.final_norm_linear,
                &seed,
                u64::from(head.params.hidden_width),
            )?;
        }
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        Ok(())
    }

    pub fn decode(&mut self, seed: u64) -> Result<(), String> {
        let command = self.runtime.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        for layer in &self.layers {
            layer.encode_decode_layer(encoder, seed)?;
        }
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        Ok(())
    }

    pub fn decode_token(&mut self, token: u32, seed: u64) -> Result<u32, String> {
        let head = self
            .head
            .as_ref()
            .ok_or("resident KDA-MoE model has no causal head")?;
        if token >= head.params.vocab_size {
            return Err(format!("token {token} exceeds the model vocabulary"));
        }
        unsafe {
            head.token.contents().cast::<u32>().write(token);
        }
        let first = &self.layers[0];
        let perturbation = first.seed(seed);
        let command = self.runtime.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(first.pipeline("packed_embedding_lookup")?);
        encoder.set_buffer(0, Some(&head.token), 0);
        first.set_packed(encoder, 1);
        encoder.set_buffer(2, Some(&first.scales), 0);
        encoder.set_buffer(3, Some(&first.biases), 0);
        encoder.set_buffer(4, Some(&self.hidden), 0);
        encoder.set_buffer(5, Some(&head.embedding_linear), 0);
        set_bytes(encoder, 6, &perturbation);
        set_bytes(encoder, 7, &head.params);
        encoder.dispatch_threads(
            thread_group(u64::from(head.params.hidden_width)),
            thread_group(256),
        );

        for layer in &self.layers {
            layer.encode_decode_layer(encoder, seed)?;
        }

        encoder.set_compute_pipeline_state(first.pipeline("decoder_rms_norm")?);
        encoder.set_buffer(0, Some(&self.hidden), 0);
        encoder.set_buffer(1, Some(&head.final_norm), 0);
        encoder.set_buffer(2, Some(&first.normalized), 0);
        set_bytes(encoder, 3, &first.params);
        encoder.dispatch_threads(thread_group(1), thread_group(1));

        first.encode_packed_projection(
            encoder,
            &first.normalized,
            &head.logits,
            &head.embedding_linear,
            &perturbation,
            u64::from(head.params.vocab_size),
        )?;
        encoder.set_compute_pipeline_state(first.pipeline("decoder_argmax")?);
        encoder.set_buffer(0, Some(&head.logits), 0);
        encoder.set_buffer(1, Some(&head.next_token), 0);
        set_bytes(encoder, 2, &head.params.vocab_size);
        encoder.dispatch_thread_groups(thread_group(1), thread_group(256));
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        Ok(unsafe { head.next_token.contents().cast::<u32>().read() })
    }

    pub fn logit_bits(&self) -> Result<Vec<u16>, String> {
        let head = self
            .head
            .as_ref()
            .ok_or("resident KDA-MoE model has no causal head")?;
        Ok(unsafe {
            std::slice::from_raw_parts(
                head.logits.contents().cast::<u16>(),
                usize::try_from(head.params.vocab_size).unwrap(),
            )
            .to_vec()
        })
    }

    pub fn generate(
        &mut self,
        prompt: &[u32],
        max_new_tokens: usize,
        seed: u64,
    ) -> Result<Vec<u32>, String> {
        let (&last, prefix) = prompt
            .split_last()
            .ok_or("generation prompt cannot be empty")?;
        for &token in prefix {
            self.decode_token(token, seed)?;
        }
        let mut next = self.decode_token(last, seed)?;
        let mut output = Vec::with_capacity(prompt.len() + max_new_tokens);
        output.extend_from_slice(prompt);
        for index in 0..max_new_tokens {
            output.push(next);
            if index + 1 < max_new_tokens {
                next = self.decode_token(next, seed)?;
            }
        }
        Ok(output)
    }

    pub fn hidden(&self) -> Vec<u16> {
        unsafe {
            std::slice::from_raw_parts(self.hidden.contents().cast::<u16>(), self.hidden_elements)
                .to_vec()
        }
    }

    pub fn layers_mut(&mut self) -> &mut [KdaMoeMetalExecutor] {
        &mut self.layers
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MetalDecoderParams {
    batch: u32,
    length: u32,
    hidden_width: u32,
    experts: u32,
    top_k: u32,
    expert_width: u32,
    residual_scale: f32,
    rms_epsilon: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MetalKdaParams {
    batch: u32,
    length: u32,
    heads: u32,
    key_width: u32,
    value_width: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MetalKdaControlParams {
    batch: u32,
    length: u32,
    heads: u32,
    head_width: u32,
    gate_rank: u32,
    rms_epsilon: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MetalEmbeddingParams {
    vocab_size: u32,
    hidden_width: u32,
    embedding_scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MetalPackedLinear {
    byte_offset: u32,
    scale_offset: u32,
    bias_offset: u32,
    input_width: u32,
    output_width: u32,
    bits: u32,
    group_size: u32,
    element_offset: u32,
    perturb_whole: u32,
    perturb_threshold: u32,
}

impl From<KdaPackedLinear> for MetalPackedLinear {
    fn from(value: KdaPackedLinear) -> Self {
        Self {
            byte_offset: u32::try_from(value.byte_offset).expect("packed byte offset exceeds u32"),
            scale_offset: u32::try_from(value.scale_offset)
                .expect("packed scale offset exceeds u32"),
            bias_offset: u32::try_from(value.bias_offset).expect("packed bias offset exceeds u32"),
            input_width: u32::try_from(value.input_width).expect("packed input width exceeds u32"),
            output_width: u32::try_from(value.output_width)
                .expect("packed output width exceeds u32"),
            bits: u32::from(value.bits),
            group_size: u32::try_from(value.group_size).expect("packed group size exceeds u32"),
            element_offset: u32::try_from(value.element_offset)
                .expect("packed element offset exceeds u32"),
            perturb_whole: value.perturb_whole,
            perturb_threshold: value.perturb_threshold,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MetalSeed {
    low: u32,
    high: u32,
    enabled: u32,
    _padding: u32,
}

impl From<u64> for MetalSeed {
    fn from(value: u64) -> Self {
        Self {
            low: value as u32,
            high: (value >> 32) as u32,
            enabled: 1,
            _padding: 0,
        }
    }
}

impl MetalSeed {
    fn disabled() -> Self {
        Self {
            low: 0,
            high: 0,
            enabled: 0,
            _padding: 0,
        }
    }
}

fn metal_linears(linears: &[KdaPackedLinear]) -> Vec<MetalPackedLinear> {
    linears
        .iter()
        .copied()
        .map(MetalPackedLinear::from)
        .collect()
}

fn set_bytes<T>(encoder: &metal::ComputeCommandEncoderRef, slot: u64, value: &T) {
    encoder.set_bytes(
        slot,
        std::mem::size_of::<T>() as u64,
        (value as *const T).cast::<c_void>(),
    );
}
