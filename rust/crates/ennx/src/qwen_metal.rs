//! Full dense Qwen2.5-Coder forward evaluation on the shared Apple GPU runtime.
//!
//! The evaluator borrows the canonical BF16 parameter buffer used by the
//! resident ENNX search state. It owns only bounded FP32 work buffers, so a
//! candidate evaluation does not duplicate the model weights.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fs::File;
use std::mem::size_of;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use memmap2::Mmap;
use metal::objc::rc::autoreleasepool;
use metal::{Buffer, CommandBufferRef, ComputePipelineState, MTLCommandBufferStatus, MTLSize};
use serde::Deserialize;

use crate::apple_gpu::{Runtime, gpu_seconds, thread_group, trace};
use crate::fbt_mps::{Matmul as MpsMatmul, Matrix as MpsMatrix};

type Result<T> = std::result::Result<T, String>;
const LINEAR_SOURCE: &str = include_str!("qwen_linear.metal");
const ATTENTION_SOURCE: &str = include_str!("qwen_attention.metal");
const QWEN_SOURCE: &str = include_str!("qwen.metal");
const LOGIT_ROWS: usize = 128;
const PREFILL_CHUNK: u32 = 256;
const MPS_CHUNK: u32 = 2048;
const REF_ROWS: u32 = 256;
const MPS_MINROWS: u32 = 32;

fn source_for(name: &str) -> &'static str {
    match name {
        "flame_linear" | "qwen_widen" | "qwen_bf16_f16" | "qwen_f32_to_f16" | "qwen_f16_to_f32"
        | "qwen_gemv" | "qwen_gemv_rows" | "qwen_mlp_rows" | "qwen_simd_gemm" => LINEAR_SOURCE,
        "qwen_qkv" => QWEN_SOURCE,
        "flame_matmul" | "flame_softmax" | "flame_xent" | "flame_mean" => ATTENTION_SOURCE,
        _ => QWEN_SOURCE,
    }
}

#[derive(Clone, Copy)]
struct QwenConfig {
    layers: u32,
    hidden: u32,
    intermediate: u32,
    heads: u32,
    kv_heads: u32,
    vocab: u32,
    eos_token_id: u32,
    context: u32,
    epsilon: f32,
    rope_theta: f32,
}

impl Default for QwenConfig {
    fn default() -> Self {
        Self {
            layers: 28,
            hidden: 1536,
            intermediate: 8960,
            heads: 12,
            kv_heads: 2,
            vocab: 151_936,
            eos_token_id: 151_643,
            context: 32_768,
            epsilon: 1e-6,
            rope_theta: 1_000_000.0,
        }
    }
}

impl QwenConfig {
    fn validate(self, max_tokens: u32) -> Result<()> {
        if self.layers == 0
            || self.hidden == 0
            || self.intermediate == 0
            || self.heads == 0
            || self.kv_heads == 0
            || self.vocab == 0
            || self.context == 0
            || self.eos_token_id >= self.vocab
            || self.hidden % self.heads != 0
            || self.heads % self.kv_heads != 0
            || (self.hidden / self.heads) % 2 != 0
            || max_tokens == 0
            || max_tokens > self.context
            || !self.epsilon.is_finite()
            || self.epsilon <= 0.0
            || self.epsilon >= 1.0
            || !self.rope_theta.is_finite()
            || self.rope_theta <= 1.0
        {
            return Err("Invalid Qwen dimensions, heads, or normalization".into());
        }
        Ok(())
    }

    fn head_dim(self) -> u32 {
        self.hidden / self.heads
    }

    fn kv_width(self) -> u32 {
        self.kv_heads * self.head_dim()
    }
}

#[derive(Default, Clone, Copy)]
struct Layer {
    input_norm: usize,
    post_norm: usize,
    q_weight: usize,
    q_bias: usize,
    k_weight: usize,
    k_bias: usize,
    v_weight: usize,
    v_bias: usize,
    o_weight: usize,
    gate_weight: usize,
    up_weight: usize,
    down_weight: usize,
}

struct Layout {
    layers: Vec<Layer>,
    embedding: usize,
    final_norm: usize,
    tensors: BTreeMap<String, (usize, usize)>,
    len: usize,
}

impl Layout {
    fn new(c: QwenConfig) -> Result<Self> {
        let mut tensors = BTreeMap::<String, (usize, usize)>::new();
        let mut add = |name: String, dims: &[usize]| -> Result<()> {
            let length = dims.iter().try_fold(1usize, |value, &dim| {
                value
                    .checked_mul(dim)
                    .ok_or_else(|| "Qwen layout shape overflow".to_string())
            })?;
            tensors.insert(name, (0, length));
            Ok(())
        };
        let h = c.hidden as usize;
        let kv = c.kv_width() as usize;
        let intermediate = c.intermediate as usize;
        add("model.embed_tokens.weight".into(), &[c.vocab as usize, h])?;
        add("model.norm.weight".into(), &[h])?;
        for layer in 0..c.layers {
            let p = format!("model.layers.{layer}.");
            add(p.clone() + "input_layernorm.weight", &[h])?;
            add(p.clone() + "post_attention_layernorm.weight", &[h])?;
            add(p.clone() + "self_attn.q_proj.weight", &[h, h])?;
            add(p.clone() + "self_attn.q_proj.bias", &[h])?;
            add(p.clone() + "self_attn.k_proj.weight", &[kv, h])?;
            add(p.clone() + "self_attn.k_proj.bias", &[kv])?;
            add(p.clone() + "self_attn.v_proj.weight", &[kv, h])?;
            add(p.clone() + "self_attn.v_proj.bias", &[kv])?;
            add(p.clone() + "self_attn.o_proj.weight", &[h, h])?;
            add(p.clone() + "mlp.gate_proj.weight", &[intermediate, h])?;
            add(p.clone() + "mlp.up_proj.weight", &[intermediate, h])?;
            add(p + "mlp.down_proj.weight", &[h, intermediate])?;
        }
        let mut len = 0usize;
        for (offset, length) in tensors.values_mut() {
            *offset = len;
            len = len
                .checked_add(*length)
                .ok_or("Qwen layout offset overflow")?;
        }
        len.checked_mul(2).ok_or("Qwen weight byte-size overflow")?;
        let mut layers = Vec::with_capacity(c.layers as usize);
        for layer in 0..c.layers {
            let p = format!("model.layers.{layer}.");
            let get = |name: &str| tensors[&(p.clone() + name)].0;
            layers.push(Layer {
                input_norm: get("input_layernorm.weight"),
                post_norm: get("post_attention_layernorm.weight"),
                q_weight: get("self_attn.q_proj.weight"),
                q_bias: get("self_attn.q_proj.bias"),
                k_weight: get("self_attn.k_proj.weight"),
                k_bias: get("self_attn.k_proj.bias"),
                v_weight: get("self_attn.v_proj.weight"),
                v_bias: get("self_attn.v_proj.bias"),
                o_weight: get("self_attn.o_proj.weight"),
                gate_weight: get("mlp.gate_proj.weight"),
                up_weight: get("mlp.up_proj.weight"),
                down_weight: get("mlp.down_proj.weight"),
            });
        }
        Ok(Self {
            layers,
            embedding: tensors["model.embed_tokens.weight"].0,
            final_norm: tensors["model.norm.weight"].0,
            tensors,
            len,
        })
    }
}

#[derive(Clone, Copy)]
struct QwenBlock {
    key: u64,
    offset: usize,
    length: usize,
    scale: f32,
    weight: f32,
}

#[derive(Deserialize)]
struct SafeTensorRecord {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

#[repr(usize)]
#[derive(Clone, Copy)]
enum W {
    X,
    Norm,
    QRaw,
    KRaw,
    VRaw,
    Q,
    K,
    V,
    KRepeat,
    VRepeat,
    Scores,
    Attended,
    Update,
    Gate,
    Up,
    Activation,
    Logits,
    Losses,
    Tokens,
    Masks,
    Invalid,
    Count,
}

fn product(values: &[usize]) -> Result<usize> {
    values.iter().try_fold(1usize, |value, &item| {
        value
            .checked_mul(item)
            .filter(|&size| size <= isize::MAX as usize)
            .ok_or_else(|| "Qwen allocation or shape overflow".to_string())
    })
}

fn tps(tokens: u32, milliseconds: f32) -> f32 {
    if tokens == 0 || milliseconds <= 0.0 {
        0.0
    } else {
        tokens as f32 / (milliseconds / 1000.0)
    }
}

fn rope_table(config: QwenConfig, max_tokens: u32) -> Result<Vec<f32>> {
    let width = config.head_dim() as usize;
    let half = width / 2;
    let elements = product(&[max_tokens as usize, width])?;
    let mut table = vec![0.0f32; elements];
    for position in 0..max_tokens as usize {
        for pair in 0..half {
            let inverse = config.rope_theta.powf(-2.0 * pair as f32 / width as f32);
            let angle = position as f32 * inverse;
            let offset = position * width + pair * 2;
            table[offset] = angle.cos();
            table[offset + 1] = angle.sin();
        }
    }
    Ok(table)
}

fn tensor_bytes<'a>(
    mapping: &'a Mmap,
    payload_offset: usize,
    offsets: [usize; 2],
    name: &str,
) -> Result<&'a [u8]> {
    let [start, end] = offsets;
    if end < start {
        return Err(format!("Invalid byte range for Qwen tensor {name}"));
    }
    let start = payload_offset
        .checked_add(start)
        .ok_or_else(|| format!("Qwen tensor {name} offset overflow"))?;
    let end = payload_offset
        .checked_add(end)
        .ok_or_else(|| format!("Qwen tensor {name} offset overflow"))?;
    if end > mapping.len() || start > end {
        return Err(format!("Qwen tensor {name} exceeds the safetensors file"));
    }
    let bytes = &mapping[start..end];
    if bytes.len() % 2 != 0 {
        return Err(format!("Qwen tensor {name} has an odd BF16 byte length"));
    }
    Ok(bytes)
}

fn bf16_stats(bytes: &[u8], name: &str) -> Result<(f32, bool)> {
    let mut peak = 0.0f64;
    for pair in bytes.chunks_exact(2) {
        let bits = u16::from_le_bytes([pair[0], pair[1]]);
        if bits & 0x7f80 == 0x7f80 {
            return Err(format!(
                "Qwen tensor {name} contains a non-finite BF16 value"
            ));
        }
        peak = peak.max(f64::from(f32::from_bits(u32::from(bits) << 16).abs()));
    }
    if peak == 0.0 {
        return Ok((0.0, false));
    }
    let mut normalized_sum = 0.0f64;
    for pair in bytes.chunks_exact(2) {
        let bits = u16::from_le_bytes([pair[0], pair[1]]);
        let value = f64::from(f32::from_bits(u32::from(bits) << 16)) / peak;
        normalized_sum += value * value;
    }
    let rms = peak * (normalized_sum / (bytes.len() / 2) as f64).sqrt();
    if !rms.is_finite() || rms <= 0.0 || rms > f32::MAX as f64 {
        return Err(format!("Qwen tensor {name} has an invalid FP32 RMS"));
    }
    Ok((rms as f32, true))
}

fn tensor_key(name: &str) -> u64 {
    name.bytes().fold(1469598103934665603u64, |key, byte| {
        (key ^ u64::from(byte)).wrapping_mul(1099511628211)
    })
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum MpsMode {
    Off,
    F32,
    F16,
}

fn mps_mode() -> Result<MpsMode> {
    match std::env::var("ENNX_QWEN_MPS") {
        Err(std::env::VarError::NotPresent) => Ok(MpsMode::Off),
        Ok(value) if value == "0" => Ok(MpsMode::Off),
        Ok(value) if value == "1" || value == "fp32" => Ok(MpsMode::F32),
        Ok(value) if value == "fp16" => Ok(MpsMode::F16),
        Ok(_) | Err(std::env::VarError::NotUnicode(_)) => {
            Err("ENNX_QWEN_MPS must be 0, fp32, or fp16".into())
        }
    }
}

fn cache_path(rows: usize) -> bool {
    rows > REF_ROWS as usize
}

fn workspace_sizes(c: QwenConfig, capacity: u32, chunk: u32) -> Result<Vec<usize>> {
    let n = capacity.min(chunk) as usize;
    let h = c.hidden as usize;
    let kv = c.kv_width() as usize;
    let intermediate = c.intermediate as usize;
    let mut sizes = vec![0usize; W::Count as usize];
    let mut add = |slot: W, elements: usize, bytes: usize| -> Result<()> {
        let elements = product(&[elements, bytes])?;
        if elements > (u32::MAX - 255) as usize {
            return Err("Qwen workspace exceeds Metal dispatch indexing range".into());
        }
        sizes[slot as usize] = elements;
        Ok(())
    };
    for slot in [W::X, W::Norm, W::QRaw, W::Q, W::Attended, W::Update] {
        add(slot, product(&[n, h])?, 4)?;
    }
    add(W::Activation, product(&[n, intermediate])?, 4)?;
    for slot in [W::KRaw, W::VRaw, W::K, W::V] {
        add(slot, product(&[n, kv])?, 4)?;
    }
    for slot in [W::KRepeat, W::VRepeat] {
        add(slot, product(&[n, h])?, 4)?;
    }
    add(W::Scores, product(&[c.heads as usize, n, n])?, 4)?;
    for slot in [W::Gate, W::Up] {
        add(slot, product(&[n, intermediate])?, 4)?;
    }
    add(
        W::Logits,
        product(&[n.min(LOGIT_ROWS), c.vocab as usize])?,
        4,
    )?;
    add(W::Losses, capacity as usize, 4)?;
    add(W::Tokens, capacity as usize, 4)?;
    add(W::Masks, capacity as usize, 1)?;
    add(W::Invalid, 1, 4)?;
    Ok(sizes)
}

#[repr(C)]
#[derive(Default)]
struct Matmul {
    m: u32,
    n: u32,
    k: u32,
    transpose_b: u32,
    stride_a: u64,
    stride_b: u64,
    stride_c: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct FlameShape {
    rows: u32,
    width: u32,
    heads: u32,
    hidden: u32,
    experts: u32,
    top_k: u32,
    start: u32,
    sequence: u32,
    epsilon: f32,
    rope_base: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct QwenShape {
    rows: u32,
    width: u32,
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    hidden: u32,
    start: u32,
    sequence: u32,
    epsilon: f32,
    rope_theta: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct QwenQkvShape {
    rows: u32,
    hidden: u32,
    kv_width: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct QwenMlpShape {
    rows: u32,
    hidden: u32,
    intermediate: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct CacheShape {
    rows: u32,
    kv_heads: u32,
    head_dim: u32,
    capacity: u32,
    position: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct DecodeRopeShape {
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    capacity: u32,
    position: u32,
    batch: u32,
    cache_stride: u32,
    rope_theta: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct DecodeAttentionShape {
    sequence: u32,
    capacity: u32,
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    batch: u32,
    cache_stride: u32,
    scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct PrefillAttentionShape {
    rows: u32,
    sequence: u32,
    capacity: u32,
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct ArgmaxShape {
    width: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct ArgmaxBatchShape {
    rows: u32,
    width: u32,
}

enum BatchDecodeOutput {
    Logits(Vec<Vec<f32>>),
    Tokens(Vec<u32>),
}

struct GenerationState {
    key_cache: Buffer,
    value_cache: Buffer,
    position_buffer: Buffer,
    capacity: u32,
    position: u32,
    batch: u32,
    batch_index: u32,
    cache_stride: usize,
}

struct MpsPrefill {
    weights: Buffer,
    input: Option<Buffer>,
    output: Option<Buffer>,
    matmul: RefCell<MpsMatmul>,
    mode: MpsMode,
}

impl GenerationState {
    fn bytes(&self) -> u64 {
        self.key_cache.length() + self.value_cache.length() + self.position_buffer.length()
    }

    fn layer_offset(&self, _config: QwenConfig, layer: usize) -> u64 {
        let elements =
            (layer * self.batch as usize + self.batch_index as usize) * self.cache_stride;
        (elements * size_of::<u16>()) as u64
    }

    fn layer_batch(&self, layer: usize) -> u64 {
        let elements = layer * self.batch as usize * self.cache_stride;
        (elements * size_of::<u16>()) as u64
    }
}

pub struct QwenEvaluator {
    runtime: Arc<Runtime>,
    config: QwenConfig,
    max_tokens: u32,
    layout: Layout,
    workspace: Vec<Buffer>,
    rope_table: Buffer,
    workspace_bytes: u64,
    pipelines: BTreeMap<&'static str, ComputePipelineState>,
    blocks: Vec<QwenBlock>,
    loss_cache: Option<GenerationState>,
    loss_profile: Option<QwenLossProfile>,
    mps: Option<MpsPrefill>,
    prefill_chunk: u32,
    tile_attn: bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct QwenLossProfile {
    pub rows: u32,
    pub tokens: u32,
    pub scored_tokens: u32,
    pub cached_tokens: u32,
    pub kv_cache_bytes: u64,
    pub write_ms: f32,
    pub forward_ms: f32,
    pub output_ms: f32,
    pub total_ms: f32,
    pub tile_attn: bool,
    pub stages: Option<QwenStageProfile>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct QwenStageProfile {
    pub embed_ms: f32,
    pub qkv_ms: f32,
    pub attn_ms: f32,
    pub attn_out_ms: f32,
    pub mlp_expand_ms: f32,
    pub mlp_reduce_ms: f32,
    pub final_norm_ms: f32,
    pub gpu_ms: f32,
    pub commands: u32,
    pub missing: u32,
    pub chunks: u32,
}

#[derive(Clone, Copy)]
enum Stage {
    Embed,
    Qkv,
    Attn,
    AttnOut,
    MlpExpand,
    MlpReduce,
    FinalNorm,
}

impl QwenStageProfile {
    fn add(&mut self, stage: Stage, command: &CommandBufferRef) {
        self.commands += 1;
        let Some(ms) = gpu_seconds(command).map(|seconds| seconds as f32 * 1000.0) else {
            self.missing += 1;
            return;
        };
        self.gpu_ms += ms;
        match stage {
            Stage::Embed => self.embed_ms += ms,
            Stage::Qkv => self.qkv_ms += ms,
            Stage::Attn => self.attn_ms += ms,
            Stage::AttnOut => self.attn_out_ms += ms,
            Stage::MlpExpand => self.mlp_expand_ms += ms,
            Stage::MlpReduce => self.mlp_reduce_ms += ms,
            Stage::FinalNorm => self.final_norm_ms += ms,
        }
    }

    fn merge(&mut self, other: Self) {
        self.embed_ms += other.embed_ms;
        self.qkv_ms += other.qkv_ms;
        self.attn_ms += other.attn_ms;
        self.attn_out_ms += other.attn_out_ms;
        self.mlp_expand_ms += other.mlp_expand_ms;
        self.mlp_reduce_ms += other.mlp_reduce_ms;
        self.final_norm_ms += other.final_norm_ms;
        self.gpu_ms += other.gpu_ms;
        self.commands += other.commands;
        self.missing += other.missing;
        self.chunks += other.chunks;
    }
}

fn stage_split<'a>(
    runtime: &'a Runtime,
    command: &CommandBufferRef,
    profile: &mut QwenStageProfile,
    stage: Stage,
    label: &str,
) -> Result<&'a CommandBufferRef> {
    finish(command)?;
    profile.add(stage, command);
    let next = runtime.queue.new_command_buffer();
    next.set_label(label);
    Ok(next)
}

#[derive(Clone, Debug)]
pub struct QwenGenerationProfile {
    pub prompt_tokens: u32,
    pub requested_generated_tokens: u32,
    pub generated_tokens: u32,
    pub prefill_ms: f32,
    pub first_token_ms: f32,
    pub decode_ms_after_first: f32,
    pub total_ms: f32,
    pub tile_attn: bool,
    pub steady_decode_tokens_per_second: f32,
    pub end_to_end_generated_tokens_per_second: f32,
    pub host_overhead_ms: f32,
    pub command_buffer_count: u32,
    pub logits_read_to_cpu: bool,
    pub token_selection_on_gpu: bool,
    pub kv_cache_bytes: u64,
    pub device_name: String,
    pub max_tokens: u32,
    pub decode_kernel_trace: Vec<&'static str>,
    pub decode_kernel_times_ms: Vec<(&'static str, f32)>,
    pub decode_bottleneck_candidates: Vec<&'static str>,
    pub lm_head_argmax_decision: &'static str,
}

impl QwenEvaluator {
    pub fn new(max_tokens: u32) -> Result<Self> {
        autoreleasepool(|| Self::new_inner(max_tokens))
    }

    fn new_inner(max_tokens: u32) -> Result<Self> {
        Self::with_config(QwenConfig::default(), max_tokens)
    }

    fn with_config(config: QwenConfig, max_tokens: u32) -> Result<Self> {
        Self::with_backend(config, max_tokens, mps_mode()?)
    }

    fn with_backend(config: QwenConfig, max_tokens: u32, mode: MpsMode) -> Result<Self> {
        config.validate(max_tokens)?;
        let tile_attn = std::env::var_os("ENNX_QWEN_TILE_ATTN").is_some();
        if tile_attn && config.head_dim() != 128 {
            return Err("Qwen tiled attention requires head dimension 128".into());
        }
        let layout = Layout::new(config)?;
        let prefill_chunk = if mode == MpsMode::Off {
            PREFILL_CHUNK
        } else {
            MPS_CHUNK
        };
        let sizes = workspace_sizes(config, max_tokens, prefill_chunk)?;
        let mut workspace_bytes = sizes.iter().try_fold(0u64, |sum, &size| {
            sum.checked_add(size as u64)
                .ok_or_else(|| "Qwen workspace size overflow".to_string())
        })?;
        let runtime = Runtime::shared()?;
        if mode != MpsMode::Off && !metal::mps::mps_supports_device(&runtime.device) {
            return Err("MPS does not support the Qwen Metal device".into());
        }
        let mps_elements = if mode != MpsMode::Off {
            product(&[config.hidden as usize, config.intermediate as usize])?
        } else {
            0
        };
        let weight_elements = if mode == MpsMode::F16 {
            mps_elements
                .checked_mul(2)
                .ok_or("Qwen MPS weight workspace overflow")?
        } else {
            mps_elements
        };
        let half_elements = product(&[
            prefill_chunk.min(max_tokens) as usize,
            config.intermediate as usize,
        ])?;
        let half_storage = weight_elements
            .checked_add(
                half_elements
                    .checked_mul(2)
                    .ok_or("Qwen MPS half workspace overflow")?,
            )
            .ok_or("Qwen MPS half workspace overflow")?;
        let mps_bytes = match mode {
            MpsMode::Off => 0,
            MpsMode::F32 => product(&[mps_elements, size_of::<f32>()])?,
            MpsMode::F16 => product(&[half_storage, size_of::<u16>()])?,
        } as u64;
        workspace_bytes = workspace_bytes
            .checked_add(mps_bytes)
            .ok_or("Qwen MPS workspace size overflow")?;
        trace("Qwen runtime acquired");
        let rope_table_values = rope_table(config, max_tokens)?;
        let rope_table = runtime.buffer_with(&rope_table_values);
        let rope_table_bytes = std::mem::size_of_val(rope_table_values.as_slice()) as u64;
        let weight_bytes = product(&[layout.len, 2])? as u64;
        let limit = runtime.device.recommended_max_working_set_size();
        let allocated = runtime.device.current_allocated_size();
        let budget = limit.saturating_sub(limit / 10);
        if allocated
            .checked_add(workspace_bytes)
            .and_then(|value| value.checked_add(rope_table_bytes))
            .and_then(|value| value.checked_add(weight_bytes))
            .is_none_or(|value| value > budget)
        {
            return Err("Qwen Metal evaluation exceeds the conservative working-set budget".into());
        }
        if weight_bytes > runtime.device.max_buffer_length() {
            return Err("Qwen BF16 weights exceed Metal maxBufferLength".into());
        }
        if rope_table_bytes > runtime.device.max_buffer_length() || rope_table.contents().is_null()
        {
            return Err("Qwen RoPE table exceeds Metal buffer limits".into());
        }
        let mut pipelines = BTreeMap::new();
        for name in [
            "flame_linear",
            "qwen_widen",
            "qwen_bf16_f16",
            "qwen_f32_to_f16",
            "qwen_f16_to_f32",
            "qwen_gemv",
            "qwen_gemv_rows",
            "qwen_mlp_rows",
            "qwen_simd_gemm",
            "qwen_qkv",
            "flame_matmul",
            "flame_softmax",
            "flame_xent",
            "flame_mean",
            "qwen_embedding",
            "qwen_rms",
            "qwen_bias",
            "qwen_rope",
            "qwen_cache",
            "qwen_arow",
            "qwen_atile",
            "qwen_attn16",
            "qwen_drope",
            "qwen_dattn",
            "qwen_argmax",
            "qwen_argn",
            "qwen_repkv",
            "qwen_aout",
            "qwen_silu",
            "qwen_silu16",
            "qwen_residual",
        ] {
            trace(&format!("Qwen pipeline {name}"));
            let pipeline = runtime.precise(source_for(name), "Qwen", name)?;
            if pipeline.thread_execution_width() != 32
                || pipeline.max_total_threads_per_threadgroup() < 256
            {
                return Err(
                    "Qwen requires Apple GPU 32-lane SIMD groups and 256-thread groups".into(),
                );
            }
            pipelines.insert(name, pipeline);
        }
        if pipelines["qwen_attn16"].static_threadgroup_memory_length()
            > runtime.device.max_threadgroup_memory_length()
        {
            return Err("Qwen tiled attention exceeds threadgroup memory".into());
        }
        let mut workspace = Vec::with_capacity(sizes.len());
        trace("allocating Qwen workspace");
        for size in sizes {
            let buffer = runtime.buffer::<u8>(size);
            if buffer.contents().is_null() {
                return Err(format!(
                    "Metal could not allocate {size} Qwen workspace bytes"
                ));
            }
            workspace.push(buffer);
        }
        let mps = if mode != MpsMode::Off {
            let weights = match mode {
                MpsMode::F32 => runtime.buffer::<f32>(weight_elements),
                MpsMode::F16 => runtime.buffer::<u16>(weight_elements),
                MpsMode::Off => unreachable!(),
            };
            if weights.contents().is_null() {
                return Err("Metal could not allocate Qwen MPS weight scratch".into());
            }
            let input = (mode == MpsMode::F16).then(|| runtime.buffer::<u16>(half_elements));
            let output = (mode == MpsMode::F16).then(|| runtime.buffer::<u16>(half_elements));
            if input
                .as_ref()
                .into_iter()
                .chain(output.as_ref())
                .any(|buffer| buffer.contents().is_null())
            {
                return Err("Metal could not allocate Qwen MPS half scratch".into());
            }
            Some(MpsPrefill {
                weights,
                input,
                output,
                matmul: RefCell::new(MpsMatmul::default()),
                mode,
            })
        } else {
            None
        };
        trace("Qwen evaluator ready");
        Ok(Self {
            runtime,
            config,
            max_tokens,
            layout,
            workspace,
            rope_table,
            workspace_bytes,
            pipelines,
            blocks: Vec::new(),
            loss_cache: None,
            loss_profile: None,
            mps,
            prefill_chunk,
            tile_attn,
        })
    }

    pub fn weights_len(&self) -> usize {
        self.layout.len
    }

    pub fn workspace_bytes(&self) -> u64 {
        self.workspace_bytes
    }

    pub fn device_name(&self) -> &str {
        &self.runtime.info().name
    }

    pub fn vocab(&self) -> usize {
        self.config.vocab as usize
    }

    pub fn blocks(&self) -> Vec<(u64, usize, usize, f32, f32)> {
        self.blocks
            .iter()
            .map(|block| {
                (
                    block.key,
                    block.offset,
                    block.length,
                    block.scale,
                    block.weight,
                )
            })
            .collect()
    }

    pub fn loss_profile(&self) -> Option<QwenLossProfile> {
        self.loss_profile
    }

    pub fn load_weights(&mut self, path: &Path) -> Result<Buffer> {
        autoreleasepool(|| self.load_file(path))
    }

    fn load_file(&mut self, path: &Path) -> Result<Buffer> {
        let file = File::open(path).map_err(|error| format!("Qwen weights open: {error}"))?;
        let mapping =
            unsafe { Mmap::map(&file) }.map_err(|error| format!("Qwen weights mmap: {error}"))?;
        if mapping.len() < 8 {
            return Err("Qwen weights are missing a safetensors header".into());
        }
        let header_size = usize::try_from(u64::from_le_bytes(
            mapping[..8]
                .try_into()
                .map_err(|_| "Invalid Qwen safetensors header length")?,
        ))
        .map_err(|_| "Qwen safetensors header is too large")?;
        let payload_offset = 8usize
            .checked_add(header_size)
            .ok_or("Qwen safetensors header offset overflow")?;
        if payload_offset > mapping.len() {
            return Err("Qwen safetensors header exceeds the file".into());
        }
        let header: BTreeMap<String, serde_json::Value> =
            serde_json::from_slice(&mapping[8..payload_offset])
                .map_err(|error| format!("Qwen safetensors header: {error}"))?;
        let mut records = BTreeMap::<String, SafeTensorRecord>::new();
        for (name, value) in header {
            if name == "__metadata__" {
                continue;
            }
            let record: SafeTensorRecord = serde_json::from_value(value)
                .map_err(|error| format!("Qwen tensor {name} metadata: {error}"))?;
            records.insert(name, record);
        }
        let expected = &self.layout.tensors;
        for name in records.keys() {
            if !expected.contains_key(name) && name != "lm_head.weight" {
                return Err(format!("Unexpected Qwen tensor: {name}"));
            }
        }
        for name in expected.keys() {
            if !records.contains_key(name) {
                return Err(format!("Missing Qwen tensor: {name}"));
            }
        }
        let embedding_range = records
            .get("model.embed_tokens.weight")
            .ok_or("Missing Qwen embedding tensor")?
            .data_offsets;
        let embedding_bytes = tensor_bytes(
            &mapping,
            payload_offset,
            embedding_range,
            "model.embed_tokens.weight",
        )?;
        if let Some(record) = records.get("lm_head.weight") {
            let lm_head = tensor_bytes(
                &mapping,
                payload_offset,
                record.data_offsets,
                "lm_head.weight",
            )?;
            if record.dtype != "BF16"
                || product(&record.shape)? != expected["model.embed_tokens.weight"].1
                || lm_head != embedding_bytes
            {
                return Err("Tied Qwen embedding and lm_head weights differ".into());
            }
        }

        let bytes = product(&[self.layout.len, 2])? as u64;
        let limit = self.runtime.device.recommended_max_working_set_size();
        let allocated = self.runtime.device.current_allocated_size();
        let budget = limit.saturating_sub(limit / 10);
        if allocated
            .checked_add(bytes)
            .is_none_or(|value| value > budget)
        {
            return Err("Qwen Metal evaluation exceeds the conservative working-set budget".into());
        }
        let buffer = self.runtime.buffer::<u16>(self.layout.len);
        if buffer.contents().is_null() {
            return Err("Metal could not allocate Qwen weights".into());
        }
        let tensor_count = expected.len();
        let mut blocks = Vec::with_capacity(tensor_count);
        for (name, &(offset, length)) in expected {
            let record = records
                .get(name)
                .ok_or_else(|| format!("Missing Qwen tensor: {name}"))?;
            if record.dtype != "BF16" || product(&record.shape)? != length {
                return Err(format!("Unexpected dtype or shape for Qwen tensor {name}"));
            }
            let bytes = tensor_bytes(&mapping, payload_offset, record.data_offsets, name)?;
            let (scale, nonzero) = bf16_stats(bytes, name)?;
            if !nonzero {
                return Err(format!("Qwen tensor {name} is all zero"));
            }
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bytes.as_ptr(),
                    buffer.contents().cast::<u8>().add(offset * 2),
                    length * 2,
                );
            }
            blocks.push(QwenBlock {
                key: tensor_key(name),
                offset,
                length,
                scale,
                weight: 1.0 / (tensor_count as f32 * length as f32 * scale * scale),
            });
        }
        self.blocks = blocks;
        Ok(buffer)
    }

    pub fn check_weights(&self, weights: &Buffer) -> Result<()> {
        if weights.length() != self.weights_len() as u64 * 2 {
            return Err(format!(
                "Qwen requires {} contiguous Metal BF16 weights",
                self.weights_len()
            ));
        }
        if weights.device().registry_id() != self.runtime.device.registry_id() {
            return Err("Qwen weights must belong to the evaluator's Metal device".into());
        }
        Ok(())
    }

    fn check_tokens(&self, tokens: &[i32]) -> Result<()> {
        if tokens.is_empty()
            || tokens.len() > self.max_tokens as usize
            || tokens
                .iter()
                .any(|&token| token < 0 || token as u32 >= self.config.vocab)
        {
            return Err(
                "Qwen tokens must be nonempty, in vocabulary, and within max_tokens".into(),
            );
        }
        Ok(())
    }

    fn check_batch(&self, tokens: &[Vec<i32>], masks: &[Vec<bool>]) -> Result<()> {
        if tokens.is_empty() || tokens.len() != masks.len() {
            return Err("Qwen requires equally sized nonempty token and mask batches".into());
        }
        for (row, mask) in tokens.iter().zip(masks) {
            self.check_tokens(row)?;
            if row.len() < 2
                || row.len() != mask.len()
                || mask[0]
                || !mask[1..].iter().any(|&value| value)
            {
                return Err("Each Qwen loss mask must match tokens, leave token zero unscored, and score a target".into());
            }
        }
        Ok(())
    }

    pub fn logits(&mut self, weights: &Buffer, tokens: &[i32]) -> Result<Vec<f32>> {
        autoreleasepool(|| self.logits_inner(weights, tokens, false))
    }

    pub fn next_logits(&mut self, weights: &Buffer, tokens: &[i32]) -> Result<Vec<f32>> {
        autoreleasepool(|| self.logits_inner(weights, tokens, true))
    }

    pub fn generate(
        &mut self,
        weights: &Buffer,
        prompt: &[i32],
        max_new_tokens: usize,
    ) -> Result<Vec<i32>> {
        autoreleasepool(|| self.generate_inner(weights, prompt, max_new_tokens))
    }

    pub fn bench_generate(
        &mut self,
        weights: &Buffer,
        prompt: &[i32],
        max_new_tokens: usize,
        mode: &str,
    ) -> Result<QwenGenerationProfile> {
        autoreleasepool(|| self.bench_profile(weights, prompt, max_new_tokens, mode))
    }

    pub fn generate_batch(
        &mut self,
        weights: &Buffer,
        prompts: &[Vec<i32>],
        max_new_tokens: usize,
    ) -> Result<Vec<Vec<i32>>> {
        autoreleasepool(|| self.batch_inner(weights, prompts, max_new_tokens, None))
    }

    pub fn sample(
        &mut self,
        weights: &Buffer,
        prompt: &[i32],
        max_new_tokens: usize,
        temperature: f32,
        top_p: f32,
        top_k: usize,
        seed: u64,
    ) -> Result<Vec<i32>> {
        autoreleasepool(|| {
            self.sample_inner(
                weights,
                prompt,
                max_new_tokens,
                temperature,
                top_p,
                top_k,
                seed,
            )
        })
    }

    pub fn sample_batch(
        &mut self,
        weights: &Buffer,
        prompts: &[Vec<i32>],
        max_new_tokens: usize,
        temperature: f32,
        top_p: f32,
        top_k: usize,
        seeds: &[u64],
    ) -> Result<Vec<Vec<i32>>> {
        autoreleasepool(|| {
            self.batch_inner(
                weights,
                prompts,
                max_new_tokens,
                Some((temperature, top_p, top_k, seeds)),
            )
        })
    }

    fn batch_inner(
        &mut self,
        weights: &Buffer,
        prompts: &[Vec<i32>],
        max_new_tokens: usize,
        sampling: Option<(f32, f32, usize, &[u64])>,
    ) -> Result<Vec<Vec<i32>>> {
        self.check_weights(weights)?;
        if prompts.is_empty() || prompts.len() > self.max_tokens as usize {
            return Err("Qwen batch size must be nonempty and fit the evaluator workspace".into());
        }
        if max_new_tokens == 0 {
            return Err("Qwen max_new_tokens must be positive".into());
        }
        if let Some((temperature, top_p, top_k, seeds)) = sampling {
            if seeds.len() != prompts.len() {
                return Err("Qwen sampled batch requires one seed per prompt".into());
            }
            if !temperature.is_finite() || temperature <= 0.0 {
                return Err("Qwen sampling temperature must be finite and positive".into());
            }
            if !top_p.is_finite() || !(0.0 < top_p && top_p <= 1.0) {
                return Err("Qwen sampling top_p must be in (0, 1]".into());
            }
            if top_k > self.config.vocab as usize {
                return Err("Qwen sampling top_k exceeds the vocabulary".into());
            }
        }
        for prompt in prompts {
            self.check_tokens(prompt)?;
            if prompt
                .len()
                .checked_add(max_new_tokens)
                .is_none_or(|length| length > self.max_tokens as usize)
            {
                return Err("Qwen prompt plus batch generation exceeds max_tokens".into());
            }
        }

        let batch = prompts.len() as u32;
        let mut state = self.batch_state(batch)?;
        let mut positions = Vec::with_capacity(prompts.len());
        let mut logits = Vec::with_capacity(prompts.len());
        for (index, prompt) in prompts.iter().enumerate() {
            self.write(W::Tokens, prompt);
            self.write(W::Masks, &vec![0u8; prompt.len()]);
            state.batch_index = index as u32;
            state.position = 0;
            self.forward(weights, prompt.len() as u32, Some(&mut state))?;
            logits.push(self.output_logits(
                weights,
                u64::from((prompt.len() - 1) as u32 % self.prefill_chunk)
                    * u64::from(self.config.hidden)
                    * 4,
            )?);
            positions.push(prompt.len() as u32);
        }

        let mut tokens: Vec<Vec<i32>> = prompts.to_vec();
        let mut active = vec![true; prompts.len()];
        let mut rngs = sampling.map(|(_, _, _, seeds)| {
            seeds
                .iter()
                .map(|&seed| if seed == 0 { 0x9E3779B97F4A7C15 } else { seed })
                .collect::<Vec<_>>()
        });
        let mut gpu_next_tokens: Option<Vec<u32>> = None;
        for step in 0..max_new_tokens {
            let decoded_tokens = gpu_next_tokens.take();
            let mut next_tokens = Vec::with_capacity(prompts.len());
            for index in 0..prompts.len() {
                let next = if let Some((temperature, top_p, top_k, _)) = sampling {
                    sample_logits(
                        &logits[index],
                        temperature,
                        top_p,
                        top_k,
                        &mut rngs.as_mut().expect("sampling RNGs missing")[index],
                    )?
                } else if let Some(decoded_tokens) = decoded_tokens.as_ref() {
                    decoded_tokens[index]
                } else {
                    argmax_logits(&logits[index])?
                };
                let token = i32::try_from(next).map_err(|_| "Qwen token ID overflow")?;
                tokens[index].push(token);
                active[index] = active[index] && next != self.config.eos_token_id;
                next_tokens.push(token);
            }
            if step + 1 == max_new_tokens || !active.iter().any(|&value| value) {
                break;
            }
            self.write(W::Tokens, &next_tokens);
            match self.batch_logits(weights, &mut state, &positions, sampling.is_none())? {
                BatchDecodeOutput::Logits(next_logits) => logits = next_logits,
                BatchDecodeOutput::Tokens(next_tokens) => gpu_next_tokens = Some(next_tokens),
            }
            for position in &mut positions {
                *position = position
                    .checked_add(1)
                    .ok_or("Qwen generation position overflow")?;
            }
        }
        Ok(tokens)
    }

    fn sample_inner(
        &mut self,
        weights: &Buffer,
        prompt: &[i32],
        max_new_tokens: usize,
        temperature: f32,
        top_p: f32,
        top_k: usize,
        seed: u64,
    ) -> Result<Vec<i32>> {
        self.check_weights(weights)?;
        self.check_tokens(prompt)?;
        if max_new_tokens == 0
            || prompt
                .len()
                .checked_add(max_new_tokens)
                .is_none_or(|length| length > self.max_tokens as usize)
        {
            return Err("Qwen prompt plus sampled generation exceeds max_tokens".into());
        }
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err("Qwen sampling temperature must be finite and positive".into());
        }
        if !top_p.is_finite() || !(0.0 < top_p && top_p <= 1.0) {
            return Err("Qwen sampling top_p must be in (0, 1]".into());
        }
        if top_k > self.config.vocab as usize {
            return Err("Qwen sampling top_k exceeds the vocabulary".into());
        }
        let mut state = self.new_state()?;
        let mut tokens = prompt.to_vec();
        self.write(W::Tokens, prompt);
        self.write(W::Masks, &vec![0u8; prompt.len()]);
        self.forward(weights, prompt.len() as u32, Some(&mut state))?;
        let mut logits = self.output_logits(
            weights,
            u64::from((prompt.len() - 1) as u32 % self.prefill_chunk)
                * u64::from(self.config.hidden)
                * 4,
        )?;
        let mut rng = if seed == 0 { 0x9E3779B97F4A7C15 } else { seed };
        for index in 0..max_new_tokens {
            let next_token = sample_logits(&logits, temperature, top_p, top_k, &mut rng)?;
            let token = i32::try_from(next_token).map_err(|_| "Qwen token ID overflow")?;
            tokens.push(token);
            if next_token == self.config.eos_token_id {
                break;
            }
            if index + 1 < max_new_tokens {
                self.write(W::Tokens, &[token]);
                logits = self.decode_logits(weights, &mut state)?;
            }
        }
        Ok(tokens)
    }

    fn generate_inner(
        &mut self,
        weights: &Buffer,
        prompt: &[i32],
        max_new_tokens: usize,
    ) -> Result<Vec<i32>> {
        self.check_weights(weights)?;
        self.check_tokens(prompt)?;
        if max_new_tokens == 0 {
            return Err("Qwen max_new_tokens must be positive".into());
        }
        if prompt
            .len()
            .checked_add(max_new_tokens)
            .is_none_or(|length| length > self.max_tokens as usize)
        {
            return Err("Qwen prompt plus generation exceeds max_tokens".into());
        }
        let mut state = self.new_state()?;
        let mut tokens = prompt.to_vec();
        self.write(W::Tokens, prompt);
        self.write(W::Masks, &vec![0u8; prompt.len()]);
        self.forward(weights, prompt.len() as u32, Some(&mut state))?;
        let mut next_token = self.argmax_output(
            weights,
            u64::from((prompt.len() - 1) as u32 % self.prefill_chunk)
                * u64::from(self.config.hidden)
                * 4,
        )?;
        for index in 0..max_new_tokens {
            let token = i32::try_from(next_token).map_err(|_| "Qwen token ID overflow")?;
            tokens.push(token);
            if next_token == self.config.eos_token_id {
                break;
            }
            if index + 1 < max_new_tokens {
                self.write(W::Tokens, &[token]);
                next_token = self.decode_token(weights, &mut state)?;
            }
        }
        Ok(tokens)
    }

    fn bench_profile(
        &mut self,
        weights: &Buffer,
        prompt: &[i32],
        max_new_tokens: usize,
        mode: &str,
    ) -> Result<QwenGenerationProfile> {
        if mode != "greedy" {
            return Err("Qwen bench_generate currently supports only greedy mode".into());
        }
        self.check_weights(weights)?;
        self.check_tokens(prompt)?;
        if max_new_tokens == 0 {
            return Err("Qwen max_new_tokens must be positive".into());
        }
        if prompt
            .len()
            .checked_add(max_new_tokens)
            .is_none_or(|length| length > self.max_tokens as usize)
        {
            return Err("Qwen prompt plus generation exceeds max_tokens".into());
        }

        let total_start = Instant::now();
        let mut state = self.new_state()?;
        let kv_cache_bytes = state
            .key_cache
            .length()
            .saturating_add(state.value_cache.length());
        let mut command_buffer_count = prompt.len().div_ceil(self.prefill_chunk as usize) as u32;

        self.write(W::Tokens, prompt);
        self.write(W::Masks, &vec![0u8; prompt.len()]);
        let prefill_start = Instant::now();
        self.forward(weights, prompt.len() as u32, Some(&mut state))?;
        let prefill_ms = prefill_start.elapsed().as_secs_f32() * 1000.0;

        let first_start = Instant::now();
        let mut next_token = self.argmax_output(
            weights,
            u64::from((prompt.len() - 1) as u32 % self.prefill_chunk)
                * u64::from(self.config.hidden)
                * 4,
        )?;
        command_buffer_count = command_buffer_count.saturating_add(1);
        let first_token_ms = first_start.elapsed().as_secs_f32() * 1000.0;

        let mut generated_tokens = 0u32;
        let mut decode_kernel_times_ms = BTreeMap::<&'static str, f32>::new();
        let decode_start = Instant::now();
        for index in 0..max_new_tokens {
            generated_tokens = generated_tokens.saturating_add(1);
            if next_token == self.config.eos_token_id {
                break;
            }
            if index + 1 < max_new_tokens {
                let token = i32::try_from(next_token).map_err(|_| "Qwen token ID overflow")?;
                self.write(W::Tokens, &[token]);
                next_token =
                    self.decode_profile(weights, &mut state, &mut decode_kernel_times_ms)?;
                command_buffer_count = command_buffer_count.saturating_add(1);
            }
        }
        let decode_ms_after_first = decode_start.elapsed().as_secs_f32() * 1000.0;
        let total_ms = total_start.elapsed().as_secs_f32() * 1000.0;
        let metal_phase_ms = prefill_ms + first_token_ms + decode_ms_after_first;
        let steady_tokens = generated_tokens.saturating_sub(1);
        let steady_decode_tokens_per_second = tps(steady_tokens, decode_ms_after_first);
        let end_to_end_generated_tokens_per_second = tps(generated_tokens, total_ms);

        let mut measured_times: Vec<_> = decode_kernel_times_ms.into_iter().collect();
        measured_times.sort_by(|left, right| {
            right
                .1
                .partial_cmp(&left.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let decode_bottleneck_candidates = if measured_times.is_empty() {
            vec!["no steady decode token measured"]
        } else {
            measured_times
                .iter()
                .take(3)
                .map(|(name, _)| *name)
                .collect()
        };

        Ok(QwenGenerationProfile {
            prompt_tokens: prompt.len() as u32,
            requested_generated_tokens: max_new_tokens as u32,
            generated_tokens,
            prefill_ms,
            first_token_ms,
            decode_ms_after_first,
            total_ms,
            tile_attn: self.tile_attn,
            steady_decode_tokens_per_second,
            end_to_end_generated_tokens_per_second,
            host_overhead_ms: (total_ms - metal_phase_ms).max(0.0),
            command_buffer_count,
            logits_read_to_cpu: false,
            token_selection_on_gpu: true,
            kv_cache_bytes,
            device_name: self.device_name().to_string(),
            max_tokens: self.max_tokens,
            decode_kernel_trace: vec![
                "qwen_embedding",
                "qwen_rms",
                "qwen_qkv",
                "qwen_drope",
                "qwen_dattn",
                "qwen_gemv_rows",
                "qwen_residual",
                "qwen_rms",
                "qwen_mlp_rows",
                "qwen_gemv_rows",
                "qwen_residual",
                "qwen_rms",
                "qwen_gemv",
                "qwen_argmax",
            ],
            decode_kernel_times_ms: measured_times,
            decode_bottleneck_candidates,
            lm_head_argmax_decision: "keep separate qwen_gemv + qwen_argmax until a measured trace shows the final LM head is the largest decode bottleneck",
        })
    }

    fn new_state(&self) -> Result<GenerationState> {
        self.batch_state(1)
    }

    fn batch_state(&self, batch: u32) -> Result<GenerationState> {
        if batch == 0 || batch > self.max_tokens {
            return Err("Qwen generation batch exceeds the evaluator workspace".into());
        }
        let capacity = self
            .max_tokens
            .checked_add(255)
            .ok_or("Qwen generation cache capacity overflow")?
            / 256
            * 256;
        let per_cache = product(&[
            self.config.layers as usize,
            self.config.kv_heads as usize,
            capacity as usize,
            self.config.head_dim() as usize,
        ])?;
        let cache_elements = product(&[per_cache, batch as usize])?;
        let cache_bytes = (per_cache * size_of::<u16>()) as u64;
        let total_bytes = cache_bytes
            .checked_mul(u64::from(batch))
            .and_then(|value| value.checked_mul(2))
            .ok_or("Qwen generation cache size overflow")?;
        if total_bytes > self.runtime.device.max_buffer_length() {
            return Err("Qwen generation cache exceeds Metal maxBufferLength".into());
        }
        let limit = self.runtime.device.recommended_max_working_set_size();
        let allocated = self.runtime.device.current_allocated_size();
        let budget = limit.saturating_sub(limit / 10);
        if allocated
            .checked_add(total_bytes)
            .is_none_or(|value| value > budget)
        {
            return Err("Qwen generation cache exceeds the conservative working-set budget".into());
        }
        let key_cache = self.runtime.buffer::<u16>(cache_elements);
        let value_cache = self.runtime.buffer::<u16>(cache_elements);
        let position_buffer = self.runtime.buffer::<u32>(batch as usize);
        if key_cache.contents().is_null()
            || value_cache.contents().is_null()
            || position_buffer.contents().is_null()
        {
            return Err("Metal could not allocate the Qwen generation cache".into());
        }
        Ok(GenerationState {
            key_cache,
            value_cache,
            position_buffer,
            capacity,
            position: 0,
            batch,
            batch_index: 0,
            cache_stride: self.config.kv_heads as usize
                * capacity as usize
                * self.config.head_dim() as usize,
        })
    }

    fn logits_inner(
        &mut self,
        weights: &Buffer,
        tokens: &[i32],
        last_only: bool,
    ) -> Result<Vec<f32>> {
        self.check_weights(weights)?;
        self.check_tokens(tokens)?;
        let first = if last_only { tokens.len() - 1 } else { 0 };
        let length = product(&[tokens.len() - first, self.vocab()])?;
        let mut output = vec![0.0f32; length];
        self.write(W::Tokens, tokens);
        self.write(W::Masks, &vec![0u8; tokens.len()]);
        self.forward(weights, tokens.len() as u32, None)?;
        self.output(
            weights,
            tokens.len() as u32,
            first as u32,
            0,
            Some(&mut output),
        )?;
        Ok(output)
    }

    pub fn losses(
        &mut self,
        weights: &Buffer,
        tokens: &[Vec<i32>],
        masks: &[Vec<bool>],
    ) -> Result<Vec<f32>> {
        self.check_weights(weights)?;
        self.check_batch(tokens, masks)?;
        self.loss_profile = None;
        let total_start = Instant::now();
        let mut profile = QwenLossProfile {
            rows: tokens.len() as u32,
            tile_attn: self.tile_attn,
            scored_tokens: masks
                .iter()
                .flat_map(|row| row.iter())
                .filter(|&&value| value)
                .count() as u32,
            ..QwenLossProfile::default()
        };
        let spans = masks
            .iter()
            .map(|mask| {
                let first = mask
                    .iter()
                    .position(|&value| value)
                    .expect("validated Qwen mask has a target");
                let end = mask
                    .iter()
                    .rposition(|&value| value)
                    .expect("validated Qwen mask has a target")
                    + 1;
                let scored = mask[..end].iter().filter(|&&value| value).count() as u32;
                (first, end, scored)
            })
            .collect::<Vec<_>>();
        profile.tokens = spans.iter().map(|&(_, end, _)| end as u32).sum();
        let needs_cache = spans.iter().any(|&(_, end, _)| cache_path(end));
        let mut cache = if needs_cache {
            Some(
                self.loss_cache
                    .take()
                    .map_or_else(|| self.new_state(), Ok)?,
            )
        } else {
            None
        };
        if let Some(state) = cache.as_ref().or(self.loss_cache.as_ref()) {
            profile.kv_cache_bytes = state.bytes();
        }
        let result = (|| -> Result<Vec<f32>> {
            let mut output = Vec::with_capacity(tokens.len());
            for ((row, mask), &(first, end, scored)) in tokens.iter().zip(masks).zip(&spans) {
                let write_start = Instant::now();
                self.write(W::Tokens, &row[..end]);
                let values: Vec<u8> = mask[..end].iter().map(|&value| u8::from(value)).collect();
                self.write(W::Masks, &values);
                profile.write_ms += write_start.elapsed().as_secs_f32() * 1000.0;
                let loss = if cache_path(end) {
                    profile.cached_tokens += end as u32;
                    let (loss, forward_ms, output_ms, stages) = self.cached_loss(
                        weights,
                        end as u32,
                        first as u32,
                        scored,
                        cache.as_mut().expect("Qwen loss cache missing"),
                    )?;
                    profile.forward_ms += forward_ms;
                    profile.output_ms += output_ms;
                    if let Some(stages) = stages {
                        profile
                            .stages
                            .get_or_insert_with(QwenStageProfile::default)
                            .merge(stages);
                    }
                    loss
                } else {
                    let forward_start = Instant::now();
                    let stages = self.forward(weights, end as u32, None)?;
                    profile.forward_ms += forward_start.elapsed().as_secs_f32() * 1000.0;
                    if let Some(stages) = stages {
                        profile
                            .stages
                            .get_or_insert_with(QwenStageProfile::default)
                            .merge(stages);
                    }
                    let output_start = Instant::now();
                    let loss =
                        self.output(weights, end as u32, (first - 1) as u32, scored, None)?;
                    profile.output_ms += output_start.elapsed().as_secs_f32() * 1000.0;
                    loss
                };
                output.push(loss);
            }
            Ok(output)
        })();
        if let Some(state) = cache {
            self.loss_cache = Some(state);
        }
        let output = result?;
        profile.total_ms = total_start.elapsed().as_secs_f32() * 1000.0;
        self.loss_profile = Some(profile);
        Ok(output)
    }

    fn cached_loss(
        &self,
        weights: &Buffer,
        rows: u32,
        first_scored: u32,
        scored: u32,
        cache: &mut GenerationState,
    ) -> Result<(f32, f32, f32, Option<QwenStageProfile>)> {
        cache.position = 0;
        self.write(W::Invalid, &[0u32]);
        self.write(W::Losses, &vec![0.0f32; rows as usize]);
        let first_row = first_scored - 1;
        let mut forward_ms = 0.0;
        let mut output_ms = 0.0;
        let mut stages = None;
        let mut start = 0;
        while start < rows {
            let chunk = (rows - start).min(self.prefill_chunk);
            let forward_start = Instant::now();
            if let Some(chunk_stages) = self.forward_chunk(weights, chunk, start, Some(cache))? {
                stages
                    .get_or_insert_with(QwenStageProfile::default)
                    .merge(chunk_stages);
            }
            forward_ms += forward_start.elapsed().as_secs_f32() * 1000.0;
            let local_first = first_row.saturating_sub(start).min(chunk);
            if local_first < chunk {
                let output_start = Instant::now();
                self.project_rows(weights, chunk, local_first, start, rows, true, None)?;
                output_ms += output_start.elapsed().as_secs_f32() * 1000.0;
            }
            cache.position = cache
                .position
                .checked_add(chunk)
                .ok_or("Qwen loss cache position overflow")?;
            start += chunk;
        }
        let output_start = Instant::now();
        let loss = self.mean_loss(rows, scored)?;
        output_ms += output_start.elapsed().as_secs_f32() * 1000.0;
        Ok((loss, forward_ms, output_ms, stages))
    }

    fn buffer(&self, slot: W) -> &Buffer {
        &self.workspace[slot as usize]
    }

    fn write<T: Copy>(&self, slot: W, values: &[T]) {
        write_buffer(self.buffer(slot), values);
    }

    fn read<T: Copy>(&self, slot: W, count: usize) -> Vec<T> {
        assert!(size_of::<T>() * count <= self.buffer(slot).length() as usize);
        unsafe {
            std::slice::from_raw_parts(self.buffer(slot).contents().cast::<T>(), count).to_vec()
        }
    }

    fn trace_values(&self, label: &str, slot: W, count: usize) {
        self.trace_at(label, slot, 0, count);
    }

    fn trace_at(&self, label: &str, slot: W, offset: usize, count: usize) {
        if std::env::var_os("ENNX_QWEN_TRACE_VALUES").is_none() {
            return;
        }
        let bytes = offset
            .checked_mul(size_of::<f32>())
            .expect("Qwen trace offset overflow");
        let buffer = self.buffer(slot);
        assert!(bytes + count * size_of::<f32>() <= buffer.length() as usize);
        let values: Vec<f32> = unsafe {
            std::slice::from_raw_parts(buffer.contents().cast::<u8>().add(bytes).cast(), count)
                .to_vec()
        };
        let finite = values.iter().filter(|value| value.is_finite()).count();
        let nonzero = values.iter().filter(|value| **value != 0.0).count();
        let peak = values
            .iter()
            .filter(|value| value.is_finite())
            .map(|value| value.abs())
            .fold(0.0f32, f32::max);
        trace(&format!(
            "Qwen {label}: finite={finite}/{count} nonzero={nonzero}/{count} peak={peak:e} first={:?}",
            &values[..values.len().min(4)]
        ));
    }

    fn qwen_shape(&self, rows: u32) -> QwenShape {
        QwenShape {
            rows,
            width: self.config.hidden,
            heads: self.config.heads,
            kv_heads: self.config.kv_heads,
            head_dim: self.config.head_dim(),
            epsilon: self.config.epsilon,
            rope_theta: self.config.rope_theta,
            ..QwenShape::default()
        }
    }

    fn flame_shape(&self, rows: u32) -> FlameShape {
        FlameShape {
            rows,
            width: self.config.hidden,
            heads: self.config.heads,
            epsilon: self.config.epsilon,
            rope_base: self.config.rope_theta,
            ..FlameShape::default()
        }
    }

    fn encode<T>(
        &self,
        command: &CommandBufferRef,
        name: &str,
        buffers: &[(&Buffer, u64)],
        params: &T,
        groups: MTLSize,
    ) {
        self.encode_threads(command, name, buffers, params, groups, 256);
    }

    fn encode_threads<T>(
        &self,
        command: &CommandBufferRef,
        name: &str,
        buffers: &[(&Buffer, u64)],
        params: &T,
        groups: MTLSize,
        threads: u64,
    ) {
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipelines[name]);
        for (index, &(buffer, offset)) in buffers.iter().enumerate() {
            encoder.set_buffer(index as u64, Some(buffer), offset);
        }
        encoder.set_bytes(
            buffers.len() as u64,
            size_of::<T>() as u64,
            (params as *const T).cast(),
        );
        encoder.dispatch_thread_groups(groups, thread_group(threads));
        encoder.end_encoding();
    }

    fn elementwise<T>(
        &self,
        command: &CommandBufferRef,
        name: &str,
        buffers: &[(&Buffer, u64)],
        params: &T,
        count: usize,
    ) {
        self.encode(
            command,
            name,
            buffers,
            params,
            thread_group((count as u64).div_ceil(256)),
        );
    }

    fn linear(
        &self,
        command: &CommandBufferRef,
        input: W,
        weights: &Buffer,
        offset: usize,
        output: W,
        rows: u32,
        inside: u32,
        outside: u32,
    ) {
        self.linear_buffers(
            command,
            self.buffer(input),
            0,
            weights,
            offset,
            self.buffer(output),
            0,
            rows,
            inside,
            outside,
        );
    }

    fn prefill_linear(
        &self,
        command: &CommandBufferRef,
        input: W,
        weights: &Buffer,
        offset: usize,
        output: W,
        rows: u32,
        inside: u32,
        outside: u32,
    ) -> Result<()> {
        let Some(mps) = self.mps.as_ref().filter(|_| rows >= MPS_MINROWS) else {
            self.linear(
                command, input, weights, offset, output, rows, inside, outside,
            );
            return Ok(());
        };
        let elements = u64::from(inside)
            .checked_mul(u64::from(outside))
            .ok_or("Qwen MPS matrix size overflow")?;
        let element_bytes = if mps.mode == MpsMode::F16 { 2 } else { 4 };
        let bytes = elements
            .checked_mul(element_bytes)
            .ok_or("Qwen MPS matrix byte-size overflow")?;
        if bytes > mps.weights.length() {
            return Err("Qwen MPS weight scratch is too small".into());
        }
        if mps.mode == MpsMode::F32 {
            self.encode(
                command,
                "qwen_widen",
                &[(weights, offset as u64 * 2), (&mps.weights, 0)],
                &elements,
                thread_group(elements.div_ceil(256)),
            );
            return mps.matmul.borrow_mut().encode(
                &self.runtime.device,
                command,
                MpsMatrix::new(self.buffer(input), rows, inside),
                MpsMatrix::new(&mps.weights, outside, inside),
                MpsMatrix::new(self.buffer(output), rows, outside),
                true,
                1.0,
            );
        }

        let input_elements = u64::from(rows)
            .checked_mul(u64::from(inside))
            .ok_or("Qwen MPS input size overflow")?;
        let output_elements = u64::from(rows)
            .checked_mul(u64::from(outside))
            .ok_or("Qwen MPS output size overflow")?;
        let input_scratch = mps.input.as_ref().expect("MPS half input missing");
        let output_scratch = mps.output.as_ref().expect("MPS half output missing");
        if input_elements
            .checked_mul(2)
            .is_none_or(|bytes| bytes > input_scratch.length())
            || output_elements
                .checked_mul(2)
                .is_none_or(|bytes| bytes > output_scratch.length())
        {
            return Err("Qwen MPS half scratch is too small".into());
        }
        self.encode(
            command,
            "qwen_f32_to_f16",
            &[(self.buffer(input), 0), (input_scratch, 0)],
            &input_elements,
            thread_group(input_elements.div_ceil(256)),
        );
        self.encode(
            command,
            "qwen_bf16_f16",
            &[(weights, offset as u64 * 2), (&mps.weights, 0)],
            &elements,
            thread_group(elements.div_ceil(256)),
        );
        mps.matmul.borrow_mut().encode(
            &self.runtime.device,
            command,
            MpsMatrix::half(input_scratch, rows, inside),
            MpsMatrix::half(&mps.weights, outside, inside),
            MpsMatrix::half(output_scratch, rows, outside),
            true,
            1.0,
        )?;
        self.encode(
            command,
            "qwen_f16_to_f32",
            &[(output_scratch, 0), (self.buffer(output), 0)],
            &output_elements,
            thread_group(output_elements.div_ceil(256)),
        );
        Ok(())
    }

    fn half_gemm(
        &self,
        command: &CommandBufferRef,
        input: &Buffer,
        weights: &Buffer,
        offset: usize,
        output: &Buffer,
        rows: u32,
        inside: u32,
        outside: u32,
    ) -> Result<()> {
        let mps = self
            .mps
            .as_ref()
            .filter(|mps| mps.mode == MpsMode::F16)
            .ok_or("Qwen half GEMM requires the FP16 MPS backend")?;
        let elements = u64::from(inside)
            .checked_mul(u64::from(outside))
            .ok_or("Qwen half GEMM weight size overflow")?;
        let input_bytes = u64::from(rows)
            .checked_mul(u64::from(inside))
            .and_then(|value| value.checked_mul(2))
            .ok_or("Qwen half GEMM input size overflow")?;
        let output_bytes = u64::from(rows)
            .checked_mul(u64::from(outside))
            .and_then(|value| value.checked_mul(2))
            .ok_or("Qwen half GEMM output size overflow")?;
        if elements
            .checked_mul(2)
            .is_none_or(|bytes| bytes > mps.weights.length())
            || input_bytes > input.length()
            || output_bytes > output.length()
        {
            return Err("Qwen half GEMM scratch is too small".into());
        }
        self.encode(
            command,
            "qwen_bf16_f16",
            &[(weights, offset as u64 * 2), (&mps.weights, 0)],
            &elements,
            thread_group(elements.div_ceil(256)),
        );
        mps.matmul.borrow_mut().encode(
            &self.runtime.device,
            command,
            MpsMatrix::half(input, rows, inside),
            MpsMatrix::half(&mps.weights, outside, inside),
            MpsMatrix::half(output, rows, outside),
            true,
            1.0,
        )
    }

    fn mlp_expand(
        &self,
        command: &CommandBufferRef,
        weights: &Buffer,
        rows: u32,
        layer: &Layer,
    ) -> Result<bool> {
        let Some(mps) = self
            .mps
            .as_ref()
            .filter(|mps| mps.mode == MpsMode::F16 && rows >= MPS_MINROWS)
        else {
            return Ok(false);
        };
        let c = self.config;
        let input = mps.input.as_ref().expect("MPS half input missing");
        let input_elements = u64::from(rows)
            .checked_mul(u64::from(c.hidden))
            .ok_or("Qwen MLP input size overflow")?;
        let output_elements = u64::from(rows)
            .checked_mul(u64::from(c.intermediate))
            .ok_or("Qwen MLP output size overflow")?;
        let weight_elements = usize::try_from(c.hidden)
            .ok()
            .and_then(|hidden| {
                usize::try_from(c.intermediate)
                    .ok()
                    .and_then(|width| hidden.checked_mul(width))
            })
            .ok_or("Qwen MLP weight size overflow")?;
        if layer
            .gate_weight
            .checked_add(weight_elements)
            .is_none_or(|offset| offset != layer.up_weight)
        {
            return Ok(false);
        }
        self.encode(
            command,
            "qwen_f32_to_f16",
            &[(self.buffer(W::Norm), 0), (input, 0)],
            &input_elements,
            thread_group(input_elements.div_ceil(256)),
        );
        self.half_gemm(
            command,
            input,
            weights,
            layer.gate_weight,
            self.buffer(W::Gate),
            rows,
            c.hidden,
            c.intermediate
                .checked_mul(2)
                .ok_or("Qwen MLP fused width overflow")?,
        )?;
        let shape = [output_elements, u64::from(c.intermediate)];
        self.encode(
            command,
            "qwen_silu16",
            &[(self.buffer(W::Gate), 0), (self.buffer(W::Activation), 0)],
            &shape,
            thread_group(output_elements.div_ceil(256)),
        );
        Ok(true)
    }

    fn mlp_reduce(
        &self,
        command: &CommandBufferRef,
        weights: &Buffer,
        rows: u32,
        layer: &Layer,
    ) -> Result<()> {
        let mps = self
            .mps
            .as_ref()
            .filter(|mps| mps.mode == MpsMode::F16)
            .ok_or("Qwen half MLP reduction requires the FP16 MPS backend")?;
        let c = self.config;
        let output = mps.output.as_ref().expect("MPS half output missing");
        self.half_gemm(
            command,
            self.buffer(W::Activation),
            weights,
            layer.down_weight,
            output,
            rows,
            c.intermediate,
            c.hidden,
        )?;
        let elements = u64::from(rows)
            .checked_mul(u64::from(c.hidden))
            .ok_or("Qwen MLP reduction size overflow")?;
        self.encode(
            command,
            "qwen_f16_to_f32",
            &[(output, 0), (self.buffer(W::Attended), 0)],
            &elements,
            thread_group(elements.div_ceil(256)),
        );
        Ok(())
    }

    fn qkv(&self, command: &CommandBufferRef, weights: &Buffer, rows: u32, layer: &Layer) {
        let c = self.config;
        let params = QwenQkvShape {
            rows,
            hidden: c.hidden,
            kv_width: c.kv_width(),
        };
        self.encode_threads(
            command,
            "qwen_qkv",
            &[
                (self.buffer(W::Norm), 0),
                (weights, layer.q_weight as u64 * 2),
                (weights, layer.k_weight as u64 * 2),
                (weights, layer.v_weight as u64 * 2),
                (weights, layer.q_bias as u64 * 2),
                (weights, layer.k_bias as u64 * 2),
                (weights, layer.v_bias as u64 * 2),
                (self.buffer(W::QRaw), 0),
                (self.buffer(W::KRaw), 0),
                (self.buffer(W::VRaw), 0),
            ],
            &params,
            thread_group((u64::from(c.hidden) + 2 * u64::from(c.kv_width())).div_ceil(32)),
            32,
        );
    }

    fn silu_rows(&self, command: &CommandBufferRef, weights: &Buffer, rows: u32, layer: &Layer) {
        let c = self.config;
        let params = QwenMlpShape {
            rows,
            hidden: c.hidden,
            intermediate: c.intermediate,
        };
        self.encode_threads(
            command,
            "qwen_mlp_rows",
            &[
                (self.buffer(W::Norm), 0),
                (weights, layer.gate_weight as u64 * 2),
                (weights, layer.up_weight as u64 * 2),
                (self.buffer(W::Activation), 0),
            ],
            &params,
            thread_group(u64::from(c.intermediate).div_ceil(32)),
            32,
        );
    }

    fn linear_buffers(
        &self,
        command: &CommandBufferRef,
        input: &Buffer,
        input_offset: u64,
        weights: &Buffer,
        offset: usize,
        output: &Buffer,
        output_offset: u64,
        rows: u32,
        inside: u32,
        outside: u32,
    ) {
        let params = Matmul {
            m: rows,
            n: outside,
            k: inside,
            transpose_b: 1,
            stride_a: 0,
            stride_b: 0,
            stride_c: 0,
        };
        let simdgroup = rows >= 64 && inside % 8 == 0 && outside % 32 == 0;
        let pipeline = if (1..=4).contains(&rows) {
            "qwen_gemv_rows"
        } else if rows < 32 {
            "qwen_gemv"
        } else if simdgroup {
            "qwen_simd_gemm"
        } else {
            "flame_linear"
        };
        let threads = if pipeline == "qwen_gemv_rows" || rows <= 4 {
            32
        } else if simdgroup {
            128
        } else {
            256
        };
        self.encode_threads(
            command,
            pipeline,
            &[
                (input, input_offset),
                (weights, offset as u64 * 2),
                (output, output_offset),
            ],
            &params,
            MTLSize {
                width: u64::from(outside).div_ceil(32),
                height: if simdgroup {
                    u64::from(rows).div_ceil(64)
                } else {
                    u64::from(rows).div_ceil(32)
                },
                depth: 1,
            },
            threads,
        );
    }

    fn rms(
        &self,
        command: &CommandBufferRef,
        input: W,
        weights: &Buffer,
        offset: usize,
        output: W,
        rows: u32,
    ) {
        let params = self.qwen_shape(rows);
        self.encode(
            command,
            "qwen_rms",
            &[
                (self.buffer(input), 0),
                (weights, offset as u64 * 2),
                (self.buffer(output), 0),
            ],
            &params,
            thread_group(u64::from(rows)),
        );
    }

    fn residual(&self, command: &CommandBufferRef, input: W, update: W, rows: u32) {
        let params = self.qwen_shape(rows);
        self.elementwise(
            command,
            "qwen_residual",
            &[(self.buffer(input), 0), (self.buffer(update), 0)],
            &params,
            rows as usize * self.config.hidden as usize,
        );
    }

    fn argmax_output(&self, weights: &Buffer, input_offset: u64) -> Result<u32> {
        self.write(W::Invalid, &[0u32]);
        let c = self.config;
        let params = Matmul {
            m: 1,
            n: c.vocab,
            k: c.hidden,
            transpose_b: 1,
            ..Matmul::default()
        };
        let argmax = ArgmaxShape { width: c.vocab };
        let command = self.runtime.queue.new_command_buffer();
        command.set_label("Qwen argmax output");
        self.encode_threads(
            &command,
            "qwen_gemv",
            &[
                (self.buffer(W::Norm), input_offset),
                (weights, self.layout.embedding as u64 * 2),
                (self.buffer(W::Logits), 0),
            ],
            &params,
            MTLSize {
                width: u64::from(c.vocab).div_ceil(32),
                height: 1,
                depth: 1,
            },
            32,
        );
        self.encode(
            &command,
            "qwen_argmax",
            &[
                (self.buffer(W::Logits), 0),
                (self.buffer(W::Tokens), 0),
                (self.buffer(W::Invalid), 0),
            ],
            &argmax,
            thread_group(1),
        );
        finish(&command)?;
        if self.read::<u32>(W::Invalid, 1)[0] != 0 {
            return Err("Qwen argmax received nonfinite logits".into());
        }
        let token = self.read::<u32>(W::Tokens, 1)[0];
        if token >= c.vocab {
            return Err("Qwen argmax returned an invalid token".into());
        }
        Ok(token)
    }

    fn decode_token(&self, weights: &Buffer, state: &mut GenerationState) -> Result<u32> {
        if state.position >= self.max_tokens {
            return Err("Qwen generation cache is full".into());
        }
        let c = self.config;
        let qwen = self.qwen_shape(1);
        let hidden_elements = c.hidden as usize;
        let position = state.position;
        let sequence = position
            .checked_add(1)
            .ok_or("Qwen generation position overflow")?;
        write_buffer(&state.position_buffer, &[position]);
        let command = self.runtime.queue.new_command_buffer();
        command.set_label("Qwen decode");
        self.write(W::Invalid, &[0u32]);
        self.elementwise(
            &command,
            "qwen_embedding",
            &[
                (weights, self.layout.embedding as u64 * 2),
                (self.buffer(W::Tokens), 0),
                (self.buffer(W::X), 0),
            ],
            &qwen,
            hidden_elements,
        );
        for (layer_index, layer) in self.layout.layers.iter().enumerate() {
            self.rms(&command, W::X, weights, layer.input_norm, W::Norm, 1);
            self.qkv(&command, weights, 1, layer);
            let rope = DecodeRopeShape {
                heads: c.heads,
                kv_heads: c.kv_heads,
                head_dim: c.head_dim(),
                capacity: state.capacity,
                position,
                batch: 1,
                cache_stride: state.cache_stride as u32,
                rope_theta: c.rope_theta,
            };
            self.encode(
                &command,
                "qwen_drope",
                &[
                    (self.buffer(W::QRaw), 0),
                    (self.buffer(W::KRaw), 0),
                    (self.buffer(W::VRaw), 0),
                    (self.buffer(W::Q), 0),
                    (&state.key_cache, state.layer_offset(c, layer_index)),
                    (&state.value_cache, state.layer_offset(c, layer_index)),
                    (&self.rope_table, 0),
                    (&state.position_buffer, 0),
                ],
                &rope,
                thread_group(u64::from(c.hidden).div_ceil(256)),
            );
            let attention = DecodeAttentionShape {
                sequence,
                capacity: state.capacity,
                heads: c.heads,
                kv_heads: c.kv_heads,
                head_dim: c.head_dim(),
                batch: 1,
                cache_stride: state.cache_stride as u32,
                scale: 1.0 / (c.head_dim() as f32).sqrt(),
            };
            self.encode_threads(
                &command,
                "qwen_dattn",
                &[
                    (self.buffer(W::Q), 0),
                    (&state.key_cache, state.layer_offset(c, layer_index)),
                    (&state.value_cache, state.layer_offset(c, layer_index)),
                    (self.buffer(W::Update), 0),
                    (&state.position_buffer, 0),
                ],
                &attention,
                thread_group(u64::from(c.heads)),
                32,
            );
            self.linear(
                &command,
                W::Update,
                weights,
                layer.o_weight,
                W::Attended,
                1,
                c.hidden,
                c.hidden,
            );
            self.residual(&command, W::X, W::Attended, 1);
            self.rms(&command, W::X, weights, layer.post_norm, W::Norm, 1);
            self.silu_rows(&command, weights, 1, layer);
            self.linear(
                &command,
                W::Activation,
                weights,
                layer.down_weight,
                W::Attended,
                1,
                c.intermediate,
                c.hidden,
            );
            self.residual(&command, W::X, W::Attended, 1);
        }
        self.rms(&command, W::X, weights, self.layout.final_norm, W::Norm, 1);
        let params = Matmul {
            m: 1,
            n: c.vocab,
            k: c.hidden,
            transpose_b: 1,
            ..Matmul::default()
        };
        self.encode_threads(
            &command,
            "qwen_gemv",
            &[
                (self.buffer(W::Norm), 0),
                (weights, self.layout.embedding as u64 * 2),
                (self.buffer(W::Logits), 0),
            ],
            &params,
            MTLSize {
                width: u64::from(c.vocab).div_ceil(32),
                height: 1,
                depth: 1,
            },
            32,
        );
        self.encode(
            &command,
            "qwen_argmax",
            &[
                (self.buffer(W::Logits), 0),
                (self.buffer(W::Tokens), 0),
                (self.buffer(W::Invalid), 0),
            ],
            &ArgmaxShape { width: c.vocab },
            thread_group(1),
        );
        finish(&command)?;
        self.trace_values("decode final norm", W::Norm, (c.hidden as usize).min(4));
        state.position = sequence;
        if self.read::<u32>(W::Invalid, 1)[0] != 0 {
            return Err("Qwen decode produced nonfinite logits".into());
        }
        let token = self.read::<u32>(W::Tokens, 1)[0];
        if token >= c.vocab {
            return Err("Qwen decode produced an invalid token".into());
        }
        Ok(token)
    }

    fn decode_profile(
        &self,
        weights: &Buffer,
        state: &mut GenerationState,
        times: &mut BTreeMap<&'static str, f32>,
    ) -> Result<u32> {
        if state.position >= self.max_tokens {
            return Err("Qwen generation cache is full".into());
        }
        let c = self.config;
        let qwen = self.qwen_shape(1);
        let hidden_elements = c.hidden as usize;
        let position = state.position;
        let sequence = position
            .checked_add(1)
            .ok_or("Qwen generation position overflow")?;
        write_buffer(&state.position_buffer, &[position]);
        self.write(W::Invalid, &[0u32]);

        macro_rules! timed_command {
            ($name:literal, $command:ident, $body:block) => {{
                let $command = self.runtime.queue.new_command_buffer();
                $command.set_label($name);
                let start = Instant::now();
                $body
                finish(&$command)?;
                *times.entry($name).or_insert(0.0) += start.elapsed().as_secs_f32() * 1000.0;
            }};
        }

        timed_command!("qwen_embedding", command, {
            self.elementwise(
                &command,
                "qwen_embedding",
                &[
                    (weights, self.layout.embedding as u64 * 2),
                    (self.buffer(W::Tokens), 0),
                    (self.buffer(W::X), 0),
                ],
                &qwen,
                hidden_elements,
            );
        });

        for (layer_index, layer) in self.layout.layers.iter().enumerate() {
            timed_command!("qwen_rms_input", command, {
                self.rms(&command, W::X, weights, layer.input_norm, W::Norm, 1);
            });
            timed_command!("qwen_qkv", command, {
                self.qkv(&command, weights, 1, layer);
            });
            let rope = DecodeRopeShape {
                heads: c.heads,
                kv_heads: c.kv_heads,
                head_dim: c.head_dim(),
                capacity: state.capacity,
                position,
                batch: 1,
                cache_stride: state.cache_stride as u32,
                rope_theta: c.rope_theta,
            };
            timed_command!("qwen_drope", command, {
                self.encode(
                    &command,
                    "qwen_drope",
                    &[
                        (self.buffer(W::QRaw), 0),
                        (self.buffer(W::KRaw), 0),
                        (self.buffer(W::VRaw), 0),
                        (self.buffer(W::Q), 0),
                        (&state.key_cache, state.layer_offset(c, layer_index)),
                        (&state.value_cache, state.layer_offset(c, layer_index)),
                        (&self.rope_table, 0),
                        (&state.position_buffer, 0),
                    ],
                    &rope,
                    thread_group(u64::from(c.hidden).div_ceil(256)),
                );
            });
            let attention = DecodeAttentionShape {
                sequence,
                capacity: state.capacity,
                heads: c.heads,
                kv_heads: c.kv_heads,
                head_dim: c.head_dim(),
                batch: 1,
                cache_stride: state.cache_stride as u32,
                scale: 1.0 / (c.head_dim() as f32).sqrt(),
            };
            timed_command!("qwen_dattn", command, {
                self.encode_threads(
                    &command,
                    "qwen_dattn",
                    &[
                        (self.buffer(W::Q), 0),
                        (&state.key_cache, state.layer_offset(c, layer_index)),
                        (&state.value_cache, state.layer_offset(c, layer_index)),
                        (self.buffer(W::Update), 0),
                        (&state.position_buffer, 0),
                    ],
                    &attention,
                    thread_group(u64::from(c.heads)),
                    32,
                );
            });
            timed_command!("qwen_attention_output_projection", command, {
                self.linear(
                    &command,
                    W::Update,
                    weights,
                    layer.o_weight,
                    W::Attended,
                    1,
                    c.hidden,
                    c.hidden,
                );
                self.residual(&command, W::X, W::Attended, 1);
            });
            timed_command!("qwen_rms_post_attention", command, {
                self.rms(&command, W::X, weights, layer.post_norm, W::Norm, 1);
            });
            timed_command!("qwen_mlp_rows", command, {
                self.silu_rows(&command, weights, 1, layer);
            });
            timed_command!("qwen_mlp_down_projection", command, {
                self.linear(
                    &command,
                    W::Activation,
                    weights,
                    layer.down_weight,
                    W::Attended,
                    1,
                    c.intermediate,
                    c.hidden,
                );
                self.residual(&command, W::X, W::Attended, 1);
            });
        }
        timed_command!("qwen_rms_final", command, {
            self.rms(&command, W::X, weights, self.layout.final_norm, W::Norm, 1);
        });
        let params = Matmul {
            m: 1,
            n: c.vocab,
            k: c.hidden,
            transpose_b: 1,
            ..Matmul::default()
        };
        timed_command!("qwen_lm_head_argmax", command, {
            self.encode_threads(
                &command,
                "qwen_gemv",
                &[
                    (self.buffer(W::Norm), 0),
                    (weights, self.layout.embedding as u64 * 2),
                    (self.buffer(W::Logits), 0),
                ],
                &params,
                MTLSize {
                    width: u64::from(c.vocab).div_ceil(32),
                    height: 1,
                    depth: 1,
                },
                32,
            );
            self.encode(
                &command,
                "qwen_argmax",
                &[
                    (self.buffer(W::Logits), 0),
                    (self.buffer(W::Tokens), 0),
                    (self.buffer(W::Invalid), 0),
                ],
                &ArgmaxShape { width: c.vocab },
                thread_group(1),
            );
        });

        state.position = sequence;
        if self.read::<u32>(W::Invalid, 1)[0] != 0 {
            return Err("Qwen decode produced nonfinite logits".into());
        }
        let token = self.read::<u32>(W::Tokens, 1)[0];
        if token >= c.vocab {
            return Err("Qwen decode produced an invalid token".into());
        }
        Ok(token)
    }

    fn decode_logits(&self, weights: &Buffer, state: &mut GenerationState) -> Result<Vec<f32>> {
        self.decode_token(weights, state)?;
        let logits = self.read::<f32>(W::Logits, self.config.vocab as usize);
        if logits.iter().any(|value| !value.is_finite()) {
            return Err("Qwen sampled decode produced nonfinite logits".into());
        }
        Ok(logits)
    }

    fn batch_logits(
        &self,
        weights: &Buffer,
        state: &mut GenerationState,
        positions: &[u32],
        greedy: bool,
    ) -> Result<BatchDecodeOutput> {
        if positions.len() != state.batch as usize {
            return Err("Qwen batch decode requires one position per sequence".into());
        }
        if positions
            .iter()
            .any(|&position| position >= state.capacity || position >= self.max_tokens)
        {
            return Err("Qwen batch generation cache is full".into());
        }
        write_buffer(&state.position_buffer, positions);
        self.write(W::Invalid, &[0u32]);
        let c = self.config;
        let rows = state.batch;
        let qwen = self.qwen_shape(rows);
        let hidden_elements = rows as usize * c.hidden as usize;
        let command = self.runtime.queue.new_command_buffer();
        command.set_label("Qwen batched decode");
        self.elementwise(
            &command,
            "qwen_embedding",
            &[
                (weights, self.layout.embedding as u64 * 2),
                (self.buffer(W::Tokens), 0),
                (self.buffer(W::X), 0),
            ],
            &qwen,
            hidden_elements,
        );
        for (layer_index, layer) in self.layout.layers.iter().enumerate() {
            self.rms(&command, W::X, weights, layer.input_norm, W::Norm, rows);
            self.qkv(&command, weights, rows, layer);
            let rope = DecodeRopeShape {
                heads: c.heads,
                kv_heads: c.kv_heads,
                head_dim: c.head_dim(),
                capacity: state.capacity,
                position: 0,
                batch: rows,
                cache_stride: state.cache_stride as u32,
                rope_theta: c.rope_theta,
            };
            self.encode(
                &command,
                "qwen_drope",
                &[
                    (self.buffer(W::QRaw), 0),
                    (self.buffer(W::KRaw), 0),
                    (self.buffer(W::VRaw), 0),
                    (self.buffer(W::Q), 0),
                    (&state.key_cache, state.layer_batch(layer_index)),
                    (&state.value_cache, state.layer_batch(layer_index)),
                    (&self.rope_table, 0),
                    (&state.position_buffer, 0),
                ],
                &rope,
                thread_group((hidden_elements as u64).div_ceil(256)),
            );
            let attention = DecodeAttentionShape {
                sequence: 0,
                capacity: state.capacity,
                heads: c.heads,
                kv_heads: c.kv_heads,
                head_dim: c.head_dim(),
                batch: rows,
                cache_stride: state.cache_stride as u32,
                scale: 1.0 / (c.head_dim() as f32).sqrt(),
            };
            self.encode_threads(
                &command,
                "qwen_dattn",
                &[
                    (self.buffer(W::Q), 0),
                    (&state.key_cache, state.layer_batch(layer_index)),
                    (&state.value_cache, state.layer_batch(layer_index)),
                    (self.buffer(W::Update), 0),
                    (&state.position_buffer, 0),
                ],
                &attention,
                MTLSize {
                    width: u64::from(rows) * u64::from(c.heads),
                    height: 1,
                    depth: 1,
                },
                32,
            );
            self.linear(
                &command,
                W::Update,
                weights,
                layer.o_weight,
                W::Attended,
                rows,
                c.hidden,
                c.hidden,
            );
            self.residual(&command, W::X, W::Attended, rows);
            self.rms(&command, W::X, weights, layer.post_norm, W::Norm, rows);
            self.silu_rows(&command, weights, rows, layer);
            self.linear(
                &command,
                W::Activation,
                weights,
                layer.down_weight,
                W::Attended,
                rows,
                c.intermediate,
                c.hidden,
            );
            self.residual(&command, W::X, W::Attended, rows);
        }
        self.rms(
            &command,
            W::X,
            weights,
            self.layout.final_norm,
            W::Norm,
            rows,
        );
        self.linear(
            &command,
            W::Norm,
            weights,
            self.layout.embedding,
            W::Logits,
            rows,
            c.hidden,
            c.vocab,
        );
        if greedy {
            self.encode(
                &command,
                "qwen_argn",
                &[
                    (self.buffer(W::Logits), 0),
                    (self.buffer(W::Tokens), 0),
                    (self.buffer(W::Invalid), 0),
                ],
                &ArgmaxBatchShape {
                    rows,
                    width: c.vocab,
                },
                thread_group(u64::from(rows)),
            );
            finish(&command)?;
            if self.read::<u32>(W::Invalid, 1)[0] != 0 {
                return Err("Qwen batched argmax received nonfinite logits".into());
            }
            let tokens = self.read::<u32>(W::Tokens, rows as usize);
            if tokens.iter().any(|&token| token >= c.vocab) {
                return Err("Qwen batched argmax returned an invalid token".into());
            }
            return Ok(BatchDecodeOutput::Tokens(tokens));
        }
        finish(&command)?;
        let logits = self.read::<f32>(W::Logits, state.batch as usize * c.vocab as usize);
        if logits.iter().any(|value| !value.is_finite()) {
            return Err("Qwen batched decode produced nonfinite logits".into());
        }
        Ok(BatchDecodeOutput::Logits(
            logits
                .chunks_exact(c.vocab as usize)
                .map(Vec::from)
                .collect(),
        ))
    }

    fn output_logits(&self, weights: &Buffer, input_offset: u64) -> Result<Vec<f32>> {
        let c = self.config;
        let params = Matmul {
            m: 1,
            n: c.vocab,
            k: c.hidden,
            transpose_b: 1,
            ..Matmul::default()
        };
        let command = self.runtime.queue.new_command_buffer();
        command.set_label("Qwen sampled output");
        self.encode_threads(
            &command,
            "qwen_gemv",
            &[
                (self.buffer(W::Norm), input_offset),
                (weights, self.layout.embedding as u64 * 2),
                (self.buffer(W::Logits), 0),
            ],
            &params,
            MTLSize {
                width: u64::from(c.vocab).div_ceil(32),
                height: 1,
                depth: 1,
            },
            32,
        );
        finish(&command)?;
        let logits = self.read::<f32>(W::Logits, c.vocab as usize);
        if logits.iter().any(|value| !value.is_finite()) {
            return Err("Qwen sampled output produced nonfinite logits".into());
        }
        Ok(logits)
    }

    fn forward(
        &self,
        weights: &Buffer,
        rows: u32,
        mut cache: Option<&mut GenerationState>,
    ) -> Result<Option<QwenStageProfile>> {
        if let Some(cache) = cache.as_deref_mut() {
            let mut start = 0;
            let mut stages = None;
            while start < rows {
                let chunk = (rows - start).min(self.prefill_chunk);
                if let Some(chunk_stages) =
                    self.forward_chunk(weights, chunk, start, Some(&*cache))?
                {
                    stages
                        .get_or_insert_with(QwenStageProfile::default)
                        .merge(chunk_stages);
                }
                cache.position = cache
                    .position
                    .checked_add(chunk)
                    .ok_or("Qwen generation position overflow")?;
                start += chunk;
            }
            return Ok(stages);
        }
        if rows > REF_ROWS {
            return Err(
                "Qwen reference forward is limited to 256 tokens; use generate for long prompts"
                    .into(),
            );
        }
        self.forward_chunk(weights, rows, 0, None)
    }

    fn forward_chunk(
        &self,
        weights: &Buffer,
        rows: u32,
        start: u32,
        cache: Option<&GenerationState>,
    ) -> Result<Option<QwenStageProfile>> {
        if let Some(cache) = cache {
            if cache
                .position
                .checked_add(rows)
                .is_none_or(|end| end > cache.capacity)
            {
                return Err("Qwen prefill exceeds the generation cache capacity".into());
            }
        }
        let c = self.config;
        let qwen = QwenShape {
            start,
            ..self.qwen_shape(rows)
        };
        let flame = self.flame_shape(rows);
        let hidden_elements = rows as usize * c.hidden as usize;
        let kv_elements = rows as usize * c.kv_width() as usize;
        let stage_mode = std::env::var_os("ENNX_QWEN_STAGE_PROFILE").is_some();
        let sync_layers = !stage_mode && std::env::var_os("ENNX_QWEN_SYNC_LAYERS").is_some();
        let mut stages = stage_mode.then(|| QwenStageProfile {
            chunks: 1,
            ..QwenStageProfile::default()
        });
        let mut command = self.runtime.queue.new_command_buffer();
        command.set_label("Qwen forward");
        self.elementwise(
            &command,
            "qwen_embedding",
            &[
                (weights, self.layout.embedding as u64 * 2),
                (self.buffer(W::Tokens), 0),
                (self.buffer(W::X), 0),
            ],
            &qwen,
            hidden_elements,
        );
        if let Some(profile) = stages.as_mut() {
            command = stage_split(&self.runtime, command, profile, Stage::Embed, "Qwen qkv")?;
        } else if sync_layers {
            finish(&command)?;
            self.trace_values("embedding", W::X, hidden_elements.min(4));
            self.trace_at(
                "embedding (last row)",
                W::X,
                (rows as usize - 1) * c.hidden as usize,
                hidden_elements.min(4),
            );
            command = self.runtime.queue.new_command_buffer();
            command.set_label("Qwen layer 0");
        }
        for (layer_index, layer) in self.layout.layers.iter().enumerate() {
            self.rms(&command, W::X, weights, layer.input_norm, W::Norm, rows);
            self.prefill_linear(
                &command,
                W::Norm,
                weights,
                layer.q_weight,
                W::QRaw,
                rows,
                c.hidden,
                c.hidden,
            )?;
            self.elementwise(
                &command,
                "qwen_bias",
                &[
                    (self.buffer(W::QRaw), 0),
                    (weights, layer.q_bias as u64 * 2),
                ],
                &qwen,
                hidden_elements,
            );
            self.prefill_linear(
                &command,
                W::Norm,
                weights,
                layer.k_weight,
                W::KRaw,
                rows,
                c.hidden,
                c.kv_width(),
            )?;
            let kv_shape = QwenShape {
                width: c.kv_width(),
                ..qwen
            };
            self.elementwise(
                &command,
                "qwen_bias",
                &[
                    (self.buffer(W::KRaw), 0),
                    (weights, layer.k_bias as u64 * 2),
                ],
                &kv_shape,
                kv_elements,
            );
            self.prefill_linear(
                &command,
                W::Norm,
                weights,
                layer.v_weight,
                W::VRaw,
                rows,
                c.hidden,
                c.kv_width(),
            )?;
            self.elementwise(
                &command,
                "qwen_bias",
                &[
                    (self.buffer(W::VRaw), 0),
                    (weights, layer.v_bias as u64 * 2),
                ],
                &kv_shape,
                kv_elements,
            );
            self.encode(
                &command,
                "qwen_rope",
                &[
                    (self.buffer(W::QRaw), 0),
                    (self.buffer(W::KRaw), 0),
                    (self.buffer(W::VRaw), 0),
                    (self.buffer(W::Q), 0),
                    (self.buffer(W::K), 0),
                    (self.buffer(W::V), 0),
                    (&self.rope_table, 0),
                ],
                &qwen,
                thread_group(hidden_elements.div_ceil(256) as u64),
            );
            if let Some(profile) = stages.as_mut() {
                command = stage_split(
                    &self.runtime,
                    command,
                    profile,
                    Stage::Qkv,
                    "Qwen attention",
                )?;
            }
            if let Some(cache) = cache {
                let cache_shape = CacheShape {
                    rows,
                    kv_heads: c.kv_heads,
                    head_dim: c.head_dim(),
                    capacity: cache.capacity,
                    position: cache.position,
                };
                self.encode(
                    &command,
                    "qwen_cache",
                    &[
                        (self.buffer(W::K), 0),
                        (self.buffer(W::V), 0),
                        (&cache.key_cache, cache.layer_offset(c, layer_index)),
                        (&cache.value_cache, cache.layer_offset(c, layer_index)),
                    ],
                    &cache_shape,
                    thread_group((rows as u64 * u64::from(c.kv_width())).div_ceil(256)),
                );
                let attention = PrefillAttentionShape {
                    rows,
                    sequence: cache.position + rows,
                    capacity: cache.capacity,
                    heads: c.heads,
                    kv_heads: c.kv_heads,
                    head_dim: c.head_dim(),
                    scale: 1.0 / (c.head_dim() as f32).sqrt(),
                };
                let buffers = [
                    (self.buffer(W::Q), 0),
                    (&cache.key_cache, cache.layer_offset(c, layer_index)),
                    (&cache.value_cache, cache.layer_offset(c, layer_index)),
                    (self.buffer(W::Update), 0),
                ];
                if self.tile_attn {
                    self.encode_threads(
                        &command,
                        "qwen_attn16",
                        &buffers,
                        &attention,
                        MTLSize {
                            width: u64::from(c.heads),
                            height: u64::from(rows).div_ceil(16),
                            depth: 1,
                        },
                        256,
                    );
                } else {
                    self.encode_threads(
                        &command,
                        "qwen_arow",
                        &buffers,
                        &attention,
                        thread_group(u64::from(rows) * u64::from(c.heads)),
                        32,
                    );
                }
            } else {
                self.elementwise(
                    &command,
                    "qwen_repkv",
                    &[(self.buffer(W::K), 0), (self.buffer(W::KRepeat), 0)],
                    &qwen,
                    hidden_elements,
                );
                self.elementwise(
                    &command,
                    "qwen_repkv",
                    &[(self.buffer(W::V), 0), (self.buffer(W::VRepeat), 0)],
                    &qwen,
                    hidden_elements,
                );
                let d = c.head_dim();
                let qk = Matmul {
                    m: rows,
                    n: rows,
                    k: d,
                    transpose_b: 1,
                    stride_a: u64::from(rows) * u64::from(d),
                    stride_b: u64::from(rows) * u64::from(d),
                    stride_c: u64::from(rows) * u64::from(rows),
                };
                self.encode(
                    &command,
                    "flame_matmul",
                    &[
                        (self.buffer(W::Q), 0),
                        (self.buffer(W::KRepeat), 0),
                        (self.buffer(W::Scores), 0),
                    ],
                    &qk,
                    MTLSize {
                        width: u64::from(rows).div_ceil(32),
                        height: u64::from(rows).div_ceil(32),
                        depth: u64::from(c.heads),
                    },
                );
                self.encode(
                    &command,
                    "flame_softmax",
                    &[(self.buffer(W::Scores), 0)],
                    &flame,
                    thread_group(u64::from(rows) * u64::from(c.heads)),
                );
                let av = Matmul {
                    m: rows,
                    n: d,
                    k: rows,
                    transpose_b: 0,
                    stride_a: u64::from(rows) * u64::from(rows),
                    stride_b: u64::from(rows) * u64::from(d),
                    stride_c: u64::from(rows) * u64::from(d),
                };
                self.encode(
                    &command,
                    "flame_matmul",
                    &[
                        (self.buffer(W::Scores), 0),
                        (self.buffer(W::VRepeat), 0),
                        (self.buffer(W::Attended), 0),
                    ],
                    &av,
                    MTLSize {
                        width: u64::from(d).div_ceil(32),
                        height: u64::from(rows).div_ceil(32),
                        depth: u64::from(c.heads),
                    },
                );
                self.elementwise(
                    &command,
                    "qwen_aout",
                    &[(self.buffer(W::Attended), 0), (self.buffer(W::Update), 0)],
                    &qwen,
                    hidden_elements,
                );
            }
            if let Some(profile) = stages.as_mut() {
                command = stage_split(
                    &self.runtime,
                    command,
                    profile,
                    Stage::Attn,
                    "Qwen attention output",
                )?;
            }
            self.prefill_linear(
                &command,
                W::Update,
                weights,
                layer.o_weight,
                W::Attended,
                rows,
                c.hidden,
                c.hidden,
            )?;
            self.residual(&command, W::X, W::Attended, rows);
            if let Some(profile) = stages.as_mut() {
                command = stage_split(
                    &self.runtime,
                    command,
                    profile,
                    Stage::AttnOut,
                    "Qwen MLP expansion",
                )?;
            }
            self.rms(&command, W::X, weights, layer.post_norm, W::Norm, rows);
            let half_mlp = self.mlp_expand(&command, weights, rows, layer)?;
            if !half_mlp {
                self.prefill_linear(
                    &command,
                    W::Norm,
                    weights,
                    layer.gate_weight,
                    W::Gate,
                    rows,
                    c.hidden,
                    c.intermediate,
                )?;
                self.prefill_linear(
                    &command,
                    W::Norm,
                    weights,
                    layer.up_weight,
                    W::Up,
                    rows,
                    c.hidden,
                    c.intermediate,
                )?;
                let activation = QwenShape {
                    hidden: c.intermediate,
                    ..qwen
                };
                self.elementwise(
                    &command,
                    "qwen_silu",
                    &[
                        (self.buffer(W::Gate), 0),
                        (self.buffer(W::Up), 0),
                        (self.buffer(W::Activation), 0),
                    ],
                    &activation,
                    rows as usize * c.intermediate as usize,
                );
            }
            if let Some(profile) = stages.as_mut() {
                command = stage_split(
                    &self.runtime,
                    command,
                    profile,
                    Stage::MlpExpand,
                    "Qwen MLP reduction",
                )?;
            }
            if half_mlp {
                self.mlp_reduce(&command, weights, rows, layer)?;
            } else {
                self.prefill_linear(
                    &command,
                    W::Activation,
                    weights,
                    layer.down_weight,
                    W::Attended,
                    rows,
                    c.intermediate,
                    c.hidden,
                )?;
            }
            self.residual(&command, W::X, W::Attended, rows);
            if let Some(profile) = stages.as_mut() {
                command = stage_split(
                    &self.runtime,
                    command,
                    profile,
                    Stage::MlpReduce,
                    "Qwen qkv",
                )?;
            } else if sync_layers {
                finish(&command)?;
                self.trace_at(
                    &format!("layer {} (last row)", layer_index),
                    W::X,
                    (rows as usize - 1) * c.hidden as usize,
                    hidden_elements.min(4),
                );
                command = self.runtime.queue.new_command_buffer();
                command.set_label(&format!("Qwen layer {}", layer_index + 1));
            }
        }
        self.rms(
            &command,
            W::X,
            weights,
            self.layout.final_norm,
            W::Norm,
            rows,
        );
        finish(&command)?;
        if let Some(profile) = stages.as_mut() {
            profile.add(Stage::FinalNorm, &command);
        }
        self.trace_at(
            "final norm (last row)",
            W::Norm,
            (rows as usize - 1) * c.hidden as usize,
            hidden_elements.min(4),
        );
        Ok(stages)
    }

    fn output(
        &self,
        weights: &Buffer,
        rows: u32,
        first_row: u32,
        scored: u32,
        host_logits: Option<&mut [f32]>,
    ) -> Result<f32> {
        self.write(W::Invalid, &[0u32]);
        if scored != 0 && first_row != 0 {
            self.write(W::Losses, &vec![0.0f32; first_row as usize]);
        }
        self.project_rows(weights, rows, first_row, 0, rows, scored != 0, host_logits)?;
        if scored == 0 {
            return Ok(0.0);
        }
        self.mean_loss(rows, scored)
    }

    fn project_rows(
        &self,
        weights: &Buffer,
        local_rows: u32,
        first_local: u32,
        global_start: u32,
        sequence: u32,
        score: bool,
        mut host_logits: Option<&mut [f32]>,
    ) -> Result<()> {
        for start in (first_local..local_rows).step_by(LOGIT_ROWS) {
            let chunk = (local_rows - start).min(LOGIT_ROWS as u32);
            let output_rows = chunk as usize * self.vocab();
            let output = self.buffer(W::Logits);
            if output_rows * size_of::<f32>() > output.length() as usize {
                return Err("Qwen logit workspace is smaller than the requested chunk".into());
            }
            let command = self.runtime.queue.new_command_buffer();
            command.set_label("Qwen output");
            self.linear_buffers(
                &command,
                self.buffer(W::Norm),
                u64::from(start) * u64::from(self.config.hidden) * 4,
                weights,
                self.layout.embedding,
                output,
                0,
                chunk,
                self.config.hidden,
                self.config.vocab,
            );
            if score {
                let shape = FlameShape {
                    rows: chunk,
                    width: self.config.vocab,
                    start: global_start + start,
                    sequence,
                    ..self.flame_shape(chunk)
                };
                self.encode(
                    &command,
                    "flame_xent",
                    &[
                        (output, 0),
                        (self.buffer(W::Tokens), 0),
                        (self.buffer(W::Masks), 0),
                        (self.buffer(W::Losses), 0),
                        (self.buffer(W::Invalid), 0),
                    ],
                    &shape,
                    thread_group(u64::from(chunk)),
                );
            }
            finish(&command)?;
            self.trace_values("output projection", W::Logits, output_rows.min(4));
            if self.read::<u32>(W::Invalid, 1)[0] != 0 {
                return Err("Qwen produced nonfinite logits or loss".into());
            }
            if let Some(output_host) = host_logits.as_deref_mut() {
                let begin = (start - first_local) as usize * self.vocab();
                output_host[begin..begin + output_rows]
                    .copy_from_slice(&self.read::<f32>(W::Logits, output_rows));
            }
        }
        Ok(())
    }

    fn mean_loss(&self, rows: u32, scored: u32) -> Result<f32> {
        let command = self.runtime.queue.new_command_buffer();
        command.set_label("Qwen loss reduction");
        let shape = FlameShape {
            hidden: scored,
            ..self.flame_shape(rows)
        };
        self.encode(
            &command,
            "flame_mean",
            &[(self.buffer(W::Losses), 0), (self.buffer(W::Invalid), 0)],
            &shape,
            thread_group(1),
        );
        finish(&command)?;
        let value = self.read::<f32>(W::Losses, 1)[0];
        if self.read::<u32>(W::Invalid, 1)[0] != 0 || !value.is_finite() || value < 0.0 {
            return Err("Qwen returned an invalid loss".into());
        }
        Ok(value)
    }
}

fn write_buffer<T: Copy>(buffer: &Buffer, values: &[T]) {
    assert!(size_of::<T>() * values.len() <= buffer.length() as usize);
    unsafe {
        std::ptr::copy_nonoverlapping(
            values.as_ptr().cast::<u8>(),
            buffer.contents().cast(),
            size_of::<T>() * values.len(),
        );
    }
}

fn argmax_logits(logits: &[f32]) -> Result<u32> {
    let mut best = None;
    for (index, &value) in logits.iter().enumerate() {
        if !value.is_finite() {
            return Err("Qwen batched output produced nonfinite logits".into());
        }
        if best.is_none_or(|(best_index, best_value)| {
            value > best_value || (value == best_value && index < best_index)
        }) {
            best = Some((index, value));
        }
    }
    best.map(|(index, _)| index as u32)
        .ok_or("Qwen batched output is empty".into())
}

fn sample_logits(
    logits: &[f32],
    temperature: f32,
    top_p: f32,
    top_k: usize,
    rng: &mut u64,
) -> Result<u32> {
    let peak = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !peak.is_finite() {
        return Err("Qwen sampling received nonfinite logits".into());
    }
    let mut probabilities: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(index, &logit)| (index, ((logit - peak) / temperature).exp()))
        .collect();
    probabilities.sort_unstable_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    let limit = if top_k == 0 {
        probabilities.len()
    } else {
        top_k.min(probabilities.len())
    };
    let mut selected = Vec::with_capacity(limit);
    let mut total = 0.0f32;
    let full_total: f32 = probabilities.iter().map(|&(_, value)| value).sum();
    if !full_total.is_finite() || full_total <= 0.0 {
        return Err("Qwen sampling probability normalization failed".into());
    }
    for &(index, value) in probabilities.iter().take(limit) {
        selected.push((index, value));
        total += value;
        if total / full_total >= top_p {
            break;
        }
    }
    if selected.is_empty() || !total.is_finite() || total <= 0.0 {
        return Err("Qwen sampling selected no probability mass".into());
    }
    *rng = (*rng)
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let unit = ((*rng >> 11) as f64 / (1u64 << 53) as f64) as f32;
    let target = unit * total;
    let mut cumulative = 0.0f32;
    for (index, value) in selected {
        cumulative += value;
        if target < cumulative {
            return u32::try_from(index).map_err(|_| "Qwen sampled token ID overflow".into());
        }
    }
    u32::try_from(probabilities[0].0).map_err(|_| "Qwen sampled token ID overflow".into())
}

fn finish(command: &CommandBufferRef) -> Result<()> {
    command.commit();
    command.wait_until_completed();
    match command.status() {
        MTLCommandBufferStatus::Completed => Ok(()),
        status => Err(format!(
            "Qwen Metal command '{}' failed with status {status:?}",
            command.label()
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_limit() {
        assert!(!cache_path(REF_ROWS as usize));
        assert!(cache_path(REF_ROWS as usize + 1));
        assert!(MPS_CHUNK > REF_ROWS);
    }

    #[test]
    fn mask_alignment() {
        autoreleasepool(|| {
            let config = QwenConfig {
                layers: 1,
                hidden: 32,
                intermediate: 64,
                heads: 2,
                kv_heads: 1,
                vocab: 64,
                eos_token_id: 63,
                context: 8,
                epsilon: 1e-6,
                rope_theta: 10000.0,
            };
            let mut evaluator = QwenEvaluator::with_config(config, 8).unwrap();
            let values: Vec<u16> = (0..evaluator.layout.len)
                .map(|index| ((((index * 7 % 29) as f32 - 14.0) / 128.0).to_bits() >> 16) as u16)
                .collect();
            let weights = evaluator.runtime.buffer_with(&values);
            let tokens = vec![3, 5, 7, 11];
            let logits = evaluator.logits(&weights, &tokens).unwrap();
            let token_loss = |index: usize| {
                let row =
                    &logits[index * config.vocab as usize..(index + 1) * config.vocab as usize];
                let peak = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let total = row.iter().map(|value| (*value - peak).exp()).sum::<f32>();
                total.ln() + peak - row[tokens[index + 1] as usize]
            };
            for mask in [
                vec![false, true, true, true],
                vec![false, false, true, true],
            ] {
                let first = mask.iter().position(|&value| value).unwrap();
                let expected = ((first - 1)..tokens.len() - 1)
                    .map(&token_loss)
                    .sum::<f32>()
                    / mask.iter().filter(|&&value| value).count() as f32;
                let actual = evaluator
                    .losses(&weights, std::slice::from_ref(&tokens), &[mask])
                    .unwrap()[0];
                assert!((actual - expected).abs() <= 1e-4 * expected.abs().max(1.0));
            }
        });
    }

    #[test]
    fn loss_parity() {
        autoreleasepool(|| {
            let config = QwenConfig {
                layers: 1,
                hidden: 32,
                intermediate: 64,
                heads: 2,
                kv_heads: 1,
                vocab: 64,
                eos_token_id: 63,
                context: 260,
                epsilon: 1e-6,
                rope_theta: 10000.0,
            };
            let mut evaluator = QwenEvaluator::with_backend(config, 260, MpsMode::Off).unwrap();
            let values: Vec<u16> = (0..evaluator.layout.len)
                .map(|index| ((((index * 7 % 29) as f32 - 14.0) / 128.0).to_bits() >> 16) as u16)
                .collect();
            let weights = evaluator.runtime.buffer_with(&values);
            let tokens = (0..128).map(|index| index % 63).collect::<Vec<_>>();
            let mask = (0..tokens.len())
                .map(|index| index >= 9)
                .collect::<Vec<_>>();
            let reference = evaluator
                .losses(
                    &weights,
                    std::slice::from_ref(&tokens),
                    std::slice::from_ref(&mask),
                )
                .unwrap()[0];
            evaluator.write(W::Tokens, &tokens);
            evaluator.write(
                W::Masks,
                &mask
                    .iter()
                    .map(|&value| u8::from(value))
                    .collect::<Vec<_>>(),
            );
            let mut cache = evaluator.new_state().unwrap();
            let (cached, _, _, _) = evaluator
                .cached_loss(
                    &weights,
                    tokens.len() as u32,
                    9,
                    mask.iter().filter(|&&value| value).count() as u32,
                    &mut cache,
                )
                .unwrap();
            assert!(
                (cached - reference).abs() <= 1e-3 * reference.abs().max(1.0),
                "reference={reference}, cached={cached}"
            );

            let long_tokens = (0..260).map(|index| index % 63).collect::<Vec<_>>();
            let long_mask = (0..260).map(|index| index > 0).collect::<Vec<_>>();
            let loss = evaluator
                .losses(&weights, &[long_tokens], &[long_mask])
                .unwrap()[0];
            let profile = evaluator.loss_profile().unwrap();
            assert!(loss.is_finite());
            assert_eq!(profile.tokens, 260);
            assert_eq!(profile.cached_tokens, 260);
            assert!(profile.kv_cache_bytes > 0);
        });
    }

    #[test]
    fn cache_parity() {
        let Some(path) = std::env::var_os("ENNX_QWEN_CHECKPOINT") else {
            return;
        };
        autoreleasepool(|| {
            let mut evaluator = QwenEvaluator::new(128).unwrap();
            let weights = evaluator.load_weights(Path::new(&path)).unwrap();
            let tokens = (0..128).map(|index| 1000 + index % 97).collect::<Vec<_>>();
            let mask = (0..tokens.len())
                .map(|index| index >= 64)
                .collect::<Vec<_>>();
            let reference = evaluator
                .losses(
                    &weights,
                    std::slice::from_ref(&tokens),
                    std::slice::from_ref(&mask),
                )
                .unwrap()[0];
            evaluator.write(W::Tokens, &tokens);
            evaluator.write(
                W::Masks,
                &mask
                    .iter()
                    .map(|&value| u8::from(value))
                    .collect::<Vec<_>>(),
            );
            let mut cache = evaluator.new_state().unwrap();
            let (cached, _, _, _) = evaluator
                .cached_loss(&weights, 128, 64, 64, &mut cache)
                .unwrap();
            assert!(
                (cached - reference).abs() <= 1e-3 * reference.abs().max(1.0),
                "reference={reference}, cached={cached}"
            );
        });
    }

    #[test]
    fn backend_parity() {
        let Some(path) = std::env::var_os("ENNX_QWEN_CHECKPOINT") else {
            return;
        };
        autoreleasepool(|| {
            let loss = |mode| {
                let mut evaluator =
                    QwenEvaluator::with_backend(QwenConfig::default(), 128, mode).unwrap();
                let weights = evaluator.load_weights(Path::new(&path)).unwrap();
                let tokens = (0..128).map(|index| 1000 + index % 97).collect::<Vec<_>>();
                let mask = (0..tokens.len())
                    .map(|index| index >= 64)
                    .collect::<Vec<_>>();
                evaluator.losses(&weights, &[tokens], &[mask]).unwrap()[0]
            };
            let fp32 = loss(MpsMode::F32);
            let fp16 = loss(MpsMode::F16);
            let delta = (fp16 - fp32).abs();
            assert!(
                delta <= 2e-4,
                "FP32 loss={fp32}, FP16 loss={fp16}, delta={delta}"
            );
            println!("Qwen backend FP32 loss={fp32:.6} FP16 loss={fp16:.6} delta={delta:.6}");
        });
    }

    #[test]
    fn rank_parity() {
        if std::env::var_os("ENNX_QWEN_RANK_PARITY").is_none() {
            return;
        }
        let path = std::env::var_os("ENNX_QWEN_CHECKPOINT")
            .expect("ENNX_QWEN_RANK_PARITY requires ENNX_QWEN_CHECKPOINT");
        autoreleasepool(|| {
            let mut loader =
                QwenEvaluator::with_backend(QwenConfig::default(), 128, MpsMode::Off).unwrap();
            let weights = loader.load_weights(Path::new(&path)).unwrap();
            let blocks = loader
                .blocks()
                .into_iter()
                .map(|(key, offset, len, scale, weight)| {
                    crate::bf16_metal::ParamBlock::new(key, offset, len, scale, weight).unwrap()
                })
                .collect();
            let base = unsafe {
                std::slice::from_raw_parts(weights.contents().cast::<u16>(), loader.weights_len())
            };
            let mut search = crate::bf16_metal::SearchState::new(
                base,
                -5.0,
                0.0,
                blocks,
                1,
                1,
                crate::TRLengthConfig::new(0.01, 0.0001, 0.08),
            )
            .unwrap();
            drop(weights);
            drop(loader);
            search.correlate(0x25f4_91ad).unwrap();

            let mut fp32 =
                QwenEvaluator::with_backend(QwenConfig::default(), 128, MpsMode::F32).unwrap();
            let mut fp16 =
                QwenEvaluator::with_backend(QwenConfig::default(), 128, MpsMode::F16).unwrap();
            let tokens = (0..128).map(|index| 1000 + index % 97).collect::<Vec<_>>();
            let mask = (0..tokens.len())
                .map(|index| index >= 64)
                .collect::<Vec<_>>();
            let order = |losses: &[f32]| {
                let mut indices = (0..losses.len()).collect::<Vec<_>>();
                indices.sort_by(|&left, &right| {
                    losses[left]
                        .total_cmp(&losses[right])
                        .then_with(|| left.cmp(&right))
                });
                indices
            };
            for root in [0x7b6d_0a13, 0xac91_7e2d, 0xe45b_38c7] {
                let mut losses32 = Vec::with_capacity(4);
                let mut losses16 = Vec::with_capacity(4);
                for index in 0..4 {
                    let candidate = search.test_candidate(root, index).unwrap();
                    losses32.push(
                        fp32.losses(
                            &candidate,
                            std::slice::from_ref(&tokens),
                            std::slice::from_ref(&mask),
                        )
                        .unwrap()[0],
                    );
                    losses16.push(
                        fp16.losses(
                            &candidate,
                            std::slice::from_ref(&tokens),
                            std::slice::from_ref(&mask),
                        )
                        .unwrap()[0],
                    );
                }
                let max_delta = losses32
                    .iter()
                    .zip(&losses16)
                    .map(|(left, right)| (left - right).abs())
                    .fold(0.0f32, f32::max);
                println!(
                    "Qwen rank root={root:#x} FP32={losses32:?} FP16={losses16:?} max_delta={max_delta:.6}"
                );
                assert!(max_delta <= 5e-4, "backend loss drift exceeds rank gate");
                assert_eq!(order(&losses16), order(&losses32));
            }

            let ask = crate::trials::Ask {
                neighbors: 1,
                ..crate::trials::Ask::default()
            };
            let phases = search.test_profile(0x7b6d_0a13, ask).unwrap();
            println!(
                "Qwen proposal pool_ms={:.3} select_ms={:.3} row_ms={:.3}",
                phases[0], phases[1], phases[2]
            );
            search.set_profiling(true);
            let round = search.ask_round(1, 4, 0x7b6d_0a13, ask).unwrap();
            let (radii, cosines, reference_cosines) = search.pool_geometry(&round).unwrap();
            let profile = search.last_profile().unwrap();
            assert!(
                radii
                    .iter()
                    .all(|radius| radius.is_finite() && *radius > 0.0)
            );
            assert!(
                cosines
                    .iter()
                    .all(|entry| entry.2.is_some_and(f32::is_finite))
            );
            assert!(
                reference_cosines
                    .iter()
                    .all(|cosine| cosine.is_some_and(f32::is_finite))
            );
            println!(
                "Qwen pool radii={radii:?} cosines={cosines:?} reference_cosines={reference_cosines:?} proposal_ms={:.3}",
                profile.total_ms
            );
        });
    }

    #[test]
    fn checkpoint_4k() {
        if std::env::var_os("ENNX_QWEN_4K").is_none() {
            return;
        }
        let path = std::env::var_os("ENNX_QWEN_CHECKPOINT")
            .expect("ENNX_QWEN_4K requires ENNX_QWEN_CHECKPOINT");
        autoreleasepool(|| {
            let mut evaluator = QwenEvaluator::new(4096).unwrap();
            let weights = evaluator.load_weights(Path::new(&path)).unwrap();
            let tokens = (0..4096).map(|index| 1000 + index % 97).collect::<Vec<_>>();
            let mask = (0..tokens.len())
                .map(|index| index >= 4032)
                .collect::<Vec<_>>();
            let loss = evaluator.losses(&weights, &[tokens], &[mask]).unwrap()[0];
            let profile = evaluator.loss_profile().unwrap();
            assert!(loss.is_finite());
            assert_eq!(profile.tokens, 4096);
            assert_eq!(profile.scored_tokens, 64);
            assert_eq!(profile.cached_tokens, 4096);
            assert!(profile.kv_cache_bytes > 0);
            println!("Qwen 4K loss={loss:.6} profile={profile:?}");
        });
    }

    #[test]
    fn decode_parity() {
        autoreleasepool(|| {
            let config = QwenConfig {
                layers: 1,
                hidden: 32,
                intermediate: 64,
                heads: 2,
                kv_heads: 1,
                vocab: 64,
                eos_token_id: 63,
                context: 8,
                epsilon: 1e-6,
                rope_theta: 10000.0,
            };
            let evaluator = QwenEvaluator::with_config(config, 8).unwrap();
            let weights: Vec<u16> = (0..evaluator.layout.len)
                .map(|i| ((((i * 7 % 29) as f32 - 14.0) / 128.0).to_bits() >> 16) as u16)
                .collect();
            let weights = evaluator.runtime.buffer_with(&weights);
            let mut ordinary = evaluator.new_state().unwrap();
            let mut profiled = evaluator.new_state().unwrap();
            let mut times = BTreeMap::new();
            for token in [3u32, 5, 7] {
                evaluator.write(W::Tokens, &[token]);
                let expected = evaluator.decode_token(&weights, &mut ordinary).unwrap();
                let logits = evaluator.read::<f32>(W::Logits, config.vocab as usize);
                evaluator.write(W::Tokens, &[token]);
                let actual = evaluator
                    .decode_profile(&weights, &mut profiled, &mut times)
                    .unwrap();
                assert_eq!(actual, expected);
                assert_eq!(profiled.position, ordinary.position);
                for (a, b) in evaluator
                    .read::<f32>(W::Logits, config.vocab as usize)
                    .iter()
                    .zip(&logits)
                {
                    assert!((a - b).abs() <= 1e-5 + 1e-5 * b.abs());
                }
            }
            assert!(times.contains_key("qwen_embedding"));
            assert!(times.contains_key("qwen_lm_head_argmax"));
            assert!(times.values().all(|time| time.is_finite() && *time >= 0.0));
        });
    }

    #[test]
    fn vocab_row() {
        autoreleasepool(|| {
            if metal::Device::system_default().is_none() {
                return;
            }
            let runtime = Runtime::shared().expect("Metal device required");
            let pipeline = runtime
                .precise(source_for("flame_linear"), "Qwen test", "flame_linear")
                .expect("linear pipeline");
            let input = runtime.buffer_with(&[99.0f32, 1.0, 2.0]);
            let vocab = 151_936usize;
            let mut weights = Vec::with_capacity(vocab * 2);
            for _ in 0..vocab {
                weights.extend([0x3f80u16, 0x4000u16]);
            }
            let weights = runtime.buffer_with(&weights);
            let output = runtime.buffer::<f32>(vocab);
            let params = Matmul {
                m: 1,
                n: vocab as u32,
                k: 2,
                transpose_b: 1,
                ..Matmul::default()
            };
            let command = runtime.queue.new_command_buffer();
            command.set_label("Qwen linear test");
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(&input), size_of::<f32>() as u64);
            encoder.set_buffer(1, Some(&weights), 0);
            encoder.set_buffer(2, Some(&output), 0);
            encoder.set_bytes(
                3,
                size_of::<Matmul>() as u64,
                (&params as *const Matmul).cast(),
            );
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: (vocab as u64).div_ceil(32),
                    height: 1,
                    depth: 1,
                },
                thread_group(256),
            );
            encoder.end_encoding();
            finish(&command).expect("linear command");
            let values =
                unsafe { std::slice::from_raw_parts(output.contents().cast::<f32>(), vocab) };
            assert!(values.iter().all(|value| *value == 5.0));
        });
    }

    #[test]
    fn gemv_shapes() {
        autoreleasepool(|| {
            if metal::Device::system_default().is_none() {
                return;
            }
            let runtime = Runtime::shared().expect("Metal device required");
            let pipeline = runtime
                .precise(source_for("qwen_gemv"), "Qwen test", "qwen_gemv")
                .expect("Qwen GEMV pipeline");
            let rows_pipeline = runtime
                .precise(source_for("qwen_gemv_rows"), "Qwen test", "qwen_gemv_rows")
                .expect("Qwen row GEMV pipeline");
            let rows = 2usize;
            let columns = 8960usize;
            let interior = 1536usize;
            let input: Vec<f32> = (0..rows * interior)
                .map(|index| 1.0 + (index % 7) as f32 * 0.125)
                .collect();
            let weights: Vec<u16> = (0..columns * interior)
                .map(|index| {
                    let value = 0.5 + (index % 5) as f32 * 0.25;
                    (value.to_bits() >> 16) as u16
                })
                .collect();
            let input_buffer = runtime.buffer_with(&input);
            let weight_buffer = runtime.buffer_with(&weights);
            let output_buffer = runtime.buffer::<f32>(rows * columns);
            let rows_output_buffer = runtime.buffer::<f32>(rows * columns);
            let params = Matmul {
                m: rows as u32,
                n: columns as u32,
                k: interior as u32,
                transpose_b: 1,
                ..Matmul::default()
            };
            let command = runtime.queue.new_command_buffer();
            command.set_label("Qwen decode GEMV test");
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(&input_buffer), 0);
            encoder.set_buffer(1, Some(&weight_buffer), 0);
            encoder.set_buffer(2, Some(&output_buffer), 0);
            encoder.set_bytes(
                3,
                size_of::<Matmul>() as u64,
                (&params as *const Matmul).cast(),
            );
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: (columns as u64).div_ceil(32),
                    height: (rows as u64).div_ceil(32),
                    depth: 1,
                },
                thread_group(256),
            );
            encoder.end_encoding();
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&rows_pipeline);
            encoder.set_buffer(0, Some(&input_buffer), 0);
            encoder.set_buffer(1, Some(&weight_buffer), 0);
            encoder.set_buffer(2, Some(&rows_output_buffer), 0);
            encoder.set_bytes(
                3,
                size_of::<Matmul>() as u64,
                (&params as *const Matmul).cast(),
            );
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: (columns as u64).div_ceil(32),
                    height: 1,
                    depth: 1,
                },
                thread_group(32),
            );
            encoder.end_encoding();
            finish(&command).expect("Qwen GEMV command");
            let output = unsafe {
                std::slice::from_raw_parts(output_buffer.contents().cast::<f32>(), rows * columns)
            };
            let rows_output = unsafe {
                std::slice::from_raw_parts(
                    rows_output_buffer.contents().cast::<f32>(),
                    rows * columns,
                )
            };
            for row in 0..rows {
                for column in 0..columns {
                    let expected = (0..interior)
                        .map(|index| {
                            input[row * interior + index]
                                * f32::from_bits(
                                    u32::from(weights[column * interior + index]) << 16,
                                )
                        })
                        .sum::<f32>();
                    assert!((output[row * columns + column] - expected).abs() < 1e-3);
                    assert!((rows_output[row * columns + column] - expected).abs() < 1e-3);
                }
            }
        });
    }

    #[test]
    fn qkv_parity() {
        autoreleasepool(|| {
            if metal::Device::system_default().is_none() {
                return;
            }
            let runtime = Runtime::shared().expect("Metal device required");
            let pipeline = runtime
                .precise(source_for("qwen_qkv"), "Qwen test", "qwen_qkv")
                .expect("Qwen fused QKV pipeline");
            let rows = 3usize;
            let hidden = 8usize;
            let kv_width = 4usize;
            let input: Vec<f32> = (0..rows * hidden)
                .map(|index| ((index % 9) as f32 - 4.0) * 0.25)
                .collect();
            let make_weights = |width: usize| {
                (0..width * hidden)
                    .map(|index| ((((index % 7) as f32 - 3.0) * 0.125).to_bits() >> 16) as u16)
                    .collect::<Vec<u16>>()
            };
            let q_weights = make_weights(hidden);
            let k_weights = make_weights(kv_width);
            let v_weights = make_weights(kv_width);
            let q_bias: Vec<u16> = (0..hidden)
                .map(|index| ((index as f32 * 0.125).to_bits() >> 16) as u16)
                .collect();
            let k_bias: Vec<u16> = (0..kv_width)
                .map(|index| ((index as f32 * -0.25).to_bits() >> 16) as u16)
                .collect();
            let v_bias: Vec<u16> = (0..kv_width)
                .map(|index| ((index as f32 * 0.5).to_bits() >> 16) as u16)
                .collect();
            let input_buffer = runtime.buffer_with(&input);
            let q_weight_buffer = runtime.buffer_with(&q_weights);
            let k_weight_buffer = runtime.buffer_with(&k_weights);
            let v_weight_buffer = runtime.buffer_with(&v_weights);
            let q_bias_buffer = runtime.buffer_with(&q_bias);
            let k_bias_buffer = runtime.buffer_with(&k_bias);
            let v_bias_buffer = runtime.buffer_with(&v_bias);
            let q_output_buffer = runtime.buffer::<f32>(rows * hidden);
            let k_output_buffer = runtime.buffer::<f32>(rows * kv_width);
            let v_output_buffer = runtime.buffer::<f32>(rows * kv_width);
            let params = QwenQkvShape {
                rows: rows as u32,
                hidden: hidden as u32,
                kv_width: kv_width as u32,
            };
            let command = runtime.queue.new_command_buffer();
            command.set_label("Qwen fused QKV test");
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline);
            for (index, buffer) in [
                &input_buffer,
                &q_weight_buffer,
                &k_weight_buffer,
                &v_weight_buffer,
                &q_bias_buffer,
                &k_bias_buffer,
                &v_bias_buffer,
                &q_output_buffer,
                &k_output_buffer,
                &v_output_buffer,
            ]
            .iter()
            .enumerate()
            {
                encoder.set_buffer(index as u64, Some(*buffer), 0);
            }
            encoder.set_bytes(
                10,
                size_of::<QwenQkvShape>() as u64,
                (&params as *const QwenQkvShape).cast(),
            );
            encoder.dispatch_thread_groups(
                thread_group((hidden as u64 + 2 * kv_width as u64).div_ceil(32)),
                thread_group(32),
            );
            encoder.end_encoding();
            finish(&command).expect("Qwen fused QKV command");

            let output = |buffer: &Buffer, length: usize| unsafe {
                std::slice::from_raw_parts(buffer.contents().cast::<f32>(), length)
            };
            let q_output = output(&q_output_buffer, rows * hidden);
            let k_output = output(&k_output_buffer, rows * kv_width);
            let v_output = output(&v_output_buffer, rows * kv_width);
            for row in 0..rows {
                for column in 0..hidden {
                    let expected = (0..hidden)
                        .map(|index| {
                            input[row * hidden + index]
                                * f32::from_bits(
                                    u32::from(q_weights[column * hidden + index]) << 16,
                                )
                        })
                        .sum::<f32>()
                        + f32::from_bits(u32::from(q_bias[column]) << 16);
                    assert!((q_output[row * hidden + column] - expected).abs() < 1e-5);
                }
                for column in 0..kv_width {
                    let k_expected = (0..hidden)
                        .map(|index| {
                            input[row * hidden + index]
                                * f32::from_bits(
                                    u32::from(k_weights[column * hidden + index]) << 16,
                                )
                        })
                        .sum::<f32>()
                        + f32::from_bits(u32::from(k_bias[column]) << 16);
                    let v_expected = (0..hidden)
                        .map(|index| {
                            input[row * hidden + index]
                                * f32::from_bits(
                                    u32::from(v_weights[column * hidden + index]) << 16,
                                )
                        })
                        .sum::<f32>()
                        + f32::from_bits(u32::from(v_bias[column]) << 16);
                    assert!((k_output[row * kv_width + column] - k_expected).abs() < 1e-5);
                    assert!((v_output[row * kv_width + column] - v_expected).abs() < 1e-5);
                }
            }
        });
    }

    #[test]
    fn silu_parity() {
        autoreleasepool(|| {
            if metal::Device::system_default().is_none() {
                return;
            }
            let runtime = Runtime::shared().expect("Metal device required");
            let pipeline = runtime
                .precise(source_for("qwen_mlp_rows"), "Qwen test", "qwen_mlp_rows")
                .expect("Qwen fused MLP pipeline");
            let rows = 3usize;
            let hidden = 8usize;
            let intermediate = 64usize;
            let input: Vec<f32> = (0..rows * hidden)
                .map(|index| ((index % 9) as f32 - 4.0) * 0.25)
                .collect();
            let make_weights = |offset: f32| {
                (0..intermediate * hidden)
                    .map(|index| {
                        ((((index % 7) as f32 - 3.0) * 0.125 + offset).to_bits() >> 16) as u16
                    })
                    .collect::<Vec<u16>>()
            };
            let gate_weights = make_weights(0.0);
            let up_weights = make_weights(0.25);
            let input_buffer = runtime.buffer_with(&input);
            let gate_buffer = runtime.buffer_with(&gate_weights);
            let up_buffer = runtime.buffer_with(&up_weights);
            let output_buffer = runtime.buffer::<f32>(rows * intermediate);
            let params = QwenMlpShape {
                rows: rows as u32,
                hidden: hidden as u32,
                intermediate: intermediate as u32,
            };
            let command = runtime.queue.new_command_buffer();
            command.set_label("Qwen fused MLP test");
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline);
            for (index, buffer) in [&input_buffer, &gate_buffer, &up_buffer, &output_buffer]
                .iter()
                .enumerate()
            {
                encoder.set_buffer(index as u64, Some(*buffer), 0);
            }
            encoder.set_bytes(
                4,
                size_of::<QwenMlpShape>() as u64,
                (&params as *const QwenMlpShape).cast(),
            );
            encoder.dispatch_thread_groups(
                thread_group((intermediate as u64).div_ceil(32)),
                thread_group(32),
            );
            encoder.end_encoding();
            finish(&command).expect("Qwen fused MLP command");

            let output = unsafe {
                std::slice::from_raw_parts(
                    output_buffer.contents().cast::<f32>(),
                    rows * intermediate,
                )
            };
            for row in 0..rows {
                for column in 0..intermediate {
                    let gate = (0..hidden)
                        .map(|index| {
                            input[row * hidden + index]
                                * f32::from_bits(
                                    u32::from(gate_weights[column * hidden + index]) << 16,
                                )
                        })
                        .sum::<f32>();
                    let up = (0..hidden)
                        .map(|index| {
                            input[row * hidden + index]
                                * f32::from_bits(
                                    u32::from(up_weights[column * hidden + index]) << 16,
                                )
                        })
                        .sum::<f32>();
                    let expected = gate * (1.0 / (1.0 + (-gate).exp())) * up;
                    assert!((output[row * intermediate + column] - expected).abs() < 1e-4);
                }
            }
        });
    }

    #[test]
    fn simd_parity() {
        autoreleasepool(|| {
            if metal::Device::system_default().is_none() {
                return;
            }
            let runtime = Runtime::shared().expect("Metal device required");
            let pipeline = runtime
                .precise(source_for("qwen_simd_gemm"), "Qwen test", "qwen_simd_gemm")
                .expect("SIMD-group linear pipeline");
            let rows = 64usize;
            let columns = 32usize;
            let interior = 8usize;
            let input: Vec<f32> = (0..rows * interior)
                .map(|index| ((index % 7) as f32 - 3.0) * 0.25)
                .collect();
            let weights: Vec<u16> = (0..columns * interior)
                .map(|index| {
                    let value = ((index % 5) as f32 - 2.0) * 0.5;
                    (value.to_bits() >> 16) as u16
                })
                .collect();
            let input_buffer = runtime.buffer_with(&input);
            let weight_buffer = runtime.buffer_with(&weights);
            let output_buffer = runtime.buffer::<f32>(rows * columns);
            let params = Matmul {
                m: rows as u32,
                n: columns as u32,
                k: interior as u32,
                transpose_b: 1,
                ..Matmul::default()
            };
            let command = runtime.queue.new_command_buffer();
            command.set_label("Qwen SIMD-group linear test");
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(&input_buffer), 0);
            encoder.set_buffer(1, Some(&weight_buffer), 0);
            encoder.set_buffer(2, Some(&output_buffer), 0);
            encoder.set_bytes(
                3,
                size_of::<Matmul>() as u64,
                (&params as *const Matmul).cast(),
            );
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
                thread_group(128),
            );
            encoder.end_encoding();
            finish(&command).expect("SIMD-group linear command");
            let output = unsafe {
                std::slice::from_raw_parts(output_buffer.contents().cast::<f32>(), rows * columns)
            };
            for row in 0..rows {
                for column in 0..columns {
                    let expected = (0..interior)
                        .map(|index| {
                            input[row * interior + index]
                                * f32::from_bits(
                                    u32::from(weights[column * interior + index]) << 16,
                                )
                        })
                        .sum::<f32>();
                    let actual = output[row * columns + column];
                    assert!(
                        (actual - expected).abs() < 1e-3,
                        "row={row} column={column} actual={actual} expected={expected}"
                    );
                }
            }
        });
    }

    #[test]
    fn mps_parity() {
        autoreleasepool(|| {
            if metal::Device::system_default().is_none() {
                return;
            }
            let config = QwenConfig {
                layers: 1,
                hidden: 32,
                intermediate: 64,
                heads: 2,
                kv_heads: 1,
                vocab: 64,
                eos_token_id: 63,
                context: 64,
                epsilon: 1e-6,
                rope_theta: 10000.0,
            };
            let rows = 64usize;
            let inside = config.hidden as usize;
            let outside = config.intermediate as usize;
            let input: Vec<f32> = (0..rows * inside)
                .map(|index| ((index % 13) as f32 - 6.0) * 0.125)
                .collect();
            let weights: Vec<u16> = (0..outside * inside)
                .map(|index| {
                    let value = ((index % 11) as f32 - 5.0) * 0.0625;
                    (value.to_bits() >> 16) as u16
                })
                .collect();
            for mode in [MpsMode::F32, MpsMode::F16] {
                let evaluator = QwenEvaluator::with_backend(config, 64, mode).unwrap();
                evaluator.write(W::Norm, &input);
                let weight_buffer = evaluator.runtime.buffer_with(&weights);
                let command = evaluator.runtime.queue.new_command_buffer();
                evaluator
                    .prefill_linear(
                        &command,
                        W::Norm,
                        &weight_buffer,
                        0,
                        W::Gate,
                        rows as u32,
                        inside as u32,
                        outside as u32,
                    )
                    .unwrap();
                finish(&command).unwrap();
                let output = evaluator.read::<f32>(W::Gate, rows * outside);
                let tolerance = if mode == MpsMode::F16 { 2e-3 } else { 1e-4 };
                for row in 0..rows {
                    for column in 0..outside {
                        let expected = (0..inside)
                            .map(|index| {
                                input[row * inside + index]
                                    * f32::from_bits(
                                        u32::from(weights[column * inside + index]) << 16,
                                    )
                            })
                            .sum::<f32>();
                        let actual = output[row * outside + column];
                        assert!(
                            (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
                            "row={row} column={column} actual={actual} expected={expected}"
                        );
                    }
                }
            }
        });
    }

    #[test]
    fn mlp_parity() {
        autoreleasepool(|| {
            if metal::Device::system_default().is_none() {
                return;
            }
            let config = QwenConfig {
                layers: 1,
                hidden: 32,
                intermediate: 64,
                heads: 2,
                kv_heads: 1,
                vocab: 64,
                eos_token_id: 63,
                context: 32,
                epsilon: 1e-6,
                rope_theta: 10000.0,
            };
            let values = (0..Layout::new(config).unwrap().len)
                .map(|index| {
                    let value = ((index * 7 % 29) as f32 - 14.0) / 128.0;
                    (value.to_bits() >> 16) as u16
                })
                .collect::<Vec<_>>();
            let tokens = (0..32).map(|index| index % 63).collect::<Vec<_>>();
            let mask = (0..32).map(|index| index >= 8).collect::<Vec<_>>();
            let loss = |mode| {
                let mut evaluator = QwenEvaluator::with_backend(config, 32, mode).unwrap();
                let weights = evaluator.runtime.buffer_with(&values);
                evaluator
                    .losses(
                        &weights,
                        std::slice::from_ref(&tokens),
                        std::slice::from_ref(&mask),
                    )
                    .unwrap()[0]
            };
            let fp32 = loss(MpsMode::F32);
            let fp16 = loss(MpsMode::F16);
            let delta = (fp16 - fp32).abs();
            assert!(
                delta <= 2e-3 * fp32.abs().max(1.0),
                "FP32 loss={fp32} FP16 half-resident loss={fp16} delta={delta}"
            );
        });
    }

    #[test]
    fn argmax_tiebreak() {
        autoreleasepool(|| {
            if metal::Device::system_default().is_none() {
                return;
            }
            let runtime = Runtime::shared().expect("Metal device required");
            let pipeline = runtime
                .precise(source_for("qwen_argmax"), "Qwen test", "qwen_argmax")
                .expect("argmax pipeline");
            let values = runtime.buffer_with(&[1.0f32, 5.0, 5.0, 4.0, -2.0]);
            let result = runtime.buffer::<u32>(1);
            let invalid = runtime.buffer_with(&[0u32]);
            let params = ArgmaxShape { width: 5 };
            let command = runtime.queue.new_command_buffer();
            command.set_label("Qwen argmax test");
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(&values), 0);
            encoder.set_buffer(1, Some(&result), 0);
            encoder.set_buffer(2, Some(&invalid), 0);
            encoder.set_bytes(
                3,
                size_of::<ArgmaxShape>() as u64,
                (&params as *const ArgmaxShape).cast(),
            );
            encoder.dispatch_thread_groups(thread_group(1), thread_group(256));
            encoder.end_encoding();
            finish(&command).expect("argmax command");
            assert_eq!(unsafe { result.contents().cast::<u32>().read() }, 1);
            assert_eq!(unsafe { invalid.contents().cast::<u32>().read() }, 0);
        });
    }

    #[test]
    fn argmax_rows() {
        autoreleasepool(|| {
            if metal::Device::system_default().is_none() {
                return;
            }
            let runtime = Runtime::shared().expect("Metal device required");
            let pipeline = runtime
                .precise(source_for("qwen_argn"), "Qwen test", "qwen_argn")
                .expect("batched argmax pipeline");
            let values =
                runtime.buffer_with(&[1.0f32, 5.0, 5.0, 4.0, -2.0, 9.0, 8.0, 8.0, 2.0, 3.0]);
            let result = runtime.buffer::<u32>(2);
            let invalid = runtime.buffer_with(&[0u32]);
            let params = ArgmaxBatchShape { rows: 2, width: 5 };
            let command = runtime.queue.new_command_buffer();
            command.set_label("Qwen batched argmax test");
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(&values), 0);
            encoder.set_buffer(1, Some(&result), 0);
            encoder.set_buffer(2, Some(&invalid), 0);
            encoder.set_bytes(
                3,
                size_of::<ArgmaxBatchShape>() as u64,
                (&params as *const ArgmaxBatchShape).cast(),
            );
            encoder.dispatch_thread_groups(thread_group(2), thread_group(256));
            encoder.end_encoding();
            finish(&command).expect("batched argmax command");
            let output = unsafe { std::slice::from_raw_parts(result.contents().cast::<u32>(), 2) };
            assert_eq!(output, &[1, 0]);
            assert_eq!(unsafe { invalid.contents().cast::<u32>().read() }, 0);
        });
    }

    #[test]
    fn attn_parity() {
        autoreleasepool(|| {
            if metal::Device::system_default().is_none() {
                return;
            }
            let runtime = Runtime::shared().expect("Metal device required");
            let pipeline = runtime
                .precise(source_for("qwen_dattn"), "Qwen test", "qwen_dattn")
                .expect("decode attention pipeline");
            let query = runtime.buffer_with(&[1.0f32, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]);
            let key_cache = runtime.buffer_with(&[
                0x3f80u16, 0, 0, 0, 0, 0x3f80, 0, 0, 0, 0, 0x3f80, 0, 0, 0, 0, 0,
            ]);
            let value_cache = runtime.buffer_with(&[
                0x3f80u16, 0x4000, 0x4040, 0x4080, 0x40a0, 0x40c0, 0x40e0, 0x4100, 0x4110, 0x4120,
                0x4130, 0x4140,
            ]);
            let output = runtime.buffer::<f32>(8);
            let positions = runtime.buffer_with(&[2u32]);
            let params = DecodeAttentionShape {
                sequence: 3,
                capacity: 4,
                heads: 2,
                kv_heads: 1,
                head_dim: 4,
                batch: 1,
                cache_stride: 16,
                scale: 0.5,
            };
            let command = runtime.queue.new_command_buffer();
            command.set_label("Qwen decode attention test");
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(&query), 0);
            encoder.set_buffer(1, Some(&key_cache), 0);
            encoder.set_buffer(2, Some(&value_cache), 0);
            encoder.set_buffer(3, Some(&output), 0);
            encoder.set_buffer(4, Some(&positions), 0);
            encoder.set_bytes(
                5,
                size_of::<DecodeAttentionShape>() as u64,
                (&params as *const DecodeAttentionShape).cast(),
            );
            encoder.dispatch_thread_groups(thread_group(2), thread_group(32));
            encoder.end_encoding();
            finish(&command).expect("decode attention command");

            let keys = [
                [1.0f32, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ];
            let values = [
                [1.0f32, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
            ];
            let queries = [[1.0f32, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]];
            let mut expected = Vec::with_capacity(8);
            for query in queries {
                let scores: Vec<f32> = keys
                    .iter()
                    .map(|key| {
                        query
                            .iter()
                            .zip(key)
                            .map(|(&left, &right)| left * right)
                            .sum::<f32>()
                            * 0.5
                    })
                    .collect();
                let peak = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let weights: Vec<f32> = scores.iter().map(|&score| (score - peak).exp()).collect();
                let denominator = weights.iter().sum::<f32>();
                for d in 0..4 {
                    expected.push(
                        values
                            .iter()
                            .zip(&weights)
                            .map(|(value, &weight)| value[d] * weight)
                            .sum::<f32>()
                            / denominator,
                    );
                }
            }
            let actual = unsafe { std::slice::from_raw_parts(output.contents().cast::<f32>(), 8) };
            for (&actual, &expected) in actual.iter().zip(&expected) {
                assert!((actual - expected).abs() < 0.02, "{actual} != {expected}");
            }
        });
    }

    #[test]
    fn tile_parity() {
        autoreleasepool(|| {
            if metal::Device::system_default().is_none() {
                return;
            }
            let runtime = Runtime::shared().expect("Metal device required");
            let old = runtime
                .precise(source_for("qwen_arow"), "Qwen test", "qwen_arow")
                .expect("serial attention pipeline");
            let tiled = runtime
                .precise(source_for("qwen_atile"), "Qwen test", "qwen_atile")
                .expect("tiled attention pipeline");
            let tile16 = runtime
                .precise(source_for("qwen_attn16"), "Qwen test", "qwen_attn16")
                .expect("16-row attention pipeline");
            assert!(
                tile16.static_threadgroup_memory_length()
                    <= runtime.device.max_threadgroup_memory_length()
            );

            let rows = 19usize;
            let heads = 12usize;
            let kv_heads = 2usize;
            let dim = 128usize;
            let capacity = 32usize;
            let sequence = 26usize;
            let query = (0..heads * rows * dim)
                .map(|i| ((i * 17 % 31) as f32 - 15.0) / 64.0)
                .collect::<Vec<_>>();
            let make_cache = |stride: usize| {
                (0..kv_heads * capacity * dim)
                    .map(|i| {
                        let value = ((i * stride % 29) as f32 - 14.0) / 32.0;
                        (value.to_bits() >> 16) as u16
                    })
                    .collect::<Vec<_>>()
            };
            let query = runtime.buffer_with(&query);
            let keys = runtime.buffer_with(&make_cache(7));
            let values = runtime.buffer_with(&make_cache(11));
            let serial = runtime.buffer::<f32>(rows * heads * dim);
            let tile = runtime.buffer::<f32>(rows * heads * dim);
            let tile16_out = runtime.buffer::<f32>(rows * heads * dim);
            let params = PrefillAttentionShape {
                rows: rows as u32,
                sequence: sequence as u32,
                capacity: capacity as u32,
                heads: heads as u32,
                kv_heads: kv_heads as u32,
                head_dim: dim as u32,
                scale: 1.0 / (dim as f32).sqrt(),
            };
            let run = |pipeline: &ComputePipelineState,
                       output: &Buffer,
                       groups: MTLSize,
                       threads: u64| {
                let command = runtime.queue.new_command_buffer();
                command.set_label("Qwen attention parity");
                let encoder = command.new_compute_command_encoder();
                encoder.set_compute_pipeline_state(pipeline);
                for (index, buffer) in [&query, &keys, &values, output].iter().enumerate() {
                    encoder.set_buffer(index as u64, Some(*buffer), 0);
                }
                encoder.set_bytes(
                    4,
                    size_of::<PrefillAttentionShape>() as u64,
                    (&params as *const PrefillAttentionShape).cast(),
                );
                encoder.dispatch_thread_groups(groups, thread_group(threads));
                encoder.end_encoding();
                finish(&command).expect("attention parity command");
            };
            run(&old, &serial, thread_group((rows * heads) as u64), 32);
            run(
                &tiled,
                &tile,
                MTLSize {
                    width: heads as u64,
                    height: rows.div_ceil(8) as u64,
                    depth: 1,
                },
                128,
            );
            run(
                &tile16,
                &tile16_out,
                MTLSize {
                    width: heads as u64,
                    height: rows.div_ceil(16) as u64,
                    depth: 1,
                },
                256,
            );
            let serial = unsafe {
                std::slice::from_raw_parts(serial.contents().cast::<f32>(), rows * heads * dim)
            };
            let tile = unsafe {
                std::slice::from_raw_parts(tile.contents().cast::<f32>(), rows * heads * dim)
            };
            let tile16_out = unsafe {
                std::slice::from_raw_parts(tile16_out.contents().cast::<f32>(), rows * heads * dim)
            };
            for (index, &expected) in serial.iter().enumerate() {
                for (actual, name) in [(tile[index], "tile8"), (tile16_out[index], "tile16")] {
                    let error = (actual - expected).abs();
                    assert!(
                        error <= 5e-4 * expected.abs().max(1.0),
                        "index={index} serial={expected} {name}={actual} error={error}"
                    );
                }
            }
        });
    }

    #[test]
    fn multi_decode() {
        autoreleasepool(|| {
            if metal::Device::system_default().is_none() {
                return;
            }
            let runtime = Runtime::shared().expect("Metal device required");
            let pipeline = runtime
                .precise(source_for("qwen_dattn"), "Qwen test", "qwen_dattn")
                .expect("decode attention pipeline");
            let head_dim = 32usize;
            let capacity = 3usize;
            let cache_stride = capacity * head_dim;
            let mut query_values = vec![0.0f32; 2 * head_dim];
            query_values[0] = 1.0;
            query_values[head_dim + 1] = 1.0;
            let query = runtime.buffer_with(&query_values);
            let key_cache = runtime.buffer_with(&vec![0x3f80u16; 2 * cache_stride]);
            let value_cache = runtime.buffer_with(&vec![0x3f80u16; 2 * cache_stride]);
            let output = runtime.buffer::<f32>(2 * head_dim);
            let positions = runtime.buffer_with(&[2u32, 2]);
            let params = DecodeAttentionShape {
                sequence: 3,
                capacity: 3,
                heads: 1,
                kv_heads: 1,
                head_dim: head_dim as u32,
                batch: 2,
                cache_stride: cache_stride as u32,
                scale: 1.0 / (head_dim as f32).sqrt(),
            };
            let command = runtime.queue.new_command_buffer();
            command.set_label("Qwen batch attention test");
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(&query), 0);
            encoder.set_buffer(1, Some(&key_cache), 0);
            encoder.set_buffer(2, Some(&value_cache), 0);
            encoder.set_buffer(3, Some(&output), 0);
            encoder.set_buffer(4, Some(&positions), 0);
            encoder.set_bytes(
                5,
                size_of::<DecodeAttentionShape>() as u64,
                (&params as *const DecodeAttentionShape).cast(),
            );
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: 2,
                    height: 1,
                    depth: 1,
                },
                thread_group(32),
            );
            encoder.end_encoding();
            finish(&command).expect("decode batch attention command");
            let actual = unsafe {
                std::slice::from_raw_parts(output.contents().cast::<f32>(), 2 * head_dim).to_vec()
            };
            assert!(actual.iter().all(|value| value.is_finite()));
            assert_eq!(actual[..head_dim], actual[head_dim..]);
        });
    }

    #[test]
    fn gqa_layout() {
        autoreleasepool(|| {
            if metal::Device::system_default().is_none() {
                return;
            }
            let runtime = Runtime::shared().expect("Metal device required");
            let pipeline = runtime
                .precise(source_for("qwen_dattn"), "Qwen test", "qwen_dattn")
                .expect("decode attention pipeline");
            let heads = 12usize;
            let kv_heads = 2usize;
            let head_dim = 128usize;
            let batch = 2usize;
            let capacity = 2usize;
            let cache_stride = kv_heads * capacity * head_dim;
            let query = runtime.buffer_with(
                &(0..heads)
                    .flat_map(|head| {
                        (0..batch).flat_map(move |row| {
                            (0..head_dim).map(move |dimension| {
                                (head as f32 + dimension as f32 * 0.001) + row as f32 * 0.0
                            })
                        })
                    })
                    .collect::<Vec<_>>(),
            );
            let mut key = vec![0u16; batch * cache_stride];
            let mut value = vec![0u16; batch * cache_stride];
            for row in 0..batch {
                for kv_head in 0..kv_heads {
                    for position in 0..capacity {
                        let base = row * cache_stride + (kv_head * capacity + position) * head_dim;
                        key[base] = 0x3f80;
                        value[base] = 0x3f80;
                        value[base + 1] = 0x4000;
                    }
                }
            }
            let key_cache = runtime.buffer_with(&key);
            let value_cache = runtime.buffer_with(&value);
            let output = runtime.buffer::<f32>(batch * heads * head_dim);
            let positions = runtime.buffer_with(&[1u32, 1]);
            let params = DecodeAttentionShape {
                sequence: 2,
                capacity: capacity as u32,
                heads: heads as u32,
                kv_heads: kv_heads as u32,
                head_dim: head_dim as u32,
                batch: batch as u32,
                cache_stride: cache_stride as u32,
                scale: 1.0 / (head_dim as f32).sqrt(),
            };
            let command = runtime.queue.new_command_buffer();
            command.set_label("Qwen GQA batch attention test");
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(&query), 0);
            encoder.set_buffer(1, Some(&key_cache), 0);
            encoder.set_buffer(2, Some(&value_cache), 0);
            encoder.set_buffer(3, Some(&output), 0);
            encoder.set_buffer(4, Some(&positions), 0);
            encoder.set_bytes(
                5,
                size_of::<DecodeAttentionShape>() as u64,
                (&params as *const DecodeAttentionShape).cast(),
            );
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: (batch * heads) as u64,
                    height: 1,
                    depth: 1,
                },
                thread_group(32),
            );
            encoder.end_encoding();
            finish(&command).expect("decode Qwen GQA attention command");
            let actual = unsafe {
                std::slice::from_raw_parts(
                    output.contents().cast::<f32>(),
                    batch * heads * head_dim,
                )
            };
            let row_width = heads * head_dim;
            assert_eq!(actual[..row_width], actual[row_width..]);
        });
    }

    #[test]
    fn rope_index() {
        autoreleasepool(|| {
            if metal::Device::system_default().is_none() {
                return;
            }
            let runtime = Runtime::shared().expect("Metal device required");
            let pipeline = runtime
                .precise(source_for("qwen_drope"), "Qwen test", "qwen_drope")
                .expect("decode rope pipeline");
            let q_input = runtime.buffer_with(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
            let k_input = runtime.buffer_with(&[11.0f32, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0]);
            let v_input = runtime.buffer_with(&[21.0f32, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0]);
            let q_output = runtime.buffer::<f32>(8);
            let key_cache = runtime.buffer::<u16>(32);
            let value_cache = runtime.buffer::<u16>(32);
            let rope_table = runtime.buffer_with(&[
                1.0f32, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
            ]);
            let positions = runtime.buffer_with(&[0u32, 1]);
            let params = DecodeRopeShape {
                heads: 1,
                kv_heads: 1,
                head_dim: 4,
                capacity: 4,
                position: 0,
                batch: 2,
                cache_stride: 16,
                rope_theta: 1.0,
            };
            let command = runtime.queue.new_command_buffer();
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline);
            for (index, buffer) in [
                &q_input,
                &k_input,
                &v_input,
                &q_output,
                &key_cache,
                &value_cache,
                &rope_table,
                &positions,
            ]
            .iter()
            .enumerate()
            {
                encoder.set_buffer(index as u64, Some(buffer), 0);
            }
            encoder.set_bytes(
                8,
                size_of::<DecodeRopeShape>() as u64,
                (&params as *const DecodeRopeShape).cast(),
            );
            encoder.dispatch_thread_groups(thread_group(1), thread_group(256));
            encoder.end_encoding();
            command.set_label("Qwen RoPE indexing test");
            finish(&command).expect("decode rope command");

            let output = unsafe {
                std::slice::from_raw_parts(q_output.contents().cast::<f32>(), 8).to_vec()
            };
            assert_eq!(output, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
            let cached =
                unsafe { std::slice::from_raw_parts(key_cache.contents().cast::<u16>(), 32) };
            assert_eq!(cached[0..4], [0x4130, 0x4140, 0x4150, 0x4160]);
            assert_eq!(cached[20..24], [0x4170, 0x4180, 0x4188, 0x4190]);
        });
    }

    #[test]
    fn rope_rows() {
        autoreleasepool(|| {
            if metal::Device::system_default().is_none() {
                return;
            }
            let runtime = Runtime::shared().expect("Metal device required");
            let pipeline = runtime
                .precise(source_for("qwen_drope"), "Qwen test", "qwen_drope")
                .expect("decode rope pipeline");
            let heads = 12usize;
            let kv_heads = 2usize;
            let head_dim = 128usize;
            let batch = 2usize;
            let capacity = 256usize;
            let q_width = heads * head_dim;
            let kv_width = kv_heads * head_dim;
            let cache_stride = kv_width * capacity;
            let query = runtime.buffer_with(&vec![1.0f32; batch * q_width]);
            let key = runtime.buffer_with(&vec![2.0f32; batch * kv_width]);
            let value = runtime.buffer_with(&vec![3.0f32; batch * kv_width]);
            let q_output = runtime.buffer::<f32>(batch * q_width);
            let key_cache = runtime.buffer::<u16>(batch * cache_stride);
            let value_cache = runtime.buffer::<u16>(batch * cache_stride);
            let rope_table = runtime.buffer_with(&vec![1.0f32; 8 * head_dim]);
            let positions = runtime.buffer_with(&[4u32, 4]);
            let params = DecodeRopeShape {
                heads: heads as u32,
                kv_heads: kv_heads as u32,
                head_dim: head_dim as u32,
                capacity: capacity as u32,
                position: 0,
                batch: batch as u32,
                cache_stride: cache_stride as u32,
                rope_theta: 1_000_000.0,
            };
            let command = runtime.queue.new_command_buffer();
            command.set_label("Qwen batch RoPE cache test");
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline);
            for (index, buffer) in [
                &query,
                &key,
                &value,
                &q_output,
                &key_cache,
                &value_cache,
                &rope_table,
                &positions,
            ]
            .iter()
            .enumerate()
            {
                encoder.set_buffer(index as u64, Some(buffer), 0);
            }
            encoder.set_bytes(
                8,
                size_of::<DecodeRopeShape>() as u64,
                (&params as *const DecodeRopeShape).cast(),
            );
            encoder.dispatch_thread_groups(thread_group(12), thread_group(256));
            encoder.end_encoding();
            finish(&command).expect("decode Qwen batch RoPE command");
            let cached = unsafe {
                std::slice::from_raw_parts(key_cache.contents().cast::<u16>(), batch * cache_stride)
            };
            assert_eq!(cached[..cache_stride], cached[cache_stride..]);
            assert!(cached[..cache_stride].iter().any(|&value| value != 0));
        });
    }
}
