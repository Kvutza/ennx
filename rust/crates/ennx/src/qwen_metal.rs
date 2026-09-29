//! Full dense Qwen2.5-Coder forward evaluation on the shared Apple GPU runtime.
//!
//! The evaluator borrows the canonical BF16 parameter buffer used by the
//! resident ENNX search state. It owns only bounded FP32 work buffers, so a
//! candidate evaluation does not duplicate the model weights. The separate
//! immutable evaluator may explicitly cache an exact FP32 readout expansion.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fs::File;
use std::mem::size_of;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use deser::Deserialize;
use memmap2::Mmap;
use metal::objc::rc::autoreleasepool;
use metal::{Buffer, CommandBufferRef, ComputePipelineState, MTLCommandBufferStatus, MTLSize};

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

#[path = "qwen_readout.rs"]
mod readout;

#[path = "qwen_metal/cache.rs"]
mod cache;
#[path = "qwen_metal/checkpoint.rs"]
mod checkpoint;
#[path = "qwen_metal/decode.rs"]
mod decode;
#[path = "qwen_metal/dispatch.rs"]
mod dispatch;
#[path = "qwen_metal/generation.rs"]
mod generation;
#[path = "qwen_metal/loss.rs"]
mod loss;
#[path = "qwen_metal/mps.rs"]
mod mps;
#[path = "qwen_metal/prefill.rs"]
mod prefill;
#[path = "qwen_metal/profiling.rs"]
mod profiling;
#[path = "qwen_metal/setup.rs"]
mod setup;
#[path = "qwen_metal/shapes.rs"]
mod shapes;
use checkpoint::*;
use shapes::*;

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
#[repr(usize)]
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
    frozen_readout: Option<readout::Readout>,
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
