//! Full FLAME BF16 forward evaluation on the shared Apple GPU runtime.
//!
//! Weights use the canonical, lexicographically sorted CUDA/JAX tensor layout.
//! No weight cache, FP16 conversion, or quantization is used. All computation
//! and scratch are FP32. Calls are synchronous and reuse a bounded workspace.
//! Borrowed weight buffers must be ready before entry and must not be mutated
//! until return; producers on other command queues must synchronize first.

use std::collections::BTreeMap;
use std::sync::Arc;

use metal::objc::rc::autoreleasepool;
use metal::{Buffer, CommandBufferRef, ComputePipelineState, MTLCommandBufferStatus, MTLSize};

use crate::apple_gpu::{Runtime, thread_group};

type Result<T> = std::result::Result<T, String>;
const SOURCE: &str = include_str!("flame.metal");
const LOGIT_ROWS: usize = 128;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FlameConfig {
    pub layers: u32,
    pub width: u32,
    pub heads: u32,
    pub vocab: u32,
    pub dense_width: u32,
    pub expert_width: u32,
    pub shared_width: u32,
    pub experts: u32,
    pub top_k: u32,
    pub context: u32,
    pub epsilon: f32,
    pub rope_base: f32,
}

impl Default for FlameConfig {
    fn default() -> Self {
        Self {
            layers: 9,
            width: 1024,
            heads: 16,
            vocab: 50304,
            dense_width: 5472,
            expert_width: 704,
            shared_width: 1408,
            experts: 64,
            top_k: 6,
            context: 2048,
            epsilon: 1e-6,
            rope_base: 10000.0,
        }
    }
}

impl FlameConfig {
    pub fn validate(&self, max_tokens: u32) -> Result<()> {
        if [
            self.layers,
            self.width,
            self.heads,
            self.vocab,
            self.dense_width,
            self.expert_width,
            self.shared_width,
            self.experts,
            self.top_k,
            self.context,
        ]
        .iter()
        .any(|&x| x == 0 || x > i32::MAX as u32)
            || self.width % self.heads != 0
            || (self.width / self.heads) % 2 != 0
            || self.top_k > self.experts
            || self.width > i32::MAX as u32 / 3
            || [self.dense_width, self.expert_width, self.shared_width]
                .iter()
                .any(|&x| x > i32::MAX as u32 / 2)
            || max_tokens == 0
            || max_tokens > self.context
            || u64::from(max_tokens) * u64::from(self.top_k) > i32::MAX as u64
            || !self.epsilon.is_finite()
            || self.epsilon <= 0.0
            || self.epsilon >= 1.0
            || !self.rope_base.is_finite()
            || self.rope_base <= 1.0
        {
            return Err(
                "Invalid FLAME dimensions, attention heads, routing, or normalization".into(),
            );
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct MetalMemoryInfo {
    pub name: String,
    pub has_unified_memory: bool,
    pub recommended_max_working_set_size: u64,
    pub current_allocated_size: u64,
    pub max_buffer_length: u64,
}

/// Metal memory accounting is advisory; OS and non-Metal allocations share RAM.
pub fn memory_info() -> Result<MetalMemoryInfo> {
    autoreleasepool(memory_infoinner)
}

fn memory_infoinner() -> Result<MetalMemoryInfo> {
    let runtime = Runtime::shared()?;
    Ok(MetalMemoryInfo {
        name: runtime.device.name().to_owned(),
        has_unified_memory: runtime.device.has_unified_memory(),
        recommended_max_working_set_size: runtime.device.recommended_max_working_set_size(),
        current_allocated_size: runtime.device.current_allocated_size(),
        max_buffer_length: runtime.device.max_buffer_length(),
    })
}

fn memory_guard(runtime: &Runtime, total: u64, largest: u64) -> Result<()> {
    if largest > runtime.device.max_buffer_length() {
        return Err(format!(
            "FLAME allocation of {largest} bytes exceeds Metal maxBufferLength {}",
            runtime.device.max_buffer_length()
        ));
    }
    let recommended = runtime.device.recommended_max_working_set_size();
    let allocated = runtime.device.current_allocated_size();
    // Leave a margin inside Metal's own recommended working set. This cannot
    // replace the caller's preflight of shared system RAM or concurrent users.
    let limit = recommended - recommended / 10;
    if allocated.checked_add(total).is_none_or(|x| x > limit) {
        return Err(format!(
            "FLAME needs {total} additional Metal bytes; {allocated} allocated, conservative budget {limit}"
        ));
    }
    Ok(())
}

/// Upload verbatim finite BF16 bits once, for sharing with evaluator and search.
/// The evaluator additionally checks this buffer's canonical model length.
pub fn upload(weights: &[u16]) -> Result<Buffer> {
    autoreleasepool(|| upload_inner(weights))
}

fn upload_inner(weights: &[u16]) -> Result<Buffer> {
    if weights.is_empty() || weights.iter().any(|x| x & 0x7f80 == 0x7f80) {
        return Err("FLAME upload requires nonempty finite BF16 weights".into());
    }
    let bytes = product(&[weights.len(), 2])? as u64;
    let runtime = Runtime::shared()?;
    memory_guard(&runtime, bytes, bytes)?;
    let buffer = runtime.buffer_with(weights);
    if buffer.contents().is_null() {
        return Err("Metal could not allocate FLAME weights".into());
    }
    Ok(buffer)
}

fn product(values: &[usize]) -> Result<usize> {
    values.iter().try_fold(1usize, |n, &v| {
        n.checked_mul(v)
            .filter(|&x| x <= isize::MAX as usize)
            .ok_or_else(|| "FLAME allocation or shape overflow".into())
    })
}

fn weight_count(c: FlameConfig) -> Result<usize> {
    let h = c.width as usize;
    let moe = (c.layers - 1) as usize;
    let terms: &[&[usize]] = &[
        &[2, c.vocab as usize, h],
        &[h],
        &[c.layers as usize, 4, h, h],
        &[c.layers as usize, 2, h],
        &[3, c.dense_width as usize, h],
        &[moe, c.experts as usize, h],
        &[moe, 3, c.shared_width as usize, h],
        &[moe, 3, c.experts as usize, c.expert_width as usize, h],
    ];
    terms.iter().try_fold(0usize, |sum, dims| {
        sum.checked_add(product(dims)?)
            .ok_or_else(|| "FLAME weight count overflow".into())
    })
}

#[derive(Default, Clone, Copy)]
struct Layer {
    attention_norm: usize,
    qkv: usize,
    projection: usize,
    mlp_norm: usize,
    first: usize,
    second: usize,
    router: usize,
    expert_first: usize,
    expert_second: usize,
}

struct Layout {
    layers: Vec<Layer>,
    embedding: usize,
    final_norm: usize,
    output: usize,
    len: usize,
    #[cfg(test)]
    tensors: BTreeMap<String, (usize, usize)>,
}

impl Layout {
    fn new(c: FlameConfig) -> Result<Self> {
        let h = c.width as usize;
        let mut tensors = BTreeMap::<String, (usize, usize)>::new();
        let mut add = |name: String, dims: &[usize]| -> Result<()> {
            tensors.insert(name, (0, product(dims)?));
            Ok(())
        };
        add(
            "embedding.word_embeddings.weight".into(),
            &[c.vocab as usize, h],
        )?;
        add("output_layer.weight".into(), &[c.vocab as usize, h])?;
        add("decoder.final_layernorm.weight".into(), &[h])?;
        for i in 0..c.layers {
            let p = format!("decoder.layers.{i}.");
            add(p.clone() + "self_attention.linear_qkv.weight", &[3, h, h])?;
            add(
                p.clone() + "self_attention.linear_qkv.layer_norm_weight",
                &[h],
            )?;
            add(p.clone() + "self_attention.linear_proj.weight", &[h, h])?;
            if i == 0 {
                add(p.clone() + "mlp.linear_fc1.layer_norm_weight", &[h])?;
                add(
                    p.clone() + "mlp.linear_fc1.weight",
                    &[2, c.dense_width as usize, h],
                )?;
                add(p + "mlp.linear_fc2.weight", &[h, c.dense_width as usize])?;
            } else {
                add(p.clone() + "pre_mlp_layernorm.weight", &[h])?;
                add(p.clone() + "mlp.router.weight", &[c.experts as usize, h])?;
                add(
                    p.clone() + "mlp.shared_experts.linear_fc1.weight",
                    &[2, c.shared_width as usize, h],
                )?;
                add(
                    p.clone() + "mlp.shared_experts.linear_fc2.weight",
                    &[h, c.shared_width as usize],
                )?;
                add(
                    p.clone() + "mlp.experts.experts.linear_fc1.weight",
                    &[c.experts as usize, 2, c.expert_width as usize, h],
                )?;
                add(
                    p + "mlp.experts.experts.linear_fc2.weight",
                    &[c.experts as usize, h, c.expert_width as usize],
                )?;
            }
        }
        let mut len = 0usize;
        for (offset, count) in tensors.values_mut() {
            *offset = len;
            len = len
                .checked_add(*count)
                .ok_or("FLAME layout offset overflow")?;
        }
        product(&[len, 2])?;
        let mut layers = Vec::with_capacity(c.layers as usize);
        for i in 0..c.layers {
            let p = format!("decoder.layers.{i}.");
            let get = |s: &str| tensors[&(p.clone() + s)].0;
            let mut layer = Layer {
                attention_norm: get("self_attention.linear_qkv.layer_norm_weight"),
                qkv: get("self_attention.linear_qkv.weight"),
                projection: get("self_attention.linear_proj.weight"),
                ..Layer::default()
            };
            if i == 0 {
                layer.mlp_norm = get("mlp.linear_fc1.layer_norm_weight");
                layer.first = get("mlp.linear_fc1.weight");
                layer.second = get("mlp.linear_fc2.weight");
            } else {
                layer.mlp_norm = get("pre_mlp_layernorm.weight");
                layer.router = get("mlp.router.weight");
                layer.first = get("mlp.shared_experts.linear_fc1.weight");
                layer.second = get("mlp.shared_experts.linear_fc2.weight");
                layer.expert_first = get("mlp.experts.experts.linear_fc1.weight");
                layer.expert_second = get("mlp.experts.experts.linear_fc2.weight");
            }
            layers.push(layer);
        }
        Ok(Self {
            layers,
            len,
            embedding: tensors["embedding.word_embeddings.weight"].0,
            final_norm: tensors["decoder.final_layernorm.weight"].0,
            output: tensors["output_layer.weight"].0,
            #[cfg(test)]
            tensors,
        })
    }
}

// Buffer indices and allocation sizes are defined together to keep accounting
// identical to the actual reusable allocations (including tokens and routing).
#[repr(usize)]
#[derive(Clone, Copy)]
enum W {
    X,
    Norm,
    Qkv,
    Q,
    K,
    V,
    Scores,
    Attended,
    Update,
    Gates,
    Activation,
    RouteLogits,
    Probs,
    Routed,
    Gathered,
    ExpertOutput,
    Logits,
    Losses,
    Tokens,
    Masks,
    Indices,
    Slots,
    Counts,
    Invalid,
}

fn workspace_sizes(c: FlameConfig, capacity: u32) -> Result<Vec<usize>> {
    let n = capacity as usize;
    let h = c.width as usize;
    let hidden = c.dense_width.max(c.expert_width).max(c.shared_width) as usize;
    let nh = product(&[n, h])?;
    let routes = product(&[n, c.top_k as usize])?;
    let nhidden = product(&[n, hidden])?;
    let mut sizes = vec![0; W::Invalid as usize + 1];
    let mut add = |w: W, dims: &[usize], element_bytes: usize| -> Result<()> {
        let elements = product(dims)?;
        // Linear elementwise dispatches use uint thread positions. Matmul
        // device addressing and BF16 model offsets use 64-bit arithmetic.
        if elements > (u32::MAX - 255) as usize {
            return Err("FLAME workspace exceeds Metal dispatch indexing range".into());
        }
        sizes[w as usize] = product(&[elements, element_bytes])?;
        Ok(())
    };
    for w in [
        W::X,
        W::Norm,
        W::Q,
        W::K,
        W::V,
        W::Attended,
        W::Update,
        W::Gathered,
        W::ExpertOutput,
    ] {
        add(w, &[nh], 4)?;
    }
    add(W::Qkv, &[nh, 3], 4)?;
    add(W::Scores, &[n, n, c.heads as usize], 4)?;
    add(W::Gates, &[nhidden, 2], 4)?;
    add(W::Activation, &[nhidden], 4)?;
    add(W::RouteLogits, &[n, c.experts as usize], 4)?;
    add(W::Probs, &[routes], 4)?;
    add(W::Routed, &[routes, h], 4)?;
    add(W::Logits, &[n.min(LOGIT_ROWS), c.vocab as usize], 4)?;
    add(W::Losses, &[n], 4)?;
    add(W::Tokens, &[n], 4)?;
    add(W::Masks, &[n], 1)?;
    add(W::Indices, &[routes], 4)?;
    add(W::Slots, &[n, c.experts as usize], 4)?;
    add(W::Counts, &[c.experts as usize], 4)?;
    add(W::Invalid, &[1], 4)?;
    Ok(sizes)
}

#[repr(C)]
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
struct Shape {
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

/// Reuses FP32 workspace while borrowing an unchanged full BF16 weight buffer.
pub struct FlameEvaluator {
    runtime: Arc<Runtime>,
    config: FlameConfig,
    max_tokens: u32,
    layout: Layout,
    workspace: Vec<Buffer>,
    workspace_bytes: u64,
    pipelines: BTreeMap<&'static str, ComputePipelineState>,
}

impl FlameEvaluator {
    pub fn new(config: FlameConfig, max_tokens: u32) -> Result<Self> {
        autoreleasepool(|| Self::new_inner(config, max_tokens))
    }

    fn new_inner(config: FlameConfig, max_tokens: u32) -> Result<Self> {
        config.validate(max_tokens)?;
        let sizes = workspace_sizes(config, max_tokens)?;
        let bytes = sizes
            .iter()
            .try_fold(0u64, |a, &b| a.checked_add(b as u64))
            .ok_or("FLAME workspace size overflow")?;
        let runtime = Runtime::shared()?;
        memory_guard(&runtime, bytes, *sizes.iter().max().unwrap() as u64)?;
        if product(&[weight_count(config)?, 2])? as u64 > runtime.device.max_buffer_length() {
            return Err("FLAME BF16 weight layout exceeds Metal maxBufferLength".into());
        }
        let layout = Layout::new(config)?;
        let mut pipelines = BTreeMap::new();
        for name in [
            "flame_linear",
            "flame_linear_small",
            "flame_matmul",
            "flame_embedding",
            "flame_rms",
            "flame_rotary",
            "flame_softmax",
            "flame_unpack",
            "flame_residual",
            "flame_silu",
            "flame_router",
            "flame_group",
            "flame_gather",
            "flame_scatter",
            "flame_combine",
            "flame_cross_entropy",
            "flame_mean",
        ] {
            let pipeline = runtime.precise(SOURCE, "FLAME", name)?;
            if pipeline.thread_execution_width() != 32
                || pipeline.max_total_threads_per_threadgroup() < 256
            {
                return Err(
                    "FLAME requires Apple GPU 32-lane SIMD groups and 256-thread groups".into(),
                );
            }
            pipelines.insert(name, pipeline);
        }
        let mut workspace = Vec::with_capacity(sizes.len());
        for size in sizes {
            let buffer = runtime.buffer::<u8>(size);
            if buffer.contents().is_null() {
                return Err(format!(
                    "Metal could not allocate {size} FLAME workspace bytes"
                ));
            }
            workspace.push(buffer);
        }
        Ok(Self {
            runtime,
            config,
            max_tokens,
            layout,
            workspace,
            workspace_bytes: bytes,
            pipelines,
        })
    }

    pub fn weights_len(&self) -> usize {
        self.layout.len
    }
    pub fn workspace_bytes(&self) -> u64 {
        self.workspace_bytes
    }
    pub fn vocab(&self) -> usize {
        self.config.vocab as usize
    }

    pub fn upload(&self, weights: &[u16]) -> Result<Buffer> {
        if weights.len() != self.weights_len() {
            return Err(format!(
                "FLAME requires {} BF16 weights, got {}",
                self.weights_len(),
                weights.len()
            ));
        }
        upload(weights)
    }

    pub fn check_tokens(&self, tokens: &[i32]) -> Result<()> {
        if tokens.is_empty()
            || tokens.len() > self.max_tokens as usize
            || tokens
                .iter()
                .any(|&x| x < 0 || x as u32 >= self.config.vocab)
        {
            return Err(
                "FLAME tokens must be nonempty, in vocabulary, and within max_tokens".into(),
            );
        }
        Ok(())
    }

    pub fn check_batch(&self, tokens: &[Vec<i32>], masks: &[Vec<bool>]) -> Result<()> {
        if tokens.is_empty() || tokens.len() != masks.len() {
            return Err("FLAME requires equally sized nonempty token and mask batches".into());
        }
        for (row, mask) in tokens.iter().zip(masks) {
            self.check_tokens(row)?;
            if row.len() < 2 || row.len() != mask.len() || mask[0] || !mask[1..].iter().any(|&x| x)
            {
                return Err("Each FLAME loss mask must match its tokens, leave token zero unscored, and score a target".into());
            }
        }
        Ok(())
    }

    pub fn check_weights(&self, weights: &Buffer) -> Result<()> {
        if weights.length() != (self.weights_len() as u64) * 2 {
            return Err(format!(
                "FLAME requires {} contiguous Metal BF16 weights",
                self.weights_len()
            ));
        }
        if weights.device().registry_id() != self.runtime.device.registry_id() {
            return Err("FLAME weights must belong to the evaluator's Metal device".into());
        }
        Ok(())
    }

    /// Row-major [tokens.len(), vocab] FP32 logits, including the last token.
    pub fn logits(&mut self, weights: &Buffer, tokens: &[i32]) -> Result<Vec<f32>> {
        autoreleasepool(|| self.logits_inner(weights, tokens, false))
    }

    /// Vocab-length FP32 logits for the next token. Recomputes the full prefix,
    /// but projects and copies only its final row, using the full-logits kernel.
    pub fn next_logits(&mut self, weights: &Buffer, tokens: &[i32]) -> Result<Vec<f32>> {
        autoreleasepool(|| self.logits_inner(weights, tokens, true))
    }

    fn logits_inner(
        &mut self,
        weights: &Buffer,
        tokens: &[i32],
        last_only: bool,
    ) -> Result<Vec<f32>> {
        self.check_weights(weights)?;
        self.check_tokens(tokens)?;
        let first_row = if last_only { tokens.len() - 1 } else { 0 };
        let len = product(&[tokens.len() - first_row, self.vocab()])?;
        product(&[len, 4])?;
        let mut output = Vec::new();
        output
            .try_reserve_exact(len)
            .map_err(|e| format!("FLAME host logits allocation: {e}"))?;
        output.resize(len, 0.0);
        self.write(W::Tokens, tokens);
        self.write(W::Masks, &vec![0u8; tokens.len()]);
        self.forward(weights, tokens.len() as u32)?;
        self.output(
            weights,
            tokens.len() as u32,
            first_row as u32,
            0,
            Some(&mut output),
        )?;
        Ok(output)
    }

    /// Mean shifted masked cross-entropy for each unpadded sequence. Validates
    /// the entire batch before submitting any GPU work; sequences run serially.
    pub fn losses(
        &mut self,
        weights: &Buffer,
        tokens: &[Vec<i32>],
        masks: &[Vec<bool>],
    ) -> Result<Vec<f32>> {
        self.check_weights(weights)?;
        self.check_batch(tokens, masks)?;
        let mut output = Vec::with_capacity(tokens.len());
        for (row, mask) in tokens.iter().zip(masks) {
            let loss = autoreleasepool(|| {
                self.write(W::Tokens, row);
                let mask: Vec<u8> = mask.iter().map(|&x| u8::from(x)).collect();
                self.write(W::Masks, &mask);
                self.forward(weights, row.len() as u32)?;
                self.output(
                    weights,
                    row.len() as u32,
                    0,
                    mask.iter().map(|&x| u32::from(x)).sum(),
                    None,
                )
            })?;
            output.push(loss);
        }
        Ok(output)
    }

    fn b(&self, w: W) -> &Buffer {
        &self.workspace[w as usize]
    }

    fn write<T: Copy>(&self, w: W, values: &[T]) {
        assert!(std::mem::size_of_val(values) as u64 <= self.b(w).length());
        // Workspace is private, shared-storage, and every previous call waited.
        unsafe {
            std::ptr::copy_nonoverlapping(
                values.as_ptr().cast::<u8>(),
                self.b(w).contents().cast(),
                std::mem::size_of_val(values),
            );
        }
    }

    fn read<T: Copy>(&self, w: W, count: usize) -> Vec<T> {
        assert!(count * size_of::<T>() <= self.b(w).length() as usize);
        // Only called after waiting for GPU completion, never for borrowed weights.
        unsafe { std::slice::from_raw_parts(self.b(w).contents().cast::<T>(), count).to_vec() }
    }

    fn shape(&self, rows: u32) -> Shape {
        Shape {
            rows,
            width: self.config.width,
            heads: self.config.heads,
            experts: self.config.experts,
            top_k: self.config.top_k,
            epsilon: self.config.epsilon,
            rope_base: self.config.rope_base,
            ..Shape::default()
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
        // Separate serial encoders give tracked scratch buffers automatic hazards.
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipelines[name]);
        for (i, &(buffer, offset)) in buffers.iter().enumerate() {
            encoder.set_buffer(i as u64, Some(buffer), offset);
        }
        encoder.set_bytes(
            buffers.len() as u64,
            size_of::<T>() as u64,
            (params as *const T).cast(),
        );
        encoder.dispatch_thread_groups(groups, thread_group(256));
        encoder.end_encoding();
    }

    fn elementwise(
        &self,
        cmd: &CommandBufferRef,
        name: &str,
        buffers: &[(&Buffer, u64)],
        p: &Shape,
        count: u64,
    ) {
        self.encode(cmd, name, buffers, p, thread_group(count.div_ceil(256)));
    }

    fn linear(
        &self,
        cmd: &CommandBufferRef,
        input: &Buffer,
        input_offset: u64,
        weights: &Buffer,
        offset: usize,
        output: &Buffer,
        rows: u32,
        inside: u32,
        outside: u32,
    ) {
        self.linear_dispatch(
            cmd,
            input,
            input_offset,
            weights,
            offset,
            output,
            rows,
            inside,
            outside,
            rows < 8,
        );
    }

    fn linear_dispatch(
        &self,
        cmd: &CommandBufferRef,
        input: &Buffer,
        input_offset: u64,
        weights: &Buffer,
        offset: usize,
        output: &Buffer,
        rows: u32,
        inside: u32,
        outside: u32,
        small: bool,
    ) {
        let p = Matmul {
            m: rows,
            n: outside,
            k: inside,
            transpose_b: 1,
            stride_a: 0,
            stride_b: 0,
            stride_c: 0,
        };
        self.encode(
            cmd,
            if small {
                "flame_linear_small"
            } else {
                "flame_linear"
            },
            &[
                (input, input_offset),
                (weights, offset as u64 * 2),
                (output, 0),
            ],
            &p,
            MTLSize {
                width: u64::from(outside).div_ceil(if small { 8 } else { 32 }),
                height: if small {
                    u64::from(rows)
                } else {
                    u64::from(rows).div_ceil(32)
                },
                depth: 1,
            },
        );
    }

    fn normalized(&self, cmd: &CommandBufferRef, weights: &Buffer, offset: usize, rows: u32) {
        self.encode(
            cmd,
            "flame_rms",
            &[
                (self.b(W::X), 0),
                (weights, offset as u64 * 2),
                (self.b(W::Norm), 0),
            ],
            &self.shape(rows),
            thread_group(u64::from(rows)),
        );
    }

    fn mlp(
        &self,
        cmd: &CommandBufferRef,
        weights: &Buffer,
        input: W,
        output: W,
        first: usize,
        second: usize,
        rows: u32,
        hidden: u32,
    ) {
        self.linear(
            cmd,
            self.b(input),
            0,
            weights,
            first,
            self.b(W::Gates),
            rows,
            self.config.width,
            2 * hidden,
        );
        let p = Shape {
            hidden,
            ..self.shape(rows)
        };
        self.elementwise(
            cmd,
            "flame_silu",
            &[(self.b(W::Gates), 0), (self.b(W::Activation), 0)],
            &p,
            u64::from(rows) * u64::from(hidden),
        );
        self.linear(
            cmd,
            self.b(W::Activation),
            0,
            weights,
            second,
            self.b(output),
            rows,
            hidden,
            self.config.width,
        );
    }

    fn forward(&self, weights: &Buffer, rows: u32) -> Result<()> {
        let c = self.config;
        let p = self.shape(rows);
        let nh = u64::from(rows) * u64::from(c.width);
        let mut command = self.runtime.queue.new_command_buffer().to_owned();
        self.elementwise(
            &command,
            "flame_embedding",
            &[
                (weights, self.layout.embedding as u64 * 2),
                (self.b(W::Tokens), 0),
                (self.b(W::X), 0),
            ],
            &p,
            nh,
        );
        for (i, layer) in self.layout.layers.iter().enumerate() {
            self.normalized(&command, weights, layer.attention_norm, rows);
            self.linear(
                &command,
                self.b(W::Norm),
                0,
                weights,
                layer.qkv,
                self.b(W::Qkv),
                rows,
                c.width,
                3 * c.width,
            );
            self.elementwise(
                &command,
                "flame_rotary",
                &[
                    (self.b(W::Qkv), 0),
                    (self.b(W::Q), 0),
                    (self.b(W::K), 0),
                    (self.b(W::V), 0),
                ],
                &p,
                nh,
            );
            let d = c.width / c.heads;
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
                &[(self.b(W::Q), 0), (self.b(W::K), 0), (self.b(W::Scores), 0)],
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
                &[(self.b(W::Scores), 0)],
                &p,
                thread_group(u64::from(rows) * u64::from(c.heads)),
            );
            let av = Matmul {
                m: rows,
                n: d,
                k: rows,
                transpose_b: 0,
                stride_a: qk.stride_c,
                stride_b: qk.stride_b,
                stride_c: qk.stride_a,
            };
            self.encode(
                &command,
                "flame_matmul",
                &[
                    (self.b(W::Scores), 0),
                    (self.b(W::V), 0),
                    (self.b(W::Qkv), 0),
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
                "flame_unpack",
                &[(self.b(W::Qkv), 0), (self.b(W::Attended), 0)],
                &p,
                nh,
            );
            self.linear(
                &command,
                self.b(W::Attended),
                0,
                weights,
                layer.projection,
                self.b(W::Update),
                rows,
                c.width,
                c.width,
            );
            self.elementwise(
                &command,
                "flame_residual",
                &[(self.b(W::X), 0), (self.b(W::Update), 0)],
                &p,
                nh,
            );
            self.normalized(&command, weights, layer.mlp_norm, rows);
            if i == 0 {
                self.mlp(
                    &command,
                    weights,
                    W::Norm,
                    W::Update,
                    layer.first,
                    layer.second,
                    rows,
                    c.dense_width,
                );
                self.elementwise(
                    &command,
                    "flame_residual",
                    &[(self.b(W::X), 0), (self.b(W::Update), 0)],
                    &p,
                    nh,
                );
                continue;
            }
            self.linear(
                &command,
                self.b(W::Norm),
                0,
                weights,
                layer.router,
                self.b(W::RouteLogits),
                rows,
                c.width,
                c.experts,
            );
            self.encode(
                &command,
                "flame_router",
                &[
                    (self.b(W::RouteLogits), 0),
                    (self.b(W::Probs), 0),
                    (self.b(W::Indices), 0),
                ],
                &p,
                thread_group(u64::from(rows)),
            );
            let grouped = Shape {
                sequence: self.max_tokens,
                ..p
            };
            self.elementwise(
                &command,
                "flame_group",
                &[
                    (self.b(W::Indices), 0),
                    (self.b(W::Slots), 0),
                    (self.b(W::Counts), 0),
                ],
                &grouped,
                u64::from(c.experts),
            );
            self.mlp(
                &command,
                weights,
                W::Norm,
                W::Update,
                layer.first,
                layer.second,
                rows,
                c.shared_width,
            );
            finish(&command)?;
            let counts = self.read::<u32>(W::Counts, c.experts as usize);
            if counts.iter().any(|&count| count > rows)
                || counts.iter().map(|&x| u64::from(x)).sum::<u64>()
                    != u64::from(rows) * u64::from(c.top_k)
            {
                return Err(
                    "FLAME invalid expert route counts (possibly nonfinite router logits)".into(),
                );
            }
            command = self.runtime.queue.new_command_buffer().to_owned();
            for (expert, &count) in counts.iter().enumerate() {
                if count == 0 {
                    continue;
                }
                let slots = (
                    self.b(W::Slots),
                    expert as u64 * u64::from(self.max_tokens) * 4,
                );
                let ep = self.shape(count);
                let elements = u64::from(count) * u64::from(c.width);
                self.elementwise(
                    &command,
                    "flame_gather",
                    &[(self.b(W::Norm), 0), slots, (self.b(W::Gathered), 0)],
                    &ep,
                    elements,
                );
                self.mlp(
                    &command,
                    weights,
                    W::Gathered,
                    W::ExpertOutput,
                    layer.expert_first + expert * 2 * c.expert_width as usize * c.width as usize,
                    layer.expert_second + expert * c.width as usize * c.expert_width as usize,
                    count,
                    c.expert_width,
                );
                self.elementwise(
                    &command,
                    "flame_scatter",
                    &[(self.b(W::ExpertOutput), 0), slots, (self.b(W::Routed), 0)],
                    &ep,
                    elements,
                );
            }
            self.elementwise(
                &command,
                "flame_combine",
                &[
                    (self.b(W::X), 0),
                    (self.b(W::Update), 0),
                    (self.b(W::Routed), 0),
                    (self.b(W::Probs), 0),
                ],
                &p,
                nh,
            );
        }
        self.normalized(&command, weights, self.layout.final_norm, rows);
        finish(&command)
    }

    fn output(
        &self,
        weights: &Buffer,
        rows: u32,
        first_row: u32,
        scored: u32,
        mut host_logits: Option<&mut [f32]>,
    ) -> Result<f32> {
        self.write(W::Invalid, &[0u32]);
        for start in (first_row..rows).step_by(LOGIT_ROWS) {
            let chunk = (rows - start).min(LOGIT_ROWS as u32);
            // A final-row projection must retain the full output chunk's
            // accumulation kernel, even when only one row is dispatched.
            let full_start = start / LOGIT_ROWS as u32 * LOGIT_ROWS as u32;
            let full_chunk = (rows - full_start).min(LOGIT_ROWS as u32);
            let command = self.runtime.queue.new_command_buffer();
            self.linear_dispatch(
                command,
                self.b(W::Norm),
                u64::from(start) * u64::from(self.config.width) * 4,
                weights,
                self.layout.output,
                self.b(W::Logits),
                chunk,
                self.config.width,
                self.config.vocab,
                full_chunk < 8,
            );
            let p = Shape {
                width: self.config.vocab,
                start,
                sequence: rows,
                ..self.shape(chunk)
            };
            self.encode(
                command,
                "flame_cross_entropy",
                &[
                    (self.b(W::Logits), 0),
                    (self.b(W::Tokens), 0),
                    (self.b(W::Masks), 0),
                    (self.b(W::Losses), 0),
                    (self.b(W::Invalid), 0),
                ],
                &p,
                thread_group(u64::from(chunk)),
            );
            finish(command)?;
            if self.read::<u32>(W::Invalid, 1)[0] != 0 {
                return Err("FLAME produced nonfinite logits or loss".into());
            }
            if let Some(output) = host_logits.as_deref_mut() {
                let begin = (start - first_row) as usize * self.vocab();
                let len = chunk as usize * self.vocab();
                // Avoid an extra host copy of each potentially large logit tile.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        self.b(W::Logits).contents().cast::<f32>(),
                        output[begin..begin + len].as_mut_ptr(),
                        len,
                    );
                }
            }
        }
        if scored == 0 {
            return Ok(0.0);
        }
        let command = self.runtime.queue.new_command_buffer();
        let p = Shape {
            hidden: scored,
            ..self.shape(rows)
        };
        self.encode(
            command,
            "flame_mean",
            &[(self.b(W::Losses), 0), (self.b(W::Invalid), 0)],
            &p,
            thread_group(1),
        );
        finish(command)?;
        let loss = self.read::<f32>(W::Losses, 1)[0];
        if self.read::<u32>(W::Invalid, 1)[0] != 0 || !loss.is_finite() || loss < 0.0 {
            return Err("FLAME returned an invalid loss".into());
        }
        Ok(loss)
    }
}

fn finish(command: &CommandBufferRef) -> Result<()> {
    command.commit();
    command.wait_until_completed();
    if command.status() != MTLCommandBufferStatus::Completed {
        return Err(format!(
            "FLAME Metal command failed with status {:?}",
            command.status()
        ));
    }
    Ok(())
}

#[cfg(test)]
#[path = "flame_metaltests.rs"]
mod tests;
