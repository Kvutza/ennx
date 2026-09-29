//! Persistent KV and bounded query work, using the production SIMD-matrix PISA kernel.
use crate::apple_gpu::{Runtime, gpu_seconds, thread_group};
use crate::context::{BLOCK, HEADS, Layout, Output, SELECTED, WIDTH};
use metal::{
    Buffer, BufferRef, ComputeCommandEncoderRef, ComputePipelineState, MTLCommandBufferStatus,
};
use std::sync::Arc;
use std::time::Instant;
#[cfg(test)]
#[path = "context_metal/tests.rs"]
mod tests;

/// GPU-resident bridge from model projection scratch to persistent KV.
pub(crate) struct ContextKernels {
    layout: Layout,
    queries: Buffer,
    pack: ComputePipelineState,
    leaves: ComputePipelineState,
    parents: ComputePipelineState,
    attention: ComputePipelineState,
    index: Option<IndexState>,
}

struct IndexState {
    kernel: ComputePipelineState,
    history: Buffer,
    saved: Buffer,
    stamps: Buffer,
    counters: Buffer,
}

#[derive(Clone, Copy)]
pub(crate) struct IndexPolicy<'a> {
    pub weights: &'a BufferRef,
    pub offset: u64,
    pub block: u32,
    pub mode: crate::config::IndexMode,
    pub layer: u32,
    pub fresh: bool,
    pub reuse: f32,
}

#[repr(C)]
struct IndexShape {
    start: u32,
    rows: u32,
    block: u32,
    mode: u32,
    layer: u32,
    fresh: u32,
    reuse: f32,
    tokens: u32,
}

impl ContextKernels {
    pub(crate) fn new(runtime: &Runtime, layout: Layout) -> Result<Self, String> {
        Self::with_index(runtime, layout, false)
    }

    pub(crate) fn with_index(
        runtime: &Runtime,
        layout: Layout,
        indexed: bool,
    ) -> Result<Self, String> {
        let layout = Layout::new(layout.tokens, layout.queries)?;
        let source = format!(
            "#define PISA_CONTEXT {}\n#define PISA_ROWS {}\n#define PISA_CACHE\n#define PISA_SKIP_IDENTITY_RESCALE\n{}\n{}",
            layout.tokens,
            layout.tokens,
            if indexed {
                "#define PISA_EXTERNAL_INDEX"
            } else {
                ""
            },
            include_str!("fbt_pisa1.metal")
        );
        let tree = include_str!("context_tree.metal");
        Ok(Self {
            layout,
            queries: runtime.buffer::<u16>((layout.queries * HEADS * WIDTH) as usize),
            pack: runtime.pipeline(tree, "context pack", "context_pack")?,
            leaves: runtime.pipeline(tree, "context tree", "context_leaves")?,
            parents: runtime.pipeline(tree, "context tree", "context_parents")?,
            attention: runtime.pipeline(
                &source,
                "cached model attention",
                "fbt_pisa1_select_attention_q4",
            )?,
            index: if indexed {
                Some(IndexState {
                    kernel: runtime.precise(
                        include_str!("pisa_index.metal"),
                        "PISA proxy index",
                        "pisa_index",
                    )?,
                    history: runtime.buffer_with(&vec![0.0f32; 5 * 1024 * 2304]),
                    saved: runtime.buffer_with(&vec![u32::MAX; 5 * 1024 * 32]),
                    stamps: runtime.buffer_with(&vec![u32::MAX; 5 * 1024]),
                    counters: runtime.buffer_with(&[0u32; 2]),
                })
            } else {
                None
            },
        })
    }

    pub(crate) fn encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        qkv: &BufferRef,
        kv: &BufferRef,
        tree: &BufferRef,
        blocks: &BufferRef,
        output: &BufferRef,
        start: u32,
        rows: u32,
    ) -> Result<(), String> {
        self.encode_policy(
            encoder,
            qkv,
            kv,
            tree,
            blocks,
            Some(output),
            start,
            rows,
            None,
        )
    }

    pub(crate) fn index_counts(&self) -> [usize; 2] {
        self.index.as_ref().map_or([0; 2], |index| unsafe {
            let counts = std::slice::from_raw_parts(index.counters.contents().cast::<u32>(), 2);
            [counts[0] as usize, counts[1] as usize]
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_policy(
        &self,
        encoder: &ComputeCommandEncoderRef,
        qkv: &BufferRef,
        kv: &BufferRef,
        tree: &BufferRef,
        blocks: &BufferRef,
        output: Option<&BufferRef>,
        start: u32,
        rows: u32,
        policy: Option<IndexPolicy<'_>>,
    ) -> Result<(), String> {
        self.layout.range(start, rows)?;
        if policy.is_some() != self.index.is_some() {
            return Err("PISA index policy does not match the compiled attention kernel".into());
        }
        encoder.set_compute_pipeline_state(&self.pack);
        for (index, buffer) in [qkv, kv, &self.queries].into_iter().enumerate() {
            encoder.set_buffer(index as u64, Some(buffer), 0);
        }
        bytes(encoder, 3, &[start, rows]);
        encoder.dispatch_threads(thread_group(u64::from(rows) * 640), thread_group(256));
        encoder.memory_barrier_with_resources(&[kv, &self.queries]);
        let (mut first, mut end) = (start / BLOCK, (start + rows).div_ceil(BLOCK));
        encoder.set_compute_pipeline_state(&self.leaves);
        encoder.set_buffer(0, Some(kv), 0);
        encoder.set_buffer(1, Some(tree), 0);
        bytes(encoder, 2, &[first, end - first]);
        encoder.dispatch_thread_groups(thread_group(u64::from(end - first)), thread_group(64));
        encoder.memory_barrier_with_resources(&[tree]);
        let (mut child, mut parent, mut count) = (0, self.layout.leaves(), self.layout.leaves());
        while count > 1 {
            first /= 2;
            end = end.div_ceil(2);
            encoder.set_compute_pipeline_state(&self.parents);
            encoder.set_buffer(0, Some(tree), 0);
            bytes(encoder, 1, &[child, parent, first, end - first]);
            encoder.dispatch_thread_groups(thread_group(u64::from(end - first)), thread_group(64));
            encoder.memory_barrier_with_resources(&[tree]);
            child = parent;
            count /= 2;
            parent += count;
        }
        self.encode_index(encoder, tree, blocks, start, rows, policy);
        let Some(output) = output else {
            return Ok(());
        };
        encoder.set_compute_pipeline_state(&self.attention);
        for (index, buffer) in [&*self.queries, tree, blocks, output]
            .into_iter()
            .enumerate()
        {
            encoder.set_buffer(index as u64, Some(buffer), 0);
        }
        bytes(encoder, 4, &[start, rows]);
        encoder.set_buffer(5, Some(kv), 0);
        if let Some(policy) = policy {
            bytes(encoder, 6, &policy.block);
        }
        encoder.dispatch_thread_groups(thread_group(u64::from(rows / 4)), thread_group(128));
        encoder.memory_barrier_with_resources(&[blocks, output]);
        Ok(())
    }
    fn encode_index(
        &self,
        encoder: &ComputeCommandEncoderRef,
        tree: &BufferRef,
        blocks: &BufferRef,
        start: u32,
        rows: u32,
        policy: Option<IndexPolicy<'_>>,
    ) {
        if let (Some(policy), Some(index)) = (policy, &self.index) {
            let shape = IndexShape {
                start,
                rows,
                block: policy.block,
                mode: match policy.mode {
                    crate::config::IndexMode::Independent => 0,
                    crate::config::IndexMode::Shared => 1,
                    crate::config::IndexMode::Refined => 2,
                },
                layer: policy.layer,
                fresh: u32::from(policy.fresh),
                reuse: policy.reuse,
                tokens: self.layout.tokens,
            };
            encoder.set_compute_pipeline_state(&index.kernel);
            for (slot, buffer) in [
                &*self.queries,
                tree,
                blocks,
                policy.weights,
                &*index.history,
                &*index.saved,
                &*index.stamps,
                &*index.counters,
            ]
            .into_iter()
            .enumerate()
            {
                encoder.set_buffer(
                    slot as u64,
                    Some(buffer),
                    if slot == 3 { policy.offset } else { 0 },
                );
            }
            bytes(encoder, 8, &shape);
            encoder.dispatch_thread_groups(thread_group(u64::from(rows / 4)), thread_group(32));
            encoder.memory_barrier_with_resources(&[
                blocks,
                &index.history,
                &index.saved,
                &index.stamps,
                &index.counters,
            ]);
        }
    }
}

pub struct ContextCache {
    runtime: Arc<Runtime>,
    layout: Layout,
    kv: Buffer,
    tree: Buffer,
    queries: Buffer,
    blocks: Buffer,
    output: Buffer,
    leaves: ComputePipelineState,
    parents: ComputePipelineState,
    attention: ComputePipelineState,
    filled: u32,
}

impl ContextCache {
    pub fn new(layout: Layout) -> Result<Self, String> {
        let layout = Layout::new(layout.tokens, layout.queries)?;
        let runtime = Runtime::shared()?;
        let source = format!(
            "#define PISA_CONTEXT {}\n#define PISA_ROWS {}\n#define PISA_CACHE\n#define PISA_SKIP_IDENTITY_RESCALE\n{}",
            layout.tokens,
            layout.tokens,
            include_str!("fbt_pisa1.metal")
        );
        let attention = runtime.pipeline(
            &source,
            "PISA cached attention",
            "fbt_pisa1_select_attention_q4",
        )?;
        if attention.thread_execution_width() != 32
            || attention.max_total_threads_per_threadgroup() < 128
            || attention.static_threadgroup_memory_length()
                > runtime.device.max_threadgroup_memory_length()
        {
            return Err("cached PISA kernel exceeds GPU threadgroup limits".into());
        }
        let kv = runtime.buffer::<u16>((layout.tokens * 2 * WIDTH) as usize);
        // SAFETY: newly allocated shared memory has no submitted GPU readers.
        unsafe {
            std::ptr::write_bytes(kv.contents().cast::<u8>(), 0, layout.kv_bytes() as usize);
        }
        let tree = runtime.buffer::<u16>((layout.nodes() * WIDTH) as usize);
        // SAFETY: untouched subtrees represent the zero-initialized future KV rows.
        unsafe {
            std::ptr::write_bytes(
                tree.contents().cast::<u8>(),
                0,
                layout.tree_bytes() as usize,
            );
        }
        Ok(Self {
            leaves: runtime.pipeline(
                include_str!("context_tree.metal"),
                "context tree",
                "context_leaves",
            )?,
            parents: runtime.pipeline(
                include_str!("context_tree.metal"),
                "context tree",
                "context_parents",
            )?,
            tree,
            queries: runtime.buffer::<u16>((layout.queries * HEADS * WIDTH) as usize),
            blocks: runtime.buffer::<u32>((layout.queries * SELECTED) as usize),
            output: runtime.buffer::<u16>((layout.queries * HEADS * WIDTH) as usize),
            runtime,
            layout,
            kv,
            attention,
            filled: 0,
        })
    }

    pub fn write(&mut self, start: u32, values: &[u16]) -> Result<f64, String> {
        let rows = u32::try_from(values.len() / (2 * WIDTH) as usize).map_err(|e| e.to_string())?;
        let end = start.checked_add(rows).ok_or("KV write range overflow")?;
        if values.is_empty()
            || values.len() % (2 * WIDTH) as usize != 0
            || start > self.filled
            || end > self.layout.tokens
        {
            return Err("KV write must extend or repair a contiguous in-range prefix".into());
        }
        // SAFETY: all methods finish submitted work before returning; the checked
        // write covers complete KV rows in this owned shared buffer.
        unsafe {
            std::ptr::copy_nonoverlapping(
                values.as_ptr(),
                self.kv
                    .contents()
                    .cast::<u16>()
                    .add((start * 2 * WIDTH) as usize),
                values.len(),
            );
        }
        let command = self.runtime.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        self.rebuild(&encoder, start / BLOCK, end.div_ceil(BLOCK));
        encoder.end_encoding();
        let ms = complete(&command)?;
        self.filled = self.filled.max(end);
        Ok(ms)
    }

    fn rebuild(&self, encoder: &ComputeCommandEncoderRef, mut first: u32, mut end: u32) {
        let range = [first, end - first];
        encoder.set_compute_pipeline_state(&self.leaves);
        encoder.set_buffer(0, Some(&self.kv), 0);
        encoder.set_buffer(1, Some(&self.tree), 0);
        bytes(encoder, 2, &range);
        encoder.dispatch_thread_groups(thread_group(u64::from(end - first)), thread_group(64));
        encoder.memory_barrier_with_resources(&[&self.tree]);
        let (mut child, mut parent, mut count) = (0, self.layout.leaves(), self.layout.leaves());
        while count > 1 {
            first /= 2;
            end = end.div_ceil(2);
            let level = [child, parent, first, end - first];
            encoder.set_compute_pipeline_state(&self.parents);
            encoder.set_buffer(0, Some(&self.tree), 0);
            bytes(encoder, 1, &level);
            encoder.dispatch_thread_groups(thread_group(u64::from(end - first)), thread_group(64));
            encoder.memory_barrier_with_resources(&[&self.tree]);
            child = parent;
            count /= 2;
            parent += count;
        }
    }

    pub fn query(&mut self, start: u32, queries: &[u16]) -> Result<Output, String> {
        let rows =
            u32::try_from(queries.len() / (HEADS * WIDTH) as usize).map_err(|e| e.to_string())?;
        self.layout.range(start, rows)?;
        if queries.len() % (HEADS * WIDTH) as usize != 0 || start + rows > self.filled {
            return Err(
                "queries must cover complete heads within the initialized KV prefix".into(),
            );
        }
        let wall = Instant::now();
        // SAFETY: prior GPU work has completed; the range validation bounds this copy.
        unsafe {
            std::ptr::copy_nonoverlapping(
                queries.as_ptr(),
                self.queries.contents().cast::<u16>(),
                queries.len(),
            );
        }
        let command = self.runtime.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.attention);
        for (index, buffer) in [&self.queries, &self.tree, &self.blocks, &self.output]
            .iter()
            .enumerate()
        {
            encoder.set_buffer(index as u64, Some(buffer), 0);
        }
        bytes(&encoder, 4, &[start, rows]);
        encoder.set_buffer(5, Some(&self.kv), 0);
        encoder.dispatch_thread_groups(thread_group(u64::from(rows / 4)), thread_group(128));
        encoder.end_encoding();
        let device_ms = complete(&command)?;
        // SAFETY: buffers are initialized by the completed dispatch over exactly these rows.
        let (blocks, values) = unsafe {
            (
                std::slice::from_raw_parts(
                    self.blocks.contents().cast::<u32>(),
                    (rows * SELECTED) as usize,
                )
                .to_vec(),
                std::slice::from_raw_parts(self.output.contents().cast::<u16>(), queries.len())
                    .to_vec(),
            )
        };
        Ok(Output {
            blocks,
            values,
            device_ms,
            wall_ms: wall.elapsed().as_secs_f64() * 1000.0,
        })
    }
}

impl crate::context::probe::Cache for ContextCache {
    fn write(&mut self, start: u32, values: &[u16]) -> Result<f64, String> {
        self.write(start, values)
    }
    fn query(&mut self, start: u32, queries: &[u16]) -> Result<Output, String> {
        self.query(start, queries)
    }
    fn tree(&mut self) -> Result<Vec<u16>, String> {
        // SAFETY: each prior write completed all tree dispatches.
        Ok(unsafe {
            std::slice::from_raw_parts(
                self.tree.contents().cast::<u16>(),
                (self.layout.nodes() * WIDTH) as usize,
            )
            .to_vec()
        })
    }
}

fn bytes<T>(encoder: &ComputeCommandEncoderRef, index: u64, value: &T) {
    encoder.set_bytes(
        index,
        std::mem::size_of_val(value) as u64,
        (value as *const T).cast(),
    );
}

fn complete(command: &metal::CommandBufferRef) -> Result<f64, String> {
    command.commit();
    command.wait_until_completed();
    if command.status() != MTLCommandBufferStatus::Completed {
        return Err(format!("context command failed: {:?}", command.status()));
    }
    gpu_seconds(command)
        .map(|seconds| seconds * 1000.0)
        .ok_or("missing context GPU timing".into())
}
