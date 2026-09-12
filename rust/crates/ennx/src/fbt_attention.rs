//! Candidate-scoped append-only KV state and encodable attention primitives.

use crate::apple_gpu::{Runtime, thread_group};
use crate::fbt::{AttentionConfig, CacheKey};
use crate::fbt_metal::{check_buffers, check_command, check_memory, dispatch};
use metal::objc::{
    __send_message as send_message,
    runtime::{Object, Sel},
};
use metal::{
    Buffer, BufferRef, CommandBuffer, CommandBufferRef, ComputePipelineState,
    MTLCommandBufferStatus as Status, MTLSize,
};
use std::sync::{Arc, OnceLock};

const SOURCE: &str = include_str!("fbt_attention.metal");

fn pipeline(runtime: &Runtime, name: &str) -> Result<ComputePipelineState, String> {
    let p = runtime.precise(SOURCE, "FBT attention", name)?;
    if p.thread_execution_width() != 32 || p.max_total_threads_per_threadgroup() < 32 {
        return Err("FBT attention requires 32-lane SIMD groups".into());
    }
    Ok(p)
}

#[repr(C)]
struct NormParams {
    width: u32,
    epsilon: f32,
}

/// Learned, bias-free RMSNorm. Gamma is BF16; inputs and outputs are FP32.
pub struct RmsNorm {
    runtime: Arc<Runtime>,
    params: NormParams,
    pipeline: ComputePipelineState,
}

impl RmsNorm {
    pub fn new(width: u32, epsilon: f32) -> Result<Self, String> {
        if width == 0 || !epsilon.is_finite() || epsilon <= 0.0 {
            return Err("Invalid FBT RMSNorm width or epsilon".into());
        }
        let runtime = Runtime::shared()?;
        let pipeline = pipeline(&runtime, "fbt_rms_affine")?;
        Ok(Self {
            runtime,
            params: NormParams { width, epsilon },
            pipeline,
        })
    }

    /// Same distinct-buffer and serial command ownership rules as Feedback::encode.
    pub fn encode(
        &self,
        command: &CommandBufferRef,
        rows: u32,
        input: &BufferRef,
        gamma: &BufferRef,
        output: &BufferRef,
    ) -> Result<(), String> {
        if rows == 0 {
            return Err("FBT RMSNorm rows must be nonzero".into());
        }
        check_command(command)?;
        let bytes = u64::from(rows)
            .checked_mul(u64::from(self.params.width))
            .and_then(|n| n.checked_mul(4))
            .ok_or("FBT RMSNorm size overflow")?;
        check_buffers(
            &self.runtime,
            &[
                (input, bytes),
                (gamma, u64::from(self.params.width) * 2),
                (output, bytes),
            ],
        )?;
        dispatch(
            command,
            &self.pipeline,
            &[input, gamma, output],
            &self.params,
            thread_group(u64::from(rows)),
        );
        Ok(())
    }
}

#[repr(C)]
struct Params {
    heads: u32,
    kv_heads: u32,
    dim: u32,
    start: u32,
    rows: u32,
    window: u32,
    epsilon: f32,
    rope_base: f32,
    score_scale: f32,
}

/// One layer, one sequence, one candidate's KV cache. Local attention masks keys;
/// storage is currently full-capacity, not a ring, so large prefill chunks cannot
/// overwrite keys still needed by earlier queries in that same chunk.
pub struct Attention {
    runtime: Arc<Runtime>,
    config: AttentionConfig,
    key: CacheKey,
    position: u32,
    max_chunk: u32,
    queries: Buffer,
    keys: Buffer,
    values: Buffer,
    prepare: ComputePipelineState,
    attend: ComputePipelineState,
    tiled: Option<ComputePipelineState>,
    pending: Vec<CommandBuffer>,
}

impl Attention {
    pub fn new(config: AttentionConfig, max_chunk: u32, key: CacheKey) -> Result<Self, String> {
        config.validate()?;
        if max_chunk == 0 || max_chunk > config.capacity {
            return Err("FBT attention chunk capacity is invalid".into());
        }
        let q = u64::from(max_chunk) * u64::from(config.heads) * u64::from(config.head_dim) * 4;
        let kv = u64::from(config.capacity)
            * u64::from(config.kv_heads)
            * u64::from(config.head_dim)
            * 2;
        let runtime = Runtime::shared()?;
        check_memory(&runtime, q + 2 * kv, q.max(kv))?;
        let prepare = pipeline(&runtime, "fbt_prepare_qkv")?;
        let attend = pipeline(&runtime, "fbt_cached_attention")?;
        let tiled = if config.head_dim == 96 {
            let tiled = pipeline(&runtime, "fbt_tiled_attention96")?;
            if tiled.max_total_threads_per_threadgroup() < 128
                || tiled.static_threadgroup_memory_length()
                    > runtime.device.max_threadgroup_memory_length()
            {
                return Err("FBT tiled attention exceeds threadgroup capability".into());
            }
            Some(tiled)
        } else {
            None
        };
        let queries = runtime.buffer::<u8>(q as usize);
        let keys = runtime.buffer::<u8>(kv as usize);
        let values = runtime.buffer::<u8>(kv as usize);
        if [queries.contents(), keys.contents(), values.contents()]
            .iter()
            .any(|p| p.is_null())
        {
            return Err("FBT attention allocation failed".into());
        }
        Ok(Self {
            runtime,
            config,
            key,
            position: 0,
            max_chunk,
            queries,
            keys,
            values,
            prepare,
            attend,
            tiled,
            pending: Vec::new(),
        })
    }

    /// Obtain a command buffer from the cache's ordered queue. It may also hold
    /// projections, RMSNorm, feedback and FFN dispatches before/after attention.
    pub fn command_buffer(&self) -> CommandBuffer {
        self.runtime.queue.new_command_buffer().to_owned()
    }

    /// Scheduled position, not proof of GPU completion. Check completion before
    /// reading outputs; failed commands require reset before subsequent encoding.
    pub fn position(&self) -> u32 {
        self.position
    }

    /// Check EVERY retained command before consuming an evaluation result. A
    /// later successful command must not hide an earlier asynchronous GPU error.
    /// Does not wait. The caller can wait on its final command, then call this.
    pub fn check_completed(&self) -> Result<(), String> {
        for command in &self.pending {
            if command.status() != Status::Completed {
                return Err(format!(
                    "FBT cache command is {:?}, not completed",
                    command.status()
                ));
            }
        }
        Ok(())
    }

    /// Reset logical validity without clearing memory. No stale entries are read
    /// because the new sequence starts at zero and overwrites each valid prefix.
    /// An outstanding or abandoned unsubmitted command forbids reuse: complete
    /// it first, or discard this cache and construct a new one.
    pub fn reset(&mut self, key: CacheKey) -> Result<(), String> {
        if self
            .pending
            .iter()
            .any(|c| !matches!(c.status(), Status::Completed | Status::Error))
        {
            return Err("FBT cache has unfinished commands".into());
        }
        self.key = key;
        self.position = 0;
        self.pending.clear();
        Ok(())
    }

    /// Append a causal chunk. Q is FP32 [rows, heads, dim]; K/V are FP32
    /// [rows, kv_heads, dim]. Gates are already-computed FP32 [rows, heads]
    /// output multipliers, NOT logits. Outputs are FP32 in Q's shape.
    /// The caller must not mutate/read resources before successful completion.
    /// All buffers are distinct; the command must use command_buffer()'s queue.
    /// Only the supplied key is accepted; new candidates/passes require reset.
    /// No submission, wait, host readback or GPU allocation occurs here.
    /// Host-side command tracking retains outstanding commands until completion.
    pub fn encode(
        &mut self,
        command: &CommandBufferRef,
        key: CacheKey,
        rows: u32,
        q: &BufferRef,
        k: &BufferRef,
        v: &BufferRef,
        gates: &BufferRef,
        output: &BufferRef,
    ) -> Result<(), String> {
        self.encode_impl(command, key, rows, q, k, v, gates, output, false)
    }

    /// Explicit tiled prefill path for 96-wide heads, with the same cache,
    /// buffer and command ownership contract as encode. Other widths reject.
    pub fn encode_tiled(
        &mut self,
        command: &CommandBufferRef,
        key: CacheKey,
        rows: u32,
        q: &BufferRef,
        k: &BufferRef,
        v: &BufferRef,
        gates: &BufferRef,
        output: &BufferRef,
    ) -> Result<(), String> {
        self.encode_impl(command, key, rows, q, k, v, gates, output, true)
    }

    fn encode_impl(
        &mut self,
        command: &CommandBufferRef,
        key: CacheKey,
        rows: u32,
        q: &BufferRef,
        k: &BufferRef,
        v: &BufferRef,
        gates: &BufferRef,
        output: &BufferRef,
        tiled: bool,
    ) -> Result<(), String> {
        if tiled && self.tiled.is_none() {
            return Err("FBT tiled attention requires head dimension 96".into());
        }
        check_command(command)?;
        // metal-rs does not expose MTLCommandBuffer.commandQueue. Query the
        // documented Objective-C property to enforce stream ordering here.
        static QUEUE_SELECTOR: OnceLock<Sel> = OnceLock::new();
        let queue: *const metal::CommandQueueRef = unsafe {
            send_message(
                command as *const CommandBufferRef as *const Object,
                *QUEUE_SELECTOR.get_or_init(|| Sel::register("commandQueue")),
                (),
            )
        }
        .map_err(|e| format!("FBT command queue lookup failed: {e}"))?;
        if queue != &*self.runtime.queue as *const metal::CommandQueueRef {
            return Err("FBT cache command uses a different queue".into());
        }
        if key != self.key {
            return Err("FBT cache candidate/sequence/pass mismatch; reset required".into());
        }
        if rows == 0 || rows > self.max_chunk || rows > self.config.capacity - self.position {
            return Err("FBT attention chunk exceeds capacity".into());
        }
        for last in &self.pending {
            if last.status() == Status::Error {
                return Err("FBT cache command failed; reset required".into());
            }
            if !std::ptr::eq(&**last, command)
                && matches!(last.status(), Status::NotEnqueued | Status::Enqueued)
            {
                return Err("FBT cache previous command has not been committed".into());
            }
        }
        let c = self.config;
        let qb = u64::from(rows) * u64::from(c.heads) * u64::from(c.head_dim) * 4;
        let kb = u64::from(rows) * u64::from(c.kv_heads) * u64::from(c.head_dim) * 4;
        check_buffers(
            &self.runtime,
            &[
                (q, qb),
                (k, kb),
                (v, kb),
                (gates, u64::from(rows) * u64::from(c.heads) * 4),
                (output, qb),
            ],
        )?;
        let params = Params {
            heads: c.heads,
            kv_heads: c.kv_heads,
            dim: c.head_dim,
            start: self.position,
            rows,
            window: c.window.unwrap_or(0),
            epsilon: c.qk_norm.epsilon()?,
            rope_base: c.rope_base,
            score_scale: c.score_scale,
        };
        dispatch(
            command,
            &self.prepare,
            &[q, k, v, &self.queries, &self.keys, &self.values],
            &params,
            MTLSize {
                width: u64::from(c.heads) + u64::from(c.kv_heads),
                height: u64::from(rows),
                depth: 1,
            },
        );
        if tiled {
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(self.tiled.as_ref().unwrap());
            for (index, buffer) in [&*self.queries, &*self.keys, &*self.values, gates, output]
                .into_iter()
                .enumerate()
            {
                encoder.set_buffer(index as u64, Some(buffer), 0);
            }
            encoder.set_bytes(
                5,
                size_of::<Params>() as u64,
                (&params as *const Params).cast(),
            );
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: u64::from(c.heads),
                    height: u64::from(rows).div_ceil(8),
                    depth: 1,
                },
                thread_group(128),
            );
            encoder.end_encoding();
        } else {
            dispatch(
                command,
                &self.attend,
                &[&self.queries, &self.keys, &self.values, gates, output],
                &params,
                MTLSize {
                    width: u64::from(c.heads),
                    height: u64::from(rows),
                    depth: 1,
                },
            );
        }
        self.position += rows;
        self.pending.retain(|c| c.status() != Status::Completed);
        if !self
            .pending
            .last()
            .is_some_and(|last| std::ptr::eq(&**last, command))
        {
            self.pending.push(command.to_owned());
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "fbt_attentiontests.rs"]
mod tests;
