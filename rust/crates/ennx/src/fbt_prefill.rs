//! Layer-major teacher-forced scoring. No recurrent/decode cache reuse.

use super::*;
use crate::fbt_mps::{Matmul, Matrix};
use metal::MTLSize;
use std::cell::RefCell;
#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};

const SOURCE: &str = include_str!("fbt_prefill.metal");
const QUERY_BLOCK: u32 = 256;

#[cfg(test)]
#[path = "fbt_boundtests.rs"]
mod boundtests;

#[cfg(test)]
static FUSED_GATE_UP_DISPATCHES: AtomicUsize = AtomicUsize::new(0);

#[cfg(test)]
pub(super) fn reset_fused_gate_up_dispatches() {
    FUSED_GATE_UP_DISPATCHES.store(0, Ordering::Relaxed);
}

#[cfg(test)]
pub(super) fn fused_gate_up_dispatches() -> usize {
    FUSED_GATE_UP_DISPATCHES.load(Ordering::Relaxed)
}

#[derive(Debug)]
pub struct BatchScore {
    /// One mean over all supplied labels per independent example.
    pub mean_nll: Vec<f64>,
    pub tokens_per_example: usize,
    pub passes: u32,
    /// Includes uploads, live-weight widening, submission, waits and readback.
    /// Also includes workspace creation if prepare_prefill was not called first.
    pub elapsed_seconds: f64,
    pub pass_seconds: Vec<f64>,
    pub encode_submit_seconds: f64,
    pub completion_wait_seconds: f64,
    /// One completed command-buffer interval per layer plus pass setup/readout.
    pub gpu_seconds: Vec<Option<f64>>,
}

#[derive(Debug)]
pub struct OperationTiming {
    pub sequence: usize,
    pub domain: &'static str,
    pub pass: Option<u32>,
    pub layer: Option<u32>,
    pub operation: &'static str,
    pub shape: String,
    pub cpu_start_seconds: f64,
    pub cpu_seconds: f64,
    pub gpu_start_seconds: Option<f64>,
    pub gpu_end_seconds: Option<f64>,
    pub gpu_seconds: Option<f64>,
}

#[derive(Debug)]
pub struct BatchTrace {
    pub score: BatchScore,
    pub operations: Vec<OperationTiming>,
    pub preparation_operations: usize,
    pub scorer_operations: usize,
    pub wall_seconds: f64,
    pub gpu_sum_seconds: f64,
    pub gpu_envelope_seconds: f64,
    pub gpu_gap_seconds: f64,
    pub gpu_overlap_seconds: f64,
}

struct TraceRecorder {
    started: std::time::Instant,
    operations: Vec<OperationTiming>,
    commands: Vec<(usize, CommandBuffer)>,
}

impl TraceRecorder {
    fn new() -> Self {
        Self {
            started: std::time::Instant::now(),
            operations: Vec::new(),
            commands: Vec::new(),
        }
    }

    fn host<T>(
        &mut self,
        operation: &'static str,
        shape: String,
        execute: impl FnOnce() -> T,
    ) -> T {
        let start = self.started.elapsed().as_secs_f64();
        let timer = std::time::Instant::now();
        let result = execute();
        self.operations.push(OperationTiming {
            sequence: self.operations.len(),
            domain: "host",
            pass: None,
            layer: None,
            operation,
            shape,
            cpu_start_seconds: start,
            cpu_seconds: timer.elapsed().as_secs_f64(),
            gpu_start_seconds: None,
            gpu_end_seconds: None,
            gpu_seconds: None,
        });
        result
    }

    fn gpu(
        &mut self,
        model: &Model,
        domain: &'static str,
        pass: u32,
        layer: Option<u32>,
        operation: &'static str,
        shape: String,
        encode: impl FnOnce(&CommandBufferRef) -> Result<(), String>,
    ) -> Result<(), String> {
        let start = self.started.elapsed().as_secs_f64();
        let timer = std::time::Instant::now();
        let command = model.runtime.queue.new_command_buffer().to_owned();
        encode(&command)?;
        command.commit();
        let index = self.operations.len();
        self.operations.push(OperationTiming {
            sequence: index,
            domain,
            pass: Some(pass),
            layer,
            operation,
            shape,
            cpu_start_seconds: start,
            cpu_seconds: timer.elapsed().as_secs_f64(),
            gpu_start_seconds: None,
            gpu_end_seconds: None,
            gpu_seconds: None,
        });
        self.commands.push((index, command));
        Ok(())
    }

    fn finish(&mut self) -> Result<(f64, f64, f64, f64, f64), String> {
        let wait = std::time::Instant::now();
        if let Some((_, command)) = self.commands.last() {
            command.wait_until_completed();
        }
        let wait_seconds = wait.elapsed().as_secs_f64();
        let intervals: Vec<_> = self
            .commands
            .iter()
            .map(|(_, command)| {
                if command.status() != MTLCommandBufferStatus::Completed {
                    return Err(format!(
                        "FBT traced operation failed: {:?}",
                        command.status()
                    ));
                }
                crate::apple_gpu::gpu_interval(command)
                    .ok_or_else(|| "Metal did not expose a completed GPU interval".to_string())
            })
            .collect::<Result<_, _>>()?;
        let origin = intervals.first().map(|(start, _)| *start).unwrap_or(0.0);
        let mut gpu_sum = 0.0;
        let mut gpu_gap = 0.0;
        let mut gpu_overlap = 0.0;
        let mut covered_end: Option<f64> = None;
        for ((index, _), (start, end)) in self.commands.iter().zip(&intervals) {
            let duration = end - start;
            if let Some(previous) = covered_end {
                if *start >= previous {
                    gpu_gap += start - previous;
                } else {
                    gpu_overlap += (end.min(previous) - start).max(0.0);
                }
            }
            covered_end = Some(covered_end.map_or(*end, |previous| previous.max(*end)));
            gpu_sum += duration;
            let operation = &mut self.operations[*index];
            operation.gpu_start_seconds = Some(start - origin);
            operation.gpu_end_seconds = Some(end - origin);
            operation.gpu_seconds = Some(duration);
        }
        let gpu_envelope = intervals.last().map(|(_, end)| end - origin).unwrap_or(0.0);
        Ok((wait_seconds, gpu_sum, gpu_envelope, gpu_gap, gpu_overlap))
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Params {
    length: u32,
    heads: u32,
    kv_heads: u32,
    dim: u32,
    start: u32,
    block: u32,
    window: u32,
    key_start: u32,
    key_rows: u32,
    epsilon: f32,
    rope_base: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct NormParams {
    width: u32,
    epsilon: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct FeedbackNormParams {
    width: u32,
    fused_epsilon: f32,
}

pub(super) struct Prefill {
    pub(super) batch: u32,
    length: u32,
    rows: u32,
    matmul: RefCell<Matmul>,
    feedback: RefCell<Feedback>,
    widen: ComputePipelineState,
    prepare: ComputePipelineState,
    softmax: ComputePipelineState,
    unpack: ComputePipelineState,
    flash_attention: ComputePipelineState,
    shift: ComputePipelineState,
    glu: ComputePipelineState,
    weights: Buffer,
    pub(super) tokens: Buffer,
    pub(super) targets: Buffer,
    mask: Buffer,
    pub(super) losses: Buffer,
    pub(super) x: Buffer,
    embed: Buffer,
    normalized: Buffer,
    branch: Buffer,
    previous: Buffer,
    history: Buffer,
    qkv: [Buffer; 3],
    head_major: [Buffer; 4],
    gates: Buffer,
    ff_gate: Buffer,
    ff_up: Buffer,
    scores: Buffer,
    logits: Buffer,
    rms_half: ComputePipelineState,
    residual_half: ComputePipelineState,
    glu_half: ComputePipelineState,
    prepare_half: ComputePipelineState,
    flash_attention_half: ComputePipelineState,
    flash_attention_half_96: ComputePipelineState,
    normalized_half: Buffer,
    embed_half: Buffer,
    branch_half: Buffer,
    qkv_half: [Buffer; 3],
    gates_half: Buffer,
    ff_gate_half: Buffer,
    ff_up_half: Buffer,
    cross_entropy_blocked_half: ComputePipelineState,
    logits_half: Buffer,
    unit_rms_half: ComputePipelineState,
    feedback_combine_norm: ComputePipelineState,
    qkvg_half: Buffer,
    prepare_half_fused: ComputePipelineState,
    ff_gate_up_half: Buffer,
    glu_half_fused: ComputePipelineState,
    history_half: Buffer,
    shift_half: ComputePipelineState,
    gemm_glu_half: ComputePipelineState,
}

impl Model {
    /// Preallocate the opt-in MPS prefill workspace. Does not change score() or
    /// its reference backend. Length is per example; all examples must match.
    pub fn prepare_prefill(&mut self, batch: u32, length: u32) -> Result<(), String> {
        if self
            .prefills
            .first()
            .is_some_and(|p| p.batch == batch && p.length == length && self.prefills.len() == 1)
        {
            return Ok(());
        }

        self.prefills.clear();
        self.prefills.push(Prefill::new(self, batch, length)?);
        self.ensure_transposed_weights()?;
        Ok(())
    }

    pub fn ensure_transposed_weights(&mut self) -> Result<(), String> {
        if self.transposed_revision == self.revision && !self.transposed_weights.is_empty() {
            return Ok(());
        }
        if self.transposed_weights.is_empty() {
            self.transposed_weights = (0..self.parameters.len()).map(|_| None).collect();
            for (idx, p) in self.parameters.iter().enumerate() {
                if p.shape.len() == 2 {
                    let is_qkvg = self
                        .layers
                        .iter()
                        .any(|l| l.q == idx || l.k == idx || l.v == idx || l.head_gate == idx);
                    let is_gate_up = self.layers.iter().any(|l| l.gate == idx || l.up == idx);
                    if is_qkvg && self.config.width == 1536 {
                        continue;
                    }
                    if is_gate_up && (self.config.width == 1536 && self.config.intermediate == 6656)
                    {
                        continue;
                    }
                    let n = p.shape[0];
                    let k = p.shape[1];
                    let elements = u64::from(n) * u64::from(k);
                    let buf = allocate::<u16>(&self.runtime, elements)?;
                    self.transposed_weights[idx] = Some(buf);
                }
            }
        }
        if self.transposed_feedback.is_none() {
            let width = self.config.width;
            let elements = u64::from(width) * u64::from(width);
            let b0 = allocate::<u16>(&self.runtime, elements)?;
            let b1 = allocate::<u16>(&self.runtime, elements)?;
            self.transposed_feedback = Some([b0, b1]);
        }
        let command = self.runtime.queue.new_command_buffer();
        let pipe = self
            .runtime
            .precise(SOURCE, "FBT full prefill", "fbt_transpose_half")?;
        for (idx, p) in self.parameters.iter().enumerate() {
            if let Some(out) = &self.transposed_weights[idx] {
                let (n, k) = (p.shape[0], p.shape[1]);
                let shape = [n, k];
                let encoder = command.new_compute_command_encoder();
                encoder.set_compute_pipeline_state(&pipe);
                encoder.set_buffer(0, Some(&p.buffer), 0);
                encoder.set_buffer(1, Some(out), 0);
                encoder.set_bytes(2, 8, shape.as_ptr().cast());
                let grid = MTLSize {
                    width: u64::from(k).div_ceil(16),
                    height: u64::from(n).div_ceil(16),
                    depth: 1,
                };
                let tg = MTLSize {
                    width: 16,
                    height: 16,
                    depth: 1,
                };
                encoder.dispatch_thread_groups(grid, tg);
                encoder.end_encoding();
            }
        }
        if let Some([b0, b1]) = &self.transposed_feedback {
            let p = &self.parameters[self.feedback_weights];
            let width = self.config.width;
            let shape = [width, width];
            let matrix_bytes = u64::from(width) * u64::from(width) * 2;
            for (out_buf, offset) in [(b0, 0), (b1, matrix_bytes)] {
                let encoder = command.new_compute_command_encoder();
                encoder.set_compute_pipeline_state(&pipe);
                encoder.set_buffer(0, Some(&p.buffer), offset);
                encoder.set_buffer(1, Some(out_buf), 0);
                encoder.set_bytes(2, 8, shape.as_ptr().cast());
                let grid = MTLSize {
                    width: u64::from(width).div_ceil(16),
                    height: u64::from(width).div_ceil(16),
                    depth: 1,
                };
                let tg = MTLSize {
                    width: 16,
                    height: 16,
                    depth: 1,
                };
                encoder.dispatch_thread_groups(grid, tg);
                encoder.end_encoding();
            }
        }
        if self.config.width == 1536 {
            if self.transposed_qkvg.is_empty() {
                let width = self.config.width;
                let elements = u64::from(width) * 3088;
                self.transposed_qkvg = (0..self.config.layers)
                    .map(|_| allocate::<u16>(&self.runtime, elements).ok())
                    .collect();
            }
            let pack_pipe = self
                .runtime
                .precise(SOURCE, "FBT full prefill", "fbt_pack_qkvg")?;
            for (idx, l) in self.layers.iter().enumerate() {
                if let Some(out) = &self.transposed_qkvg[idx] {
                    let encoder = command.new_compute_command_encoder();
                    encoder.set_compute_pipeline_state(&pack_pipe);
                    encoder.set_buffer(0, Some(&self.parameters[l.q].buffer), 0);
                    encoder.set_buffer(1, Some(&self.parameters[l.k].buffer), 0);
                    encoder.set_buffer(2, Some(&self.parameters[l.v].buffer), 0);
                    encoder.set_buffer(3, Some(&self.parameters[l.head_gate].buffer), 0);
                    encoder.set_buffer(4, Some(out), 0);
                    encoder.set_bytes(5, 4, (&self.config.width as *const u32).cast());
                    let grid = MTLSize {
                        width: 3088,
                        height: u64::from(self.config.width),
                        depth: 1,
                    };
                    let tg = MTLSize {
                        width: 32,
                        height: 8,
                        depth: 1,
                    };
                    encoder.dispatch_threads(grid, tg);
                    encoder.end_encoding();
                }
            }
        }
        if self.config.width == 1536 && self.config.intermediate == 6656 {
            if self.transposed_gate_up.is_empty() {
                let width = self.config.width;
                let elements = u64::from(width) * 13312;
                self.transposed_gate_up = (0..self.config.layers)
                    .map(|_| allocate::<u16>(&self.runtime, elements).ok())
                    .collect();
            }
            let pack_pipe = self
                .runtime
                .precise(SOURCE, "FBT full prefill", "fbt_pack_gate_up")?;
            for (idx, l) in self.layers.iter().enumerate() {
                if let Some(out) = &self.transposed_gate_up[idx] {
                    let encoder = command.new_compute_command_encoder();
                    encoder.set_compute_pipeline_state(&pack_pipe);
                    encoder.set_buffer(0, Some(&self.parameters[l.gate].buffer), 0);
                    encoder.set_buffer(1, Some(&self.parameters[l.up].buffer), 0);
                    encoder.set_buffer(2, Some(out), 0);
                    encoder.set_bytes(3, 4, (&self.config.width as *const u32).cast());
                    let grid = MTLSize {
                        width: 13312,
                        height: u64::from(self.config.width),
                        depth: 1,
                    };
                    let tg = MTLSize {
                        width: 32,
                        height: 8,
                        depth: 1,
                    };
                    encoder.dispatch_threads(grid, tg);
                    encoder.end_encoding();
                }
            }
        }
        command.commit();
        command.wait_until_completed();
        if command.status() != MTLCommandBufferStatus::Completed {
            return Err("Failed to transpose weights".into());
        }
        self.transposed_revision = self.revision;
        Ok(())
    }

    fn enqueue_transposed_weights_trace(
        &self,
        trace: &mut TraceRecorder,
    ) -> Result<(usize, bool), String> {
        if self.transposed_revision == self.revision && !self.transposed_weights.is_empty() {
            trace.host(
                "weight_layout_reuse",
                format!("revision={}", self.revision),
                || {},
            );
            return Ok((1, false));
        }
        if self.transposed_weights.len() != self.parameters.len()
            || self.transposed_feedback.is_none()
            || self.transposed_qkvg.len() != self.config.layers as usize
            || self.transposed_gate_up.len() != self.config.layers as usize
        {
            return Err("FBT traced weight layouts were not preallocated".into());
        }

        let start = trace.operations.len();
        let transpose = self
            .runtime
            .precise(SOURCE, "FBT full prefill", "fbt_transpose_half")?;
        for (index, parameter) in self.parameters.iter().enumerate() {
            if let Some(output) = &self.transposed_weights[index] {
                let (n, k) = (parameter.shape[0], parameter.shape[1]);
                let shape = [n, k];
                trace.gpu(
                    self,
                    "metal",
                    0,
                    None,
                    "weight_transpose",
                    format!("parameter={} n={n} k={k}", parameter.name),
                    |command| {
                        let encoder = command.new_compute_command_encoder();
                        encoder.set_compute_pipeline_state(&transpose);
                        encoder.set_buffer(0, Some(&parameter.buffer), 0);
                        encoder.set_buffer(1, Some(output), 0);
                        encoder.set_bytes(2, 8, shape.as_ptr().cast());
                        encoder.dispatch_thread_groups(
                            MTLSize {
                                width: u64::from(k).div_ceil(16),
                                height: u64::from(n).div_ceil(16),
                                depth: 1,
                            },
                            MTLSize {
                                width: 16,
                                height: 16,
                                depth: 1,
                            },
                        );
                        encoder.end_encoding();
                        Ok(())
                    },
                )?;
            }
        }

        let feedback = self.transposed_feedback.as_ref().unwrap();
        let parameter = &self.parameters[self.feedback_weights];
        let width = self.config.width;
        let shape = [width, width];
        let matrix_bytes = u64::from(width) * u64::from(width) * 2;
        for (matrix, (output, offset)) in [
            (feedback[0].as_ref(), 0),
            (feedback[1].as_ref(), matrix_bytes),
        ]
        .into_iter()
        .enumerate()
        {
            trace.gpu(
                self,
                "metal",
                0,
                None,
                "feedback_weight_transpose",
                format!("matrix={matrix} n={width} k={width}"),
                |command| {
                    let encoder = command.new_compute_command_encoder();
                    encoder.set_compute_pipeline_state(&transpose);
                    encoder.set_buffer(0, Some(&parameter.buffer), offset);
                    encoder.set_buffer(1, Some(output), 0);
                    encoder.set_bytes(2, 8, shape.as_ptr().cast());
                    encoder.dispatch_thread_groups(
                        MTLSize {
                            width: u64::from(width).div_ceil(16),
                            height: u64::from(width).div_ceil(16),
                            depth: 1,
                        },
                        MTLSize {
                            width: 16,
                            height: 16,
                            depth: 1,
                        },
                    );
                    encoder.end_encoding();
                    Ok(())
                },
            )?;
        }

        let pack_qkvg = self
            .runtime
            .precise(SOURCE, "FBT full prefill", "fbt_pack_qkvg")?;
        for (index, layer) in self.layers.iter().enumerate() {
            let output = self.transposed_qkvg[index]
                .as_ref()
                .ok_or_else(|| format!("Missing traced QKVG layout for layer {index}"))?;
            trace.gpu(
                self,
                "metal",
                0,
                Some(index as u32),
                "qkvg_weight_pack",
                format!("width={} packed=3088", self.config.width),
                |command| {
                    let encoder = command.new_compute_command_encoder();
                    encoder.set_compute_pipeline_state(&pack_qkvg);
                    encoder.set_buffer(0, Some(&self.parameters[layer.q].buffer), 0);
                    encoder.set_buffer(1, Some(&self.parameters[layer.k].buffer), 0);
                    encoder.set_buffer(2, Some(&self.parameters[layer.v].buffer), 0);
                    encoder.set_buffer(3, Some(&self.parameters[layer.head_gate].buffer), 0);
                    encoder.set_buffer(4, Some(output), 0);
                    encoder.set_bytes(5, 4, (&self.config.width as *const u32).cast());
                    encoder.dispatch_threads(
                        MTLSize {
                            width: 3088,
                            height: u64::from(self.config.width),
                            depth: 1,
                        },
                        MTLSize {
                            width: 32,
                            height: 8,
                            depth: 1,
                        },
                    );
                    encoder.end_encoding();
                    Ok(())
                },
            )?;
        }

        let pack_gate_up = self
            .runtime
            .precise(SOURCE, "FBT full prefill", "fbt_pack_gate_up")?;
        for (index, layer) in self.layers.iter().enumerate() {
            let output = self.transposed_gate_up[index]
                .as_ref()
                .ok_or_else(|| format!("Missing traced gate/up layout for layer {index}"))?;
            trace.gpu(
                self,
                "metal",
                0,
                Some(index as u32),
                "gate_up_weight_pack",
                format!("width={} packed=13312", self.config.width),
                |command| {
                    let encoder = command.new_compute_command_encoder();
                    encoder.set_compute_pipeline_state(&pack_gate_up);
                    encoder.set_buffer(0, Some(&self.parameters[layer.gate].buffer), 0);
                    encoder.set_buffer(1, Some(&self.parameters[layer.up].buffer), 0);
                    encoder.set_buffer(2, Some(output), 0);
                    encoder.set_bytes(3, 4, (&self.config.width as *const u32).cast());
                    encoder.dispatch_threads(
                        MTLSize {
                            width: 13312,
                            height: u64::from(self.config.width),
                            depth: 1,
                        },
                        MTLSize {
                            width: 32,
                            height: 8,
                            depth: 1,
                        },
                    );
                    encoder.end_encoding();
                    Ok(())
                },
            )?;
        }
        Ok((trace.operations.len() - start, true))
    }

    /// Score equal-length independent examples with layer-major batched GEMMs.
    /// Standard and two-pass Fused modes only. Sequential recurrence cannot be
    /// reordered into full-context prefill and is explicitly rejected.
    pub fn score_batch(
        &mut self,
        examples: &[(&[u32], &[u32])],
        mode: ScoreMode,
    ) -> Result<BatchScore, String> {
        let start = std::time::Instant::now();
        if mode == ScoreMode::Sequential || examples.is_empty() {
            return Err("FBT prefill requires a nonempty Standard/Fused batch".into());
        }
        let length = examples[0].0.len();
        if length == 0
            || length > self.config.capacity as usize
            || examples.iter().any(|(x, y)| {
                x.len() != length
                    || y.len() != length
                    || x.iter().chain(*y).any(|&t| t >= self.config.vocab)
            })
        {
            return Err("Invalid FBT prefill IDs or unequal sequence lengths".into());
        }
        let batch = u32::try_from(examples.len()).map_err(|_| "FBT batch exceeds u32")?;
        self.prepare_prefill(batch, length as u32)?;
        self.ensure_transposed_weights()?;

        let mut result = metal::objc::rc::autoreleasepool(|| {
            if self.prefills.len() == 1 && self.prefills[0].batch == batch {
                self.prefills[0].score(self, examples, mode)
            } else {
                let mut combined = self.prefills[0].score(self, &[examples[0]], mode)?;
                for i in 1..batch as usize {
                    let r = self.prefills[i].score(self, &[examples[i]], mode)?;
                    combined.mean_nll.extend_from_slice(&r.mean_nll);
                }
                Ok(combined)
            }
        })?;
        result.elapsed_seconds = start.elapsed().as_secs_f64();
        Ok(result)
    }

    /// Profile each logical operation in the optimized full-sequence scorer.
    /// This deliberately changes command-buffer granularity and is not a timing
    /// substitute for `score_batch`; it exists to attribute that scorer's work.
    pub fn score_batch_traced(
        &mut self,
        examples: &[(&[u32], &[u32])],
        mode: ScoreMode,
    ) -> Result<BatchTrace, String> {
        if mode == ScoreMode::Sequential || examples.is_empty() {
            return Err("FBT traced prefill requires a nonempty Standard/Fused batch".into());
        }
        let length = examples[0].0.len();
        if length == 0
            || length > self.config.capacity as usize
            || examples.iter().any(|(x, y)| {
                x.len() != length
                    || y.len() != length
                    || x.iter().chain(*y).any(|&t| t >= self.config.vocab)
            })
        {
            return Err("Invalid FBT traced prefill IDs or unequal sequence lengths".into());
        }
        if !self.optimized
            || self.config.width != 1536
            || self.config.intermediate != 6656
            || self.config.width / self.config.heads != 96
        {
            return Err("FBT operation tracing supports the optimized LocalV1 scorer only".into());
        }
        let batch = u32::try_from(examples.len()).map_err(|_| "FBT batch exceeds u32")?;
        self.prepare_prefill(batch, length as u32)?;
        let mut trace = TraceRecorder::new();
        let (preparation_operations, refreshed) =
            self.enqueue_transposed_weights_trace(&mut trace)?;
        let result = metal::objc::rc::autoreleasepool(|| {
            self.prefills[0].score_traced(self, examples, mode, trace, preparation_operations)
        })?;
        if refreshed {
            self.transposed_revision = self.revision;
        }
        Ok(result)
    }
}

impl Prefill {
    #[cfg(test)]
    pub(super) fn token_losses(&self) -> Vec<f32> {
        unsafe {
            std::slice::from_raw_parts(self.losses.contents().cast::<f32>(), self.rows as usize)
                .to_vec()
        }
    }

    fn new(model: &Model, batch: u32, length: u32) -> Result<Self, String> {
        let c = model.config;
        let supported_half_attention =
            c.width / c.heads == 128 || (c.width == 1536 && c.heads == 16 && c.kv_heads == 8);
        if model.optimized && !supported_half_attention {
            return Err(
                "Optimized prefill requires 128-wide heads or the 1536-wide 16Q/8KV configuration"
                    .into(),
            );
        }
        if batch == 0 || length == 0 || length > c.capacity {
            return Err("Invalid FBT prefill batch/length".into());
        }
        let rows = batch
            .checked_mul(length)
            .ok_or("FBT prefill row overflow")?;
        let heads = batch
            .checked_mul(c.heads)
            .ok_or("FBT prefill head overflow")?;
        rows.checked_mul(c.width.max(c.intermediate))
            .ok_or("FBT prefill coordinate overflow")?;
        let d = u64::from(c.width);
        let kv = u64::from(c.kv_heads) * (d / u64::from(c.heads));
        let n = u64::from(rows);
        let f = u64::from(c.intermediate);
        let weights = if model.optimized {
            1
        } else {
            d * u64::from(c.vocab).max(f).max(d)
        };
        let attention = if model.optimized && (c.width / c.heads == 128 || c.width / c.heads == 96)
        {
            1
        } else {
            u64::from(heads) * u64::from(length.min(QUERY_BLOCK)) * u64::from(length)
        };
        let partials_elements = u64::from(rows) * u64::from(c.vocab).div_ceil(32) * 4;
        let logits = if model.optimized {
            partials_elements
        } else {
            u64::from(c.chunk.min(rows)) * u64::from(c.vocab)
        };
        // Six graph states, raw QKV, four head-major states, two FFN states;
        // Feedback additionally owns a row scale and one full-width scratch.
        let total = if model.optimized {
            4 * (n * (11 * d + 5) + logits)
        } else {
            4 * (n * (12 * d + 2 * kv + u64::from(c.heads) + 2 * f + 5)
                + weights
                + attention
                + logits)
        };
        let largest = [weights, attention, logits, n * d, n * f, n * kv]
            .into_iter()
            .max()
            .unwrap()
            * 4;
        check_memory(&model.runtime, total, largest)?;
        if !metal::mps::mps_supports_device(&model.runtime.device) {
            return Err("MPS does not support this Metal device".into());
        }
        let alloc = |elements| allocate::<f32>(&model.runtime, elements);
        let pipe = |name| model.runtime.precise(SOURCE, "FBT full prefill", name);
        Ok(Self {
            batch,
            length,
            rows,
            matmul: RefCell::new(Matmul::default()),
            feedback: RefCell::new(Feedback::new(
                FeedbackConfig {
                    width: c.width,
                    token_norm: c.feedback_token_norm,
                    fused_norm: c.feedback_fused_norm,
                },
                rows,
            )?),
            widen: pipe("fbt_widen_weights")?,
            prepare: pipe("fbt_prefill_prepare")?,
            softmax: pipe("fbt_prefill_softmax")?,
            unpack: pipe("fbt_prefill_unpack")?,
            flash_attention: pipe("fbt_prefill_flash_attention")?,
            shift: pipe("fbt_prefill_shift")?,
            glu: pipe("fbt_prefill_glu")?,
            weights: alloc(weights)?,
            tokens: allocate::<u32>(&model.runtime, n)?,
            targets: allocate::<u32>(&model.runtime, n)?,
            mask: allocate::<u32>(&model.runtime, n)?,
            losses: alloc(n)?,
            x: alloc(n * d)?,
            embed: alloc(n * d)?,
            normalized: alloc(n * d)?,
            branch: alloc(n * d)?,
            previous: alloc(n * d)?,
            history: alloc(n * d)?,
            qkv: if model.optimized {
                [alloc(1)?, alloc(1)?, alloc(1)?]
            } else {
                [alloc(n * d)?, alloc(n * kv)?, alloc(n * kv)?]
            },
            head_major: [
                allocate::<u16>(&model.runtime, n * d)?,
                allocate::<u16>(&model.runtime, n * d)?,
                allocate::<u16>(&model.runtime, n * d)?,
                if model.optimized {
                    alloc(1)?
                } else {
                    alloc(n * d)?
                },
            ],
            gates: if model.optimized {
                alloc(1)?
            } else {
                alloc(n * u64::from(c.heads))?
            },
            ff_gate: if model.optimized {
                alloc(1)?
            } else {
                alloc(n * f)?
            },
            ff_up: if model.optimized {
                alloc(1)?
            } else {
                alloc(n * f)?
            },
            scores: alloc(attention)?,
            logits: alloc(logits)?,
            rms_half: pipe("fbt_prefill_rms_half")?,
            residual_half: pipe("fbt_prefill_residual_half")?,
            glu_half: pipe("fbt_prefill_glu_half")?,
            prepare_half: pipe("fbt_prefill_prepare_half")?,
            flash_attention_half: pipe("fbt_prefill_flash_attention_half")?,
            flash_attention_half_96: pipe("fbt_prefill_flash_attention_half_96")?,
            normalized_half: allocate::<u16>(&model.runtime, n * d)?,
            embed_half: allocate::<u16>(&model.runtime, n * d)?,
            branch_half: allocate::<u16>(&model.runtime, n * d)?,
            qkv_half: [
                allocate::<u16>(&model.runtime, n * d)?,
                allocate::<u16>(&model.runtime, n * kv)?,
                allocate::<u16>(&model.runtime, n * kv)?,
            ],
            gates_half: allocate::<u16>(&model.runtime, n * u64::from(c.heads))?,
            ff_gate_half: allocate::<u16>(&model.runtime, n * f)?,
            ff_up_half: allocate::<u16>(&model.runtime, n * f)?,
            cross_entropy_blocked_half: pipe("fbt_cross_entropy_blocked_half")?,
            logits_half: allocate::<u16>(
                &model.runtime,
                u64::from(2048u32.min(rows)) * u64::from(c.vocab),
            )?,
            unit_rms_half: pipe("fbt_prefill_unit_rms_half")?,
            feedback_combine_norm: pipe("fbt_feedback_combine_norm")?,
            qkvg_half: allocate::<u16>(&model.runtime, n * 3088)?,
            prepare_half_fused: pipe("fbt_prefill_prepare_half_fused")?,
            ff_gate_up_half: allocate::<u16>(&model.runtime, n * 2 * f)?,
            glu_half_fused: pipe("fbt_prefill_glu_half_fused")?,
            history_half: allocate::<u16>(&model.runtime, n * d)?,
            shift_half: pipe("fbt_prefill_shift_half")?,
            gemm_glu_half: pipe("fbt_gemm_glu_half")?,
        })
    }

    fn widen(&self, command: &CommandBufferRef, parameter: &Parameter) {
        dispatch(
            command,
            &self.widen,
            &[&parameter.buffer, &self.weights],
            &(parameter.elements as u64),
            thread_group((parameter.elements as u64).div_ceil(32)),
        );
    }

    fn linear(
        &self,
        model: &Model,
        command: &CommandBufferRef,
        parameter: usize,
        input: &BufferRef,
        output: &BufferRef,
    ) -> Result<(), String> {
        let p = &model.parameters[parameter];
        let (n, k) = (p.shape[0], p.shape[1]);
        self.widen(command, p);
        self.matmul.borrow_mut().encode(
            &model.runtime.device,
            command,
            Matrix::new(input, self.rows, k),
            Matrix::new(&self.weights, n, k),
            Matrix::new(output, self.rows, n),
            true,
            1.0,
        )
    }

    fn gemm_glu_half(
        &self,
        command: &CommandBufferRef,
        a: &BufferRef,
        b: &BufferRef,
        c: &BufferRef,
        m: u32,
        n: u32,
        k: u32,
    ) {
        #[cfg(test)]
        FUSED_GATE_UP_DISPATCHES.fetch_add(1, Ordering::Relaxed);
        let p = [m, n, k];
        let grid = MTLSize {
            width: u64::from(n).div_ceil(64),
            height: u64::from(m).div_ceil(64),
            depth: 1,
        };
        let tg = MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        };
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.gemm_glu_half);
        encoder.set_buffer(0, Some(a), 0);
        encoder.set_buffer(1, Some(b), 0);
        encoder.set_buffer(2, Some(c), 0);
        encoder.set_bytes(3, 12, p.as_ptr().cast());
        encoder.dispatch_thread_groups(grid, tg);
        encoder.end_encoding();
    }

    fn linear_half(
        &self,
        model: &Model,
        command: &CommandBufferRef,
        parameter: usize,
        input: &BufferRef,
        output: &BufferRef,
    ) -> Result<(), String> {
        let p = &model.parameters[parameter];
        let (n, k) = (p.shape[0], p.shape[1]);
        let tw = model.transposed_weights[parameter]
            .as_ref()
            .ok_or_else(|| format!("Missing FP16 transpose for parameter {parameter}"))?;
        self.matmul.borrow_mut().encode(
            &model.runtime.device,
            command,
            Matrix::half(input, self.rows, k),
            Matrix::half(tw, k, n),
            Matrix::half(output, self.rows, n),
            false,
            1.0,
        )
    }

    fn rms_half(
        &self,
        command: &CommandBufferRef,
        input: &BufferRef,
        gamma: &BufferRef,
        output: &BufferRef,
        width: u32,
        epsilon: f32,
    ) {
        let p = NormParams { width, epsilon };
        dispatch(
            command,
            &self.rms_half,
            &[input, gamma, output],
            &p,
            thread_group(self.rows as u64),
        );
    }

    fn attention_half(
        &self,
        model: &Model,
        command: &CommandBufferRef,
        layer: u32,
    ) -> Result<(), String> {
        let c = model.config;
        let p = Params {
            length: self.length,
            heads: c.heads,
            kv_heads: c.kv_heads,
            dim: c.width / c.heads,
            start: 0,
            block: 0,
            window: c.attention(layer).window.unwrap_or(0),
            key_start: 0,
            key_rows: 0,
            epsilon: c.epsilon,
            rope_base: c.rope_base,
        };
        let groups = MTLSize {
            width: c.heads as u64,
            height: self.rows as u64,
            depth: 1,
        };
        if model.optimized && c.width == 1536 && p.dim == 96 {
            dispatch(
                command,
                &self.prepare_half_fused,
                &[
                    &self.qkvg_half,
                    &self.head_major[0],
                    &self.head_major[1],
                    &self.head_major[2],
                    &self.gates_half,
                ],
                &p,
                groups,
            );
        } else {
            dispatch(
                command,
                &self.prepare_half,
                &[
                    &self.qkv_half[0],
                    &self.qkv_half[1],
                    &self.qkv_half[2],
                    &self.head_major[0],
                    &self.head_major[1],
                    &self.head_major[2],
                ],
                &p,
                groups,
            );
        }
        let heads = self.batch * c.heads;
        let encoder = command.new_compute_command_encoder();
        let is_fast_96 = p.dim == 96 && model.optimized && c.width == 1536;
        let pipeline = if is_fast_96 {
            &self.flash_attention_half_96
        } else {
            &self.flash_attention_half
        };
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&self.head_major[0]), 0);
        encoder.set_buffer(1, Some(&self.head_major[1]), 0);
        encoder.set_buffer(2, Some(&self.head_major[2]), 0);
        encoder.set_buffer(3, Some(&self.gates_half), 0);
        encoder.set_buffer(4, Some(&self.branch_half), 0);
        encoder.set_bytes(
            5,
            std::mem::size_of::<Params>() as u64,
            (&p as *const Params).cast(),
        );
        let (tile_q, threads) = if is_fast_96 { (32, 128) } else { (16, 64) };
        let grid_size = MTLSize {
            width: u64::from(self.length).div_ceil(tile_q),
            height: u64::from(heads),
            depth: 1,
        };
        let tg_size = MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(grid_size, tg_size);
        encoder.end_encoding();
        Ok(())
    }

    fn attention(
        &self,
        model: &Model,
        command: &CommandBufferRef,
        layer: u32,
    ) -> Result<(), String> {
        let c = model.config;
        let mut p = Params {
            length: self.length,
            heads: c.heads,
            kv_heads: c.kv_heads,
            dim: c.width / c.heads,
            start: 0,
            block: 0,
            window: c.attention(layer).window.unwrap_or(0),
            key_start: 0,
            key_rows: 0,
            epsilon: c.epsilon,
            rope_base: c.rope_base,
        };
        let groups = MTLSize {
            width: c.heads as u64,
            height: self.rows as u64,
            depth: 1,
        };
        dispatch(
            command,
            &self.prepare,
            &[
                &self.qkv[0],
                &self.qkv[1],
                &self.qkv[2],
                &self.head_major[0],
                &self.head_major[1],
                &self.head_major[2],
            ],
            &p,
            groups,
        );
        let heads = self.batch * c.heads;
        if model.optimized && p.dim == 128 {
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.flash_attention);
            encoder.set_buffer(0, Some(&self.head_major[0]), 0);
            encoder.set_buffer(1, Some(&self.head_major[1]), 0);
            encoder.set_buffer(2, Some(&self.head_major[2]), 0);
            encoder.set_buffer(3, Some(&self.gates), 0);
            encoder.set_buffer(4, Some(&self.branch), 0);
            encoder.set_bytes(
                5,
                std::mem::size_of::<Params>() as u64,
                (&p as *const Params).cast(),
            );
            let grid_size = MTLSize {
                width: u64::from(self.length).div_ceil(16),
                height: u64::from(heads),
                depth: 1,
            };
            let tg_size = MTLSize {
                width: 64,
                height: 1,
                depth: 1,
            };
            encoder.dispatch_thread_groups(grid_size, tg_size);
            encoder.end_encoding();
            return Ok(());
        }
        let full_half = |buffer| {
            Matrix::half(buffer, self.length, p.dim).layout(
                heads,
                u64::from(self.length) * u64::from(p.dim),
                0,
            )
        };
        for start in (0..self.length).step_by(QUERY_BLOCK as usize) {
            p.start = start;
            p.block = QUERY_BLOCK.min(self.length - start);
            p.key_start = if p.window == 0 {
                0
            } else {
                (start + 1).saturating_sub(p.window)
            };
            p.key_rows = start + p.block - p.key_start;
            let query =
                |buffer| full_half(buffer).row_view(p.block, u64::from(start) * u64::from(p.dim));
            let scores = Matrix::half(&self.scores, p.block, p.key_rows).layout(
                heads,
                u64::from(p.block) * u64::from(p.key_rows),
                0,
            );
            let keys = |buffer| {
                full_half(buffer).row_view(p.key_rows, u64::from(p.key_start) * u64::from(p.dim))
            };
            self.matmul.borrow_mut().encode(
                &model.runtime.device,
                command,
                query(&self.head_major[0]),
                keys(&self.head_major[1]),
                scores,
                true,
                f64::from(1.0 / (p.dim as f32).sqrt()),
            )?;
            dispatch(
                command,
                &self.softmax,
                &[&self.scores],
                &p,
                MTLSize {
                    width: heads as u64,
                    height: p.block as u64,
                    depth: 1,
                },
            );
            self.matmul.borrow_mut().encode(
                &model.runtime.device,
                command,
                scores,
                keys(&self.head_major[2]),
                query(&self.head_major[3]),
                false,
                1.0,
            )?;
        }
        dispatch(
            command,
            &self.unpack,
            &[&self.head_major[3], &self.gates, &self.branch],
            &p,
            groups,
        );
        Ok(())
    }

    pub(super) fn layer(
        &self,
        model: &Model,
        command: &mut CommandBuffer,
        index: usize,
    ) -> Result<(), String> {
        let c = model.config;
        let l = &model.layers[index];
        let w = |index: usize| model.parameters[index].buffer.as_ref();
        let p = GraphParams {
            width: c.width,
            rows: self.rows,
            start: 0,
            vocab: c.vocab,
            scale: c.residual_scale,
        };
        let groups = thread_group(self.rows as u64);

        if model.optimized && (c.width / c.heads == 128 || c.width / c.heads == 96) {
            self.rms_half(
                command,
                &self.x,
                w(l.norm_attn),
                &self.normalized_half,
                c.width,
                c.epsilon,
            );
            if c.width == 1536 && model.transposed_qkvg.len() > index {
                let qkvg_w = model.transposed_qkvg[index].as_ref().unwrap();
                self.matmul.borrow_mut().encode(
                    &model.runtime.device,
                    command,
                    Matrix::half(&self.normalized_half, self.rows, c.width),
                    Matrix::half(qkvg_w, c.width, 3088),
                    Matrix::half(&self.qkvg_half, self.rows, 3088),
                    false,
                    1.0,
                )?;
            } else {
                self.linear_half(
                    model,
                    command,
                    l.q,
                    &self.normalized_half,
                    &self.qkv_half[0],
                )?;
                self.linear_half(
                    model,
                    command,
                    l.k,
                    &self.normalized_half,
                    &self.qkv_half[1],
                )?;
                self.linear_half(
                    model,
                    command,
                    l.v,
                    &self.normalized_half,
                    &self.qkv_half[2],
                )?;
                self.linear_half(
                    model,
                    command,
                    l.head_gate,
                    &self.normalized_half,
                    &self.gates_half,
                )?;
            }
            self.attention_half(model, command, index as u32)?;
            self.linear_half(model, command, l.out, &self.branch_half, &self.embed_half)?;
            dispatch(
                command,
                &self.residual_half,
                &[&self.embed_half, &self.x],
                &p,
                groups,
            );

            self.rms_half(
                command,
                &self.x,
                w(l.norm_ffn),
                &self.normalized_half,
                c.width,
                c.epsilon,
            );
            if c.width == 1536 && model.transposed_gate_up.len() > index {
                let gate_up_w = model.transposed_gate_up[index].as_ref().unwrap();
                if model.gate_up_implementation == super::GateUpImplementation::FusedMetal {
                    self.gemm_glu_half(
                        command,
                        &self.normalized_half,
                        gate_up_w,
                        &self.ff_up_half,
                        self.rows,
                        c.intermediate,
                        c.width,
                    );
                } else {
                    self.matmul.borrow_mut().encode(
                        &model.runtime.device,
                        command,
                        Matrix::half(&self.normalized_half, self.rows, c.width),
                        Matrix::half(gate_up_w, c.width, 13312),
                        Matrix::half(&self.ff_gate_up_half, self.rows, 13312),
                        false,
                        1.0,
                    )?;
                    dispatch(
                        command,
                        &self.glu_half_fused,
                        &[&self.ff_gate_up_half, &self.ff_up_half],
                        &c.intermediate,
                        groups,
                    );
                }
            } else {
                self.linear_half(
                    model,
                    command,
                    l.gate,
                    &self.normalized_half,
                    &self.ff_gate_half,
                )?;
                self.linear_half(
                    model,
                    command,
                    l.up,
                    &self.normalized_half,
                    &self.ff_up_half,
                )?;
                dispatch(
                    command,
                    &self.glu_half,
                    &[&self.ff_gate_half, &self.ff_up_half],
                    &c.intermediate,
                    groups,
                );
            }
            self.linear_half(model, command, l.down, &self.ff_up_half, &self.embed_half)?;
            dispatch(
                command,
                &self.residual_half,
                &[&self.embed_half, &self.x],
                &p,
                groups,
            );
            return Ok(());
        }

        model.norm.encode(
            command,
            self.rows,
            &self.x,
            w(l.norm_attn),
            &self.normalized,
        )?;
        for (parameter, output) in [
            (l.q, &self.qkv[0]),
            (l.k, &self.qkv[1]),
            (l.v, &self.qkv[2]),
            (l.head_gate, &self.gates),
        ] {
            self.linear(model, command, parameter, &self.normalized, output)?;
        }
        self.attention(model, command, index as u32)?;
        self.linear(model, command, l.out, &self.branch, &self.embed)?;
        dispatch(
            command,
            &model.residual,
            &[&self.embed, &self.x],
            &p,
            groups,
        );
        model
            .norm
            .encode(command, self.rows, &self.x, w(l.norm_ffn), &self.normalized)?;

        self.linear(model, command, l.gate, &self.normalized, &self.ff_gate)?;
        self.linear(model, command, l.up, &self.normalized, &self.ff_up)?;
        dispatch(
            command,
            &self.glu,
            &[&self.ff_gate, &self.ff_up],
            &c.intermediate,
            groups,
        );
        self.linear(model, command, l.down, &self.ff_up, &self.embed)?;

        dispatch(
            command,
            &model.residual,
            &[&self.embed, &self.x],
            &p,
            groups,
        );
        Ok(())
    }

    pub(super) fn start_pass(
        &self,
        model: &Model,
        command: &CommandBufferRef,
        pass: u32,
    ) -> Result<(), String> {
        let c = model.config;
        let p = GraphParams {
            width: c.width,
            rows: self.rows,
            start: 0,
            vocab: c.vocab,
            scale: c.residual_scale,
        };
        let groups = thread_group(self.rows as u64);
        dispatch(
            command,
            &model.lookup,
            &[
                &model.parameters[model.embedding].buffer,
                &self.tokens,
                if pass == 0 { &self.x } else { &self.embed },
            ],
            &p,
            groups,
        );
        if pass > 0 {
            if model.optimized {
                dispatch(
                    command,
                    &self.shift_half,
                    &[&self.history_half, &self.branch_half, &self.mask],
                    &[c.width, self.length],
                    groups,
                );
                let p_norm = NormParams {
                    width: c.width,
                    epsilon: match c.feedback_token_norm {
                        InputNorm::UnitRms { epsilon } => epsilon,
                        _ => 1e-5,
                    },
                };
                dispatch(
                    command,
                    &self.unit_rms_half,
                    &[&self.embed, &self.normalized_half],
                    &p_norm,
                    thread_group(self.rows as u64),
                );
                let [w_state, w_gate] = model.transposed_feedback.as_ref().unwrap();
                self.matmul.borrow_mut().encode(
                    &model.runtime.device,
                    command,
                    Matrix::half(&self.branch_half, self.rows, c.width),
                    Matrix::half(w_state, c.width, c.width),
                    Matrix::half(&self.qkv_half[0], self.rows, c.width),
                    false,
                    1.0,
                )?;
                self.matmul.borrow_mut().encode(
                    &model.runtime.device,
                    command,
                    Matrix::half(&self.normalized_half, self.rows, c.width),
                    Matrix::half(w_gate, c.width, c.width),
                    Matrix::half(&self.embed_half, self.rows, c.width),
                    false,
                    1.0,
                )?;
                let p_feedback = FeedbackNormParams {
                    width: c.width,
                    fused_epsilon: match c.feedback_fused_norm {
                        InputNorm::UnitRms { epsilon } => epsilon,
                        _ => 1e-5,
                    },
                };
                dispatch(
                    command,
                    &self.feedback_combine_norm,
                    &[
                        &self.qkv_half[0],
                        &self.embed_half,
                        &self.embed,
                        &self.mask,
                        &self.x,
                    ],
                    &p_feedback,
                    thread_group(self.rows as u64),
                );
            } else {
                dispatch(
                    command,
                    &self.shift,
                    &[&self.history, &self.previous, &self.mask],
                    &[c.width, self.length],
                    groups,
                );
                self.feedback.borrow_mut().encode(
                    command,
                    self.rows,
                    &model.parameters[model.feedback_weights].buffer,
                    &self.previous,
                    &self.embed,
                    &self.mask,
                    &self.x,
                )?;
            }
        }
        Ok(())
    }

    pub(super) fn finish_pass(
        &self,
        model: &Model,
        command: &CommandBufferRef,
        score: bool,
    ) -> Result<(), String> {
        let c = model.config;
        if model.optimized {
            if !score {
                self.rms_half(
                    command,
                    &self.x,
                    &model.parameters[model.final_norm].buffer,
                    &self.history_half,
                    c.width,
                    c.epsilon,
                );
                return Ok(());
            }
            self.rms_half(
                command,
                &self.x,
                &model.parameters[model.final_norm].buffer,
                &self.normalized_half,
                c.width,
                c.epsilon,
            );
            let embed = model.transposed_weights[model.embedding].as_ref().unwrap();
            let chunk_size = 2048u32.min(self.rows);
            for start in (0..self.rows).step_by(chunk_size as usize) {
                let rows = chunk_size.min(self.rows - start);
                let input = Matrix::half(&self.normalized_half, self.rows, c.width)
                    .row_view(rows, u64::from(start) * u64::from(c.width));
                for column in (0..c.vocab).step_by(8192) {
                    let width = 8192u32.min(c.vocab - column);
                    self.matmul.borrow_mut().encode(
                        &model.runtime.device,
                        command,
                        input,
                        Matrix::half(embed, c.width, c.vocab).column_view(width, column),
                        Matrix::half(&self.logits_half, rows, width)
                            .row_view(rows, u64::from(column) * u64::from(rows)),
                        false,
                        1.0,
                    )?;
                }

                let p = GraphParams {
                    width: c.width,
                    rows,
                    start,
                    vocab: c.vocab,
                    scale: c.residual_scale,
                };
                let encoder = command.new_compute_command_encoder();
                encoder.set_compute_pipeline_state(&self.cross_entropy_blocked_half);
                encoder.set_buffer(0, Some(&self.logits_half), 0);
                encoder.set_buffer(1, Some(&self.targets), 0);
                encoder.set_buffer(2, Some(&self.losses), 0);
                encoder.set_bytes(
                    3,
                    std::mem::size_of::<GraphParams>() as u64,
                    (&p as *const GraphParams).cast(),
                );
                encoder.dispatch_thread_groups(
                    MTLSize {
                        width: rows as u64,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: 256,
                        height: 1,
                        depth: 1,
                    },
                );
                encoder.end_encoding();
            }
            return Ok(());
        }
        model.norm.encode(
            command,
            self.rows,
            &self.x,
            &model.parameters[model.final_norm].buffer,
            &self.normalized,
        )?;
        if !score {
            let blit = command.new_blit_command_encoder();
            blit.copy_from_buffer(
                &self.normalized,
                0,
                &self.history,
                0,
                u64::from(self.rows) * u64::from(c.width) * 4,
            );
            blit.end_encoding();
            return Ok(());
        } else {
            self.widen(command, &model.parameters[model.embedding]);
            for start in (0..self.rows).step_by(c.chunk as usize) {
                let rows = c.chunk.min(self.rows - start);
                let input = Matrix::new(&self.normalized, rows, c.width)
                    .row_view(rows, u64::from(start) * u64::from(c.width));
                self.matmul.borrow_mut().encode(
                    &model.runtime.device,
                    command,
                    input,
                    Matrix::new(&self.weights, c.vocab, c.width),
                    Matrix::new(&self.logits, rows, c.vocab),
                    true,
                    1.0,
                )?;
                let p = GraphParams {
                    width: c.width,
                    rows,
                    start,
                    vocab: c.vocab,
                    scale: c.residual_scale,
                };
                dispatch(
                    command,
                    &model.cross_entropy,
                    &[&self.logits, &self.targets, &self.losses],
                    &p,
                    thread_group(rows as u64),
                );
            }
        }
        Ok(())
    }

    fn score_traced(
        &self,
        model: &Model,
        examples: &[(&[u32], &[u32])],
        mode: ScoreMode,
        mut trace: TraceRecorder,
        preparation_operations: usize,
    ) -> Result<BatchTrace, String> {
        let total = trace.started;
        let scorer_command_start = trace.commands.len();
        trace.host(
            "input_upload",
            format!(
                "batch={} length={} buffers=tokens,targets",
                examples.len(),
                self.length
            ),
            || {
                for (sample, (tokens, targets)) in examples.iter().enumerate() {
                    let offset = sample * self.length as usize;
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            tokens.as_ptr(),
                            self.tokens.contents().cast::<u32>().add(offset),
                            tokens.len(),
                        );
                        std::ptr::copy_nonoverlapping(
                            targets.as_ptr(),
                            self.targets.contents().cast::<u32>().add(offset),
                            targets.len(),
                        );
                    }
                }
            },
        );

        let c = model.config;
        let passes = if mode == ScoreMode::Fused { 2 } else { 1 };
        let rows = self.rows;
        let groups = thread_group(u64::from(rows));
        for pass_index in 0..passes {
            let pass = pass_index + 1;
            let graph = GraphParams {
                width: c.width,
                rows,
                start: 0,
                vocab: c.vocab,
                scale: c.residual_scale,
            };
            trace.gpu(
                model,
                "metal",
                pass,
                None,
                "embedding_lookup",
                format!("rows={rows} width={} vocab={}", c.width, c.vocab),
                |command| {
                    dispatch(
                        command,
                        &model.lookup,
                        &[
                            &model.parameters[model.embedding].buffer,
                            &self.tokens,
                            if pass_index == 0 {
                                &self.x
                            } else {
                                &self.embed
                            },
                        ],
                        &graph,
                        groups,
                    );
                    Ok(())
                },
            )?;
            if pass_index > 0 {
                trace.gpu(
                    model,
                    "metal",
                    pass,
                    None,
                    "feedback_shift",
                    format!("rows={rows} width={} length={}", c.width, self.length),
                    |command| {
                        dispatch(
                            command,
                            &self.shift_half,
                            &[&self.history_half, &self.branch_half, &self.mask],
                            &[c.width, self.length],
                            groups,
                        );
                        Ok(())
                    },
                )?;
                let norm = NormParams {
                    width: c.width,
                    epsilon: match c.feedback_token_norm {
                        InputNorm::UnitRms { epsilon } => epsilon,
                        _ => 1e-5,
                    },
                };
                trace.gpu(
                    model,
                    "metal",
                    pass,
                    None,
                    "feedback_token_rms",
                    format!("rows={rows} width={}", c.width),
                    |command| {
                        dispatch(
                            command,
                            &self.unit_rms_half,
                            &[&self.embed, &self.normalized_half],
                            &norm,
                            groups,
                        );
                        Ok(())
                    },
                )?;
                let [state_weight, gate_weight] = model.transposed_feedback.as_ref().unwrap();
                trace.gpu(
                    model,
                    "mps",
                    pass,
                    None,
                    "feedback_state_gemm",
                    format!("m={rows} n={} k={}", c.width, c.width),
                    |command| {
                        self.matmul.borrow_mut().encode(
                            &model.runtime.device,
                            command,
                            Matrix::half(&self.branch_half, rows, c.width),
                            Matrix::half(state_weight, c.width, c.width),
                            Matrix::half(&self.qkv_half[0], rows, c.width),
                            false,
                            1.0,
                        )
                    },
                )?;
                trace.gpu(
                    model,
                    "mps",
                    pass,
                    None,
                    "feedback_gate_gemm",
                    format!("m={rows} n={} k={}", c.width, c.width),
                    |command| {
                        self.matmul.borrow_mut().encode(
                            &model.runtime.device,
                            command,
                            Matrix::half(&self.normalized_half, rows, c.width),
                            Matrix::half(gate_weight, c.width, c.width),
                            Matrix::half(&self.embed_half, rows, c.width),
                            false,
                            1.0,
                        )
                    },
                )?;
                let feedback = FeedbackNormParams {
                    width: c.width,
                    fused_epsilon: match c.feedback_fused_norm {
                        InputNorm::UnitRms { epsilon } => epsilon,
                        _ => 1e-5,
                    },
                };
                trace.gpu(
                    model,
                    "metal",
                    pass,
                    None,
                    "feedback_combine_norm",
                    format!("rows={rows} width={}", c.width),
                    |command| {
                        dispatch(
                            command,
                            &self.feedback_combine_norm,
                            &[
                                &self.qkv_half[0],
                                &self.embed_half,
                                &self.embed,
                                &self.mask,
                                &self.x,
                            ],
                            &feedback,
                            groups,
                        );
                        Ok(())
                    },
                )?;
            }

            for index in 0..c.layers as usize {
                let layer = &model.layers[index];
                let layer_number = index as u32;
                let norm_attn = &model.parameters[layer.norm_attn].buffer;
                trace.gpu(
                    model,
                    "metal",
                    pass,
                    Some(layer_number),
                    "attention_rms",
                    format!("rows={rows} width={}", c.width),
                    |command| {
                        self.rms_half(
                            command,
                            &self.x,
                            norm_attn,
                            &self.normalized_half,
                            c.width,
                            c.epsilon,
                        );
                        Ok(())
                    },
                )?;
                let qkvg = model.transposed_qkvg[index].as_ref().unwrap();
                trace.gpu(
                    model,
                    "mps",
                    pass,
                    Some(layer_number),
                    "qkvg_gemm",
                    format!("m={rows} n=3088 k={}", c.width),
                    |command| {
                        self.matmul.borrow_mut().encode(
                            &model.runtime.device,
                            command,
                            Matrix::half(&self.normalized_half, rows, c.width),
                            Matrix::half(qkvg, c.width, 3088),
                            Matrix::half(&self.qkvg_half, rows, 3088),
                            false,
                            1.0,
                        )
                    },
                )?;
                let attention = Params {
                    length: self.length,
                    heads: c.heads,
                    kv_heads: c.kv_heads,
                    dim: c.width / c.heads,
                    start: 0,
                    block: 0,
                    window: c.attention(layer_number).window.unwrap_or(0),
                    key_start: 0,
                    key_rows: 0,
                    epsilon: c.epsilon,
                    rope_base: c.rope_base,
                };
                let attention_groups = MTLSize {
                    width: u64::from(c.heads),
                    height: u64::from(rows),
                    depth: 1,
                };
                trace.gpu(
                    model,
                    "metal",
                    pass,
                    Some(layer_number),
                    "attention_prepare",
                    format!(
                        "batch={} length={} heads={} kv_heads={} dim={}",
                        self.batch, self.length, c.heads, c.kv_heads, attention.dim
                    ),
                    |command| {
                        dispatch(
                            command,
                            &self.prepare_half_fused,
                            &[
                                &self.qkvg_half,
                                &self.head_major[0],
                                &self.head_major[1],
                                &self.head_major[2],
                                &self.gates_half,
                            ],
                            &attention,
                            attention_groups,
                        );
                        Ok(())
                    },
                )?;
                trace.gpu(
                    model,
                    "metal",
                    pass,
                    Some(layer_number),
                    "flash_attention",
                    format!(
                        "batch={} length={} heads={} kv_heads={} dim={} window={}",
                        self.batch,
                        self.length,
                        c.heads,
                        c.kv_heads,
                        attention.dim,
                        attention.window
                    ),
                    |command| {
                        let encoder = command.new_compute_command_encoder();
                        encoder.set_compute_pipeline_state(&self.flash_attention_half_96);
                        encoder.set_buffer(0, Some(&self.head_major[0]), 0);
                        encoder.set_buffer(1, Some(&self.head_major[1]), 0);
                        encoder.set_buffer(2, Some(&self.head_major[2]), 0);
                        encoder.set_buffer(3, Some(&self.gates_half), 0);
                        encoder.set_buffer(4, Some(&self.branch_half), 0);
                        encoder.set_bytes(
                            5,
                            std::mem::size_of::<Params>() as u64,
                            (&attention as *const Params).cast(),
                        );
                        encoder.dispatch_thread_groups(
                            MTLSize {
                                width: u64::from(self.length).div_ceil(32),
                                height: u64::from(self.batch * c.heads),
                                depth: 1,
                            },
                            MTLSize {
                                width: 128,
                                height: 1,
                                depth: 1,
                            },
                        );
                        encoder.end_encoding();
                        Ok(())
                    },
                )?;
                trace.gpu(
                    model,
                    "mps",
                    pass,
                    Some(layer_number),
                    "attention_output_gemm",
                    format!("m={rows} n={} k={}", c.width, c.width),
                    |command| {
                        self.linear_half(
                            model,
                            command,
                            layer.out,
                            &self.branch_half,
                            &self.embed_half,
                        )
                    },
                )?;
                trace.gpu(
                    model,
                    "metal",
                    pass,
                    Some(layer_number),
                    "attention_residual",
                    format!("rows={rows} width={}", c.width),
                    |command| {
                        dispatch(
                            command,
                            &self.residual_half,
                            &[&self.embed_half, &self.x],
                            &graph,
                            groups,
                        );
                        Ok(())
                    },
                )?;
                let norm_ffn = &model.parameters[layer.norm_ffn].buffer;
                trace.gpu(
                    model,
                    "metal",
                    pass,
                    Some(layer_number),
                    "ffn_rms",
                    format!("rows={rows} width={}", c.width),
                    |command| {
                        self.rms_half(
                            command,
                            &self.x,
                            norm_ffn,
                            &self.normalized_half,
                            c.width,
                            c.epsilon,
                        );
                        Ok(())
                    },
                )?;

                let gate_up = model.transposed_gate_up[index].as_ref().unwrap();
                if model.gate_up_implementation == super::GateUpImplementation::FusedMetal {
                    trace.gpu(
                        model,
                        "metal",
                        pass,
                        Some(layer_number),
                        "gate_up_glu",
                        format!(
                            "m={rows} n={} k={} implementation=fused_metal",
                            c.intermediate, c.width
                        ),
                        |command| {
                            self.gemm_glu_half(
                                command,
                                &self.normalized_half,
                                gate_up,
                                &self.ff_up_half,
                                rows,
                                c.intermediate,
                                c.width,
                            );
                            Ok(())
                        },
                    )?;
                } else {
                    trace.gpu(
                        model,
                        "mps",
                        pass,
                        Some(layer_number),
                        "gate_up_gemm",
                        format!("m={rows} n=13312 k={}", c.width),
                        |command| {
                            self.matmul.borrow_mut().encode(
                                &model.runtime.device,
                                command,
                                Matrix::half(&self.normalized_half, rows, c.width),
                                Matrix::half(gate_up, c.width, 13312),
                                Matrix::half(&self.ff_gate_up_half, rows, 13312),
                                false,
                                1.0,
                            )
                        },
                    )?;
                    trace.gpu(
                        model,
                        "metal",
                        pass,
                        Some(layer_number),
                        "glu",
                        format!("rows={rows} intermediate={}", c.intermediate),
                        |command| {
                            dispatch(
                                command,
                                &self.glu_half_fused,
                                &[&self.ff_gate_up_half, &self.ff_up_half],
                                &c.intermediate,
                                groups,
                            );
                            Ok(())
                        },
                    )?;
                }

                trace.gpu(
                    model,
                    "mps",
                    pass,
                    Some(layer_number),
                    "down_gemm",
                    format!("m={rows} n={} k={}", c.width, c.intermediate),
                    |command| {
                        self.linear_half(
                            model,
                            command,
                            layer.down,
                            &self.ff_up_half,
                            &self.embed_half,
                        )
                    },
                )?;
                trace.gpu(
                    model,
                    "metal",
                    pass,
                    Some(layer_number),
                    "ffn_residual",
                    format!("rows={rows} width={}", c.width),
                    |command| {
                        dispatch(
                            command,
                            &self.residual_half,
                            &[&self.embed_half, &self.x],
                            &graph,
                            groups,
                        );
                        Ok(())
                    },
                )?;
            }

            let score = pass == passes;
            trace.gpu(
                model,
                "metal",
                pass,
                None,
                "final_rms",
                format!(
                    "rows={rows} width={} output={}",
                    c.width,
                    if score { "readout" } else { "history" }
                ),
                |command| {
                    self.rms_half(
                        command,
                        &self.x,
                        &model.parameters[model.final_norm].buffer,
                        if score {
                            &self.normalized_half
                        } else {
                            &self.history_half
                        },
                        c.width,
                        c.epsilon,
                    );
                    Ok(())
                },
            )?;
            if score {
                let embedding = model.transposed_weights[model.embedding].as_ref().unwrap();
                let chunk_size = 2048u32.min(rows);
                for start in (0..rows).step_by(chunk_size as usize) {
                    let chunk_rows = chunk_size.min(rows - start);
                    for column in (0..c.vocab).step_by(8192) {
                        let width = 8192u32.min(c.vocab - column);
                        trace.gpu(
                            model,
                            "mps",
                            pass,
                            None,
                            "readout_gemm",
                            format!(
                                "start={start} column={column} m={chunk_rows} n={width} k={}",
                                c.width
                            ),
                            |command| {
                                let input = Matrix::half(&self.normalized_half, rows, c.width)
                                    .row_view(chunk_rows, u64::from(start) * u64::from(c.width));
                                self.matmul.borrow_mut().encode(
                                    &model.runtime.device,
                                    command,
                                    input,
                                    Matrix::half(embedding, c.width, c.vocab)
                                        .column_view(width, column),
                                    Matrix::half(&self.logits_half, chunk_rows, width).row_view(
                                        chunk_rows,
                                        u64::from(column) * u64::from(chunk_rows),
                                    ),
                                    false,
                                    1.0,
                                )
                            },
                        )?;
                    }
                    let params = GraphParams {
                        width: c.width,
                        rows: chunk_rows,
                        start,
                        vocab: c.vocab,
                        scale: c.residual_scale,
                    };
                    trace.gpu(
                        model,
                        "metal",
                        pass,
                        None,
                        "cross_entropy",
                        format!("start={start} rows={chunk_rows} vocab={}", c.vocab),
                        |command| {
                            let encoder = command.new_compute_command_encoder();
                            encoder.set_compute_pipeline_state(&self.cross_entropy_blocked_half);
                            encoder.set_buffer(0, Some(&self.logits_half), 0);
                            encoder.set_buffer(1, Some(&self.targets), 0);
                            encoder.set_buffer(2, Some(&self.losses), 0);
                            encoder.set_bytes(
                                3,
                                std::mem::size_of::<GraphParams>() as u64,
                                (&params as *const GraphParams).cast(),
                            );
                            encoder.dispatch_thread_groups(
                                MTLSize {
                                    width: u64::from(chunk_rows),
                                    height: 1,
                                    depth: 1,
                                },
                                MTLSize {
                                    width: 256,
                                    height: 1,
                                    depth: 1,
                                },
                            );
                            encoder.end_encoding();
                            Ok(())
                        },
                    )?;
                }
            }
        }

        let gpu_operations = trace.commands.len() - scorer_command_start;
        let layer_operations =
            if model.gate_up_implementation == super::GateUpImplementation::FusedMetal {
                10
            } else {
                11
            };
        let readout_chunks = self.rows.div_ceil(2048u32.min(self.rows)) as usize;
        let readout_blocks = c.vocab.div_ceil(8192) as usize;
        let readout_operations = readout_chunks * (readout_blocks + 1);
        let expected = if passes == 2 {
            4 + c.layers as usize * layer_operations * 2 + 5 + readout_operations
        } else {
            2 + c.layers as usize * layer_operations + readout_operations
        };
        if gpu_operations != expected {
            return Err(format!(
                "Incomplete FBT operation trace: expected {expected}, recorded {gpu_operations}"
            ));
        }
        let encode_submit_seconds = trace
            .operations
            .iter()
            .filter(|operation| operation.domain != "host")
            .map(|operation| operation.cpu_seconds)
            .sum();
        let (
            completion_wait_seconds,
            gpu_sum_seconds,
            gpu_envelope_seconds,
            gpu_gap_seconds,
            gpu_overlap_seconds,
        ) = trace.finish()?;
        let mean_nll = trace.host(
            "loss_reduce",
            format!("batch={} rows={rows}", examples.len()),
            || {
                let losses = unsafe {
                    std::slice::from_raw_parts(
                        self.losses.contents().cast::<f32>(),
                        self.rows as usize,
                    )
                };
                if losses.iter().any(|value| !value.is_finite()) {
                    return Err("Non-finite FBT traced prefill score".to_string());
                }
                Ok(losses
                    .chunks_exact(self.length as usize)
                    .map(|chunk| {
                        chunk.iter().map(|&value| f64::from(value)).sum::<f64>()
                            / f64::from(self.length)
                    })
                    .collect::<Vec<_>>())
            },
        )?;
        let pass_seconds = (1..=passes)
            .map(|pass| {
                trace
                    .operations
                    .iter()
                    .filter(|operation| operation.pass == Some(pass))
                    .filter_map(|operation| operation.gpu_seconds)
                    .sum()
            })
            .collect();
        let gpu_seconds = trace
            .operations
            .iter()
            .filter(|operation| operation.domain != "host")
            .map(|operation| operation.gpu_seconds)
            .collect();
        let wall_seconds = total.elapsed().as_secs_f64();
        Ok(BatchTrace {
            score: BatchScore {
                mean_nll,
                tokens_per_example: self.length as usize,
                passes,
                elapsed_seconds: wall_seconds,
                pass_seconds,
                encode_submit_seconds,
                completion_wait_seconds,
                gpu_seconds,
            },
            operations: trace.operations,
            preparation_operations,
            scorer_operations: gpu_operations + 2,
            wall_seconds,
            gpu_sum_seconds,
            gpu_envelope_seconds,
            gpu_gap_seconds,
            gpu_overlap_seconds,
        })
    }

    fn score(
        &self,
        model: &Model,
        examples: &[(&[u32], &[u32])],
        mode: ScoreMode,
    ) -> Result<BatchScore, String> {
        for (sample, (tokens, targets)) in examples.iter().enumerate() {
            let offset = sample * self.length as usize;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    tokens.as_ptr(),
                    self.tokens.contents().cast::<u32>().add(offset),
                    tokens.len(),
                );
                std::ptr::copy_nonoverlapping(
                    targets.as_ptr(),
                    self.targets.contents().cast::<u32>().add(offset),
                    targets.len(),
                );
            }
        }
        let passes = if mode == ScoreMode::Fused { 2 } else { 1 };
        let mut result = BatchScore {
            mean_nll: Vec::new(),
            tokens_per_example: self.length as usize,
            passes,
            elapsed_seconds: 0.0,
            pass_seconds: Vec::new(),
            encode_submit_seconds: 0.0,
            completion_wait_seconds: 0.0,
            gpu_seconds: Vec::new(),
        };
        let mut all_commands = Vec::new();
        let total_start = std::time::Instant::now();
        for pass in 0..passes {
            let mut command = model.runtime.queue.new_command_buffer().to_owned();
            for stage in 0..model.config.layers + 2 {
                let outcome = if stage == 0 {
                    self.start_pass(model, &command, pass)
                } else if stage <= model.config.layers {
                    self.layer(model, &mut command, stage as usize - 1)
                } else {
                    self.finish_pass(model, &command, pass + 1 == passes)
                };
                if let Err(e) = outcome {
                    return Err(e);
                }
            }
            command.commit();
            all_commands.push(command);
        }
        result.encode_submit_seconds = total_start.elapsed().as_secs_f64();
        let wait = std::time::Instant::now();
        if let Some(last) = all_commands.last() {
            last.wait_until_completed();
        }
        result.completion_wait_seconds = wait.elapsed().as_secs_f64();
        for command in &all_commands {
            if command.status() != MTLCommandBufferStatus::Completed {
                return Err(format!(
                    "FBT prefill GPU command failed: {:?}",
                    command.status()
                ));
            }
            let secs = gpu_seconds(command);
            result.gpu_seconds.push(secs);
            if let Some(s) = secs {
                result.pass_seconds.push(s);
            }
        }
        let losses = unsafe {
            std::slice::from_raw_parts(self.losses.contents().cast::<f32>(), self.rows as usize)
        };
        if losses.iter().any(|v| !v.is_finite()) {
            return Err("Non-finite FBT prefill score".into());
        }
        result.mean_nll = losses
            .chunks_exact(self.length as usize)
            .map(|chunk| chunk.iter().map(|&v| f64::from(v)).sum::<f64>() / f64::from(self.length))
            .collect();
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode_half(bits: u16) -> f64 {
        let exponent = (bits >> 10) & 31;
        let fraction = f64::from(bits & 1023);
        let magnitude = match exponent {
            0 => fraction * 2.0f64.powi(-24),
            31 => f64::NAN,
            _ => (1024.0 + fraction) * 2.0f64.powi(i32::from(exponent) - 25),
        };
        if bits & 0x8000 == 0 {
            magnitude
        } else {
            -magnitude
        }
    }

    #[test]
    #[ignore = "GPU diagnostic; exercised by tools/fbt-bo --check"]
    fn gpu_scorer_attention96() {
        metal::objc::rc::autoreleasepool(|| {
            let runtime = crate::apple_gpu::Runtime::shared().unwrap();
            let pipeline = runtime
                .precise(
                    SOURCE,
                    "FBT full prefill",
                    "fbt_prefill_flash_attention_half_96",
                )
                .unwrap();
            let values = |count: usize, seed: u32| -> Vec<u16> {
                let choices = [
                    0x0000, 0x3400, 0xb400, 0x3800, 0xb800, 0x3c00, 0xbc00, 0x4000, 0xc000,
                ];
                (0..count)
                    .map(|i| {
                        let hash = (i as u32).wrapping_mul(2654435761) ^ ((i as u32) >> 7) ^ seed;
                        choices[hash as usize % choices.len()]
                    })
                    .collect()
            };
            let (batch, heads, kv_heads, dim) = (2usize, 4usize, 2usize, 96usize);
            for length in [1usize, 31, 32, 33, 65, 4096] {
                let q = values(batch * heads * length * dim, 17);
                let k = values(batch * kv_heads * length * dim, 123);
                let v = values(batch * kv_heads * length * dim, 456);
                let gates = values(batch * length * heads, 789);
                let buffers = [&q, &k, &v, &gates].map(|x| runtime.buffer_with(x));
                let [q, k, v, gates] = [&q, &k, &v, &gates]
                    .map(|x| x.iter().map(|&bits| decode_half(bits)).collect::<Vec<_>>());
                let output_len = batch * length * heads * dim;
                let windows: &[usize] = if length == 4096 {
                    &[0, 2048]
                } else {
                    &[0, 1, 17, 32]
                };
                for &window in windows {
                    let output = runtime.buffer_with(&vec![0x7e00u16; output_len + 64]);
                    let p = Params {
                        length: length as u32,
                        heads: heads as u32,
                        kv_heads: kv_heads as u32,
                        dim: dim as u32,
                        start: 0,
                        block: 0,
                        window: window as u32,
                        key_start: 0,
                        key_rows: 0,
                        epsilon: 1e-5,
                        rope_base: 10000.0,
                    };
                    let command = runtime.queue.new_command_buffer();
                    let encoder = command.new_compute_command_encoder();
                    encoder.set_compute_pipeline_state(&pipeline);
                    for (index, buffer) in buffers.iter().enumerate() {
                        encoder.set_buffer(index as u64, Some(buffer), 0);
                    }
                    encoder.set_buffer(4, Some(&output), 0);
                    encoder.set_bytes(
                        5,
                        std::mem::size_of_val(&p) as u64,
                        (&p as *const Params).cast(),
                    );
                    encoder.dispatch_thread_groups(
                        MTLSize {
                            width: length.div_ceil(32) as u64,
                            height: (batch * heads) as u64,
                            depth: 1,
                        },
                        MTLSize {
                            width: 128,
                            height: 1,
                            depth: 1,
                        },
                    );
                    encoder.end_encoding();
                    command.commit();
                    command.wait_until_completed();
                    assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
                    let actual = unsafe {
                        std::slice::from_raw_parts(output.contents().cast::<u16>(), output_len + 64)
                    };
                    assert!(actual[output_len..].iter().all(|&bits| bits == 0x7e00));
                    let positions: Vec<usize> = if length == 4096 {
                        vec![0, 31, 32, 2047, 2048, 4095]
                    } else {
                        (0..length).collect()
                    };
                    let mut max_error = 0.0f64;
                    for sample in 0..batch {
                        for head in 0..heads {
                            let kv_head = head / (heads / kv_heads);
                            for &pos in &positions {
                                let lower = if window == 0 {
                                    0
                                } else {
                                    (pos + 1).saturating_sub(window)
                                };
                                let qi = ((sample * heads + head) * length + pos) * dim;
                                let ki = (sample * kv_heads + kv_head) * length * dim;
                                let scores: Vec<f64> = (lower..=pos)
                                    .map(|key| {
                                        (0..dim)
                                            .map(|d| q[qi + d] * k[ki + key * dim + d])
                                            .sum::<f64>()
                                            / (dim as f64).sqrt()
                                    })
                                    .collect();
                                let maximum =
                                    scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                                let weights: Vec<f64> =
                                    scores.iter().map(|s| (s - maximum).exp()).collect();
                                let denominator: f64 = weights.iter().sum();
                                let row = (sample * length + pos) * heads + head;
                                let gate = 1.0 / (1.0 + (-gates[row]).exp());
                                for d in 0..dim {
                                    let expected = weights
                                        .iter()
                                        .enumerate()
                                        .map(|(key, weight)| {
                                            weight * v[ki + (lower + key) * dim + d]
                                        })
                                        .sum::<f64>()
                                        / denominator
                                        * gate;
                                    let actual = decode_half(actual[row * dim + d]);
                                    let error = (actual - expected).abs();
                                    assert!(
                                        actual.is_finite() && error < 0.003,
                                        "length={length} window={window} sample={sample} head={head} pos={pos} d={d} actual={actual} expected={expected}"
                                    );
                                    max_error = max_error.max(error);
                                }
                            }
                        }
                    }
                    eprintln!(
                        "FBT_ATTENTION_CHECK length={length} window={window} max_error={max_error:.9} gpu_seconds={:.6}",
                        gpu_seconds(command).unwrap()
                    );
                }
            }
        });
    }

    #[test]
    fn cache_parity() {
        metal::objc::rc::autoreleasepool(|| {
            let c = ModelConfig {
                width: 192,
                intermediate: 129,
                layers: 2,
                vocab: 97,
                heads: 2,
                kv_heads: 1,
                capacity: 257,
                chunk: 17,
                local_window: 3,
                full_every: 2,
                epsilon: 1e-5,
                rope_base: 10000.0,
                residual_scale: 0.5,
                feedback_token_norm: InputNorm::None,
                feedback_fused_norm: InputNorm::None,
                tiled_attention: true,
            };
            let model = Model::new(c, 42).unwrap();
            let p = Prefill::new(&model, 2, 257).unwrap();
            let mut inputs = Vec::new();
            for (index, buffer) in p.qkv.iter().enumerate() {
                let count = (buffer.length() / 4) as usize;
                let values: Vec<f32> = (0..count)
                    .map(|i| ((i * 7 + index * 13) % 113) as f32 * 0.01 - 0.5)
                    .collect();
                unsafe {
                    std::ptr::copy_nonoverlapping(values.as_ptr(), buffer.contents().cast(), count);
                }
                inputs.push(values);
            }
            unsafe {
                std::ptr::write_bytes(p.gates.contents(), 0, p.gates.length() as usize);
            }
            for layer in [0, 1] {
                let command = model.runtime.queue.new_command_buffer();
                p.attention(&model, command, layer).unwrap();
                command.commit();
                command.wait_until_completed();
                assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
                let actual = unsafe {
                    std::slice::from_raw_parts(p.branch.contents().cast::<f32>(), 514 * 192)
                };
                for sample in 0..2 {
                    let key = CacheKey {
                        candidate: 0,
                        sequence: sample as u64,
                        pass: 0,
                    };
                    let mut reference = Attention::new(c.attention(layer), 257, key).unwrap();
                    let input: Vec<_> = inputs
                        .iter()
                        .map(|v| {
                            let n = v.len() / 2;
                            model.runtime.buffer_with(&v[sample * n..(sample + 1) * n])
                        })
                        .collect();
                    let gates = model.runtime.buffer_with(&vec![0.5f32; 257 * 2]);
                    let output = model.runtime.buffer::<f32>(257 * 192);
                    let command = reference.command_buffer();
                    reference
                        .encode(
                            &command, key, 257, &input[0], &input[1], &input[2], &gates, &output,
                        )
                        .unwrap();
                    command.commit();
                    command.wait_until_completed();
                    reference.check_completed().unwrap();
                    let expected = unsafe {
                        std::slice::from_raw_parts(output.contents().cast::<f32>(), 257 * 192)
                    };
                    let mut max_diff = 0.0f32;
                    for (&a, &b) in actual[sample * 257 * 192..(sample + 1) * 257 * 192]
                        .iter()
                        .zip(expected)
                    {
                        max_diff = max_diff.max((a - b).abs());
                    }
                    eprintln!("cache_parity sample={sample} layer={layer} max_diff={max_diff}");
                    assert!(
                        max_diff < 0.003,
                        "layer={layer} sample={sample} max_diff={max_diff}"
                    );
                }
            }
        });
    }
}
