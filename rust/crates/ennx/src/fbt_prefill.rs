//! Layer-major teacher-forced scoring. No recurrent/decode cache reuse.

pub(super) use super::*;
pub(super) use crate::fbt_mps::{Matmul, Matrix};
pub(super) use metal::{CommandBuffer, MTLCommandBufferStatus, MTLSize};
pub(super) use std::cell::RefCell;
#[cfg(test)]
pub(super) use std::sync::atomic::{AtomicUsize, Ordering};

pub(super) const SOURCE: &str = include_str!("fbt_prefill.metal");
pub(super) const QUERY_BLOCK: u32 = 256;

#[cfg(test)]
#[path = "fbt_boundtests.rs"]
mod boundtests;

#[cfg(test)]
pub(super) static FUSED_DISPATCHES: AtomicUsize = AtomicUsize::new(0);

#[cfg(test)]
pub(super) fn reset_dispatches() {
    FUSED_DISPATCHES.store(0, Ordering::Relaxed);
}

#[cfg(test)]
pub(super) fn fused_dispatches() -> usize {
    FUSED_DISPATCHES.load(Ordering::Relaxed)
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

pub(super) struct TraceRecorder {
    pub(super) started: std::time::Instant,
    pub(super) operations: Vec<OperationTiming>,
    pub(super) commands: Vec<(usize, CommandBuffer)>,
}

impl TraceRecorder {
    pub(super) fn new() -> Self {
        Self {
            started: std::time::Instant::now(),
            operations: Vec::new(),
            commands: Vec::new(),
        }
    }

    pub(super) fn host<T>(
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

    pub(super) fn gpu(
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

    pub(super) fn finish(&mut self) -> Result<(f64, f64, f64, f64, f64), String> {
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
pub(super) struct Params {
    pub(super) length: u32,
    pub(super) heads: u32,
    pub(super) kv_heads: u32,
    pub(super) dim: u32,
    pub(super) start: u32,
    pub(super) block: u32,
    pub(super) window: u32,
    pub(super) key_start: u32,
    pub(super) key_rows: u32,
    pub(super) epsilon: f32,
    pub(super) rope_base: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct NormParams {
    pub(super) width: u32,
    pub(super) epsilon: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct FeedbackNormParams {
    pub(super) width: u32,
    pub(super) fused_epsilon: f32,
}

pub(super) struct Prefill {
    pub(super) batch: u32,
    pub(super) length: u32,
    pub(super) rows: u32,
    pub(super) matmul: RefCell<Matmul>,
    pub(super) feedback: RefCell<Feedback>,
    pub(super) widen: ComputePipelineState,
    pub(super) prepare: ComputePipelineState,
    pub(super) softmax: ComputePipelineState,
    pub(super) unpack: ComputePipelineState,
    pub(super) flash_attention: ComputePipelineState,
    pub(super) shift: ComputePipelineState,
    pub(super) glu: ComputePipelineState,
    pub(super) weights: Buffer,
    pub(super) tokens: Buffer,
    pub(super) targets: Buffer,
    pub(super) mask: Buffer,
    pub(super) losses: Buffer,
    pub(super) x: Buffer,
    pub(super) embed: Buffer,
    pub(super) normalized: Buffer,
    pub(super) branch: Buffer,
    pub(super) previous: Buffer,
    pub(super) history: Buffer,
    pub(super) qkv: [Buffer; 3],
    pub(super) head_major: [Buffer; 4],
    pub(super) gates: Buffer,
    pub(super) ff_gate: Buffer,
    pub(super) ff_up: Buffer,
    pub(super) scores: Buffer,
    pub(super) logits: Buffer,
    pub(super) rms_half: ComputePipelineState,
    pub(super) residual_half: ComputePipelineState,
    pub(super) glu_half: ComputePipelineState,
    pub(super) prepare_half: ComputePipelineState,
    pub(super) flash_attention_half: ComputePipelineState,
    pub(super) flash_attention_half_96: ComputePipelineState,
    pub(super) normalized_half: Buffer,
    pub(super) embed_half: Buffer,
    pub(super) branch_half: Buffer,
    pub(super) qkv_half: [Buffer; 3],
    pub(super) gates_half: Buffer,
    pub(super) ff_gate_half: Buffer,
    pub(super) ff_up_half: Buffer,
    pub(super) cross_entropy_blocked_half: ComputePipelineState,
    pub(super) logits_half: Buffer,
    pub(super) unit_rms_half: ComputePipelineState,
    pub(super) feedback_combine_norm: ComputePipelineState,
    pub(super) qkvg_half: Buffer,
    pub(super) prepare_half_fused: ComputePipelineState,
    pub(super) ff_gate_up_half: Buffer,
    pub(super) glu_half_fused: ComputePipelineState,
    pub(super) history_half: Buffer,
    pub(super) shift_half: ComputePipelineState,
    pub(super) glu_gemm: ComputePipelineState,
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
        self.ensure_transposed()?;
        Ok(())
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
        self.ensure_transposed()?;

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
    pub fn score_traced(
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
        let (preparation_operations, refreshed) = self.trace_transposed(&mut trace)?;
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
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(super) fn decode_half(bits: u16) -> f64 {
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
    pub(super) fn gpu_attention96() {
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
    pub(super) fn cache_parity() {
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
