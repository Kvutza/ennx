use super::*;
use ::metal::{CommandBuffer, MTLCommandBufferStatus};
fn validate_commands(
    commands: &[CommandBuffer],
    mut failure: Option<String>,
) -> Result<(), String> {
    for command in commands {
        if command.status() != MTLCommandBufferStatus::Completed {
            failure = Some(format!(
                "FBT model GPU command failed: {:?}",
                command.status()
            ));
        }
    }
    if let Some(error) = failure {
        return Err(error);
    }
    Ok(())
}

impl Model {
    fn upload_inputs(&self, tokens: &[u32], targets: &[u32]) {
        unsafe {
            std::ptr::copy_nonoverlapping(
                tokens.as_ptr(),
                self.tokens.contents().cast(),
                tokens.len(),
            );
            std::ptr::copy_nonoverlapping(
                targets.as_ptr(),
                self.targets.contents().cast(),
                targets.len(),
            );
        }
    }
    fn feedback_input(
        &mut self,
        command: &CommandBufferRef,
        key: CacheKey,
        rows: u32,
        mode: ScoreMode,
        p: &GraphParams,
    ) -> Result<(), String> {
        let groups = thread_group(u64::from(rows));
        let w = |index: usize| self.parameters[index].buffer.as_ref();
        let fused = mode == ScoreMode::Sequential || key.pass > 0;
        dispatch(
            command,
            &self.lookup,
            &[
                w(self.embedding),
                &self.tokens,
                if fused { &self.embed } else { &self.x },
            ],
            p,
            groups,
        );
        if fused {
            let source = if mode == ScoreMode::Sequential {
                0
            } else {
                (key.pass - 1) as usize
            };
            dispatch(
                command,
                &self.shift,
                &[&self.histories[source], &self.previous, &self.mask],
                p,
                groups,
            );
            self.feedback.encode(
                command,
                rows,
                w(self.feedback_weights),
                &self.previous,
                &self.embed,
                &self.mask,
                &self.x,
            )?;
        }
        Ok(())
    }
    pub(super) fn score_inner(
        &mut self,
        tokens: &[u32],
        targets: &[u32],
        mode: ScoreMode,
    ) -> Result<Score, String> {
        let start_time = std::time::Instant::now();
        let c = self.config;
        validate_inputs(c, tokens, targets)?;
        self.sequence = self
            .sequence
            .checked_add(1)
            .ok_or("FBT sequence ID exhausted")?;
        self.upload_inputs(tokens, targets);
        let passes = if mode == ScoreMode::Fused { 2 } else { 1 };
        let mut pass_seconds = Vec::with_capacity(passes as usize);
        let mut encode_submit_seconds = 0.0;
        let mut completion_wait_seconds = 0.0;
        let mut chunks = Vec::new();
        for pass in 0..passes {
            let pass_start = std::time::Instant::now();
            let key = CacheKey {
                candidate: self.revision,
                sequence: self.sequence,
                pass,
            };
            for layer in &mut self.layers {
                layer.attention.reset(key)?;
            }
            let chunk = if mode == ScoreMode::Sequential {
                1
            } else {
                c.chunk
            };
            let mut commands: Vec<CommandBuffer> = Vec::new();
            let mut failure = None;
            let encode_start = std::time::Instant::now();
            for start in (0..tokens.len() as u32).step_by(chunk as usize) {
                let rows = chunk.min(tokens.len() as u32 - start);
                let command = self.runtime.queue.new_command_buffer().to_owned();
                let result =
                    self.encode_chunk(&command, key, start, rows, mode, pass + 1 == passes);
                // Submit even a partially encoded chunk before draining on error;
                // caches may retain this command and scratch cannot outlive work.
                command.commit();
                commands.push(command);
                if let Err(error) = result {
                    failure = Some(error);
                    break;
                }
            }
            encode_submit_seconds += encode_start.elapsed().as_secs_f64();
            let wait_start = std::time::Instant::now();
            if let Some(command) = commands.last() {
                command.wait_until_completed();
            }
            completion_wait_seconds += wait_start.elapsed().as_secs_f64();
            validate_commands(&commands, failure)?;
            for (index, command) in commands.iter().enumerate() {
                let start = index as u32 * chunk;
                chunks.push(ChunkTiming {
                    pass,
                    start,
                    rows: chunk.min(tokens.len() as u32 - start),
                    gpu_seconds: gpu_seconds(command),
                });
            }
            for layer in &self.layers {
                layer.attention.check_completed()?;
            }
            pass_seconds.push(pass_start.elapsed().as_secs_f64());
        }
        let loss = unsafe {
            std::slice::from_raw_parts(self.losses.contents().cast::<f32>(), tokens.len())
        };
        if loss.iter().any(|x| !x.is_finite()) {
            return Err("Non-finite FBT score".into());
        }
        Ok(Score {
            mean_nll: loss.iter().map(|&x| f64::from(x)).sum::<f64>() / tokens.len() as f64,
            tokens: tokens.len(),
            passes,
            elapsed_seconds: start_time.elapsed().as_secs_f64(),
            pass_seconds,
            encode_submit_seconds,
            completion_wait_seconds,
            chunks,
        })
    }

    fn encode_chunk(
        &mut self,
        command: &CommandBufferRef,
        key: CacheKey,
        start: u32,
        rows: u32,
        mode: ScoreMode,
        score: bool,
    ) -> Result<(), String> {
        let c = self.config;
        let p = GraphParams {
            width: c.width,
            rows,
            start,
            vocab: c.vocab,
            scale: c.residual_scale,
        };
        let groups = thread_group(u64::from(rows));
        self.feedback_input(command, key, rows, mode, &p)?;
        let w = |index: usize| self.parameters[index].buffer.as_ref();
        let linear = |op: &Linear, weights: &BufferRef, input: &BufferRef, output: &BufferRef| {
            if self.optimized && rows >= 64 {
                op.encode_fused(
                    command,
                    rows,
                    weights,
                    input,
                    output,
                    GemmEpilogue::Projection,
                )
            } else if rows >= 64 {
                op.encode_tiled(command, rows, weights, input, output)
            } else {
                op.encode(command, rows, weights, input, output)
            }
        };
        for layer in &mut self.layers {
            self.norm
                .encode(command, rows, &self.x, w(layer.norm_attn), &self.normalized)?;
            linear(&self.square, w(layer.q), &self.normalized, &self.q)?;
            linear(&self.kv, w(layer.k), &self.normalized, &self.k)?;
            linear(&self.kv, w(layer.v), &self.normalized, &self.v)?;
            linear(
                &self.head_gate,
                w(layer.head_gate),
                &self.normalized,
                &self.gates,
            )?;
            if c.tiled_attention && rows >= 8 {
                layer.attention.encode_tiled(
                    command,
                    key,
                    rows,
                    &self.q,
                    &self.k,
                    &self.v,
                    &self.gates,
                    &self.attended,
                )?;
            } else {
                layer.attention.encode(
                    command,
                    key,
                    rows,
                    &self.q,
                    &self.k,
                    &self.v,
                    &self.gates,
                    &self.attended,
                )?;
            }
            if self.optimized && rows >= 8 {
                self.square.encode_fused(
                    command,
                    rows,
                    w(layer.out),
                    &self.attended,
                    &self.x,
                    GemmEpilogue::Residual(c.residual_scale),
                )?;
            } else {
                linear(&self.square, w(layer.out), &self.attended, &self.branch)?;
                dispatch(
                    command,
                    &self.residual,
                    &[&self.branch, &self.x],
                    &p,
                    groups,
                );
            }
            self.norm
                .encode(command, rows, &self.x, w(layer.norm_ffn), &self.normalized)?;
            linear(&self.up, w(layer.gate), &self.normalized, &self.ff_gate)?;
            if self.optimized && rows >= 8 {
                self.up.encode_fused(
                    command,
                    rows,
                    w(layer.up),
                    &self.normalized,
                    &self.ff_hidden,
                    GemmEpilogue::Gated(&self.ff_gate),
                )?;
                self.down.encode_fused(
                    command,
                    rows,
                    w(layer.down),
                    &self.ff_hidden,
                    &self.x,
                    GemmEpilogue::Residual(c.residual_scale),
                )?;
            } else {
                linear(&self.up, w(layer.up), &self.normalized, &self.ff_up)?;
                let fp = GraphParams {
                    width: c.intermediate,
                    ..p
                };
                dispatch(
                    command,
                    &self.glu,
                    &[&self.ff_gate, &self.ff_up, &self.ff_hidden],
                    &fp,
                    groups,
                );
                linear(&self.down, w(layer.down), &self.ff_hidden, &self.branch)?;
                dispatch(
                    command,
                    &self.residual,
                    &[&self.branch, &self.x],
                    &p,
                    groups,
                );
            }
        }
        self.norm
            .encode(command, rows, &self.x, w(self.final_norm), &self.normalized)?;
        dispatch(
            command,
            &self.capture,
            &[&self.normalized, &self.histories[key.pass as usize]],
            &p,
            groups,
        );
        if score {
            if self.optimized {
                self.readout.encode_fused(
                    command,
                    rows,
                    w(self.embedding),
                    &self.normalized,
                    &self.logits,
                    GemmEpilogue::Loss {
                        targets: &self.targets,
                        start,
                    },
                )?;
                dispatch(
                    command,
                    &self.cross_entropy_partials,
                    &[&self.logits, &self.losses],
                    &p,
                    groups,
                );
            } else {
                linear(
                    &self.readout,
                    w(self.embedding),
                    &self.normalized,
                    &self.logits,
                )?;
                dispatch(
                    command,
                    &self.cross_entropy,
                    &[&self.logits, &self.targets, &self.losses],
                    &p,
                    groups,
                );
            }
        }
        Ok(())
    }
}
