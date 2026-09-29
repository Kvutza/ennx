use super::prefill::*;

impl Prefill {
    pub(super) fn score(
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
