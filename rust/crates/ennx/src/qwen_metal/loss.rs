use super::*;

impl QwenEvaluator {
    pub(super) fn check_batch(&self, tokens: &[Vec<i32>], masks: &[Vec<bool>]) -> Result<()> {
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

    pub(super) fn logits_inner(
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

    pub(super) fn cached_loss(
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

    pub(super) fn output(
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

    pub(super) fn project_rows(
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
            let cached = self
                .frozen_readout
                .as_ref()
                .map(|readout| readout.encode(self, &command, weights, start, chunk))
                .transpose()?
                .unwrap_or(false);
            if !cached {
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
            }
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

    pub(super) fn mean_loss(&self, rows: u32, scored: u32) -> Result<f32> {
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
