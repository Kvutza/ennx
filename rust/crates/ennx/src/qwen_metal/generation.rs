use super::*;

impl QwenEvaluator {
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

    pub(super) fn batch_inner(
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

    pub(super) fn sample_inner(
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

    pub(super) fn generate_inner(
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
}
