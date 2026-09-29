use super::*;

impl QwenEvaluator {
    pub(super) fn bench_profile(
        &mut self,
        weights: &Buffer,
        prompt: &[i32],
        max_new_tokens: usize,
        mode: &str,
    ) -> Result<QwenGenerationProfile> {
        if mode != "greedy" {
            return Err("Qwen bench_generate currently supports only greedy mode".into());
        }
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

        let total_start = Instant::now();
        let mut state = self.new_state()?;
        let kv_cache_bytes = state
            .key_cache
            .length()
            .saturating_add(state.value_cache.length());
        let mut command_buffer_count = prompt.len().div_ceil(self.prefill_chunk as usize) as u32;

        self.write(W::Tokens, prompt);
        self.write(W::Masks, &vec![0u8; prompt.len()]);
        let prefill_start = Instant::now();
        self.forward(weights, prompt.len() as u32, Some(&mut state))?;
        let prefill_ms = prefill_start.elapsed().as_secs_f32() * 1000.0;

        let first_start = Instant::now();
        let mut next_token = self.argmax_output(
            weights,
            u64::from((prompt.len() - 1) as u32 % self.prefill_chunk)
                * u64::from(self.config.hidden)
                * 4,
        )?;
        command_buffer_count = command_buffer_count.saturating_add(1);
        let first_token_ms = first_start.elapsed().as_secs_f32() * 1000.0;

        let mut generated_tokens = 0u32;
        let mut decode_kernel_times_ms = BTreeMap::<&'static str, f32>::new();
        let decode_start = Instant::now();
        for index in 0..max_new_tokens {
            generated_tokens = generated_tokens.saturating_add(1);
            if next_token == self.config.eos_token_id {
                break;
            }
            if index + 1 < max_new_tokens {
                let token = i32::try_from(next_token).map_err(|_| "Qwen token ID overflow")?;
                self.write(W::Tokens, &[token]);
                next_token =
                    self.decode_profile(weights, &mut state, &mut decode_kernel_times_ms)?;
                command_buffer_count = command_buffer_count.saturating_add(1);
            }
        }
        let decode_ms_after_first = decode_start.elapsed().as_secs_f32() * 1000.0;
        let total_ms = total_start.elapsed().as_secs_f32() * 1000.0;
        let metal_phase_ms = prefill_ms + first_token_ms + decode_ms_after_first;
        let steady_tokens = generated_tokens.saturating_sub(1);
        let steady_decode_tokens_per_second = tps(steady_tokens, decode_ms_after_first);
        let end_to_end_generated_tokens_per_second = tps(generated_tokens, total_ms);

        let mut measured_times: Vec<_> = decode_kernel_times_ms.into_iter().collect();
        measured_times.sort_by(|left, right| {
            right
                .1
                .partial_cmp(&left.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let decode_bottleneck_candidates = if measured_times.is_empty() {
            vec!["no steady decode token measured"]
        } else {
            measured_times
                .iter()
                .take(3)
                .map(|(name, _)| *name)
                .collect()
        };

        Ok(QwenGenerationProfile {
            prompt_tokens: prompt.len() as u32,
            requested_generated_tokens: max_new_tokens as u32,
            generated_tokens,
            prefill_ms,
            first_token_ms,
            decode_ms_after_first,
            total_ms,
            tile_attn: self.tile_attn,
            steady_decode_tokens_per_second,
            end_to_end_generated_tokens_per_second,
            host_overhead_ms: (total_ms - metal_phase_ms).max(0.0),
            command_buffer_count,
            logits_read_to_cpu: false,
            token_selection_on_gpu: true,
            kv_cache_bytes,
            device_name: self.device_name().to_string(),
            max_tokens: self.max_tokens,
            decode_kernel_trace: vec![
                "qwen_embedding",
                "qwen_rms",
                "qwen_qkv",
                "qwen_drope",
                "qwen_dattn",
                "qwen_gemv_rows",
                "qwen_residual",
                "qwen_rms",
                "qwen_mlp_rows",
                "qwen_gemv_rows",
                "qwen_residual",
                "qwen_rms",
                "qwen_gemv",
                "qwen_argmax",
            ],
            decode_kernel_times_ms: measured_times,
            decode_bottleneck_candidates,
            lm_head_argmax_decision: "keep separate qwen_gemv + qwen_argmax until a measured trace shows the final LM head is the largest decode bottleneck",
        })
    }

    pub(super) fn decode_profile(
        &self,
        weights: &Buffer,
        state: &mut GenerationState,
        times: &mut BTreeMap<&'static str, f32>,
    ) -> Result<u32> {
        if state.position >= self.max_tokens {
            return Err("Qwen generation cache is full".into());
        }
        let c = self.config;
        let qwen = self.qwen_shape(1);
        let hidden_elements = c.hidden as usize;
        let position = state.position;
        let sequence = position
            .checked_add(1)
            .ok_or("Qwen generation position overflow")?;
        write_buffer(&state.position_buffer, &[position]);
        self.write(W::Invalid, &[0u32]);

        macro_rules! timed_command {
            ($name:literal, $command:ident, $body:block) => {{
                let $command = self.runtime.queue.new_command_buffer();
                $command.set_label($name);
                let start = Instant::now();
                $body
                finish(&$command)?;
                *times.entry($name).or_insert(0.0) += start.elapsed().as_secs_f32() * 1000.0;
            }};
        }

        timed_command!("qwen_embedding", command, {
            self.elementwise(
                &command,
                "qwen_embedding",
                &[
                    (weights, self.layout.embedding as u64 * 2),
                    (self.buffer(W::Tokens), 0),
                    (self.buffer(W::X), 0),
                ],
                &qwen,
                hidden_elements,
            );
        });

        for (layer_index, layer) in self.layout.layers.iter().enumerate() {
            timed_command!("qwen_rms_input", command, {
                self.rms(&command, W::X, weights, layer.input_norm, W::Norm, 1);
            });
            timed_command!("qwen_qkv", command, {
                self.qkv(&command, weights, 1, layer);
            });
            let rope = DecodeRopeShape {
                heads: c.heads,
                kv_heads: c.kv_heads,
                head_dim: c.head_dim(),
                capacity: state.capacity,
                position,
                batch: 1,
                cache_stride: state.cache_stride as u32,
                rope_theta: c.rope_theta,
            };
            timed_command!("qwen_drope", command, {
                self.encode(
                    &command,
                    "qwen_drope",
                    &[
                        (self.buffer(W::QRaw), 0),
                        (self.buffer(W::KRaw), 0),
                        (self.buffer(W::VRaw), 0),
                        (self.buffer(W::Q), 0),
                        (&state.key_cache, state.layer_offset(c, layer_index)),
                        (&state.value_cache, state.layer_offset(c, layer_index)),
                        (&self.rope_table, 0),
                        (&state.position_buffer, 0),
                    ],
                    &rope,
                    thread_group(u64::from(c.hidden).div_ceil(256)),
                );
            });
            let attention = DecodeAttentionShape {
                sequence,
                capacity: state.capacity,
                heads: c.heads,
                kv_heads: c.kv_heads,
                head_dim: c.head_dim(),
                batch: 1,
                cache_stride: state.cache_stride as u32,
                scale: 1.0 / (c.head_dim() as f32).sqrt(),
            };
            timed_command!("qwen_dattn", command, {
                self.encode_threads(
                    &command,
                    "qwen_dattn",
                    &[
                        (self.buffer(W::Q), 0),
                        (&state.key_cache, state.layer_offset(c, layer_index)),
                        (&state.value_cache, state.layer_offset(c, layer_index)),
                        (self.buffer(W::Update), 0),
                        (&state.position_buffer, 0),
                    ],
                    &attention,
                    thread_group(u64::from(c.heads)),
                    32,
                );
            });
            timed_command!("qwen_attention_output_projection", command, {
                self.linear(
                    &command,
                    W::Update,
                    weights,
                    layer.o_weight,
                    W::Attended,
                    1,
                    c.hidden,
                    c.hidden,
                );
                self.residual(&command, W::X, W::Attended, 1);
            });
            timed_command!("qwen_rms_post_attention", command, {
                self.rms(&command, W::X, weights, layer.post_norm, W::Norm, 1);
            });
            timed_command!("qwen_mlp_rows", command, {
                self.silu_rows(&command, weights, 1, layer);
            });
            timed_command!("qwen_mlp_down_projection", command, {
                self.linear(
                    &command,
                    W::Activation,
                    weights,
                    layer.down_weight,
                    W::Attended,
                    1,
                    c.intermediate,
                    c.hidden,
                );
                self.residual(&command, W::X, W::Attended, 1);
            });
        }
        timed_command!("qwen_rms_final", command, {
            self.rms(&command, W::X, weights, self.layout.final_norm, W::Norm, 1);
        });
        let params = Matmul {
            m: 1,
            n: c.vocab,
            k: c.hidden,
            transpose_b: 1,
            ..Matmul::default()
        };
        timed_command!("qwen_lm_head_argmax", command, {
            self.encode_threads(
                &command,
                "qwen_gemv",
                &[
                    (self.buffer(W::Norm), 0),
                    (weights, self.layout.embedding as u64 * 2),
                    (self.buffer(W::Logits), 0),
                ],
                &params,
                MTLSize {
                    width: u64::from(c.vocab).div_ceil(32),
                    height: 1,
                    depth: 1,
                },
                32,
            );
            self.encode(
                &command,
                "qwen_argmax",
                &[
                    (self.buffer(W::Logits), 0),
                    (self.buffer(W::Tokens), 0),
                    (self.buffer(W::Invalid), 0),
                ],
                &ArgmaxShape { width: c.vocab },
                thread_group(1),
            );
        });

        state.position = sequence;
        if self.read::<u32>(W::Invalid, 1)[0] != 0 {
            return Err("Qwen decode produced nonfinite logits".into());
        }
        let token = self.read::<u32>(W::Tokens, 1)[0];
        if token >= c.vocab {
            return Err("Qwen decode produced an invalid token".into());
        }
        Ok(token)
    }
}
