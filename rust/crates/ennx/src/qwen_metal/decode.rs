use super::*;

impl QwenEvaluator {
    pub(super) fn argmax_output(&self, weights: &Buffer, input_offset: u64) -> Result<u32> {
        self.write(W::Invalid, &[0u32]);
        let c = self.config;
        let params = Matmul {
            m: 1,
            n: c.vocab,
            k: c.hidden,
            transpose_b: 1,
            ..Matmul::default()
        };
        let argmax = ArgmaxShape { width: c.vocab };
        let command = self.runtime.queue.new_command_buffer();
        command.set_label("Qwen argmax output");
        self.encode_threads(
            &command,
            "qwen_gemv",
            &[
                (self.buffer(W::Norm), input_offset),
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
            &argmax,
            thread_group(1),
        );
        finish(&command)?;
        if self.read::<u32>(W::Invalid, 1)[0] != 0 {
            return Err("Qwen argmax received nonfinite logits".into());
        }
        let token = self.read::<u32>(W::Tokens, 1)[0];
        if token >= c.vocab {
            return Err("Qwen argmax returned an invalid token".into());
        }
        Ok(token)
    }

    pub(super) fn decode_token(
        &self,
        weights: &Buffer,
        state: &mut GenerationState,
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
        let command = self.runtime.queue.new_command_buffer();
        command.set_label("Qwen decode");
        self.write(W::Invalid, &[0u32]);
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
        for (layer_index, layer) in self.layout.layers.iter().enumerate() {
            self.rms(&command, W::X, weights, layer.input_norm, W::Norm, 1);
            self.qkv(&command, weights, 1, layer);
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
            self.rms(&command, W::X, weights, layer.post_norm, W::Norm, 1);
            self.silu_rows(&command, weights, 1, layer);
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
        }
        self.rms(&command, W::X, weights, self.layout.final_norm, W::Norm, 1);
        let params = Matmul {
            m: 1,
            n: c.vocab,
            k: c.hidden,
            transpose_b: 1,
            ..Matmul::default()
        };
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
        finish(&command)?;
        self.trace_values("decode final norm", W::Norm, (c.hidden as usize).min(4));
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

    pub(super) fn decode_logits(
        &self,
        weights: &Buffer,
        state: &mut GenerationState,
    ) -> Result<Vec<f32>> {
        self.decode_token(weights, state)?;
        let logits = self.read::<f32>(W::Logits, self.config.vocab as usize);
        if logits.iter().any(|value| !value.is_finite()) {
            return Err("Qwen sampled decode produced nonfinite logits".into());
        }
        Ok(logits)
    }

    pub(super) fn batch_logits(
        &self,
        weights: &Buffer,
        state: &mut GenerationState,
        positions: &[u32],
        greedy: bool,
    ) -> Result<BatchDecodeOutput> {
        if positions.len() != state.batch as usize {
            return Err("Qwen batch decode requires one position per sequence".into());
        }
        if positions
            .iter()
            .any(|&position| position >= state.capacity || position >= self.max_tokens)
        {
            return Err("Qwen batch generation cache is full".into());
        }
        write_buffer(&state.position_buffer, positions);
        self.write(W::Invalid, &[0u32]);
        let c = self.config;
        let rows = state.batch;
        let qwen = self.qwen_shape(rows);
        let hidden_elements = rows as usize * c.hidden as usize;
        let command = self.runtime.queue.new_command_buffer();
        command.set_label("Qwen batched decode");
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
        for (layer_index, layer) in self.layout.layers.iter().enumerate() {
            self.rms(&command, W::X, weights, layer.input_norm, W::Norm, rows);
            self.qkv(&command, weights, rows, layer);
            let rope = DecodeRopeShape {
                heads: c.heads,
                kv_heads: c.kv_heads,
                head_dim: c.head_dim(),
                capacity: state.capacity,
                position: 0,
                batch: rows,
                cache_stride: state.cache_stride as u32,
                rope_theta: c.rope_theta,
            };
            self.encode(
                &command,
                "qwen_drope",
                &[
                    (self.buffer(W::QRaw), 0),
                    (self.buffer(W::KRaw), 0),
                    (self.buffer(W::VRaw), 0),
                    (self.buffer(W::Q), 0),
                    (&state.key_cache, state.layer_batch(layer_index)),
                    (&state.value_cache, state.layer_batch(layer_index)),
                    (&self.rope_table, 0),
                    (&state.position_buffer, 0),
                ],
                &rope,
                thread_group((hidden_elements as u64).div_ceil(256)),
            );
            let attention = DecodeAttentionShape {
                sequence: 0,
                capacity: state.capacity,
                heads: c.heads,
                kv_heads: c.kv_heads,
                head_dim: c.head_dim(),
                batch: rows,
                cache_stride: state.cache_stride as u32,
                scale: 1.0 / (c.head_dim() as f32).sqrt(),
            };
            self.encode_threads(
                &command,
                "qwen_dattn",
                &[
                    (self.buffer(W::Q), 0),
                    (&state.key_cache, state.layer_batch(layer_index)),
                    (&state.value_cache, state.layer_batch(layer_index)),
                    (self.buffer(W::Update), 0),
                    (&state.position_buffer, 0),
                ],
                &attention,
                MTLSize {
                    width: u64::from(rows) * u64::from(c.heads),
                    height: 1,
                    depth: 1,
                },
                32,
            );
            self.linear(
                &command,
                W::Update,
                weights,
                layer.o_weight,
                W::Attended,
                rows,
                c.hidden,
                c.hidden,
            );
            self.residual(&command, W::X, W::Attended, rows);
            self.rms(&command, W::X, weights, layer.post_norm, W::Norm, rows);
            self.silu_rows(&command, weights, rows, layer);
            self.linear(
                &command,
                W::Activation,
                weights,
                layer.down_weight,
                W::Attended,
                rows,
                c.intermediate,
                c.hidden,
            );
            self.residual(&command, W::X, W::Attended, rows);
        }
        self.rms(
            &command,
            W::X,
            weights,
            self.layout.final_norm,
            W::Norm,
            rows,
        );
        self.linear(
            &command,
            W::Norm,
            weights,
            self.layout.embedding,
            W::Logits,
            rows,
            c.hidden,
            c.vocab,
        );
        if greedy {
            self.encode(
                &command,
                "qwen_argn",
                &[
                    (self.buffer(W::Logits), 0),
                    (self.buffer(W::Tokens), 0),
                    (self.buffer(W::Invalid), 0),
                ],
                &ArgmaxBatchShape {
                    rows,
                    width: c.vocab,
                },
                thread_group(u64::from(rows)),
            );
            finish(&command)?;
            if self.read::<u32>(W::Invalid, 1)[0] != 0 {
                return Err("Qwen batched argmax received nonfinite logits".into());
            }
            let tokens = self.read::<u32>(W::Tokens, rows as usize);
            validate_batch(&tokens, c.vocab)?;
            return Ok(BatchDecodeOutput::Tokens(tokens));
        }
        finish(&command)?;
        let logits = self.read::<f32>(W::Logits, state.batch as usize * c.vocab as usize);
        if logits.iter().any(|value| !value.is_finite()) {
            return Err("Qwen batched decode produced nonfinite logits".into());
        }
        Ok(BatchDecodeOutput::Logits(
            logits
                .chunks_exact(c.vocab as usize)
                .map(Vec::from)
                .collect(),
        ))
    }

    pub(super) fn output_logits(&self, weights: &Buffer, input_offset: u64) -> Result<Vec<f32>> {
        let c = self.config;
        let params = Matmul {
            m: 1,
            n: c.vocab,
            k: c.hidden,
            transpose_b: 1,
            ..Matmul::default()
        };
        let command = self.runtime.queue.new_command_buffer();
        command.set_label("Qwen sampled output");
        self.encode_threads(
            &command,
            "qwen_gemv",
            &[
                (self.buffer(W::Norm), input_offset),
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
        finish(&command)?;
        let logits = self.read::<f32>(W::Logits, c.vocab as usize);
        if logits.iter().any(|value| !value.is_finite()) {
            return Err("Qwen sampled output produced nonfinite logits".into());
        }
        Ok(logits)
    }
}

fn validate_batch(tokens: &[u32], vocab: u32) -> Result<()> {
    if tokens.iter().any(|&token| token >= vocab) {
        return Err("Qwen batched argmax returned an invalid token".into());
    }
    Ok(())
}
