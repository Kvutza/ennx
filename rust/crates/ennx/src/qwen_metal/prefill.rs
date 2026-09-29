use super::*;

impl QwenEvaluator {
    pub(super) fn forward(
        &self,
        weights: &Buffer,
        rows: u32,
        mut cache: Option<&mut GenerationState>,
    ) -> Result<Option<QwenStageProfile>> {
        if let Some(cache) = cache.as_deref_mut() {
            let mut start = 0;
            let mut stages = None;
            while start < rows {
                let chunk = (rows - start).min(self.prefill_chunk);
                if let Some(chunk_stages) =
                    self.forward_chunk(weights, chunk, start, Some(&*cache))?
                {
                    stages
                        .get_or_insert_with(QwenStageProfile::default)
                        .merge(chunk_stages);
                }
                cache.position = cache
                    .position
                    .checked_add(chunk)
                    .ok_or("Qwen generation position overflow")?;
                start += chunk;
            }
            return Ok(stages);
        }
        if rows > REF_ROWS {
            return Err(
                "Qwen reference forward is limited to 256 tokens; use generate for long prompts"
                    .into(),
            );
        }
        self.forward_chunk(weights, rows, 0, None)
    }

    pub(super) fn forward_chunk(
        &self,
        weights: &Buffer,
        rows: u32,
        start: u32,
        cache: Option<&GenerationState>,
    ) -> Result<Option<QwenStageProfile>> {
        if let Some(cache) = cache {
            if cache
                .position
                .checked_add(rows)
                .is_none_or(|end| end > cache.capacity)
            {
                return Err("Qwen prefill exceeds the generation cache capacity".into());
            }
        }
        let c = self.config;
        let qwen = QwenShape {
            start,
            ..self.qwen_shape(rows)
        };
        let hidden_elements = rows as usize * c.hidden as usize;
        let stage_mode = std::env::var_os("ENNX_QWEN_STAGE_PROFILE").is_some();
        let sync_layers = !stage_mode && std::env::var_os("ENNX_QWEN_SYNC_LAYERS").is_some();
        let mut stages = stage_mode.then(|| QwenStageProfile {
            chunks: 1,
            ..QwenStageProfile::default()
        });
        let mut command = self.runtime.queue.new_command_buffer();
        command.set_label("Qwen forward");
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
        if let Some(profile) = stages.as_mut() {
            command = stage_split(&self.runtime, command, profile, Stage::Embed, "Qwen qkv")?;
        } else if sync_layers {
            finish(&command)?;
            self.trace_values("embedding", W::X, hidden_elements.min(4));
            self.trace_at(
                "embedding (last row)",
                W::X,
                (rows as usize - 1) * c.hidden as usize,
                hidden_elements.min(4),
            );
            command = self.runtime.queue.new_command_buffer();
            command.set_label("Qwen layer 0");
        }
        for (layer_index, layer) in self.layout.layers.iter().enumerate() {
            self.prefill_qkv(command, weights, layer, qwen)?;
            if let Some(profile) = stages.as_mut() {
                command = stage_split(
                    &self.runtime,
                    command,
                    profile,
                    Stage::Qkv,
                    "Qwen attention",
                )?;
            }
            self.prefill_attention(command, layer_index, cache, qwen);
            if let Some(profile) = stages.as_mut() {
                command = stage_split(
                    &self.runtime,
                    command,
                    profile,
                    Stage::Attn,
                    "Qwen attention output",
                )?;
            }
            self.prefill_linear(
                &command,
                W::Update,
                weights,
                layer.o_weight,
                W::Attended,
                rows,
                c.hidden,
                c.hidden,
            )?;
            self.residual(&command, W::X, W::Attended, rows);
            if let Some(profile) = stages.as_mut() {
                command = stage_split(
                    &self.runtime,
                    command,
                    profile,
                    Stage::AttnOut,
                    "Qwen MLP expansion",
                )?;
            }
            let half_mlp = self.prefill_mlp(command, weights, layer, qwen)?;
            if let Some(profile) = stages.as_mut() {
                command = stage_split(
                    &self.runtime,
                    command,
                    profile,
                    Stage::MlpExpand,
                    "Qwen MLP reduction",
                )?;
            }
            if half_mlp {
                self.mlp_reduce(&command, weights, rows, layer)?;
            } else {
                self.prefill_linear(
                    &command,
                    W::Activation,
                    weights,
                    layer.down_weight,
                    W::Attended,
                    rows,
                    c.intermediate,
                    c.hidden,
                )?;
            }
            self.residual(&command, W::X, W::Attended, rows);
            if let Some(profile) = stages.as_mut() {
                command = stage_split(
                    &self.runtime,
                    command,
                    profile,
                    Stage::MlpReduce,
                    "Qwen qkv",
                )?;
            } else if sync_layers {
                finish(&command)?;
                self.trace_at(
                    &format!("layer {} (last row)", layer_index),
                    W::X,
                    (rows as usize - 1) * c.hidden as usize,
                    hidden_elements.min(4),
                );
                command = self.runtime.queue.new_command_buffer();
                command.set_label(&format!("Qwen layer {}", layer_index + 1));
            }
        }
        self.rms(
            &command,
            W::X,
            weights,
            self.layout.final_norm,
            W::Norm,
            rows,
        );
        finish(&command)?;
        if let Some(profile) = stages.as_mut() {
            profile.add(Stage::FinalNorm, &command);
        }
        self.trace_at(
            "final norm (last row)",
            W::Norm,
            (rows as usize - 1) * c.hidden as usize,
            hidden_elements.min(4),
        );
        Ok(stages)
    }
    fn prefill_qkv(
        &self,
        command: &CommandBufferRef,
        weights: &Buffer,
        layer: &Layer,
        qwen: QwenShape,
    ) -> Result<()> {
        let c = self.config;
        let rows = qwen.rows;
        let kv_elements = rows as usize * c.kv_width() as usize;
        let hidden_elements = rows as usize * c.hidden as usize;
        self.rms(&command, W::X, weights, layer.input_norm, W::Norm, rows);
        self.prefill_linear(
            &command,
            W::Norm,
            weights,
            layer.q_weight,
            W::QRaw,
            rows,
            c.hidden,
            c.hidden,
        )?;
        self.elementwise(
            &command,
            "qwen_bias",
            &[
                (self.buffer(W::QRaw), 0),
                (weights, layer.q_bias as u64 * 2),
            ],
            &qwen,
            hidden_elements,
        );
        self.prefill_linear(
            &command,
            W::Norm,
            weights,
            layer.k_weight,
            W::KRaw,
            rows,
            c.hidden,
            c.kv_width(),
        )?;
        let kv_shape = QwenShape {
            width: c.kv_width(),
            ..qwen
        };
        self.elementwise(
            &command,
            "qwen_bias",
            &[
                (self.buffer(W::KRaw), 0),
                (weights, layer.k_bias as u64 * 2),
            ],
            &kv_shape,
            kv_elements,
        );
        self.prefill_linear(
            &command,
            W::Norm,
            weights,
            layer.v_weight,
            W::VRaw,
            rows,
            c.hidden,
            c.kv_width(),
        )?;
        self.elementwise(
            &command,
            "qwen_bias",
            &[
                (self.buffer(W::VRaw), 0),
                (weights, layer.v_bias as u64 * 2),
            ],
            &kv_shape,
            kv_elements,
        );
        self.encode(
            &command,
            "qwen_rope",
            &[
                (self.buffer(W::QRaw), 0),
                (self.buffer(W::KRaw), 0),
                (self.buffer(W::VRaw), 0),
                (self.buffer(W::Q), 0),
                (self.buffer(W::K), 0),
                (self.buffer(W::V), 0),
                (&self.rope_table, 0),
            ],
            &qwen,
            thread_group(hidden_elements.div_ceil(256) as u64),
        );
        Ok(())
    }

    fn prefill_attention(
        &self,
        command: &CommandBufferRef,
        layer_index: usize,
        cache: Option<&GenerationState>,
        qwen: QwenShape,
    ) -> () {
        let c = self.config;
        let rows = qwen.rows;
        let flame = self.flame_shape(rows);
        let hidden_elements = rows as usize * c.hidden as usize;
        if let Some(cache) = cache {
            let cache_shape = CacheShape {
                rows,
                kv_heads: c.kv_heads,
                head_dim: c.head_dim(),
                capacity: cache.capacity,
                position: cache.position,
            };
            self.encode(
                &command,
                "qwen_cache",
                &[
                    (self.buffer(W::K), 0),
                    (self.buffer(W::V), 0),
                    (&cache.key_cache, cache.layer_offset(c, layer_index)),
                    (&cache.value_cache, cache.layer_offset(c, layer_index)),
                ],
                &cache_shape,
                thread_group((rows as u64 * u64::from(c.kv_width())).div_ceil(256)),
            );
            let attention = PrefillAttentionShape {
                rows,
                sequence: cache.position + rows,
                capacity: cache.capacity,
                heads: c.heads,
                kv_heads: c.kv_heads,
                head_dim: c.head_dim(),
                scale: 1.0 / (c.head_dim() as f32).sqrt(),
            };
            let buffers = [
                (self.buffer(W::Q), 0),
                (&cache.key_cache, cache.layer_offset(c, layer_index)),
                (&cache.value_cache, cache.layer_offset(c, layer_index)),
                (self.buffer(W::Update), 0),
            ];
            if self.tile_attn {
                self.encode_threads(
                    &command,
                    "qwen_attn16",
                    &buffers,
                    &attention,
                    MTLSize {
                        width: u64::from(c.heads),
                        height: u64::from(rows).div_ceil(16),
                        depth: 1,
                    },
                    256,
                );
            } else {
                self.encode_threads(
                    &command,
                    "qwen_arow",
                    &buffers,
                    &attention,
                    thread_group(u64::from(rows) * u64::from(c.heads)),
                    32,
                );
            }
        } else {
            self.elementwise(
                &command,
                "qwen_repkv",
                &[(self.buffer(W::K), 0), (self.buffer(W::KRepeat), 0)],
                &qwen,
                hidden_elements,
            );
            self.elementwise(
                &command,
                "qwen_repkv",
                &[(self.buffer(W::V), 0), (self.buffer(W::VRepeat), 0)],
                &qwen,
                hidden_elements,
            );
            let d = c.head_dim();
            let qk = Matmul {
                m: rows,
                n: rows,
                k: d,
                transpose_b: 1,
                stride_a: u64::from(rows) * u64::from(d),
                stride_b: u64::from(rows) * u64::from(d),
                stride_c: u64::from(rows) * u64::from(rows),
            };
            self.encode(
                &command,
                "flame_matmul",
                &[
                    (self.buffer(W::Q), 0),
                    (self.buffer(W::KRepeat), 0),
                    (self.buffer(W::Scores), 0),
                ],
                &qk,
                MTLSize {
                    width: u64::from(rows).div_ceil(32),
                    height: u64::from(rows).div_ceil(32),
                    depth: u64::from(c.heads),
                },
            );
            self.encode(
                &command,
                "flame_softmax",
                &[(self.buffer(W::Scores), 0)],
                &flame,
                thread_group(u64::from(rows) * u64::from(c.heads)),
            );
            let av = Matmul {
                m: rows,
                n: d,
                k: rows,
                transpose_b: 0,
                stride_a: u64::from(rows) * u64::from(rows),
                stride_b: u64::from(rows) * u64::from(d),
                stride_c: u64::from(rows) * u64::from(d),
            };
            self.encode(
                &command,
                "flame_matmul",
                &[
                    (self.buffer(W::Scores), 0),
                    (self.buffer(W::VRepeat), 0),
                    (self.buffer(W::Attended), 0),
                ],
                &av,
                MTLSize {
                    width: u64::from(d).div_ceil(32),
                    height: u64::from(rows).div_ceil(32),
                    depth: u64::from(c.heads),
                },
            );
            self.elementwise(
                &command,
                "qwen_aout",
                &[(self.buffer(W::Attended), 0), (self.buffer(W::Update), 0)],
                &qwen,
                hidden_elements,
            );
        }
    }

    fn prefill_mlp(
        &self,
        command: &CommandBufferRef,
        weights: &Buffer,
        layer: &Layer,
        qwen: QwenShape,
    ) -> Result<bool> {
        let c = self.config;
        let rows = qwen.rows;
        self.rms(&command, W::X, weights, layer.post_norm, W::Norm, rows);
        let half_mlp = self.mlp_expand(&command, weights, rows, layer)?;
        if !half_mlp {
            self.prefill_linear(
                &command,
                W::Norm,
                weights,
                layer.gate_weight,
                W::Gate,
                rows,
                c.hidden,
                c.intermediate,
            )?;
            self.prefill_linear(
                &command,
                W::Norm,
                weights,
                layer.up_weight,
                W::Up,
                rows,
                c.hidden,
                c.intermediate,
            )?;
            let activation = QwenShape {
                hidden: c.intermediate,
                ..qwen
            };
            self.elementwise(
                &command,
                "qwen_silu",
                &[
                    (self.buffer(W::Gate), 0),
                    (self.buffer(W::Up), 0),
                    (self.buffer(W::Activation), 0),
                ],
                &activation,
                rows as usize * c.intermediate as usize,
            );
        }
        Ok(half_mlp)
    }
}
