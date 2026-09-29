use super::prefill::*;

impl Prefill {
    pub(super) fn score_traced(
        &self,
        model: &Model,
        examples: &[(&[u32], &[u32])],
        mode: ScoreMode,
        mut trace: TraceRecorder,
        preparation_operations: usize,
    ) -> Result<BatchTrace, String> {
        let total = trace.started;
        let scorer_command_start = trace.commands.len();
        self.trace_input(&mut trace, examples)?;
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
            self.trace_feedback(model, &mut trace, pass_index, pass)?;
            for index in 0..c.layers as usize {
                self.trace_attention(model, &mut trace, index, pass)?;
                self.trace_ffn(model, &mut trace, index, pass)?;
            }

            self.trace_finish(model, &mut trace, pass, passes)?;
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
            || self.traced_losses(),
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

    fn traced_losses(&self) -> Result<Vec<f64>, String> {
        let losses = unsafe {
            std::slice::from_raw_parts(self.losses.contents().cast::<f32>(), self.rows as usize)
        };
        if losses.iter().any(|value| !value.is_finite()) {
            return Err("Non-finite FBT traced prefill score".to_string());
        }
        Ok(losses
            .chunks_exact(self.length as usize)
            .map(|chunk| {
                chunk.iter().map(|&value| f64::from(value)).sum::<f64>() / f64::from(self.length)
            })
            .collect::<Vec<_>>())
    }

    fn trace_input(
        &self,
        trace: &mut TraceRecorder,
        examples: &[(&[u32], &[u32])],
    ) -> Result<(), String> {
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

        Ok(())
    }

    fn trace_attention(
        &self,
        model: &Model,
        trace: &mut TraceRecorder,
        index: usize,
        pass: u32,
    ) -> Result<(), String> {
        let c = model.config;
        let rows = self.rows;
        let groups = thread_group(u64::from(rows));
        let graph = GraphParams {
            width: c.width,
            rows,
            start: 0,
            vocab: c.vocab,
            scale: c.residual_scale,
        };
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
                self.batch, self.length, c.heads, c.kv_heads, attention.dim, attention.window
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
        Ok(())
    }

    fn trace_ffn(
        &self,
        model: &Model,
        trace: &mut TraceRecorder,
        index: usize,
        pass: u32,
    ) -> Result<(), String> {
        let c = model.config;
        let rows = self.rows;
        let groups = thread_group(u64::from(rows));
        let graph = GraphParams {
            width: c.width,
            rows,
            start: 0,
            vocab: c.vocab,
            scale: c.residual_scale,
        };
        let layer = &model.layers[index];
        let layer_number = index as u32;
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
                    self.glu_gemm(
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
        Ok(())
    }

    fn trace_feedback(
        &self,
        model: &Model,
        trace: &mut TraceRecorder,
        pass_index: u32,
        pass: u32,
    ) -> Result<(), String> {
        let c = model.config;
        let rows = self.rows;
        let groups = thread_group(u64::from(rows));
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

        Ok(())
    }

    fn trace_readout(
        &self,
        model: &Model,
        trace: &mut TraceRecorder,
        pass: u32,
    ) -> Result<(), String> {
        let c = model.config;
        let rows = self.rows;
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
                            Matrix::half(embedding, c.width, c.vocab).column_view(width, column),
                            Matrix::half(&self.logits_half, chunk_rows, width)
                                .row_view(chunk_rows, u64::from(column) * u64::from(chunk_rows)),
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
        Ok(())
    }

    fn trace_finish(
        &self,
        model: &Model,
        trace: &mut TraceRecorder,
        pass: u32,
        passes: u32,
    ) -> Result<(), String> {
        let c = model.config;
        let rows = self.rows;
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
            self.trace_readout(model, trace, pass)?;
        }
        Ok(())
    }
}
