use super::prefill::*;

impl Prefill {
    pub(in super::super) fn layer(
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
                    self.glu_gemm(
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

    pub(in super::super) fn start_pass(
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

    pub(in super::super) fn finish_pass(
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
}
