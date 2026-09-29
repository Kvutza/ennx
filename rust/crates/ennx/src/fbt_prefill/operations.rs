use super::prefill::*;

impl Prefill {
    pub(super) fn widen(&self, command: &CommandBufferRef, parameter: &Parameter) {
        dispatch(
            command,
            &self.widen,
            &[&parameter.buffer, &self.weights],
            &(parameter.elements as u64),
            thread_group((parameter.elements as u64).div_ceil(32)),
        );
    }

    pub(super) fn linear(
        &self,
        model: &Model,
        command: &CommandBufferRef,
        parameter: usize,
        input: &BufferRef,
        output: &BufferRef,
    ) -> Result<(), String> {
        let p = &model.parameters[parameter];
        let (n, k) = (p.shape[0], p.shape[1]);
        self.widen(command, p);
        self.matmul.borrow_mut().encode(
            &model.runtime.device,
            command,
            Matrix::new(input, self.rows, k),
            Matrix::new(&self.weights, n, k),
            Matrix::new(output, self.rows, n),
            true,
            1.0,
        )
    }

    pub(super) fn glu_gemm(
        &self,
        command: &CommandBufferRef,
        a: &BufferRef,
        b: &BufferRef,
        c: &BufferRef,
        m: u32,
        n: u32,
        k: u32,
    ) {
        #[cfg(test)]
        FUSED_DISPATCHES.fetch_add(1, Ordering::Relaxed);
        let p = [m, n, k];
        let grid = MTLSize {
            width: u64::from(n).div_ceil(64),
            height: u64::from(m).div_ceil(64),
            depth: 1,
        };
        let tg = MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        };
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.glu_gemm);
        encoder.set_buffer(0, Some(a), 0);
        encoder.set_buffer(1, Some(b), 0);
        encoder.set_buffer(2, Some(c), 0);
        encoder.set_bytes(3, 12, p.as_ptr().cast());
        encoder.dispatch_thread_groups(grid, tg);
        encoder.end_encoding();
    }

    pub(super) fn linear_half(
        &self,
        model: &Model,
        command: &CommandBufferRef,
        parameter: usize,
        input: &BufferRef,
        output: &BufferRef,
    ) -> Result<(), String> {
        let p = &model.parameters[parameter];
        let (n, k) = (p.shape[0], p.shape[1]);
        let tw = model.transposed_weights[parameter]
            .as_ref()
            .ok_or_else(|| format!("Missing FP16 transpose for parameter {parameter}"))?;
        self.matmul.borrow_mut().encode(
            &model.runtime.device,
            command,
            Matrix::half(input, self.rows, k),
            Matrix::half(tw, k, n),
            Matrix::half(output, self.rows, n),
            false,
            1.0,
        )
    }

    pub(super) fn rms_half(
        &self,
        command: &CommandBufferRef,
        input: &BufferRef,
        gamma: &BufferRef,
        output: &BufferRef,
        width: u32,
        epsilon: f32,
    ) {
        let p = NormParams { width, epsilon };
        dispatch(
            command,
            &self.rms_half,
            &[input, gamma, output],
            &p,
            thread_group(self.rows as u64),
        );
    }

    pub(super) fn attention_half(
        &self,
        model: &Model,
        command: &CommandBufferRef,
        layer: u32,
    ) -> Result<(), String> {
        let c = model.config;
        let p = Params {
            length: self.length,
            heads: c.heads,
            kv_heads: c.kv_heads,
            dim: c.width / c.heads,
            start: 0,
            block: 0,
            window: c.attention(layer).window.unwrap_or(0),
            key_start: 0,
            key_rows: 0,
            epsilon: c.epsilon,
            rope_base: c.rope_base,
        };
        let groups = MTLSize {
            width: c.heads as u64,
            height: self.rows as u64,
            depth: 1,
        };
        if model.optimized && c.width == 1536 && p.dim == 96 {
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
                &p,
                groups,
            );
        } else {
            dispatch(
                command,
                &self.prepare_half,
                &[
                    &self.qkv_half[0],
                    &self.qkv_half[1],
                    &self.qkv_half[2],
                    &self.head_major[0],
                    &self.head_major[1],
                    &self.head_major[2],
                ],
                &p,
                groups,
            );
        }
        let heads = self.batch * c.heads;
        let encoder = command.new_compute_command_encoder();
        let is_fast_96 = p.dim == 96 && model.optimized && c.width == 1536;
        let pipeline = if is_fast_96 {
            &self.flash_attention_half_96
        } else {
            &self.flash_attention_half
        };
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&self.head_major[0]), 0);
        encoder.set_buffer(1, Some(&self.head_major[1]), 0);
        encoder.set_buffer(2, Some(&self.head_major[2]), 0);
        encoder.set_buffer(3, Some(&self.gates_half), 0);
        encoder.set_buffer(4, Some(&self.branch_half), 0);
        encoder.set_bytes(
            5,
            std::mem::size_of::<Params>() as u64,
            (&p as *const Params).cast(),
        );
        let (tile_q, threads) = if is_fast_96 { (32, 128) } else { (16, 64) };
        let grid_size = MTLSize {
            width: u64::from(self.length).div_ceil(tile_q),
            height: u64::from(heads),
            depth: 1,
        };
        let tg_size = MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(grid_size, tg_size);
        encoder.end_encoding();
        Ok(())
    }

    pub(super) fn attention(
        &self,
        model: &Model,
        command: &CommandBufferRef,
        layer: u32,
    ) -> Result<(), String> {
        let c = model.config;
        let mut p = Params {
            length: self.length,
            heads: c.heads,
            kv_heads: c.kv_heads,
            dim: c.width / c.heads,
            start: 0,
            block: 0,
            window: c.attention(layer).window.unwrap_or(0),
            key_start: 0,
            key_rows: 0,
            epsilon: c.epsilon,
            rope_base: c.rope_base,
        };
        let groups = MTLSize {
            width: c.heads as u64,
            height: self.rows as u64,
            depth: 1,
        };
        dispatch(
            command,
            &self.prepare,
            &[
                &self.qkv[0],
                &self.qkv[1],
                &self.qkv[2],
                &self.head_major[0],
                &self.head_major[1],
                &self.head_major[2],
            ],
            &p,
            groups,
        );
        let heads = self.batch * c.heads;
        if model.optimized && p.dim == 128 {
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.flash_attention);
            encoder.set_buffer(0, Some(&self.head_major[0]), 0);
            encoder.set_buffer(1, Some(&self.head_major[1]), 0);
            encoder.set_buffer(2, Some(&self.head_major[2]), 0);
            encoder.set_buffer(3, Some(&self.gates), 0);
            encoder.set_buffer(4, Some(&self.branch), 0);
            encoder.set_bytes(
                5,
                std::mem::size_of::<Params>() as u64,
                (&p as *const Params).cast(),
            );
            let grid_size = MTLSize {
                width: u64::from(self.length).div_ceil(16),
                height: u64::from(heads),
                depth: 1,
            };
            let tg_size = MTLSize {
                width: 64,
                height: 1,
                depth: 1,
            };
            encoder.dispatch_thread_groups(grid_size, tg_size);
            encoder.end_encoding();
            return Ok(());
        }
        let full_half = |buffer| {
            Matrix::half(buffer, self.length, p.dim).layout(
                heads,
                u64::from(self.length) * u64::from(p.dim),
                0,
            )
        };
        for start in (0..self.length).step_by(QUERY_BLOCK as usize) {
            p.start = start;
            p.block = QUERY_BLOCK.min(self.length - start);
            p.key_start = if p.window == 0 {
                0
            } else {
                (start + 1).saturating_sub(p.window)
            };
            p.key_rows = start + p.block - p.key_start;
            let query =
                |buffer| full_half(buffer).row_view(p.block, u64::from(start) * u64::from(p.dim));
            let scores = Matrix::half(&self.scores, p.block, p.key_rows).layout(
                heads,
                u64::from(p.block) * u64::from(p.key_rows),
                0,
            );
            let keys = |buffer| {
                full_half(buffer).row_view(p.key_rows, u64::from(p.key_start) * u64::from(p.dim))
            };
            self.matmul.borrow_mut().encode(
                &model.runtime.device,
                command,
                query(&self.head_major[0]),
                keys(&self.head_major[1]),
                scores,
                true,
                f64::from(1.0 / (p.dim as f32).sqrt()),
            )?;
            dispatch(
                command,
                &self.softmax,
                &[&self.scores],
                &p,
                MTLSize {
                    width: heads as u64,
                    height: p.block as u64,
                    depth: 1,
                },
            );
            self.matmul.borrow_mut().encode(
                &model.runtime.device,
                command,
                scores,
                keys(&self.head_major[2]),
                query(&self.head_major[3]),
                false,
                1.0,
            )?;
        }
        dispatch(
            command,
            &self.unpack,
            &[&self.head_major[3], &self.gates, &self.branch],
            &p,
            groups,
        );
        Ok(())
    }
}
