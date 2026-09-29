use super::*;

impl QwenEvaluator {
    pub(super) fn buffer(&self, slot: W) -> &Buffer {
        &self.workspace[slot as usize]
    }

    pub(super) fn write<T: Copy>(&self, slot: W, values: &[T]) {
        write_buffer(self.buffer(slot), values);
    }

    pub(super) fn read<T: Copy>(&self, slot: W, count: usize) -> Vec<T> {
        assert!(size_of::<T>() * count <= self.buffer(slot).length() as usize);
        unsafe {
            std::slice::from_raw_parts(self.buffer(slot).contents().cast::<T>(), count).to_vec()
        }
    }

    pub(super) fn trace_values(&self, label: &str, slot: W, count: usize) {
        self.trace_at(label, slot, 0, count);
    }

    pub(super) fn trace_at(&self, label: &str, slot: W, offset: usize, count: usize) {
        if std::env::var_os("ENNX_QWEN_TRACE_VALUES").is_none() {
            return;
        }
        let bytes = offset
            .checked_mul(size_of::<f32>())
            .expect("Qwen trace offset overflow");
        let buffer = self.buffer(slot);
        assert!(bytes + count * size_of::<f32>() <= buffer.length() as usize);
        let values: Vec<f32> = unsafe {
            std::slice::from_raw_parts(buffer.contents().cast::<u8>().add(bytes).cast(), count)
                .to_vec()
        };
        let finite = values.iter().filter(|value| value.is_finite()).count();
        let nonzero = values.iter().filter(|value| **value != 0.0).count();
        let peak = values
            .iter()
            .filter(|value| value.is_finite())
            .map(|value| value.abs())
            .fold(0.0f32, f32::max);
        trace(&format!(
            "Qwen {label}: finite={finite}/{count} nonzero={nonzero}/{count} peak={peak:e} first={:?}",
            &values[..values.len().min(4)]
        ));
    }

    pub(super) fn qwen_shape(&self, rows: u32) -> QwenShape {
        QwenShape {
            rows,
            width: self.config.hidden,
            heads: self.config.heads,
            kv_heads: self.config.kv_heads,
            head_dim: self.config.head_dim(),
            epsilon: self.config.epsilon,
            rope_theta: self.config.rope_theta,
            ..QwenShape::default()
        }
    }

    pub(super) fn flame_shape(&self, rows: u32) -> FlameShape {
        FlameShape {
            rows,
            width: self.config.hidden,
            heads: self.config.heads,
            epsilon: self.config.epsilon,
            rope_base: self.config.rope_theta,
            ..FlameShape::default()
        }
    }

    pub(super) fn encode<T>(
        &self,
        command: &CommandBufferRef,
        name: &str,
        buffers: &[(&Buffer, u64)],
        params: &T,
        groups: MTLSize,
    ) {
        self.encode_threads(command, name, buffers, params, groups, 256);
    }

    pub(super) fn encode_threads<T>(
        &self,
        command: &CommandBufferRef,
        name: &str,
        buffers: &[(&Buffer, u64)],
        params: &T,
        groups: MTLSize,
        threads: u64,
    ) {
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipelines[name]);
        for (index, &(buffer, offset)) in buffers.iter().enumerate() {
            encoder.set_buffer(index as u64, Some(buffer), offset);
        }
        encoder.set_bytes(
            buffers.len() as u64,
            size_of::<T>() as u64,
            (params as *const T).cast(),
        );
        encoder.dispatch_thread_groups(groups, thread_group(threads));
        encoder.end_encoding();
    }

    pub(super) fn elementwise<T>(
        &self,
        command: &CommandBufferRef,
        name: &str,
        buffers: &[(&Buffer, u64)],
        params: &T,
        count: usize,
    ) {
        self.encode(
            command,
            name,
            buffers,
            params,
            thread_group((count as u64).div_ceil(256)),
        );
    }

    pub(super) fn linear(
        &self,
        command: &CommandBufferRef,
        input: W,
        weights: &Buffer,
        offset: usize,
        output: W,
        rows: u32,
        inside: u32,
        outside: u32,
    ) {
        self.linear_buffers(
            command,
            self.buffer(input),
            0,
            weights,
            offset,
            self.buffer(output),
            0,
            rows,
            inside,
            outside,
        );
    }

    pub(super) fn qkv(
        &self,
        command: &CommandBufferRef,
        weights: &Buffer,
        rows: u32,
        layer: &Layer,
    ) {
        let c = self.config;
        let params = QwenQkvShape {
            rows,
            hidden: c.hidden,
            kv_width: c.kv_width(),
        };
        self.encode_threads(
            command,
            "qwen_qkv",
            &[
                (self.buffer(W::Norm), 0),
                (weights, layer.q_weight as u64 * 2),
                (weights, layer.k_weight as u64 * 2),
                (weights, layer.v_weight as u64 * 2),
                (weights, layer.q_bias as u64 * 2),
                (weights, layer.k_bias as u64 * 2),
                (weights, layer.v_bias as u64 * 2),
                (self.buffer(W::QRaw), 0),
                (self.buffer(W::KRaw), 0),
                (self.buffer(W::VRaw), 0),
            ],
            &params,
            thread_group((u64::from(c.hidden) + 2 * u64::from(c.kv_width())).div_ceil(32)),
            32,
        );
    }

    pub(super) fn silu_rows(
        &self,
        command: &CommandBufferRef,
        weights: &Buffer,
        rows: u32,
        layer: &Layer,
    ) {
        let c = self.config;
        let params = QwenMlpShape {
            rows,
            hidden: c.hidden,
            intermediate: c.intermediate,
        };
        self.encode_threads(
            command,
            "qwen_mlp_rows",
            &[
                (self.buffer(W::Norm), 0),
                (weights, layer.gate_weight as u64 * 2),
                (weights, layer.up_weight as u64 * 2),
                (self.buffer(W::Activation), 0),
            ],
            &params,
            thread_group(u64::from(c.intermediate).div_ceil(32)),
            32,
        );
    }

    pub(super) fn linear_buffers(
        &self,
        command: &CommandBufferRef,
        input: &Buffer,
        input_offset: u64,
        weights: &Buffer,
        offset: usize,
        output: &Buffer,
        output_offset: u64,
        rows: u32,
        inside: u32,
        outside: u32,
    ) {
        let params = Matmul {
            m: rows,
            n: outside,
            k: inside,
            transpose_b: 1,
            stride_a: 0,
            stride_b: 0,
            stride_c: 0,
        };
        let simdgroup = rows >= 64 && inside % 8 == 0 && outside % 32 == 0;
        let pipeline = if (1..=4).contains(&rows) {
            "qwen_gemv_rows"
        } else if rows < 32 {
            "qwen_gemv"
        } else if simdgroup {
            "qwen_simd_gemm"
        } else {
            "flame_linear"
        };
        let threads = if pipeline == "qwen_gemv_rows" || rows <= 4 {
            32
        } else if simdgroup {
            128
        } else {
            256
        };
        self.encode_threads(
            command,
            pipeline,
            &[
                (input, input_offset),
                (weights, offset as u64 * 2),
                (output, output_offset),
            ],
            &params,
            MTLSize {
                width: u64::from(outside).div_ceil(32),
                height: if simdgroup {
                    u64::from(rows).div_ceil(64)
                } else {
                    u64::from(rows).div_ceil(32)
                },
                depth: 1,
            },
            threads,
        );
    }

    pub(super) fn rms(
        &self,
        command: &CommandBufferRef,
        input: W,
        weights: &Buffer,
        offset: usize,
        output: W,
        rows: u32,
    ) {
        let params = self.qwen_shape(rows);
        self.encode(
            command,
            "qwen_rms",
            &[
                (self.buffer(input), 0),
                (weights, offset as u64 * 2),
                (self.buffer(output), 0),
            ],
            &params,
            thread_group(u64::from(rows)),
        );
    }

    pub(super) fn residual(&self, command: &CommandBufferRef, input: W, update: W, rows: u32) {
        let params = self.qwen_shape(rows);
        self.elementwise(
            command,
            "qwen_residual",
            &[(self.buffer(input), 0), (self.buffer(update), 0)],
            &params,
            rows as usize * self.config.hidden as usize,
        );
    }
}
