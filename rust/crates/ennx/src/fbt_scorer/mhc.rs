use super::*;

impl Scorer<'_> {
    fn mhc_site(execution: u32, attention: bool) -> u32 {
        execution * 2 + u32::from(!attention)
    }

    fn mhc_coefficients(
        &self,
        encoder: &ComputeCommandEncoderRef,
        stream: &BufferRef,
        stream_offset: u64,
        predictor: u64,
        bias: u64,
        control: u64,
        output: u64,
        shape: &[u32; 4],
    ) {
        encoder.set_compute_pipeline_state(&self.pipelines.mhc_scale);
        encoder.set_buffer(0, Some(stream), stream_offset);
        encoder.set_buffer(1, Some(self.weights.buffer), control);
        encoder.set_buffer(2, Some(&self.buffers.mhc_coefficients), output);
        encoder.set_bytes(
            3,
            std::mem::size_of_val(shape) as u64,
            shape.as_ptr().cast(),
        );
        encoder.dispatch_thread_groups(thread_group(u64::from(self.rows)), thread_group(256));
        encoder.memory_barrier_with_resources(&[&self.buffers.mhc_coefficients]);
        encoder.set_compute_pipeline_state(&self.pipelines.mhc_predict_rows);
        encoder.set_buffer(0, Some(stream), stream_offset);
        encoder.set_buffer(1, Some(self.weights.buffer), predictor);
        encoder.set_buffer(2, Some(self.weights.buffer), bias);
        encoder.set_buffer(3, Some(self.weights.buffer), control);
        encoder.set_buffer(4, Some(&self.buffers.mhc_coefficients), output);
        encoder.set_bytes(
            5,
            std::mem::size_of_val(shape) as u64,
            shape.as_ptr().cast(),
        );
        encoder.dispatch_thread_groups(
            thread_group(u64::from(self.rows.div_ceil(64))),
            thread_group(128),
        );
        encoder.memory_barrier_with_resources(&[&self.buffers.mhc_coefficients]);
    }

    pub(super) fn mhc_prepare(
        &self,
        layer: u32,
        execution: u32,
        attention: bool,
    ) -> Result<(), String> {
        let encoder = self.active_encoder();
        let site = Self::mhc_site(execution, attention);
        let stream = (site % 2) as usize;
        let stream_offset = half_bytes(u64::from(self.scratch_start() * MHC_STREAMS * WIDTH));
        let coefficient_offset =
            u64::from(self.scratch_start() * MHC_COEFFICIENTS) * std::mem::size_of::<f32>() as u64;
        let predictor_stride = u64::from(MHC_INPUT * MHC_COEFFICIENTS);
        let bias_stride = u64::from(MHC_COEFFICIENTS);
        let (predictor, bias, control, norm) = if attention {
            (
                self.weights.mhc_attention_predictor,
                self.weights.mhc_attention_bias,
                self.weights.mhc_attention_control,
                self.weights.attention_norm,
            )
        } else {
            (
                self.weights.mhc_moe_predictor,
                self.weights.mhc_moe_bias,
                self.weights.mhc_moe_control,
                self.weights.ffn_norm,
            )
        };
        let shape = [self.rows, WIDTH, self.weights.architecture.kernel_code(), 0];
        self.mhc_coefficients(
            encoder,
            &self.buffers.mhc_streams[stream],
            stream_offset,
            predictor + half_bytes(u64::from(layer) * predictor_stride),
            bias + half_bytes(u64::from(layer) * bias_stride),
            control + half_bytes(u64::from(layer) * 4),
            coefficient_offset,
            &shape,
        );
        encoder.set_compute_pipeline_state(&self.pipelines.mhc_mix_rms);
        encoder.set_buffer(0, Some(&self.buffers.mhc_streams[stream]), stream_offset);
        encoder.set_buffer(1, Some(&self.buffers.mhc_coefficients), coefficient_offset);
        encoder.set_buffer(
            2,
            Some(self.weights.buffer),
            norm + half_bytes(u64::from(layer * WIDTH)),
        );
        encoder.set_buffer(3, Some(&self.buffers.normalized), self.activation_offset());
        encoder.set_bytes(
            4,
            std::mem::size_of_val(&shape) as u64,
            shape.as_ptr().cast(),
        );
        encoder.dispatch_thread_groups(thread_group(u64::from(self.rows)), thread_group(128));
        encoder.memory_barrier_with_resources(&[&self.buffers.normalized]);
        Ok(())
    }

    pub(super) fn mhc_update(
        &self,
        layer: u32,
        execution: u32,
        attention: bool,
        branch: &BufferRef,
    ) -> Result<(), String> {
        let encoder = self.active_encoder();
        let site = Self::mhc_site(execution, attention);
        let source = (site % 2) as usize;
        let destination = 1 - source;
        let stream_offset = half_bytes(u64::from(self.scratch_start() * MHC_STREAMS * WIDTH));
        let coefficient_offset =
            u64::from(self.scratch_start() * MHC_COEFFICIENTS) * std::mem::size_of::<f32>() as u64;
        let shape = [self.rows, WIDTH, self.weights.architecture.kernel_code(), 0];
        encoder.set_compute_pipeline_state(&self.pipelines.mhc_update);
        encoder.set_buffer(0, Some(&self.buffers.mhc_streams[source]), stream_offset);
        encoder.set_buffer(1, Some(branch), self.activation_offset());
        encoder.set_buffer(2, Some(&self.buffers.mhc_coefficients), coefficient_offset);
        encoder.set_buffer(
            3,
            Some(&self.buffers.mhc_streams[destination]),
            stream_offset,
        );
        encoder.set_bytes(
            4,
            std::mem::size_of_val(&shape) as u64,
            shape.as_ptr().cast(),
        );
        encoder.dispatch_threads(
            thread_group(u64::from(self.rows * WIDTH / 4)),
            thread_group(
                self.pipelines
                    .mhc_update
                    .max_total_threads_per_threadgroup()
                    .min(256),
            ),
        );
        encoder.memory_barrier_with_resources(&[&self.buffers.mhc_streams[destination]]);
        Ok(())
    }

    pub(super) fn mhc_finalize(&self) -> Result<(), String> {
        let encoder = self.active_encoder();
        let shape = [self.rows, WIDTH, self.weights.architecture.kernel_code(), 0];
        encoder.set_compute_pipeline_state(&self.pipelines.mhc_mean_rms);
        encoder.set_buffer(
            0,
            Some(&self.buffers.mhc_streams[0]),
            half_bytes(u64::from(self.scratch_start() * MHC_STREAMS * WIDTH)),
        );
        encoder.set_buffer(1, Some(self.weights.buffer), self.weights.final_norm);
        encoder.set_buffer(2, Some(&self.buffers.normalized), self.activation_offset());
        encoder.set_bytes(
            3,
            std::mem::size_of_val(&shape) as u64,
            shape.as_ptr().cast(),
        );
        encoder.dispatch_thread_groups(thread_group(u64::from(self.rows)), thread_group(128));
        encoder.memory_barrier_with_resources(&[&self.buffers.normalized]);
        Ok(())
    }

    pub(super) fn feed_mhc(&self, layer: u32) -> Result<(), String> {
        let pipelines = self.pipelines;
        let buffers = self.buffers;
        let weights = self.weights;
        let activation_offset = self.activation_offset();
        let (router_offset, gate_up_offset, down_offset) = ffn_offsets(layer);
        let fine = &pipelines.fine_grained;
        let fine_buffers = &buffers.fine_grained;
        self.start_stage("route")?;
        fine.route_rows(
            self.active_encoder(),
            fine_buffers,
            &buffers.normalized,
            activation_offset,
            weights.buffer,
            weights.router + router_offset,
            self.rows,
        )?;
        self.start_stage("gate")?;
        let gate_weights = if fine.int8_gate() {
            buffers.quantized_gateup()?
        } else if fine.interleaved_gate() {
            buffers.interleaved_gateup()?
        } else {
            weights.buffer
        };
        let gate_offset = if fine.int8_gate() {
            gate_up_offset / 2
        } else if fine.interleaved_gate() {
            gate_up_offset
        } else {
            weights.gate_up + gate_up_offset
        };
        fine.gate_rows(
            self.active_encoder(),
            fine_buffers,
            &buffers.normalized,
            activation_offset,
            gate_weights,
            gate_offset,
            self.rows,
        )?;
        self.start_stage("down")?;
        fine.down_rows(
            self.active_encoder(),
            fine_buffers,
            weights.buffer,
            weights.down + down_offset,
            self.rows,
        )?;
        self.start_stage("combine_branch")?;
        fine.combine_branch(
            self.active_encoder(),
            fine_buffers,
            (&buffers.output, activation_offset),
            self.rows,
        );
        Ok(())
    }
}
