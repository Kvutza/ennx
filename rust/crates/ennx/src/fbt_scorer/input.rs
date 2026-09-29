use super::*;

impl Scorer<'_> {
    pub(super) fn embed(&self) -> Result<(), String> {
        let encoder = self.active_encoder();
        let pipelines = self.pipelines;
        let buffers = self.buffers;
        let weights = self.weights;
        let activation_offset = self.activation_offset();
        encoder.set_compute_pipeline_state(if self.diffusion.is_some() {
            &pipelines.denoise_embed
        } else {
            &pipelines.embed
        });
        encoder.set_buffer(0, Some(weights.buffer), weights.readout);
        encoder.set_buffer(
            1,
            Some(&buffers.tokens),
            u64::from(self.row_start) * std::mem::size_of::<u32>() as u64,
        );
        encoder.set_buffer(2, Some(&buffers.input), activation_offset);
        if let Some(input) = self.diffusion {
            encoder.set_buffer(3, Some(weights.buffer), weights.mask_embed);
            encoder.set_buffer(4, Some(input.confidence), u64::from(self.row_start) * 4);
            encoder.dispatch_thread_groups(thread_group(u64::from(self.rows)), thread_group(128));
        } else {
            encoder.dispatch_threads(
                thread_group(u64::from(self.rows) * u64::from(WIDTH)),
                thread_group(pipelines.embed.max_total_threads_per_threadgroup().min(256)),
            );
        }
        encoder.memory_barrier_with_resources(&[&buffers.input]);

        if weights.architecture.is_multistream() {
            let shape = [self.rows, WIDTH, weights.architecture.kernel_code(), 0];
            encoder.set_compute_pipeline_state(&pipelines.mhc_replicate);
            encoder.set_buffer(0, Some(&buffers.input), activation_offset);
            encoder.set_buffer(
                1,
                Some(&buffers.mhc_streams[0]),
                half_bytes(u64::from(self.scratch_start() * MHC_STREAMS * WIDTH)),
            );
            encoder.set_bytes(
                2,
                std::mem::size_of_val(&shape) as u64,
                shape.as_ptr().cast(),
            );
            encoder.dispatch_threads(
                thread_group(u64::from(self.rows * WIDTH)),
                thread_group(
                    pipelines
                        .mhc_replicate
                        .max_total_threads_per_threadgroup()
                        .min(256),
                ),
            );
            encoder.memory_barrier_with_resources(&[&buffers.mhc_streams[0]]);
            return Ok(());
        }

        rms_offsets(
            &encoder,
            pipelines,
            (&buffers.input, activation_offset),
            (weights.buffer, weights.attention_norm),
            (&buffers.normalized, activation_offset),
            self.rows,
        );
        encoder.memory_barrier_with_resources(&[&buffers.normalized]);
        Ok(())
    }

    pub(super) fn layer(
        &self,
        input: &BufferRef,
        layer: u32,
        pass: u32,
        execution: u32,
    ) -> Result<(), String> {
        if self.weights.architecture.is_multistream() {
            self.start_stage("mhc_attention_input")?;
            self.mhc_prepare(layer, execution, true)?;
            self.attention(layer, execution)?;
            self.start_stage("mhc_attention_update")?;
            self.mhc_update(layer, execution, true, &self.buffers.projected)?;
            self.start_stage("mhc_moe_input")?;
            self.mhc_prepare(layer, execution, false)?;
            self.feed_mhc(layer)?;
            self.start_stage("mhc_moe_update")?;
            self.mhc_update(layer, execution, false, &self.buffers.output)?;
            return Ok(());
        }
        self.attention(layer, execution)?;
        self.feed_forward(input, layer, pass)
    }
}
