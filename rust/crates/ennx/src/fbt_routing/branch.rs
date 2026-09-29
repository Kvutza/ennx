use super::*;

impl FineGrainedMoePipelines {
    pub(crate) fn combine_branch(
        &self,
        encoder: &ComputeCommandEncoderRef,
        buffers: &FineGrainedMoeBuffers,
        output: (&BufferRef, u64),
        rows: u32,
    ) {
        let shape = Top3RouteShape::new(rows, WIDTH);
        encoder.set_compute_pipeline_state(&self.combine);
        let bindings: [(&BufferRef, u64); 5] = [
            (&buffers.route_weights, 0),
            (&buffers.packed_rows, 0),
            (&buffers.routed_output, 0),
            (&buffers.shared_output, 0),
            output,
        ];
        for (index, (buffer, offset)) in bindings.into_iter().enumerate() {
            encoder.set_buffer(index as u64, Some(buffer), offset);
        }
        encoder.set_bytes(
            5,
            std::mem::size_of::<Top3RouteShape>() as u64,
            (&shape as *const Top3RouteShape).cast(),
        );
        encoder.dispatch_thread_groups(
            thread_group(u64::from(rows)),
            thread_group(u64::from(WIDTH / 4)),
        );
        encoder.memory_barrier_with_resources(&[output.0]);
    }
}
