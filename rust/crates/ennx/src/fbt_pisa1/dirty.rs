use super::*;

impl Pisa1 {
    pub(crate) fn pyramid_range(
        &self,
        encoder: &ComputeCommandEncoderRef,
        qkv: &BufferRef,
        pyramid: &BufferRef,
        start: u32,
        rows: u32,
    ) {
        debug_assert!(rows > 0);
        debug_assert!(start + rows <= self.context);
        let first = start / BLOCK;
        let end = (start + rows).div_ceil(BLOCK);
        let range = [first, end - first, 0, 0];
        encoder.set_compute_pipeline_state(&self.pipelines.leaf_range);
        encoder.set_buffer(0, Some(qkv), 0);
        encoder.set_buffer(1, Some(pyramid), 0);
        encoder.set_bytes(
            2,
            std::mem::size_of_val(&range) as u64,
            range.as_ptr().cast(),
        );
        encoder.dispatch_thread_groups(
            thread_group(u64::from(end - first)),
            thread_group(u64::from(HEAD_DIM)),
        );
        encoder.memory_barrier_with_resources(&[pyramid]);
        encoder.set_compute_pipeline_state(&self.pipelines.upper_range);
        encoder.set_buffer(0, Some(pyramid), 0);
        encoder.set_bytes(
            1,
            std::mem::size_of_val(&range) as u64,
            range.as_ptr().cast(),
        );
        encoder.dispatch_thread_groups(thread_group(1), thread_group(u64::from(HEAD_DIM)));
        encoder.memory_barrier_with_resources(&[pyramid]);
    }

    pub(crate) fn attention_from(
        &self,
        encoder: &ComputeCommandEncoderRef,
        qkv: &BufferRef,
        pyramid: &BufferRef,
        start: u32,
        rows: u32,
    ) {
        debug_assert!(rows > 0);
        debug_assert!(start + rows <= ROWS);
        debug_assert_eq!(start % self.pipelines.query_tile, 0);
        debug_assert_eq!(rows % self.pipelines.query_tile, 0);
        let range = [start, rows];
        encoder.set_compute_pipeline_state(&self.pipelines.select_attention_q4);
        for (index, buffer) in [qkv, pyramid, &self.blocks, &self.output]
            .iter()
            .enumerate()
        {
            encoder.set_buffer(index as u64, Some(buffer), 0);
        }
        encoder.set_bytes(
            4,
            std::mem::size_of_val(&range) as u64,
            range.as_ptr().cast(),
        );
        encoder.dispatch_thread_groups(
            thread_group(u64::from(rows / self.pipelines.query_tile)),
            thread_group(u64::from(self.pipelines.query_tile * 32)),
        );
        encoder.memory_barrier_with_resources(&[&self.output]);
    }
}
