use super::*;

impl Pisa1 {
    pub(crate) fn index_counts(&self) -> [usize; 2] {
        self.indexed
            .get()
            .and_then(|result| result.as_ref().ok())
            .map_or([0; 2], |kernels| kernels.index_counts())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn denoise(
        &self,
        encoder: &ComputeCommandEncoderRef,
        qkv: &BufferRef,
        kv: &BufferRef,
        tree: &BufferRef,
        start: u32,
        rows: u32,
        policy: crate::context_metal::IndexPolicy<'_>,
        attention: bool,
    ) -> Result<(), String> {
        let kernels = self
            .indexed
            .get_or_init(|| {
                let runtime = Runtime::shared()?;
                crate::context_metal::ContextKernels::with_index(
                    runtime.as_ref(),
                    crate::context::Layout::new(self.context, 4096)?,
                    true,
                )
            })
            .as_ref()
            .map_err(Clone::clone)?;
        kernels.encode_policy(
            encoder,
            qkv,
            kv,
            tree,
            &self.blocks,
            attention.then_some(&*self.output),
            start,
            rows,
            Some(policy),
        )
    }

    pub(crate) fn cached_attention(
        &self,
        encoder: &ComputeCommandEncoderRef,
        qkv: &BufferRef,
        kv: &BufferRef,
        tree: &BufferRef,
        start: u32,
        rows: u32,
    ) -> Result<(), String> {
        let kernels = self
            .cached
            .get_or_init(|| {
                let runtime = Runtime::shared()?;
                crate::context_metal::ContextKernels::new(
                    runtime.as_ref(),
                    crate::context::Layout::new(self.context, 4096)?,
                )
            })
            .as_ref()
            .map_err(Clone::clone)?;
        kernels.encode(
            encoder,
            qkv,
            kv,
            tree,
            &self.blocks,
            &self.output,
            start,
            rows,
        )
    }
}
