use super::*;

impl FbtModel {
    /// Reuse bounded vocabulary scratch. Sampling is keyed by absolute row,
    /// so changing chunk boundaries preserves the generated token sequence.
    pub(super) fn readout(&mut self, temperature: f32, seed: u64) -> CudaResult<()> {
        let capacity = self.scratch.logits.len() / VOCAB;
        // Prefix logits before the final prompt position cannot contribute to
        // generation. Keep absolute row seeds for the positions we do sample.
        let start = self.cache.as_ref().map_or(0, |cache| {
            cache.sample_from.saturating_sub(self.first).min(self.rows)
        });
        for first in (start..self.rows).step_by(capacity) {
            let count = capacity.min(self.rows - first);
            matmul_at(
                self.synth.as_ref(),
                &self.module,
                &self.stream,
                &self.scratch.normalized,
                &self.weights.readout,
                &mut self.scratch.logits,
                count,
                VOCAB,
                WIDTH,
                first,
            )?;
            if let Some(state) = &mut self.diffusion {
                let launch = state
                    .module
                    .prepare_diffusion_sample(LaunchConfig1D::new(count as u32, 256, 0))
                    .map_err(cuda_error)?;
                state
                    .module
                    .diffusion_sample(
                        &self.stream,
                        &launch,
                        &self.scratch.logits,
                        &mut self.scratch.sampled,
                        &mut state.confidence,
                        count as u32,
                        temperature,
                        seed,
                        (self.first + first) as u32,
                    )
                    .map_err(cuda_error)?;
            } else {
                let launch = self
                    .module
                    .prepare_sample(LaunchConfig1D::new(count as u32, 256, 0))
                    .map_err(cuda_error)?;
                self.module
                    .sample(
                        &self.stream,
                        &launch,
                        &self.scratch.logits,
                        &mut self.scratch.sampled,
                        count as u32,
                        temperature,
                        seed,
                        (self.first + first) as u32,
                    )
                    .map_err(cuda_error)?;
            }
        }
        Ok(())
    }
}
