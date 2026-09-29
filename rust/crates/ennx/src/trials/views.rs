use super::*;

impl Search {
    pub fn row(&self, trial: Trial) -> Result<Vec<u8>, String> {
        let slot = self.row_slot(trial)?;
        self.engine.read(slot, self.row_bytes)
    }

    pub fn device_view(&self, trial: Trial) -> Result<DeviceView<'_>, String> {
        let slot = self.row_slot(trial)?;
        let _ = slot;
        match &self.engine {
            #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
            Engine::Cuda(engine) => {
                let (ptr, row_bytes, stream) = engine.device_row(slot)?;
                Ok(DeviceView::cuda(ptr, row_bytes, stream))
            }
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Engine::Metal(engine) => {
                let (buffer, offset) = engine.row_buffer(slot)?;
                Ok(DeviceView::metal(buffer, offset, self.row_bytes))
            }
            #[cfg(feature = "opencl")]
            Engine::OpenCl(engine) => {
                let row = engine.device_view(slot)?;
                Ok(DeviceView::opencl(
                    row.context(),
                    row.queue(),
                    row.buffer(),
                    row.offset(),
                    row.row_bytes(),
                ))
            }
            Engine::Cpu(_) => Err("pending row is not stored on a resident device".to_string()),
        }
    }

    pub fn byte_sum(&self, trial: Trial) -> Result<u64, String> {
        let slot = self.row_slot(trial)?;
        self.engine.byte_sum(slot, self.row_bytes)
    }

    /// Borrow the pending packed row through its CUDA device address.
    ///
    /// The returned address is owned by this search and becomes invalid when
    /// the search is dropped. The CUDA stream is synchronized before return.
    #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
    pub fn device_row(&self, trial: Trial) -> Result<(u64, usize, usize), String> {
        let slot = self.row_slot(trial)?;
        match &self.engine {
            Engine::Cuda(engine) => engine.device_row(slot),
            _ => Err("pending row is not stored on CUDA".to_string()),
        }
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
    pub fn device_batch(&self, trials: &[Trial]) -> Result<Vec<(u64, usize, usize)>, String> {
        let mut slots = Vec::with_capacity(trials.len());
        for trial in trials {
            slots.push(self.row_slot(*trial)?);
        }
        match &self.engine {
            Engine::Cuda(engine) => engine.device_rows(&slots),
            _ => Err("pending rows are not stored on CUDA".to_string()),
        }
    }

    /// Materialize a lazily selected trial into its resident row slot.
    ///
    /// Calling this for a trial returned by [`Search::ask`] is a no-op. Lazy
    /// trials are also materialized automatically by [`Search::tell`] before
    /// they are added to history.
    pub fn materialize_pending(&mut self, trial: Trial) -> Result<(), String> {
        let pending = self.pending_for(trial)?;
        if pending.materialized {
            return Ok(());
        }
        self.engine.materialize(
            self.base,
            pending.slot,
            pending.seed,
            &self.leaves,
            pending.length,
        )?;
        self.pending
            .iter_mut()
            .find(|candidate| candidate.id == trial.id)
            .expect("the pending trial was validated above")
            .materialized = true;
        Ok(())
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(crate) fn metal_row(&self, trial: Trial) -> Result<(::metal::Buffer, usize), String> {
        let slot = self.row_slot(trial)?;
        match &self.engine {
            Engine::Metal(engine) => engine.row_buffer(slot),
            _ => Err("the resident search is not using Metal".to_string()),
        }
    }

    #[cfg(feature = "opencl")]
    #[allow(dead_code)]
    pub(crate) fn opencl_row(&self, trial: Trial) -> Result<OpenClResidentRow<'_>, String> {
        let slot = self.row_slot(trial)?;
        match &self.engine {
            Engine::OpenCl(engine) => engine.device_view(slot),
            _ => Err("the resident search is not using OpenCL".to_string()),
        }
    }
}
