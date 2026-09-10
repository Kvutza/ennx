use super::Search;

impl Search {
    pub(super) fn row_slot(&self, trial: super::Trial) -> Result<usize, String> {
        let pending = self.pending_for(trial)?;
        if !pending.materialized {
            return Err("trial row is not materialized".into());
        }
        if self.device_state {
            let slot = self.state_word(4 + pending.slot)? as usize;
            if slot >= self.slots {
                return Err("trial has no resident row slot".into());
            }
            Ok(slot)
        } else {
            Ok(pending.slot)
        }
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(crate) fn metal_region(&mut self, buffer: ::metal::Buffer) {
        if let super::Engine::Metal(engine) = &mut self.engine {
            engine.bind_region(buffer);
        }
    }

    #[cfg(feature = "opencl")]
    pub(crate) fn opencl_context(&self) -> Option<&opencl3::context::Context> {
        match &self.engine {
            super::Engine::OpenCl(engine) => Some(&engine.context),
            _ => None,
        }
    }

    #[cfg(feature = "opencl")]
    pub(crate) fn opencl_region(&mut self, buffer: std::sync::Arc<opencl3::memory::Buffer<u64>>) {
        if let super::Engine::OpenCl(engine) = &mut self.engine {
            engine.bind_region(buffer);
        }
    }
}

impl Search {
    pub(crate) fn start_state(&mut self, value: f32) -> Result<(), String> {
        let _ = value;
        let started = match &mut self.engine {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            super::Engine::Metal(engine) => {
                engine.start_state(self.capacity, value)?;
                true
            }
            #[cfg(feature = "opencl")]
            super::Engine::OpenCl(engine) => {
                engine.start_state(self.capacity, value)?;
                true
            }
            _ => false,
        };
        if started {
            self.device_state = true;
        }
        Ok(())
    }

    pub(crate) fn observe(&mut self, trial: super::Trial, value: f32) -> Result<(), String> {
        if !value.is_finite() {
            return Err("trial value must be finite".into());
        }
        let slot = self.pending_for(trial)?.slot;
        let _ = slot;
        let observed = match &self.engine {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            super::Engine::Metal(engine) => {
                engine.observe(slot, value)?;
                true
            }
            #[cfg(feature = "opencl")]
            super::Engine::OpenCl(engine) => {
                engine.observe(slot, value)?;
                true
            }
            _ => false,
        };
        if !observed {
            return Err("search state is not resident".into());
        }
        self.pending.retain(|item| item.id != trial.id);
        Ok(())
    }

    pub(crate) fn state_word(&self, index: usize) -> Result<u32, String> {
        let _ = index;
        match &self.engine {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            super::Engine::Metal(engine) => engine.state_word(index),
            #[cfg(feature = "opencl")]
            super::Engine::OpenCl(engine) => engine.state_word(index),
            _ => Err("search state is not resident".into()),
        }
    }

    pub(super) fn history_bound(&self) -> usize {
        if self.device_state {
            self.capacity
        } else {
            self.history.len()
        }
    }

    pub(super) fn history_view(&self) -> Vec<(usize, f32)> {
        if self.device_state {
            Vec::new()
        } else {
            self.history
                .iter()
                .map(|record| (record.slot, record.value))
                .collect()
        }
    }
}
