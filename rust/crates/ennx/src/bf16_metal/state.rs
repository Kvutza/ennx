use super::*;

impl SearchState {
    /// Start a fresh bounded ENN window from the current incumbent.
    pub(crate) fn compact_history(&mut self) -> Result<(), String> {
        self.check_idle()?;
        let missing = self.history != 0 && !self.identities[..self.history].contains(&self.base_id);
        if !self.implicit_history || self.history == 0 || (self.history < MAX_HISTORY && !missing) {
            return Ok(());
        }
        if self.exact_history {
            self.copy(&self.base, self.replay_origin.as_ref().unwrap())?;
            unsafe {
                std::ptr::write_bytes(
                    self.replay_steps.contents().cast::<u8>(),
                    0,
                    self.replay_steps.length() as usize,
                );
                std::ptr::write_bytes(
                    self.replay_scales.contents().cast::<u8>(),
                    0,
                    self.replay_scales.length() as usize,
                );
            }
        }
        self.copy(&self.base, &self.history_rows[0])?;
        self.resident_history = 1;
        self.resident_identities[0] = self.base_id;
        self.history = 1;
        self.objective_history.retain_incumbent();
        self.outcomes[0] = self.best;
        self.variances[0] = self.best_variance;
        self.identities[0] = self.base_id;
        self.pairwise_distances.fill(0.0);
        self.retain_axis();
        if let Some(family) = &mut self.family {
            family.components.fill([0.0; FAMILIES]);
            family.weights = [1.0; FAMILIES];
        }
        self.apply_family()?;
        self.fitted_enn = None;
        if let Some(threshold) = &mut self.threshold {
            threshold.compact(self.base_id);
        }
        Ok(())
    }
    pub fn restarts(&self) -> Result<usize, String> {
        self.check_healthy()?;
        Ok(self.restart_count)
    }
    pub fn set_profiling(&mut self, enabled: bool) {
        self.profiling = enabled;
        if !enabled {
            self.last_profile = None;
        }
    }
    pub fn last_profile(&self) -> Option<AskProfile> {
        self.last_profile
    }
    pub fn tell_profile(&self) -> Option<TellProfile> {
        self.tell_profile
    }
    pub fn memory_info(&self) -> MemoryInfo {
        MemoryInfo {
            row_bytes: self.row_bytes(),
            resident_bytes: self.resident_bytes,
            max_buffer_length: self.runtime.device.max_buffer_length(),
            recommended_max_working_set_size: self
                .runtime
                .device
                .recommended_max_working_set_size(),
            current_allocated_size: self.runtime.device.current_allocated_size(),
        }
    }

    pub fn controller_info(&self) -> Result<ControllerInfo, String> {
        self.check_healthy()?;
        if let Some(morbo) = self
            .objective_selector
            .as_ref()
            .and_then(|selector| selector.morbo.as_ref())
        {
            let trust = morbo.trust();
            return Ok(ControllerInfo {
                dimensions: self.dimensions,
                evaluated_arms: 1,
                length: trust.length(),
                length_min: self.length_config.length_min,
                length_max: self.length_config.length_max,
                success_tolerance: trust.success_tolerance(),
                failure_tolerance: trust.failure_tolerance(),
                success_counter: trust.success_counter(),
                failure_counter: trust.failure_counter(),
                restarts: self.restart_count,
            });
        }
        Ok(ControllerInfo {
            dimensions: self.dimensions,
            evaluated_arms: 1,
            length: self.length,
            length_min: self.length_config.length_min,
            length_max: self.length_config.length_max,
            success_tolerance: self.trust.success_tolerance(),
            failure_tolerance: self.trust.failure_tolerance(),
            success_counter: self.trust.success_counter(),
            failure_counter: self.trust.failure_counter(),
            restarts: self.restart_count,
        })
    }

    pub fn reliability_info(
        &self,
    ) -> Result<Option<reliability_region::ReliabilityTelemetry>, String> {
        self.check_healthy()?;
        Ok(self
            .reliability
            .as_ref()
            .map(reliability_region::ReliabilityController::telemetry))
    }

    pub(super) fn row_bytes(&self) -> u64 {
        self.dimensions as u64 * 2
    }
    pub(super) fn check_healthy(&self) -> Result<(), String> {
        if self.poisoned {
            Err("Metal BF16 state is unusable after a failed GPU update".into())
        } else {
            Ok(())
        }
    }
    pub(super) fn check_idle(&self) -> Result<(), String> {
        self.check_healthy()?;
        if self.pending.is_some() || self.queued.is_some() || self.async_command.is_some() {
            Err("Tell the pending Metal BF16 proposal and sync before another mutation".into())
        } else {
            Ok(())
        }
    }
}
