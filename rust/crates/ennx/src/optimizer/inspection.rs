use super::*;

impl Optimizer {
    /// Get current telemetry.
    pub fn telemetry(&self) -> &Telemetry {
        &self.telemetry
    }

    /// Get bounds.
    pub fn bounds(&self) -> &Array2<f64> {
        &self.bounds
    }

    /// Get number of dimensions.
    pub fn num_dim(&self) -> usize {
        self.num_dim
    }

    /// Get configuration.
    pub fn config(&self) -> &OptimizerConfig {
        &self.config
    }

    /// Get trust region state.
    pub fn trust_region(&self) -> &TrustRegionState {
        &self.tr_state
    }

    /// Get mutable trust region state.
    pub fn trust_mut(&mut self) -> &mut TrustRegionState {
        &mut self.tr_state
    }

    /// Trust region length (TuRBO or Morbo inner).
    pub fn tr_length(&self) -> f64 {
        self.tr_state.length()
    }

    /// Row-level observation access (ENN surrogate or fallback store).
    pub fn obs_access(&self) -> obs_access::ObsAccess<'_> {
        obs_access::ObsAccess::new(self)
    }

    /// Get surrogate.
    pub fn surrogate(&self) -> Option<&(dyn Surrogate + Send + Sync)> {
        self.surrogate.as_ref().map(|s| s.as_ref())
    }

    /// Get mutable surrogate.
    pub fn surrogate_mut(&mut self) -> Option<&mut (dyn Surrogate + Send + Sync)> {
        match self.surrogate.as_mut() {
            Some(s) => Some(s.as_mut()),
            None => None,
        }
    }

    /// Get observations in unit space (ENN model or fallback store).
    pub fn x_obs(&self) -> Option<Array2<f64>> {
        if let Some(surrogate) = self.surrogate.as_ref() {
            return surrogate.observations_x().ok().flatten();
        }
        if self.fallback_x.is_empty() {
            return None;
        }
        Some(obs_access::stack_rows(&self.fallback_x))
    }

    /// Get observation values (ENN model or fallback store).
    pub fn y_obs(&self) -> Option<Array2<f64>> {
        if let Some(surrogate) = self.surrogate.as_ref() {
            return surrogate.observations_y().ok().flatten();
        }
        if self.fallback_y.is_empty() {
            return None;
        }
        Some(obs_access::stack_rows(&self.fallback_y))
    }

    /// Get incumbent x in unit space.
    pub fn unit_incumbent(&self) -> Option<&Array1<f64>> {
        self.unit_incumbent.as_ref()
    }

    /// Get incumbent y scalar.
    pub fn y_scalar(&self) -> Option<&Array1<f64>> {
        self.y_scalar.as_ref()
    }

    /// Increment restart generation.
    pub fn increment_generation(&mut self) {
        self.restart_generation += 1;
    }

    /// Get restart generation.
    pub fn restart_generation(&self) -> usize {
        self.restart_generation
    }

    /// Get sobol engine.
    pub fn sobol_mut(&mut self) -> Option<&mut SobolEngine> {
        self.sobol_engine.as_mut()
    }

    /// Get sobol seed base.
    pub fn sobol_base(&self) -> u64 {
        self.sobol_base
    }

    /// Get init progress from strategy.
    pub fn init_progress(&self) -> Option<(usize, usize)> {
        self.strategy.init_progress()
    }

    /// Current number of stored observations.
    pub fn obs_count(&self) -> usize {
        if let Some(surrogate) = self.surrogate.as_ref() {
            return surrogate.observation_count().unwrap_or(0);
        }
        self.fallback_x.len()
    }
}
