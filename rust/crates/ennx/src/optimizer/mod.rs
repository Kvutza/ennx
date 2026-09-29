//! Optimizer state machine for ask/tell pattern.

mod incumbent;
mod inspection;
pub mod multi_tr;
pub mod obs_access;
mod observation_delta;
mod tr_state;

pub use multi_tr::{
    MultiTrustRegionConfig, MultiTrustRegionState, RegionBatch, RegionCandidate, SharingPolicy,
};
pub use observation_delta::ObservationDelta;

use ndarray::{Array1, Array2, ArrayView2};
use rand::RngCore;

use crate::candidates::SobolEngine;
use crate::config::{InitStrategy, OptimizerConfig, SurrogateConfig};
use crate::error::ENNError;
use crate::incumbent_tracker::{IncumbentTracker, enn_k, tracker_surrogate};
use crate::strategy::Strategy;
use crate::surrogate::{BoxedSurrogate, ENNSurrogate, Surrogate};
use tr_state::TrustRegionState;

/// Telemetry for timing.
#[derive(Debug, Clone, Default)]
pub struct Telemetry {
    pub dt_fit: f64,
    pub dt_gen: f64,
    pub dt_sel: f64,
    pub dt_tell: f64,
    pub num_candidates: usize,
}

/// Optimizer state machine.
pub struct Optimizer {
    bounds: Array2<f64>,
    num_dim: usize,
    config: OptimizerConfig,
    tr_state: TrustRegionState,
    surrogate: Option<BoxedSurrogate>,
    strategy: Strategy,
    pub(crate) fallback_x: Vec<Array1<f64>>,
    pub(crate) fallback_y: Vec<Array1<f64>>,
    incumbent_idx: Option<usize>,
    unit_incumbent: Option<Array1<f64>>,
    y_scalar: Option<Array1<f64>>,
    restart_generation: usize,
    sobol_engine: Option<SobolEngine>,
    sobol_base: u64,
    telemetry: Telemetry,
    incumbent_tracker: IncumbentTracker,
}

impl Optimizer {
    /// Create a new optimizer.
    pub fn new(
        bounds: Array2<f64>,
        config: OptimizerConfig,
        rng: &mut dyn RngCore,
    ) -> Result<Self, ENNError> {
        Self::new_strategy(bounds, config, Strategy::hybrid(InitStrategy::LHD, 10), rng)
    }

    /// Create a new optimizer with an explicit strategy.
    pub fn new_strategy(
        bounds: Array2<f64>,
        config: OptimizerConfig,
        strategy: Strategy,
        rng: &mut dyn RngCore,
    ) -> Result<Self, ENNError> {
        Self::new_inner(bounds, config, strategy, None, rng)
    }

    /// Create an optimizer whose surrogate is supplied by an external frontend.
    ///
    /// This is the coarse extension boundary used by language bindings: the
    /// optimizer, trust region, candidates, acquisition, and observations remain
    /// native while the supplied surrogate performs batched fit/predict/sample
    /// operations.
    pub fn new_surrogate(
        bounds: Array2<f64>,
        config: OptimizerConfig,
        strategy: Strategy,
        surrogate: BoxedSurrogate,
        rng: &mut dyn RngCore,
    ) -> Result<Self, ENNError> {
        Self::new_inner(bounds, config, strategy, Some(surrogate), rng)
    }

    fn new_inner(
        bounds: Array2<f64>,
        config: OptimizerConfig,
        strategy: Strategy,
        provided_surrogate: Option<BoxedSurrogate>,
        rng: &mut dyn RngCore,
    ) -> Result<Self, ENNError> {
        let num_dim = bounds.nrows();
        if bounds.ncols() != 2 {
            return Err(ENNError::InvalidShape {
                expected: vec![num_dim, 2],
                got: vec![num_dim, bounds.ncols()],
            });
        }

        let mut tr_state = TrustRegionState::from_config(num_dim, &config.trust_region, rng)
            .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
        if let Some(dim) = config.failure_tolerance_dim {
            if let TrustRegionState::Turbo(t) = &mut tr_state {
                t.set_dim(dim);
            }
        }

        let surrogate = provided_surrogate.or_else(|| match &config.surrogate {
            SurrogateConfig::ENN(enn_config) => {
                Some(Box::new(ENNSurrogate::new(enn_config.clone())))
            }
            SurrogateConfig::None => None,
        });

        let sobol_engine =
            if config.candidates.candidate_rv == crate::candidates::CandidateRV::Sobol {
                let mut eng = SobolEngine::new(num_dim)?;
                eng.scramble(rng);
                Some(eng)
            } else {
                None
            };

        let mut seed_bytes = [0u8; 8];
        rng.fill_bytes(&mut seed_bytes);
        let sobol_base = u64::from_le_bytes(seed_bytes) % (1u64 << 31);
        let num_metrics = tr_state.num_metrics();
        let tracker_m = match &config.surrogate {
            SurrogateConfig::ENN(enn_config) => enn_k(enn_config.k),
            SurrogateConfig::None => tracker_surrogate(),
        };
        let noise_aware =
            config.noise_aware || tr_state.morbo().map(|m| m.noise_aware()).unwrap_or(false);
        let incumbent_tracker = IncumbentTracker::new(tracker_m, noise_aware, num_metrics);

        Ok(Self {
            bounds,
            num_dim,
            config,
            tr_state,
            surrogate,
            strategy,
            fallback_x: Vec::new(),
            fallback_y: Vec::new(),
            incumbent_idx: None,
            unit_incumbent: None,
            y_scalar: None,
            restart_generation: 0,
            sobol_engine,
            sobol_base,
            telemetry: Telemetry::default(),
            incumbent_tracker,
        })
    }

    /// Ask for candidates.
    pub fn ask(&mut self, num_arms: usize, rng: &mut dyn RngCore) -> Result<Array2<f64>, ENNError> {
        let start = std::time::Instant::now();

        let mut strategy = std::mem::replace(&mut self.strategy, Strategy::turbo());
        let mut telemetry = std::mem::take(&mut self.telemetry);
        let result = strategy.ask(self, num_arms, &mut telemetry, rng);
        self.strategy = strategy;
        self.telemetry = telemetry;

        self.telemetry.dt_gen = start.elapsed().as_secs_f64();
        if result.is_ok() {
            if let Some(surrogate) = self.surrogate.as_ref() {
                surrogate.schedule_flush()?;
            }
        }
        result
    }

    /// Tell observations.
    pub fn tell(
        &mut self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        rng: &mut dyn RngCore,
    ) -> Result<(), ENNError> {
        self.tell_variance(x, y, None, rng)
    }

    /// Tell observations with optional known observation variance.
    pub fn tell_variance(
        &mut self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
        rng: &mut dyn RngCore,
    ) -> Result<(), ENNError> {
        let start = std::time::Instant::now();

        if let Some(surrogate) = self.surrogate.as_ref() {
            surrogate.wait_flush()?;
        }

        let mut strategy = std::mem::replace(&mut self.strategy, Strategy::turbo());
        let mut telemetry = std::mem::take(&mut self.telemetry);
        let result = strategy.tell(self, x, y, yvar, &mut telemetry, rng);
        self.strategy = strategy;
        self.telemetry = telemetry;

        self.telemetry.dt_tell = start.elapsed().as_secs_f64();
        if result.is_ok() {
            // Soft-threshold drain for disk: tell-all never reaches ask between
            // rows, so schedule here (ask already schedules after propose).
            // Skip after bulk seed chunks: overlapping soft-sync with the next
            // ask regresses ask@1e6; bulk fit_append already ensure_sync'd.
            if x.nrows() < 64 {
                if let Some(surrogate) = self.surrogate.as_ref() {
                    surrogate.schedule_flush()?;
                }
            }
        }
        result
    }

    /// Add observations (internal).
    pub fn add_observations(
        &mut self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
    ) -> Result<ObservationDelta, ENNError> {
        let delta = self.prepare_observations(x, y)?;
        self.commit_observations(&delta);
        Ok(delta)
    }

    pub(crate) fn prepare_observations(
        &self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
    ) -> Result<ObservationDelta, ENNError> {
        if x.nrows() != y.nrows() {
            return Err(ENNError::InvalidShape {
                expected: vec![x.nrows(), y.ncols()],
                got: vec![y.nrows(), y.ncols()],
            });
        }
        let old_n = self.obs_count();
        observation_delta::observation_batch(old_n, x, y)
    }

    pub(crate) fn commit_observations(&mut self, delta: &ObservationDelta) {
        for i in 0..delta.x_new.nrows() {
            let y_row: Array1<f64> = delta.y_new.row(i).to_owned();
            self.incumbent_tracker.tell(delta.old_n + i, &y_row);
            if self.surrogate.is_none() {
                self.fallback_x.push(delta.x_new.row(i).to_owned());
                self.fallback_y.push(y_row);
            }
        }
    }
}

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_incremental;
#[cfg(test)]
mod testsmbinc;
#[cfg(test)]
mod testsmbnzawrinc;
