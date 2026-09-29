//! Optimization strategies for ask/tell pattern.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rand::RngCore;
use rand::seq::SliceRandom;

use crate::util::argmax_tie;

mod selection;
use selection::*;

use crate::acquisition::{ParetoAcquisition, RandomAcquisition, UCBAcquisition};
use crate::candidates::{generate_candidates, generate_lhd, generate_uniform};
use crate::config::{AcquisitionConfig, InitStrategy};
use crate::error::ENNError;
use crate::optimizer::{
    MultiTrustRegionConfig, MultiTrustRegionState, Optimizer, SharingPolicy, Telemetry,
};

/// Strategy state for initialization phase.
#[derive(Debug, Clone)]
pub struct InitStrategyState {
    pub strategy_type: InitStrategy,
    pub num_init: usize,
    pub completed: usize,
}

impl InitStrategyState {
    pub fn new(strategy_type: InitStrategy, num_init: usize) -> Self {
        Self {
            strategy_type,
            num_init,
            completed: 0,
        }
    }
}

/// Strategy state for TuRBO normal phase.
#[derive(Debug, Clone, Default)]
pub struct TurboStrategyState;

/// Strategy state for the experimental multi-trust-region phase.
#[derive(Debug, Clone)]
pub struct StrategyState {
    pub init: InitStrategyState,
    pub multi_tr: MultiTrustRegionState,
    pub in_init: bool,
    pub pending_regions: Option<Vec<usize>>,
}

#[derive(Clone, Copy)]
struct CandidateSegment {
    start: usize,
    end: usize,
    arms: usize,
    region: usize,
}

impl StrategyState {
    pub fn new(
        num_dim: usize,
        num_regions: usize,
        num_init: usize,
        rng: &mut dyn RngCore,
    ) -> Result<Self, ENNError> {
        let mut config = MultiTrustRegionConfig::new(num_regions, Default::default());
        config.sharing_policy = SharingPolicy::Shared;
        let multi_tr = MultiTrustRegionState::new(num_dim, config, None, rng)
            .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
        Ok(Self {
            init: InitStrategyState::new(InitStrategy::LHD, num_init),
            multi_tr,
            in_init: num_init > 0,
            pending_regions: None,
        })
    }
}

/// Strategy enum - uses concrete types instead of trait objects.
#[derive(Debug, Clone)]
pub enum Strategy {
    /// Initialization-only strategy.
    Init(InitStrategyState),
    /// TuRBO normal strategy.
    Turbo(TurboStrategyState),
    /// Hybrid: initialization then TuRBO.
    Hybrid {
        init: InitStrategyState,
        turbo: TurboStrategyState,
        in_init: bool,
    },
    /// Experimental multi-trust-region strategy.
    Experimental(StrategyState),
}

impl Strategy {
    /// Create a new initialization-only strategy.
    pub fn init(strategy_type: InitStrategy, num_init: usize) -> Self {
        Strategy::Init(InitStrategyState::new(strategy_type, num_init))
    }

    /// Create a new TuRBO strategy.
    pub fn turbo() -> Self {
        Strategy::Turbo(TurboStrategyState)
    }

    /// Create a new experimental multi-trust-region strategy.
    pub fn experimental(
        num_dim: usize,
        num_regions: usize,
        num_init: usize,
        rng: &mut dyn RngCore,
    ) -> Result<Self, ENNError> {
        Ok(Strategy::Experimental(StrategyState::new(
            num_dim,
            num_regions,
            num_init,
            rng,
        )?))
    }

    /// Create a new hybrid strategy.
    pub fn hybrid(init_strategy: InitStrategy, num_init: usize) -> Self {
        Strategy::Hybrid {
            init: InitStrategyState::new(init_strategy, num_init),
            turbo: TurboStrategyState,
            in_init: true,
        }
    }

    /// Generate candidates (ask).
    pub fn ask(
        &mut self,
        optimizer: &mut Optimizer,
        num_arms: usize,
        telemetry: &mut Telemetry,
        rng: &mut dyn RngCore,
    ) -> Result<Array2<f64>, ENNError> {
        match self {
            Strategy::Init(state) => ask_init(state, optimizer, num_arms, rng),
            Strategy::Turbo(_) => ask_turbo(optimizer, num_arms, telemetry, rng),
            Strategy::Hybrid {
                init,
                in_init: true,
                ..
            } => ask_hybrid(init, optimizer, num_arms, rng),
            Strategy::Hybrid { .. } => ask_turbo(optimizer, num_arms, telemetry, rng),
            Strategy::Experimental(state) => {
                ask_experimental(state, optimizer, num_arms, telemetry, rng)
            }
        }
    }

    /// Process observations, optionally with known observation variance.
    pub fn tell(
        &mut self,
        optimizer: &mut Optimizer,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
        telemetry: &mut Telemetry,
        rng: &mut dyn RngCore,
    ) -> Result<(), ENNError> {
        match self {
            Strategy::Init(state) => tell_init(state, optimizer, x, y, yvar, rng),
            Strategy::Turbo(_) => tell_turbo(optimizer, x, y, yvar, telemetry, rng),
            Strategy::Hybrid {
                init,
                turbo: _,
                in_init,
            } => {
                if *in_init {
                    tell_init(init, optimizer, x, y, yvar, rng)?;
                    // Check if init is complete
                    if init.completed >= init.num_init {
                        *in_init = false;
                    }
                    Ok(())
                } else {
                    tell_turbo(optimizer, x, y, yvar, telemetry, rng)
                }
            }
            Strategy::Experimental(state) => {
                tell_experimental(state, optimizer, x, y, yvar, telemetry, rng)
            }
        }
    }

    /// Get initialization progress if applicable.
    pub fn init_progress(&self) -> Option<(usize, usize)> {
        match self {
            Strategy::Init(state) => Some((state.completed, state.num_init)),
            Strategy::Hybrid {
                init,
                in_init: true,
                ..
            } => Some((init.completed, init.num_init)),
            Strategy::Experimental(state) if state.in_init => {
                Some((state.init.completed, state.init.num_init))
            }
            _ => None,
        }
    }
}

/// Ask for initialization phase.
fn ask_init(
    state: &InitStrategyState,
    optimizer: &mut Optimizer,
    num_arms: usize,
    rng: &mut dyn RngCore,
) -> Result<Array2<f64>, ENNError> {
    let num_dim = optimizer.num_dim();
    let lower = Array1::zeros(num_dim);
    let upper = Array1::ones(num_dim);

    let candidates = match state.strategy_type {
        InitStrategy::LHD => {
            let mut unit_bounds = Array2::zeros((num_dim, 2));
            for j in 0..num_dim {
                unit_bounds[[j, 1]] = 1.0;
            }
            generate_lhd(num_arms, num_dim, &unit_bounds.view(), rng)
        }
        InitStrategy::Random => generate_uniform(&lower, &upper, num_arms, rng)?,
    };

    Ok(candidates)
}

/// Ask for initialization phase in hybrid mode.
fn ask_hybrid(
    state: &InitStrategyState,
    optimizer: &mut Optimizer,
    num_arms: usize,
    rng: &mut dyn RngCore,
) -> Result<Array2<f64>, ENNError> {
    ask_init(state, optimizer, num_arms, rng)
}

fn observe_morbo(optimizer: &mut Optimizer) -> Result<(), ENNError> {
    if !optimizer.trust_region().is_morbo() {
        return Ok(());
    }
    let Some(y_all) = optimizer.y_obs() else {
        return Ok(());
    };
    if y_all.nrows() == 0 {
        return Ok(());
    }
    optimizer.trust_mut().morbo_only(&y_all.view())
}

/// Common tell logic: add observations, fit surrogate, update incumbent.
fn tell_common(
    optimizer: &mut Optimizer,
    x: &ArrayView2<f64>,
    y: &ArrayView2<f64>,
    yvar: Option<&ArrayView2<f64>>,
    telemetry: Option<&mut Telemetry>,
    rng: &mut dyn RngCore,
) -> Result<(), ENNError> {
    if let Some(nm) = optimizer.surrogate().and_then(|s| s.fitted_metrics()) {
        if nm != y.ncols() {
            return Err(ENNError::InvalidParameter(format!(
                "y has {} metric columns but the fitted model has {nm}; changing output width is unsupported",
                y.ncols()
            )));
        }
    }
    let delta = optimizer.prepare_observations(x, y)?;
    if let Some(surrogate) = optimizer.surrogate_mut() {
        let start = std::time::Instant::now();
        surrogate.fit_append(&delta.x_view(), &delta.y_view(), yvar, rng)?;
        if let Some(tel) = telemetry {
            tel.dt_fit = start.elapsed().as_secs_f64();
        }
    }
    optimizer.commit_observations(&delta);

    if optimizer.trust_region().is_morbo() && delta.new_n > delta.old_n {
        optimizer.trust_mut().morbo_only(&delta.y_view())?;
    }

    optimizer.update_incumbent(rng)?;

    if optimizer.trust_region().is_morbo() {
        let num_obs = delta.new_n;
        if num_obs > 0 {
            let y_inc = optimizer
                .y_scalar()
                .ok_or_else(|| ENNError::InvalidParameter("Missing incumbent y".to_string()))?
                .to_owned();
            optimizer.trust_mut().update_morbo(&y_inc.view(), num_obs)?;
        }
    }

    // noise_aware incumbent predict (and similar) re-faults disk observation pages
    // after fit_append's release; drop them again so bulk seed RSS stays bounded.
    // Skip remap on tiny tells (e.g. --tell-all): O(N) remaps dominate mid-N wall time.
    if x.nrows() >= 64 {
        if let Some(surrogate) = optimizer.surrogate() {
            surrogate.release_pages()?;
        }
    }

    Ok(())
}

/// Tell for initialization phase.
fn tell_init(
    state: &mut InitStrategyState,
    optimizer: &mut Optimizer,
    x: &ArrayView2<f64>,
    y: &ArrayView2<f64>,
    yvar: Option<&ArrayView2<f64>>,
    rng: &mut dyn RngCore,
) -> Result<(), ENNError> {
    tell_common(optimizer, x, y, yvar, None, rng)?;
    state.completed += x.nrows();
    Ok(())
}

/// Ask for TuRBO phase.
fn ask_turbo(
    optimizer: &mut Optimizer,
    num_arms: usize,
    telemetry: &mut Telemetry,
    rng: &mut dyn RngCore,
) -> Result<Array2<f64>, ENNError> {
    optimizer.trust_mut().resample_propose(rng);
    optimizer.trust_mut().set_arms(num_arms);

    if optimizer.trust_region().is_morbo() {
        let num_obs = optimizer.obs_count();
        if num_obs > 0 {
            optimizer.trust_mut().rescale_morbo(num_obs)?;
        }
    }

    // Fetch incumbent center and lengthscales once (B5: was duplicated)
    let default_center = Array1::from_elem(optimizer.num_dim(), 0.5);
    let x_center = optimizer
        .unit_incumbent()
        .map(|x| x.to_owned())
        .unwrap_or(default_center);
    let lengthscales = optimizer.surrogate().and_then(|s| s.lengthscales());
    let ls_ref: Option<ArrayView1<f64>> = lengthscales.as_ref().map(|ls| ls.view());

    let (lower_1d, upper_1d) = optimizer
        .trust_region()
        .compute_bounds(&x_center.view(), ls_ref.as_ref());

    // Generate candidates
    let num_dim = optimizer.num_dim();
    let config = optimizer.config().candidates.clone();
    let num_candidates = config.num_candidates(num_dim, num_arms);
    telemetry.num_candidates = num_candidates;

    let x_cand_unit = {
        generate_candidates(
            || (lower_1d.clone(), upper_1d.clone()),
            &x_center.view(),
            ls_ref.as_ref(),
            num_candidates,
            config.candidate_rv,
            rng,
            optimizer.sobol_mut(),
            config.num_pert,
        )?
    };

    // Select arms using acquisition function (with timing)
    let start = std::time::Instant::now();
    let selected = { select_arms(optimizer, &x_cand_unit.view(), num_arms, rng)? };
    telemetry.dt_sel = start.elapsed().as_secs_f64();

    Ok(selected)
}

fn experimental_regions(state: &mut StrategyState, optimizer: &Optimizer) -> Result<(), ENNError> {
    if state.multi_tr.active_count() > 0 {
        return Ok(());
    }

    let center = optimizer
        .unit_incumbent()
        .cloned()
        .unwrap_or_else(|| Array1::from_elem(optimizer.num_dim(), 0.5));
    let center_view = center.view();
    for region in 0..state.multi_tr.num_regions() {
        state
            .multi_tr
            .restart_region(region, &center_view)
            .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
    }
    Ok(())
}

fn ask_experimental(
    state: &mut StrategyState,
    optimizer: &mut Optimizer,
    num_arms: usize,
    telemetry: &mut Telemetry,
    rng: &mut dyn RngCore,
) -> Result<Array2<f64>, ENNError> {
    if state.in_init {
        state.pending_regions = None;
        return ask_init(&state.init, optimizer, num_arms, rng);
    }

    experimental_regions(state, optimizer)?;

    let batches = state
        .multi_tr
        .allocate(num_arms)
        .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
    let num_dim = optimizer.num_dim();
    let config = optimizer.config().candidates.clone();
    let lengthscales = optimizer.surrogate().and_then(|s| s.lengthscales());
    let ls_ref: Option<ArrayView1<f64>> = lengthscales.as_ref().map(|ls| ls.view());

    let mut candidate_blocks = Vec::with_capacity(batches.len());
    let mut segments = Vec::with_capacity(batches.len());
    let mut offset = 0;

    for batch in batches {
        let x_center = state.multi_tr.centers.row(batch.region).to_owned();
        let (lower_1d, upper_1d) = state.multi_tr.compute_bounds(batch.region, ls_ref.as_ref());
        let num_candidates = config.num_candidates(num_dim, batch.len);
        telemetry.num_candidates += num_candidates;

        let x_cand = {
            let sobol_engine = optimizer.sobol_mut();
            generate_candidates(
                || (lower_1d.clone(), upper_1d.clone()),
                &x_center.view(),
                ls_ref.as_ref(),
                num_candidates,
                config.candidate_rv,
                rng,
                sobol_engine,
                config.num_pert,
            )?
        };

        let end = offset + x_cand.nrows();
        segments.push(CandidateSegment {
            start: offset,
            end,
            arms: batch.len,
            region: batch.region,
        });
        offset = end;
        candidate_blocks.push(x_cand);
    }

    if candidate_blocks.is_empty() {
        return Err(ENNError::InvalidParameter(
            "experimental strategy produced no candidates".to_string(),
        ));
    }

    let views = candidate_blocks
        .iter()
        .map(|block| block.view())
        .collect::<Vec<_>>();
    let candidates = ndarray::concatenate(Axis(0), &views)
        .map_err(|error| ENNError::InvalidParameter(error.to_string()))?;
    let start = std::time::Instant::now();
    let indices = select_segment(optimizer, &candidates.view(), &segments, rng)?;
    telemetry.dt_sel += start.elapsed().as_secs_f64();
    state.pending_regions = Some(
        segments
            .iter()
            .flat_map(|segment| std::iter::repeat_n(segment.region, segment.arms))
            .collect(),
    );
    Ok(select_indices(&candidates.view(), &indices))
}

/// Tell for TuRBO phase.
fn tell_turbo(
    optimizer: &mut Optimizer,
    x: &ArrayView2<f64>,
    y: &ArrayView2<f64>,
    yvar: Option<&ArrayView2<f64>>,
    telemetry: &mut Telemetry,
    rng: &mut dyn RngCore,
) -> Result<(), ENNError> {
    tell_common(optimizer, x, y, yvar, Some(telemetry), rng)?;

    let num_obs = optimizer.obs_count();
    let y_incumbent = optimizer
        .y_scalar()
        .ok_or_else(|| ENNError::InvalidParameter("Missing incumbent y".to_string()))?
        .to_owned();
    optimizer.trust_mut().set_arms(x.nrows());
    if !optimizer.trust_region().is_morbo() {
        // Init-phase tells do not advance TR prev_obs. After init (or a
        // restart that cleared hist), advance the watermark without loading
        // full y history — critical for disk-backed N ≫ 1e6.
        if optimizer.trust_region().turbo_obs() == 0 {
            let prev = num_obs.saturating_sub(y.nrows());
            optimizer.trust_mut().set_obs(prev);
        }
        optimizer
            .trust_mut()
            .tell_batch(y, &y_incumbent.view(), num_obs)?;
    }
    if optimizer.trust_region().needs_restart() {
        optimizer.trust_mut().restart(Some(rng));
        optimizer.increment_generation();
        // Do not reset the incumbent tracker: clearing observation_count forces
        // the next tell to rebuild from full y_obs() (Θ(N) RAM on disk). The
        // tracker is already maintained incrementally in add_observations.
        observe_morbo(optimizer)?;
    }

    Ok(())
}

fn tell_experimental(
    state: &mut StrategyState,
    optimizer: &mut Optimizer,
    x: &ArrayView2<f64>,
    y: &ArrayView2<f64>,
    yvar: Option<&ArrayView2<f64>>,
    telemetry: &mut Telemetry,
    rng: &mut dyn RngCore,
) -> Result<(), ENNError> {
    if y.ncols() != 1 {
        return Err(ENNError::InvalidParameter(format!(
            "experimental multi-trust-region strategy expects scalar y, got {} columns",
            y.ncols()
        )));
    }

    tell_common(optimizer, x, y, yvar, Some(telemetry), rng)?;

    state.init.completed += x.nrows();
    if state.in_init && state.init.completed >= state.init.num_init {
        state.in_init = false;
    }

    let y_scalar = y.column(0);
    let pending_regions = state.pending_regions.take();
    if let Some(regions) = pending_regions.as_deref() {
        state
            .multi_tr
            .tell(x, &y_scalar, Some(regions))
            .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
    } else {
        state
            .multi_tr
            .tell_update(x, &y_scalar)
            .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
    }

    if state.multi_tr.active_count() == 0 {
        experimental_regions(state, optimizer)?;
    } else {
        let restart_center = optimizer
            .unit_incumbent()
            .cloned()
            .unwrap_or_else(|| Array1::from_elem(optimizer.num_dim(), 0.5));
        let restart_center_view = restart_center.view();
        for region in 0..state.multi_tr.num_regions() {
            if !state.multi_tr.active_mask[region] {
                state
                    .multi_tr
                    .restart_region(region, &restart_center_view)
                    .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
            }
        }
    }

    Ok(())
}

/// Select arms randomly.
#[cfg(test)]
mod tests_init;
#[cfg(test)]
mod tests_selection;
#[cfg(test)]
mod testsmbacq;
