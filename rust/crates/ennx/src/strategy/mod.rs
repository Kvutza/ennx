//! Optimization strategies for ask/tell pattern.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rand::seq::SliceRandom;
use rand::RngCore;

use crate::util::argmax_random_tie;

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
pub struct ExperimentalStrategyState {
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

impl ExperimentalStrategyState {
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
    Experimental(ExperimentalStrategyState),
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
        Ok(Strategy::Experimental(ExperimentalStrategyState::new(
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
            } => ask_init_hybrid(init, optimizer, num_arms, rng),
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
fn ask_init_hybrid(
    state: &InitStrategyState,
    optimizer: &mut Optimizer,
    num_arms: usize,
    rng: &mut dyn RngCore,
) -> Result<Array2<f64>, ENNError> {
    ask_init(state, optimizer, num_arms, rng)
}

fn morbo_sync_ranges_from_obs(optimizer: &mut Optimizer) -> Result<(), ENNError> {
    if !optimizer.trust_region().is_morbo() {
        return Ok(());
    }
    let Some(y_all) = optimizer.y_obs() else {
        return Ok(());
    };
    if y_all.nrows() == 0 {
        return Ok(());
    }
    optimizer
        .trust_region_mut()
        .morbo_update_ranges_only(&y_all.view())
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
    if let Some(nm) = optimizer.surrogate().and_then(|s| s.fitted_num_metrics()) {
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
        surrogate.fit_append(&delta.x_new_view(), &delta.y_new_view(), yvar, rng)?;
        if let Some(tel) = telemetry {
            tel.dt_fit = start.elapsed().as_secs_f64();
        }
    }
    optimizer.commit_observations(&delta);

    if optimizer.trust_region().is_morbo() && delta.new_n > delta.old_n {
        optimizer
            .trust_region_mut()
            .morbo_update_ranges_only(&delta.y_new_view())?;
    }

    optimizer.update_incumbent(rng)?;

    if optimizer.trust_region().is_morbo() {
        let num_obs = delta.new_n;
        if num_obs > 0 {
            let y_inc = optimizer
                .incumbent_y_scalar()
                .ok_or_else(|| ENNError::InvalidParameter("Missing incumbent y".to_string()))?
                .to_owned();
            optimizer
                .trust_region_mut()
                .morbo_update_incumbent_only(&y_inc.view(), num_obs)?;
        }
    }

    // noise_aware incumbent predict (and similar) re-faults disk observation pages
    // after fit_append's release; drop them again so bulk seed RSS stays bounded.
    // Skip remap on tiny tells (e.g. --tell-all): O(N) remaps dominate mid-N wall time.
    if x.nrows() >= 64 {
        if let Some(surrogate) = optimizer.surrogate() {
            surrogate.release_observation_pages()?;
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
    optimizer.trust_region_mut().resample_on_propose(rng);
    optimizer.trust_region_mut().set_num_arms(num_arms);

    if optimizer.trust_region().is_morbo() {
        let num_obs = optimizer.obs_count();
        if num_obs > 0 {
            optimizer
                .trust_region_mut()
                .morbo_rescalarize_incumbent(num_obs)?;
        }
    }

    // Fetch incumbent center and lengthscales once (B5: was duplicated)
    let default_center = Array1::from_elem(optimizer.num_dim(), 0.5);
    let x_center = optimizer
        .incumbent_x_unit()
        .map(|x| x.to_owned())
        .unwrap_or(default_center);
    let lengthscales = optimizer.surrogate().and_then(|s| s.lengthscales());
    let ls_ref: Option<ArrayView1<f64>> = lengthscales.as_ref().map(|ls| ls.view());

    let (lower_1d, upper_1d) = optimizer
        .trust_region()
        .compute_bounds_1d(&x_center.view(), ls_ref.as_ref());

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
            optimizer.sobol_engine_mut(),
            config.num_pert,
        )?
    };

    // Select arms using acquisition function (with timing)
    let start = std::time::Instant::now();
    let selected = { select_arms(optimizer, &x_cand_unit.view(), num_arms, rng)? };
    telemetry.dt_sel = start.elapsed().as_secs_f64();

    Ok(selected)
}

fn experimental_restart_all_regions(
    state: &mut ExperimentalStrategyState,
    optimizer: &Optimizer,
) -> Result<(), ENNError> {
    if state.multi_tr.active_count() > 0 {
        return Ok(());
    }

    let center = optimizer
        .incumbent_x_unit()
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
    state: &mut ExperimentalStrategyState,
    optimizer: &mut Optimizer,
    num_arms: usize,
    telemetry: &mut Telemetry,
    rng: &mut dyn RngCore,
) -> Result<Array2<f64>, ENNError> {
    if state.in_init {
        state.pending_regions = None;
        return ask_init(&state.init, optimizer, num_arms, rng);
    }

    experimental_restart_all_regions(state, optimizer)?;

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
        let (lower_1d, upper_1d) = state
            .multi_tr
            .compute_bounds_1d(batch.region, ls_ref.as_ref());
        let num_candidates = config.num_candidates(num_dim, batch.len);
        telemetry.num_candidates += num_candidates;

        let x_cand = {
            let sobol_engine = optimizer.sobol_engine_mut();
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
    let indices = select_segmented_arms(optimizer, &candidates.view(), &segments, rng)?;
    telemetry.dt_sel += start.elapsed().as_secs_f64();
    state.pending_regions = Some(
        segments
            .iter()
            .flat_map(|segment| std::iter::repeat_n(segment.region, segment.arms))
            .collect(),
    );
    Ok(select_by_indices(&candidates.view(), &indices))
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
        .incumbent_y_scalar()
        .ok_or_else(|| ENNError::InvalidParameter("Missing incumbent y".to_string()))?
        .to_owned();
    optimizer.trust_region_mut().set_num_arms(x.nrows());
    if !optimizer.trust_region().is_morbo() {
        // Init-phase tells do not advance TR prev_num_obs. After init (or a
        // restart that cleared hist), advance the watermark without loading
        // full y history — critical for disk-backed N ≫ 1e6.
        if optimizer.trust_region().turbo_prev_num_obs() == 0 {
            let prev = num_obs.saturating_sub(y.nrows());
            optimizer.trust_region_mut().set_turbo_prev_num_obs(prev);
        }
        optimizer
            .trust_region_mut()
            .tell_update_new_batch(y, &y_incumbent.view(), num_obs)?;
    }
    if optimizer.trust_region().needs_restart() {
        optimizer.trust_region_mut().restart(Some(rng));
        optimizer.increment_restart_generation();
        // Do not reset the incumbent tracker: clearing observation_count forces
        // the next tell to rebuild from full y_obs() (Θ(N) RAM on disk). The
        // tracker is already maintained incrementally in add_observations.
        morbo_sync_ranges_from_obs(optimizer)?;
    }

    Ok(())
}

fn tell_experimental(
    state: &mut ExperimentalStrategyState,
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
        experimental_restart_all_regions(state, optimizer)?;
    } else {
        let restart_center = optimizer
            .incumbent_x_unit()
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
fn select_with_random(
    x_cand: &ArrayView2<f64>,
    num_arms: usize,
    rng: &mut dyn RngCore,
) -> Result<Array2<f64>, ENNError> {
    let random_acq = RandomAcquisition;
    let indices = random_acq
        .select(x_cand.nrows(), num_arms, rng)
        .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
    Ok(select_by_indices(x_cand, &indices))
}

/// Select arms via Thompson sampling (posterior draw).
fn select_with_thompson(
    optimizer: &Optimizer,
    surrogate: &(dyn crate::surrogate::Surrogate + Send + Sync),
    x_cand: &ArrayView2<f64>,
    num_arms: usize,
    rng: &mut dyn RngCore,
) -> Result<Array2<f64>, ENNError> {
    let samples = surrogate.sample(x_cand, num_arms, rng)?;
    let n_candidates = x_cand.nrows();
    if optimizer.trust_region().is_morbo() {
        let num_metrics = samples.shape()[2];
        let mut flat = ndarray::Array2::zeros((num_arms * n_candidates, num_metrics));
        for arm in 0..num_arms {
            for cand in 0..n_candidates {
                for m in 0..num_metrics {
                    flat[[arm * n_candidates + cand, m]] = samples[[arm, cand, m]];
                }
            }
        }
        let flat_scores = optimizer
            .trust_region()
            .morbo_scalarize(&flat.view(), false)
            .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
        let mut all_scores = ndarray::Array2::zeros((num_arms, n_candidates));
        for arm in 0..num_arms {
            for cand in 0..n_candidates {
                all_scores[[arm, cand]] = flat_scores[arm * n_candidates + cand];
            }
        }
        let mut indices = Vec::with_capacity(num_arms);
        for arm in 0..num_arms {
            let mut arm_scores = vec![f64::NEG_INFINITY; n_candidates];
            for cand in 0..n_candidates {
                arm_scores[cand] = all_scores[[arm, cand]];
            }
            for &prev in &indices {
                arm_scores[prev] = f64::NEG_INFINITY;
            }
            indices.push(argmax_random_tie(&arm_scores, rng));
        }
        return Ok(select_by_indices(x_cand, &indices));
    }
    let sample_values: Vec<f64> = (0..n_candidates).map(|i| samples[[0, i, 0]]).collect();
    let mut indices: Vec<usize> = (0..n_candidates).collect();
    indices.sort_by(|&a, &b| {
        sample_values[b]
            .partial_cmp(&sample_values[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let selected: Vec<usize> = indices.into_iter().take(num_arms).collect();
    Ok(select_by_indices(x_cand, &selected))
}

/// Select arms via UCB (upper confidence bound).
fn select_with_ucb(
    optimizer: &Optimizer,
    surrogate: &(dyn crate::surrogate::Surrogate + Send + Sync),
    x_cand: &ArrayView2<f64>,
    num_arms: usize,
    beta: f64,
    rng: &mut dyn RngCore,
) -> Result<Array2<f64>, ENNError> {
    let pred = surrogate.predict(x_cand)?;
    if optimizer.trust_region().is_morbo() {
        let ucb_vals = &pred.mu + &(pred.se * beta);
        let scores = optimizer
            .trust_region()
            .morbo_scalarize(&ucb_vals.view(), false)
            .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
        let mut indices: Vec<usize> = (0..scores.len()).collect();
        indices.shuffle(rng);
        indices.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
        let selected: Vec<usize> = indices.into_iter().take(num_arms).collect();
        return Ok(select_by_indices(x_cand, &selected));
    }
    let mu = pred.mu.column(0);
    let sigma = pred.se.column(0);
    let ucb = UCBAcquisition::new(beta);
    let indices = ucb
        .select(&mu, &sigma, num_arms, rng)
        .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
    Ok(select_by_indices(x_cand, &indices))
}

/// Select arms via Pareto frontier.
fn select_with_pareto(
    surrogate: &(dyn crate::surrogate::Surrogate + Send + Sync),
    x_cand: &ArrayView2<f64>,
    num_arms: usize,
    rng: &mut dyn RngCore,
) -> Result<Array2<f64>, ENNError> {
    let pred = surrogate.predict(x_cand)?;
    let pareto = ParetoAcquisition::new();
    let indices = pareto
        .select(&pred.mu.view(), &pred.se.view(), num_arms, rng)
        .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
    Ok(select_by_indices(x_cand, &indices))
}

/// Select fixed arm counts from candidate segments after one surrogate pass.
fn select_segmented_arms(
    optimizer: &Optimizer,
    candidates: &ArrayView2<f64>,
    segments: &[CandidateSegment],
    rng: &mut dyn RngCore,
) -> Result<Vec<usize>, ENNError> {
    let config = optimizer.config().acquisition;
    let surrogate = optimizer.surrogate();
    if surrogate.is_none() || matches!(config, AcquisitionConfig::Random) {
        return select_segmented_random(segments, rng);
    }
    let surrogate = surrogate.expect("checked above");

    match config {
        AcquisitionConfig::Random => unreachable!("handled above"),
        AcquisitionConfig::Thompson => {
            select_segmented_thompson(optimizer, surrogate, candidates, segments, rng)
        }
        AcquisitionConfig::UCB { beta } => {
            select_segmented_ucb(optimizer, surrogate, candidates, segments, beta, rng)
        }
        AcquisitionConfig::Pareto => select_segmented_pareto(surrogate, candidates, segments, rng),
    }
}

fn select_segmented_random(
    segments: &[CandidateSegment],
    rng: &mut dyn RngCore,
) -> Result<Vec<usize>, ENNError> {
    let mut selected = Vec::new();
    for segment in segments {
        let local = RandomAcquisition
            .select(segment.end - segment.start, segment.arms, rng)
            .map_err(|error| ENNError::InvalidParameter(error.to_string()))?;
        selected.extend(local.into_iter().map(|index| segment.start + index));
    }
    Ok(selected)
}

fn select_segmented_thompson(
    optimizer: &Optimizer,
    surrogate: &(dyn crate::surrogate::Surrogate + Send + Sync),
    candidates: &ArrayView2<f64>,
    segments: &[CandidateSegment],
    rng: &mut dyn RngCore,
) -> Result<Vec<usize>, ENNError> {
    use ndarray::s;

    let max_arms = segments
        .iter()
        .map(|segment| segment.arms)
        .max()
        .unwrap_or(1);
    let samples = surrogate.sample(candidates, max_arms, rng)?;
    let mut scores = Array2::zeros((max_arms, candidates.nrows()));
    if optimizer.trust_region().is_morbo() {
        let metrics = samples.shape()[2];
        let flat = samples
            .to_shape((max_arms * candidates.nrows(), metrics))
            .map_err(|error| ENNError::InvalidParameter(error.to_string()))?;
        let scalar = optimizer
            .trust_region()
            .morbo_scalarize(&flat.view(), false)
            .map_err(|error| ENNError::InvalidParameter(error.to_string()))?;
        scores
            .as_slice_mut()
            .expect("scores are contiguous")
            .copy_from_slice(scalar.as_slice().expect("scalar scores are contiguous"));
    } else {
        scores.row_mut(0).assign(&samples.slice(s![0, .., 0]));
    }

    let mut selected = Vec::new();
    for segment in segments {
        let mut local = Vec::with_capacity(segment.arms);
        for arm in 0..segment.arms {
            let row = usize::from(optimizer.trust_region().is_morbo()) * arm;
            let mut values = scores.slice(s![row, segment.start..segment.end]).to_vec();
            for &previous in &local {
                values[previous] = f64::NEG_INFINITY;
            }
            local.push(argmax_random_tie(&values, rng));
        }
        selected.extend(local.into_iter().map(|index| segment.start + index));
    }
    Ok(selected)
}

fn select_segmented_ucb(
    optimizer: &Optimizer,
    surrogate: &(dyn crate::surrogate::Surrogate + Send + Sync),
    candidates: &ArrayView2<f64>,
    segments: &[CandidateSegment],
    beta: f64,
    rng: &mut dyn RngCore,
) -> Result<Vec<usize>, ENNError> {
    let prediction = surrogate.predict(candidates)?;
    let scores = if optimizer.trust_region().is_morbo() {
        let values = &prediction.mu + &(prediction.se * beta);
        optimizer
            .trust_region()
            .morbo_scalarize(&values.view(), false)
            .map_err(|error| ENNError::InvalidParameter(error.to_string()))?
    } else {
        &prediction.mu.column(0) + &(&prediction.se.column(0) * beta)
    };
    let mut selected = Vec::new();
    for segment in segments {
        let mut local = (segment.start..segment.end).collect::<Vec<_>>();
        local.shuffle(rng);
        local.sort_by(|&left, &right| scores[right].total_cmp(&scores[left]));
        selected.extend(local.into_iter().take(segment.arms));
    }
    Ok(selected)
}

fn select_segmented_pareto(
    surrogate: &(dyn crate::surrogate::Surrogate + Send + Sync),
    candidates: &ArrayView2<f64>,
    segments: &[CandidateSegment],
    rng: &mut dyn RngCore,
) -> Result<Vec<usize>, ENNError> {
    use ndarray::s;

    let prediction = surrogate.predict(candidates)?;
    let pareto = ParetoAcquisition::new();
    let mut selected = Vec::new();
    for segment in segments {
        let local = pareto
            .select(
                &prediction.mu.slice(s![segment.start..segment.end, ..]),
                &prediction.se.slice(s![segment.start..segment.end, ..]),
                segment.arms,
                rng,
            )
            .map_err(|error| ENNError::InvalidParameter(error.to_string()))?;
        selected.extend(local.into_iter().map(|index| segment.start + index));
    }
    Ok(selected)
}

/// Select arms using acquisition function.
fn select_arms(
    optimizer: &Optimizer,
    x_cand: &ArrayView2<f64>,
    num_arms: usize,
    rng: &mut dyn RngCore,
) -> Result<Array2<f64>, ENNError> {
    let config = optimizer.config().acquisition;

    match config {
        AcquisitionConfig::Random => select_with_random(x_cand, num_arms, rng),
        AcquisitionConfig::Thompson => match optimizer.surrogate() {
            Some(s) => select_with_thompson(optimizer, s, x_cand, num_arms, rng),
            None => select_with_random(x_cand, num_arms, rng),
        },
        AcquisitionConfig::UCB { beta } => match optimizer.surrogate() {
            Some(s) => select_with_ucb(optimizer, s, x_cand, num_arms, beta, rng),
            None => select_with_random(x_cand, num_arms, rng),
        },
        AcquisitionConfig::Pareto => match optimizer.surrogate() {
            Some(s) => select_with_pareto(s, x_cand, num_arms, rng),
            None => select_with_random(x_cand, num_arms, rng),
        },
    }
}

/// Select rows by indices.
fn select_by_indices(x: &ArrayView2<f64>, indices: &[usize]) -> Array2<f64> {
    use ndarray::Axis;
    let rows: Vec<_> = indices.iter().map(|&i| x.row(i).to_owned()).collect();
    ndarray::stack(Axis(0), &rows.iter().map(|r| r.view()).collect::<Vec<_>>())
        .expect("stack should succeed for same-shaped rows")
}

#[cfg(test)]
mod tests_init;
#[cfg(test)]
mod tests_morbo_acq;
#[cfg(test)]
mod tests_selection;
