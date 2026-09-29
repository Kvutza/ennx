//! Acquisition selection over candidate segments.

use super::*;

pub(super) fn select_random(
    x_cand: &ArrayView2<f64>,
    num_arms: usize,
    rng: &mut dyn RngCore,
) -> Result<Array2<f64>, ENNError> {
    let random_acq = RandomAcquisition;
    let indices = random_acq
        .select(x_cand.nrows(), num_arms, rng)
        .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
    Ok(select_indices(x_cand, &indices))
}

/// Select arms via Thompson sampling (posterior draw).
pub(super) fn select_thompson(
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
            indices.push(argmax_tie(&arm_scores, rng));
        }
        return Ok(select_indices(x_cand, &indices));
    }
    let sample_values: Vec<f64> = (0..n_candidates).map(|i| samples[[0, i, 0]]).collect();
    let mut indices: Vec<usize> = (0..n_candidates).collect();
    indices.sort_by(|&a, &b| {
        sample_values[b]
            .partial_cmp(&sample_values[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let selected: Vec<usize> = indices.into_iter().take(num_arms).collect();
    Ok(select_indices(x_cand, &selected))
}

/// Select arms via UCB (upper confidence bound).
pub(super) fn select_ucb(
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
        return Ok(select_indices(x_cand, &selected));
    }
    let mu = pred.mu.column(0);
    let sigma = pred.se.column(0);
    let ucb = UCBAcquisition::new(beta);
    let indices = ucb
        .select(&mu, &sigma, num_arms, rng)
        .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
    Ok(select_indices(x_cand, &indices))
}

/// Select arms via Pareto frontier.
pub(super) fn select_pareto(
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
    Ok(select_indices(x_cand, &indices))
}

/// Select fixed arm counts from candidate segments after one surrogate pass.
pub(super) fn select_segment(
    optimizer: &Optimizer,
    candidates: &ArrayView2<f64>,
    segments: &[CandidateSegment],
    rng: &mut dyn RngCore,
) -> Result<Vec<usize>, ENNError> {
    let config = optimizer.config().acquisition;
    let surrogate = optimizer.surrogate();
    if surrogate.is_none() || matches!(config, AcquisitionConfig::Random) {
        return random_segments(segments, rng);
    }
    let surrogate = surrogate.expect("checked above");

    match config {
        AcquisitionConfig::Random => unreachable!("handled above"),
        AcquisitionConfig::Thompson => {
            thompson_segments(optimizer, surrogate, candidates, segments, rng)
        }
        AcquisitionConfig::UCB { beta } => {
            ucb_segments(optimizer, surrogate, candidates, segments, beta, rng)
        }
        AcquisitionConfig::Pareto => pareto_segments(surrogate, candidates, segments, rng),
    }
}

pub(super) fn random_segments(
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

pub(super) fn thompson_segments(
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
            local.push(argmax_tie(&values, rng));
        }
        selected.extend(local.into_iter().map(|index| segment.start + index));
    }
    Ok(selected)
}

pub(super) fn ucb_segments(
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

pub(super) fn pareto_segments(
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
pub(super) fn select_arms(
    optimizer: &Optimizer,
    x_cand: &ArrayView2<f64>,
    num_arms: usize,
    rng: &mut dyn RngCore,
) -> Result<Array2<f64>, ENNError> {
    let config = optimizer.config().acquisition;

    match config {
        AcquisitionConfig::Random => select_random(x_cand, num_arms, rng),
        AcquisitionConfig::Thompson => match optimizer.surrogate() {
            Some(s) => select_thompson(optimizer, s, x_cand, num_arms, rng),
            None => select_random(x_cand, num_arms, rng),
        },
        AcquisitionConfig::UCB { beta } => match optimizer.surrogate() {
            Some(s) => select_ucb(optimizer, s, x_cand, num_arms, beta, rng),
            None => select_random(x_cand, num_arms, rng),
        },
        AcquisitionConfig::Pareto => match optimizer.surrogate() {
            Some(s) => select_pareto(s, x_cand, num_arms, rng),
            None => select_random(x_cand, num_arms, rng),
        },
    }
}

/// Select rows by indices.
pub(super) fn select_indices(x: &ArrayView2<f64>, indices: &[usize]) -> Array2<f64> {
    use ndarray::Axis;
    let rows: Vec<_> = indices.iter().map(|&i| x.row(i).to_owned()).collect();
    ndarray::stack(Axis(0), &rows.iter().map(|r| r.view()).collect::<Vec<_>>())
        .expect("stack should succeed for same-shaped rows")
}
