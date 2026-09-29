//! Ported from yubo-research/enn, commit 506e98c506eeb849cffbf53d9ddf3a3a799c6830.
//! AUTO metric policy: reservoir, refit schedule, rescale versus re-partition.

use crate::error::ENNError;
use crate::metric_loo::loo_loglik;
use crate::metric_rows::RowReservoir;
use crate::metric_sobol::DEPENDENCE_ROWS;
use crate::metric_weights::{dependence_weights, insufficient, validate_tied};

pub const REBUILD_DRIFT: f64 = std::f64::consts::LN_2;
pub const DRIFT_FLOOR: f64 = 9.210_340_371_976_184;
pub const HELDOUT_GAIN: f64 = 0.0;
pub const RESERVOIR_CAPACITY: usize = 1000;
pub const AUTO_K: usize = 10;
pub const REFIT_GROWTH: f64 = 1.5;
pub const RESCALE_TOL: f64 = 0.01;

/// `None` leaves the metric fixed. `Auto` learns a diagonal metric from a reservoir.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MetricLearning {
    #[default]
    None,
    Auto,
}

impl MetricLearning {
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "none" | "NONE" | "None" => Some(Self::None),
            "auto" | "AUTO" | "Auto" => Some(Self::Auto),
            _ => None,
        }
    }
}

fn floored_log(w: &[f64]) -> Vec<f64> {
    let log_w: Vec<f64> = w.iter().map(|v| v.ln()).collect();
    let max = log_w.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    log_w
        .into_iter()
        .map(|v| v.max(max - DRIFT_FLOOR))
        .collect()
}

pub fn weight_drift(weights: &[f64], built: &[f64]) -> f64 {
    if weights.len() != built.len()
        || weights
            .iter()
            .chain(built)
            .any(|v| !v.is_finite() || *v <= 0.0)
    {
        return f64::INFINITY;
    }
    let a = floored_log(weights);
    let b = floored_log(built);
    0.5 * a
        .iter()
        .zip(b.iter())
        .map(|(u, v)| (u - v).abs())
        .fold(0.0, f64::max)
}

pub fn accept_gain(heldout_gain: f64) -> bool {
    heldout_gain.is_finite() && heldout_gain > HELDOUT_GAIN
}

pub fn auto_weights(
    x: &[f64],
    n: usize,
    d: usize,
    y: &[f64],
    m: usize,
    k: usize,
    tied: &[Vec<usize>],
) -> Result<(Vec<f64>, f64), ENNError> {
    crate::metric_weights::validate_rows(x, n, d, y, m)?;
    if k == 0 {
        return Err(ENNError::InvalidParameter(
            "metric neighbor count must be positive".into(),
        ));
    }
    validate_tied(tied, d)?;
    if insufficient(n) {
        return Ok((vec![1.0; d], f64::NEG_INFINITY));
    }
    let w = dependence_weights(x, n, d, y, m, tied, crate::metric_weights::DEPENDENCE_FLOOR)?;
    let ones = vec![1.0; d];
    let gain = loo_loglik(x, n, d, y, m, &w, k) - loo_loglik(x, n, d, y, m, &ones, k);
    Ok((w, gain))
}

#[derive(Clone, Copy, Debug, Default)]
pub struct MetricCounters {
    pub num_refits: usize,
    pub num_rescales: usize,
    pub num_rebuilds: usize,
}

#[derive(Clone, Debug)]
pub struct MetricSnapshot {
    pub num_seen: usize,
    pub counters: MetricCounters,
    pub heldout_gain: Option<f64>,
    pub uses_learned: bool,
    pub weights: Vec<f64>,
    pub built: Vec<f64>,
}

pub struct ScaleUpdate {
    pub x_scale: Vec<f64>,
    pub rebuild: bool,
}

pub struct AutoMetric {
    rows: RowReservoir,
    tied: Vec<Vec<usize>>,
    weights: Vec<f64>,
    built: Vec<f64>,
    pub rebuild_drift: f64,
    refit_growth: f64,
    next_refit: usize,
    pub heldout_gain: Option<f64>,
    counters: MetricCounters,
}

impl AutoMetric {
    pub fn new(
        num_dim: usize,
        num_outputs: usize,
        tied: Vec<Vec<usize>>,
        seed: u64,
    ) -> Result<Self, ENNError> {
        if num_dim == 0 || num_outputs == 0 {
            return Err(ENNError::InvalidParameter(
                "metric learning needs a positive shape".into(),
            ));
        }
        validate_tied(&tied, num_dim)?;
        Ok(Self {
            rows: RowReservoir::new(RESERVOIR_CAPACITY, num_dim, num_outputs, seed)?,
            tied,
            weights: vec![1.0; num_dim],
            built: vec![1.0; num_dim],
            rebuild_drift: REBUILD_DRIFT,
            refit_growth: REFIT_GROWTH,
            next_refit: DEPENDENCE_ROWS,
            heldout_gain: None,
            counters: MetricCounters::default(),
        })
    }

    pub fn tied(&self) -> &[Vec<usize>] {
        &self.tied
    }

    /// Update only the fields that are `Some`. Omitted fields stay as they are.
    pub fn configure(
        &mut self,
        refit_growth: Option<f64>,
        rebuild_drift: Option<f64>,
        seed: Option<u64>,
        capacity: Option<usize>,
    ) -> Result<(), ENNError> {
        if let Some(growth) = refit_growth {
            if !growth.is_finite() || growth <= 1.0 {
                return Err(ENNError::InvalidParameter(format!(
                    "refit_growth must be > 1, got {growth}"
                )));
            }
        }
        if let Some(drift) = rebuild_drift {
            if drift.is_nan() || drift < 0.0 {
                return Err(ENNError::InvalidParameter(format!(
                    "rebuild_drift must be >= 0, got {drift}"
                )));
            }
        }
        if seed.is_some() || capacity.is_some() {
            self.rows.configure(
                seed.unwrap_or_else(|| self.rows.seed()),
                capacity.unwrap_or_else(|| self.rows.capacity()),
            )?;
        }
        if let Some(growth) = refit_growth {
            self.refit_growth = growth;
        }
        if let Some(drift) = rebuild_drift {
            self.rebuild_drift = drift;
        }
        Ok(())
    }

    pub fn seed(&self) -> u64 {
        self.rows.seed()
    }

    pub fn capacity(&self) -> usize {
        self.rows.capacity()
    }

    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    pub fn built(&self) -> &[f64] {
        &self.built
    }

    pub fn num_seen(&self) -> usize {
        self.rows.num_seen()
    }

    pub fn snapshot(&self) -> MetricSnapshot {
        MetricSnapshot {
            num_seen: self.num_seen(),
            counters: self.counters,
            heldout_gain: self.heldout_gain,
            uses_learned: self.is_learned(),
            weights: self.weights.clone(),
            built: self.built.clone(),
        }
    }

    pub fn is_learned(&self) -> bool {
        self.heldout_gain.is_some_and(accept_gain)
    }

    pub fn set_weights(&mut self, weights: &[f64]) -> Result<ScaleUpdate, ENNError> {
        let update = self.preview(weights)?;
        self.weights = weights.to_vec();
        if update.rebuild {
            self.built = weights.to_vec();
            self.counters.num_rebuilds += 1;
        } else {
            self.counters.num_rescales += 1;
        }
        Ok(update)
    }

    pub(crate) fn preview(&self, weights: &[f64]) -> Result<ScaleUpdate, ENNError> {
        if weights.len() != self.rows.num_dim()
            || !weights.iter().all(|w| w.is_finite() && *w > 0.0)
        {
            return Err(ENNError::InvalidParameter(
                "weights must be finite, > 0, and match the dimension".into(),
            ));
        }
        let rebuild = weight_drift(weights, &self.built) > self.rebuild_drift;
        let x_scale: Vec<f64> = weights.iter().map(|w| 1.0 / w.sqrt()).collect();
        Ok(ScaleUpdate { x_scale, rebuild })
    }

    pub fn refit(&mut self) -> Result<Option<ScaleUpdate>, ENNError> {
        self.propose()?
            .map(|weights| self.set_weights(&weights))
            .transpose()
    }

    pub(crate) fn propose(&mut self) -> Result<Option<Vec<f64>>, ENNError> {
        let n = self.rows.len();
        let (w, gain) = auto_weights(
            self.rows.x(),
            n,
            self.rows.num_dim(),
            self.rows.y(),
            self.rows.num_outputs(),
            AUTO_K,
            &self.tied,
        )?;
        self.heldout_gain = Some(gain);
        self.counters.num_refits += 1;
        let grown = (self.refit_growth * self.num_seen() as f64).ceil() as usize;
        self.next_refit = grown.max(DEPENDENCE_ROWS);
        let target = if accept_gain(gain) {
            w
        } else {
            vec![1.0; self.rows.num_dim()]
        };
        let change = 0.5
            * target
                .iter()
                .zip(self.weights.iter())
                .map(|(t, w)| (t / w).ln().abs())
                .fold(0.0, f64::max);
        if change > RESCALE_TOL {
            return Ok(Some(target));
        }
        Ok(None)
    }

    pub fn observe(
        &mut self,
        x: &[f64],
        y: &[f64],
        n: usize,
    ) -> Result<Option<ScaleUpdate>, ENNError> {
        if self.ingest(x, y, n)? {
            self.refit()
        } else {
            Ok(None)
        }
    }

    pub(crate) fn ingest(&mut self, x: &[f64], y: &[f64], n: usize) -> Result<bool, ENNError> {
        if x.len() != n.checked_mul(self.rows.num_dim()).unwrap_or(usize::MAX)
            || y.len() != n.checked_mul(self.rows.num_outputs()).unwrap_or(usize::MAX)
            || x.iter().chain(y).any(|v| !v.is_finite())
        {
            return Err(ENNError::InvalidParameter(
                "metric rows must be finite and match the configured shape".into(),
            ));
        }
        for i in 0..n {
            let xr = &x[i * self.rows.num_dim()..(i + 1) * self.rows.num_dim()];
            let yr = &y[i * self.rows.num_outputs()..(i + 1) * self.rows.num_outputs()];
            self.rows.push_row(xr, yr);
        }
        Ok(self.num_seen() >= self.next_refit)
    }
}

#[cfg(test)]
#[path = "metric_checks.rs"]
mod checks;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metric_config() {
        let mut metric = AutoMetric::new(2, 1, vec![], 0).unwrap();
        metric
            .configure(Some(3.0), Some(0.25), Some(7), Some(4))
            .unwrap();
        assert_eq!(metric.refit_growth, 3.0);
        assert_eq!(metric.rebuild_drift, 0.25);
        assert_eq!(metric.seed(), 7);
        assert_eq!(metric.capacity(), 4);
        assert!(
            metric
                .configure(Some(3.0), Some(0.25), Some(7), Some(0))
                .is_err()
        );
        metric
            .observe(&[0.0, 1.0, 0.2, 0.3], &[0.0, 1.0], 2)
            .unwrap();
        assert!(
            metric
                .configure(Some(3.0), Some(0.25), Some(7), Some(1))
                .is_err()
        );
    }

    #[test]
    fn metric_defaults() {
        let mut metric = AutoMetric::new(2, 1, vec![], 4).unwrap();
        let seed = metric.seed();
        let capacity = metric.capacity();
        let drift = metric.rebuild_drift;
        metric.configure(Some(3.0), None, None, None).unwrap();
        assert_eq!(metric.refit_growth, 3.0);
        assert_eq!(metric.rebuild_drift, drift);
        assert_eq!(metric.seed(), seed);
        assert_eq!(metric.capacity(), capacity);
        metric.configure(None, Some(0.1), None, None).unwrap();
        assert_eq!(metric.refit_growth, 3.0);
        assert_eq!(metric.seed(), seed);
        assert_eq!(metric.capacity(), capacity);
    }

    #[test]
    fn metric_neighbors() {
        let n = 120;
        let d = 3;
        let mut x = vec![0.0; n * d];
        let mut y = vec![0.0; n];
        for i in 0..n {
            for j in 0..d {
                x[i * d + j] = ((i + 1) as f64 * (j + 3) as f64 * 0.017 + 0.001 * i as f64) % 1.0;
            }
            y[i] = (6.0 * std::f64::consts::PI * x[i * d]).sin() + 0.01 * x[i * d + 1];
        }
        let (w10, g10) = auto_weights(&x, n, d, &y, 1, 10, &[]).unwrap();
        let (w1, g1) = auto_weights(&x, n, d, &y, 1, 1, &[]).unwrap();
        assert_eq!(w10, w1);
        assert_ne!(g10, g1);
    }
}
