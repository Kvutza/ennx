use ndarray::Array1;

use crate::trials::{Ask, Leaf, Search, Trial};
use crate::trust_region::{TRLengthConfig, TurboTrustRegion};
use crate::weights::ComputeBackend;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TurboTrial {
    inner: Trial,
    pub index: usize,
    pub seed: u64,
    pub score: f32,
    pub length: f32,
    pub probability: f32,
}

/// TuRBO control state around a resident packed-weight search engine.
pub struct TurboSearch {
    search: Search,
    trust: TurboTrustRegion,
    dimensions: usize,
    num_pert: usize,
    outcomes: Vec<f64>,
    best: f32,
    restarts: usize,
}

impl TurboSearch {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        base: &[u8],
        base_value: f32,
        leaves: Vec<Leaf>,
        capacity: usize,
        backend: ComputeBackend,
        num_pert: usize,
        length: TRLengthConfig,
    ) -> Result<Self, String> {
        let dimensions = leaves.iter().try_fold(0usize, |total, leaf| {
            total
                .checked_add(leaf.length)
                .ok_or("parameter count overflow")
        })?;
        if dimensions == 0 {
            return Err("resident TuRBO requires parameters".to_string());
        }
        if num_pert == 0 {
            return Err("num_pert must be positive".to_string());
        }
        let mut trust = TurboTrustRegion::new(dimensions, length);
        trust.set_num_arms(1);
        Ok(Self {
            search: Search::new(base, base_value, leaves, capacity, backend)?,
            trust,
            dimensions,
            num_pert,
            outcomes: vec![f64::from(base_value)],
            best: base_value,
            restarts: 0,
        })
    }

    pub fn ask(&mut self, seeds: &[u64], mut config: Ask) -> Result<TurboTrial, String> {
        let length = self.trust.length() as f32;
        let probability = (self.num_pert as f64 / self.dimensions as f64).min(1.0) as f32;
        config.length = probability * length;
        let inner = self.search.ask(seeds, config)?;
        Ok(TurboTrial {
            inner,
            index: inner.index,
            seed: inner.seed,
            score: inner.score,
            length,
            probability,
        })
    }

    pub fn row(&self, trial: TurboTrial) -> Result<Vec<u8>, String> {
        self.search.row(trial.inner)
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
    pub fn device_row(&self, trial: TurboTrial) -> Result<(u64, usize, usize), String> {
        self.search.device_row(trial.inner)
    }

    pub fn tell(&mut self, trial: TurboTrial, value: f32) -> Result<bool, String> {
        if !value.is_finite() {
            return Err("trial value must be finite".to_string());
        }
        let accepted = value > self.best;
        self.search.tell(trial.inner, value, accepted)?;
        self.best = self.best.max(value);
        self.outcomes.push(f64::from(value));
        let outcomes = Array1::from_vec(self.outcomes.clone());
        self.trust
            .update(&outcomes.view(), outcomes.len())
            .map_err(|error| error.to_string())?;
        if self.trust.needs_restart() {
            self.trust.restart();
            self.search.restart(self.best)?;
            self.restarts += 1;
        }
        Ok(accepted)
    }

    pub fn length(&self) -> f64 {
        self.trust.length()
    }

    pub fn probability(&self) -> f64 {
        (self.num_pert as f64 / self.dimensions as f64).min(1.0)
    }

    pub fn best(&self) -> f32 {
        self.best
    }

    pub fn restarts(&self) -> usize {
        self.restarts
    }

    pub fn history_len(&self) -> usize {
        self.search.history_len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn leaves() -> Vec<Leaf> {
        vec![Leaf::new(0, 100, 8, 0.1, 1.0, 0.1).unwrap()]
    }

    #[test]
    fn owns_state() {
        let mut search = TurboSearch::new(
            &[8; 100],
            0.0,
            leaves(),
            8,
            ComputeBackend::Cpu,
            20,
            TRLengthConfig::default(),
        )
        .unwrap();
        let ask = Ask {
            neighbors: 1,
            ..Ask::default()
        };
        let trial = search.ask(&[7, 11], ask).unwrap();
        assert_eq!(trial.probability, 0.2);
        assert!(search.tell(trial, 1.0).unwrap());
        assert_eq!(search.best(), 1.0);
        let trial = search.ask(&[13, 17], ask).unwrap();
        assert!(!search.tell(trial, 0.5).unwrap());
        assert_eq!(search.history_len(), 3);
    }

    #[test]
    fn adapts_length() {
        let mut search = TurboSearch::new(
            &[8; 100],
            0.0,
            leaves(),
            8,
            ComputeBackend::Cpu,
            20,
            TRLengthConfig::default(),
        )
        .unwrap();
        let initial = search.length();
        let ask = Ask {
            neighbors: 1,
            ..Ask::default()
        };
        for (seed, reward) in [(7, 1.0), (11, 2.0), (13, 3.0), (17, 4.0)] {
            let trial = search.ask(&[seed], ask).unwrap();
            assert!(search.tell(trial, reward).unwrap());
        }
        assert!(search.length() > initial);
    }
}
