use ndarray::{Array1, ArrayView1, ArrayView2};
use rand::RngCore;

use crate::error::ENNError;
use crate::mbtrregn::{MorboTrustRegion, Rescalarize};
use crate::trregncfg::TrustRegionConfig;
use crate::trust_region::{TrustRegionError, TurboTrustRegion};

#[derive(Debug)]
pub enum TrustRegionState {
    Turbo(TurboTrustRegion),
    Morbo(Box<MorboTrustRegion>),
}

impl TrustRegionState {
    pub fn from_config(
        num_dim: usize,
        config: &TrustRegionConfig,
        rng: &mut dyn RngCore,
    ) -> Result<Self, TrustRegionError> {
        match config {
            TrustRegionConfig::Turbo(cfg) => Ok(TrustRegionState::Turbo(TurboTrustRegion::new(
                num_dim, *cfg,
            ))),
            TrustRegionConfig::Morbo(settings) => Ok(TrustRegionState::Morbo(Box::new(
                MorboTrustRegion::new(num_dim, settings.clone(), rng)?,
            ))),
        }
    }

    pub fn is_morbo(&self) -> bool {
        matches!(self, TrustRegionState::Morbo(_))
    }

    pub fn morbo(&self) -> Option<&MorboTrustRegion> {
        match self {
            TrustRegionState::Morbo(m) => Some(m.as_ref()),
            _ => None,
        }
    }

    pub fn morbo_mut(&mut self) -> Option<&mut MorboTrustRegion> {
        match self {
            TrustRegionState::Morbo(m) => Some(m.as_mut()),
            _ => None,
        }
    }

    pub fn length(&self) -> f64 {
        match self {
            TrustRegionState::Turbo(t) => t.length(),
            TrustRegionState::Morbo(m) => m.as_ref().length(),
        }
    }

    /// Observations accounted for by the TuRBO length-adaptation state (0 for Morbo).
    pub fn turbo_obs(&self) -> usize {
        match self {
            TrustRegionState::Turbo(t) => t.prev_obs(),
            TrustRegionState::Morbo(_) => 0,
        }
    }

    pub fn set_obs(&mut self, prev_obs: usize) {
        match self {
            TrustRegionState::Turbo(t) => t.set_watermark(prev_obs),
            TrustRegionState::Morbo(_) => {}
        }
    }

    pub fn set_arms(&mut self, num_arms: usize) {
        match self {
            TrustRegionState::Turbo(t) => t.set_arms(num_arms),
            TrustRegionState::Morbo(m) => m.as_mut().set_arms(num_arms),
        }
    }

    pub fn compute_bounds(
        &self,
        x_center: &ArrayView1<f64>,
        lengthscales: Option<&ArrayView1<f64>>,
    ) -> (Array1<f64>, Array1<f64>) {
        match self {
            TrustRegionState::Turbo(t) => t.compute_bounds(x_center, lengthscales),
            TrustRegionState::Morbo(m) => m.as_ref().compute_bounds(x_center, lengthscales),
        }
    }

    pub fn needs_restart(&self) -> bool {
        match self {
            TrustRegionState::Turbo(t) => t.needs_restart(),
            TrustRegionState::Morbo(m) => m.as_ref().needs_restart(),
        }
    }

    pub fn restart(&mut self, rng: Option<&mut dyn RngCore>) {
        match self {
            TrustRegionState::Turbo(t) => t.restart(),
            TrustRegionState::Morbo(m) => m.as_mut().restart(rng),
        }
    }

    pub fn resample_propose(&mut self, rng: &mut dyn RngCore) {
        if let TrustRegionState::Morbo(m) = self {
            if m.as_ref().rescalarize() == Rescalarize::OnPropose {
                m.as_mut().resample_weights(rng);
            }
        }
    }

    pub fn num_metrics(&self) -> usize {
        match self {
            TrustRegionState::Turbo(_) => 1,
            TrustRegionState::Morbo(m) => m.as_ref().num_metrics(),
        }
    }

    pub fn morbo_scalarize(
        &self,
        y: &ArrayView2<f64>,
        clip: bool,
    ) -> Result<Array1<f64>, TrustRegionError> {
        match self {
            TrustRegionState::Morbo(m) => m.as_ref().scalarize(y, clip),
            _ => Err(TrustRegionError::InvalidState(
                "scalarize requires Morbo trust region".to_string(),
            )),
        }
    }

    pub fn morbo_only(&mut self, y_new: &ArrayView2<f64>) -> Result<(), ENNError> {
        match self {
            TrustRegionState::Morbo(m) => {
                m.as_mut().update_incremental(y_new);
                Ok(())
            }
            _ => Err(ENNError::InvalidParameter(
                "morbo_only requires Morbo".to_string(),
            )),
        }
    }

    pub fn update_morbo(
        &mut self,
        y_incumbent: &ArrayView1<f64>,
        num_obs: usize,
    ) -> Result<(), ENNError> {
        match self {
            TrustRegionState::Morbo(m) => m
                .as_mut()
                .update_only(y_incumbent, num_obs)
                .map_err(|e| ENNError::InvalidParameter(e.to_string())),
            _ => Err(ENNError::InvalidParameter(
                "update_morbo requires Morbo".to_string(),
            )),
        }
    }

    pub fn rescale_morbo(&mut self, num_obs: usize) -> Result<(), ENNError> {
        match self {
            TrustRegionState::Morbo(m) => m
                .as_mut()
                .rescalarize_weights(num_obs)
                .map_err(|e| ENNError::InvalidParameter(e.to_string())),
            _ => Ok(()),
        }
    }

    pub fn tell_update(
        &mut self,
        y_all: &ArrayView2<f64>,
        y_incumbent: &ArrayView1<f64>,
        num_obs: usize,
    ) -> Result<(), ENNError> {
        match self {
            TrustRegionState::Turbo(t) => {
                if y_all.ncols() != 1 {
                    return Err(ENNError::InvalidParameter(format!(
                        "Turbo TR expects 1 objective column, got {}",
                        y_all.ncols()
                    )));
                }
                if y_incumbent.len() != 1 {
                    return Err(ENNError::InvalidParameter(format!(
                        "Turbo TR expects 1 incumbent scalar, got {}",
                        y_incumbent.len()
                    )));
                }
                let y_1d = y_all.column(0);
                t.update_history(&y_1d, num_obs, y_incumbent[0])
                    .map_err(|e| ENNError::InvalidParameter(e.to_string()))
            }
            TrustRegionState::Morbo(m) => m
                .as_mut()
                .update(&y_all.view(), y_incumbent)
                .map_err(|e| ENNError::InvalidParameter(e.to_string())),
        }
    }

    /// TuRBO trust-region update from the newly told batch only (O(batch), not O(history)).
    pub fn tell_batch(
        &mut self,
        y_new: &ArrayView2<f64>,
        y_incumbent: &ArrayView1<f64>,
        num_obs: usize,
    ) -> Result<(), ENNError> {
        match self {
            TrustRegionState::Turbo(t) => {
                if y_new.ncols() != 1 {
                    return Err(ENNError::InvalidParameter(format!(
                        "Turbo TR expects 1 objective column, got {}",
                        y_new.ncols()
                    )));
                }
                if y_incumbent.len() != 1 {
                    return Err(ENNError::InvalidParameter(format!(
                        "Turbo TR expects 1 incumbent scalar, got {}",
                        y_incumbent.len()
                    )));
                }
                let y_1d = y_new.column(0);
                t.update_batch(&y_1d, num_obs, y_incumbent[0])
                    .map_err(|e| ENNError::InvalidParameter(e.to_string()))
            }
            TrustRegionState::Morbo(m) => m
                .as_mut()
                .update(&y_new.view(), y_incumbent)
                .map_err(|e| ENNError::InvalidParameter(e.to_string())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    use ndarray::array;

    use crate::mbtrregn::{MorboTRSettings, Rescalarize};
    use crate::trregncfg::TrustRegionConfig;
    use crate::trust_region::TRLengthConfig;

    #[test]
    fn morbo_config() {
        let mut rng = StdRng::seed_from_u64(99);
        let cfg = TrustRegionConfig::Morbo(MorboTRSettings {
            num_metrics: 2,
            alpha: 0.05,
            length: TRLengthConfig::default(),
            rescalarize: Rescalarize::OnPropose,
            noise_aware: false,
        });
        let mut tr = TrustRegionState::from_config(3, &cfg, &mut rng).unwrap();
        assert!(tr.is_morbo());
        assert_eq!(tr.num_metrics(), 2);
        tr.resample_propose(&mut rng);
    }

    #[test]
    fn morbo_propose2() {
        let mut rng = StdRng::seed_from_u64(11);
        let cfg = TrustRegionConfig::Morbo(MorboTRSettings {
            num_metrics: 2,
            alpha: 0.05,
            length: TRLengthConfig::default(),
            rescalarize: Rescalarize::OnPropose,
            noise_aware: false,
        });
        let mut tr = TrustRegionState::from_config(2, &cfg, &mut rng).unwrap();
        let w0 = tr.morbo_mut().expect("morbo").weights().to_owned();
        tr.resample_propose(&mut rng);
        let w1 = tr.morbo_mut().expect("morbo").weights().to_owned();
        tr.resample_propose(&mut rng);
        let w2 = tr.morbo_mut().expect("morbo").weights().to_owned();
        assert_ne!(w0, w1);
        assert_ne!(w1, w2);
    }

    #[test]
    fn turbo_max() {
        use ndarray::Array2;

        let config = TRLengthConfig::default();
        let mut tr = TrustRegionState::Turbo(TurboTrustRegion::new(2, config));
        tr.set_arms(1);

        let y0 = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();
        let inc0 = array![1.0];
        tr.tell_update(&y0.view(), &inc0.view(), 1).unwrap();
        let len_before = tr.length();

        for (i, inc_val) in [2.0_f64, 3.0, 4.0].into_iter().enumerate() {
            let n = i + 2;
            let mut vals = vec![1.0];
            vals.extend(std::iter::repeat_n(0.5, n - 1));
            let y_mat = Array2::from_shape_vec((n, 1), vals).unwrap();
            let inc = array![inc_val];
            tr.tell_update(&y_mat.view(), &inc.view(), n).unwrap();
        }

        assert!(
            tr.length() > len_before,
            "trust region length should expand after three incumbent improvements \
             (incumbent 2→3→4 with flat observed batch max 1.0); \
             batch-only update treats each tell as failure"
        );
    }

    #[test]
    fn morbo_restart() {
        let mut rng = StdRng::seed_from_u64(12);
        let cfg = TrustRegionConfig::Morbo(MorboTRSettings {
            num_metrics: 2,
            alpha: 0.05,
            length: TRLengthConfig::default(),
            rescalarize: Rescalarize::OnRestart,
            noise_aware: false,
        });
        let mut tr = TrustRegionState::from_config(2, &cfg, &mut rng).unwrap();
        let w0 = tr.morbo_mut().expect("morbo").weights().to_owned();
        tr.resample_propose(&mut rng);
        let w1 = tr.morbo_mut().expect("morbo").weights().to_owned();
        assert_eq!(w0, w1);
        tr.restart(Some(&mut rng));
        let w2 = tr.morbo_mut().expect("morbo").weights().to_owned();
        assert_ne!(w1, w2);
    }
}
