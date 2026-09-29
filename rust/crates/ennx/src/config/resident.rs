use super::*;

/// Fully resolved ENN policy for accelerator-resident search.
#[derive(Debug, Clone, Copy)]
pub struct ResidentEnnConfig {
    pub pool: crate::procedural_pool::ProceduralPool,
    pub proposal_method: crate::procedural_pool::ProposalMethod,
    pub ask: crate::trials::Ask,
    pub num_candidates: usize,
    pub num_samples: usize,
    pub fit_neighbors: bool,
    pub distance_scaling: DistanceScaling,
    pub history_geometry: HistoryGeometry,
    pub local_scale_neighbors: usize,
}

impl ConfigOverrides {
    /// Resolve the shared ENN/acquisition parameters consumed by Metal, OpenCL,
    /// and CUDA resident search kernels.
    pub fn resident_enn(&self, seed: u64) -> Result<ResidentEnnConfig, String> {
        self.resident_pool()?;
        let defaults = crate::trials::Ask::default();
        let effective = self.apply_to(turbo_enn());
        let SurrogateConfig::ENN(enn) = effective.surrogate else {
            return Err("resident search requires an ENN surrogate".into());
        };
        if enn.num_candidates == 0 {
            return Err("resident ENN num_candidates must be positive".into());
        }
        if enn.num_samples == 0 {
            return Err("resident ENN num_samples must be positive".into());
        }
        let epistemic_scale = self
            .epistemic_scale
            .unwrap_or(f64::from(defaults.epistemic_scale));
        let aleatoric_scale = self
            .aleatoric_scale
            .unwrap_or(f64::from(defaults.aleatoric_scale));
        let params = crate::params::ENNParams::new(enn.k, epistemic_scale, aleatoric_scale)
            .map_err(|error| format!("resident ENN parameters: {error}"))?;
        let neighbors = usize::try_from(params.k_neighbors)
            .map_err(|_| "resident ENN k_neighbors does not fit usize")?;
        let y_scale = self.y_scale.unwrap_or(f64::from(defaults.y_scale));
        if !y_scale.is_finite() || y_scale < 0.0 || !(y_scale as f32).is_finite() {
            return Err("resident ENN y_scale must be finite, nonnegative, and fit FP32".into());
        }
        for (name, value) in [
            ("epistemic_scale", params.epistemic_scale),
            ("aleatoric_scale", params.aleatoric_scale),
        ] {
            if !(value as f32).is_finite() {
                return Err(format!("resident ENN {name} must fit FP32"));
            }
        }
        let (acquisition, beta) = match effective.acquisition {
            AcquisitionConfig::UCB { beta } => {
                if !beta.is_finite() || !(beta as f32).is_finite() {
                    return Err("resident ENN UCB beta must be finite FP32".into());
                }
                (crate::weights::AcquisitionKind::Ucb, beta as f32)
            }
            AcquisitionConfig::Thompson => {
                (crate::weights::AcquisitionKind::Thompson, defaults.beta)
            }
            AcquisitionConfig::Random | AcquisitionConfig::Pareto => {
                return Err("resident ENN supports only UCB or Thompson acquisition".into());
            }
        };
        let (distance_scaling, local_scale_neighbors) = self.resident_geometry()?;
        Ok(ResidentEnnConfig {
            pool: self.resident_pool()?,
            proposal_method: self.proposal_method.unwrap_or_default(),
            ask: crate::trials::Ask {
                neighbors,
                epistemic_scale: params.epistemic_scale as f32,
                aleatoric_scale: params.aleatoric_scale as f32,
                y_scale: y_scale as f32,
                beta,
                acquisition,
                seed,
                ..defaults
            },
            num_candidates: enn.num_candidates,
            num_samples: enn.num_samples,
            fit_neighbors: self.fit_neighbors.unwrap_or(false),
            distance_scaling,
            history_geometry: self.history_geometry.unwrap_or_default(),
            local_scale_neighbors,
        })
    }

    fn resident_geometry(&self) -> Result<(DistanceScaling, usize), String> {
        let scaling = self.distance_scaling.unwrap_or_default();
        let neighbors = self.local_scale_neighbors.unwrap_or(8);
        if neighbors == 0 || neighbors > 128 {
            return Err("resident ENN local_scale_neighbors must be in 1..=128".into());
        }
        if scaling == DistanceScaling::Global && self.local_scale_neighbors.is_some() {
            return Err("local_scale_neighbors requires distance_scaling = 'self_tuning'".into());
        }
        Ok((scaling, neighbors))
    }

    /// Resolve an ask for backends that retain only a bounded live history.
    pub fn resident_ask(
        &self,
        live_history: usize,
        seed: u64,
    ) -> Result<crate::trials::Ask, String> {
        if live_history == 0 {
            return Err("resident ENN requires at least one live observation".into());
        }
        let mut ask = self.resident_enn(seed)?.ask;
        ask.neighbors = ask.neighbors.min(live_history);
        Ok(ask)
    }
}
