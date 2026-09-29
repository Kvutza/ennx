//! Explicit vector acquisition policy, independent of backend scalar control.
use super::*;

#[cfg(test)]
#[path = "objectives_tests.rs"]
mod tests;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[deser(deny_unknown_fields, rename_all = "kebab-case")]
pub struct ResidentObjectiveConfig {
    pub mode: ResidentObjectiveMode,
    pub scales: Vec<f32>,
    pub preferences: Option<Vec<f32>>,
    pub regions: Option<usize>,
    pub alpha: Option<f32>,
    pub seed_domain: Option<String>,
    pub rescalarize: Option<Rescalarize>,
    pub clip: Option<bool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[deser(rename_all = "kebab-case")]
pub enum ResidentObjectiveMode {
    Pareto,
    Morbo,
}

impl ResidentObjectiveConfig {
    pub fn validate(&self) -> Result<(), String> {
        if !(2..=crate::objective_observation::MAX_OBJECTIVES).contains(&self.scales.len())
            || self.scales.iter().any(|v| !v.is_finite() || *v <= 0.0)
        {
            return Err(
                "objective_acquisition.scales requires 2..=8 positive finite values".into(),
            );
        }
        let explicit = self.preferences.is_some()
            || self.regions.is_some()
            || self.alpha.is_some()
            || self.seed_domain.is_some()
            || self.rescalarize.is_some()
            || self.clip.is_some();
        match self.mode {
            ResidentObjectiveMode::Pareto if explicit => {
                Err("Pareto does not accept MORBO scalarization fields".into())
            }
            ResidentObjectiveMode::Morbo => self.validate_morbo(),
            ResidentObjectiveMode::Pareto => Ok(()),
        }
    }

    fn validate_morbo(&self) -> Result<(), String> {
        let preferences = self
            .preferences
            .as_ref()
            .ok_or("MORBO requires preferences")?;
        if preferences.len() != self.scales.len()
            || preferences.iter().any(|v| !v.is_finite() || *v <= 0.0)
            || !(2..=8).contains(&self.regions.unwrap_or(4))
            || !self.alpha.is_some_and(|v| v.is_finite() && v >= 0.0)
            || !self
                .seed_domain
                .as_ref()
                .is_some_and(|v| !v.trim().is_empty())
            || self.rescalarize.is_none()
            || self.clip != Some(true)
        {
            return Err("MORBO requires positive preferences, 2..=8 regions, nonnegative alpha, nonempty seed_domain, explicit rescalarize and clip=true".into());
        }
        Ok(())
    }
}

impl ConfigOverrides {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn resident_objectives(
        &self,
        rep: u32,
    ) -> Result<Option<crate::bf16_metal::ObjectivePolicy>, String> {
        use crate::bf16_metal::{ObjectiveAcquisition, ObjectivePolicy};
        self.objective_acquisition
            .as_ref()
            .map(|config| {
                config.validate()?;
                let acquisition = match config.mode {
                    ResidentObjectiveMode::Pareto => ObjectiveAcquisition::Pareto,
                    ResidentObjectiveMode::Morbo => ObjectiveAcquisition::Morbo {
                        weights: config.preferences.clone().unwrap(),
                        regions: config.regions.unwrap_or(4),
                        alpha: config.alpha.unwrap(),
                        seed: self.derived_seed(rep, config.seed_domain.as_ref().unwrap(), 0),
                        rescalarize: config.rescalarize.unwrap(),
                        clip: config.clip.unwrap(),
                    },
                };
                Ok(ObjectivePolicy {
                    acquisition,
                    scales: config.scales.clone(),
                })
            })
            .transpose()
    }

    pub(super) fn validate_objectives(&self) -> Result<(), String> {
        let vector = self.generation.as_ref().is_some_and(|generation| {
            (!generation.causal_pretrain() || self.objective_acquisition.is_some())
                && matches!(
                    generation.reward,
                    GenerationReward::CodeObjectives { .. }
                        | GenerationReward::DraftedCode { .. }
                        | GenerationReward::CodeExecution { .. }
                        | GenerationReward::CommandObjectives { .. }
                )
        });
        if let Some(config) = &self.objective_acquisition {
            config.validate()?;
            if self.generation.as_ref().is_some_and(|generation| {
                matches!(generation.reward, GenerationReward::DraftedCode { .. })
            }) && config.scales.len() != 3
            {
                return Err("drafted-code requires three acquisition scales".into());
            }
            if config.mode == ResidentObjectiveMode::Morbo
                && self.history_geometry != Some(HistoryGeometry::Latent)
            {
                return Err("MORBO requires latent history geometry".into());
            }
            if config.mode == ResidentObjectiveMode::Morbo
                && self.reliability_controller().is_some()
            {
                return Err(
                    "MORBO owns trust-region adaptation and cannot use the reliability controller"
                        .into(),
                );
            }
            if !vector {
                return Err("objective_acquisition requires independent vector observations; this experiment currently reports scalar rewards only (no implicit scalarization)".into());
            }
        } else if vector {
            return Err("vector generation rewards require explicit objective_acquisition".into());
        }
        Ok(())
    }
}
