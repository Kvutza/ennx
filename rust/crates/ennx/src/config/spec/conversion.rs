//! Typed adaptation to the existing programmatic optimizer API.
use super::*;
use super::{spec_execution::*, spec_optimizer::*};

impl TuneSpec {
    pub fn from_overrides(config: &ConfigOverrides) -> Result<Self, String> {
        config.validate_experiment()?;
        let pool = config.resident_pool()?;
        Ok(Self {
            version: 2,
            experiment: config.experiment.unwrap(),
            model: config.model,
            corpus: config.corpus,
            output: config.output.clone(),
            run: RunSpec {
                rounds: config.rounds(),
                reps: config.reps(),
                target_ms: config.target_ms(),
                selection: config.selection,
                validation_interval: config.validation_interval,
            },
            data: DataSpec {
                train: config.dataset.clone(),
                validation: config.validation_dataset.clone(),
            },
            proposal: ProposalSpec {
                method: config.proposal_method.unwrap_or_default(),
                distribution: config.perturbation,
                candidates: pool.count(),
                arms: pool.arms(),
            },
            enn: EnnSpec::from_config(config)?,
            acquisition: AcquisitionSpec::from_config(config),
            trust_region: config.reliability_controller().map_or_else(
                || {
                    let bounds = TrustRegionBounds::from_config(config);
                    if config
                        .objective_acquisition
                        .as_ref()
                        .is_some_and(|objective| objective.mode == ResidentObjectiveMode::Morbo)
                    {
                        let objective = config.objective_acquisition.as_ref().unwrap();
                        TrustRegionSpec::Morbo {
                            bounds,
                            regions: objective.regions.unwrap(),
                            rescalarize: objective.rescalarize.unwrap(),
                            clip: objective.clip.unwrap(),
                        }
                    } else {
                        TrustRegionSpec::Turbo { bounds }
                    }
                },
                |policy| TrustRegionSpec::Reliability {
                    bounds: TrustRegionBounds::from_config(config),
                    policy,
                },
            ),
            objective: ObjectiveSpec {
                reference: config.objective_reference,
            },
            seeds: SeedSpec {
                model: config.model_seed,
                reference: config.reference_seed,
                proposal: config.proposal_seed,
                acquisition: config.acquisition_seed,
            },
            diagnostics: DiagnosticSpec {
                trace: config.trace(),
                stage_samples: config.scorer_stage_samples,
                kernels: config.kernel_trial.clone(),
            },
            generation: config.generation.clone(),
        })
    }

    pub fn overrides(&self) -> Result<ConfigOverrides, String> {
        if self.proposal.arms == 0
            || self.proposal.candidates == 0
            || !self.proposal.candidates.is_multiple_of(self.proposal.arms)
        {
            return Err(
                "proposal.candidates must be positive and divisible by proposal.arms".into(),
            );
        }
        let mut config = ConfigOverrides {
            experiment: Some(self.experiment),
            model: self.model,
            corpus: self.corpus,
            output: self.output.clone(),
            rounds: Some(self.run.rounds),
            reps: Some(self.run.reps),
            target_round_ms: Some(self.run.target_ms),
            selection: self.run.selection,
            validation_interval: self.run.validation_interval,
            dataset: self.data.train.clone(),
            validation_dataset: self.data.validation.clone(),
            proposal_pool: Some(crate::procedural_pool::ProceduralPoolConfig {
                arms: self.proposal.arms,
                candidates_per_arm: self.proposal.candidates / self.proposal.arms,
            }),
            proposal_method: Some(self.proposal.method),
            perturbation: self.proposal.distribution,
            k_neighbors: Some(self.enn.neighbors),
            epistemic_scale: Some(self.enn.epistemic_scale),
            aleatoric_scale: Some(self.enn.aleatoric_scale),
            y_scale: Some(self.enn.y_scale),
            history_geometry: (self.enn.geometry != HistoryGeometry::default())
                .then_some(self.enn.geometry),
            distance_scaling: (self.enn.scaling != DistanceScaling::default())
                .then_some(self.enn.scaling),
            local_scale_neighbors: self.enn.local_neighbors,
            fit_neighbors: Some(self.enn.fit.neighbors),
            num_candidates: Some(self.enn.fit.candidates),
            num_samples: Some(self.enn.fit.samples),
            length_init: Some(self.trust_region.bounds().initial),
            length_min: Some(self.trust_region.bounds().min),
            length_max: Some(self.trust_region.bounds().max),
            trust_region_shape: self.trust_region.bounds().shape,
            objective_reference: self.objective.reference,
            model_seed: self.seeds.model,
            reference_seed: self.seeds.reference,
            proposal_seed: self.seeds.proposal,
            acquisition_seed: self.seeds.acquisition,
            trace: self.diagnostics.trace.then_some(true),
            scorer_stage_samples: self.diagnostics.stage_samples,
            kernel_trial: self.diagnostics.kernels.clone(),
            generation: self.generation.clone(),
            ..Default::default()
        };
        self.acquisition.apply(&mut config);
        match &self.trust_region {
            TrustRegionSpec::Turbo { .. } => {
                if matches!(self.acquisition, AcquisitionSpec::AugmentedChebyshev { .. }) {
                    return Err(
                        "augmented-chebyshev acquisition requires trust-region method='morbo'"
                            .into(),
                    );
                }
                config.trust_region_kind = Some(TrustRegionKind::Turbo);
            }
            TrustRegionSpec::Morbo {
                regions,
                rescalarize,
                clip,
                ..
            } => {
                if !matches!(self.acquisition, AcquisitionSpec::AugmentedChebyshev { .. }) {
                    return Err(
                        "trust-region method='morbo' requires augmented-chebyshev acquisition"
                            .into(),
                    );
                }
                let objective = config.objective_acquisition.as_mut().unwrap();
                objective.regions = Some(*regions);
                objective.rescalarize = Some(*rescalarize);
                objective.clip = Some(*clip);
                config.trust_region_kind = Some(TrustRegionKind::Turbo);
            }
            TrustRegionSpec::Reliability { policy, .. } => {
                if matches!(self.acquisition, AcquisitionSpec::AugmentedChebyshev { .. }) {
                    return Err(
                        "MORBO owns trust-region adaptation and cannot use reliability".into(),
                    );
                }
                config.trust_region_kind = Some(TrustRegionKind::Reliability);
                config.reliability_controller = Some(*policy);
            }
        }
        config.validate_experiment()?;
        Ok(config)
    }
}

impl AcquisitionSpec {
    fn from_config(config: &ConfigOverrides) -> Self {
        match config.objective_acquisition.as_ref() {
            Some(vector) if vector.mode == ResidentObjectiveMode::Pareto => Self::Pareto {
                scales: vector.scales.clone(),
            },
            Some(vector) => Self::AugmentedChebyshev {
                scales: vector.scales.clone(),
                preferences: vector.preferences.clone().unwrap(),
                alpha: vector.alpha.unwrap(),
                seed_domain: vector.seed_domain.clone().unwrap(),
            },
            None => match config.acquisition.unwrap_or_default() {
                AcquisitionConfig::UCB { beta } => Self::Ucb { beta },
                AcquisitionConfig::Thompson => Self::Thompson,
                _ => unreachable!("validated resident acquisition"),
            },
        }
    }

    fn apply(&self, config: &mut ConfigOverrides) {
        config.acquisition = Some(match self {
            Self::Ucb { beta } => AcquisitionConfig::UCB { beta: *beta },
            Self::Thompson => AcquisitionConfig::Thompson,
            // Vector acquisition owns selection. The scalar ask carries the
            // compatible posterior evaluator already used by resident kernels.
            Self::Pareto { .. } | Self::AugmentedChebyshev { .. } => AcquisitionConfig::default(),
        });
        config.objective_acquisition = match self {
            Self::Pareto { scales } => Some(ResidentObjectiveConfig {
                mode: ResidentObjectiveMode::Pareto,
                scales: scales.clone(),
                preferences: None,
                regions: None,
                alpha: None,
                seed_domain: None,
                rescalarize: None,
                clip: None,
            }),
            Self::AugmentedChebyshev {
                scales,
                preferences,
                alpha,
                seed_domain,
            } => Some(ResidentObjectiveConfig {
                mode: ResidentObjectiveMode::Morbo,
                scales: scales.clone(),
                preferences: Some(preferences.clone()),
                regions: None,
                alpha: Some(*alpha),
                seed_domain: Some(seed_domain.clone()),
                rescalarize: None,
                clip: None,
            }),
            _ => None,
        };
    }
}

impl EnnSpec {
    fn from_config(config: &ConfigOverrides) -> Result<Self, String> {
        let resident = config.resident_enn(config.acquisition_seed())?;
        let defaults = Self::default();
        Ok(Self {
            neighbors: config.k_neighbors.unwrap_or(defaults.neighbors),
            epistemic_scale: config.epistemic_scale.unwrap_or(defaults.epistemic_scale),
            aleatoric_scale: config.aleatoric_scale.unwrap_or(defaults.aleatoric_scale),
            y_scale: config.y_scale.unwrap_or(defaults.y_scale),
            geometry: resident.history_geometry,
            scaling: resident.distance_scaling,
            local_neighbors: (resident.distance_scaling == DistanceScaling::SelfTuning)
                .then_some(resident.local_scale_neighbors),
            fit: FitSpec {
                neighbors: resident.fit_neighbors,
                candidates: resident.num_candidates,
                samples: resident.num_samples,
            },
        })
    }
}

impl TrustRegionBounds {
    fn from_config(config: &ConfigOverrides) -> Self {
        let length = config.length();
        Self {
            initial: length.length_init,
            min: length.length_min,
            max: length.length_max,
            shape: config.trust_region_shape,
        }
    }
}
