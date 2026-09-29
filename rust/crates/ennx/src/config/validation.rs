//! Experiment validation stays independent of configuration defaults and lowering.

use super::*;

impl ConfigOverrides {
    pub(super) fn validate_selection(&self) -> Result<bool, String> {
        if !matches!(
            self.experiment,
            Some(
                TurboEnnExperiment::EndToEnd
                    | TurboEnnExperiment::MoeLayer
                    | TurboEnnExperiment::Pretrain
                    | TurboEnnExperiment::Generation
            )
        ) {
            return Err(
                "experiment must be 'end-to-end', 'moe-layer', 'pretrain', or 'generation'".into(),
            );
        }
        if self.rounds() == 0 {
            return Err("rounds must be positive".into());
        }
        if self.reps() == 0 {
            return Err("reps must be positive".into());
        }
        if self.target_ms() == 0 {
            return Err("target_round_ms must be positive".into());
        }
        if self.output().as_os_str().is_empty() {
            return Err("output must be nonempty".into());
        }
        Ok(matches!(
            self.experiment,
            Some(TurboEnnExperiment::Pretrain | TurboEnnExperiment::Generation)
        ))
    }

    pub(super) fn validate_fields(&self, pretrain: bool) -> Result<(), String> {
        self.validate_objectives()?;
        self.resident_pool()?;
        self.validate_identity(pretrain)?;
        if self.validation_dataset.is_some() && (!pretrain || self.generation.is_some()) {
            return Err("validation_dataset requires corpus-prefix pretraining".into());
        }
        if self.validation_interval == Some(0)
            || (self.generation.is_none()
                && self.validation_interval.is_some()
                && self.validation_dataset.is_none()
                && self.dataset.is_some())
        {
            return Err(
                "validation_interval requires a held-out dataset and must be positive".into(),
            );
        }
        if (self.validation_interval.is_some() || self.selection.is_some()) && !pretrain {
            return Err(
                "validation intervals and selection ablations require pretrain experiments".into(),
            );
        }
        if self
            .generation
            .as_ref()
            .and_then(|config| config.signal_gate.as_ref())
            .is_some_and(|gate| gate.rounds > self.rounds())
        {
            return Err("signal gate must finish within the configured rounds".into());
        }
        if self.corpus == Some(PretrainCorpus::Fineweb10btPilotV1) && self.generation.is_some() {
            return Err(
                "FineWeb optimizer pilot uses corpus-prefix NLL, not generation rewards".into(),
            );
        }
        self.validate_geometry(pretrain)
    }

    fn validate_identity(&self, pretrain: bool) -> Result<(), String> {
        let diffusion = self
            .generation
            .as_ref()
            .is_some_and(|generation| generation.draft.is_some());
        let diffusion_model = self.model == Some(PretrainModel::FbtPisa1DiffusionMhc4V1);
        if diffusion != diffusion_model {
            return Err("fbt-pisa1-diffusion-mhc4-v1 requires generation.draft; causal models cannot run diffusion".into());
        }
        if self.kernel_trial.is_some() && !pretrain {
            return Err("kernel trials require a pretrain experiment".into());
        }
        if !pretrain && self.scorer_stage_samples.is_some() {
            return Err("scorer stage timing is supported only by pretrain experiments".into());
        }
        if self.scorer_stage_samples == Some(0) {
            return Err("scorer_stage_samples must be positive when configured".into());
        }
        if pretrain && self.model.is_none() {
            let models = PretrainModel::ALL.map(PretrainModel::id).join(", ");
            return Err(format!("pretrain model must be one of: {models}"));
        }
        if self.experiment == Some(TurboEnnExperiment::Pretrain)
            && !matches!(
                self.corpus,
                Some(
                    PretrainCorpus::StackV3PythonPilotV1
                        | PretrainCorpus::StackV3Python800kV1
                        | PretrainCorpus::Fineweb10btPilotV1
                )
            )
        {
            return Err(
                "pretrain corpus must be stack-v3-python-pilot-v1, stack-v3-python-800k-v1, or fineweb-10bt-pilot-v1".into(),
            );
        }
        Ok(())
    }

    fn validate_geometry(&self, pretrain: bool) -> Result<(), String> {
        self.validate_program(pretrain)?;
        if !pretrain && self.perturbation.is_some() {
            return Err("perturbation is supported only by pretrain experiments".into());
        }
        if !pretrain && self.objective_reference.is_some() {
            return Err("objective reference is supported only by pretrain experiments".into());
        }
        if !pretrain
            && (self.distance_scaling.is_some()
                || self.history_geometry.is_some()
                || self.local_scale_neighbors.is_some())
        {
            return Err("distance scaling is supported only by pretrain experiments".into());
        }
        if self.history_geometry == Some(HistoryGeometry::Latent)
            && self.perturbation() != crate::Perturbation::Rademacher
        {
            return Err("latent history geometry requires Rademacher perturbations".into());
        }
        if !pretrain && self.reps() > 1 {
            return Err("reps greater than one are supported only by pretrain".into());
        }
        if self.experiment == Some(TurboEnnExperiment::MoeLayer) {
            if self.model.is_some() || self.corpus.is_some() {
                return Err("model and corpus presets are supported only by pretrain".into());
            }
            return Ok(());
        }
        if !pretrain && (self.dataset.is_some() || self.model.is_some() || self.corpus.is_some()) {
            return Err(
                "dataset, model, and corpus are supported only by pretrain experiments".into(),
            );
        }
        Ok(())
    }

    fn validate_program(&self, pretrain: bool) -> Result<(), String> {
        let spectral = matches!(
            self.proposal_method,
            Some(
                crate::procedural_pool::ProposalMethod::SpectralBasis
                    | crate::procedural_pool::ProposalMethod::PolynomialThreshold
            )
        );
        let supported = pretrain
            && self.perturbation() == crate::Perturbation::Rademacher
            && self.history_geometry == Some(HistoryGeometry::Latent);
        if spectral && !supported {
            return Err(
                "spectral-basis and polynomial-threshold proposals require pretraining, Rademacher coordinates, and latent history geometry".into(),
            );
        }
        Ok(())
    }

    pub(super) fn validate_seeds(&self) -> Result<(), String> {
        if self.reps() > 1
            && [
                self.model_seed,
                self.reference_seed,
                self.proposal_seed,
                self.acquisition_seed,
                self.generation
                    .as_ref()
                    .and_then(|generation| generation.seed),
            ]
            .iter()
            .any(Option::is_some)
        {
            return Err("reps greater than one require internally derived seeds".into());
        }
        let fixed_seeds = [self.model_seed(), self.reference_seed()];
        if fixed_seeds.iter().any(|&seed| seed > i64::MAX as u64) {
            return Err("seeds must fit TOML signed 64-bit integers".into());
        }
        for rep in 0..self.reps() {
            let seeds = [self.proposal_seeded(rep), self.acquisition_seeded(rep)];
            if seeds.iter().any(|&seed| seed > i64::MAX as u64) {
                return Err("seeds must fit TOML signed 64-bit integers".into());
            }
            for seed in [self.proposal_seeded(rep), self.acquisition_seeded(rep)] {
                seed.checked_add(u64::from(self.rounds() - 1))
                    .ok_or("round seed overflow")?;
            }
        }
        Ok(())
    }

    pub(super) fn validate_resident(&self, pretrain: bool) -> Result<(), String> {
        if self.trust_region_kind == Some(TrustRegionKind::Morbo) {
            return Err("the LocalV1 FBT latency run requires the TuRBO trust region".into());
        }
        if !pretrain && self.trust_region_shape.is_some() {
            return Err("trust-region shape is supported only by pretrain experiments".into());
        }
        match (self.trust_region_kind, self.reliability_controller) {
            (Some(TrustRegionKind::Reliability), Some(config)) if pretrain => {
                config.validate()?;
            }
            (Some(TrustRegionKind::Reliability), _) if !pretrain => {
                return Err(
                    "the reliability controller is supported only by pretrain experiments".into(),
                );
            }
            (Some(TrustRegionKind::Reliability), None) => {
                return Err(
                    "trust-region method 'reliability' requires [trust-region.reliability]".into(),
                );
            }
            (_, Some(_)) => {
                return Err("[trust-region.reliability] requires method = 'reliability'".into());
            }
            _ => {}
        }
        let effective = self.apply_to(turbo_enn());
        match effective.acquisition {
            AcquisitionConfig::Random | AcquisitionConfig::Pareto => {
                return Err(
                    "the TuRBO-ENN round experiment supports only UCB or Thompson acquisition"
                        .into(),
                );
            }
            AcquisitionConfig::UCB { beta } if !beta.is_finite() || !(beta as f32).is_finite() => {
                return Err("turbo_enn UCB beta must be finite FP32".into());
            }
            AcquisitionConfig::UCB { .. } | AcquisitionConfig::Thompson => {}
        }
        self.resident_enn(self.acquisition_seed())?;
        let TrustRegionConfig::Turbo(length) = effective.trust_region else {
            return Err("the LocalV1 FBT latency run requires the TuRBO trust region".into());
        };
        validate_lengths(length)
    }

    pub(super) fn validate_controller(&self) -> Result<(), String> {
        let unsupported = self.candidate_rv.is_some()
            || self.num_candidates_factor.is_some()
            || self.min_candidates.is_some()
            || self.max_candidates.is_some()
            || self.num_candidates_per_arm.is_some()
            || self.num_pert.is_some()
            || self.index_driver.is_some()
            || self.scale_x.is_some()
            || self.noise_aware.is_some()
            || self.failure_tolerance_dim.is_some()
            || self.enn_storage.is_some()
            || self.work_dir.is_some()
            || self.y_bounds.is_some()
            || self.num_metrics.is_some()
            || self.alpha.is_some()
            || self.rescalarize.is_some();
        if unsupported {
            return Err("the TuRBO-ENN round experiment received optimizer fields that its fixed resident controller does not implement".into());
        }
        Ok(())
    }
}
