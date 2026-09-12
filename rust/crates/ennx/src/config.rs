//! Configuration types for the optimizer.

use crate::backend::EnnStorage;
use crate::candidates::CandidateRV;
use crate::index::IndexDriver;
use crate::mbtrregn::{MorboTRSettings, Rescalarize};
use crate::surrogate::ENNSurrogateConfig;
use crate::trregncfg::TrustRegionConfig;
use crate::trust_region::TRLengthConfig;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

mod study;
use study::lower_structured_tune;
pub use study::{DistanceScaling, TrustRegionShape};

/// Optimizer configuration.
#[derive(Debug, Clone)]
pub struct OptimizerConfig {
    /// Surrogate configuration.
    pub surrogate: SurrogateConfig,
    /// Trust region configuration.
    pub trust_region: TrustRegionConfig,
    /// Candidate generation configuration.
    pub candidates: CandidateConfig,
    /// Acquisition function configuration.
    pub acquisition: AcquisitionConfig,
    /// Use surrogate posterior mean for incumbent selection among candidates.
    pub noise_aware: bool,
    pub failure_tolerance_dim: Option<f64>,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            surrogate: SurrogateConfig::ENN(ENNSurrogateConfig::default()),
            trust_region: TrustRegionConfig::default(),
            candidates: CandidateConfig::default(),
            acquisition: AcquisitionConfig::default(),
            noise_aware: false,
            failure_tolerance_dim: None,
        }
    }
}

/// Surrogate type configuration.
#[derive(Debug, Clone)]
pub enum SurrogateConfig {
    /// ENN surrogate.
    ENN(ENNSurrogateConfig),
    /// No surrogate (for LHD/random).
    None,
}

impl Default for SurrogateConfig {
    fn default() -> Self {
        SurrogateConfig::ENN(ENNSurrogateConfig::default())
    }
}

/// Candidate generation configuration.
#[derive(Debug, Clone)]
pub struct CandidateConfig {
    /// Base multiplier for number of candidates.
    pub num_candidates_factor: f64,
    /// Minimum number of candidates.
    pub min_candidates: usize,
    /// Maximum number of candidates (None = no cap). Matches Python default_candidates cap.
    pub max_candidates: Option<usize>,
    /// Optional per-arm multiplier: pool is at least num_arms * this value.
    pub num_candidates_per_arm: Option<usize>,
    /// Number of coordinates to perturb for RAASP-style candidates.
    pub num_pert: usize,
    /// Random variable type for candidates.
    pub candidate_rv: CandidateRV,
}

impl Default for CandidateConfig {
    fn default() -> Self {
        Self {
            num_candidates_factor: 1000.0,
            min_candidates: 100,
            max_candidates: None,
            num_candidates_per_arm: None,
            num_pert: 20,
            candidate_rv: CandidateRV::Uniform,
        }
    }
}

impl CandidateConfig {
    /// Compute number of candidates based on dimension and arms.
    ///
    /// Matches Python `CandidateGenConfig.resolve_candidates`: default base
    /// `min(max_candidates, factor * dim)` when set, optional `max(fixed, per_arm * arms)`,
    /// no `num_arms` multiplier. `max_candidates` caps the formula base only when
    /// `num_candidates_per_arm` is set; otherwise exact-fixed mode uses min=max as pool size.
    pub fn num_candidates(&self, num_dim: usize, num_arms: usize) -> usize {
        let is_exact_fixed = self.num_candidates_factor == 1.0
            && self.max_candidates == Some(self.min_candidates)
            && self.num_candidates_per_arm.is_none();

        let mut base = if is_exact_fixed {
            self.min_candidates
        } else {
            let raw = (self.num_candidates_factor * num_dim as f64) as usize;
            let formula = match self.max_candidates {
                Some(cap) if self.num_candidates_per_arm.is_some() => raw.min(cap),
                Some(cap) if (self.num_candidates_factor - 100.0).abs() < f64::EPSILON => {
                    raw.min(cap)
                }
                _ => raw,
            };
            formula.max(self.min_candidates)
        };

        if let Some(m) = self.num_candidates_per_arm {
            base = base.max(num_arms * m);
        }

        base
    }
}

/// Shared TuRBO-ENN optimizer overrides.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ConfigOverrides {
    /// File-driven study to run. Programmatic optimizer overrides may omit it.
    pub study: Option<TurboEnnStudy>,
    pub acquisition: Option<AcquisitionConfig>,
    /// Neighbors used by the resident accelerator ENN.
    pub k_neighbors: Option<i32>,
    /// Fixed epistemic distance scale for the resident accelerator ENN.
    pub epistemic_scale: Option<f64>,
    /// Fixed aleatoric variance floor for the resident accelerator ENN.
    pub aleatoric_scale: Option<f64>,
    /// Output uncertainty scale used by accelerator acquisition kernels.
    pub y_scale: Option<f64>,
    /// Geometry normalization used before ENN neighbor weighting.
    pub distance_scaling: Option<DistanceScaling>,
    /// Neighbor rank defining each point's local radius under self-tuning scaling.
    pub local_scale_neighbors: Option<usize>,
    /// Select the resident ENN neighbor count by leave-one-out predictive likelihood.
    pub fit_neighbors: Option<bool>,
    pub candidate_rv: Option<CandidateRV>,
    pub num_candidates_factor: Option<f64>,
    pub min_candidates: Option<usize>,
    pub max_candidates: Option<usize>,
    pub num_candidates_per_arm: Option<usize>,
    pub num_pert: Option<usize>,
    pub length_init: Option<f64>,
    pub length_min: Option<f64>,
    pub length_max: Option<f64>,
    pub index_driver: Option<IndexDriver>,
    pub num_samples: Option<usize>,
    pub num_candidates: Option<usize>,
    pub scale_x: Option<bool>,
    pub noise_aware: Option<bool>,
    pub failure_tolerance_dim: Option<f64>,
    pub enn_storage: Option<EnnStorage>,
    pub work_dir: Option<PathBuf>,
    pub y_bounds: Option<Vec<[f64; 2]>>,
    pub trust_region_kind: Option<TrustRegionKind>,
    pub num_metrics: Option<usize>,
    pub alpha: Option<f64>,
    pub rescalarize: Option<Rescalarize>,
    /// Artifact root for a file-driven TuRBO-ENN run.
    pub output: Option<PathBuf>,
    /// Number of complete optimization rounds to measure.
    pub rounds: Option<u32>,
    /// Deterministic repetitions of the same study.
    pub reps: Option<u32>,
    /// Required maximum latency for every measured round.
    pub target_round_ms: Option<u32>,
    /// Emit perturbative per-operation timing records for the fixed round study.
    pub trace: Option<bool>,
    /// Seed for model initialization.
    pub model_seed: Option<u64>,
    /// Seed for the persistent proposal direction.
    pub reference_seed: Option<u64>,
    /// Base seed for per-round proposals.
    pub proposal_seed: Option<u64>,
    /// Base seed for per-round acquisition.
    pub acquisition_seed: Option<u64>,
    /// Fixed ENNXPTN1 token stream for causal pretraining.
    pub dataset: Option<PathBuf>,
    /// Versioned model architecture preset for a pretraining study.
    pub model: Option<PretrainModel>,
    /// Versioned immutable corpus recipe for a pretraining study.
    pub corpus: Option<PretrainCorpus>,
    /// Full-coordinate perturbation distribution generated by the accelerator.
    pub perturbation: Option<crate::Perturbation>,
    /// Trust-region shape policy for the full-weight pretrain study.
    pub trust_region_shape: Option<TrustRegionShape>,
    /// Reliability-aware controller parameters for full-weight pretraining.
    pub reliability_controller: Option<crate::ReliabilityControllerConfig>,
}

/// Fully resolved ENN policy for accelerator-resident search.
#[derive(Debug, Clone, Copy)]
pub struct ResidentEnnConfig {
    pub ask: crate::trials::Ask,
    pub num_candidates: usize,
    pub num_samples: usize,
    pub fit_neighbors: bool,
    pub distance_scaling: DistanceScaling,
    pub local_scale_neighbors: usize,
}

/// Complete workload selected by a file-driven TuRBO-ENN configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TurboEnnStudy {
    EndToEnd,
    MoeLayer,
    Pretrain,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PretrainModel {
    FbtPisa1MoeV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PretrainCorpus {
    StackV3PythonPilotV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrustRegionKind {
    Turbo,
    Morbo,
    Reliability,
}

fn apply_enn(config: &mut OptimizerConfig, overrides: &ConfigOverrides) {
    let SurrogateConfig::ENN(enn_cfg) = &config.surrogate else {
        return;
    };
    let mut enn = enn_cfg.clone();
    if let Some(k) = overrides.k_neighbors {
        enn.k = k;
    }
    if let Some(driver) = overrides.index_driver {
        enn.index_driver = driver;
    }
    if let Some(nfs) = overrides.num_samples {
        enn.num_samples = nfs;
    }
    if let Some(nfc) = overrides.num_candidates {
        enn.num_candidates = nfc;
    }
    if let Some(sx) = overrides.scale_x {
        enn.scale_x = sx;
    }
    if let Some(storage) = overrides.enn_storage {
        enn.storage = storage;
    }
    if let Some(dir) = overrides.work_dir.clone() {
        enn.work_dir = Some(dir);
    }
    if let Some(rows) = overrides.y_bounds.as_ref() {
        let flat = rows.iter().flatten().copied().collect::<Vec<_>>();
        enn.y_bounds = Some(
            ndarray::Array2::from_shape_vec((rows.len(), 2), flat)
                .expect("fixed-width y_bounds rows"),
        );
    }
    config.surrogate = SurrogateConfig::ENN(enn);
}

fn apply_region(overrides: &ConfigOverrides, config: &mut OptimizerConfig) {
    if let Some(kind) = overrides.trust_region_kind {
        if kind == TrustRegionKind::Morbo {
            let num_metrics = overrides.num_metrics.unwrap_or(2);
            let alpha = overrides.alpha.unwrap_or(0.05);
            let length = TRLengthConfig {
                length_init: overrides.length_init.unwrap_or(0.8),
                length_min: overrides.length_min.unwrap_or(0.5f64.powi(7)),
                length_max: overrides.length_max.unwrap_or(1.6),
            };
            let rescalarize = overrides.rescalarize.unwrap_or(Rescalarize::OnPropose);
            config.trust_region = TrustRegionConfig::Morbo(MorboTRSettings {
                num_metrics,
                alpha,
                length,
                rescalarize,
                noise_aware: overrides.noise_aware.unwrap_or(false),
            });
            return;
        }
    }
    if overrides.length_init.is_none()
        && overrides.length_min.is_none()
        && overrides.length_max.is_none()
    {
        return;
    }
    let TRLengthConfig {
        length_init,
        length_min,
        length_max,
    } = match &config.trust_region {
        TrustRegionConfig::Turbo(cfg) => *cfg,
        TrustRegionConfig::Morbo(m) => m.length,
    };
    let updated = TRLengthConfig {
        length_init: overrides.length_init.unwrap_or(length_init),
        length_min: overrides.length_min.unwrap_or(length_min),
        length_max: overrides.length_max.unwrap_or(length_max),
    };
    config.trust_region = match &config.trust_region {
        TrustRegionConfig::Turbo(_) => TrustRegionConfig::Turbo(updated),
        TrustRegionConfig::Morbo(m) => {
            let mut morbo = m.clone();
            morbo.length = updated;
            TrustRegionConfig::Morbo(morbo)
        }
    };
}

impl ConfigOverrides {
    fn validate_study_selection(&self) -> Result<bool, String> {
        if !matches!(
            self.study,
            Some(TurboEnnStudy::EndToEnd | TurboEnnStudy::MoeLayer | TurboEnnStudy::Pretrain)
        ) {
            return Err("study must be 'end_to_end', 'moe_layer', or 'pretrain'".into());
        }
        if self.rounds() == 0 {
            return Err("rounds must be positive".into());
        }
        if self.reps() == 0 {
            return Err("reps must be positive".into());
        }
        if self.target_round_ms() == 0 {
            return Err("target_round_ms must be positive".into());
        }
        if self.output().as_os_str().is_empty() {
            return Err("output must be nonempty".into());
        }
        Ok(self.study == Some(TurboEnnStudy::Pretrain))
    }

    fn validate_study_fields(&self, pretrain: bool) -> Result<(), String> {
        if pretrain && self.model != Some(PretrainModel::FbtPisa1MoeV1) {
            return Err("pretrain model must be 'fbt_pisa1_moe_v1'".into());
        }
        if pretrain && self.corpus != Some(PretrainCorpus::StackV3PythonPilotV1) {
            return Err("pretrain corpus must be 'stack_v3_python_pilot_v1'".into());
        }
        if !pretrain && self.perturbation.is_some() {
            return Err("perturbation is supported only by pretrain studies".into());
        }
        if !pretrain && (self.distance_scaling.is_some() || self.local_scale_neighbors.is_some()) {
            return Err("distance scaling is supported only by pretrain studies".into());
        }
        if self.study == Some(TurboEnnStudy::MoeLayer) {
            if self.model.is_some() || self.corpus.is_some() {
                return Err("model and corpus presets are supported only by pretrain".into());
            }
            return Ok(());
        }
        if !pretrain && (self.dataset.is_some() || self.model.is_some() || self.corpus.is_some()) {
            return Err("dataset, model, and corpus are supported only by pretrain studies".into());
        }
        Ok(())
    }

    fn validate_seeds(&self) -> Result<(), String> {
        let seeds = [
            self.model_seed(),
            self.reference_seed(),
            self.proposal_seed(),
            self.acquisition_seed(),
        ];
        if seeds.iter().any(|&seed| seed > i64::MAX as u64) {
            return Err("seeds must fit TOML signed 64-bit integers".into());
        }
        for seed in [self.proposal_seed(), self.acquisition_seed()] {
            seed.checked_add(u64::from(self.rounds() - 1))
                .ok_or("round seed overflow")?;
        }
        Ok(())
    }

    fn validate_resident_policy(&self, pretrain: bool) -> Result<(), String> {
        if self.trust_region_kind == Some(TrustRegionKind::Morbo) {
            return Err("the LocalV1 FBT latency run requires the TuRBO trust region".into());
        }
        if !pretrain && self.trust_region_shape.is_some() {
            return Err("trust-region shape is supported only by pretrain studies".into());
        }
        match (self.trust_region_kind, self.reliability_controller) {
            (Some(TrustRegionKind::Reliability), Some(config)) if pretrain => {
                config.validate()?;
            }
            (Some(TrustRegionKind::Reliability), _) if !pretrain => {
                return Err(
                    "the reliability controller is supported only by pretrain studies".into(),
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
                    "the TuRBO-ENN round study supports only UCB or Thompson acquisition".into(),
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

    fn validate_fixed_controller_fields(&self) -> Result<(), String> {
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
            return Err("the TuRBO-ENN round study received optimizer fields that its fixed resident controller does not implement".into());
        }
        Ok(())
    }

    /// Apply overrides to an existing config.
    pub fn apply_to(&self, mut config: OptimizerConfig) -> OptimizerConfig {
        if let Some(acq) = self.acquisition {
            config.acquisition = acq;
        }
        if let Some(rv) = self.candidate_rv {
            config.candidates.candidate_rv = rv;
        }
        if let Some(f) = self.num_candidates_factor {
            config.candidates.num_candidates_factor = f;
        }
        if let Some(m) = self.min_candidates {
            config.candidates.min_candidates = m;
        }
        if let Some(cap) = self.max_candidates {
            config.candidates.max_candidates = Some(cap);
        }
        if let Some(m) = self.num_candidates_per_arm {
            config.candidates.num_candidates_per_arm = Some(m);
        }
        if let Some(n) = self.num_pert {
            config.candidates.num_pert = n.max(1);
        }
        apply_region(self, &mut config);
        if self.k_neighbors.is_some()
            || self.index_driver.is_some()
            || self.num_samples.is_some()
            || self.num_candidates.is_some()
            || self.scale_x.is_some()
            || self.enn_storage.is_some()
            || self.work_dir.is_some()
            || self.y_bounds.is_some()
        {
            apply_enn(&mut config, self);
        }
        if let Some(na) = self.noise_aware {
            config.noise_aware = na;
        }
        if let Some(d) = self.failure_tolerance_dim {
            config.failure_tolerance_dim = Some(d);
        }
        config
    }
}

/// Acquisition function configuration.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AcquisitionConfig {
    /// Upper Confidence Bound.
    #[serde(rename = "ucb")]
    UCB { beta: f64 },
    /// Thompson sampling.
    Thompson,
    /// Random acquisition.
    Random,
    /// Pareto front acquisition (multi-objective).
    Pareto,
}

#[cfg(test)]
mod serde_tests {
    use super::{
        AcquisitionConfig, ConfigOverrides, DistanceScaling, TRLengthConfig, TrustRegionKind,
        TrustRegionShape, TurboEnnStudy, parse_turbo_enn_config, turbo_enn,
    };
    use crate::{CandidateRV, Rescalarize};

    #[test]
    fn config_names() {
        let parsed: ConfigOverrides = toml::from_str(
            r#"
            acquisition = { ucb = { beta = 3.5 } }
            candidate_rv = "sobol"
            num_samples = 14
            trust_region_kind = "morbo"
            rescalarize = "on_propose"
            "#,
        )
        .unwrap();

        assert!(matches!(
            parsed.acquisition,
            Some(AcquisitionConfig::UCB { beta: 3.5 })
        ));
        assert_eq!(parsed.candidate_rv, Some(CandidateRV::Sobol));
        assert_eq!(parsed.num_samples, Some(14));
        assert_eq!(parsed.trust_region_kind, Some(TrustRegionKind::Morbo));
        assert_eq!(parsed.rescalarize, Some(Rescalarize::OnPropose));
    }

    #[test]
    fn config_fields() {
        let error = toml::from_str::<ConfigOverrides>("fit_samples = 10").unwrap_err();
        assert!(error.to_string().contains("unknown field"));
        let parsed = toml::from_str::<ConfigOverrides>("rounds = 3").unwrap();
        assert_eq!(parsed.rounds(), 3);
        assert_eq!(crate::TurboEnnStudy::EndToEnd, TurboEnnStudy::EndToEnd);
        assert_eq!(
            crate::prelude::TurboEnnStudy::EndToEnd,
            TurboEnnStudy::EndToEnd
        );
    }

    #[test]
    fn tune_uses_flat_turbo_enn_config() {
        let config = parse_turbo_enn_config(
            r#"
            version = 1
            study = "end_to_end"
            acquisition = "ucb"
            beta = 1.0
            trust_region_kind = "turbo"
            length_init = 0.01
            length_min = 0.0001
            length_max = 0.1
            output = "results/turbo-enn"
            rounds = 3
            target_round_ms = 1000
            "#,
        )
        .unwrap();
        assert_eq!(config.target_round_ms(), 1000);
        assert_eq!(config.length().length_init, 0.01);
        assert!(matches!(
            config.acquisition,
            Some(AcquisitionConfig::UCB { beta: 1.0 })
        ));
        assert!(!config.trace());

        let defaults = parse_turbo_enn_config("version=1\nstudy='end_to_end'\ntrace=true").unwrap();
        assert!(matches!(
            defaults.apply_to(turbo_enn()).acquisition,
            AcquisitionConfig::UCB { beta: 2.0 }
        ));
        assert_eq!(defaults.rounds(), 3);
        assert!(defaults.trace());

        let local = parse_turbo_enn_config(
            r#"
            version = 1
            study = "pretrain"
            model = "fbt_pisa1_moe_v1"
            corpus = "stack_v3_python_pilot_v1"
            distance_scaling = "self_tuning"
            local_scale_neighbors = 7
            "#,
        )
        .unwrap();
        let resident = local.resident_enn(11).unwrap();
        assert_eq!(resident.distance_scaling, DistanceScaling::SelfTuning);
        assert_eq!(resident.local_scale_neighbors, 7);
    }

    #[test]
    fn checked_in_turbo_enn_study_uses_the_shared_schema() {
        macro_rules! tuning_example {
            ($buck:literal, $cargo:literal) => {{
                #[cfg(feature = "buck2-test-data")]
                {
                    include_str!($buck)
                }
                #[cfg(not(feature = "buck2-test-data"))]
                {
                    include_str!($cargo)
                }
            }};
        }

        let config = parse_turbo_enn_config(tuning_example!(
            "../turbo-enn.toml/turbo-enn.toml",
            "../../../../examples/tuning/turbo-enn.toml"
        ))
        .unwrap();
        assert_eq!(config.rounds(), 3);
        assert_eq!(config.target_round_ms(), 1000);
        assert_eq!(
            config.output(),
            std::path::PathBuf::from("../../results/turbo-enn")
        );
        assert!(!config.trace());

        let trace_config = parse_turbo_enn_config(tuning_example!(
            "../turbo-enn.toml/turbo-enn-trace.toml",
            "../../../../examples/tuning/turbo-enn-trace.toml"
        ))
        .unwrap();
        assert_eq!(trace_config.rounds(), 1);
        assert!(trace_config.trace());
        assert_eq!(
            trace_config.output(),
            std::path::PathBuf::from("../../results/turbo-enn-trace")
        );

        let one_round_config = parse_turbo_enn_config(tuning_example!(
            "../turbo-enn.toml/turbo-enn-one-round.toml",
            "../../../../examples/tuning/turbo-enn-one-round.toml"
        ))
        .unwrap();
        assert_eq!(one_round_config.rounds(), 1);
        assert!(!one_round_config.trace());
        assert_eq!(
            one_round_config.output(),
            std::path::PathBuf::from("../../results/turbo-enn-one-round")
        );

        let pretrain_text = tuning_example!(
            "../turbo-enn.toml/code-pretrain.toml",
            "../../../../examples/tuning/code-pretrain.toml"
        );
        let pretrain = parse_turbo_enn_config(pretrain_text).unwrap();
        assert_eq!(pretrain.study, Some(TurboEnnStudy::Pretrain));
        assert!(pretrain.rounds() > 0);
        assert!(pretrain.output.is_none());
        assert!(pretrain.dataset().is_none());
        assert_eq!(pretrain.perturbation(), crate::Perturbation::Gaussian);
        assert_eq!(
            pretrain.trust_region_shape,
            Some(TrustRegionShape::TensorFamilyLearned)
        );
        let resident = pretrain.resident_enn(pretrain.acquisition_seed()).unwrap();
        assert_eq!(resident.ask.neighbors, 10);
        assert_eq!(resident.num_candidates, 30);
        assert_eq!(resident.num_samples, 10);
        assert_eq!(
            resident.ask.acquisition,
            crate::weights::AcquisitionKind::Ucb
        );
        assert_eq!(
            resident.ask.epistemic_scale,
            pretrain.epistemic_scale.unwrap() as f32
        );
        assert_eq!(resident.ask.aleatoric_scale, 0.05);
        assert_eq!(resident.ask.y_scale, 1.0);
        assert_eq!(resident.ask.seed, pretrain.acquisition_seed());
        assert_eq!(resident.distance_scaling, DistanceScaling::Global);
        assert_eq!(resident.local_scale_neighbors, 8);
        assert_eq!(pretrain.length(), TRLengthConfig::new(0.01, 0.0001, 0.1));
        assert_ne!(pretrain.proposal_seed(), pretrain.acquisition_seed());
        assert_eq!(pretrain.reps(), 1);
        let learned = parse_turbo_enn_config(tuning_example!(
            "../turbo-enn.toml/code-pretrain-family-learned.toml",
            "../../../../examples/tuning/code-pretrain-family-learned.toml"
        ))
        .unwrap();
        let learned_enn = learned.resident_enn(learned.acquisition_seed()).unwrap();
        assert!(learned_enn.fit_neighbors);
        assert_eq!(learned_enn.distance_scaling, DistanceScaling::SelfTuning);
        assert_eq!(learned_enn.local_scale_neighbors, 8);
        let reliability = parse_turbo_enn_config(tuning_example!(
            "../turbo-enn.toml/code-pretrain-ablation-local-reliability.toml",
            "../../../../examples/tuning/code-pretrain-ablation-local-reliability.toml"
        ))
        .unwrap();
        assert_eq!(
            reliability.trust_region_kind,
            Some(TrustRegionKind::Reliability)
        );
        assert_eq!(
            reliability
                .reliability_controller()
                .unwrap()
                .local_scale_neighbors,
            8
        );
        let legacy_text = pretrain_text
            .replace("[surrogate]", "[surrogate.resident-enn]")
            .replace("method = \"enn\"\n", "")
            .replace("fit_candidates = 30", "candidates = 30")
            .replace("fit_samples = 10", "samples = 10");
        let legacy = parse_turbo_enn_config(&legacy_text).unwrap();
        assert_eq!(
            serde_json::to_value(&pretrain).unwrap(),
            serde_json::to_value(&legacy).unwrap()
        );
        assert_eq!(pretrain.proposal_seed(), legacy.proposal_seed());
        assert_eq!(pretrain.acquisition_seed(), legacy.acquisition_seed());
        let baseline =
            parse_turbo_enn_config(&pretrain_text.replace("distribution = \"gaussian\"\n", ""))
                .unwrap();
        assert_eq!(baseline.perturbation(), crate::Perturbation::Gaussian);

        let self_tuning = parse_turbo_enn_config(&pretrain_text.replace(
            "method = \"enn\"\n",
            "method = \"enn\"\ndistance_scaling = \"self_tuning\"\nlocal_scale_neighbors = 7\n",
        ))
        .unwrap();
        let resident = self_tuning
            .resident_enn(self_tuning.acquisition_seed())
            .unwrap();
        assert_eq!(resident.distance_scaling, DistanceScaling::SelfTuning);
        assert_eq!(resident.local_scale_neighbors, 7);
    }

    #[test]
    fn tune_rejects_false_configuration() {
        let base = "version=1\nstudy='end_to_end'\nacquisition='thompson'\nlength_init=0.01\nlength_min=0.0001\nlength_max=0.1\noutput='x'\nrounds=3\ntarget_round_ms=1000";
        assert!(
            parse_turbo_enn_config("version=1")
                .unwrap_err()
                .contains("study must be 'end_to_end'")
        );
        assert!(
            parse_turbo_enn_config("version=1\nstudy='kernel_probe'")
                .unwrap_err()
                .contains("unknown variant")
        );
        assert!(
            parse_turbo_enn_config(&base.replace("rounds=3", "rounds=0"))
                .unwrap_err()
                .contains("rounds must be positive")
        );
        assert!(
            parse_turbo_enn_config(&base.replace(
                "target_round_ms=1000",
                "target_round_ms=1000\nunexpected=true"
            ))
            .unwrap_err()
            .contains("unknown field")
        );
        assert!(
            parse_turbo_enn_config(&base.replace("acquisition='thompson'", "acquisition='pareto'"))
                .unwrap_err()
                .contains("only UCB or Thompson")
        );
        assert!(
            parse_turbo_enn_config(
                &base.replace("length_init=0.01", "length_init=0.01\nnum_samples=0")
            )
            .unwrap_err()
            .contains("num_samples must be positive")
        );
        assert!(
            parse_turbo_enn_config("version=1\n[full_space_bo]\nrounds=3")
                .unwrap_err()
                .contains("unknown field")
        );
        let pretrain = "version=1\nstudy='pretrain'\nmodel='fbt_pisa1_moe_v1'\ncorpus='stack_v3_python_pilot_v1'";
        assert!(
            parse_turbo_enn_config(&format!("{pretrain}\nk_neighbors=0"))
                .unwrap_err()
                .contains("k_neighbors")
        );
        assert!(
            parse_turbo_enn_config(&format!("{pretrain}\nepistemic_scale=-1"))
                .unwrap_err()
                .contains("epistemic_scale")
        );
        assert!(
            parse_turbo_enn_config(&format!("{pretrain}\ny_scale=-1"))
                .unwrap_err()
                .contains("y_scale")
        );
        assert!(
            parse_turbo_enn_config(&format!("{pretrain}\nnum_candidates=0"))
                .unwrap_err()
                .contains("num_candidates must be positive")
        );
        assert!(
            parse_turbo_enn_config(&format!("{base}\nperturbation='rademacher'"))
                .unwrap_err()
                .contains("only by pretrain")
        );
        assert!(
            parse_turbo_enn_config(&format!(
                "{base}\ntrust_region_shape='tensor_family_static'"
            ))
            .unwrap_err()
            .contains("shape is supported only by pretrain")
        );
    }
}

impl Default for AcquisitionConfig {
    fn default() -> Self {
        AcquisitionConfig::UCB { beta: 2.0 }
    }
}

/// Initialization strategy type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InitStrategy {
    /// Latin Hypercube Design.
    #[default]
    LHD,
    /// Random uniform.
    Random,
}

/// Create a TuRBO-ENN configuration.
pub fn turbo_enn() -> OptimizerConfig {
    OptimizerConfig {
        surrogate: SurrogateConfig::ENN(ENNSurrogateConfig {
            k: 10,
            num_candidates: 30,
            num_samples: 10,
            ..Default::default()
        }),
        trust_region: TrustRegionConfig::default(),
        candidates: CandidateConfig {
            num_candidates_factor: 1000.0,
            min_candidates: 100,
            max_candidates: None,
            num_candidates_per_arm: None,
            num_pert: 20,
            candidate_rv: CandidateRV::Uniform,
        },
        acquisition: AcquisitionConfig::UCB { beta: 2.0 },
        noise_aware: false,
        failure_tolerance_dim: None,
    }
}

/// Parse one `tune` document using the existing flat TuRBO-ENN field schema.
pub fn parse_turbo_enn_config(text: &str) -> Result<ConfigOverrides, String> {
    let document: toml::Value = toml::from_str(text).map_err(|error| error.to_string())?;
    let mut root = document
        .as_table()
        .cloned()
        .ok_or("tune config must be a TOML table")?;
    if root.remove("version").and_then(|value| value.as_integer()) != Some(1) {
        return Err("version must be 1".into());
    }
    lower_structured_tune(&mut root)?;
    let beta = root.remove("beta");
    match root.get("acquisition").and_then(toml::Value::as_str) {
        Some("ucb") => {
            let beta = match beta {
                Some(toml::Value::Float(value)) => value,
                Some(toml::Value::Integer(value)) => value as f64,
                Some(_) => return Err("beta must be a number".into()),
                None => 2.0,
            };
            root.insert(
                "acquisition".into(),
                toml::Value::Table(toml::Table::from_iter([(
                    "ucb".into(),
                    toml::Value::Table(toml::Table::from_iter([(
                        "beta".into(),
                        toml::Value::Float(beta),
                    )])),
                )])),
            );
        }
        _ if beta.is_some() => return Err("beta is valid only with acquisition = 'ucb'".into()),
        _ => {}
    }
    let config: ConfigOverrides = toml::Value::Table(root)
        .try_into()
        .map_err(|error| format!("invalid TuRBO-ENN fields: {error}"))?;
    config.validate_round_study()?;
    Ok(config)
}

pub fn load_turbo_enn_config(path: &std::path::Path) -> Result<(ConfigOverrides, String), String> {
    let result = || {
        let text = std::fs::read_to_string(path).map_err(|error| error.to_string())?;
        let mut config = parse_turbo_enn_config(&text)?;
        let absolute = std::fs::canonicalize(path).map_err(|error| error.to_string())?;
        let parent = absolute
            .parent()
            .ok_or("configuration path has no parent")?;
        let output = parent.join(config.output());
        config.output = Some(output.clone());
        if let Some(dataset) = config.dataset.as_ref() {
            let dataset = if dataset.is_absolute() {
                dataset.clone()
            } else {
                parent.join(dataset)
            };
            let dataset = std::fs::canonicalize(&dataset)
                .map_err(|error| format!("dataset path {}: {error}", dataset.display()))?;
            config.dataset = Some(dataset);
        }
        let mut resolved: toml::Value = toml::from_str(&text).map_err(|error| error.to_string())?;
        let table = resolved
            .as_table_mut()
            .ok_or("tune config must be a TOML table")?;
        table.insert(
            "output".into(),
            toml::Value::String(output.to_string_lossy().into_owned()),
        );
        if let Some(dataset) = config.dataset.as_ref() {
            table.insert(
                "dataset".into(),
                toml::Value::String(dataset.to_string_lossy().into_owned()),
            );
        }
        let resolved = toml::to_string_pretty(&resolved).map_err(|error| error.to_string())?;
        Ok((config, resolved))
    };
    result().map_err(|error: String| format!("TuRBO-ENN config {}: {error}", path.display()))
}

fn validate_lengths(length: TRLengthConfig) -> Result<(), String> {
    for value in [length.length_init, length.length_min, length.length_max] {
        if !value.is_finite() || !(value as f32).is_finite() || value as f32 <= 0.0 {
            return Err("trust-region lengths must be positive finite FP32 values".into());
        }
    }
    if length.length_min >= length.length_max
        || length.length_init < length.length_min
        || length.length_init > length.length_max
    {
        return Err("trust-region lengths must satisfy min <= init <= max and min < max".into());
    }
    Ok(())
}

impl ConfigOverrides {
    pub fn validate_round_study(&self) -> Result<(), String> {
        let pretrain = self.validate_study_selection()?;
        self.validate_study_fields(pretrain)?;
        self.validate_seeds()?;
        self.validate_resident_policy(pretrain)?;
        self.validate_fixed_controller_fields()
    }

    pub fn length(&self) -> TRLengthConfig {
        let effective = self.apply_to(turbo_enn());
        let TrustRegionConfig::Turbo(length) = effective.trust_region else {
            unreachable!("validated TuRBO-ENN configuration")
        };
        length
    }

    pub fn reliability_controller(&self) -> Option<crate::ReliabilityControllerConfig> {
        (self.trust_region_kind == Some(TrustRegionKind::Reliability))
            .then_some(self.reliability_controller)
            .flatten()
    }

    pub fn output(&self) -> PathBuf {
        self.output
            .clone()
            .unwrap_or_else(|| PathBuf::from("results/turbo-enn"))
    }

    pub fn rounds(&self) -> u32 {
        self.rounds.unwrap_or(3)
    }

    pub fn reps(&self) -> u32 {
        self.reps.unwrap_or(1)
    }

    pub fn target_round_ms(&self) -> u32 {
        self.target_round_ms.unwrap_or(1000)
    }

    pub fn trace(&self) -> bool {
        self.trace.unwrap_or(false)
    }

    pub fn model_seed(&self) -> u64 {
        self.model_seed
            .unwrap_or_else(|| self.derived_seed(0, "model", 0))
    }

    pub fn reference_seed(&self) -> u64 {
        self.reference_seed
            .unwrap_or_else(|| self.derived_seed(0, "reference", 0))
    }

    pub fn proposal_seed(&self) -> u64 {
        self.proposal_seed
            .unwrap_or_else(|| self.derived_seed(0, "proposal", 0))
    }

    pub fn acquisition_seed(&self) -> u64 {
        self.acquisition_seed
            .unwrap_or_else(|| self.derived_seed(0, "acquisition", 0))
    }

    pub(crate) fn derived_seed(&self, rep: u32, domain: &str, index: u64) -> u64 {
        let mut state = 0xcbf2_9ce4_8422_2325u64;
        let mut parts = vec![
            b"ennx-tune-v1".to_vec(),
            domain.as_bytes().to_vec(),
            rep.to_le_bytes().to_vec(),
            index.to_le_bytes().to_vec(),
            format!("{:?}", self.study).into_bytes(),
            format!("{:?}", self.model).into_bytes(),
            format!("{:?}", self.corpus).into_bytes(),
            format!("{:?}", self.perturbation()).into_bytes(),
            format!("{:?}", self.acquisition).into_bytes(),
            self.rounds().to_string().into_bytes(),
            self.reps().to_string().into_bytes(),
            self.target_round_ms().to_string().into_bytes(),
        ];
        parts.sort();
        for bytes in parts {
            for byte in bytes {
                state ^= u64::from(byte);
                state = state.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        crate::hash::splitmix64(state) & 0x3fff_ffff_ffff_ffff
    }

    pub fn dataset(&self) -> Option<&std::path::Path> {
        self.dataset.as_deref()
    }

    pub fn perturbation(&self) -> crate::Perturbation {
        self.perturbation.unwrap_or_default()
    }

    /// Resolve the shared ENN/acquisition parameters consumed by Metal, OpenCL,
    /// and CUDA resident search kernels.
    pub fn resident_enn(&self, seed: u64) -> Result<ResidentEnnConfig, String> {
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
        let distance_scaling = self.distance_scaling.unwrap_or_default();
        let local_scale_neighbors = self.local_scale_neighbors.unwrap_or(8);
        if local_scale_neighbors == 0 || local_scale_neighbors > 128 {
            return Err("resident ENN local_scale_neighbors must be in 1..=128".into());
        }
        if distance_scaling == DistanceScaling::Global && self.local_scale_neighbors.is_some() {
            return Err("local_scale_neighbors requires distance_scaling = 'self_tuning'".into());
        }
        Ok(ResidentEnnConfig {
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
            local_scale_neighbors,
        })
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

/// Create a TuRBO-ZERO configuration.
pub fn turbo_zero() -> OptimizerConfig {
    OptimizerConfig {
        surrogate: SurrogateConfig::None,
        trust_region: TrustRegionConfig::default(),
        candidates: CandidateConfig {
            num_candidates_factor: 1000.0,
            min_candidates: 100,
            max_candidates: None,
            num_candidates_per_arm: None,
            num_pert: 20,
            candidate_rv: CandidateRV::Uniform,
        },
        acquisition: AcquisitionConfig::Random,
        noise_aware: false,
        failure_tolerance_dim: None,
    }
}

/// Create an LHD-only configuration.
pub fn lhd_only() -> OptimizerConfig {
    OptimizerConfig {
        surrogate: SurrogateConfig::None,
        trust_region: TrustRegionConfig::default(),
        candidates: CandidateConfig {
            num_candidates_factor: 1.0,
            min_candidates: 1,
            max_candidates: None,
            num_candidates_per_arm: None,
            num_pert: 20,
            candidate_rv: CandidateRV::Uniform,
        },
        acquisition: AcquisitionConfig::Random,
        noise_aware: false,
        failure_tolerance_dim: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::EnnStorage;
    use crate::candidates::CandidateRV;
    use std::path::Path;

    #[test]
    fn test_003() {
        let config = CandidateConfig::default();

        // Basic case: 2D, 1 arm
        let n = config.num_candidates(2, 1);
        assert!(n >= 100); // At least min_candidates

        // Larger dimension
        let n_large = config.num_candidates(10, 1);
        assert!(n_large >= 1000);

        // More arms
        let n_arms = config.num_candidates(2, 10);
        assert!(n_arms >= 100); // 10 * 10 = 100
    }

    #[test]
    fn test_004() {
        // Python default: min(5000, 100*num_dim). Cap at 5000 for high dim.
        let config = CandidateConfig {
            num_candidates_factor: 100.0,
            min_candidates: 100,
            max_candidates: Some(5000),
            num_candidates_per_arm: None,
            num_pert: 20,
            candidate_rv: CandidateRV::Uniform,
        };
        assert_eq!(config.num_candidates(60, 1), 5000);
        assert_eq!(config.num_candidates(100, 1), 5000);
        assert_eq!(config.num_candidates(10, 1), 1000);
    }

    #[test]
    fn test_configdefaults() {
        let config = OptimizerConfig::default();
        assert!(matches!(config.acquisition, AcquisitionConfig::UCB { .. }));
    }

    #[test]
    fn test_turboennconfig() {
        let config = turbo_enn();
        assert!(matches!(config.surrogate, SurrogateConfig::ENN(_)));
        assert!(matches!(config.acquisition, AcquisitionConfig::UCB { .. }));
    }

    #[test]
    fn test_turbozeroconfig() {
        let config = turbo_zero();
        assert!(matches!(config.surrogate, SurrogateConfig::None));
        assert!(matches!(config.acquisition, AcquisitionConfig::Random));
    }

    #[test]
    fn test_lhdonlyconfig() {
        let config = lhd_only();
        assert!(matches!(config.surrogate, SurrogateConfig::None));
        let n = config.candidates.num_candidates(10, 1);
        assert_eq!(n, 10);
    }

    #[test]
    fn test_initstrategyenum() {
        let init_default = InitStrategy::default();
        assert_eq!(init_default, InitStrategy::LHD);
        assert_eq!(InitStrategy::Random as u8, 1);
    }

    #[test]
    fn test_010() {
        use crate::index::IndexDriver;

        let overrides = ConfigOverrides {
            acquisition: Some(AcquisitionConfig::Thompson),
            candidate_rv: Some(CandidateRV::Sobol),
            index_driver: Some(IndexDriver::Exact),
            num_samples: Some(123),
            num_candidates: Some(456),
            scale_x: Some(true),
            ..Default::default()
        };

        let config = turbo_enn();
        let applied = overrides.apply_to(config);

        assert!(matches!(applied.acquisition, AcquisitionConfig::Thompson));
        assert_eq!(applied.candidates.candidate_rv, CandidateRV::Sobol);
        if let SurrogateConfig::ENN(enn) = &applied.surrogate {
            assert_eq!(enn.index_driver, IndexDriver::Exact);
            assert_eq!(enn.num_samples, 123);
            assert_eq!(enn.num_candidates, 456);
            assert!(enn.scale_x);
        } else {
            panic!("expected ENN surrogate");
        }
    }

    #[test]
    fn test_011() {
        let overrides = ConfigOverrides {
            scale_x: Some(true),
            ..Default::default()
        };
        let applied = overrides.apply_to(turbo_enn());
        let SurrogateConfig::ENN(enn) = applied.surrogate else {
            panic!("expected ENN surrogate");
        };
        assert!(enn.scale_x);
    }

    #[test]
    fn morbo_metrics() {
        use crate::mbtrregn::MorboTrustRegion;
        use crate::trregncfg::TrustRegionConfig;
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        let overrides = ConfigOverrides {
            trust_region_kind: Some(TrustRegionKind::Morbo),
            num_metrics: Some(1),
            ..Default::default()
        };
        let applied = overrides.apply_to(turbo_enn());
        let TrustRegionConfig::Morbo(settings) = applied.trust_region else {
            panic!("expected Morbo trust region");
        };
        let mut rng = StdRng::seed_from_u64(8);
        let result = MorboTrustRegion::new(2, settings, &mut rng);
        assert!(
            result.is_err(),
            "PyO3/override path must reject num_metrics=1 like Python Morbo config"
        );
    }

    #[test]
    fn candidate_arms() {
        let cfg = CandidateConfig {
            num_candidates_factor: 1.0,
            min_candidates: 10,
            max_candidates: None,
            num_candidates_per_arm: Some(25),
            num_pert: 20,
            candidate_rv: CandidateRV::Uniform,
        };
        assert_eq!(cfg.num_candidates(2, 3), 75);
        assert_eq!(cfg.num_candidates(2, 8), 200);
    }

    #[test]
    fn config_pool() {
        let overrides = ConfigOverrides {
            num_candidates_factor: Some(1.0),
            min_candidates: Some(10),
            num_candidates_per_arm: Some(40),
            ..Default::default()
        };
        let applied = overrides.apply_to(turbo_zero());
        assert_eq!(applied.candidates.num_candidates(2, 3), 120);
        assert_eq!(applied.candidates.num_candidates(2, 8), 320);
    }

    #[test]
    fn enn_fields() {
        let overrides = ConfigOverrides {
            num_samples: Some(7),
            num_candidates: Some(11),
            scale_x: Some(true),
            ..Default::default()
        };
        let applied = overrides.apply_to(turbo_enn());
        let SurrogateConfig::ENN(enn) = applied.surrogate else {
            panic!("expected ENN surrogate");
        };
        assert_eq!(enn.num_samples, 7);
        assert_eq!(enn.num_candidates, 11);
        assert!(enn.scale_x);
    }

    #[test]
    fn enn_dir() {
        use crate::index::IndexDriver;
        use std::path::PathBuf;

        let overrides = ConfigOverrides {
            index_driver: Some(IndexDriver::BpAnnDisk),
            enn_storage: Some(EnnStorage::Disk),
            work_dir: Some(PathBuf::from("/tmp/enn_work")),
            ..Default::default()
        };
        let applied = overrides.apply_to(turbo_enn());
        let SurrogateConfig::ENN(enn) = applied.surrogate else {
            panic!("expected ENN surrogate");
        };
        assert_eq!(enn.index_driver, IndexDriver::BpAnnDisk);
        assert_eq!(enn.storage, EnnStorage::Disk);
        assert_eq!(enn.work_dir.as_deref(), Some(Path::new("/tmp/enn_work")));
    }

    #[test]
    fn morbo_propose() {
        let overrides = ConfigOverrides {
            trust_region_kind: Some(TrustRegionKind::Morbo),
            num_metrics: Some(2),
            ..Default::default()
        };
        let applied = overrides.apply_to(turbo_enn());
        let TrustRegionConfig::Morbo(settings) = applied.trust_region else {
            panic!("expected Morbo trust region");
        };
        assert_eq!(
            settings.rescalarize,
            Rescalarize::OnPropose,
            "missing rescalarize should match Python MorboTRConfig default ON_PROPOSE"
        );
    }
}
