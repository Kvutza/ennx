//! Configuration types for the optimizer.

use crate::backend::EnnStorage;
use crate::candidates::CandidateRV;
use crate::index::IndexDriver;
use crate::mbtrregn::{MorboTRSettings, Rescalarize};
use crate::surrogate::ENNSurrogateConfig;
use crate::trregncfg::TrustRegionConfig;
use crate::trust_region::TRLengthConfig;
use deser::{Deserialize, Serialize};
use std::path::PathBuf;

mod diffusion;
mod experiment;
mod feedback;
mod generation;
pub use diffusion::{DiffusionConfig, IndexMode};
mod initialization;
mod kernel;
mod objectives;
mod resident;
mod seeds;
pub use resident::ResidentEnnConfig;
mod procedural;
mod spec;
#[path = "config/spec/conversion.rs"]
mod spec_conversion;
#[path = "config/spec/execution.rs"]
mod spec_execution;
#[path = "config/spec/optimizer.rs"]
mod spec_optimizer;
mod validation;
pub use feedback::FeedbackTransition;
pub use generation::{
    FrozenAttention, FrozenBackend, FrozenReadout, GenerationConfig, GenerationPurpose,
    GenerationReward, GenerationTask, SignalGate, VerifyConfig, VerifyMode,
};
pub use initialization::ModelInitialization;
pub use kernel::KernelTrial;
pub use objectives::{ResidentObjectiveConfig, ResidentObjectiveMode};
pub use spec::TuneSpec;
pub use spec_execution::{DataSpec, DiagnosticSpec, ObjectiveSpec, RunSpec, SeedSpec};
pub use spec_optimizer::{
    AcquisitionSpec, EnnSpec, FitSpec, ProposalSpec, TrustRegionBounds, TrustRegionSpec,
};
#[cfg(test)]
#[path = "config/spec/tests.rs"]
mod spec_tests;
pub use experiment::{
    DistanceScaling, HistoryGeometry, ObjectiveReference, PretrainSelection, TrustRegionShape,
};

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
#[deser(default, deny_unknown_fields)]
pub struct ConfigOverrides {
    /// Explicit independent objective acquisition; never inferred from scalar rewards.
    pub objective_acquisition: Option<ResidentObjectiveConfig>,
    pub generation: Option<GenerationConfig>,
    /// Logical procedural proposal layout. Current resident kernels accept the
    /// legacy one-arm/four-slot layout only; larger requests fail validation.
    pub proposal_pool: Option<crate::procedural_pool::ProceduralPoolConfig>,
    /// Program used to turn procedural random bits into a full-coordinate direction.
    pub proposal_method: Option<crate::procedural_pool::ProposalMethod>,
    /// File-driven experiment to run. Programmatic optimizer overrides may omit it.
    pub experiment: Option<TurboEnnExperiment>,
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
    /// Coordinate system used for resident ENN history distances.
    pub history_geometry: Option<HistoryGeometry>,
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
    /// Deterministic repetitions of the same experiment.
    pub reps: Option<u32>,
    /// Required maximum latency for every measured round.
    pub target_round_ms: Option<u32>,
    /// Emit perturbative per-operation timing records for the fixed round experiment.
    pub trace: Option<bool>,
    /// Opt-in shader replacement and full-loop numerical capture for kernel search.
    pub kernel_trial: Option<KernelTrial>,
    /// Timestamp complete scorer stages; also profiles the selected gate experiment.
    pub scorer_stage_samples: Option<u32>,
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
    /// Immutable held-out stream, never supplied to optimizer acceptance.
    pub validation_dataset: Option<PathBuf>,
    pub validation_interval: Option<u32>,
    /// Candidate-selection ablation; the remaining ENN policy stays unchanged.
    pub selection: Option<PretrainSelection>,
    /// Versioned model architecture preset for a pretraining experiment.
    pub model: Option<PretrainModel>,
    /// Versioned immutable corpus recipe for a pretraining experiment.
    pub corpus: Option<PretrainCorpus>,
    /// Full-coordinate perturbation distribution generated by the accelerator.
    pub perturbation: Option<crate::Perturbation>,
    /// Trust-region shape policy for the full-weight pretrain experiment.
    pub trust_region_shape: Option<TrustRegionShape>,
    /// Per-minibatch control variate used by the pretraining objective.
    pub objective_reference: Option<ObjectiveReference>,
    /// Reliability-aware controller parameters for full-weight pretraining.
    pub reliability_controller: Option<crate::ReliabilityPolicy>,
}

/// Complete workload selected by a file-driven TuRBO-ENN configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[deser(rename_all = "kebab-case")]
pub enum TurboEnnExperiment {
    EndToEnd,
    MoeLayer,
    Pretrain,
    Generation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[deser(rename_all = "kebab-case")]
pub enum PretrainModel {
    #[deser(rename = "fbt-pisa1-legacy-v1")]
    FbtPisa1MoeV1,
    FbtPisa1Residual1V1,
    FbtPisa1ProjectedBoundaryV1,
    FbtPisa1Hc4V1,
    FbtPisa1Mhc4V1,
    FbtPisa1LoopedMhc4V1,
    FbtPisa1DiffusionMhc4V1,
}

impl PretrainModel {
    pub const ALL: [Self; 7] = [
        Self::FbtPisa1MoeV1,
        Self::FbtPisa1Residual1V1,
        Self::FbtPisa1ProjectedBoundaryV1,
        Self::FbtPisa1Hc4V1,
        Self::FbtPisa1Mhc4V1,
        Self::FbtPisa1LoopedMhc4V1,
        Self::FbtPisa1DiffusionMhc4V1,
    ];

    pub const fn id(self) -> &'static str {
        match self {
            Self::FbtPisa1MoeV1 => "fbt-pisa1-legacy-v1",
            Self::FbtPisa1Residual1V1 => "fbt-pisa1-residual1-v1",
            Self::FbtPisa1ProjectedBoundaryV1 => "fbt-pisa1-projected-boundary-v1",
            Self::FbtPisa1Hc4V1 => "fbt-pisa1-hc4-v1",
            Self::FbtPisa1Mhc4V1 => "fbt-pisa1-mhc4-v1",
            Self::FbtPisa1LoopedMhc4V1 => "fbt-pisa1-looped-mhc4-v1",
            Self::FbtPisa1DiffusionMhc4V1 => "fbt-pisa1-diffusion-mhc4-v1",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[deser(rename_all = "kebab-case")]
pub enum PretrainCorpus {
    StackV3PythonPilotV1,
    #[deser(rename = "stack-v3-python-800k-v1")]
    StackV3Python800kV1,
    #[deser(rename = "fineweb-10bt-pilot-v1")]
    Fineweb10btPilotV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[deser(rename_all = "snake_case")]
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
#[deser(rename_all = "snake_case")]
pub enum AcquisitionConfig {
    /// Upper Confidence Bound.
    #[deser(rename = "ucb")]
    UCB { beta: f64 },
    /// Thompson sampling.
    Thompson,
    /// Random acquisition.
    Random,
    /// Pareto front acquisition (multi-objective).
    Pareto,
}

#[cfg(test)]
mod serialization_tests {
    use super::TuneSpec;
    use super::{
        AcquisitionConfig, ConfigOverrides, DistanceScaling, GenerationPurpose, TRLengthConfig,
        TrustRegionKind, TrustRegionShape, TurboEnnExperiment, parse_tune, turbo_enn,
    };
    use crate::{CandidateRV, Rescalarize};

    #[test]
    fn config_names() {
        let parsed: ConfigOverrides = ennx_wire::toml::from_str(
            r#"
            acquisition = { ucb = { beta = 3.5 } }
            candidate_rv = "sobol"
            num_samples = 14
            trust_region_kind = "morbo"
            rescalarize = "on-propose"
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
        let error = ennx_wire::toml::from_str::<ConfigOverrides>("fit_samples = 10").unwrap_err();
        assert!(error.to_string().contains("unknown field"));
        let parsed = ennx_wire::toml::from_str::<ConfigOverrides>("rounds = 3").unwrap();
        assert_eq!(parsed.rounds(), 3);
        assert_eq!(
            crate::TurboEnnExperiment::EndToEnd,
            TurboEnnExperiment::EndToEnd
        );
        assert_eq!(
            crate::prelude::TurboEnnExperiment::EndToEnd,
            TurboEnnExperiment::EndToEnd
        );
    }

    #[test]
    fn fineweb_seeds() {
        let input = "version=2\nexperiment='pretrain'\nmodel='fbt-pisa1-legacy-v1'\ncorpus='fineweb-10bt-pilot-v1'\n[run]\nselection='enn'\nvalidation-interval=4";
        let enn = parse_tune(input).unwrap();
        let random = parse_tune(&input.replace("selection='enn'", "selection='random'")).unwrap();
        assert_eq!(enn.model_seed(), random.model_seed());
        assert_eq!(enn.proposal_seed(), random.proposal_seed());
        assert_eq!(enn.acquisition_seed(), random.acquisition_seed());
        assert_ne!(enn.model_seeded(0), enn.model_seeded(1));
        assert!(
            parse_tune(&input.replace("validation-interval=4", "validation-interval=0")).is_err()
        );
    }

    #[test]
    fn generated_seeds() {
        #[cfg(feature = "buck2-test-data")]
        let (enn_text, random_text) = (
            include_str!("../turbo-enn.toml/code-generation-enn.toml"),
            include_str!("../turbo-enn.toml/code-generation-random.toml"),
        );
        #[cfg(not(feature = "buck2-test-data"))]
        let (enn_text, random_text) = (
            include_str!("../../../../examples/tuning/code-generation-enn.toml"),
            include_str!("../../../../examples/tuning/code-generation-random.toml"),
        );
        let enn = parse_tune(enn_text).unwrap();
        let random = parse_tune(random_text).unwrap();
        enn.validate_experiment().unwrap();
        random.validate_experiment().unwrap();
        assert_eq!(enn.reps(), 3);
        assert_eq!(enn.rounds(), 512);
        assert_eq!(enn.generation.as_ref().unwrap().max_tokens, 4096);
        assert!(enn.generation.as_ref().unwrap().seed.is_none());
        for rep in 0..enn.reps() {
            assert_eq!(enn.model_seeded(rep), random.model_seeded(rep));
            assert_eq!(enn.proposal_seeded(rep), random.proposal_seeded(rep));
            assert_eq!(enn.acquisition_seeded(rep), random.acquisition_seeded(rep));
            assert_eq!(enn.sample_seeded(rep), random.sample_seeded(rep));
        }
        assert_ne!(enn.selection, random.selection);
    }

    #[test]
    fn shared_schema() {
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

        let config = parse_tune(tuning_example!(
            "../turbo-enn.toml/turbo-enn.toml",
            "../../../../examples/tuning/turbo-enn.toml"
        ))
        .unwrap();
        assert_eq!(config.rounds(), 3);
        assert_eq!(config.target_ms(), 1000);
        assert_eq!(
            config.output(),
            std::path::PathBuf::from("../../results/turbo-enn")
        );
        assert!(!config.trace());

        let trace_config = parse_tune(tuning_example!(
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

        let one_round_config = parse_tune(tuning_example!(
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
        let pretrain = parse_tune(pretrain_text).unwrap();
        assert_eq!(pretrain.experiment, Some(TurboEnnExperiment::Pretrain));
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
        let generated = parse_tune(tuning_example!(
            "../turbo-enn.toml/code-pretrain-generated.toml",
            "../../../../examples/tuning/code-pretrain-generated.toml"
        ))
        .unwrap();
        assert_eq!(generated.experiment, Some(TurboEnnExperiment::Pretrain));
        let generation = generated.generation.as_ref().unwrap();
        assert_eq!(generation.purpose, GenerationPurpose::SystemsProbe);
        assert_eq!(generation.max_tokens, 4096);
        assert!(generation.corpus_prompt.is_empty());
        assert_eq!(generation.corpus_prompt_tokens, Some(128));
        assert!(generation.tasks.is_empty());
        assert_eq!(
            generated
                .resident_enn(generated.acquisition_seed())
                .unwrap()
                .history_geometry,
            crate::config::HistoryGeometry::Latent
        );
        assert!(
            parse_tune(
                &tuning_example!(
                    "../turbo-enn.toml/code-pretrain-generated.toml",
                    "../../../../examples/tuning/code-pretrain-generated.toml"
                )
                .replace("systems-probe", "coding-optimization")
            )
            .unwrap_err()
            .contains("requires checkpoint and qualification_manifest")
        );
        let learned = parse_tune(tuning_example!(
            "../turbo-enn.toml/code-pretrain-family-learned.toml",
            "../../../../examples/tuning/code-pretrain-family-learned.toml"
        ))
        .unwrap();
        let learned_enn = learned.resident_enn(learned.acquisition_seed()).unwrap();
        assert!(learned_enn.fit_neighbors);
        assert_eq!(learned_enn.distance_scaling, DistanceScaling::SelfTuning);
        assert_eq!(learned_enn.local_scale_neighbors, 8);
        let reliability = parse_tune(tuning_example!(
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
        let repeated = parse_tune(&pretrain_text.replace("[run]\n", "[run]\nreps = 3\n")).unwrap();
        assert_eq!(repeated.reps(), 3);
        assert_ne!(repeated.proposal_seeded(0), repeated.proposal_seeded(1));
        assert_ne!(
            repeated.acquisition_seeded(1),
            repeated.acquisition_seeded(2)
        );
        assert_eq!(repeated.proposal_seeded(0), pretrain.proposal_seeded(0));
        let rademacher = parse_tune(&pretrain_text.replace("gaussian", "rademacher")).unwrap();
        assert_eq!(rademacher.proposal_seeded(0), pretrain.proposal_seeded(0));
        assert_eq!(
            rademacher.acquisition_seeded(0),
            pretrain.acquisition_seeded(0)
        );
        assert_eq!(
            repeated
                .resident_enn(repeated.acquisition_seeded(2))
                .unwrap()
                .ask
                .seed,
            repeated.acquisition_seeded(2)
        );
        let explicit_repeated = ConfigOverrides {
            reps: Some(2),
            proposal_seed: Some(7),
            ..Default::default()
        };
        assert_eq!(
            explicit_repeated.validate_seeds().unwrap_err(),
            "reps greater than one require internally derived seeds"
        );
        let baseline =
            parse_tune(&pretrain_text.replace("distribution = \"gaussian\"\n", "")).unwrap();
        assert_eq!(baseline.perturbation(), crate::Perturbation::Gaussian);

        let self_tuning = parse_tune(&pretrain_text.replace(
            "[enn]\n",
            "[enn]\nscaling = \"self-tuning\"\nlocal-neighbors = 7\n",
        ))
        .unwrap();
        let resident = self_tuning
            .resident_enn(self_tuning.acquisition_seed())
            .unwrap();
        assert_eq!(resident.distance_scaling, DistanceScaling::SelfTuning);
        assert_eq!(resident.local_scale_neighbors, 7);
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

/// Parse one closed, typed version 2 experiment document.
pub fn parse_tune(text: &str) -> Result<ConfigOverrides, String> {
    TuneSpec::parse(text)?.overrides()
}

pub fn load_tune(path: &std::path::Path) -> Result<(ConfigOverrides, String), String> {
    let result = || {
        let text = std::fs::read_to_string(path).map_err(|error| error.to_string())?;
        let mut config = parse_tune(&text)?;
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
        if let Some(dataset) = config.validation_dataset.as_ref() {
            config.validation_dataset = Some(std::fs::canonicalize(parent.join(dataset)).map_err(
                |error| format!("held-out dataset path {}: {error}", dataset.display()),
            )?);
        }
        if let Some(trial) = &mut config.kernel_trial {
            trial.resolve(parent)?;
        }
        if let Some(generation) = &mut config.generation {
            generation.resolve(parent)?;
        }
        let resolved = TuneSpec::from_overrides(&config)?.to_toml()?;
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
    pub fn validate_experiment(&self) -> Result<(), String> {
        let pretrain = self.validate_selection()?;
        if self.experiment == Some(TurboEnnExperiment::Generation) {
            let generation = self
                .generation
                .as_ref()
                .ok_or("generation experiment requires [generation]")?;
            generation.validate()?;
            if self.reps() != 1
                || self.corpus.is_some()
                || self.dataset.is_some()
                || self.objective_reference.is_some()
                || self.kernel_trial.is_some()
                || self.trace == Some(true)
            {
                return Err("generation uses explicit tasks, one repetition, and no pretrain objective or diagnostics".into());
            }
            if !generation.corpus_prompt.is_empty() || generation.corpus_prompt_tokens.is_some() {
                return Err("generation experiment requires explicit tasks; corpus_prompt is for generated pretraining".into());
            }
            let fields = ennx_wire::json::to_value(self).map_err(|error| error.to_string())?;
            if fields.as_map().unwrap().iter().any(|(key, value)| {
                !value.is_null()
                    && (key.as_str().is_some_and(|key| key.ends_with("_pairs"))
                        || key == "scorer_stage_samples")
            }) {
                return Err("generation cannot contain scorer diagnostics".into());
            }
        } else if let Some(generation) = &self.generation {
            if self.experiment != Some(TurboEnnExperiment::Pretrain) {
                return Err("[generation] requires a generation or pretrain experiment".into());
            }
            generation.validate()?;
            if (generation.corpus_prompt.is_empty() && generation.corpus_prompt_tokens.is_none())
                || !generation.tasks.is_empty()
                || self.corpus.is_none()
            {
                return Err("generated pretraining requires corpus, corpus_prompt, and no explicit generation tasks".into());
            }
            if self.kernel_trial.is_some() || self.trace == Some(true) {
                return Err("generated pretraining cannot include kernel diagnostics".into());
            }
        }
        self.validate_fields(pretrain)?;
        self.validate_seeds()?;
        self.validate_resident(pretrain)?;
        self.validate_controller()
    }

    pub fn length(&self) -> TRLengthConfig {
        let effective = self.apply_to(turbo_enn());
        let TrustRegionConfig::Turbo(length) = effective.trust_region else {
            unreachable!("validated TuRBO-ENN configuration")
        };
        length
    }

    pub fn reliability_controller(&self) -> Option<crate::ReliabilityPolicy> {
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

    pub fn target_ms(&self) -> u32 {
        self.target_round_ms.unwrap_or(1000)
    }

    pub fn trace(&self) -> bool {
        self.trace.unwrap_or(false)
    }

    pub fn objective_reference(&self) -> ObjectiveReference {
        self.objective_reference.unwrap_or_default()
    }

    pub fn dataset(&self) -> Option<&std::path::Path> {
        self.dataset.as_deref()
    }

    pub fn perturbation(&self) -> crate::Perturbation {
        self.perturbation.unwrap_or_default()
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
