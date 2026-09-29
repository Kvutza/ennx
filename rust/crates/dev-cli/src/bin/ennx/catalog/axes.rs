//! Categorical substitutions into existing typed policies, never a second parser.
use ennx::config::{
    AcquisitionSpec, DistanceScaling, FeedbackTransition, FrozenAttention, FrozenBackend,
    FrozenReadout, GenerationReward, HistoryGeometry, ModelInitialization, ObjectiveReference,
    PretrainSelection, TrustRegionShape, TrustRegionSpec, TuneSpec, TurboEnnExperiment, VerifyMode,
};
use ennx::procedural_pool::ProposalMethod;
use ennx::{Perturbation, ReliabilityPolicy, Rescalarize};

#[derive(Clone)]
pub(super) enum Choice {
    Proposal(ProposalMethod),
    Distribution(Perturbation),
    Geometry(HistoryGeometry),
    Scaling(DistanceScaling, usize),
    Fit(bool),
    Shape(TrustRegionShape),
    Acquisition(AcquisitionSpec),
    TrustRegion(TrustRegionSpec),
    Selection(PretrainSelection),
    Reference(ObjectiveReference),
    Initialization(ModelInitialization),
    Feedback(FeedbackTransition),
    Verify(VerifyMode),
    Attention(FrozenAttention),
    Readout(FrozenReadout),
    Backend(FrozenBackend),
    Arms(u32),
}

impl Choice {
    pub(super) fn apply(&self, spec: &mut TuneSpec) {
        match self {
            Self::Proposal(value) => spec.proposal.method = *value,
            Self::Distribution(value) => spec.proposal.distribution = Some(*value),
            Self::Geometry(value) => spec.enn.geometry = *value,
            Self::Scaling(value, neighbors) => {
                spec.enn.scaling = *value;
                spec.enn.local_neighbors =
                    (*value == DistanceScaling::SelfTuning).then_some(*neighbors);
            }
            Self::Fit(value) => spec.enn.fit.neighbors = *value,
            Self::Shape(value) => spec.trust_region.bounds_mut().shape = Some(*value),
            Self::Acquisition(value) => spec.acquisition = value.clone(),
            Self::TrustRegion(value) => spec.trust_region = value.clone(),
            Self::Selection(value) => spec.run.selection = Some(*value),
            Self::Reference(value) => spec.objective.reference = Some(*value),
            Self::Arms(value) => spec.proposal.arms = *value,
            _ => self.apply_generation(spec),
        }
    }

    fn apply_generation(&self, spec: &mut TuneSpec) {
        match self {
            Self::Initialization(value) => {
                spec.generation.as_mut().unwrap().initialization = *value
            }
            Self::Feedback(value) => spec.generation.as_mut().unwrap().feedback_transition = *value,
            Self::Verify(value) => spec.generation.as_mut().unwrap().verify.mode = *value,
            Self::Attention(value) => {
                if let GenerationReward::FrozenQwen { attention, .. } =
                    &mut spec.generation.as_mut().unwrap().reward
                {
                    *attention = *value;
                }
            }
            Self::Readout(value) => {
                if let GenerationReward::FrozenQwen { readout, .. } =
                    &mut spec.generation.as_mut().unwrap().reward
                {
                    *readout = *value;
                }
            }
            Self::Backend(value) => {
                if let GenerationReward::FrozenQwen { backend, .. } =
                    &mut spec.generation.as_mut().unwrap().reward
                {
                    *backend = *value;
                }
            }
            _ => unreachable!("search choices are applied separately"),
        }
    }
}

pub(super) struct Axis {
    pub name: &'static str,
    pub choices: Vec<(&'static str, Choice)>,
}

macro_rules! axis {
    ($name:literal, $variant:ident, [$($label:literal => $value:expr),+ $(,)?]) => {
        Axis { name: $name, choices: vec![$(($label, Choice::$variant($value))),+] }
    };
}

pub(super) fn axes(spec: &TuneSpec) -> Vec<Axis> {
    let mut result = vec![acquisition(spec)];
    if !matches!(
        spec.experiment,
        TurboEnnExperiment::Pretrain | TurboEnnExperiment::Generation
    ) {
        return result;
    }
    result.extend(search_axes(spec));
    if spec.experiment == TurboEnnExperiment::Pretrain {
        result.push(axis!("run.selection", Selection, [
            "enn" => PretrainSelection::Enn, "random" => PretrainSelection::Random,
        ]));
    }
    if spec.generation.is_some() {
        result.extend(generation_axes(spec));
    } else {
        result.push(axis!("objective.reference", Reference, [
            "moving-incumbent" => ObjectiveReference::MovingIncumbent,
            "frozen-initial" => ObjectiveReference::FrozenInitial,
        ]));
    }
    result
}

fn search_axes(spec: &TuneSpec) -> Vec<Axis> {
    let neighbors = spec.enn.local_neighbors.unwrap_or(8);
    let policy = match spec.trust_region {
        TrustRegionSpec::Reliability { policy, .. } => policy,
        _ => ReliabilityPolicy::default(),
    };
    let bounds = spec.trust_region.bounds().clone();
    vec![
        axis!("proposal.method", Proposal, [
            "independent" => ProposalMethod::Independent,
            "spectral-basis" => ProposalMethod::SpectralBasis,
            "polynomial-threshold" => ProposalMethod::PolynomialThreshold,
        ]),
        axis!("proposal.distribution", Distribution, [
            "gaussian" => Perturbation::Gaussian, "rademacher" => Perturbation::Rademacher,
        ]),
        axis!("proposal.arms", Arms, ["one-arm" => 1, "two-arms" => 2, "four-arms" => 4]),
        axis!("enn.geometry", Geometry, [
            "realized" => HistoryGeometry::Realized, "latent" => HistoryGeometry::Latent,
        ]),
        Axis {
            name: "enn.scaling",
            choices: vec![
                (
                    "global",
                    Choice::Scaling(DistanceScaling::Global, neighbors),
                ),
                (
                    "self-tuning",
                    Choice::Scaling(DistanceScaling::SelfTuning, neighbors),
                ),
            ],
        },
        axis!("enn.fit.neighbors", Fit, ["fixed" => false, "fitted" => true]),
        axis!("trust-region.shape", Shape, [
            "scalar" => TrustRegionShape::Scalar,
            "tensor-family-static" => TrustRegionShape::TensorFamilyStatic,
            "tensor-family-learned" => TrustRegionShape::TensorFamilyLearned,
        ]),
        axis!("trust-region.method", TrustRegion, [
            "turbo" => TrustRegionSpec::Turbo { bounds: bounds.clone() },
            "morbo" => TrustRegionSpec::Morbo {
                bounds: bounds.clone(), regions: 4,
                rescalarize: Rescalarize::OnRestart, clip: true,
            },
            "reliability" => TrustRegionSpec::Reliability { bounds, policy },
        ]),
    ]
}

fn acquisition(spec: &TuneSpec) -> Axis {
    // Preserve supplied numerical policy. Never manufacture vector preferences.
    let choices = match &spec.acquisition {
        AcquisitionSpec::Ucb { .. } => vec![
            ("ucb", Choice::Acquisition(spec.acquisition.clone())),
            ("thompson", Choice::Acquisition(AcquisitionSpec::Thompson)),
        ],
        AcquisitionSpec::Thompson => vec![
            ("ucb", Choice::Acquisition(AcquisitionSpec::default())),
            ("thompson", Choice::Acquisition(AcquisitionSpec::Thompson)),
        ],
        AcquisitionSpec::Pareto { .. } => {
            vec![("pareto", Choice::Acquisition(spec.acquisition.clone()))]
        }
        AcquisitionSpec::AugmentedChebyshev { scales, .. } => {
            vec![
                (
                    "pareto",
                    Choice::Acquisition(AcquisitionSpec::Pareto {
                        scales: scales.clone(),
                    }),
                ),
                (
                    "augmented-chebyshev",
                    Choice::Acquisition(spec.acquisition.clone()),
                ),
            ]
        }
    };
    Axis {
        name: "acquisition.method",
        choices,
    }
}

fn generation_axes(spec: &TuneSpec) -> Vec<Axis> {
    let generation = spec.generation.as_ref().unwrap();
    let mut result = vec![axis!("generation.feedback-transition", Feedback, [
        "identity" => FeedbackTransition::Identity,
        "projected-sigmoid" => FeedbackTransition::ProjectedSigmoid,
    ])];
    // Initializers do no work when a checkpoint supplies the weights.
    if generation.checkpoint.is_none() {
        result.push(axis!("generation.initialization", Initialization, [
            "patterned" => ModelInitialization::Patterned,
            "gpt2" => ModelInitialization::Gpt2,
            "megatron" => ModelInitialization::Megatron,
            "xavier-uniform" => ModelInitialization::XavierUniform,
        ]));
    }
    // Auto is a dispatch policy, not a third implementation. Emit concrete paths.
    result.push(axis!("generation.verify.mode", Verify, [
        "serial" => VerifyMode::Serial, "accepted-prefix" => VerifyMode::AcceptedPrefix,
    ]));
    if matches!(generation.reward, GenerationReward::FrozenQwen { .. }) {
        result.extend([
            axis!("generation.reward.attention", Attention, [
                "reference" => FrozenAttention::Reference, "tiled16" => FrozenAttention::Tiled16,
            ]),
            axis!("generation.reward.readout", Readout, [
                "reference" => FrozenReadout::Reference, "mps-fp32" => FrozenReadout::MpsFp32,
            ]),
            axis!("generation.reward.backend", Backend, [
                "reference" => FrozenBackend::Reference, "fp32" => FrozenBackend::Fp32,
                "fp16" => FrozenBackend::Fp16,
            ]),
        ]);
    }
    result
}
