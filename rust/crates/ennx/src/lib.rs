//! Core ENN algorithm implementations in Rust.
//!
//! This crate provides the algorithmic core of the Epistemic Nearest Neighbors
//! library, with implementations designed for parity with the Python reference.

#![allow(clippy::pedantic, clippy::nursery, clippy::cargo)]

pub mod acquisition;
pub mod agent_contract;
#[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
mod apple_gpu;
pub mod backend;
pub mod base_policy;
#[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
mod bf16_metal;
#[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
mod bf16_search;
pub mod candidates;
pub mod capability;
#[cfg(any(feature = "metal", test))]
mod code_overlap;
pub mod coding_outcome;
pub mod config;
pub mod context;
#[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
mod context_metal;
mod dense;
pub mod disk_bpann;
pub mod draw;
pub mod error;
pub mod experimental;
pub mod fbt;
#[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
mod fbt_attention;
#[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
mod fbt_metal;
#[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
mod fbt_model;
#[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
mod fbt_moe;
#[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
mod fbt_mps;
#[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
mod fbt_pisa1;
pub mod file_config;
pub mod fit;
pub mod fitter;
#[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
mod flame_metal;
#[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
mod forward_metal;
mod forward_program;
mod forward_weights;
pub mod hash;
pub mod hypervolume;
pub mod incumbent_tracker;
pub mod index;
mod knn;
pub mod mbtrregn;
pub mod metric_auto;
pub mod metric_loo;
mod metric_rng;
mod metric_rows;
mod metric_seed;
mod metric_sobol;
pub mod metric_weights;
pub mod model;
pub mod objective_observation;
pub mod optimizer;
pub mod optimizer_factory;
pub mod params;
pub mod perturb;
pub mod posterior;
pub mod prelude;
#[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
mod pretrain_data;
pub mod procedural_pool;
mod quantization;
#[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
mod qwen_metal;
#[cfg(any(test, all(target_os = "macos", feature = "metal")))]
mod reconstruction;
pub mod reliability_region;
pub mod search;
pub mod stats;
pub mod strategy;
pub mod surrogate;
pub mod tensor_store;
pub mod text;
pub mod threshold;
pub mod traits;
mod trials;
pub mod trregncfg;
pub mod trust_region;
pub mod util;
mod weights;
pub mod y_bounds;

#[cfg(test)]
pub(crate) mod test_helpers;

pub use acquisition::{
    AcquisitionError, ParetoAcquisition, RandomAcquisition, ThompsonAcquisition, UCBAcquisition,
};
pub use agent_contract::{
    AGENT_SCHEMA, AgentMessage, AgentTranscript, AssistantSegment, DataSplit, EpisodeTermination,
    TaskProvenance,
};
pub use backend::DiskBpannEnnBackend;
pub use backend::{EnnBackend, EnnStorage, InMemoryEnnBackend};
pub use base_policy::{BasePolicyManifest, POLICY_SCHEMA, PolicyQualification};
pub use candidates::{CandidateRV, from_unit, generate_candidates, generate_lhd, to_unit};
pub use capability::{
    Backend, Capability, Operation, Support, backends, matrix, operations, support,
};
pub use coding_outcome::{
    CODING_SCHEMA, CandidateEfficiency, CandidateEligibility, CodingOutcome, ExecutableChecks,
};
pub use config::{
    AcquisitionConfig, CandidateConfig, ConfigOverrides, DistanceScaling, InitStrategy,
    ObjectiveReference, OptimizerConfig, PretrainCorpus, PretrainModel, ResidentEnnConfig,
    SurrogateConfig, TrustRegionKind, TurboEnnExperiment, lhd_only, turbo_enn, turbo_zero,
};
pub use draw::{Candidates, ConditionalDraw, DrawInternals, NeighborData};
pub use error::{ENNError, EPS_VAR};
pub use file_config::{BpannConfig, Config, ConfigFile, bpann_config, config_path, set_path};
pub use fit::{row_loglik, subsample_loglik, subsample_model};
pub use fitter::ENNFitter;
pub use hash::normal_hash;
pub use hypervolume::{hypervolume_max, hypervolume2d_max};
pub use incumbent_tracker::IncumbentTracker;
pub use index::{ENNIndex, IndexDriver, IndexError};
pub use mbtrregn::{MorboTRSettings, MorboTrustRegion, Rescalarize};
pub use model::ENN;
pub use model::{EnnIndexAccess, EnnRowAccess, ModelOptions};
pub use optimizer::obs_access::ObsAccess;
pub use optimizer::{Optimizer, Telemetry};
pub use optimizer_factory::{create_lhd, create_optimizer, enn_optimizer};
pub use params::{ENNNormal, ENNParams, ParamsError, PosteriorFlags};
pub use perturb::Perturbation;
pub use posterior::{WeightedPosteriorData, compute_internals, conditional_internals};
pub use reliability_region::{
    ReliabilityAction, ReliabilityController, ReliabilityEvidence, ReliabilityPolicy,
    ReliabilityTelemetry,
};
pub use stats::WeightedStats;
pub use strategy::Strategy;
pub use surrogate::{ENNSurrogate, ENNSurrogateConfig, Surrogate, SurrogatePrediction};
pub use traits::PosteriorComputation;
pub use trregncfg::TrustRegionConfig;
pub use trust_region::{NoTrustRegion, TRLengthConfig, TrustRegionError, TurboTrustRegion};
pub use util::{argmax_tie, pareto2d_max, sobol_indices, standardize_y};
