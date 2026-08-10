//! Core ENN algorithm implementations in Rust.
//!
//! This crate provides the algorithmic core of the Epistemic Nearest Neighbors
//! library, with implementations designed for parity with the Python reference.

#![allow(clippy::pedantic, clippy::nursery, clippy::cargo)]

pub mod acquisition;
#[cfg(all(target_os = "macos", feature = "metal"))]
pub mod apple_gpu;
pub mod backend;
pub mod candidates;
pub mod config;
mod dense;
pub mod disk_bpann;
pub mod draw;
pub mod error;
pub mod experimental;
pub mod file_config;
pub mod fit;
pub mod fitter;
#[cfg(all(target_os = "macos", feature = "metal"))]
pub mod forward_metal;
pub mod forward_program;
pub mod forward_weights;
pub mod hash;
pub mod hypervolume;
pub mod incumbent_tracker;
pub mod index;
pub mod knn;
pub mod model;
pub mod morbo_trust_region;
pub mod optimizer;
pub mod optimizer_factory;
pub mod params;
pub mod posterior;
pub mod stats;
pub mod strategy;
pub mod surrogate;
pub mod tracy;
#[cfg(all(target_os = "macos", feature = "metal"))]
mod tracy_metal;
pub mod traits;
pub mod trials;
pub mod trust_region;
pub mod trust_region_config;
pub mod util;
pub mod weights;
pub mod y_bounds;

#[cfg(test)]
pub(crate) mod test_helpers;

pub use acquisition::{
    AcquisitionError, ParetoAcquisition, RandomAcquisition, ThompsonAcquisition, UCBAcquisition,
};
pub use backend::DiskBpannEnnBackend;
pub use backend::{EnnBackend, EnnStorage, InMemoryEnnBackend};
pub use candidates::{from_unit, generate_candidates, generate_lhd, to_unit, CandidateRV};
pub use config::{
    lhd_only_config, turbo_enn_config, turbo_zero_config, AcquisitionConfig, CandidateConfig,
    ConfigOverrides, InitStrategy, OptimizerConfig, SurrogateConfig, TrustRegionKind,
};
pub use draw::{Candidates, ConditionalPosteriorDrawInternals, DrawInternals, NeighborData};
pub use error::{ENNError, EPS_VAR};
pub use file_config::{
    default_config_path, install_bpann_tuning_from_config, set_config_path, BpannConfig, Config,
    ConfigFile,
};
pub use fit::{subsample_loglik, subsample_loglik_model};
pub use fitter::ENNFitter;
pub use forward_program::{
    ForwardEvaluator, ForwardOp, ForwardProgram, KdaControlRequest, KdaDispatch, KdaEncoder,
    KdaForwardRequest, KdaMoeDispatch, KdaMoeLayerRequest, KdaPackedLinear, KdaTensorLayout,
    KernelPlan, PackedAffinePlan, ResidentBoState, ResidentRound, WorkAxis, WorkGrid, WorkTile,
};
pub use forward_weights::PackedModel;
pub use hash::{normal_hash_batch_multi_seed, normal_hash_batch_multi_seed_fast};
pub use hypervolume::hypervolume_2d_max;
pub use incumbent_tracker::IncrementalIncumbentTracker;
pub use index::{ENNIndex, IndexDriver, IndexError};
pub use model::EpistemicNearestNeighbors;
pub use model::{EnnIndexAccess, EnnRowAccess, ModelOptions};
pub use morbo_trust_region::{MorboTRSettings, MorboTrustRegion, Rescalarize};
pub use optimizer::obs_access::ObsAccess;
pub use optimizer::{
    MultiTrustRegionConfig, MultiTrustRegionState, ObservationDelta, Optimizer, ProgramTrial,
    RegionBatch, RegionCandidate, SharingPolicy, Telemetry,
};
pub use optimizer_factory::{
    create_optimizer_enn, create_optimizer_enn_multi_tr, create_optimizer_lhd,
    create_optimizer_zero,
};
pub use params::{ENNNormal, ENNParams, ParamsError, PosteriorFlags};
pub use posterior::{
    compute_conditional_posterior_internals, compute_posterior_internals, WeightedPosteriorData,
};
pub use stats::WeightedStats;
pub use strategy::Strategy;
pub use surrogate::{ENNSurrogate, ENNSurrogateConfig, Surrogate, SurrogatePrediction};
pub use traits::PosteriorComputation;
pub use trials::{
    Ask as WeightAsk, BpannHistory, Center as WeightCenter, EncodingType, IndexedObservation,
    Leaf as WeightLeaf, ObservationId, Search as WeightSearch, Trial as WeightTrial,
};
pub use trust_region::{NoTrustRegion, TRLengthConfig, TrustRegionError, TurboTrustRegion};
pub use trust_region_config::TrustRegionConfig;
pub use util::{
    argmax_random_tie, calculate_sobol_indices, pareto_front_2d_maximize, standardize_y,
};
pub use weights::{
    apply_sparse, blocks_for_words, draw_sparse, merge_values, missing_words, select_weights,
    sparse_union, sparse_xor, take_words, AcquisitionKind, ComputeBackend, WeightBlock,
    WeightSelectConfig, WeightSelectResult,
};
