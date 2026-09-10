//! Core ENN algorithm implementations in Rust.
//!
//! This crate provides the algorithmic core of the Epistemic Nearest Neighbors
//! library, with implementations designed for parity with the Python reference.

#![allow(clippy::pedantic, clippy::nursery, clippy::cargo)]

pub mod acquisition;
#[cfg(all(target_os = "macos", feature = "metal"))]
mod apple_gpu;
pub mod backend;
#[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
mod bf16_search;
pub mod candidates;
pub mod capability;
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
mod forward_metal;
mod forward_program;
mod forward_weights;
pub mod hash;
pub mod hypervolume;
pub mod incumbent_tracker;
pub mod index;
mod knn;
pub mod mbtrregn;
pub mod model;
pub mod optimizer;
pub mod optimizer_factory;
pub mod params;
pub mod posterior;
pub mod prelude;
mod quantization;
pub mod search;
pub mod stats;
pub mod strategy;
pub mod surrogate;
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
pub use backend::DiskBpannEnnBackend;
pub use backend::{EnnBackend, EnnStorage, InMemoryEnnBackend};
pub use candidates::{from_unit, generate_candidates, generate_lhd, to_unit, CandidateRV};
pub use capability::{
    backends, matrix, operations, support, Backend, Capability, Operation, Support,
};
pub use config::{
    lhd_only, turbo_enn, turbo_zero, AcquisitionConfig, CandidateConfig, ConfigOverrides,
    InitStrategy, OptimizerConfig, SurrogateConfig, TrustRegionKind,
};
pub use draw::{Candidates, ConditionalDraw, DrawInternals, NeighborData};
pub use error::{ENNError, EPS_VAR};
pub use file_config::{bpann_config, config_path, set_path, BpannConfig, Config, ConfigFile};
pub use fit::{subsample_loglik, subsample_model};
pub use fitter::ENNFitter;
pub use hash::normal_hash;
pub use hypervolume::hypervolume2d_max;
pub use incumbent_tracker::IncumbentTracker;
pub use index::{ENNIndex, IndexDriver, IndexError};
pub use mbtrregn::{MorboTRSettings, MorboTrustRegion, Rescalarize};
pub use model::ENN;
pub use model::{EnnIndexAccess, EnnRowAccess, ModelOptions};
pub use optimizer::obs_access::ObsAccess;
pub use optimizer::{Optimizer, Telemetry};
pub use optimizer_factory::{create_lhd, create_optimizer, enn_optimizer};
pub use params::{ENNNormal, ENNParams, ParamsError, PosteriorFlags};
pub use posterior::{compute_internals, conditional_internals, WeightedPosteriorData};
pub use stats::WeightedStats;
pub use strategy::Strategy;
pub use surrogate::{ENNSurrogate, ENNSurrogateConfig, Surrogate, SurrogatePrediction};
pub use traits::PosteriorComputation;
pub use trregncfg::TrustRegionConfig;
pub use trust_region::{NoTrustRegion, TRLengthConfig, TrustRegionError, TurboTrustRegion};
pub use util::{argmax_tie, pareto2d_max, sobol_indices, standardize_y};
