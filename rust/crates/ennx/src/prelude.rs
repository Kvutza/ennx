//! Curated stable Rust API surface.
//!
//! Lower-level native, packed-weight, and accelerator-frontier APIs live under
//! [`crate::experimental`].

pub use crate::acquisition::{
    AcquisitionError, ParetoAcquisition, RandomAcquisition, ThompsonAcquisition, UCBAcquisition,
};
pub use crate::backend::EnnStorage;
pub use crate::candidates::{from_unit, generate_candidates, generate_lhd, to_unit, CandidateRV};
pub use crate::config::{
    lhd_only, turbo_enn, turbo_zero, AcquisitionConfig, CandidateConfig, ConfigOverrides,
    InitStrategy, OptimizerConfig, SurrogateConfig, TrustRegionKind,
};
pub use crate::draw::{Candidates, ConditionalDraw, DrawInternals, NeighborData};
pub use crate::error::{ENNError, EPS_VAR};
pub use crate::fit::{subsample_loglik, subsample_model};
pub use crate::fitter::ENNFitter;
pub use crate::hypervolume::hypervolume2d_max;
pub use crate::index::IndexDriver;
pub use crate::model::{ModelOptions, ENN};
pub use crate::optimizer::{Optimizer, Telemetry};
pub use crate::optimizer_factory::{create_lhd, create_optimizer, enn_optimizer};
pub use crate::params::{ENNNormal, ENNParams, ParamsError, PosteriorFlags};
pub use crate::posterior::{compute_internals, conditional_internals, WeightedPosteriorData};
pub use crate::stats::WeightedStats;
pub use crate::strategy::Strategy;
pub use crate::surrogate::{ENNSurrogate, ENNSurrogateConfig, Surrogate, SurrogatePrediction};
pub use crate::traits::PosteriorComputation;
pub use crate::trregncfg::TrustRegionConfig;
pub use crate::trust_region::{NoTrustRegion, TRLengthConfig, TrustRegionError, TurboTrustRegion};
pub use crate::util::{argmax_tie, pareto2d_max, sobol_indices, standardize_y};
