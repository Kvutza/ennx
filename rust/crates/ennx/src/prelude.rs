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
    lhd_only_config, turbo_enn_config, turbo_zero_config, AcquisitionConfig, CandidateConfig,
    ConfigOverrides, InitStrategy, OptimizerConfig, SurrogateConfig, TrustRegionKind,
};
pub use crate::draw::{Candidates, ConditionalPosteriorDrawInternals, DrawInternals, NeighborData};
pub use crate::error::{ENNError, EPS_VAR};
pub use crate::fit::{subsample_loglik, subsample_loglik_model};
pub use crate::fitter::ENNFitter;
pub use crate::hypervolume::hypervolume_2d_max;
pub use crate::index::IndexDriver;
pub use crate::model::{EpistemicNearestNeighbors, ModelOptions};
pub use crate::optimizer::{Optimizer, Telemetry};
pub use crate::optimizer_factory::{
    create_optimizer_enn, create_optimizer_lhd, create_optimizer_zero,
};
pub use crate::params::{ENNNormal, ENNParams, ParamsError, PosteriorFlags};
pub use crate::posterior::{
    compute_conditional_posterior_internals, compute_posterior_internals, WeightedPosteriorData,
};
pub use crate::stats::WeightedStats;
pub use crate::strategy::Strategy;
pub use crate::surrogate::{ENNSurrogate, ENNSurrogateConfig, Surrogate, SurrogatePrediction};
pub use crate::traits::PosteriorComputation;
pub use crate::trust_region::{NoTrustRegion, TRLengthConfig, TrustRegionError, TurboTrustRegion};
pub use crate::trust_region_config::TrustRegionConfig;
pub use crate::util::{
    argmax_random_tie, calculate_sobol_indices, pareto_front_2d_maximize, standardize_y,
};
