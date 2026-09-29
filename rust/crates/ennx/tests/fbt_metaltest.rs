//! Isolated native feedback tests, independent of other model implementations.
#![cfg(all(target_os = "macos", feature = "metal"))]
#![allow(dead_code, unused_imports)]

#[path = "../src/apple_gpu.rs"]
mod apple_gpu;
#[path = "../src/fbt.rs"]
mod fbt;
#[path = "../src/fbt_attention.rs"]
mod fbt_attention;
#[path = "../src/fbt_metal.rs"]
mod fbt_metal;
#[path = "../src/fbt_model.rs"]
mod fbt_model;
#[path = "../src/fbt_mps.rs"]
mod fbt_mps;

#[path = "../src/bf16_metal.rs"]
mod bf16_metal;
#[path = "../src/objective_observation.rs"]
mod objective_observation;
use ennx::procedural_pool;
pub use ennx::{
    Perturbation, Rescalarize, config, fit, fitter, hash, hypervolume, mbtrregn, params,
    reliability_region, tensor_store, threshold, trust_region,
};
mod trials {
    pub use ennx::experimental::SearchConfig as Ask;
}
mod weights {
    pub use ennx::experimental::AcquisitionKind;
}
