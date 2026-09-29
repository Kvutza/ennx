//! Compile and exercise the standalone backend without changing library registrations.
#![cfg(all(target_os = "macos", feature = "metal"))]
#![allow(dead_code)]

#[path = "../src/apple_gpu.rs"]
mod apple_gpu;
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
