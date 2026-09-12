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
pub use ennx::{Perturbation, config, fit, fitter, hash, params, reliability_region, trust_region};
mod trials {
    pub use ennx::experimental::SearchConfig as Ask;
}
mod weights {
    pub use ennx::experimental::AcquisitionKind;
}
