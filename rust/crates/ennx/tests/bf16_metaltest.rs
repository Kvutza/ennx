//! Compile and exercise the standalone backend without changing library registrations.
#![cfg(all(target_os = "macos", feature = "metal"))]
#![allow(dead_code)]

#[path = "../src/apple_gpu.rs"]
mod apple_gpu;
#[path = "../src/bf16_metal.rs"]
mod bf16_metal;

pub use ennx::{Perturbation, config, fit, fitter, hash, params, reliability_region, trust_region};
mod trials {
    pub use ennx::experimental::SearchConfig as Ask;
}
mod weights {
    pub use ennx::experimental::AcquisitionKind;
}
