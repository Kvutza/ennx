//! Standalone harness: exercises the new evaluator even before lib.rs wiring.
#![cfg(all(target_os = "macos", feature = "metal"))]
#![allow(dead_code)]

#[path = "../src/apple_gpu.rs"]
mod apple_gpu;
#[path = "../src/flame_metal.rs"]
mod flame_metal;
