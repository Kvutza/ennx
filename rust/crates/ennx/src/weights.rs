use rand::distributions::{Distribution, Standard};
use rand::rngs::StdRng;
use rand::SeedableRng;
#[cfg(all(target_os = "macos", feature = "metal"))]
use std::collections::HashMap;
#[cfg(all(target_os = "macos", feature = "metal"))]
use std::sync::{Mutex, OnceLock};
#[cfg(all(target_os = "macos", feature = "metal"))]
use std::time::Instant;

use crate::util::insert_neighbor;

#[cfg(all(target_os = "macos", feature = "metal"))]
mod metal_weights;

#[cfg(feature = "opencl")]
mod opencl_weights;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcquisitionKind {
    Ucb,
    Thompson,
    Pareto,
}

impl AcquisitionKind {
    pub fn parse(name: &str) -> Result<Self, String> {
        match name.trim().to_ascii_lowercase().as_str() {
            "ucb" => Ok(Self::Ucb),
            "thompson" => Ok(Self::Thompson),
            "pareto" => Ok(Self::Pareto),
            other => Err(format!(
                "unknown acquisition {other:?}; expected 'ucb', 'thompson', or 'pareto'"
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeDevice {
    Auto,
    Cpu,
    Metal,
    Agx,
    OpenCl,
    Cuda,
}

impl ComputeDevice {
    pub fn parse(name: &str) -> Result<Self, String> {
        match name.trim().to_ascii_lowercase().as_str() {
            "" | "auto" => Ok(Self::Auto),
            "cpu" => Ok(Self::Cpu),
            "metal" => Ok(Self::Metal),
            "agx" => Ok(Self::Agx),
            "opencl" | "ocl" => Ok(Self::OpenCl),
            "cuda" => Ok(Self::Cuda),
            other => Err(format!(
                "unknown compute device {other:?}; expected 'auto', 'cpu', 'metal', 'agx', 'opencl', or 'cuda'"
            )),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct WeightBlock {
    pub offset: usize,
    pub length: usize,
    pub bits: u8,
    pub encoding: crate::trials::EncodingType,
    pub quantization_scale: f32,
    pub metric_scale: f32,
    pub weight: f32,
}

impl WeightBlock {
    pub fn new(
        offset: usize,
        length: usize,
        bits: u8,
        quantization_scale: f32,
        metric_scale: f32,
        weight: f32,
    ) -> Result<Self, String> {
        Self::new_with_encoding(
            offset,
            length,
            bits,
            crate::trials::EncodingType::parse(bits, None)?,
            quantization_scale,
            metric_scale,
            weight,
        )
    }

    pub fn new_with_encoding(
        offset: usize,
        length: usize,
        bits: u8,
        encoding: crate::trials::EncodingType,
        quantization_scale: f32,
        metric_scale: f32,
        weight: f32,
    ) -> Result<Self, String> {
        if length == 0 {
            return Err("quantized-weight block length must be positive".to_string());
        }
        if bits != 4 && bits != 8 {
            return Err(format!(
                "quantized-weight block bits must be 4 or 8, got {bits}"
            ));
        }
        for (name, value) in [
            ("quantization_scale", quantization_scale),
            ("metric_scale", metric_scale),
            ("weight", weight),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(format!("{name} must be finite and nonnegative"));
            }
        }
        Ok(Self {
            offset,
            length,
            bits,
            encoding,
            quantization_scale,
            metric_scale,
            weight,
        })
    }

    fn row_bytes(&self) -> usize {
        match self.bits {
            4 => self.length.div_ceil(2),
            8 => self.length,
            _ => unreachable!("block bits are checked at construction"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct WeightSelectConfig {
    pub neighbors: usize,
    pub epistemic_scale: f32,
    pub aleatoric_scale: f32,
    pub y_scale: f32,
    pub beta: f32,
    pub acquisition: AcquisitionKind,
    pub seed: u64,
    pub device: ComputeDevice,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WeightSelectResult {
    pub index: usize,
    pub score: f32,
}

#[derive(Debug, Clone, Copy)]
struct Prediction {
    mean: f32,
    se: f32,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
struct WeightSelection<'a> {
    observations: &'a [u8],
    observation_count: usize,
    outcomes: &'a [f32],
    candidates: &'a [u8],
    candidate_count: usize,
    blocks: &'a [WeightBlock],
    row_bytes: usize,
}

pub fn select_weights(
    observations: &[u8],
    observation_count: usize,
    outcomes: &[f32],
    candidates: &[u8],
    candidate_count: usize,
    blocks: &[WeightBlock],
    config: WeightSelectConfig,
) -> Result<WeightSelectResult, String> {
    let row_bytes = check_weight_inputs(
        observations,
        observation_count,
        outcomes,
        candidates,
        candidate_count,
        blocks,
        config.neighbors,
    )?;

    match config.device {
        ComputeDevice::Cpu => {}
        ComputeDevice::Metal => {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            {
                return metal_weights::select(
                    observations,
                    observation_count,
                    outcomes,
                    candidates,
                    candidate_count,
                    blocks,
                    row_bytes,
                    config,
                );
            }
            #[cfg(not(all(target_os = "macos", feature = "metal")))]
            {
                return Err("Metal ENN device is not available in this build".to_string());
            }
        }
        ComputeDevice::Agx => {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            {
                return metal_weights::select(
                    observations,
                    observation_count,
                    outcomes,
                    candidates,
                    candidate_count,
                    blocks,
                    row_bytes,
                    config,
                );
            }
            #[cfg(not(all(target_os = "macos", feature = "metal")))]
            {
                return Err("AGX ENN device is not available in this build".to_string());
            }
        }
        ComputeDevice::OpenCl => {
            #[cfg(feature = "opencl")]
            {
                return opencl_weights::select(
                    observations,
                    observation_count,
                    outcomes,
                    candidates,
                    candidate_count,
                    blocks,
                    row_bytes,
                    config,
                );
            }
            #[cfg(not(feature = "opencl"))]
            {
                return Err("OpenCL ENN device is not available in this build".to_string());
            }
        }
        ComputeDevice::Cuda => {
            return Err(
                "CUDA materialized weight selection is not available; use resident trial search"
                    .to_string(),
            );
        }
        ComputeDevice::Auto => {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            {
                return select_weights_auto(
                    WeightSelection {
                        observations,
                        observation_count,
                        outcomes,
                        candidates,
                        candidate_count,
                        blocks,
                        row_bytes,
                    },
                    config,
                );
            }
            #[cfg(all(feature = "opencl", not(all(target_os = "macos", feature = "metal"))))]
            {
                return opencl_weights::select(
                    observations,
                    observation_count,
                    outcomes,
                    candidates,
                    candidate_count,
                    blocks,
                    row_bytes,
                    config,
                );
            }
        }
    }

    select_weight_cpu(
        observations,
        observation_count,
        outcomes,
        candidates,
        candidate_count,
        blocks,
        row_bytes,
        config,
    )
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn select_weights_auto(
    input: WeightSelection<'_>,
    config: WeightSelectConfig,
) -> Result<WeightSelectResult, String> {
    static ROUTES: OnceLock<Mutex<HashMap<u32, bool>>> = OnceLock::new();
    let work = input
        .observation_count
        .saturating_mul(input.candidate_count)
        .saturating_mul(input.row_bytes)
        .max(1);
    let bucket = work.ilog2();
    let routes = ROUTES.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(&gpu) = routes
        .lock()
        .map_err(|_| "weight route cache poisoned")?
        .get(&bucket)
    {
        let device = if gpu {
            ComputeDevice::Agx
        } else {
            ComputeDevice::Cpu
        };
        return select_weights(
            input.observations,
            input.observation_count,
            input.outcomes,
            input.candidates,
            input.candidate_count,
            input.blocks,
            WeightSelectConfig { device, ..config },
        );
    }

    let cpu_start = Instant::now();
    let cpu = select_weights(
        input.observations,
        input.observation_count,
        input.outcomes,
        input.candidates,
        input.candidate_count,
        input.blocks,
        WeightSelectConfig {
            device: ComputeDevice::Cpu,
            ..config
        },
    )?;
    let cpu_time = cpu_start.elapsed();
    let gpu_start = Instant::now();
    let gpu = metal_weights::select(
        input.observations,
        input.observation_count,
        input.outcomes,
        input.candidates,
        input.candidate_count,
        input.blocks,
        input.row_bytes,
        config,
    )?;
    let gpu_time = gpu_start.elapsed();
    let agrees = cpu.index == gpu.index && (cpu.score - gpu.score).abs() <= 1e-4;
    let use_gpu = agrees && gpu_time < cpu_time;
    routes
        .lock()
        .map_err(|_| "weight route cache poisoned")?
        .insert(bucket, use_gpu);
    Ok(if use_gpu { gpu } else { cpu })
}

pub(crate) fn thompson_draws(count: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..count).map(|_| standard_normal(&mut rng)).collect()
}

#[cfg(any(
    all(target_os = "linux", target_arch = "x86_64", feature = "cuda"),
    all(target_os = "macos", feature = "metal"),
    feature = "opencl"
))]
pub(crate) fn acquisition_code(acquisition: AcquisitionKind) -> u32 {
    match acquisition {
        AcquisitionKind::Ucb => 0,
        AcquisitionKind::Thompson => 1,
        AcquisitionKind::Pareto => 2,
    }
}

fn check_weight_inputs(
    observations: &[u8],
    observation_count: usize,
    outcomes: &[f32],
    candidates: &[u8],
    candidate_count: usize,
    blocks: &[WeightBlock],
    neighbors: usize,
) -> Result<usize, String> {
    if observation_count == 0 {
        return Err("quantized-weight ENN selection requires at least one observation".to_string());
    }
    if candidate_count == 0 {
        return Err("quantized-weight ENN selection requires at least one candidate".to_string());
    }
    if outcomes.len() != observation_count {
        return Err(format!(
            "outcome count {} does not match observation count {observation_count}",
            outcomes.len()
        ));
    }
    if neighbors == 0 || neighbors > observation_count {
        return Err(format!(
            "neighbor count must be between one and {observation_count}"
        ));
    }
    if blocks.is_empty() {
        return Err(
            "quantized-weight ENN selection requires at least one metric block".to_string(),
        );
    }
    let row_bytes: usize = blocks.iter().map(WeightBlock::row_bytes).sum();
    if row_bytes == 0 {
        return Err("quantized-weight row byte width must be positive".to_string());
    }
    if observations.len() != observation_count.saturating_mul(row_bytes) {
        return Err(format!(
            "observation bytes {} do not match shape ({observation_count}, {row_bytes})",
            observations.len()
        ));
    }
    if candidates.len() != candidate_count.saturating_mul(row_bytes) {
        return Err(format!(
            "candidate bytes {} do not match shape ({candidate_count}, {row_bytes})",
            candidates.len()
        ));
    }
    if outcomes.iter().any(|value| !value.is_finite()) {
        return Err("outcomes must be finite".to_string());
    }
    Ok(row_bytes)
}

fn select_weight_cpu(
    observations: &[u8],
    observation_count: usize,
    outcomes: &[f32],
    candidates: &[u8],
    candidate_count: usize,
    blocks: &[WeightBlock],
    row_bytes: usize,
    config: WeightSelectConfig,
) -> Result<WeightSelectResult, String> {
    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut best = WeightSelectResult {
        index: 0,
        score: f32::NEG_INFINITY,
    };
    for candidate_index in 0..candidate_count {
        let prediction = predict_candidate_cpu(
            observations,
            observation_count,
            outcomes,
            &candidates[candidate_index * row_bytes..(candidate_index + 1) * row_bytes],
            blocks,
            row_bytes,
            config,
        );
        let score = match config.acquisition {
            AcquisitionKind::Ucb => prediction.mean + config.beta * prediction.se,
            AcquisitionKind::Pareto => prediction.mean + prediction.se,
            AcquisitionKind::Thompson => {
                let z = standard_normal(&mut rng);
                prediction.mean + prediction.se * z
            }
        };
        if score > best.score || (score == best.score && candidate_index < best.index) {
            best = WeightSelectResult {
                index: candidate_index,
                score,
            };
        }
    }
    Ok(best)
}

fn predict_candidate_cpu(
    observations: &[u8],
    observation_count: usize,
    outcomes: &[f32],
    candidate: &[u8],
    blocks: &[WeightBlock],
    row_bytes: usize,
    config: WeightSelectConfig,
) -> Prediction {
    let mut nearest = vec![(f32::INFINITY, 0usize); config.neighbors];
    for observation_index in 0..observation_count {
        let observation =
            &observations[observation_index * row_bytes..(observation_index + 1) * row_bytes];
        let distance = weight_distance(candidate, observation, blocks);
        insert_neighbor(&mut nearest, distance, observation_index);
    }
    weighted_prediction(&nearest, outcomes, config)
}

fn weighted_prediction(
    nearest: &[(f32, usize)],
    outcomes: &[f32],
    config: WeightSelectConfig,
) -> Prediction {
    let mut weight_sum = 0.0f32;
    let mut weighted_outcome = 0.0f32;
    for &(distance, index) in nearest {
        let variance = 1.0e-9f32 + config.epistemic_scale * distance + config.aleatoric_scale;
        let weight = 1.0 / variance.max(1.0e-12);
        weight_sum += weight;
        weighted_outcome += weight * outcomes[index];
    }
    let mean = weighted_outcome / weight_sum.max(1.0e-12);
    let se = (1.0 / weight_sum.max(1.0e-12)).sqrt() * config.y_scale;
    Prediction { mean, se }
}

pub fn weight_distance(left: &[u8], right: &[u8], blocks: &[WeightBlock]) -> f32 {
    let mut distance = 0.0f32;
    let mut byte_base = 0usize;
    for block in blocks {
        let scale = block.quantization_scale;
        let weight = block.weight;
        match block.bits {
            4 => {
                for element in 0..block.length {
                    let byte = byte_base + element / 2;
                    let shift = if element % 2 == 0 { 0 } else { 4 };
                    let code_a = u32::from((left[byte] >> shift) & 0x0f);
                    let code_b = u32::from((right[byte] >> shift) & 0x0f);
                    let a = crate::trials::decode_code(code_a, block.encoding, scale);
                    let b = crate::trials::decode_code(code_b, block.encoding, scale);
                    let delta = a - b;
                    distance = delta.mul_add(delta * weight, distance);
                }
            }
            8 => {
                for element in 0..block.length {
                    let byte = byte_base + element;
                    let code_a = u32::from(left[byte]);
                    let code_b = u32::from(right[byte]);
                    let a = crate::trials::decode_code(code_a, block.encoding, scale);
                    let b = crate::trials::decode_code(code_b, block.encoding, scale);
                    let delta = a - b;
                    distance = delta.mul_add(delta * weight, distance);
                }
            }
            _ => unreachable!("block bits are checked at construction"),
        }
        byte_base += block.row_bytes();
    }
    distance
}

fn standard_normal(rng: &mut StdRng) -> f32 {
    let mut u1: f32 = Standard.sample(rng);
    let u2: f32 = Standard.sample(rng);
    u1 = u1.clamp(1.0e-7, 1.0 - 1.0e-7);
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

mod sparse;
pub use sparse::{
    apply_sparse, blocks_for_words, draw_sparse, merge_values, missing_words, sparse_union,
    sparse_xor, take_words,
};
#[cfg(test)]
#[path = "weights/tests.rs"]
mod tests;
