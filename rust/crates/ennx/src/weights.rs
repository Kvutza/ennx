use rand::distributions::{Distribution, Standard};
use rand::rngs::StdRng;
use rand::Rng;
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
pub enum ComputeBackend {
    Auto,
    Cpu,
    Metal,
    Agx,
    OpenCl,
    Cuda,
}

impl ComputeBackend {
    pub fn parse(name: &str) -> Result<Self, String> {
        match name.trim().to_ascii_lowercase().as_str() {
            "" | "auto" => Ok(Self::Auto),
            "cpu" => Ok(Self::Cpu),
            "metal" => Ok(Self::Metal),
            "agx" => Ok(Self::Agx),
            "opencl" | "ocl" => Ok(Self::OpenCl),
            "cuda" => Ok(Self::Cuda),
            other => Err(format!(
                "unknown compute backend {other:?}; expected 'auto', 'cpu', 'metal', 'agx', 'opencl', or 'cuda'"
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
    pub backend: ComputeBackend,
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

    match config.backend {
        ComputeBackend::Cpu => {}
        ComputeBackend::Metal => {
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
                return Err("Metal ENN backend is not available in this build".to_string());
            }
        }
        ComputeBackend::Agx => {
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
                return Err("AGX ENN backend is not available in this build".to_string());
            }
        }
        ComputeBackend::OpenCl => {
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
                return Err("OpenCL ENN backend is not available in this build".to_string());
            }
        }
        ComputeBackend::Cuda => {
            return Err(
                "CUDA materialized weight selection is not available; use resident trial search"
                    .to_string(),
            );
        }
        ComputeBackend::Auto => {
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
        let backend = if gpu {
            ComputeBackend::Agx
        } else {
            ComputeBackend::Cpu
        };
        return select_weights(
            input.observations,
            input.observation_count,
            input.outcomes,
            input.candidates,
            input.candidate_count,
            input.blocks,
            WeightSelectConfig { backend, ..config },
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
            backend: ComputeBackend::Cpu,
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

pub fn sparse_union(rows: &[&[u32]]) -> Vec<u32> {
    let mut level: Vec<Vec<u32>> = rows.iter().map(|row| row.to_vec()).collect();
    if level.is_empty() {
        return Vec::new();
    }
    while level.len() > 1 {
        let mut next = Vec::with_capacity(level.len().div_ceil(2));
        let mut iter = level.into_iter();
        while let Some(left) = iter.next() {
            if let Some(right) = iter.next() {
                next.push(merge_words(&left, &right));
            } else {
                next.push(left);
            }
        }
        level = next;
    }
    level.pop().unwrap_or_default()
}

fn merge_words(left: &[u32], right: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(left.len() + right.len());
    let mut i = 0usize;
    let mut j = 0usize;
    while i < left.len() || j < right.len() {
        let value = if j == right.len() || (i < left.len() && left[i] < right[j]) {
            let value = left[i];
            i += 1;
            value
        } else if i == left.len() || right[j] < left[i] {
            let value = right[j];
            j += 1;
            value
        } else {
            let value = left[i];
            i += 1;
            j += 1;
            value
        };
        if out.last().copied() != Some(value) {
            out.push(value);
        }
    }
    out
}

pub fn sparse_xor(
    left_words: &[u32],
    left_masks: &[u32],
    right_words: &[u32],
    right_masks: &[u32],
) -> Result<(Vec<u32>, Vec<u32>), String> {
    check_move(left_words, left_masks)?;
    check_move(right_words, right_masks)?;
    let mut words = Vec::with_capacity(left_words.len() + right_words.len());
    let mut masks = Vec::with_capacity(left_masks.len() + right_masks.len());
    let mut i = 0usize;
    let mut j = 0usize;
    while i < left_words.len() || j < right_words.len() {
        if j == right_words.len() || (i < left_words.len() && left_words[i] < right_words[j]) {
            words.push(left_words[i]);
            masks.push(left_masks[i]);
            i += 1;
        } else if i == left_words.len() || right_words[j] < left_words[i] {
            words.push(right_words[j]);
            masks.push(right_masks[j]);
            j += 1;
        } else {
            let mask = left_masks[i] ^ right_masks[j];
            if mask != 0 {
                words.push(left_words[i]);
                masks.push(mask);
            }
            i += 1;
            j += 1;
        }
    }
    Ok((words, masks))
}

pub fn check_move(words: &[u32], masks: &[u32]) -> Result<(), String> {
    if words.len() != masks.len() {
        return Err("move words and masks must have the same length".to_string());
    }
    for i in 0..words.len() {
        if masks[i] == 0 {
            return Err("move masks must be nonzero".to_string());
        }
        if i > 0 && words[i - 1] >= words[i] {
            return Err("move words must be strictly increasing".to_string());
        }
    }
    Ok(())
}

pub fn missing_words(cached: &[u32], query: &[u32]) -> Vec<u32> {
    let mut out = Vec::new();
    let mut i = 0usize;
    for &word in query {
        while i < cached.len() && cached[i] < word {
            i += 1;
        }
        if i == cached.len() || cached[i] != word {
            out.push(word);
        }
    }
    out
}

pub fn merge_values(
    words: &[u32],
    values: &[u32],
    extra_words: &[u32],
    extra_values: &[u32],
) -> Result<(Vec<u32>, Vec<u32>), String> {
    if words.len() != values.len() || extra_words.len() != extra_values.len() {
        return Err("word and value arrays must have matching lengths".to_string());
    }
    let mut out_words = Vec::with_capacity(words.len() + extra_words.len());
    let mut out_values = Vec::with_capacity(values.len() + extra_values.len());
    let mut i = 0usize;
    let mut j = 0usize;
    while i < words.len() || j < extra_words.len() {
        if j == extra_words.len() || (i < words.len() && words[i] < extra_words[j]) {
            out_words.push(words[i]);
            out_values.push(values[i]);
            i += 1;
        } else if i == words.len() || extra_words[j] < words[i] {
            out_words.push(extra_words[j]);
            out_values.push(extra_values[j]);
            j += 1;
        } else {
            out_words.push(words[i]);
            out_values.push(extra_values[j]);
            i += 1;
            j += 1;
        }
    }
    Ok((out_words, out_values))
}

pub fn take_words(words: &[u32], values: &[u32], query: &[u32]) -> Result<Vec<u32>, String> {
    if words.len() != values.len() {
        return Err("word and value arrays must have matching lengths".to_string());
    }
    let mut out = Vec::with_capacity(query.len());
    let mut i = 0usize;
    for &word in query {
        while i < words.len() && words[i] < word {
            i += 1;
        }
        if i == words.len() || words[i] != word {
            return Err(format!("word {word} is missing from cache"));
        }
        out.push(values[i]);
    }
    Ok(out)
}

pub fn apply_sparse(
    words: &[u32],
    values: &[u32],
    move_words: &[u32],
    move_masks: &[u32],
) -> Result<Vec<u32>, String> {
    if words.len() != values.len() {
        return Err("word and value arrays must have matching lengths".to_string());
    }
    check_move(move_words, move_masks)?;
    let mut out = values.to_vec();
    let mut j = 0usize;
    for (i, &word) in words.iter().enumerate() {
        while j < move_words.len() && move_words[j] < word {
            j += 1;
        }
        if j < move_words.len() && move_words[j] == word {
            out[i] ^= move_masks[j];
        }
    }
    Ok(out)
}

pub fn blocks_for_words(
    words: &[u32],
    word_ends: &[u32],
    widths: &[u8],
) -> Result<(Vec<(usize, usize, u8)>, usize), String> {
    if word_ends.len() != widths.len() {
        return Err("word_ends and widths must have the same length".to_string());
    }
    if words.is_empty() {
        return Ok((Vec::new(), 0));
    }
    let mut bits_by_word = Vec::with_capacity(words.len());
    for &word in words {
        let spec = word_ends.partition_point(|&end| end <= word);
        if spec >= word_ends.len() {
            return Err(format!("word {word} is outside the weight layout"));
        }
        let bits = widths[spec];
        if bits != 4 && bits != 8 {
            return Err(format!("weight word width must be 4 or 8, got {bits}"));
        }
        bits_by_word.push(bits);
    }
    let mut blocks = Vec::new();
    let mut dimension = 0usize;
    let mut start = 0usize;
    while start < bits_by_word.len() {
        let bits = bits_by_word[start];
        let mut end = start + 1;
        while end < bits_by_word.len() && bits_by_word[end] == bits {
            end += 1;
        }
        let length = (end - start) * (32 / usize::from(bits));
        blocks.push((dimension, length, bits));
        dimension += length;
        start = end;
    }
    Ok((blocks, dimension))
}

pub fn draw_sparse(
    count: usize,
    size: usize,
    dimension: u64,
    parameter_starts: &[u64],
    parameter_ends: &[u64],
    word_offsets: &[u32],
    widths: &[u8],
    seed: u64,
) -> Result<Vec<(Vec<u32>, Vec<u32>)>, String> {
    if count == 0 || size == 0 || dimension == 0 {
        return Err("count, size, and dimension must be positive".to_string());
    }
    let n = parameter_ends.len();
    if parameter_starts.len() != n || word_offsets.len() != n || widths.len() != n || n == 0 {
        return Err("weight layout arrays must have the same positive length".to_string());
    }
    for i in 0..n {
        if parameter_starts[i] >= parameter_ends[i] {
            return Err("layout parameter ranges must be nonempty".to_string());
        }
        if i > 0 && parameter_starts[i] < parameter_ends[i - 1] {
            return Err("layout parameter ranges must be sorted and nonoverlapping".to_string());
        }
        if widths[i] != 4 && widths[i] != 8 {
            return Err(format!("weight width must be 4 or 8, got {}", widths[i]));
        }
    }
    if parameter_starts[0] != 0 || parameter_ends[n - 1] != dimension {
        return Err("layout parameter ranges must cover the full metric dimension".to_string());
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut rows = Vec::with_capacity(count);
    for _ in 0..count {
        let mut pairs = Vec::<(u32, u32)>::with_capacity(size);
        for _ in 0..size {
            let parameter = rng.gen_range(0..dimension);
            let spec = parameter_ends.partition_point(|&end| end <= parameter);
            let local = parameter - parameter_starts[spec];
            let width = u32::from(widths[spec]);
            let fields_per_word = 32 / width;
            let word = word_offsets[spec] + (local as u32 / fields_per_word);
            let bit = (local as u32 % fields_per_word) * width;
            let code = rng.gen_range(1..(1u32 << width));
            pairs.push((word, code << bit));
        }
        pairs.sort_unstable_by_key(|&(word, _)| word);
        let mut words = Vec::new();
        let mut masks = Vec::new();
        for (word, mask) in pairs {
            if words.last().copied() == Some(word) {
                let last = masks.last_mut().expect("last mask exists");
                *last ^= mask;
                if *last == 0 {
                    words.pop();
                    masks.pop();
                }
            } else {
                words.push(word);
                masks.push(mask);
            }
        }
        rows.push((words, masks));
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_xor_cancels_matching_words() {
        let (words, masks) = sparse_xor(&[1, 3], &[7, 4], &[3, 5], &[4, 2]).unwrap();
        assert_eq!(words, vec![1, 5]);
        assert_eq!(masks, vec![7, 2]);
    }

    #[test]
    fn weight_ucb_prefers_near_good_observation() {
        let obs = [0u8, 7, 15];
        let cand = [1u8, 6];
        let blocks = [WeightBlock::new(0, 2, 4, 1.0, 1.0, 1.0).unwrap()];
        let result = select_weights(
            &obs,
            3,
            &[0.0, 10.0, -2.0],
            &cand,
            2,
            &blocks,
            WeightSelectConfig {
                neighbors: 1,
                epistemic_scale: 0.7,
                aleatoric_scale: 0.05,
                y_scale: 1.0,
                beta: 0.0,
                acquisition: AcquisitionKind::Ucb,
                seed: 0,
                backend: ComputeBackend::Cpu,
            },
        )
        .unwrap();
        assert_eq!(result.index, 1);
        assert_eq!(result.score, 10.0);
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn agx_weight_selection_matches_cpu() {
        let observations = [0u8, 7, 15];
        let candidates = [1u8, 6];
        let outcomes = [0.0, 10.0, -2.0];
        let blocks = [WeightBlock::new(0, 2, 4, 1.0, 1.0, 1.0).unwrap()];
        let config = WeightSelectConfig {
            neighbors: 1,
            epistemic_scale: 0.7,
            aleatoric_scale: 0.05,
            y_scale: 1.0,
            beta: 0.0,
            acquisition: AcquisitionKind::Ucb,
            seed: 0,
            backend: ComputeBackend::Cpu,
        };
        let cpu =
            select_weights(&observations, 3, &outcomes, &candidates, 2, &blocks, config).unwrap();
        let agx = select_weights(
            &observations,
            3,
            &outcomes,
            &candidates,
            2,
            &blocks,
            WeightSelectConfig {
                backend: ComputeBackend::Agx,
                ..config
            },
        )
        .unwrap();
        assert_eq!(agx.index, cpu.index);
        assert!((agx.score - cpu.score).abs() <= 1e-5);
    }
}
