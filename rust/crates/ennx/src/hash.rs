//! Hash-based deterministic random number generation.
//!
//! This module provides deterministic RNG functions using SplitMix64
//! and Box-Muller transform for generating standard normal variates.

use ndarray::{Array, ArrayD, IxDyn};
use rayon::prelude::*;
use std::collections::HashMap;
use thiserror::Error;

use bpann::parallel::use_rayon;

/// Errors that can occur during hash-based RNG.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum HashError {
    /// Invalid number of metrics.
    #[error("num_metrics must be positive, got {0}")]
    InvalidNumMetrics(i64),
    /// Dimension mismatch in inputs.
    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),
}

/// SplitMix64 algorithm constants.
const SM64_RATIO: u64 = 0x9E3779B97F4A7C15;
const SM64_1: u64 = 0xBF58476D1CE4E5B9;
const SM64_2: u64 = 0x94D049BB133111EB;
const SM64_OFFSET: u64 = 0xD2B74407B1CE6E93;

/// Prime constant for seed combination.
const SEED_PRIME: u64 = 1_000_003;

/// 2^53 as f64 constant.
const INV_2P53: f64 = 1.0 / 9007199254740992.0;

/// Minimum clip value to avoid log(0).
const CLIP_MIN: f64 = 1e-12;

/// Maximum clip value (exclusive upper bound).
const CLIP_MAX: f64 = 1.0 - 1e-12;

/// SplitMix64 hash function.
///
/// This is a fast, high-quality hash function for 64-bit integers.
/// The implementation must match the Python version exactly for parity.
#[inline(always)]
pub fn splitmix64(x: u64) -> u64 {
    let x = x.wrapping_add(SM64_RATIO);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(SM64_1);
    z = (z ^ (z >> 27)).wrapping_mul(SM64_2);
    z ^ (z >> 31)
}

/// Convert u64 to f64 in [0, 1) using high 53 bits.
#[inline(always)]
pub fn u64_f53(x: u64) -> f64 {
    ((x >> 11) as f64) * INV_2P53
}

/// Box-Muller transform for standard normal variates.
///
/// Takes two uniform [0, 1) values and produces one standard normal.
#[inline(always)]
pub fn box_muller(u1: f64, u2: f64) -> f64 {
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

#[inline(always)]
pub fn normal_metric(seed_u64: u64, unique_idx: i64, metric: usize) -> f64 {
    let unique_u64 = unique_idx as u64;
    let base =
        (seed_u64.wrapping_mul(SEED_PRIME).wrapping_add(unique_u64)).wrapping_mul(SEED_PRIME);
    let metric_u64 = metric as u64;
    let combined1 = base.wrapping_add(metric_u64);
    let r1 = splitmix64(combined1);
    let combined2 = combined1 ^ SM64_OFFSET;
    let r2 = splitmix64(combined2);
    let mut u1 = u64_f53(r1);
    let u2 = u64_f53(r2);
    u1 = u1.clamp(CLIP_MIN, CLIP_MAX);
    box_muller(u1, u2)
}

/// Build sorted-unique neighbor indices and an inverse map for `data_indices`.
pub fn unique_inverse(data_indices: &[i64]) -> (Vec<i64>, Vec<usize>) {
    let unique_indices: Vec<i64> = {
        let mut v = data_indices.to_vec();
        v.sort_unstable();
        v.dedup();
        v
    };
    let unique_pos: HashMap<i64, usize> = unique_indices
        .iter()
        .enumerate()
        .map(|(pos, &value)| (value, pos))
        .collect();
    let inverse: Vec<usize> = data_indices
        .iter()
        .map(|&idx| {
            *unique_pos
                .get(&idx)
                .expect("all data_indices must exist in unique_indices")
        })
        .collect();
    (unique_indices, inverse)
}

/// Fast hash-based RNG for multiple seeds, indices, and metrics.
///
/// This is the optimized SplitMix64 + Box-Muller version that matches
/// the Python `normal_hash` function.
///
/// Seeds are processed in parallel; each seed's outputs are bit-identical to the
/// serial implementation (deterministic per `(seed, index, metric)`).
///
/// # Arguments
///
/// * `function_seeds` - Array of int64 seed values
/// * `data_indices` - Array of int indices (arbitrary shape)
/// * `num_metrics` - Number of metrics to generate per (seed, index) pair
///
/// # Returns
///
/// Array of shape (num_seeds, *data_indices.shape, num_metrics) containing
/// standard normal variates.
///
/// # Errors
///
/// Returns `HashError::InvalidNumMetrics` if num_metrics <= 0.
pub fn normal_hash(
    function_seeds: &[i64],
    data_indices: &[i64],
    num_metrics: i64,
) -> Result<ArrayD<f64>, HashError> {
    if num_metrics <= 0 {
        return Err(HashError::InvalidNumMetrics(num_metrics));
    }

    let num_seeds = function_seeds.len();
    let num_indices = data_indices.len();
    let num_metrics = num_metrics as usize;
    let row_len = num_indices * num_metrics;

    let (unique_indices, inverse) = unique_inverse(data_indices);

    let mut flat = vec![0.0f64; num_seeds * row_len];

    if use_rayon() {
        // Flat buffer (num_seeds, num_indices, num_metrics) in C order; fill by seed in parallel.
        flat.par_chunks_mut(row_len)
            .zip(function_seeds.par_iter())
            .for_each_init(
                || vec![0.0f64; unique_indices.len() * num_metrics],
                |unique_cache, (seed_out, &seed)| {
                    let seed_u64 = seed as u64;
                    for (ui, &unique_idx) in unique_indices.iter().enumerate() {
                        for metric in 0..num_metrics {
                            unique_cache[ui * num_metrics + metric] =
                                normal_metric(seed_u64, unique_idx, metric);
                        }
                    }
                    for (di, &inv) in inverse.iter().enumerate() {
                        let src = inv * num_metrics;
                        let dst = di * num_metrics;
                        seed_out[dst..dst + num_metrics]
                            .copy_from_slice(&unique_cache[src..src + num_metrics]);
                    }
                },
            );
    } else {
        let mut row_cache = vec![0.0f64; unique_indices.len() * num_metrics];
        for (seed_offset, &seed) in function_seeds.iter().enumerate() {
            let seed_u64 = seed as u64;
            for (ui, &unique_idx) in unique_indices.iter().enumerate() {
                for metric in 0..num_metrics {
                    row_cache[ui * num_metrics + metric] =
                        normal_metric(seed_u64, unique_idx, metric);
                }
            }
            let seed_out = &mut flat[seed_offset * row_len..(seed_offset + 1) * row_len];
            for (di, &inv) in inverse.iter().enumerate() {
                let src = inv * num_metrics;
                let dst = di * num_metrics;
                seed_out[dst..dst + num_metrics]
                    .copy_from_slice(&row_cache[src..src + num_metrics]);
            }
        }
    }

    let output = Array::from_shape_vec(IxDyn(&[num_seeds, num_indices, num_metrics]), flat)
        .expect("flat buffer size matches output shape");
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_001() {
        // Known test value from SplitMix64 reference
        let x = 0x123456789ABCDEF0u64;
        let result = splitmix64(x);
        // Just verify it doesn't panic and produces deterministic output
        let result2 = splitmix64(x);
        assert_eq!(result, result2);
    }

    #[test]
    fn test_u64tof53range() {
        // u64_f53 should produce values in [0, 1)
        for x in [0u64, u64::MAX, 0x123456789ABCDEF0] {
            let f = u64_f53(x);
            assert!(
                (0.0..1.0).contains(&f),
                "u64_f53({}) = {} out of range",
                x,
                f
            );
        }
    }

    #[test]
    fn test_boxmullerfinite() {
        // Box-Muller should produce finite values for valid inputs
        let n = box_muller(0.5, 0.5);
        assert!(n.is_finite());
    }

    #[test]
    fn test_hashdeterminism() {
        let seeds = vec![42i64];
        let indices = vec![0i64, 1i64];

        let result1 = normal_hash(&seeds, &indices, 2).unwrap();
        let result2 = normal_hash(&seeds, &indices, 2).unwrap();

        assert_eq!(result1, result2);
    }

    #[test]
    fn test_hashseedsensitivity() {
        let seeds1 = vec![42i64];
        let seeds2 = vec![99i64];
        let indices = vec![0i64, 1i64];

        let result1 = normal_hash(&seeds1, &indices, 2).unwrap();
        let result2 = normal_hash(&seeds2, &indices, 2).unwrap();

        assert_ne!(result1, result2);
    }

    #[test]
    fn test_invalidnummetrics() {
        let seeds = vec![42i64];
        let indices = vec![0i64];

        assert!(matches!(
            normal_hash(&seeds, &indices, 0),
            Err(HashError::InvalidNumMetrics(0))
        ));

        assert!(matches!(
            normal_hash(&seeds, &indices, -1),
            Err(HashError::InvalidNumMetrics(-1))
        ));
    }

    #[test]
    fn test_outputshape() {
        let seeds = vec![1i64, 2i64]; // 2 seeds
        let indices = vec![0i64, 1i64, 2i64]; // 3 indices
        let num_metrics = 4;

        let result = normal_hash(&seeds, &indices, num_metrics).unwrap();

        assert_eq!(result.shape(), &[2, 3, 4]);
    }

    fn normal_serial(
        function_seeds: &[i64],
        data_indices: &[i64],
        num_metrics: i64,
    ) -> Result<ArrayD<f64>, HashError> {
        // Serial reference matching the pre-rayon algorithm (for parity checks).
        if num_metrics <= 0 {
            return Err(HashError::InvalidNumMetrics(num_metrics));
        }
        let num_seeds = function_seeds.len();
        let num_indices = data_indices.len();
        let num_metrics = num_metrics as usize;
        let unique_indices: Vec<i64> = {
            let mut v = data_indices.to_vec();
            v.sort_unstable();
            v.dedup();
            v
        };
        let unique_pos: HashMap<i64, usize> = unique_indices
            .iter()
            .enumerate()
            .map(|(pos, &value)| (value, pos))
            .collect();
        let inverse: Vec<usize> = data_indices
            .iter()
            .map(|&idx| *unique_pos.get(&idx).expect("index present"))
            .collect();
        let mut output = Array::zeros(IxDyn(&[num_seeds, num_indices, num_metrics]));
        for (si, &seed) in function_seeds.iter().enumerate() {
            let seed_u64 = seed as u64;
            let mut unique_cache: Vec<Vec<f64>> = Vec::with_capacity(unique_indices.len());
            for &unique_idx in &unique_indices {
                let mut metric_values = Vec::with_capacity(num_metrics);
                for metric in 0..num_metrics {
                    metric_values.push(normal_metric(seed_u64, unique_idx, metric));
                }
                unique_cache.push(metric_values);
            }
            for (di, &inv) in inverse.iter().enumerate() {
                for metric in 0..num_metrics {
                    output[[si, di, metric]] = unique_cache[inv][metric];
                }
            }
        }
        Ok(output)
    }

    #[test]
    fn test_009() {
        let seeds: Vec<i64> = (0..64).collect();
        let indices = vec![0i64, 5, 5, 2, 9, 2];
        let parallel = normal_hash(&seeds, &indices, 2).unwrap();
        let serial = normal_serial(&seeds, &indices, 2).unwrap();
        assert_eq!(parallel, serial);
    }
}
