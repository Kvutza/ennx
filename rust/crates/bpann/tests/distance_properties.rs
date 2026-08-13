//! Deterministic property-style coverage for BPANN distance helpers.
//!
//! These tests are intentionally Antithesis-shaped without depending on an
//! external simulator: fixed corpus seeds, adversarial generated workloads,
//! independent scalar oracles, and failure messages that include replay state.

use bpann::distance::{bpann_row_to_f32, l2_sq_f32, row_sq_l2};
use ndarray::ArrayView1;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

const CORPUS_SEEDS: &[u64] = &[
    0x4449_5354_0000_0001,
    0x4449_5354_0000_0002,
    0x4449_5354_0000_0003,
    0x4449_5354_0000_0004,
];

const ADVERSARIAL_LENGTHS: &[usize] = &[
    0, 1, 2, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255,
];

fn scalar_sq_l2_f32_as_f64(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = f64::from(x) - f64::from(y);
            d * d
        })
        .sum()
}

fn scalar_sq_l2_f64(a: &[f64], b: &[f64], scale_x: Option<&[f64]>) -> f64 {
    match scale_x {
        Some(scale) => a
            .iter()
            .zip(b.iter())
            .zip(scale.iter())
            .map(|((&x, &y), &s)| {
                let d = x / s - y / s;
                d * d
            })
            .sum(),
        None => a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| {
                let d = x - y;
                d * d
            })
            .sum(),
    }
}

fn finite_vec_f32(rng: &mut ChaCha8Rng, len: usize, scale: f32) -> Vec<f32> {
    (0..len).map(|_| rng.gen_range(-scale..=scale)).collect()
}

fn finite_vec_f64(rng: &mut ChaCha8Rng, len: usize, scale: f64) -> Vec<f64> {
    (0..len).map(|_| rng.gen_range(-scale..=scale)).collect()
}

fn close_f32_reduction(candidate: f32, reference: f64, len: usize) -> bool {
    let tolerance = 1.0e-4 * (1.0 + len as f64);
    let absolute_error = (f64::from(candidate) - reference).abs();
    absolute_error <= tolerance * reference.abs().max(1.0)
}

fn close_f64(candidate: f64, reference: f64) -> bool {
    (candidate - reference).abs() <= 1.0e-10 * reference.abs().max(1.0)
}

#[test]
fn l2_sq_f32_obeys_distance_laws_under_generated_workloads() {
    for &seed in CORPUS_SEEDS {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        for &len in ADVERSARIAL_LENGTHS {
            for case in 0..32 {
                let scale = rng.gen_range(1.0e-3f32..=1.0e3);
                let a = finite_vec_f32(&mut rng, len, scale);
                let b = finite_vec_f32(&mut rng, len, scale);
                let got = l2_sq_f32(&a, &b);
                let reversed = l2_sq_f32(&b, &a);
                let reference = scalar_sq_l2_f32_as_f64(&a, &b);

                assert!(
                    got.is_finite() && got >= 0.0,
                    "distance must be finite and non-negative seed={seed} len={len} case={case} got={got}"
                );
                assert_eq!(
                    got, reversed,
                    "distance must be symmetric seed={seed} len={len} case={case}"
                );
                assert_eq!(
                    l2_sq_f32(&a, &a),
                    0.0,
                    "self-distance must be zero seed={seed} len={len} case={case}"
                );
                assert!(
                    close_f32_reduction(got, reference, len),
                    "distance diverged from scalar oracle seed={seed} len={len} case={case} got={got} reference={reference}"
                );
            }
        }
    }
}

#[test]
fn row_sq_l2_matches_scalar_oracle_with_and_without_scaling() {
    for &seed in CORPUS_SEEDS {
        let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0x5343_414c_455f_5831);
        for len in [1usize, 2, 3, 8, 17, 64, 129] {
            for case in 0..24 {
                let a = finite_vec_f64(&mut rng, len, 100.0);
                let b = finite_vec_f64(&mut rng, len, 100.0);
                let scale: Vec<f64> = (0..len).map(|_| rng.gen_range(0.01f64..=100.0)).collect();

                let unscaled = row_sq_l2(
                    ArrayView1::from(&a),
                    ArrayView1::from(&b),
                    false,
                    ArrayView1::from(&scale),
                );
                let scaled = row_sq_l2(
                    ArrayView1::from(&a),
                    ArrayView1::from(&b),
                    true,
                    ArrayView1::from(&scale),
                );
                let unscaled_reference = scalar_sq_l2_f64(&a, &b, None);
                let scaled_reference = scalar_sq_l2_f64(&a, &b, Some(&scale));

                assert!(
                    close_f64(unscaled, unscaled_reference),
                    "unscaled row distance mismatch seed={seed} len={len} case={case} got={unscaled} reference={unscaled_reference}"
                );
                assert!(
                    close_f64(scaled, scaled_reference),
                    "scaled row distance mismatch seed={seed} len={len} case={case} got={scaled} reference={scaled_reference}"
                );
                assert!(
                    unscaled >= 0.0 && scaled >= 0.0,
                    "row distances must be non-negative seed={seed} len={len} case={case}"
                );
            }
        }
    }
}

#[test]
fn bpann_row_to_f32_applies_the_same_scaling_as_row_sq_l2() {
    for &seed in CORPUS_SEEDS {
        let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0x524f_575f_4633_3200);
        for len in [1usize, 4, 8, 17, 65] {
            for case in 0..24 {
                let row = finite_vec_f64(&mut rng, len, 25.0);
                let query = finite_vec_f64(&mut rng, len, 25.0);
                let scale: Vec<f64> = (0..len).map(|_| rng.gen_range(0.05f64..=20.0)).collect();

                let mut encoded_row = Vec::new();
                bpann_row_to_f32(&row, true, &scale, &mut encoded_row);
                let encoded_query: Vec<f32> = query
                    .iter()
                    .zip(scale.iter())
                    .map(|(&q, &s)| (q / s) as f32)
                    .collect();

                let encoded_distance = l2_sq_f32(&encoded_query, &encoded_row);
                let reference = row_sq_l2(
                    ArrayView1::from(&query),
                    ArrayView1::from(&row),
                    true,
                    ArrayView1::from(&scale),
                );

                assert!(
                    close_f32_reduction(encoded_distance, reference, len),
                    "scaled f32 encoding mismatch seed={seed} len={len} case={case} got={encoded_distance} reference={reference}"
                );
            }
        }
    }
}
