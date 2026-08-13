#![cfg(all(target_os = "macos", feature = "metal"))]

use ennx::experimental::{KnnIndex, KnnPlan};
use ennx::IndexDriver;
use ndarray::{array, Array2};

#[test]
fn k_16() {
    let train = Array2::from_shape_fn((2050, 33), |(row, col)| {
        ((row * 17 + col * 11) % 1009) as f64 / 1009.0
    });
    let query = Array2::from_shape_fn((17, 33), |(row, col)| {
        ((row * 13 + col * 19 + 3) % 1013) as f64 / 1013.0
    });
    let exact = KnnIndex::new(&train.view(), IndexDriver::Exact).unwrap();
    let metal = KnnIndex::new(&train.view(), IndexDriver::Metal).unwrap();
    let expected = exact.search(&query.view(), 16).unwrap();
    let actual = metal.search(&query.view(), 16).unwrap();
    assert_eq!(actual.1, expected.1);
    for (actual, expected) in actual.0.iter().zip(expected.0.iter()) {
        assert!((actual - expected).abs() < 1.0e-4, "{actual} != {expected}");
    }
}

#[test]
fn k_2048() {
    let train = Array2::from_shape_fn((2050, 1), |(row, _)| row as f64);
    let query = array![[1024.25]];
    let exact = KnnIndex::new(&train.view(), IndexDriver::Exact).unwrap();
    let metal = KnnIndex::new(&train.view(), IndexDriver::Metal).unwrap();
    let expected = exact.search(&query.view(), 2048).unwrap();
    let actual = metal.search(&query.view(), 2048).unwrap();
    assert_eq!(actual.1, expected.1);
    for (actual, expected) in actual.0.iter().zip(expected.0.iter()) {
        assert!((actual - expected).abs() < 1.0e-4, "{actual} != {expected}");
    }
}

#[test]
fn wide() {
    let train = Array2::from_shape_fn((2050, 1), |(row, _)| row as f64);
    let query = array![[512.25], [1536.75]];
    let exact = KnnIndex::new(&train.view(), IndexDriver::Exact).unwrap();
    let metal = KnnIndex::with_plan(&train.view(), IndexDriver::Metal, KnnPlan::Wide).unwrap();
    let expected = exact.search(&query.view(), 1536).unwrap();
    let actual = metal.search(&query.view(), 1536).unwrap();
    assert_eq!(actual.1, expected.1);
    for (actual, expected) in actual.0.iter().zip(expected.0.iter()) {
        assert!((actual - expected).abs() < 1.0e-4, "{actual} != {expected}");
    }
}

#[test]
fn tree() {
    let train = Array2::from_shape_fn((65_538, 5), |(row, col)| {
        ((row * 23 + col * 29) % 65_521) as f64 / 65_521.0
    });
    let query = Array2::from_shape_fn((3, 5), |(row, col)| {
        ((row * 31 + col * 37 + 5) % 65_519) as f64 / 65_519.0
    });
    let exact = KnnIndex::new(&train.view(), IndexDriver::Exact).unwrap();
    let metal = KnnIndex::with_plan(&train.view(), IndexDriver::Metal, KnnPlan::Tree).unwrap();
    let expected = exact.search(&query.view(), 16).unwrap();
    let actual = metal.search(&query.view(), 16).unwrap();
    assert_eq!(actual.1, expected.1);
    for (actual, expected) in actual.0.iter().zip(expected.0.iter()) {
        assert!((actual - expected).abs() < 1.0e-4, "{actual} != {expected}");
    }
}

#[test]
fn gram() {
    let mut state = 59_u32;
    let train = Array2::from_shape_fn((2051, 257), |_| {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((state >> 8) as f32 / (1_u32 << 24) as f32) as f64
    });
    let query = Array2::from_shape_fn((11, 257), |_| {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((state >> 8) as f32 / (1_u32 << 24) as f32) as f64
    });
    let exact = KnnIndex::new(&train.view(), IndexDriver::Exact).unwrap();
    let metal = KnnIndex::with_plan(&train.view(), IndexDriver::Metal, KnnPlan::Gram).unwrap();
    let expected = exact.search(&query.view(), 16).unwrap();
    let actual = metal.search(&query.view(), 16).unwrap();
    for query in 0..query.nrows() {
        let mut actual_ids = actual.1.row(query).to_vec();
        let mut expected_ids = expected.1.row(query).to_vec();
        actual_ids.sort_unstable();
        expected_ids.sort_unstable();
        assert_eq!(actual_ids, expected_ids);
        for rank in 0..16 {
            let id = actual.1[[query, rank]];
            let expected_rank = expected
                .1
                .row(query)
                .iter()
                .position(|&candidate| candidate == id)
                .unwrap();
            let actual_distance = actual.0[[query, rank]];
            let expected_distance = expected.0[[query, expected_rank]];
            assert!(
                (actual_distance - expected_distance).abs() < 2.0e-4,
                "{actual_distance} != {expected_distance}"
            );
        }
    }
}

#[test]
fn q_1() {
    let train = Array2::from_shape_fn((1025, 65), |(row, col)| {
        ((row * 41 + col * 43) % 4093) as f64 / 4093.0
    });
    let query = Array2::from_shape_fn((1, 65), |(_, col)| (col * 47 % 4091) as f64 / 4091.0);
    let metal = KnnIndex::new(&train.view(), IndexDriver::Metal).unwrap();
    metal.search(&query.view(), 16).unwrap();
    assert_ne!(metal.plan(), "gram");
}
