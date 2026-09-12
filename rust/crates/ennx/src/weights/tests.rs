use super::*;

#[test]
fn removed_rejected() {
    assert!(ComputeDevice::parse("agx").is_err());
    assert!(ComputeDevice::parse("AGX").is_err());
    assert_eq!(ComputeDevice::parse("metal").unwrap(), ComputeDevice::Metal);
}

#[test]
fn xor_cancel() {
    let (words, masks) = sparse_xor(&[1, 3], &[7, 4], &[3, 5], &[4, 2]).unwrap();
    assert_eq!(words, vec![1, 5]);
    assert_eq!(masks, vec![7, 2]);
}

#[test]
fn ucb_prefer() {
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
            device: ComputeDevice::Cpu,
        },
    )
    .unwrap();
    assert_eq!(result.index, 1);
    assert_eq!(result.score, 10.0);
}

#[test]
fn pareto_mean() {
    let observations = [0u8, 0, 255];
    let candidates = [0u8, 255];
    let outcomes = [5.0, 5.0, 5.0];
    let blocks = [WeightBlock::new(0, 1, 8, 1.0, 1.0, 1.0).unwrap()];

    let result = select_weights(
        &observations,
        3,
        &outcomes,
        &candidates,
        2,
        &blocks,
        WeightSelectConfig {
            neighbors: 2,
            epistemic_scale: 1.0,
            aleatoric_scale: 0.0,
            y_scale: 1.0,
            beta: 0.0,
            acquisition: AcquisitionKind::Pareto,
            seed: 0,
            device: ComputeDevice::Cpu,
        },
    )
    .unwrap();

    assert_eq!(result.index, 1);
    assert!(result.score > 5.0);
}

#[test]
fn thompson_repeat() {
    let first = thompson_draws(4096, 0x1234_5678_9abc_def0);
    let second = thompson_draws(4096, 0x1234_5678_9abc_def0);
    assert_eq!(first, second);
    assert!(first.iter().all(|value| value.is_finite()));
}

#[test]
fn thompson_posterior_noise() {
    let draws = thompson_draws(4, 7);
    let posterior = crate::hash::normal_hash(&[7], &[0, 1, 2, 3], 1).unwrap();
    assert_eq!(
        draws,
        posterior.iter().map(|&z| z as f32).collect::<Vec<_>>()
    );
    assert_eq!(draws, thompson_draws(8, 7)[..4]);
    let history = [(3, 0.0), (1, 0.0), (0, 0.0)];
    assert_eq!(
        thompson_history_draws(&history, 7),
        [draws[3], draws[1], draws[0]]
    );
}

fn thompson_config() -> WeightSelectConfig {
    WeightSelectConfig {
        neighbors: 2,
        epistemic_scale: 0.7,
        aleatoric_scale: 0.05,
        y_scale: 1.0,
        beta: 0.0,
        acquisition: AcquisitionKind::Thompson,
        seed: 17,
        device: ComputeDevice::Cpu,
    }
}

#[test]
fn thompson_candidate_order_and_batches() {
    let observations = [0u8, 7, 15, 20];
    let outcomes = [0.0, 0.1, -0.2, 0.3];
    let candidates = [1u8, 6, 12];
    let blocks = [WeightBlock::new(0, 1, 8, 1.0, 1.0, 1.0).unwrap()];
    let select = |candidates: &[u8]| {
        select_weights(
            &observations,
            4,
            &outcomes,
            candidates,
            candidates.len(),
            &blocks,
            thompson_config(),
        )
        .unwrap()
    };
    let together = select(&candidates);
    let singles: Vec<_> = candidates
        .iter()
        .map(|candidate| select(&[*candidate]).score)
        .collect();
    assert_eq!(together.score, singles[together.index]);
    assert!(singles.iter().all(|&score| score <= together.score));
    let reversed = [12u8, 6, 1];
    let reverse_result = select(&reversed);
    assert_eq!(reverse_result.score, together.score);
    assert_eq!(reversed[reverse_result.index], candidates[together.index]);
    for candidate in candidates {
        let duplicate = select(&[candidate; 16]);
        assert_eq!(duplicate.index, 0);
        assert_eq!(duplicate.score, select(&[candidate]).score);
    }
}

#[test]
fn thompson_shared_neighbor_covariance() {
    let config = thompson_config();
    let outcomes = [0.0; 4];
    let mut means = [0.0f64; 3];
    let mut second_moments = [0.0f64; 3];
    let mut overlap = 0.0f64;
    let mut disjoint = 0.0f64;
    let count = 20_000;
    for seed in 0..count {
        let noise = thompson_draws(4, seed);
        let draws = [
            weighted_prediction(&[(1.0, 0), (1.0, 1)], &outcomes, config, &noise).draw,
            weighted_prediction(&[(1.0, 1), (1.0, 2)], &outcomes, config, &noise).draw,
            weighted_prediction(&[(1.0, 2), (1.0, 3)], &outcomes, config, &noise).draw,
        ]
        .map(f64::from);
        for i in 0..3 {
            means[i] += draws[i];
            second_moments[i] += draws[i] * draws[i];
        }
        overlap += draws[0] * draws[1];
        disjoint += draws[0] * draws[2];
    }
    let n = count as f64;
    for i in 0..3 {
        let mean = means[i] / n;
        let variance = second_moments[i] / n - mean * mean;
        assert!(mean.abs() < 0.03, "mean={mean}");
        assert!((variance - 1.0).abs() < 0.04, "variance={variance}");
    }
    let covariance = overlap / n - means[0] * means[1] / (n * n);
    assert!(
        (covariance - 0.5).abs() < 0.03,
        "overlap covariance={covariance}"
    );
    let covariance = disjoint / n - means[0] * means[2] / (n * n);
    assert!(covariance.abs() < 0.03, "disjoint covariance={covariance}");
}

#[test]
fn thompson_draw_normalization_handles_small_weights() {
    let config = WeightSelectConfig {
        epistemic_scale: 1.0,
        aleatoric_scale: 0.0,
        ..thompson_config()
    };
    let noise = thompson_draws(3, config.seed);
    let expected = (noise[0] as f64 + noise[2] as f64 / 2.0) / 1.25f64.sqrt();
    for distance in [1.0, 1.0e25, 1.0e35] {
        let prediction = weighted_prediction(
            &[(distance, 0), (2.0 * distance, 2)],
            &[0.0; 3],
            config,
            &noise,
        );
        assert!((prediction.draw as f64 - expected).abs() < 1e-6);
        assert!(scalar_score(prediction, config.acquisition, config.beta).is_finite());
    }
}

#[test]
fn thompson_moments() {
    let draws = thompson_draws(100_000, 0xfeed_beef);
    let n = draws.len() as f64;
    let mean = draws.iter().map(|&value| f64::from(value)).sum::<f64>() / n;
    let var = draws
        .iter()
        .map(|&value| {
            let delta = f64::from(value) - mean;
            delta * delta
        })
        .sum::<f64>()
        / n;
    assert!(mean.abs() < 0.01, "mean={mean}");
    assert!((var - 1.0).abs() < 0.02, "var={var}");
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn metal_unavailable4(error: &str) -> bool {
    error.contains("no default Metal device found")
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn metal_match() {
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
        device: ComputeDevice::Cpu,
    };
    let cpu = select_weights(&observations, 3, &outcomes, &candidates, 2, &blocks, config).unwrap();
    let metal = match select_weights(
        &observations,
        3,
        &outcomes,
        &candidates,
        2,
        &blocks,
        WeightSelectConfig {
            device: ComputeDevice::Metal,
            ..config
        },
    ) {
        Ok(result) => result,
        Err(error) if metal_unavailable4(&error) => return,
        Err(error) => panic!("Metal weight selection failed: {error}"),
    };
    assert_eq!(metal.index, cpu.index);
    assert!((metal.score - cpu.score).abs() <= 1e-5);
}

#[cfg(any(all(target_os = "macos", feature = "metal"), feature = "opencl"))]
fn gpu_posterior_matches_reference(device: ComputeDevice) -> Result<(), String> {
    const OBSERVATIONS: [u8; 4] = [15, 0, 9, 4];
    const OUTCOMES: [f32; 4] = [2.0, -1.0, 0.5, 4.0];

    fn reference_score(candidate: u8, config: WeightSelectConfig) -> f32 {
        let mut nearest: Vec<_> = OBSERVATIONS
            .iter()
            .enumerate()
            .map(|(row, &observation)| {
                let delta = f64::from(candidate) - f64::from(observation);
                (delta * delta, row)
            })
            .collect();
        nearest.sort_by(|a, b| a.partial_cmp(b).unwrap());
        nearest.truncate(config.neighbors);
        let weights: Vec<_> = nearest
            .iter()
            .map(|&(distance, _)| {
                1.0 / (1.0e-9
                    + f64::from(config.epistemic_scale) * distance
                    + f64::from(config.aleatoric_scale))
            })
            .collect();
        let weight_sum: f64 = weights.iter().sum();
        let mut mean = 0.0;
        let mut noise = 0.0;
        let mut norm_squared = 0.0;
        for ((_, row), weight) in nearest.iter().zip(weights) {
            let normalized_weight = weight / weight_sum;
            mean += weight / weight_sum.max(1.0e-12) * f64::from(OUTCOMES[*row]);
            noise += normalized_weight
                * f64::from(crate::hash::normal_metric(config.seed, *row as i64, 0) as f32);
            norm_squared += normalized_weight * normalized_weight;
        }
        let se = (1.0 / weight_sum.max(1.0e-12)).sqrt() * f64::from(config.y_scale);
        match config.acquisition {
            AcquisitionKind::Thompson => (mean + se / norm_squared.sqrt() * noise) as f32,
            AcquisitionKind::Ucb => (mean + f64::from(config.beta) * se) as f32,
            AcquisitionKind::Pareto => (mean + se) as f32,
        }
    }

    let check = |bits, candidates: &[u8], config: WeightSelectConfig| -> Result<(), String> {
        let blocks = [WeightBlock::new(0, 1, bits, 1.0, 1.0, 1.0)?];
        let mut expected = WeightSelectResult {
            index: 0,
            score: f32::NEG_INFINITY,
        };
        for (index, &candidate) in candidates.iter().enumerate() {
            let score = reference_score(candidate, config);
            if score > expected.score {
                expected = WeightSelectResult { index, score };
            }
        }
        let actual = select_weights(
            &OBSERVATIONS,
            OBSERVATIONS.len(),
            &OUTCOMES,
            candidates,
            candidates.len(),
            &blocks,
            config,
        )?;
        assert_eq!(actual.index, expected.index, "{config:?}, {candidates:?}");
        assert!(
            (actual.score - expected.score).abs() <= 2.0e-5 * expected.score.abs().max(1.0),
            "{config:?}, {candidates:?}: {actual:?} != {expected:?}"
        );
        Ok(())
    };
    let base_config = WeightSelectConfig {
        device,
        y_scale: 1.3,
        beta: 2.5,
        ..thompson_config()
    };
    // Fewer/more candidates than observations, reordering, and duplicate tie-breaking.
    let batches: &[&[u8]] = &[
        &[14],
        &[1],
        &[6],
        &[10],
        &[14, 1, 6, 10, 14, 6],
        &[6, 14, 10, 6, 1, 14],
        &[10; 6],
    ];
    for bits in [4, 8] {
        for neighbors in [1, 3] {
            for seed in [7, 0x1234_5678_9abc_def0] {
                for acquisition in [
                    AcquisitionKind::Thompson,
                    AcquisitionKind::Ucb,
                    AcquisitionKind::Pareto,
                ] {
                    let config = WeightSelectConfig {
                        neighbors,
                        seed,
                        acquisition,
                        ..base_config
                    };
                    for candidates in batches {
                        check(bits, candidates, config)?;
                    }
                }
            }
        }
    }
    // Raw inverse-variance weights have squares below the f32 subnormal range.
    check(
        8,
        &[6],
        WeightSelectConfig {
            neighbors: 3,
            epistemic_scale: 1.0e25,
            aleatoric_scale: 1.0e25,
            seed: 7,
            ..base_config
        },
    )
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn metal_materialized_posterior_is_observation_indexed() {
    match gpu_posterior_matches_reference(ComputeDevice::Metal) {
        Ok(()) => {}
        Err(error) if metal_unavailable4(&error) => {
            eprintln!("skipping Metal posterior test: {error}")
        }
        Err(error) => panic!("{error}"),
    }
}

#[cfg(feature = "opencl")]
#[test]
fn opencl_materialized_posterior_is_observation_indexed() {
    match gpu_posterior_matches_reference(ComputeDevice::OpenCl) {
        Ok(()) => {}
        Err(error)
            if error.contains("no OpenCL GPU or CPU device found")
                || error.contains("CL_PLATFORM_NOT_FOUND_KHR") =>
        {
            eprintln!("skipping OpenCL posterior test: {error}");
        }
        Err(error) => panic!("{error}"),
    }
}
