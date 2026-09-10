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
fn thompson_ziggurat() {
    let draws = thompson_draws(4, 7);
    assert_eq!(
        draws
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        [3213202411, 3205735870, 3207992515, 1039019314]
    );
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
