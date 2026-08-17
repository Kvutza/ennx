use super::*;

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

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn agx_match() {
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
    let agx = select_weights(
        &observations,
        3,
        &outcomes,
        &candidates,
        2,
        &blocks,
        WeightSelectConfig {
            device: ComputeDevice::Agx,
            ..config
        },
    )
    .unwrap();
    assert_eq!(agx.index, cpu.index);
    assert!((agx.score - cpu.score).abs() <= 1e-5);
}
