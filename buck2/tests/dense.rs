use ennx::experimental::{
    apply_dense, dense_dist2, dense_linear, ComputeDevice, DenseLeaf, DenseTerm, DenseView,
};

fn input() -> (Vec<f32>, Vec<DenseLeaf>, Vec<DenseTerm>) {
    (
        vec![0.5, -1.0, 2.0, 0.25, 4.0, -2.0, 0.75, -0.125],
        vec![
            DenseLeaf::new(11, 0, 4, 0.5).unwrap(),
            DenseLeaf::new(29, 4, 4, 1.25).unwrap(),
        ],
        vec![
            DenseTerm::new(0x1234_5678_9abc_def0, 0.01).unwrap(),
            DenseTerm::new(91, -0.0025).unwrap(),
        ],
    )
}

#[test]
fn dense_apply() {
    let (base, leaves, terms) = input();
    let result = apply_dense(&base, &leaves, &terms, ComputeDevice::Cpu).unwrap();
    assert_eq!(result.changed, base.len());
    assert!(base
        .iter()
        .zip(result.values)
        .all(|(left, right)| *left != right));

    let origin: Vec<DenseTerm> = Vec::new();
    assert!(dense_dist2(&leaves, &terms, &origin).unwrap() > 0.0);
}

fn linear(device: ComputeDevice) -> Vec<f32> {
    dense_linear(
        &[0.25, -0.5, 1.5, 2.0],
        &[0.5, -1.0, 0.75, 0.25, -0.5, 2.0, 1.25, -0.75],
        Some(&[0.125, -0.25]),
        DenseView::new(11, 0, 0.02).unwrap(),
        Some(DenseView::new(29, 0, 0.01).unwrap()),
        &[
            DenseTerm::new(0x1234_5678_9abc_def0, 0.5).unwrap(),
            DenseTerm::new(91, -0.125).unwrap(),
        ],
        device,
    )
    .unwrap()
}

#[test]
fn dense_weights() {
    assert_eq!(linear(ComputeDevice::Cpu).len(), 2);
}

#[cfg(target_os = "macos")]
#[test]
fn gpu_dense() {
    let (base, leaves, terms) = input();
    let cpu = apply_dense(&base, &leaves, &terms, ComputeDevice::Cpu).unwrap();
    let cpu_linear = linear(ComputeDevice::Cpu);
    let device = ComputeDevice::Metal;
    let gpu = apply_dense(&base, &leaves, &terms, device).unwrap();
    assert_eq!(gpu.changed, cpu.changed);
    for (left, right) in gpu.values.iter().zip(&cpu.values) {
        assert!((left - right).abs() <= f32::EPSILON);
    }
    for (left, right) in linear(device).iter().zip(&cpu_linear) {
        assert!((left - right).abs() <= 1.0e-5);
    }
}

#[cfg(not(target_os = "macos"))]
#[test]
fn opencl_exists() {
    let (base, leaves, terms) = input();
    let cpu = apply_dense(&base, &leaves, &terms, ComputeDevice::Cpu).unwrap();
    let opencl = match apply_dense(&base, &leaves, &terms, ComputeDevice::OpenCl) {
        Ok(result) => result,
        Err(error)
            if error.contains("no OpenCL GPU or CPU device")
                || error.contains("failed to enumerate OpenCL GPU devices") =>
        {
            return;
        }
        Err(error) => panic!("{error}"),
    };
    assert_eq!(opencl.changed, cpu.changed);
    for (left, right) in opencl.values.iter().zip(cpu.values) {
        assert!((left - right).abs() <= f32::EPSILON);
    }
    for (left, right) in linear(ComputeDevice::OpenCl)
        .iter()
        .zip(linear(ComputeDevice::Cpu))
    {
        assert!((left - right).abs() <= 1.0e-5);
    }
}
