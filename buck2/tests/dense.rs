use ennx::experimental::{
    apply_dense, dense_dist2, dense_linear, ComputeBackend, DenseLeaf, DenseTerm, DenseView,
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
fn zig_changes_the_complete_pytree() {
    let (base, leaves, terms) = input();
    let result = apply_dense(&base, &leaves, &terms, ComputeBackend::Cpu).unwrap();
    assert_eq!(result.changed, base.len());
    assert!(base
        .iter()
        .zip(result.values)
        .all(|(left, right)| *left != right));

    let origin: Vec<DenseTerm> = Vec::new();
    assert!(dense_dist2(&leaves, &terms, &origin).unwrap() > 0.0);
}

fn linear(backend: ComputeBackend) -> Vec<f32> {
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
        backend,
    )
    .unwrap()
}

#[test]
fn zig_linear_consumes_procedural_weights() {
    assert_eq!(linear(ComputeBackend::Cpu).len(), 2);
}

#[cfg(target_os = "macos")]
#[test]
fn apple_gpu_matches_zig() {
    let (base, leaves, terms) = input();
    let zig = apply_dense(&base, &leaves, &terms, ComputeBackend::Cpu).unwrap();
    let zig_linear = linear(ComputeBackend::Cpu);
    for backend in [ComputeBackend::Metal, ComputeBackend::Agx] {
        let gpu = apply_dense(&base, &leaves, &terms, backend).unwrap();
        assert_eq!(gpu.changed, zig.changed);
        for (left, right) in gpu.values.iter().zip(&zig.values) {
            assert!((left - right).abs() <= f32::EPSILON);
        }
        for (left, right) in linear(backend).iter().zip(&zig_linear) {
            assert!((left - right).abs() <= 1.0e-5);
        }
    }
}

#[cfg(not(target_os = "macos"))]
#[test]
fn opencl_matches_zig_when_a_device_exists() {
    let (base, leaves, terms) = input();
    let zig = apply_dense(&base, &leaves, &terms, ComputeBackend::Cpu).unwrap();
    let opencl = match apply_dense(&base, &leaves, &terms, ComputeBackend::OpenCl) {
        Ok(result) => result,
        Err(error)
            if error.contains("no OpenCL GPU or CPU device")
                || error.contains("failed to enumerate OpenCL GPU devices") =>
        {
            return;
        }
        Err(error) => panic!("{error}"),
    };
    assert_eq!(opencl.changed, zig.changed);
    for (left, right) in opencl.values.iter().zip(zig.values) {
        assert!((left - right).abs() <= f32::EPSILON);
    }
    for (left, right) in linear(ComputeBackend::OpenCl)
        .iter()
        .zip(linear(ComputeBackend::Cpu))
    {
        assert!((left - right).abs() <= 1.0e-5);
    }
}
