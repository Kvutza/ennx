use super::{Ask, ComputeDevice, Optimizer, Parameter, TRLengthConfig};

fn compare(device: ComputeDevice) {
    let create = |device| {
        Optimizer::new_batch(
            &[8; 4],
            0.0,
            vec![Parameter::new(0, 4, 8, 0.1, 1.0, 0.3).unwrap()],
            16,
            device,
            2,
            TRLengthConfig::new(0.5, 0.25, 2.0),
            2,
        )
        .unwrap()
    };
    let mut cpu = create(ComputeDevice::Cpu);
    let mut gpu = create(device);
    let mut lengths = Vec::new();
    for step in 0..80 {
        // A host length that would produce entirely different perturbations if consumed.
        let config = Ask {
            length: if step % 2 == 0 { 0.0 } else { 1e20 },
            neighbors: 1,
            ..Ask::default()
        };
        let ask = |optimizer: &mut Optimizer| match step % 4 {
            0 => vec![optimizer.ask(&[step + 7], config).unwrap()],
            1 => vec![optimizer.ask_stream(step + 7, 1, config).unwrap()],
            2 => optimizer
                .ask_batch(&[step + 7, step + 11], 2, config)
                .unwrap(),
            _ => optimizer.batch_stream(step + 7, 2, 1, config).unwrap(),
        };
        let expected = ask(&mut cpu);
        let actual = ask(&mut gpu);
        for (left, right) in expected.into_iter().zip(actual) {
            assert_eq!(
                cpu.row(left).unwrap(),
                gpu.row(right).unwrap(),
                "row at step {step}"
            );
            let value = if step < 12 { step as f32 + 1.0 } else { 0.0 };
            assert_eq!(
                cpu.tell(left, value).unwrap(),
                gpu.tell(right, value).unwrap()
            );
        }
        assert_eq!(
            cpu.length().unwrap().to_bits(),
            gpu.length().unwrap().to_bits()
        );
        assert_eq!(cpu.restarts().unwrap(), gpu.restarts().unwrap());
        lengths.push(gpu.length().unwrap());
    }
    assert!(lengths.iter().any(|length| *length > 0.5));
    assert!(lengths.iter().any(|length| *length < 0.5));
    assert!(gpu.restarts().unwrap() > 0);
}

fn closed_loop(device: ComputeDevice) {
    let create = |device| {
        Optimizer::new_batch(
            &[8; 4],
            0.0,
            vec![Parameter::new(0, 4, 8, 0.1, 1.0, 0.3).unwrap()],
            4,
            device,
            2,
            TRLengthConfig::new(0.5, 0.25, 2.0),
            2,
        )
        .unwrap()
    };
    let mut cpu = create(ComputeDevice::Cpu);
    let mut gpu = create(device);
    let config = Ask {
        neighbors: 1,
        ..Ask::default()
    };
    assert!(gpu
        .ask_stream(
            7,
            4,
            Ask {
                neighbors: 2,
                ..config
            }
        )
        .is_err());
    for step in 0..200 {
        let expected = cpu.batch_stream(step + 7, 2, 4, config).unwrap();
        let actual = gpu.batch_stream(step + 7, 2, 4, config).unwrap();
        for (left, right) in expected.into_iter().zip(actual).rev() {
            assert_eq!(
                left.seed, right.seed,
                "selection at step {step}: CPU {left:?}, device {right:?}"
            );
            let value = if step < 12 { step as f32 + 1.0 } else { 0.0 };
            cpu.observe(left, value).unwrap();
            gpu.observe(right, value).unwrap();
            assert!(gpu.observe(right, value).is_err());
        }
        // No region, history, slot, or row telemetry is requested inside this loop.
    }
    let left = cpu.ask_stream(900, 8, config).unwrap();
    let right = gpu.ask_stream(900, 8, config).unwrap();
    assert_eq!(cpu.row(left).unwrap(), gpu.row(right).unwrap());
    assert_eq!(cpu.best().unwrap(), gpu.best().unwrap());
    assert_eq!(cpu.length().unwrap(), gpu.length().unwrap());
    assert_eq!(cpu.history_len().unwrap(), gpu.history_len().unwrap());
    assert_eq!(cpu.restarts().unwrap(), gpu.restarts().unwrap());
    assert!(gpu.restarts().unwrap() > 0);
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn metal_residency() {
    compare(ComputeDevice::Metal);
    closed_loop(ComputeDevice::Metal);
}

#[cfg(feature = "opencl")]
#[test]
fn opencl_residency() {
    compare(ComputeDevice::OpenCl);
    closed_loop(ComputeDevice::OpenCl);
}
