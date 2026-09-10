use super::region::TrustRegion;
use crate::trust_region::TRLengthConfig;
use crate::weights::ComputeDevice;

fn compare(device: ComputeDevice) {
    for config in [
        TRLengthConfig::default(),
        TRLengthConfig::new(0.5, 0.25, 1.0),
        TRLengthConfig::new(f64::MIN_POSITIVE, f64::from_bits(1), 0.5),
    ] {
        let mut cpu = TrustRegion::new(4, 2, 0.0, config).unwrap();
        let mut gpu = TrustRegion::new(4, 2, 0.0, config).unwrap();
        gpu.attach(device).unwrap();
        let mut random = 71u64;
        for step in 0..320 {
            random ^= random << 13;
            random ^= random >> 7;
            random ^= random << 17;
            let value = match step {
                0..=8 => step as f32,
                9..=14 => [8.007, 8.008, 8.009, 8.01, 8.016, 8.017][step - 9],
                15..=90 => 0.0,
                91..=95 => f32::MAX,
                96..=100 => -f32::MAX,
                _ => f32::from_bits((random as u32) & 0xfeffffff),
            };
            let pending = usize::from(step % 7 != 0);
            if step % 13 == 0 {
                let length = gpu.length().unwrap().to_bits();
                gpu.prepare(f32::MAX, pending).unwrap();
                assert_eq!(gpu.length().unwrap().to_bits(), length);
            }
            assert_eq!(
                cpu.prepare(value, pending).unwrap(),
                gpu.prepare(value, pending).unwrap()
            );
            assert_eq!(
                cpu.finish(value, pending).unwrap(),
                gpu.finish(value, pending).unwrap()
            );
            assert_eq!(
                cpu.length().unwrap().to_bits(),
                gpu.length().unwrap().to_bits(),
                "length at step {step}"
            );
            assert_eq!(
                cpu.best().unwrap().to_bits(),
                gpu.best().unwrap().to_bits(),
                "best at step {step}"
            );
            assert_eq!(
                cpu.restarts().unwrap(),
                gpu.restarts().unwrap(),
                "restarts at step {step}"
            );
        }
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn metal_region() {
    compare(ComputeDevice::Metal);
}

#[cfg(feature = "opencl")]
#[test]
fn opencl_region() {
    let mut region = TrustRegion::new(4, 2, 0.0, TRLengthConfig::default()).unwrap();
    match region.attach(ComputeDevice::OpenCl) {
        Ok(()) => {}
        Err(error) if opencl_unavailable(&error) => return,
        Err(error) => panic!("{error}"),
    }
    compare(ComputeDevice::OpenCl);
}

#[cfg(feature = "opencl")]
fn opencl_unavailable(error: &str) -> bool {
    error.contains("OpenCL platform")
        || error.contains("OpenCL GPU")
        || error.contains("failed to enumerate OpenCL")
}
