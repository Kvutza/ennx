use super::{Ask, ComputeDevice, DeviceView, Optimizer, Parameter, TRLengthConfig};
use crate::objective_observation::ObjectiveObservation;
use crate::traits::Oracle;

#[cfg(feature = "opencl")]
use super::opencl_unavailable;

fn read_view(view: DeviceView<'_>) -> Result<Vec<u8>, String> {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    if let Some((buffer, offset)) = view.as_metal() {
        return Ok(unsafe {
            std::slice::from_raw_parts(buffer.contents().cast::<u8>().add(offset), view.row_bytes())
                .to_vec()
        });
    }
    #[cfg(feature = "opencl")]
    if let Some((_, queue, buffer, offset)) = view.as_opencl() {
        let mut row = vec![0u8; view.row_bytes()];
        unsafe {
            queue
                .enqueue_read_buffer(buffer, opencl3::types::CL_BLOCKING, offset, &mut row, &[])
                .map_err(|error| error.to_string())?;
        }
        return Ok(row);
    }
    Err("test requires a Metal or OpenCL view".into())
}

enum TestOutcome {
    Error,
    Value(f32),
    Row(Vec<u8>, f32),
}

struct TestOracle(TestOutcome);

impl Oracle for TestOracle {
    type Evidence = f32;

    fn observe(
        &mut self,
        candidate: DeviceView<'_>,
    ) -> Result<(ObjectiveObservation, Self::Evidence), String> {
        let value = match &self.0 {
            TestOutcome::Error => return Err("evaluation failed".into()),
            TestOutcome::Value(value) => *value,
            TestOutcome::Row(expected, value) => {
                assert_eq!(read_view(candidate)?, *expected);
                *value
            }
        };
        Ok((ObjectiveObservation::scalar(value, 0.0)?, value))
    }
}

fn recycled_views(device: ComputeDevice) {
    let mut optimizer = Optimizer::new_batch(
        &[8; 4],
        0.0,
        vec![Parameter::new(0, 4, 8, 0.1, 1.0, 0.3).unwrap()],
        2,
        device,
        2,
        TRLengthConfig::new(0.5, 0.25, 2.0),
        2,
    )
    .unwrap();
    let config = Ask {
        neighbors: 1,
        ..Ask::default()
    };
    for step in 0..20 {
        let trials = optimizer.batch_stream(step + 9, 2, 3, config).unwrap();
        let rows = trials
            .iter()
            .map(|trial| optimizer.row(*trial).unwrap())
            .collect::<Vec<_>>();
        let views = optimizer.device_views(&trials).unwrap();
        for (view, expected) in views.into_iter().zip(&rows) {
            assert_eq!(read_view(view).unwrap(), *expected);
        }
        for (trial, expected) in trials.into_iter().zip(rows).rev() {
            let mut failed = TestOracle(TestOutcome::Error);
            assert!(optimizer.evaluate(trial, &mut failed).is_err());
            let mut invalid = TestOracle(TestOutcome::Value(f32::NAN));
            assert!(optimizer.evaluate(trial, &mut invalid).is_err());
            assert_eq!(optimizer.row(trial).unwrap(), expected);
            let mut valid = TestOracle(TestOutcome::Row(expected, step as f32));
            let (_, value) = optimizer.evaluate(trial, &mut valid).unwrap();
            assert_eq!(value, step as f32);
            assert!(optimizer.device_view(trial).is_err());
        }
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn metal_views() {
    recycled_views(ComputeDevice::Metal);
}

#[cfg(feature = "opencl")]
#[test]
fn opencl_views() {
    match Optimizer::new_batch(
        &[8; 4],
        0.0,
        vec![Parameter::new(0, 4, 8, 0.1, 1.0, 0.3).unwrap()],
        2,
        ComputeDevice::OpenCl,
        2,
        TRLengthConfig::new(0.5, 0.25, 2.0),
        2,
    ) {
        Ok(_) => {}
        Err(error) if opencl_unavailable(&error) => return,
        Err(error) => panic!("{error}"),
    }
    recycled_views(ComputeDevice::OpenCl);
}
