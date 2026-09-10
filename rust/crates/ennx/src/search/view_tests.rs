use super::{Ask, ComputeDevice, DeviceView, Optimizer, Parameter, TRLengthConfig};

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
            assert!(optimizer
                .evaluate(trial, |_| Err("evaluation failed".into()))
                .is_err());
            assert!(optimizer.evaluate(trial, |_| Ok(f32::NAN)).is_err());
            assert_eq!(optimizer.row(trial).unwrap(), expected);
            let value = optimizer
                .evaluate(trial, |view| {
                    assert_eq!(read_view(view)?, expected);
                    Ok(step as f32)
                })
                .unwrap();
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

#[cfg(feature = "opencl")]
fn opencl_unavailable(error: &str) -> bool {
    error.contains("OpenCL platform")
        || error.contains("OpenCL GPU")
        || error.contains("failed to enumerate OpenCL")
        || error.contains("CL_INVALID_VALUE")
}
