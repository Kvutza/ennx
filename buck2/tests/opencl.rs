use opencl3::context::Context;
use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_ALL};
use opencl3::program::Program;

const SOURCE: &str = r#"
__kernel void add_one(__global float *values) {
    const size_t index = get_global_id(0);
    values[index] += 1.0f;
}
"#;

#[test]
fn opencl_kernel() {
    let device_id = match get_all_devices(CL_DEVICE_TYPE_ALL) {
        Ok(devices) => match devices.into_iter().next() {
            Some(device) => device,
            None => {
                eprintln!("no OpenCL device is available; skipping runtime check");
                return;
            }
        },
        Err(error) => {
            eprintln!("no OpenCL platform is available ({error}); skipping runtime check");
            return;
        }
    };
    let context = Context::from_device(&Device::new(device_id)).expect("create OpenCL context");
    Program::create_and_build_from_source(&context, SOURCE, "").expect("compile OpenCL kernel");
}
