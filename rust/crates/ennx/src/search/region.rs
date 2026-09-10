use ndarray::ArrayView1;

use crate::trust_region::{TRLengthConfig, TurboTrustRegion};
use crate::weights::ComputeDevice;

/// Adaptation state; independent of candidate buffers and execution backend.
pub struct TrustRegion {
    trust: TurboTrustRegion,
    dimensions: usize,
    num_pert: usize,
    outcomes: Vec<f64>,
    best: f32,
    restarts: usize,
    config: TRLengthConfig,
    device: Option<DeviceRegion>,
}

impl TrustRegion {
    pub fn new(
        dimensions: usize,
        num_pert: usize,
        base_value: f32,
        length: TRLengthConfig,
    ) -> Result<Self, String> {
        if dimensions == 0 || num_pert == 0 {
            return Err("parameter and perturbation counts must be positive".to_string());
        }
        if !base_value.is_finite() {
            return Err("base value must be finite".to_string());
        }
        if !length.length_min.is_finite()
            || !length.length_init.is_finite()
            || !length.length_max.is_finite()
            || length.length_min <= 0.0
            || length.length_min > length.length_init
            || length.length_init > length.length_max
        {
            return Err("region lengths must be finite, positive, and ordered".to_string());
        }
        let mut trust = TurboTrustRegion::new(dimensions, length);
        trust.set_arms(1);
        Ok(Self {
            trust,
            dimensions,
            num_pert,
            outcomes: vec![f64::from(base_value)],
            best: base_value,
            restarts: 0,
            config: length,
            device: None,
        })
    }

    /// Reading device telemetry may synchronize; proposal dispatch does not call this.
    pub fn length(&self) -> Result<f64, String> {
        match &self.device {
            Some(device) => device.length(),
            None => Ok(self.trust.length()),
        }
    }

    pub fn probability(&self) -> f64 {
        (self.num_pert as f64 / self.dimensions as f64).min(1.0)
    }

    pub fn best(&self) -> Result<f32, String> {
        if let Some(device) = &self.device {
            return Ok(f64::from_bits(device.decision()?.best) as f32);
        }
        Ok(self.best)
    }

    pub fn restarts(&self) -> Result<usize, String> {
        if let Some(device) = &self.device {
            return Ok(device.decision()?.restarts as usize);
        }
        Ok(self.restarts)
    }

    pub(super) fn accepted(&self) -> Result<bool, String> {
        Ok(self
            .device
            .as_ref()
            .ok_or("region is not resident")?
            .decision()?
            .accepted
            != 0)
    }

    pub(super) fn num_pert(&self) -> usize {
        self.num_pert
    }

    pub(super) fn dimensions(&self) -> usize {
        self.dimensions
    }

    pub(super) fn attach(&mut self, device: ComputeDevice) -> Result<(), String> {
        if matches!(device, ComputeDevice::Metal | ComputeDevice::OpenCl) {
            self.device = Some(DeviceRegion::new(
                device,
                Snapshot::new(self.dimensions, self.best, self.config),
            )?);
        }
        Ok(())
    }

    #[cfg(feature = "opencl")]
    pub(super) fn attach_context(&mut self, context: &Context) -> Result<(), String> {
        let state = Snapshot::new(self.dimensions, self.best, self.config);
        self.device = Some(DeviceRegion {
            engine: Engine::OpenCl(OpenClRegion::with_context(state, context)?),
            current: Decision::new(state.best),
            staged: Decision::new(state.best),
        });
        Ok(())
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(super) fn metal_buffer(&self) -> Option<::metal::Buffer> {
        match &self.device.as_ref()?.engine {
            Engine::Metal(engine) => Some(engine.current.clone()),
            #[allow(unreachable_patterns)]
            _ => None,
        }
    }

    #[cfg(feature = "opencl")]
    pub(super) fn opencl_buffer(&self) -> Option<std::sync::Arc<Buffer<u64>>> {
        match &self.device.as_ref()?.engine {
            Engine::OpenCl(engine) => Some(engine.current.clone()),
            #[allow(unreachable_patterns)]
            _ => None,
        }
    }

    pub(super) fn prepare(&mut self, value: f32, pending: usize) -> Result<bool, String> {
        match &mut self.device {
            Some(device) => device.prepare(value, pending),
            None => Ok(value > self.best),
        }
    }

    pub(super) fn finish(&mut self, value: f32, pending: usize) -> Result<bool, String> {
        if let Some(device) = &mut self.device {
            return Ok(device.commit());
        }
        self.best = self.best.max(value);
        self.outcomes.push(f64::from(value));
        self.trust
            .update(&ArrayView1::from(&self.outcomes), self.outcomes.len())
            .map_err(|error| error.to_string())?;
        let restarted = self.trust.needs_restart() && pending == 0;
        if restarted {
            self.trust.restart();
            self.restarts += 1;
        }
        Ok(restarted)
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(super) struct Snapshot {
    pub length: u64,
    pub initial: u64,
    pub minimum: u64,
    pub maximum: u64,
    pub significant: u64,
    pub lowest: u64,
    pub highest: u64,
    pub observations: u64,
    pub successes: u64,
    pub failures: u64,
    pub tolerance: u64,
    pub initialized: u64,
    pub best: u64,
    pub accepted: u64,
    pub restarted: u64,
    pub restarts: u64,
}

impl Snapshot {
    pub fn new(dimensions: usize, value: f32, config: TRLengthConfig) -> Self {
        let value = f64::from(value).to_bits();
        Self {
            length: config.length_init.to_bits(),
            initial: config.length_init.to_bits(),
            minimum: config.length_min.to_bits(),
            maximum: config.length_max.to_bits(),
            best: value,
            significant: value,
            lowest: value,
            highest: value,
            observations: 1,
            successes: 0,
            failures: 0,
            tolerance: dimensions.max(4).min(i32::MAX as usize) as u64,
            initialized: 0,
            accepted: 0,
            restarted: 0,
            restarts: 0,
        }
    }
}

enum Engine {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    Metal(MetalRegion),
    #[cfg(feature = "opencl")]
    OpenCl(OpenClRegion),
}

pub(super) struct DeviceRegion {
    engine: Engine,
    current: Decision,
    staged: Decision,
}

#[derive(Clone, Copy)]
struct Decision {
    best: u64,
    accepted: u64,
    restarted: u64,
    restarts: u64,
}

impl Decision {
    fn new(best: u64) -> Self {
        Self {
            best,
            accepted: 0,
            restarted: 0,
            restarts: 0,
        }
    }
}

impl DeviceRegion {
    fn decision(&self) -> Result<Decision, String> {
        match &self.engine {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Engine::Metal(engine) => {
                let state = unsafe { &*engine.current.contents().cast::<Snapshot>() };
                Ok(Decision {
                    best: state.best,
                    accepted: state.accepted,
                    restarted: state.restarted,
                    restarts: state.restarts,
                })
            }
            #[cfg(feature = "opencl")]
            Engine::OpenCl(engine) => {
                let mut words = [0u64; 4];
                unsafe {
                    engine.queue.enqueue_read_buffer(
                        engine.current.as_ref(),
                        CL_BLOCKING,
                        std::mem::offset_of!(Snapshot, best),
                        &mut words,
                        &[],
                    )
                }
                .map_err(|error| error.to_string())?;
                Ok(Decision {
                    best: words[0],
                    accepted: words[1],
                    restarted: words[2],
                    restarts: words[3],
                })
            }
            #[allow(unreachable_patterns)]
            _ => Err("region is not resident".into()),
        }
    }

    fn length(&self) -> Result<f64, String> {
        match &self.engine {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Engine::Metal(engine) => Ok(f64::from_bits(unsafe {
                engine.current.contents().cast::<u64>().read()
            })),
            #[cfg(feature = "opencl")]
            Engine::OpenCl(engine) => {
                let mut value = [0u64];
                unsafe {
                    engine.queue.enqueue_read_buffer(
                        engine.current.as_ref(),
                        CL_BLOCKING,
                        0,
                        &mut value,
                        &[],
                    )
                }
                .map_err(|error| error.to_string())?;
                Ok(f64::from_bits(value[0]))
            }
            #[allow(unreachable_patterns)]
            _ => Err("device region telemetry is unavailable".to_string()),
        }
    }

    #[allow(unused_variables, unreachable_code)]
    pub fn new(device: ComputeDevice, state: Snapshot) -> Result<Self, String> {
        let engine = match device {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            ComputeDevice::Metal => Engine::Metal(MetalRegion::new(state)?),
            #[cfg(feature = "opencl")]
            ComputeDevice::OpenCl => Engine::OpenCl(OpenClRegion::new(state)?),
            _ => return Err("device region adaptation is unavailable for this backend".to_string()),
        };
        Ok(Self {
            engine,
            current: Decision::new(state.best),
            staged: Decision::new(state.best),
        })
    }

    #[allow(unused_variables, unreachable_code)]
    pub fn prepare(&mut self, value: f32, pending: usize) -> Result<bool, String> {
        self.staged = match &mut self.engine {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Engine::Metal(engine) => engine.prepare(value, pending)?,
            #[cfg(feature = "opencl")]
            Engine::OpenCl(engine) => engine.prepare(value, pending)?,
            #[allow(unreachable_patterns)]
            _ => return Err("device region adaptation is unavailable".to_string()),
        };
        Ok(self.staged.accepted != 0)
    }

    pub fn commit(&mut self) -> bool {
        match &mut self.engine {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Engine::Metal(engine) => engine.commit(),
            #[cfg(feature = "opencl")]
            Engine::OpenCl(engine) => engine.commit(),
            #[allow(unreachable_patterns)]
            _ => {}
        }
        self.current = self.staged;
        self.current.restarted != 0
    }
}

#[cfg(any(all(target_os = "macos", feature = "metal"), feature = "opencl"))]
fn source(prefix: &str, entry: &str) -> String {
    format!(
        "{prefix}\n{}\n{}\n{entry}",
        include_str!("arithmetic.cl"),
        include_str!("adaptation.cl")
    )
}

#[cfg(all(target_os = "macos", feature = "metal"))]
use crate::apple_gpu::{thread_group, Runtime};
#[cfg(all(target_os = "macos", feature = "metal"))]
use std::sync::Arc;

#[cfg(all(target_os = "macos", feature = "metal"))]
const METAL_ENTRY: &str = r#"
kernel void adapt(const device Region *current [[buffer(0)]],
                  device Region *next [[buffer(1)]],
                  constant Word &value [[buffer(2)]],
                  constant Word &pending [[buffer(3)]]) {
    adapt_region(current, next, value, pending);
}
"#;

#[cfg(all(target_os = "macos", feature = "metal"))]
struct MetalRegion {
    runtime: Arc<Runtime>,
    pipeline: ::metal::ComputePipelineState,
    current: ::metal::Buffer,
    next: ::metal::Buffer,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl MetalRegion {
    pub fn new(state: Snapshot) -> Result<Self, String> {
        let runtime = Runtime::shared()?;
        let text = source(
            "#include <metal_stdlib>\nusing namespace metal;\n#define DEVICE device",
            METAL_ENTRY,
        );
        let pipeline = runtime.pipeline(&text, "region", "adapt")?;
        let current = runtime.buffer_with(&[state]);
        let next = runtime.buffer::<Snapshot>(1);
        Ok(Self {
            runtime,
            pipeline,
            current,
            next,
        })
    }

    pub fn prepare(&mut self, value: f32, pending: usize) -> Result<Decision, String> {
        let value = f64::from(value).to_bits();
        let pending = pending as u64;
        let command = self.runtime.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_buffer(0, Some(&self.current), 0);
        encoder.set_buffer(1, Some(&self.next), 0);
        encoder.set_bytes(2, size_of::<u64>() as u64, (&value as *const u64).cast());
        encoder.set_bytes(3, size_of::<u64>() as u64, (&pending as *const u64).cast());
        encoder.dispatch_thread_groups(thread_group(1), thread_group(1));
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        if command.status() == ::metal::MTLCommandBufferStatus::Error {
            return Err("Metal region update failed".to_string());
        }
        let state = unsafe { &*self.next.contents().cast::<Snapshot>() };
        Ok(Decision {
            best: state.best,
            accepted: state.accepted,
            restarted: state.restarted,
            restarts: state.restarts,
        })
    }

    pub fn commit(&mut self) {
        std::mem::swap(&mut self.current, &mut self.next);
    }
}

#[cfg(feature = "opencl")]
use opencl3::command_queue::CommandQueue;
#[cfg(feature = "opencl")]
use opencl3::context::Context;
#[cfg(feature = "opencl")]
use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_CPU, CL_DEVICE_TYPE_GPU};
#[cfg(feature = "opencl")]
use opencl3::kernel::{ExecuteKernel, Kernel};
#[cfg(feature = "opencl")]
use opencl3::memory::{Buffer, CL_MEM_READ_WRITE};
#[cfg(feature = "opencl")]
use opencl3::program::Program;
#[cfg(feature = "opencl")]
use opencl3::types::CL_BLOCKING;
#[cfg(feature = "opencl")]
use std::ptr;

#[cfg(feature = "opencl")]
const OPENCL_ENTRY: &str = r#"
__kernel void adapt(__global const Region *current, __global Region *next,
                    Word value, Word pending) {
    adapt_region(current, next, value, pending);
}
"#;

#[cfg(feature = "opencl")]
struct OpenClRegion {
    queue: CommandQueue,
    kernel: Kernel,
    current: std::sync::Arc<Buffer<u64>>,
    next: std::sync::Arc<Buffer<u64>>,
}

#[cfg(feature = "opencl")]
impl OpenClRegion {
    pub fn new(state: Snapshot) -> Result<Self, String> {
        let id = get_all_devices(CL_DEVICE_TYPE_GPU)
            .map_err(|error| format!("failed to enumerate OpenCL GPU devices: {error}"))?
            .into_iter()
            .next()
            .or_else(|| get_all_devices(CL_DEVICE_TYPE_CPU).ok()?.into_iter().next())
            .ok_or("no OpenCL GPU or CPU device found for region adaptation")?;
        let context = Context::from_device(&Device::new(id)).map_err(|error| error.to_string())?;
        Self::with_context(state, &context)
    }

    pub fn with_context(state: Snapshot, context: &Context) -> Result<Self, String> {
        let queue = CommandQueue::create_default(context, 0).map_err(|error| error.to_string())?;
        let program = Program::create_and_build_from_source(
            &context,
            &source("#define DEVICE __global", OPENCL_ENTRY),
            "",
        )
        .map_err(|error| error.to_string())?;
        let kernel = Kernel::create(&program, "adapt").map_err(|error| error.to_string())?;
        let mut current =
            unsafe { Buffer::create(context, CL_MEM_READ_WRITE, 16, ptr::null_mut()) }
                .map_err(|error| error.to_string())?;
        let next = unsafe { Buffer::create(context, CL_MEM_READ_WRITE, 16, ptr::null_mut()) }
            .map_err(|error| error.to_string())?;
        unsafe {
            queue.enqueue_write_buffer(
                &mut current,
                CL_BLOCKING,
                0,
                &std::mem::transmute::<Snapshot, [u64; 16]>(state),
                &[],
            )
        }
        .map_err(|error| error.to_string())?;
        Ok(Self {
            queue,
            kernel,
            current: std::sync::Arc::new(current),
            next: std::sync::Arc::new(next),
        })
    }

    pub fn prepare(&mut self, value: f32, pending: usize) -> Result<Decision, String> {
        let mut result = [0u64; 4];
        unsafe {
            ExecuteKernel::new(&self.kernel)
                .set_arg(self.current.as_ref())
                .set_arg(self.next.as_ref())
                .set_arg(&f64::from(value).to_bits())
                .set_arg(&(pending as u64))
                .set_global_work_size(1)
                .enqueue_nd_range(&self.queue)
                .map_err(|error| error.to_string())?;
            self.queue
                .enqueue_read_buffer(
                    self.next.as_ref(),
                    CL_BLOCKING,
                    12 * size_of::<u64>(),
                    &mut result,
                    &[],
                )
                .map_err(|error| error.to_string())?;
        }
        Ok(Decision {
            best: result[0],
            accepted: result[1],
            restarted: result[2],
            restarts: result[3],
        })
    }

    pub fn commit(&mut self) {
        std::mem::swap(&mut self.current, &mut self.next);
    }
}
