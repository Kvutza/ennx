use std::cell::RefCell;
use std::ptr;

use opencl3::command_queue::CommandQueue;
use opencl3::context::Context as ClContext;
use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_CPU, CL_DEVICE_TYPE_GPU};
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY};
use opencl3::program::Program;
use opencl3::types::{CL_BLOCKING, CL_NON_BLOCKING};

use super::{tiles, DenseLeaf, DenseTerm, DenseTile};

const SOURCE: &str = concat!(include_str!("ops.cl"), "\n", include_str!("dense.cl"));
const THREADS: usize = 256;

struct Context {
    context: ClContext,
    queue: CommandQueue,
    kernel: Kernel,
}

thread_local! {
    static CONTEXT: RefCell<Option<Context>> = const { RefCell::new(None) };
}

pub(super) fn apply(
    base: &[f32],
    leaves: &[DenseLeaf],
    terms: &[DenseTerm],
) -> Result<Vec<f32>, String> {
    CONTEXT.with(|cell| {
        if cell.borrow().is_none() {
            *cell.borrow_mut() = Some(Context::new()?);
        }
        cell.borrow()
            .as_ref()
            .expect("OpenCL dense context initialized")
            .apply(base, leaves, terms)
    })
}

impl Context {
    fn new() -> Result<Self, String> {
        let device_id = get_all_devices(CL_DEVICE_TYPE_GPU)
            .map_err(|error| format!("failed to enumerate OpenCL GPU devices: {error}"))?
            .into_iter()
            .next()
            .or_else(|| {
                get_all_devices(CL_DEVICE_TYPE_CPU)
                    .ok()
                    .and_then(|devices| devices.into_iter().next())
            })
            .ok_or("no OpenCL GPU or CPU device found")?;
        let device = Device::new(device_id);
        let context = ClContext::from_device(&device)
            .map_err(|error| format!("failed to create OpenCL dense context: {error}"))?;
        let queue = CommandQueue::create_default(&context, 0)
            .map_err(|error| format!("failed to create OpenCL dense queue: {error}"))?;
        let program = Program::create_and_build_from_source(&context, SOURCE, "")
            .map_err(|error| format!("failed to build OpenCL dense kernel: {error}"))?;
        let kernel = Kernel::create(&program, "apply_dense")
            .map_err(|error| format!("missing OpenCL dense kernel: {error}"))?;
        Ok(Self {
            context,
            queue,
            kernel,
        })
    }

    fn apply(
        &self,
        base: &[f32],
        leaves: &[DenseLeaf],
        terms: &[DenseTerm],
    ) -> Result<Vec<f32>, String> {
        let tiles = tiles(leaves)?;
        let mut base_buffer = buffer::<f32>(&self.context, base.len(), CL_MEM_READ_ONLY)?;
        let mut leaf_buffer = buffer::<DenseLeaf>(&self.context, leaves.len(), CL_MEM_READ_ONLY)?;
        let mut term_buffer = buffer::<DenseTerm>(&self.context, terms.len(), CL_MEM_READ_ONLY)?;
        let mut tile_buffer = buffer::<DenseTile>(&self.context, tiles.len(), CL_MEM_READ_ONLY)?;
        let out_buffer = buffer::<f32>(&self.context, base.len(), CL_MEM_WRITE_ONLY)?;
        let term_count = u32::try_from(terms.len()).map_err(|_| "dense term count exceeds u32")?;

        unsafe {
            self.queue
                .enqueue_write_buffer(&mut base_buffer, CL_NON_BLOCKING, 0, base, &[])
                .map_err(|error| format!("failed to write OpenCL dense base: {error}"))?;
            self.queue
                .enqueue_write_buffer(&mut leaf_buffer, CL_NON_BLOCKING, 0, leaves, &[])
                .map_err(|error| format!("failed to write OpenCL dense leaves: {error}"))?;
            self.queue
                .enqueue_write_buffer(&mut term_buffer, CL_NON_BLOCKING, 0, terms, &[])
                .map_err(|error| format!("failed to write OpenCL dense terms: {error}"))?;
            self.queue
                .enqueue_write_buffer(&mut tile_buffer, CL_NON_BLOCKING, 0, &tiles, &[])
                .map_err(|error| format!("failed to write OpenCL dense tiles: {error}"))?;
            ExecuteKernel::new(&self.kernel)
                .set_arg(&base_buffer)
                .set_arg(&leaf_buffer)
                .set_arg(&term_buffer)
                .set_arg(&tile_buffer)
                .set_arg(&out_buffer)
                .set_arg(&term_count)
                .set_global_work_size(
                    tiles
                        .len()
                        .checked_mul(THREADS)
                        .ok_or("OpenCL dense work size overflow")?,
                )
                .set_local_work_size(THREADS)
                .enqueue_nd_range(&self.queue)
                .map_err(|error| format!("failed to launch OpenCL dense kernel: {error}"))?;
        }

        let mut out = vec![0.0; base.len()];
        unsafe {
            self.queue
                .enqueue_read_buffer(&out_buffer, CL_BLOCKING, 0, &mut out, &[])
                .map_err(|error| format!("failed to read OpenCL dense result: {error}"))?;
        }
        Ok(out)
    }
}

fn buffer<T>(
    context: &ClContext,
    len: usize,
    flags: opencl3::types::cl_mem_flags,
) -> Result<Buffer<T>, String> {
    unsafe {
        Buffer::create(context, flags, len, ptr::null_mut())
            .map_err(|error| format!("failed to allocate OpenCL dense buffer: {error}"))
    }
}
