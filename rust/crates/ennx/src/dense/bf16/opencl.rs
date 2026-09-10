use std::ptr;

use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_CPU, CL_DEVICE_TYPE_GPU};
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE};
use opencl3::program::Program;
use opencl3::types::{CL_BLOCKING, CL_NON_BLOCKING};

use super::{DenseLeaf, DenseTerm};
use crate::dense::{tiles, DenseTile};

const SOURCE: &str = concat!(include_str!("../ops.cl"), "\n", include_str!("bf16.cl"));
const THREADS: usize = 256;

const _: [(); 32] = [(); size_of::<DenseLeaf>()];
const _: [(); 16] = [(); size_of::<DenseTerm>()];
const _: [(); 16] = [(); size_of::<DenseTile>()];

pub(super) struct Resident {
    context: Context,
    queue: CommandQueue,
    kernel: Kernel,
    base: Buffer<u16>,
    candidate: Buffer<u16>,
    leaves: Buffer<DenseLeaf>,
    tiles: Buffer<DenseTile>,
    tile_count: usize,
    len: usize,
}

impl Resident {
    pub(super) fn new(base: &[u16], leaves: &[DenseLeaf]) -> Result<Self, String> {
        let device = device()?;
        let context = Context::from_device(&device)
            .map_err(|error| format!("failed to create OpenCL BF16 context: {error}"))?;
        let queue = CommandQueue::create_default(&context, 0)
            .map_err(|error| format!("failed to create OpenCL BF16 queue: {error}"))?;
        let program = Program::create_and_build_from_source(&context, SOURCE, "")
            .map_err(|error| format!("failed to build OpenCL BF16 kernel: {error}"))?;
        let kernel = Kernel::create(&program, "materialize_bf16")
            .map_err(|error| format!("missing OpenCL BF16 kernel: {error}"))?;
        let dense_tiles = tiles(leaves)?;
        let mut resident = Self {
            base: buffer(&context, base.len(), CL_MEM_READ_ONLY)?,
            candidate: buffer(&context, base.len(), CL_MEM_READ_WRITE)?,
            leaves: buffer(&context, leaves.len(), CL_MEM_READ_ONLY)?,
            tiles: buffer(&context, dense_tiles.len(), CL_MEM_READ_ONLY)?,
            tile_count: dense_tiles.len(),
            len: base.len(),
            context,
            queue,
            kernel,
        };
        unsafe {
            resident
                .queue
                .enqueue_write_buffer(&mut resident.base, CL_NON_BLOCKING, 0, base, &[])
                .map_err(|error| format!("failed to write OpenCL BF16 base: {error}"))?;
            resident
                .queue
                .enqueue_write_buffer(&mut resident.candidate, CL_NON_BLOCKING, 0, base, &[])
                .map_err(|error| format!("failed to write OpenCL BF16 candidate: {error}"))?;
            resident
                .queue
                .enqueue_write_buffer(&mut resident.leaves, CL_NON_BLOCKING, 0, leaves, &[])
                .map_err(|error| format!("failed to write OpenCL BF16 leaves: {error}"))?;
            resident
                .queue
                .enqueue_write_buffer(&mut resident.tiles, CL_NON_BLOCKING, 0, &dense_tiles, &[])
                .map_err(|error| format!("failed to write OpenCL BF16 tiles: {error}"))?;
        }
        resident
            .queue
            .finish()
            .map_err(|error| format!("failed to initialize OpenCL BF16 buffers: {error}"))?;
        Ok(resident)
    }

    pub(super) fn materialize(&mut self, terms: &[DenseTerm]) -> Result<(), String> {
        let mut term_buffer = buffer(&self.context, terms.len(), CL_MEM_READ_ONLY)?;
        let term_count = u32::try_from(terms.len()).map_err(|_| "BF16 term count exceeds u32")?;
        unsafe {
            self.queue
                .enqueue_write_buffer(&mut term_buffer, CL_NON_BLOCKING, 0, terms, &[])
                .map_err(|error| format!("failed to write OpenCL BF16 terms: {error}"))?;
            ExecuteKernel::new(&self.kernel)
                .set_arg(&self.base)
                .set_arg(&self.leaves)
                .set_arg(&self.tiles)
                .set_arg(&self.candidate)
                .set_arg(&term_buffer)
                .set_arg(&term_count)
                .set_global_work_size(
                    self.tile_count
                        .checked_mul(THREADS)
                        .ok_or("OpenCL BF16 work size overflow")?,
                )
                .set_local_work_size(THREADS)
                .enqueue_nd_range(&self.queue)
                .map_err(|error| format!("failed to launch OpenCL BF16 kernel: {error}"))?;
        }
        self.queue
            .finish()
            .map_err(|error| format!("failed to finish OpenCL BF16 kernel: {error}"))
    }

    pub(super) fn candidate(&self) -> Result<Vec<u16>, String> {
        let mut values = vec![0; self.len];
        unsafe {
            self.queue
                .enqueue_read_buffer(&self.candidate, CL_BLOCKING, 0, &mut values, &[])
                .map_err(|error| format!("failed to read OpenCL BF16 candidate: {error}"))?;
        }
        Ok(values)
    }
}

fn device() -> Result<Device, String> {
    let id = get_all_devices(CL_DEVICE_TYPE_GPU)
        .map_err(|error| format!("failed to enumerate OpenCL GPU devices: {error}"))?
        .into_iter()
        .next()
        .or_else(|| {
            get_all_devices(CL_DEVICE_TYPE_CPU)
                .ok()
                .and_then(|devices| devices.into_iter().next())
        })
        .ok_or("no OpenCL GPU or CPU device found")?;
    Ok(Device::new(id))
}

fn buffer<T>(
    context: &Context,
    len: usize,
    flags: opencl3::types::cl_mem_flags,
) -> Result<Buffer<T>, String> {
    unsafe {
        Buffer::create(context, flags, len, ptr::null_mut())
            .map_err(|error| format!("failed to allocate OpenCL BF16 buffer: {error}"))
    }
}
