use std::cell::RefCell;
use std::ptr;

use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_CPU, CL_DEVICE_TYPE_GPU};
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY};
use opencl3::program::Program;
use opencl3::types::{cl_mem_flags, CL_BLOCKING, CL_NON_BLOCKING};

use super::{
    acquisition_code, thompson_draws, AcquisitionKind, WeightBlock, WeightSelectConfig,
    WeightSelectResult,
};

const THREADS: usize = 256;
const MAX_NEIGHBORS: usize = 2048;

const SOURCE: &str = include_str!("weights.cl");

#[repr(C)]
#[derive(Clone, Copy)]
struct ClBlock {
    offset: u32,
    length: u32,
    bits: u32,
    quantization_scale: f32,
    metric_scale: f32,
    weight: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ClParams {
    row_bytes: u32,
    observations: u32,
    candidates: u32,
    blocks: u32,
    neighbors: u32,
    epistemic_scale: f32,
    aleatoric_scale: f32,
    y_scale: f32,
    beta: f32,
    acquisition: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ClBest {
    index: u32,
    score: f32,
}

struct OpenClCtx {
    context: Context,
    queue: CommandQueue,
    score_kernel: Kernel,
    select_kernel: Kernel,
}

thread_local! {
    static CTX: RefCell<Option<OpenClCtx>> = const { RefCell::new(None) };
}

pub(super) fn select(
    observations: &[u8],
    observation_count: usize,
    outcomes: &[f32],
    candidates: &[u8],
    candidate_count: usize,
    blocks: &[WeightBlock],
    row_bytes: usize,
    config: WeightSelectConfig,
) -> Result<WeightSelectResult, String> {
    if config.neighbors > MAX_NEIGHBORS {
        return Err(format!(
            "OpenCL quantized-weight ENN supports at most {MAX_NEIGHBORS} neighbors"
        ));
    }
    let cl_blocks = cl_blocks(blocks)?;
    CTX.with(|cell| {
        if cell.borrow().is_none() {
            *cell.borrow_mut() = Some(OpenClCtx::new()?);
        }
        let borrow = cell.borrow();
        let ctx = borrow.as_ref().expect("OpenCL context initialized");
        ctx.select(
            observations,
            observation_count,
            outcomes,
            candidates,
            candidate_count,
            &cl_blocks,
            row_bytes,
            config,
        )
    })
}

impl OpenClCtx {
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
        let context = Context::from_device(&device)
            .map_err(|error| format!("failed to create OpenCL context: {error}"))?;
        let queue = CommandQueue::create_default(&context, 0)
            .map_err(|error| format!("failed to create OpenCL command queue: {error}"))?;
        let program = Program::create_and_build_from_source(&context, SOURCE, "")
            .map_err(|error| format!("failed to build OpenCL quantized-weight kernels: {error}"))?;
        let score_kernel = Kernel::create(&program, "score_weight_neighbors")
            .map_err(|error| format!("missing OpenCL quantized-weight kernel: {error}"))?;
        let select_kernel = Kernel::create(&program, "select_best")
            .map_err(|error| format!("missing OpenCL select kernel: {error}"))?;
        Ok(Self {
            context,
            queue,
            score_kernel,
            select_kernel,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn select(
        &self,
        observations: &[u8],
        observation_count: usize,
        outcomes: &[f32],
        candidates: &[u8],
        candidate_count: usize,
        blocks: &[ClBlock],
        row_bytes: usize,
        config: WeightSelectConfig,
    ) -> Result<WeightSelectResult, String> {
        let mut obs_buffer = self.buffer::<u8>(observations.len(), CL_MEM_READ_ONLY)?;
        let mut outcome_buffer = self.buffer::<f32>(outcomes.len(), CL_MEM_READ_ONLY)?;
        let mut candidate_buffer = self.buffer::<u8>(candidates.len(), CL_MEM_READ_ONLY)?;
        let mut block_buffer = self.buffer::<ClBlock>(blocks.len(), CL_MEM_READ_ONLY)?;
        let score_buffer = self.buffer::<f32>(candidate_count, CL_MEM_READ_WRITE)?;
        let best_buffer = self.buffer::<ClBest>(1, CL_MEM_WRITE_ONLY)?;
        let draws = if config.acquisition == AcquisitionKind::Thompson {
            thompson_draws(observation_count, config.seed)
        } else {
            vec![0.0]
        };
        let mut draw_buffer = self.buffer::<f32>(draws.len(), CL_MEM_READ_ONLY)?;

        unsafe {
            self.queue
                .enqueue_write_buffer(&mut obs_buffer, CL_BLOCKING, 0, observations, &[])
                .map_err(|error| format!("failed to write OpenCL observation buffer: {error}"))?;
            self.queue
                .enqueue_write_buffer(&mut outcome_buffer, CL_BLOCKING, 0, outcomes, &[])
                .map_err(|error| format!("failed to write OpenCL outcome buffer: {error}"))?;
            self.queue
                .enqueue_write_buffer(&mut candidate_buffer, CL_BLOCKING, 0, candidates, &[])
                .map_err(|error| format!("failed to write OpenCL candidate buffer: {error}"))?;
            self.queue
                .enqueue_write_buffer(&mut block_buffer, CL_BLOCKING, 0, blocks, &[])
                .map_err(|error| format!("failed to write OpenCL block buffer: {error}"))?;
            self.queue
                .enqueue_write_buffer(&mut draw_buffer, CL_BLOCKING, 0, &draws, &[])
                .map_err(|error| format!("failed to write OpenCL Thompson draw buffer: {error}"))?;
        }

        let params = ClParams {
            row_bytes: to_u32(row_bytes, "row_bytes")?,
            observations: to_u32(observation_count, "observations")?,
            candidates: to_u32(candidate_count, "candidates")?,
            blocks: to_u32(blocks.len(), "blocks")?,
            neighbors: to_u32(config.neighbors, "neighbors")?,
            epistemic_scale: config.epistemic_scale,
            aleatoric_scale: config.aleatoric_scale,
            y_scale: config.y_scale,
            beta: config.beta,
            acquisition: acquisition_code(config.acquisition),
        };

        unsafe {
            ExecuteKernel::new(&self.score_kernel)
                .set_arg(&obs_buffer)
                .set_arg(&outcome_buffer)
                .set_arg(&candidate_buffer)
                .set_arg(&block_buffer)
                .set_arg(&score_buffer)
                .set_arg(&draw_buffer)
                .set_arg(&params)
                .set_global_work_size(candidate_count * THREADS)
                .set_local_work_size(THREADS)
                .enqueue_nd_range(&self.queue)
                .map_err(|error| {
                    format!("failed to launch OpenCL quantized-weight kernel: {error}")
                })?;
            ExecuteKernel::new(&self.select_kernel)
                .set_arg(&score_buffer)
                .set_arg(&best_buffer)
                .set_arg(&to_u32(candidate_count, "candidates")?)
                .set_global_work_size(THREADS)
                .set_local_work_size(THREADS)
                .enqueue_nd_range(&self.queue)
                .map_err(|error| format!("failed to launch OpenCL select kernel: {error}"))?;
            self.queue.finish().map_err(|error| {
                format!("failed to finish OpenCL quantized-weight kernel: {error}")
            })?;
        }

        let mut best = [ClBest {
            index: 0,
            score: f32::NEG_INFINITY,
        }];
        unsafe {
            self.queue
                .enqueue_read_buffer(&best_buffer, CL_NON_BLOCKING, 0, &mut best, &[])
                .map_err(|error| format!("failed to read OpenCL selected score: {error}"))?
                .wait()
                .map_err(|error| format!("failed waiting for OpenCL selected score: {error}"))?;
        }

        Ok(WeightSelectResult {
            index: best[0].index as usize,
            score: best[0].score,
        })
    }

    fn buffer<T>(&self, len: usize, flags: cl_mem_flags) -> Result<Buffer<T>, String> {
        unsafe {
            Buffer::<T>::create(&self.context, flags, len, ptr::null_mut())
                .map_err(|error| format!("failed to allocate OpenCL buffer: {error}"))
        }
    }
}

fn cl_blocks(blocks: &[WeightBlock]) -> Result<Vec<ClBlock>, String> {
    let mut out = Vec::with_capacity(blocks.len());
    for block in blocks {
        out.push(ClBlock {
            offset: to_u32(block.offset, "block offset")?,
            length: to_u32(block.length, "block length")?,
            bits: u32::from(block.bits),
            quantization_scale: block.quantization_scale,
            metric_scale: block.metric_scale,
            weight: block.weight,
        });
    }
    Ok(out)
}

fn to_u32(value: usize, name: &str) -> Result<u32, String> {
    u32::try_from(value).map_err(|_| format!("{name} exceeds u32 range"))
}
