use std::cell::RefCell;
use std::ptr;
use std::rc::Rc;

use opencl3::command_queue::CommandQueue;
use opencl3::context::Context as ClContext;
use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_CPU, CL_DEVICE_TYPE_GPU};
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY};
use opencl3::program::Program;
use opencl3::types::{CL_BLOCKING, CL_NON_BLOCKING};

use super::DenseView;
use crate::dense::DenseTerm;

const SOURCE: &str = concat!(include_str!("../ops.cl"), "\n", include_str!("linear.cl"));
const THREADS: usize = 256;

#[repr(C)]
#[derive(Clone, Copy)]
struct Params {
    rows: u32,
    columns: u32,
    has_bias: u32,
    term_count: u32,
    weight_key: u64,
    weight_start: u64,
    bias_key: u64,
    bias_start: u64,
    weight_scale: f32,
    bias_scale: f32,
    pad0: u32,
    pad1: u32,
}

struct Context {
    context: ClContext,
    queue: CommandQueue,
    kernel: Kernel,
}

pub(super) struct Resident {
    inner: Rc<Context>,
    weight: Buffer<f32>,
    bias: Buffer<f32>,
    rows: usize,
    columns: usize,
    has_bias: bool,
    weight_view: DenseView,
    bias_view: DenseView,
}

thread_local! {
    static CONTEXT: RefCell<Option<Rc<Context>>> = const { RefCell::new(None) };
}

pub(super) fn linear(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    weight_view: DenseView,
    bias_view: Option<DenseView>,
    terms: &[DenseTerm],
    rows: usize,
) -> Result<Vec<f32>, String> {
    context()?.linear(input, weight, bias, weight_view, bias_view, terms, rows)
}

fn context() -> Result<Rc<Context>, String> {
    CONTEXT.with(|cell| {
        if cell.borrow().is_none() {
            *cell.borrow_mut() = Some(Rc::new(Context::new()?));
        }
        Ok(Rc::clone(
            cell.borrow()
                .as_ref()
                .expect("OpenCL dense linear context initialized"),
        ))
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
            .map_err(|error| format!("failed to create OpenCL dense linear context: {error}"))?;
        let queue = CommandQueue::create_default(&context, 0)
            .map_err(|error| format!("failed to create OpenCL dense linear queue: {error}"))?;
        let program = Program::create_and_build_from_source(&context, SOURCE, "")
            .map_err(|error| format!("failed to build OpenCL dense linear: {error}"))?;
        let kernel = Kernel::create(&program, "dense_linear")
            .map_err(|error| format!("missing OpenCL dense linear: {error}"))?;
        Ok(Self {
            context,
            queue,
            kernel,
        })
    }

    fn linear(
        &self,
        input: &[f32],
        weight: &[f32],
        bias: Option<&[f32]>,
        weight_view: DenseView,
        bias_view: Option<DenseView>,
        terms: &[DenseTerm],
        rows: usize,
    ) -> Result<Vec<f32>, String> {
        let no_bias = [0.0f32];
        let bias_values = bias.unwrap_or(&no_bias);
        let bias_view = bias_view.unwrap_or(DenseView {
            key: 0,
            start: 0,
            scale: 1.0,
        });
        let params = Params {
            rows: u32::try_from(rows).map_err(|_| "dense linear rows exceed u32")?,
            columns: u32::try_from(input.len()).map_err(|_| "dense linear columns exceed u32")?,
            has_bias: u32::from(bias.is_some()),
            term_count: u32::try_from(terms.len())
                .map_err(|_| "dense linear term count exceeds u32")?,
            weight_key: weight_view.key,
            weight_start: weight_view.start,
            bias_key: bias_view.key,
            bias_start: bias_view.start,
            weight_scale: weight_view.scale,
            bias_scale: bias_view.scale,
            pad0: 0,
            pad1: 0,
        };
        let mut input_buffer = buffer::<f32>(&self.context, input.len(), CL_MEM_READ_ONLY)?;
        let mut weight_buffer = buffer::<f32>(&self.context, weight.len(), CL_MEM_READ_ONLY)?;
        let mut bias_buffer = buffer::<f32>(&self.context, bias_values.len(), CL_MEM_READ_ONLY)?;
        let mut term_buffer = buffer::<DenseTerm>(&self.context, terms.len(), CL_MEM_READ_ONLY)?;
        let out_buffer = buffer::<f32>(&self.context, rows, CL_MEM_WRITE_ONLY)?;

        unsafe {
            self.queue
                .enqueue_write_buffer(&mut input_buffer, CL_NON_BLOCKING, 0, input, &[])
                .map_err(|error| format!("failed to write OpenCL dense linear input: {error}"))?;
            self.queue
                .enqueue_write_buffer(&mut weight_buffer, CL_NON_BLOCKING, 0, weight, &[])
                .map_err(|error| format!("failed to write OpenCL dense linear weight: {error}"))?;
            self.queue
                .enqueue_write_buffer(&mut bias_buffer, CL_NON_BLOCKING, 0, bias_values, &[])
                .map_err(|error| format!("failed to write OpenCL dense linear bias: {error}"))?;
            self.queue
                .enqueue_write_buffer(&mut term_buffer, CL_NON_BLOCKING, 0, terms, &[])
                .map_err(|error| format!("failed to write OpenCL dense linear terms: {error}"))?;
            ExecuteKernel::new(&self.kernel)
                .set_arg(&input_buffer)
                .set_arg(&weight_buffer)
                .set_arg(&bias_buffer)
                .set_arg(&term_buffer)
                .set_arg(&out_buffer)
                .set_arg(&params)
                .set_global_work_size(
                    rows.checked_mul(THREADS)
                        .ok_or("OpenCL dense linear work size overflow")?,
                )
                .set_local_work_size(THREADS)
                .enqueue_nd_range(&self.queue)
                .map_err(|error| format!("failed to launch OpenCL dense linear: {error}"))?;
        }

        let mut out = vec![0.0; rows];
        unsafe {
            self.queue
                .enqueue_read_buffer(&out_buffer, CL_BLOCKING, 0, &mut out, &[])
                .map_err(|error| format!("failed to read OpenCL dense linear: {error}"))?;
        }
        Ok(out)
    }
}

impl Resident {
    pub(super) fn new(
        weight: &[f32],
        columns: usize,
        bias: Option<&[f32]>,
        weight_view: DenseView,
        bias_view: Option<DenseView>,
    ) -> Result<Self, String> {
        let inner = context()?;
        let no_bias = [0.0f32];
        let bias_values = bias.unwrap_or(&no_bias);
        let mut weight_buffer = buffer::<f32>(&inner.context, weight.len(), CL_MEM_READ_ONLY)?;
        let mut bias_buffer = buffer::<f32>(&inner.context, bias_values.len(), CL_MEM_READ_ONLY)?;
        unsafe {
            inner
                .queue
                .enqueue_write_buffer(&mut weight_buffer, CL_BLOCKING, 0, weight, &[])
                .map_err(|error| {
                    format!("failed to write resident OpenCL dense linear weight: {error}")
                })?;
            inner
                .queue
                .enqueue_write_buffer(&mut bias_buffer, CL_BLOCKING, 0, bias_values, &[])
                .map_err(|error| {
                    format!("failed to write resident OpenCL dense linear bias: {error}")
                })?;
        }
        Ok(Self {
            inner,
            weight: weight_buffer,
            bias: bias_buffer,
            rows: weight.len() / columns,
            columns,
            has_bias: bias.is_some(),
            weight_view,
            bias_view: bias_view.unwrap_or(DenseView {
                key: 0,
                start: 0,
                scale: 1.0,
            }),
        })
    }

    pub(super) fn eval(&mut self, input: &[f32], terms: &[DenseTerm]) -> Result<Vec<f32>, String> {
        let params = Params {
            rows: u32::try_from(self.rows).map_err(|_| "dense linear rows exceed u32")?,
            columns: u32::try_from(self.columns).map_err(|_| "dense linear columns exceed u32")?,
            has_bias: u32::from(self.has_bias),
            term_count: u32::try_from(terms.len())
                .map_err(|_| "dense linear term count exceeds u32")?,
            weight_key: self.weight_view.key,
            weight_start: self.weight_view.start,
            bias_key: self.bias_view.key,
            bias_start: self.bias_view.start,
            weight_scale: self.weight_view.scale,
            bias_scale: self.bias_view.scale,
            pad0: 0,
            pad1: 0,
        };
        let mut input_buffer = buffer::<f32>(&self.inner.context, input.len(), CL_MEM_READ_ONLY)?;
        let mut term_buffer =
            buffer::<DenseTerm>(&self.inner.context, terms.len(), CL_MEM_READ_ONLY)?;
        let out_buffer = buffer::<f32>(&self.inner.context, self.rows, CL_MEM_WRITE_ONLY)?;
        unsafe {
            self.inner
                .queue
                .enqueue_write_buffer(&mut input_buffer, CL_NON_BLOCKING, 0, input, &[])
                .map_err(|error| {
                    format!("failed to write resident OpenCL dense linear input: {error}")
                })?;
            self.inner
                .queue
                .enqueue_write_buffer(&mut term_buffer, CL_NON_BLOCKING, 0, terms, &[])
                .map_err(|error| {
                    format!("failed to write resident OpenCL dense linear terms: {error}")
                })?;
            ExecuteKernel::new(&self.inner.kernel)
                .set_arg(&input_buffer)
                .set_arg(&self.weight)
                .set_arg(&self.bias)
                .set_arg(&term_buffer)
                .set_arg(&out_buffer)
                .set_arg(&params)
                .set_global_work_size(
                    self.rows
                        .checked_mul(THREADS)
                        .ok_or("OpenCL dense linear work size overflow")?,
                )
                .set_local_work_size(THREADS)
                .enqueue_nd_range(&self.inner.queue)
                .map_err(|error| {
                    format!("failed to launch resident OpenCL dense linear: {error}")
                })?;
        }
        let mut out = vec![0.0; self.rows];
        unsafe {
            self.inner
                .queue
                .enqueue_read_buffer(&out_buffer, CL_BLOCKING, 0, &mut out, &[])
                .map_err(|error| format!("failed to read resident OpenCL dense linear: {error}"))?;
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
            .map_err(|error| format!("failed to allocate OpenCL dense linear buffer: {error}"))
    }
}
