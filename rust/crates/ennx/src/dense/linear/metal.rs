use std::cell::RefCell;
use std::ffi::c_void;
use std::rc::Rc;
use std::sync::Arc;

extern crate metal as metal_crate;

use metal_crate::ComputePipelineState;

use super::DenseView;
use crate::apple_gpu::{thread_group, Runtime};
use crate::dense::DenseTerm;

const SOURCE: &str = concat!(
    include_str!("../ops.metal"),
    "\n",
    include_str!("linear.metal")
);
const THREADS: u64 = 256;

#[repr(C)]
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
    runtime: Arc<Runtime>,
    pipeline: ComputePipelineState,
}

pub(super) struct Resident {
    context: Rc<Context>,
    weight: metal_crate::Buffer,
    bias: metal_crate::Buffer,
    rows: usize,
    columns: usize,
    has_bias: bool,
    weight_view: DenseView,
    bias_view: DenseView,
}

thread_local! {
    static CONTEXT: RefCell<Option<Rc<Context>>> = const { RefCell::new(None) };
    static AGX_CONTEXT: RefCell<Option<Rc<Context>>> = const { RefCell::new(None) };
}

pub(super) fn linear(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    weight_view: DenseView,
    bias_view: Option<DenseView>,
    terms: &[DenseTerm],
    rows: usize,
    agx: bool,
) -> Result<Vec<f32>, String> {
    context(agx)?.linear(input, weight, bias, weight_view, bias_view, terms, rows)
}

fn context(agx: bool) -> Result<Rc<Context>, String> {
    let context = if agx { &AGX_CONTEXT } else { &CONTEXT };
    context.with(|cell| {
        if cell.borrow().is_none() {
            *cell.borrow_mut() = Some(Rc::new(Context::new(agx)?));
        }
        Ok(Rc::clone(
            cell.borrow()
                .as_ref()
                .expect("Metal dense linear context initialized"),
        ))
    })
}

impl Context {
    fn new(agx: bool) -> Result<Self, String> {
        let runtime = Runtime::shared()?;
        let pipeline = if agx {
            runtime.agx_pipeline(SOURCE, "dense-linear", "dense_linear")?
        } else {
            runtime.pipeline(SOURCE, "dense-linear", "dense_linear")?
        };
        Ok(Self { runtime, pipeline })
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
        let input_buffer = self.buffer(input);
        let weight_buffer = self.buffer(weight);
        let bias_buffer = self.buffer(bias_values);
        let term_buffer = self.buffer(terms);
        let out_buffer = self.runtime.buffer::<f32>(rows);

        let command = self.runtime.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_buffer(0, Some(&input_buffer), 0);
        encoder.set_buffer(1, Some(&weight_buffer), 0);
        encoder.set_buffer(2, Some(&bias_buffer), 0);
        encoder.set_buffer(3, Some(&term_buffer), 0);
        encoder.set_buffer(4, Some(&out_buffer), 0);
        encoder.set_bytes(
            5,
            size_of::<Params>() as u64,
            (&params as *const Params).cast::<c_void>(),
        );
        encoder.dispatch_thread_groups(thread_group(rows as u64), thread_group(THREADS));
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();

        Ok(unsafe {
            std::slice::from_raw_parts(out_buffer.contents().cast::<f32>(), rows).to_vec()
        })
    }

    fn buffer<T>(&self, values: &[T]) -> metal_crate::Buffer {
        self.runtime.buffer_with(values)
    }
}

impl Resident {
    pub(super) fn new(
        weight: &[f32],
        columns: usize,
        bias: Option<&[f32]>,
        weight_view: DenseView,
        bias_view: Option<DenseView>,
        agx: bool,
    ) -> Result<Self, String> {
        let context = context(agx)?;
        let no_bias = [0.0f32];
        let bias_values = bias.unwrap_or(&no_bias);
        let weight_buffer = context.buffer(weight);
        let bias_buffer = context.buffer(bias_values);
        Ok(Self {
            context,
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
        let input_buffer = self.context.buffer(input);
        let term_buffer = self.context.buffer(terms);
        let out_buffer = self.context.runtime.buffer::<f32>(self.rows);
        let command = self.context.runtime.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.context.pipeline);
        encoder.set_buffer(0, Some(&input_buffer), 0);
        encoder.set_buffer(1, Some(&self.weight), 0);
        encoder.set_buffer(2, Some(&self.bias), 0);
        encoder.set_buffer(3, Some(&term_buffer), 0);
        encoder.set_buffer(4, Some(&out_buffer), 0);
        encoder.set_bytes(
            5,
            size_of::<Params>() as u64,
            (&params as *const Params).cast::<c_void>(),
        );
        encoder.dispatch_thread_groups(thread_group(self.rows as u64), thread_group(THREADS));
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        Ok(unsafe {
            std::slice::from_raw_parts(out_buffer.contents().cast::<f32>(), self.rows).to_vec()
        })
    }
}
