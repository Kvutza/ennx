use std::cell::RefCell;
use std::ffi::c_void;
use std::rc::Rc;
use std::sync::Arc;

extern crate metal as metal_crate;

use metal_crate::ComputePipelineState;

use super::{DenseLeaf, DenseTerm};
use crate::apple_gpu::{thread_group, Runtime};
use crate::dense::{tiles, DenseTile};

const SOURCE: &str = concat!(
    include_str!("../ops.metal"),
    "\n",
    include_str!("bf16.metal")
);
const THREADS: u64 = 256;
const MAX_TERM_BYTES: usize = 4_096;

const _: [(); 32] = [(); size_of::<DenseLeaf>()];
const _: [(); 16] = [(); size_of::<DenseTerm>()];
const _: [(); 16] = [(); size_of::<DenseTile>()];

struct Context {
    runtime: Arc<Runtime>,
    pipeline: ComputePipelineState,
}

pub(super) struct Resident {
    context: Rc<Context>,
    base: metal_crate::Buffer,
    candidate: metal_crate::Buffer,
    leaves: metal_crate::Buffer,
    tiles: metal_crate::Buffer,
    tile_count: usize,
    len: usize,
}

thread_local! {
    static CONTEXT: RefCell<Option<Rc<Context>>> = const { RefCell::new(None) };
    static AGX_CONTEXT: RefCell<Option<Rc<Context>>> = const { RefCell::new(None) };
}

impl Resident {
    pub(super) fn new(base: &[u16], leaves: &[DenseLeaf], agx: bool) -> Result<Self, String> {
        let context = context(agx)?;
        let dense_tiles = tiles(leaves)?;
        Ok(Self {
            base: context.runtime.buffer_with(base),
            candidate: context.runtime.buffer_with(base),
            leaves: context.runtime.buffer_with(leaves),
            tiles: context.runtime.buffer_with(&dense_tiles),
            tile_count: dense_tiles.len(),
            len: base.len(),
            context,
        })
    }

    pub(super) fn materialize(&mut self, terms: &[DenseTerm]) -> Result<(), String> {
        let bytes = std::mem::size_of_val(terms);
        if bytes > MAX_TERM_BYTES {
            return Err(format!(
                "BF16 pytree terms require {bytes} bytes, maximum is {MAX_TERM_BYTES}"
            ));
        }
        let term_count = u32::try_from(terms.len()).map_err(|_| "BF16 term count exceeds u32")?;
        let command = self.context.runtime.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.context.pipeline);
        encoder.set_buffer(0, Some(&self.base), 0);
        encoder.set_buffer(1, Some(&self.leaves), 0);
        encoder.set_buffer(2, Some(&self.tiles), 0);
        encoder.set_buffer(3, Some(&self.candidate), 0);
        encoder.set_bytes(4, bytes as u64, terms.as_ptr().cast::<c_void>());
        encoder.set_bytes(
            5,
            size_of::<u32>() as u64,
            (&term_count as *const u32).cast::<c_void>(),
        );
        encoder.dispatch_thread_groups(thread_group(self.tile_count as u64), thread_group(THREADS));
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        let status = command.status();
        if status == metal_crate::MTLCommandBufferStatus::Completed {
            Ok(())
        } else {
            Err(format!("BF16 Metal command failed: {status:?}"))
        }
    }

    pub(super) fn candidate(&self) -> Vec<u16> {
        unsafe {
            std::slice::from_raw_parts(self.candidate.contents().cast::<u16>(), self.len).to_vec()
        }
    }

    pub(super) fn buffer(&self) -> &metal_crate::Buffer {
        &self.candidate
    }
}

fn context(agx: bool) -> Result<Rc<Context>, String> {
    let context = if agx { &AGX_CONTEXT } else { &CONTEXT };
    context.with(|cell| {
        if cell.borrow().is_none() {
            let runtime = Runtime::shared()?;
            let pipeline = if agx {
                runtime.agx_pipeline(SOURCE, "dense-bf16", "materialize_bf16")?
            } else {
                runtime.pipeline(SOURCE, "dense-bf16", "materialize_bf16")?
            };
            *cell.borrow_mut() = Some(Rc::new(Context { runtime, pipeline }));
        }
        Ok(Rc::clone(
            cell.borrow().as_ref().expect("BF16 context initialized"),
        ))
    })
}
