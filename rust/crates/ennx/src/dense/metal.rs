use std::cell::RefCell;
use std::ffi::c_void;
use std::sync::Arc;

use metal::ComputePipelineState;

use super::{tiles, DenseLeaf, DenseTerm};
use crate::apple_gpu::{thread_group, Runtime};

const SOURCE: &str = concat!(include_str!("ops.metal"), "\n", include_str!("dense.metal"));
const THREADS: u64 = 256;

struct Context {
    runtime: Arc<Runtime>,
    pipeline: ComputePipelineState,
}

thread_local! {
    static CONTEXT: RefCell<Option<Context>> = const { RefCell::new(None) };
    static AGX_CONTEXT: RefCell<Option<Context>> = const { RefCell::new(None) };
}

pub(super) fn apply(
    base: &[f32],
    leaves: &[DenseLeaf],
    terms: &[DenseTerm],
    agx: bool,
) -> Result<Vec<f32>, String> {
    let context = if agx { &AGX_CONTEXT } else { &CONTEXT };
    context.with(|cell| {
        if cell.borrow().is_none() {
            *cell.borrow_mut() = Some(Context::new(agx)?);
        }
        cell.borrow()
            .as_ref()
            .expect("Metal dense context initialized")
            .apply(base, leaves, terms)
    })
}

impl Context {
    fn new(agx: bool) -> Result<Self, String> {
        let runtime = Runtime::shared()?;
        let pipeline = if agx {
            runtime.agx_pipeline(SOURCE, "dense", "apply_dense")?
        } else {
            runtime.pipeline(SOURCE, "dense", "apply_dense")?
        };
        Ok(Self { runtime, pipeline })
    }

    fn apply(
        &self,
        base: &[f32],
        leaves: &[DenseLeaf],
        terms: &[DenseTerm],
    ) -> Result<Vec<f32>, String> {
        let tiles = tiles(leaves)?;
        let base_buffer = self.buffer(base);
        let leaf_buffer = self.buffer(leaves);
        let term_buffer = self.buffer(terms);
        let tile_buffer = self.buffer(&tiles);
        let out_buffer = self.runtime.buffer::<f32>(base.len());
        let term_count = u32::try_from(terms.len()).map_err(|_| "dense term count exceeds u32")?;

        let command = self.runtime.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_buffer(0, Some(&base_buffer), 0);
        encoder.set_buffer(1, Some(&leaf_buffer), 0);
        encoder.set_buffer(2, Some(&term_buffer), 0);
        encoder.set_buffer(3, Some(&tile_buffer), 0);
        encoder.set_buffer(4, Some(&out_buffer), 0);
        encoder.set_bytes(
            5,
            size_of::<u32>() as u64,
            (&term_count as *const u32).cast::<c_void>(),
        );
        encoder.dispatch_thread_groups(thread_group(tiles.len() as u64), thread_group(THREADS));
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();

        Ok(unsafe {
            std::slice::from_raw_parts(out_buffer.contents().cast::<f32>(), base.len()).to_vec()
        })
    }

    fn buffer<T>(&self, values: &[T]) -> metal::Buffer {
        self.runtime.buffer_with(values)
    }
}
