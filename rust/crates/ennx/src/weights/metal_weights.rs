use std::cell::RefCell;
use std::ffi::c_void;
use std::sync::Arc;

use metal::{ComputePipelineState, MTLSize};

use super::{
    acquisition_code, thompson_draws, WeightBlock, WeightSelectConfig, WeightSelectResult,
};
use crate::apple_gpu::Runtime;

const THREADS: u64 = 256;
const MAX_NEIGHBORS: usize = 2048;

const SOURCE: &str = include_str!("weights.metal");

#[repr(C)]
#[derive(Clone, Copy)]
struct MetalBlock {
    offset: u32,
    length: u32,
    bits: u32,
    quantization_scale: f32,
    metric_scale: f32,
    weight: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MetalParams {
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

struct MetalCtx {
    runtime: Arc<Runtime>,
    pipeline: ComputePipelineState,
}

thread_local! {
    static CTX: RefCell<Option<MetalCtx>> = const { RefCell::new(None) };
    static AGX_CTX: RefCell<Option<MetalCtx>> = const { RefCell::new(None) };
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
            "Metal quantized-weight ENN supports at most {MAX_NEIGHBORS} neighbors"
        ));
    }
    let metal_blocks = to_metal_blocks(blocks)?;
    let context = if matches!(
        config.device,
        crate::weights::ComputeDevice::Agx | crate::weights::ComputeDevice::Auto
    ) {
        &AGX_CTX
    } else {
        &CTX
    };
    context.with(|cell| {
        if cell.borrow().is_none() {
            *cell.borrow_mut() = Some(MetalCtx::new(matches!(
                config.device,
                crate::weights::ComputeDevice::Agx | crate::weights::ComputeDevice::Auto
            ))?);
        }
        let borrow = cell.borrow();
        let ctx = borrow.as_ref().expect("Metal context initialized");
        ctx.select(
            observations,
            observation_count,
            outcomes,
            candidates,
            candidate_count,
            &metal_blocks,
            row_bytes,
            config,
        )
    })
}

impl MetalCtx {
    fn new(agx: bool) -> Result<Self, String> {
        let runtime = Runtime::shared()?;
        let pipeline = if agx {
            runtime.agx_pipeline(SOURCE, "quantized-weight", "score_weight_neighbors")?
        } else {
            runtime.pipeline(SOURCE, "quantized-weight", "score_weight_neighbors")?
        };
        Ok(Self { runtime, pipeline })
    }

    #[allow(clippy::too_many_arguments)]
    fn select(
        &self,
        observations: &[u8],
        observation_count: usize,
        outcomes: &[f32],
        candidates: &[u8],
        candidate_count: usize,
        blocks: &[MetalBlock],
        row_bytes: usize,
        config: WeightSelectConfig,
    ) -> Result<WeightSelectResult, String> {
        let obs_buffer = self.buffer_with_slice(observations);
        let outcome_buffer = self.buffer_with_slice(outcomes);
        let candidate_buffer = self.buffer_with_slice(candidates);
        let block_buffer = self.buffer_with_slice(blocks);
        let draws = thompson_draws(candidate_count, config.seed);
        let draw_buffer = self.buffer_with_slice(&draws);
        let score_buffer = self.runtime.buffer::<f32>(candidate_count);
        let params = MetalParams {
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

        let command_buffer = self.runtime.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_buffer(0, Some(&obs_buffer), 0);
        encoder.set_buffer(1, Some(&outcome_buffer), 0);
        encoder.set_buffer(2, Some(&candidate_buffer), 0);
        encoder.set_buffer(3, Some(&block_buffer), 0);
        encoder.set_buffer(4, Some(&score_buffer), 0);
        encoder.set_buffer(5, Some(&draw_buffer), 0);
        encoder.set_bytes(
            6,
            std::mem::size_of::<MetalParams>() as u64,
            (&params as *const MetalParams).cast::<c_void>(),
        );
        encoder.dispatch_thread_groups(
            MTLSize {
                width: candidate_count as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: THREADS,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let scores = unsafe {
            std::slice::from_raw_parts(score_buffer.contents().cast::<f32>(), candidate_count)
        };
        let mut best = WeightSelectResult {
            index: 0,
            score: f32::NEG_INFINITY,
        };
        for (index, &score) in scores.iter().enumerate() {
            if score > best.score || (score == best.score && index < best.index) {
                best = WeightSelectResult { index, score };
            }
        }
        Ok(best)
    }

    fn buffer_with_slice<T>(&self, values: &[T]) -> metal::Buffer {
        self.runtime.buffer_with(values)
    }
}

fn to_metal_blocks(blocks: &[WeightBlock]) -> Result<Vec<MetalBlock>, String> {
    let mut out = Vec::with_capacity(blocks.len());
    for block in blocks {
        out.push(MetalBlock {
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
