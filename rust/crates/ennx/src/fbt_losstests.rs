//! Complete-vocabulary loss oracle, including partial token tiles.

use super::*;
use metal::objc::rc::autoreleasepool;

const TEST_ROWS: usize = 129;
const LABELS: [u32; 6] = [0, 63, 64, 127, 4096, 8191];

struct LossCase {
    input: Buffer,
    weights: Buffer,
    labels: Buffer,
    partials: Buffer,
    losses: Buffer,
}

fn logit_bits(column: usize) -> u16 {
    [0xc400, 0xb800, 0x0000, 0x3800, 0x4400, 0x5400][column % 6]
}

impl LossCase {
    fn new(runtime: &Runtime) -> Self {
        let mut input = vec![0u16; TEST_ROWS * WIDTH as usize];
        for row in 0..TEST_ROWS {
            input[row * WIDTH as usize] = [0x3c00, 0xbc00, 0x0000][row % 3];
        }
        let mut weights = vec![0u16; (WIDTH * VOCAB) as usize];
        for (column, value) in weights[..VOCAB as usize].iter_mut().enumerate() {
            *value = logit_bits(column);
        }
        let labels: Vec<_> = (0..TEST_ROWS)
            .map(|row| LABELS[row % LABELS.len()])
            .collect();
        Self {
            input: runtime.buffer_with(&input),
            weights: runtime.buffer_with(&weights),
            labels: runtime.buffer_with(&labels),
            partials: runtime.buffer::<[f32; 4]>(TEST_ROWS * (VOCAB / 64) as usize),
            losses: runtime.buffer_with(&vec![-123456.0f32; TEST_ROWS + 1]),
        }
    }

    fn tiles(&self, encoder: &ComputeCommandEncoderRef, pipeline: &ComputePipelineState) {
        let shape = [TEST_ROWS as u32, VOCAB];
        encoder.set_compute_pipeline_state(pipeline);
        for (index, buffer) in [&self.input, &self.weights, &self.labels, &self.partials]
            .into_iter()
            .enumerate()
        {
            encoder.set_buffer(index as u64, Some(buffer), 0);
        }
        encoder.set_bytes(
            4,
            std::mem::size_of_val(&shape) as u64,
            shape.as_ptr().cast(),
        );
        encoder.dispatch_thread_groups(
            MTLSize {
                width: u64::from(VOCAB / 64),
                height: TEST_ROWS.div_ceil(128) as u64,
                depth: 1,
            },
            thread_group(128),
        );
        encoder.memory_barrier_with_resources(&[&self.partials]);
    }

    fn reduce(&self, encoder: &ComputeCommandEncoderRef, pipeline: &ComputePipelineState) {
        let tiles = VOCAB / 64;
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&self.partials), 0);
        encoder.set_buffer(1, Some(&self.losses), 0);
        encoder.set_bytes(
            2,
            std::mem::size_of_val(&tiles) as u64,
            (&tiles as *const u32).cast(),
        );
        encoder.dispatch_thread_groups(thread_group(TEST_ROWS as u64), thread_group(128));
    }

    fn check(&self) {
        let losses = unsafe {
            std::slice::from_raw_parts(self.losses.contents().cast::<f32>(), TEST_ROWS + 1)
        };
        for (row, &loss) in losses[..TEST_ROWS].iter().enumerate() {
            let expected = reference_loss(row);
            assert!(loss.is_finite());
            assert!(
                (f64::from(loss) - expected).abs() < 0.00002,
                "row {row}: loss {loss}, expected {expected}"
            );
        }
        assert_eq!(losses[TEST_ROWS], -123456.0);
    }
}

fn reference_loss(row: usize) -> f64 {
    let scale = [1.0, -1.0, 0.0][row % 3];
    let logits: Vec<_> = (0..VOCAB as usize)
        .map(|column| scale * decode_half(logit_bits(column)))
        .collect();
    let maximum = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let sum: f64 = logits.iter().map(|value| (value - maximum).exp()).sum();
    sum.ln() + maximum - logits[LABELS[row % LABELS.len()] as usize]
}

#[test]
fn vocabulary_loss() -> Result<(), String> {
    autoreleasepool(|| {
        let runtime = Runtime::shared()?;
        let tiles = runtime.precise_metal4(
            include_str!("fbt_loss_tensorops.metal"),
            "complete-vocabulary loss",
            "fbt_readout_loss_tiles",
            &[],
        )?;
        let reduce = runtime.precise(
            include_str!("fbt_moe.metal"),
            "complete-vocabulary loss reduction",
            "fbt_readout_loss_reduce",
        )?;
        let case = LossCase::new(&runtime);
        let command = runtime.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        case.tiles(encoder, &tiles);
        case.reduce(encoder, &reduce);
        encoder.end_encoding();
        complete(command)?;
        case.check();
        Ok(())
    })
}

#[test]
fn vocabulary_proposals() -> Result<(), String> {
    autoreleasepool(|| {
        let runtime = Runtime::shared()?;
        let tiles = runtime.precise_metal4(
            include_str!("fbt_loss_tensorops.metal"),
            "complete-vocabulary proposals",
            "fbt_readout_proposal_tiles",
            &[],
        )?;
        let reduce = runtime.precise(
            include_str!("fbt_moe.metal"),
            "complete-vocabulary proposal reduction",
            "fbt_readout_proposal_reduce",
        )?;
        let case = LossCase::new(&runtime);
        let proposals = runtime.buffer_with(&vec![u32::MAX; TEST_ROWS + 1]);
        let shape = [TEST_ROWS as u32, VOCAB, 0, CONTEXT];
        let seeds = runtime.buffer_with(&[rand::random::<u64>(), rand::random::<u64>()]);
        let temperature = 0.0f32;
        let tile_count = VOCAB / 64;
        let command = runtime.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&tiles);
        encoder.set_buffer(0, Some(&case.input), 0);
        encoder.set_buffer(1, Some(&case.weights), 0);
        encoder.set_buffer(2, Some(&case.partials), 0);
        encoder.set_bytes(3, size_of_val(&shape) as u64, shape.as_ptr().cast());
        encoder.set_buffer(4, Some(&seeds), 0);
        encoder.set_bytes(
            5,
            size_of_val(&temperature) as u64,
            std::ptr::from_ref(&temperature).cast(),
        );
        let score_targets = false;
        encoder.set_buffer(6, Some(&case.labels), 0);
        encoder.set_buffer(7, Some(&case.partials), 0);
        encoder.set_bytes(
            8,
            size_of_val(&score_targets) as u64,
            std::ptr::from_ref(&score_targets).cast(),
        );
        encoder.dispatch_thread_groups(
            MTLSize {
                width: u64::from(tile_count),
                height: TEST_ROWS.div_ceil(128) as u64,
                depth: 1,
            },
            thread_group(128),
        );
        encoder.memory_barrier_with_resources(&[&case.partials]);
        encoder.set_compute_pipeline_state(&reduce);
        encoder.set_buffer(0, Some(&case.partials), 0);
        encoder.set_buffer(1, Some(&proposals), 0);
        encoder.set_bytes(
            2,
            size_of_val(&tile_count) as u64,
            std::ptr::from_ref(&tile_count).cast(),
        );
        encoder.dispatch_thread_groups(thread_group(TEST_ROWS as u64), thread_group(128));
        encoder.end_encoding();
        complete(command)?;
        let proposals = unsafe {
            std::slice::from_raw_parts(proposals.contents().cast::<u32>(), TEST_ROWS + 1)
        };
        for (row, &token) in proposals[..TEST_ROWS].iter().enumerate() {
            assert_eq!(token, [5, 0, 0][row % 3], "row {row}");
        }
        assert_eq!(proposals[TEST_ROWS], u32::MAX);
        Ok(())
    })
}
