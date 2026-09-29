//! Exact BF16-to-FP32 expansion for an explicitly immutable dense evaluator.
//! Mutable BO candidates must not use this cache.

use super::*;

pub(super) struct Readout {
    source: Buffer,
    expanded: Buffer,
    matmul: RefCell<MpsMatmul>,
}

impl Readout {
    pub(super) fn new(evaluator: &QwenEvaluator, weights: &Buffer) -> Result<Self> {
        evaluator.check_weights(weights)?;
        if !metal::mps::mps_supports_device(&evaluator.runtime.device) {
            return Err("frozen readout requires MPS support".into());
        }
        let elements = product(&[
            evaluator.config.hidden as usize,
            evaluator.config.vocab as usize,
        ])?;
        let expanded = evaluator.runtime.buffer::<f32>(elements);
        if expanded.contents().is_null() {
            return Err("could not allocate frozen FP32 readout".into());
        }
        let command = evaluator.runtime.queue.new_command_buffer();
        command.set_label("Frozen readout BF16 expansion");
        evaluator.encode(
            command,
            "qwen_widen",
            &[
                (weights, evaluator.layout.embedding as u64 * 2),
                (&expanded, 0),
            ],
            &(elements as u64),
            thread_group((elements as u64).div_ceil(256)),
        );
        finish(command)?;
        Ok(Self {
            source: weights.clone(),
            expanded,
            matmul: RefCell::new(MpsMatmul::default()),
        })
    }

    pub(super) fn bytes(&self) -> u64 {
        self.expanded.length()
    }

    pub(super) fn encode(
        &self,
        evaluator: &QwenEvaluator,
        command: &CommandBufferRef,
        weights: &Buffer,
        start: u32,
        rows: u32,
    ) -> Result<bool> {
        if weights.contents() != self.source.contents() || rows < MPS_MINROWS {
            return Ok(false);
        }
        let c = evaluator.config;
        let input = MpsMatrix::new(evaluator.buffer(W::Norm), start + rows, c.hidden)
            .row_view(rows, u64::from(start) * u64::from(c.hidden));
        self.matmul.borrow_mut().encode(
            &evaluator.runtime.device,
            command,
            input,
            MpsMatrix::new(&self.expanded, c.vocab, c.hidden),
            MpsMatrix::new(evaluator.buffer(W::Logits), rows, c.vocab),
            true,
            1.0,
        )?;
        Ok(true)
    }
}

#[cfg(test)]
#[path = "qwen_readouttests.rs"]
mod tests;
