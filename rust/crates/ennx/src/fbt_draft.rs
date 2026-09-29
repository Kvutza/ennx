//! Candidate-conditioned context for the full-length speculative drafter.

use super::*;

pub(super) const FEATURE_COUNT: u32 = 5;

#[repr(C)]
struct CaptureShape {
    vectors: u32,
    slot: u32,
    row_start: u32,
    context: u32,
}

pub(super) struct Context {
    capture: ComputePipelineState,
    features: Buffer,
    context: u32,
}

impl Context {
    pub(super) fn new(runtime: &Runtime) -> Result<Self, String> {
        Self::with_context(runtime, CONTEXT)
    }

    pub(super) fn with_context(runtime: &Runtime, context: u32) -> Result<Self, String> {
        Ok(Self {
            capture: runtime.precise(
                include_str!("fbt_draft.metal"),
                "4096-token draft target context",
                "fbt_draft_capture",
            )?,
            features: runtime
                .buffer::<u16>(FEATURE_COUNT as usize * context as usize * WIDTH as usize),
            context,
        })
    }

    pub(super) fn capture(
        &self,
        encoder: &ComputeCommandEncoderRef,
        hidden: &BufferRef,
        execution: u32,
        executions: u32,
        row_start: u32,
        rows: u32,
    ) {
        let Some(slot) = feature_slot(execution, executions) else {
            return;
        };
        let vectors = rows * WIDTH / 4;
        let shape = CaptureShape {
            vectors,
            slot,
            row_start,
            context: self.context,
        };
        encoder.set_compute_pipeline_state(&self.capture);
        encoder.set_buffer(
            0,
            Some(hidden),
            half_bytes(u64::from(row_start) * u64::from(WIDTH)),
        );
        encoder.set_buffer(1, Some(&self.features), 0);
        encoder.set_bytes(
            2,
            size_of::<CaptureShape>() as u64,
            std::ptr::from_ref(&shape).cast(),
        );
        encoder.dispatch_threads(
            thread_group(u64::from(vectors)),
            thread_group(self.capture.max_total_threads_per_threadgroup().min(256)),
        );
        encoder.memory_barrier_with_resources(&[&self.features]);
    }

    pub(super) fn features(&self) -> &BufferRef {
        &self.features
    }
}

const fn feature_visit(slot: u32, executions: u32) -> u32 {
    let last = if executions - 3 > FEATURE_COUNT {
        executions - 3
    } else {
        FEATURE_COUNT
    };
    1 + slot * (last - 1) / (FEATURE_COUNT - 1)
}

fn feature_slot(execution: u32, executions: u32) -> Option<u32> {
    (executions >= 7)
        .then(|| (0..FEATURE_COUNT).find(|&slot| feature_visit(slot, executions) == execution))
        .flatten()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feature_visits() {
        let visits = (0..FEATURE_COUNT)
            .map(|slot| feature_visit(slot, 10))
            .collect::<Vec<_>>();
        assert_eq!(visits, [1, 2, 4, 5, 7]);
        assert!(visits.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(
            visits
                .iter()
                .all(|&visit| feature_slot(visit, 10).is_some())
        );
        assert_eq!(feature_slot(0, 10), None);
        assert_eq!(feature_slot(9, 10), None);
        assert_eq!(
            (0..FEATURE_COUNT)
                .map(|slot| feature_visit(slot, 7))
                .collect::<Vec<_>>(),
            [1, 2, 3, 4, 5]
        );
    }
}
