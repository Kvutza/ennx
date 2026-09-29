use crate::apple_gpu::Runtime;
use metal::{
    CommandBufferRef, ComputeCommandEncoderRef, ComputePassDescriptor, CounterSampleBuffer,
    CounterSampleBufferDescriptor, MTLCounterSamplingPoint, MTLDispatchType, MTLStorageMode,
    NSRange,
};
use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;

const LAYER_STAGES: u64 = 11;

pub(in crate::fbt_moe) struct ScorerStageTrace {
    counters: CounterSampleBuffer,
    resolved: metal::Buffer,
    sample_count: Cell<u64>,
    stages: RefCell<Vec<&'static str>>,
}

impl ScorerStageTrace {
    pub(in crate::fbt_moe) fn new(runtime: &Runtime) -> Result<Self, String> {
        let layers = crate::forward_program::RecurrentCore::selective_fbt()
            .layer_visits(super::MODEL_LAYERS as usize, 4)?
            .len()
            .max((super::MODEL_LAYERS * super::FEEDBACK_PASSES) as usize);
        let capacity = 2 * (8 + layers as u64 * LAYER_STAGES);
        if !runtime
            .device
            .supports_counter_sampling(MTLCounterSamplingPoint::AtStageBoundary)
        {
            return Err("this Metal device does not support encoder-stage counters".into());
        }
        let counter_sets = runtime.device.counter_sets();
        let timestamp = counter_sets
            .iter()
            .find(|set| set.name() == "timestamp")
            .ok_or("this Metal device has no timestamp counter set")?;
        let descriptor = CounterSampleBufferDescriptor::new();
        descriptor.set_storage_mode(MTLStorageMode::Shared);
        descriptor.set_sample_count(capacity);
        descriptor.set_counter_set(timestamp);
        Ok(Self {
            counters: runtime
                .device
                .new_counter_sample_buffer_with_descriptor(&descriptor)?,
            resolved: runtime.buffer::<u64>(capacity as usize),
            sample_count: Cell::new(0),
            stages: RefCell::new(Vec::with_capacity(capacity as usize / 2)),
        })
    }

    pub(super) fn reset(&self) {
        self.sample_count.set(0);
        self.stages.borrow_mut().clear();
    }

    pub(super) fn encoder<'a>(
        &self,
        command: &'a CommandBufferRef,
        stage: &'static str,
    ) -> Result<&'a ComputeCommandEncoderRef, String> {
        let index = self.sample_count.get();
        if (index + 2) * 8 > self.resolved.length() {
            return Err("scorer stage counter capacity exceeded".into());
        }
        let descriptor = ComputePassDescriptor::new();
        descriptor.set_dispatch_type(MTLDispatchType::Concurrent);
        let attachment = descriptor
            .sample_buffer_attachments()
            .object_at(0)
            .ok_or("Metal stage counter attachment unavailable")?;
        attachment.set_sample_buffer(&self.counters);
        attachment.set_start_of_encoder_sample_index(index);
        attachment.set_end_of_encoder_sample_index(index + 1);
        self.stages.borrow_mut().push(stage);
        self.sample_count.set(index + 2);
        Ok(command.compute_command_encoder_with_descriptor(&descriptor))
    }

    pub(in crate::fbt_moe) fn resolve(&self, command: &CommandBufferRef) {
        let encoder = command.new_blit_command_encoder();
        encoder.resolve_counters(
            &self.counters,
            NSRange::new(0, self.sample_count.get()),
            &self.resolved,
            0,
        );
        encoder.end_encoding();
    }

    pub(in crate::fbt_moe) fn durations_ms(
        &self,
        scale_ns_per_tick: f64,
    ) -> Result<BTreeMap<&'static str, f64>, String> {
        let count = self.sample_count.get() as usize;
        let samples =
            unsafe { std::slice::from_raw_parts(self.resolved.contents().cast::<u64>(), count) };
        let stages = self.stages.borrow();
        let mut durations = BTreeMap::new();
        for (stage_index, stage) in stages.iter().enumerate() {
            let start = stage_index * 2;
            let ticks = samples[start + 1]
                .checked_sub(samples[start])
                .ok_or("nonmonotonic Metal stage timestamp")?;
            *durations.entry(*stage).or_insert(0.0) += ticks as f64 * scale_ns_per_tick / 1.0e6;
        }
        Ok(durations)
    }
}
