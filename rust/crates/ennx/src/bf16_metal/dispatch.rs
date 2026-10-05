use super::*;
use crate::apple_gpu::thread_group;

impl SearchState {
    pub(super) fn encode_proposal(&self, command: &CommandBufferRef, params: Params) {
        let encoder = command.new_compute_command_encoder();
        let pipeline = if self.independent_fp16 && params.history == 1 && params.base_slot == 0 {
            &self.initial_pipeline
        } else {
            &self.propose_pipeline
        };
        encoder.set_compute_pipeline_state(pipeline);
        for (index, buffer) in [
            &self.base,
            &self.anchor,
            &self.rejected,
            self.reference.as_ref().unwrap_or(&self.base),
            &self.reference_scales,
            &self.leaves_gpu,
            &self.tiles_gpu,
            &self.proposal,
            &self.partials,
            &self.pool_geometry,
        ]
        .iter()
        .enumerate()
        {
            encoder.set_buffer(index as u64, Some(buffer), 0);
        }
        encoder.set_buffer(11, Some(&self.threshold_tables), 0);
        encoder.set_bytes(
            10,
            size_of::<Params>() as u64,
            (&params as *const Params).cast(),
        );
        encoder.dispatch_thread_groups(thread_group(self.tiles.len() as u64), thread_group(256));
        encoder.end_encoding();
    }
    pub(super) fn encode_pool(&self, command: &CommandBufferRef, params: Params) {
        for pair in 0..self.resident_history.div_ceil(2) {
            let first = pair * 2;
            let history = (self.resident_history - first).min(2);
            let base_slot = usize::try_from(params.base_slot)
                .ok()
                .and_then(|slot| slot.checked_sub(first))
                .filter(|&slot| slot < history)
                .map_or(2, |slot| slot as u32);
            let params = Params {
                history: history as u32,
                initialize: pair as u32,
                base_slot,
                ..params
            };
            self.encode_pair(command, params, first);
        }
    }
    pub(super) fn encode_pair(&self, command: &CommandBufferRef, params: Params, first: usize) {
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pool_pipeline);
        for (index, buffer) in [
            &self.base,
            &self.history_rows[first],
            &self.history_rows[(first + 1).min(self.resident_history - 1)],
            self.reference.as_ref().unwrap_or(&self.base),
            &self.reference_scales,
            &self.leaves_gpu,
            &self.tiles_gpu,
            &self.proposal,
            &self.partials,
            &self.pool_geometry,
        ]
        .iter()
        .enumerate()
        {
            encoder.set_buffer(index as u64, Some(buffer), 0);
        }
        encoder.set_buffer(11, Some(&self.threshold_tables), 0);
        encoder.set_bytes(
            10,
            size_of::<Params>() as u64,
            (&params as *const Params).cast(),
        );
        encoder.dispatch_thread_groups(thread_group(self.tiles.len() as u64), thread_group(256));
        encoder.end_encoding();
    }
    pub(super) fn encode_select(
        &self,
        command: &CommandBufferRef,
        root: u64,
        config: Ask,
        forced_candidate: Option<usize>,
    ) {
        let params = self.select_params(root, config, forced_candidate);
        let fused_replay =
            self.realized_history() && self.history > self.resident_history && self.history <= 16;
        let needs_replay = forced_candidate.is_none() || self.family.is_some();
        if needs_replay && !fused_replay && !self.analytic_pool() {
            let reducer = command.new_compute_command_encoder();
            reducer.set_compute_pipeline_state(&self.reduction_pipeline);
            for (index, buffer) in [&self.partials, &self.pool_geometry, &self.pool_aggregates]
                .iter()
                .enumerate()
            {
                reducer.set_buffer(index as u64, Some(buffer), 0);
            }
            reducer.set_bytes(
                3,
                size_of::<SelectionParams>() as u64,
                (&params as *const SelectionParams).cast(),
            );
            reducer.dispatch_thread_groups(
                thread_group(self.resident_history.div_ceil(2) as u64),
                thread_group(256),
            );
            reducer.end_encoding();
        }

        if needs_replay && self.realized_history() && self.history > self.resident_history {
            self.encode_replay(command, root);
        }

        let encoder = command.new_compute_command_encoder();
        if let Some(selector) = &self.objective_selector {
            selector.upload(self, root, config);
            encoder.set_compute_pipeline_state(&selector.pipeline);
            encoder.set_buffer(4, Some(&selector.parameters), 0);
            encoder.set_buffer(5, Some(&selector.report), 0);
        } else {
            encoder.set_compute_pipeline_state(&self.selection_pipeline);
        }
        for (index, buffer) in [&self.pool_aggregates, &self.decision, &self.pool_distances]
            .iter()
            .enumerate()
        {
            encoder.set_buffer(index as u64, Some(buffer), 0);
        }
        encoder.set_bytes(
            3,
            size_of::<SelectionParams>() as u64,
            (&params as *const SelectionParams).cast(),
        );
        encoder.dispatch_thread_groups(thread_group(1), thread_group(1));
        encoder.end_encoding();
    }
    fn encode_replay(&self, command: &CommandBufferRef, root: u64) {
        let blit = command.new_blit_command_encoder();
        blit.copy_from_buffer(
            self.replay_origin.as_ref().unwrap(),
            0,
            &self.history_rows[0],
            0,
            self.row_bytes(),
        );
        blit.end_encoding();
        let params = Params {
            history: self.history as u32,
            blocks: self.blocks.len() as u32,
            ..self.pool_params(root)
        };
        if self.history <= 16 {
            let replay = command.new_compute_command_encoder();
            replay.set_compute_pipeline_state(&self.replay_short_pipeline);
            for (index, buffer) in [
                &self.history_rows[0],
                &self.base,
                &self.leaves_gpu,
                &self.tiles_gpu,
                &self.replay_steps,
                &self.replay_scales,
                &self.family_base_weights,
                &self.replay_partials,
                &self.replay_components,
                &self.partials,
                &self.pool_geometry,
            ]
            .iter()
            .enumerate()
            {
                replay.set_buffer(index as u64, Some(buffer), 0);
            }
            replay.set_bytes(
                11,
                size_of::<Params>() as u64,
                (&params as *const Params).cast(),
            );
            replay.dispatch_thread_groups(thread_group(self.tiles.len() as u64), thread_group(256));
            replay.end_encoding();
        } else {
            for start in (0..self.history).step_by(32) {
                let stage = Params {
                    initialize: start as u32,
                    ..params
                };
                let replay = command.new_compute_command_encoder();
                replay.set_compute_pipeline_state(&self.replay_pipeline);
                for (index, buffer) in [
                    &self.history_rows[0],
                    &self.base,
                    &self.leaves_gpu,
                    &self.tiles_gpu,
                    &self.replay_steps,
                    &self.replay_scales,
                    &self.family_base_weights,
                    &self.replay_partials,
                    &self.replay_components,
                ]
                .iter()
                .enumerate()
                {
                    replay.set_buffer(index as u64, Some(buffer), 0);
                }
                replay.set_bytes(
                    9,
                    size_of::<Params>() as u64,
                    (&stage as *const Params).cast(),
                );
                replay.dispatch_thread_groups(
                    thread_group(self.tiles.len() as u64),
                    thread_group(256),
                );
                replay.end_encoding();
            }
        }
        let replay = command.new_compute_command_encoder();
        replay.set_compute_pipeline_state(&self.replay_reduction_pipeline);
        for (index, buffer) in [
            &self.replay_partials,
            &self.replay_components,
            &self.family_groups,
            &self.tiles_gpu,
            &self.pool_distances,
            &self.pool_family_distances,
        ]
        .iter()
        .enumerate()
        {
            replay.set_buffer(index as u64, Some(buffer), 0);
        }
        replay.set_bytes(
            6,
            size_of::<Params>() as u64,
            (&params as *const Params).cast(),
        );
        replay.dispatch_thread_groups(thread_group((4 * self.history) as u64), thread_group(256));
        replay.end_encoding();
    }
    pub(super) fn encode_row(&self, command: &CommandBufferRef) {
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.materialize_pipeline);
        for (index, buffer) in [
            &self.base,
            self.reference.as_ref().unwrap_or(&self.base),
            &self.reference_scales,
            &self.leaves_gpu,
            &self.tiles_gpu,
            &self.proposal,
            &self.decision,
            &self.threshold_tables,
        ]
        .iter()
        .enumerate()
        {
            encoder.set_buffer(index as u64, Some(buffer), 0);
        }
        let account = u32::from(self.analytic_pool());
        encoder.set_bytes(8, size_of::<u32>() as u64, (&account as *const u32).cast());
        encoder.set_buffer(9, Some(&self.partials), 0);
        let params = Params {
            tiles: self.tiles.len() as u32,
            ..Params::default()
        };
        encoder.set_bytes(
            10,
            size_of::<Params>() as u64,
            (&params as *const Params).cast(),
        );
        encoder.dispatch_thread_groups(thread_group(self.tiles.len() as u64), thread_group(256));
        encoder.end_encoding();
    }
    pub(super) fn encode_ref(&self, command: &CommandBufferRef, params: Params) {
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.reference_pipeline);
        for (index, buffer) in [
            self.reference.as_ref().unwrap(),
            &self.reference_scales,
            &self.leaves_gpu,
            &self.tiles_gpu,
            &self.reference_partials,
        ]
        .iter()
        .enumerate()
        {
            encoder.set_buffer(index as u64, Some(buffer), 0);
        }
        encoder.set_bytes(
            5,
            size_of::<Params>() as u64,
            (&params as *const Params).cast(),
        );
        encoder.dispatch_thread_groups(thread_group(self.tiles.len() as u64), thread_group(256));
        encoder.end_encoding();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.rms_pipeline);
        for (index, buffer) in [
            &self.reference_partials,
            &self.leaves_gpu,
            &self.offsets_gpu,
            &self.reference_scales,
        ]
        .iter()
        .enumerate()
        {
            encoder.set_buffer(index as u64, Some(buffer), 0);
        }
        encoder.dispatch_thread_groups(thread_group(self.blocks.len() as u64), thread_group(256));
        encoder.end_encoding();
    }
    pub(super) fn copy(&self, source: &Buffer, destination: &Buffer) -> Result<(), String> {
        autoreleasepool(|| self.copy_inner(source, destination))
    }

    pub(super) fn copy_inner(&self, source: &Buffer, destination: &Buffer) -> Result<(), String> {
        let command = self.runtime.queue.new_command_buffer();
        let blit = command.new_blit_command_encoder();
        blit.copy_from_buffer(source, 0, destination, 0, self.row_bytes());
        blit.end_encoding();
        finish(command)
    }
}
