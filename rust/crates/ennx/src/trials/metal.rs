use std::ffi::c_void;
use std::sync::Arc;

use metal::{Buffer, ComputePipelineState, MTLSize};

use super::{make_steps, make_tiles, Ask, Center, Leaf, LeafStep, Tile};
use crate::apple_gpu::{thread_group, Runtime};

const THREADS: u64 = 256;
const HISTORY_BATCH: usize = 8;
const SOURCE: &str = include_str!("trials.metal");

#[repr(C)]
#[derive(Clone, Copy)]
struct Seed {
    low: u32,
    high: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Params {
    row_stride: u32,
    history: u32,
    candidates: u32,
    leaves: u32,
    tiles: u32,
    neighbors: u32,
    base_slot: u32,
    trial_slot: u32,
    center_count: u32,
    acquisition: u32,
    epistemic_scale: f32,
    aleatoric_scale: f32,
    y_scale: f32,
    beta: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CenterStep {
    parent: u32,
    seed: Seed,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MultiTrParams {
    num_regions: u32,
    candidates_per_region: u32,
}

struct Scratch {
    history_slots: Buffer,
    outcomes: Buffer,
    seeds: Buffer,
    draws: Buffer,
    scores: Buffer,
    partials: Buffer,
    choice: Buffer,
    selected_scores: Buffer,
    leaves: Buffer,
    tiles: Buffer,
    centers: Buffer,
    candidate_centers: Buffer,
    candidate_capacity: usize,
    center_capacity: usize,
}

#[derive(Default)]
struct Resident {
    history: Vec<(usize, f32)>,
    steps: Vec<LeafStep>,
    centers: Vec<Center>,
    region_centers: Vec<usize>,
    candidates_per_region: usize,
}

pub(super) struct Engine {
    runtime: Arc<Runtime>,
    rows: Buffer,
    row_bytes: usize,
    row_stride: usize,
    tile_count: usize,
    distance: ComputePipelineState,
    score: ComputePipelineState,
    pick: ComputePipelineState,
    multi_tr_pick: ComputePipelineState,
    write: ComputePipelineState,
    scratch: Scratch,
    resident: Resident,
}

impl Engine {
    pub(super) fn new(base: &[u8], leaves: &[Leaf], slots: usize) -> Result<Self, String> {
        Self::with_agx(base, leaves, slots, false)
    }

    pub(super) fn new_agx(base: &[u8], leaves: &[Leaf], slots: usize) -> Result<Self, String> {
        Self::with_agx(base, leaves, slots, true)
    }

    fn with_agx(base: &[u8], leaves: &[Leaf], slots: usize, agx: bool) -> Result<Self, String> {
        let runtime = Runtime::shared()?;
        let pipeline = |name| {
            if agx {
                runtime.agx_pipeline(SOURCE, "trial", name)
            } else {
                runtime.pipeline(SOURCE, "trial", name)
            }
        };
        let distance = pipeline("distance_trials")?;
        let score = pipeline("score_trials")?;
        let pick = pipeline("pick_trial")?;
        let multi_tr_pick = pipeline("multi_tr_pick_trials")?;
        let write = pipeline("write_trial")?;
        let row_bytes = base.len();
        let row_stride = row_bytes
            .checked_add(3)
            .ok_or("model row stride overflow")?
            & !3;
        let tiles = make_tiles(leaves);
        let rows = shared(
            &runtime,
            slots
                .checked_mul(row_stride)
                .ok_or("model row arena size overflow")?,
            "model rows",
        )?;
        copy_to(&rows, base);
        let scratch = Scratch {
            history_slots: shared(
                &runtime,
                super::MAX_HISTORY * size_of::<u32>(),
                "history slots",
            )?,
            outcomes: shared(&runtime, super::MAX_HISTORY * size_of::<f32>(), "outcomes")?,
            seeds: shared(&runtime, size_of::<Seed>(), "seeds")?,
            draws: shared(&runtime, size_of::<f32>(), "draws")?,
            scores: shared(&runtime, size_of::<f32>(), "scores")?,
            partials: shared(
                &runtime,
                super::MAX_HISTORY
                    .saturating_mul(tiles.len())
                    .saturating_mul(size_of::<f32>()),
                "partial distances",
            )?,
            choice: shared(&runtime, size_of::<u32>(), "choice")?,
            selected_scores: shared(&runtime, size_of::<f32>(), "selected scores")?,
            leaves: shared(
                &runtime,
                leaves.len().saturating_mul(size_of::<LeafStep>()),
                "leaves",
            )?,
            tiles: shared(
                &runtime,
                tiles.len().saturating_mul(size_of::<Tile>()),
                "tiles",
            )?,
            centers: shared(&runtime, size_of::<CenterStep>(), "centers")?,
            candidate_centers: shared(&runtime, size_of::<u32>(), "candidate centers")?,
            candidate_capacity: 1,
            center_capacity: 1,
        };
        copy_to(&scratch.tiles, &tiles);
        Ok(Self {
            runtime,
            rows,
            row_bytes,
            row_stride,
            tile_count: tiles.len(),
            distance,
            score,
            pick,
            multi_tr_pick,
            write,
            scratch,
            resident: Resident::default(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn ask(
        &mut self,
        base_slot: usize,
        history: &[(usize, f32)],
        trial_slot: usize,
        seeds: &[u64],
        leaves: &[Leaf],
        config: Ask,
        materialize_row: bool,
    ) -> Result<(usize, f32), String> {
        self.ensure_candidates(seeds.len())?;
        let distance_groups = distance_groups(seeds.len(), history.len(), self.tile_count)?;
        let steps = make_steps(leaves, config.length);
        self.sync_history(history)?;
        self.sync_steps(&steps);
        self.write_seeds(seeds);
        self.sync_draws(seeds.len(), config);

        let params = Params {
            row_stride: to_u32(self.row_stride, "row stride")?,
            history: to_u32(history.len(), "history length")?,
            candidates: to_u32(seeds.len(), "candidate count")?,
            leaves: to_u32(leaves.len(), "leaf count")?,
            tiles: to_u32(self.tile_count, "tile count")?,
            neighbors: to_u32(config.neighbors, "neighbor count")?,
            base_slot: to_u32(base_slot, "base slot")?,
            trial_slot: to_u32(trial_slot, "trial slot")?,
            center_count: 0,
            acquisition: crate::weights::acquisition_code(config.acquisition),
            epistemic_scale: config.epistemic_scale,
            aleatoric_scale: config.aleatoric_scale,
            y_scale: config.y_scale,
            beta: config.beta,
        };

        let command = self.runtime.queue.new_command_buffer();
        let mut gpu = self.runtime.trace(3 + usize::from(materialize_row))?;
        let encoder = gpu.encoder(command, "trials.distance")?;
        encoder.set_compute_pipeline_state(&self.distance);
        encoder.set_buffer(0, Some(&self.rows), 0);
        encoder.set_buffer(1, Some(&self.scratch.history_slots), 0);
        encoder.set_buffer(2, Some(&self.scratch.seeds), 0);
        encoder.set_buffer(3, Some(&self.scratch.leaves), 0);
        encoder.set_buffer(4, Some(&self.scratch.tiles), 0);
        encoder.set_buffer(5, Some(&self.scratch.partials), 0);
        encoder.set_buffer(6, Some(&self.scratch.centers), 0);
        encoder.set_buffer(7, Some(&self.scratch.candidate_centers), 0);
        set_params(&encoder, 8, &params);
        encoder.dispatch_thread_groups(thread_group(distance_groups), thread_group(THREADS));
        drop(encoder);

        let encoder = gpu.encoder(command, "trials.score")?;
        encoder.set_compute_pipeline_state(&self.score);
        encoder.set_buffer(0, Some(&self.scratch.partials), 0);
        encoder.set_buffer(1, Some(&self.scratch.outcomes), 0);
        encoder.set_buffer(2, Some(&self.scratch.draws), 0);
        encoder.set_buffer(3, Some(&self.scratch.scores), 0);
        set_params(&encoder, 4, &params);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: seeds.len() as u64,
                height: 1,
                depth: 1,
            },
            thread_group(THREADS),
        );
        drop(encoder);

        let encoder = gpu.encoder(command, "trials.pick")?;
        encoder.set_compute_pipeline_state(&self.pick);
        encoder.set_buffer(0, Some(&self.scratch.scores), 0);
        encoder.set_buffer(1, Some(&self.scratch.choice), 0);
        set_params(&encoder, 2, &params);
        encoder.dispatch_thread_groups(
            thread_group(1),
            thread_group(selection_threads(seeds.len())),
        );
        drop(encoder);

        if materialize_row {
            let encoder = gpu.encoder(command, "trials.write")?;
            encoder.set_compute_pipeline_state(&self.write);
            encoder.set_buffer(0, Some(&self.rows), 0);
            encoder.set_buffer(1, Some(&self.scratch.seeds), 0);
            encoder.set_buffer(2, Some(&self.scratch.choice), 0);
            encoder.set_buffer(3, Some(&self.scratch.leaves), 0);
            encoder.set_buffer(4, Some(&self.scratch.tiles), 0);
            set_params(&encoder, 5, &params);
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: params.tiles as u64,
                    height: 1,
                    depth: 1,
                },
                thread_group(THREADS),
            );
            drop(encoder);
        }
        gpu.resolve(command);
        command.commit();
        command.wait_until_completed();
        gpu.upload()?;

        let index = read_one::<u32>(&self.scratch.choice) as usize;
        let scores = read_slice::<f32>(&self.scratch.scores, seeds.len());
        Ok((index, scores[index]))
    }

    pub(super) fn materialize(
        &mut self,
        base_slot: usize,
        trial_slot: usize,
        seed: u64,
        steps: &[LeafStep],
    ) -> Result<(), String> {
        self.ensure_candidates(1)?;
        self.sync_steps(steps);
        self.write_seeds(&[seed]);
        copy_one(&self.scratch.choice, 0, 0_u32);

        let params = Params {
            row_stride: to_u32(self.row_stride, "row stride")?,
            history: 0,
            candidates: 1,
            leaves: to_u32(steps.len(), "leaf count")?,
            tiles: to_u32(self.tile_count, "tile count")?,
            neighbors: 0,
            base_slot: to_u32(base_slot, "base slot")?,
            trial_slot: to_u32(trial_slot, "trial slot")?,
            center_count: 0,
            acquisition: 0,
            epistemic_scale: 0.0,
            aleatoric_scale: 0.0,
            y_scale: 0.0,
            beta: 0.0,
        };

        let command = self.runtime.queue.new_command_buffer();
        let mut gpu = self.runtime.trace(1)?;
        let encoder = gpu.encoder(command, "trials.write")?;
        encoder.set_compute_pipeline_state(&self.write);
        encoder.set_buffer(0, Some(&self.rows), 0);
        encoder.set_buffer(1, Some(&self.scratch.seeds), 0);
        encoder.set_buffer(2, Some(&self.scratch.choice), 0);
        encoder.set_buffer(3, Some(&self.scratch.leaves), 0);
        encoder.set_buffer(4, Some(&self.scratch.tiles), 0);
        set_params(&encoder, 5, &params);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: params.tiles as u64,
                height: 1,
                depth: 1,
            },
            thread_group(THREADS),
        );
        drop(encoder);
        gpu.resolve(command);
        command.commit();
        command.wait_until_completed();
        gpu.upload()?;
        Ok(())
    }

    #[allow(dead_code)]
    pub(super) fn ask_multi_tr(
        &mut self,
        base_slot: usize,
        history: &[(usize, f32)],
        num_regions: usize,
        seeds_per_region: usize,
        seeds: &[u64],
        leaves: &[Leaf],
        config: Ask,
    ) -> Result<Vec<(usize, f32)>, String> {
        self.ask_multi_tr_impl(
            base_slot,
            history,
            num_regions,
            seeds_per_region,
            None,
            seeds,
            leaves,
            config,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn ask_multi_tr_tree(
        &mut self,
        base_slot: usize,
        history: &[(usize, f32)],
        seeds_per_region: usize,
        centers: &[Center],
        region_centers: &[usize],
        seeds: &[u64],
        leaves: &[Leaf],
        config: Ask,
    ) -> Result<Vec<(usize, f32)>, String> {
        self.ask_multi_tr_impl(
            base_slot,
            history,
            region_centers.len(),
            seeds_per_region,
            Some((centers, region_centers)),
            seeds,
            leaves,
            config,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn ask_multi_tr_impl(
        &mut self,
        base_slot: usize,
        history: &[(usize, f32)],
        num_regions: usize,
        seeds_per_region: usize,
        tree: Option<(&[Center], &[usize])>,
        seeds: &[u64],
        leaves: &[Leaf],
        config: Ask,
    ) -> Result<Vec<(usize, f32)>, String> {
        if num_regions == 0 || seeds_per_region == 0 {
            return Err("multi-TR search requires non-zero regions and candidates".to_string());
        }
        let total_candidates = num_regions
            .checked_mul(seeds_per_region)
            .ok_or("multi-TR candidate count overflow")?;
        if seeds.len() != total_candidates {
            return Err(format!(
                "expected {total_candidates} seeds for {num_regions} regions, got {}",
                seeds.len()
            ));
        }
        self.ensure_candidates(total_candidates)?;
        let center_count = self.write_centers(tree, seeds_per_region)?;
        let distance_groups = distance_groups(seeds.len(), history.len(), self.tile_count)?;
        let steps = make_steps(leaves, config.length);
        self.sync_history(history)?;
        self.sync_steps(&steps);
        self.write_seeds(seeds);
        self.sync_draws(seeds.len(), config);

        let params = Params {
            row_stride: to_u32(self.row_stride, "row stride")?,
            history: to_u32(history.len(), "history length")?,
            candidates: to_u32(seeds.len(), "candidate count")?,
            leaves: to_u32(leaves.len(), "leaf count")?,
            tiles: to_u32(self.tile_count, "tile count")?,
            neighbors: to_u32(config.neighbors, "neighbor count")?,
            base_slot: to_u32(base_slot, "base slot")?,
            trial_slot: to_u32(base_slot, "trial slot")?,
            center_count: to_u32(center_count, "center count")?,
            acquisition: crate::weights::acquisition_code(config.acquisition),
            epistemic_scale: config.epistemic_scale,
            aleatoric_scale: config.aleatoric_scale,
            y_scale: config.y_scale,
            beta: config.beta,
        };

        let command = self.runtime.queue.new_command_buffer();
        let mut gpu = self.runtime.trace(3)?;
        let encoder = gpu.encoder(command, "trials.distance")?;
        encoder.set_compute_pipeline_state(&self.distance);
        encoder.set_buffer(0, Some(&self.rows), 0);
        encoder.set_buffer(1, Some(&self.scratch.history_slots), 0);
        encoder.set_buffer(2, Some(&self.scratch.seeds), 0);
        encoder.set_buffer(3, Some(&self.scratch.leaves), 0);
        encoder.set_buffer(4, Some(&self.scratch.tiles), 0);
        encoder.set_buffer(5, Some(&self.scratch.partials), 0);
        encoder.set_buffer(6, Some(&self.scratch.centers), 0);
        encoder.set_buffer(7, Some(&self.scratch.candidate_centers), 0);
        set_params(&encoder, 8, &params);
        encoder.dispatch_thread_groups(thread_group(distance_groups), thread_group(THREADS));
        drop(encoder);

        let encoder = gpu.encoder(command, "trials.score")?;
        encoder.set_compute_pipeline_state(&self.score);
        encoder.set_buffer(0, Some(&self.scratch.partials), 0);
        encoder.set_buffer(1, Some(&self.scratch.outcomes), 0);
        encoder.set_buffer(2, Some(&self.scratch.draws), 0);
        encoder.set_buffer(3, Some(&self.scratch.scores), 0);
        set_params(&encoder, 4, &params);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: seeds.len() as u64,
                height: 1,
                depth: 1,
            },
            thread_group(THREADS),
        );
        drop(encoder);

        let multi_tr_params = MultiTrParams {
            num_regions: to_u32(num_regions, "region count")?,
            candidates_per_region: to_u32(seeds_per_region, "candidates per region")?,
        };
        let encoder = gpu.encoder(command, "trials.pick")?;
        encoder.set_compute_pipeline_state(&self.multi_tr_pick);
        encoder.set_buffer(0, Some(&self.scratch.scores), 0);
        encoder.set_buffer(1, Some(&self.scratch.choice), 0);
        encoder.set_buffer(2, Some(&self.scratch.selected_scores), 0);
        encoder.set_bytes(
            3,
            size_of::<MultiTrParams>() as u64,
            (&multi_tr_params as *const MultiTrParams).cast::<c_void>(),
        );
        encoder.dispatch_thread_groups(
            thread_group(num_regions as u64),
            thread_group(selection_threads(seeds_per_region)),
        );
        drop(encoder);
        gpu.resolve(command);
        command.commit();
        command.wait_until_completed();
        gpu.upload()?;

        let choices = read_slice::<u32>(&self.scratch.choice, num_regions);
        let scores = read_slice::<f32>(&self.scratch.selected_scores, num_regions);
        Ok(choices
            .iter()
            .zip(scores)
            .map(|(&index, &score)| (index as usize, score))
            .collect())
    }

    pub(super) fn read(&self, slot: usize, row_bytes: usize) -> Vec<u8> {
        let start = slot * self.row_stride;
        unsafe {
            std::slice::from_raw_parts(self.rows.contents().cast::<u8>().add(start), row_bytes)
                .to_vec()
        }
    }

    pub(super) fn row_buffer(&self, slot: usize) -> Result<(Buffer, usize), String> {
        if self.row_bytes == 0 || slot >= self.rows.length() as usize / self.row_stride {
            return Err(format!("model row slot {slot} is out of range"));
        }
        Ok((self.rows.to_owned(), slot * self.row_stride))
    }

    pub(super) fn write(&self, slot: usize, row: &[u8]) {
        let start = slot * self.row_stride;
        unsafe {
            std::ptr::copy_nonoverlapping(
                row.as_ptr(),
                self.rows.contents().cast::<u8>().add(start),
                row.len(),
            );
        }
    }

    fn ensure_candidates(&mut self, count: usize) -> Result<(), String> {
        if count <= self.scratch.candidate_capacity {
            return Ok(());
        }
        let capacity = count.next_power_of_two();
        self.scratch.seeds = shared(
            &self.runtime,
            capacity.saturating_mul(size_of::<Seed>()),
            "seeds",
        )?;
        self.scratch.draws = shared(
            &self.runtime,
            capacity.saturating_mul(size_of::<f32>()),
            "draws",
        )?;
        self.scratch.scores = shared(
            &self.runtime,
            capacity.saturating_mul(size_of::<f32>()),
            "scores",
        )?;
        self.scratch.choice = shared(
            &self.runtime,
            capacity.saturating_mul(size_of::<u32>()),
            "choices",
        )?;
        self.scratch.selected_scores = shared(
            &self.runtime,
            capacity.saturating_mul(size_of::<f32>()),
            "selected scores",
        )?;
        self.scratch.candidate_centers = shared(
            &self.runtime,
            capacity.saturating_mul(size_of::<u32>()),
            "candidate centers",
        )?;
        let partial_count = capacity
            .checked_mul(super::MAX_HISTORY)
            .and_then(|value| value.checked_mul(self.tile_count))
            .ok_or("partial distance buffer size overflow")?;
        self.scratch.partials = shared(
            &self.runtime,
            partial_count.saturating_mul(size_of::<f32>()),
            "partial distances",
        )?;
        self.scratch.candidate_capacity = capacity;
        self.resident.region_centers.clear();
        Ok(())
    }

    fn sync_history(&mut self, history: &[(usize, f32)]) -> Result<(), String> {
        let prefix = self
            .resident
            .history
            .iter()
            .zip(history)
            .take_while(|((old_slot, old_value), (slot, value))| {
                old_slot == slot && old_value.to_bits() == value.to_bits()
            })
            .count();
        if prefix == history.len() && history.len() == self.resident.history.len() {
            return Ok(());
        }
        let start = if prefix == self.resident.history.len() {
            prefix
        } else {
            0
        };
        for (index, &(slot, value)) in history.iter().enumerate().skip(start) {
            let slot = to_u32(slot, "history slot")?;
            copy_one(&self.scratch.history_slots, index, slot);
            copy_one(&self.scratch.outcomes, index, value);
        }
        self.resident.history.clear();
        self.resident.history.extend_from_slice(history);
        Ok(())
    }

    fn sync_steps(&mut self, steps: &[LeafStep]) {
        if self.resident.steps == steps {
            return;
        }
        copy_to(&self.scratch.leaves, steps);
        self.resident.steps.clear();
        self.resident.steps.extend_from_slice(steps);
    }

    fn sync_draws(&self, count: usize, config: Ask) {
        if config.acquisition == crate::weights::AcquisitionKind::Thompson {
            let draws = crate::weights::thompson_draws(count, config.seed);
            copy_to(&self.scratch.draws, &draws);
        }
    }

    fn write_seeds(&self, seeds: &[u64]) {
        debug_assert_eq!(size_of::<Seed>(), size_of::<u64>());
        unsafe {
            std::ptr::copy_nonoverlapping(
                seeds.as_ptr().cast::<u8>(),
                self.scratch.seeds.contents().cast::<u8>(),
                std::mem::size_of_val(seeds),
            );
        }
    }

    fn write_centers(
        &mut self,
        tree: Option<(&[Center], &[usize])>,
        candidates_per_region: usize,
    ) -> Result<usize, String> {
        let Some((centers, region_centers)) = tree else {
            return Ok(0);
        };
        if self.resident.centers == centers
            && self.resident.region_centers == region_centers
            && self.resident.candidates_per_region == candidates_per_region
        {
            return Ok(centers.len());
        }
        if centers.len() > self.scratch.center_capacity {
            let capacity = centers.len().next_power_of_two();
            self.scratch.centers = shared(
                &self.runtime,
                capacity.saturating_mul(size_of::<CenterStep>()),
                "centers",
            )?;
            self.scratch.center_capacity = capacity;
        }
        for (index, center) in centers.iter().enumerate() {
            let parent = center
                .parent
                .map(|parent| to_u32(parent, "center parent"))
                .transpose()?
                .unwrap_or(u32::MAX);
            copy_one(
                &self.scratch.centers,
                index,
                CenterStep {
                    parent,
                    seed: Seed {
                        low: center.seed as u32,
                        high: (center.seed >> 32) as u32,
                    },
                },
            );
        }
        for (region, &center) in region_centers.iter().enumerate() {
            let center = to_u32(center, "region center")?;
            let start = region * candidates_per_region;
            for candidate in start..start + candidates_per_region {
                copy_one(&self.scratch.candidate_centers, candidate, center);
            }
        }
        self.resident.centers.clear();
        self.resident.centers.extend_from_slice(centers);
        self.resident.region_centers.clear();
        self.resident
            .region_centers
            .extend_from_slice(region_centers);
        self.resident.candidates_per_region = candidates_per_region;
        Ok(centers.len())
    }
}

fn shared(runtime: &Runtime, bytes: usize, name: &str) -> Result<Buffer, String> {
    if bytes == 0 {
        return Err(format!("{name} buffer cannot be empty"));
    }
    let max_bytes = runtime.device.max_buffer_length();
    if bytes as u64 > max_bytes as u64 {
        return Err(format!(
            "{name} buffer requires {bytes} bytes, exceeding the Metal device limit of {max_bytes} bytes"
        ));
    }
    let buffer = runtime.buffer::<u8>(bytes);
    if buffer.length() < bytes as u64 || buffer.contents().is_null() {
        return Err(format!(
            "Metal could not allocate the {name} buffer ({bytes} bytes)"
        ));
    }
    Ok(buffer)
}

fn copy_to<T>(buffer: &Buffer, values: &[T]) {
    unsafe {
        std::ptr::copy_nonoverlapping(
            values.as_ptr().cast::<u8>(),
            buffer.contents().cast::<u8>(),
            std::mem::size_of_val(values),
        );
    }
}

fn copy_one<T: Copy>(buffer: &Buffer, index: usize, value: T) {
    unsafe {
        buffer.contents().cast::<T>().add(index).write(value);
    }
}

fn read_one<T: Copy>(buffer: &Buffer) -> T {
    unsafe { *buffer.contents().cast::<T>() }
}

fn read_slice<T: Copy>(buffer: &Buffer, len: usize) -> &[T] {
    unsafe { std::slice::from_raw_parts(buffer.contents().cast::<T>(), len) }
}

fn set_params(encoder: &metal::ComputeCommandEncoderRef, index: u64, params: &Params) {
    encoder.set_bytes(
        index,
        size_of::<Params>() as u64,
        (params as *const Params).cast::<c_void>(),
    );
}

fn to_u32(value: usize, name: &str) -> Result<u32, String> {
    u32::try_from(value).map_err(|_| format!("{name} exceeds u32 range"))
}

fn distance_groups(candidates: usize, history: usize, tiles: usize) -> Result<u64, String> {
    candidates
        .div_ceil(2)
        .checked_mul(history.div_ceil(HISTORY_BATCH))
        .and_then(|value| value.checked_mul(tiles))
        .and_then(|value| u64::try_from(value).ok())
        .ok_or("distance dispatch size overflow".to_string())
}

fn selection_threads(candidates: usize) -> u64 {
    candidates.next_power_of_two().clamp(32, THREADS as usize) as u64
}

const fn size_of<T>() -> usize {
    std::mem::size_of::<T>()
}
