use std::ffi::c_void;
use std::sync::Arc;

use metal::{Buffer, ComputePipelineState, MTLSize};

use super::{make_steps, make_tiles, Ask, Center, LeafStep, Parameter, SparseEdit, Tile};
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
    num_pert: u32,
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

#[repr(C)]
#[derive(Clone, Copy)]
struct RowSumParams {
    row_stride: u32,
    row_bytes: u32,
    slot: u32,
    pad: u32,
}

struct Scratch {
    params: Buffer,
    history_slots: Buffer,
    outcomes: Buffer,
    seeds: Buffer,
    draws: Buffer,
    scores: Buffer,
    partials: Buffer,
    base_distances: Buffer,
    choice: Buffer,
    selected_scores: Buffer,
    leaves: Buffer,
    tiles: Buffer,
    centers: Buffer,
    candidate_centers: Buffer,
    edits: Buffer,
    row_sum: Buffer,
    candidate_capacity: usize,
    center_capacity: usize,
    edit_capacity: usize,
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
    base_distance: ComputePipelineState,
    score: ComputePipelineState,
    score_sparse: ComputePipelineState,
    fill_seeds: ComputePipelineState,
    fill_edits: ComputePipelineState,
    pick: ComputePipelineState,
    multi_tr_pick: ComputePipelineState,
    row_sum: ComputePipelineState,
    write: ComputePipelineState,
    write_sparse: ComputePipelineState,
    scratch: Scratch,
    resident: Resident,
    region: Option<Buffer>,
    radii: Buffer,
    region_steps: ComputePipelineState,
    state: Option<Buffer>,
    capacity: usize,
    configure: ComputePipelineState,
    advance: ComputePipelineState,
    copy_row: ComputePipelineState,
    shadow: Buffer,
}

impl Engine {
    pub(super) fn new(base: &[u8], leaves: &[Parameter], slots: usize) -> Result<Self, String> {
        let runtime = Runtime::shared()?;
        let source = format!(
            "{SOURCE}\n{}\n#define REGION_METAL\n{}",
            include_str!("../search/arithmetic.cl"),
            include_str!("region_steps.cl")
        );
        let source = format!(
            "{source}\n#define DEVICE device\n{}\n{}",
            include_str!("../search/adaptation.cl"),
            include_str!("control.cl")
        );
        let pipeline = |name| runtime.precise(&source, "trial", name);
        let configure = pipeline("configure")?;
        let advance = pipeline("advance")?;
        let copy_row = pipeline("copy_row")?;
        let shadow = runtime.buffer::<u64>(16);
        let region_steps = pipeline("region_steps")?;
        let radii = runtime.buffer_with(&leaves.iter().map(|leaf| leaf.radius).collect::<Vec<_>>());
        let distance = pipeline("distance_trials")?;
        let base_distance = pipeline("base_distance")?;
        let score = pipeline("score_trials")?;
        let score_sparse = pipeline("score_sparse")?;
        let fill_seeds = pipeline("fill_seeds")?;
        let fill_edits = pipeline("fill_edits")?;
        let pick = pipeline("pick_trial")?;
        let multi_tr_pick = pipeline("multi_tr_pick_trials")?;
        let row_sum = pipeline("row_sum")?;
        let write = pipeline("write_trial")?;
        let write_sparse = pipeline("write_sparse")?;
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
            params: runtime.buffer::<Params>(1),
            history_slots: shared(
                &runtime,
                super::MAX_HISTORY * size_of::<u32>(),
                "history slots",
            )?,
            outcomes: shared(&runtime, super::MAX_HISTORY * size_of::<f32>(), "outcomes")?,
            seeds: shared(&runtime, size_of::<Seed>(), "seeds")?,
            draws: shared(&runtime, slots * size_of::<f32>(), "draws")?,
            scores: shared(&runtime, size_of::<f32>(), "scores")?,
            partials: shared(
                &runtime,
                super::MAX_HISTORY
                    .saturating_mul(tiles.len())
                    .saturating_mul(size_of::<f32>()),
                "partial distances",
            )?,
            base_distances: shared(
                &runtime,
                super::MAX_HISTORY.saturating_mul(size_of::<f32>()),
                "base distances",
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
            edits: shared(&runtime, size_of::<SparseEdit>(), "sparse edits")?,
            row_sum: shared(&runtime, size_of::<u64>(), "row sum")?,
            candidate_capacity: 1,
            center_capacity: 1,
            edit_capacity: 1,
        };
        copy_to(&scratch.tiles, &tiles);
        copy_to(&scratch.leaves, &make_steps(leaves, 0.0));
        Ok(Self {
            runtime,
            rows,
            row_bytes,
            row_stride,
            tile_count: tiles.len(),
            distance,
            base_distance,
            score,
            score_sparse,
            fill_seeds,
            fill_edits,
            pick,
            multi_tr_pick,
            row_sum,
            write,
            write_sparse,
            scratch,
            resident: Resident::default(),
            region: None,
            state: None,
            capacity: 0,
            configure,
            advance,
            copy_row,
            shadow,
            radii,
            region_steps,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn ask(
        &mut self,
        base_slot: usize,
        history: &[(usize, f32)],
        trial_slot: usize,
        seeds: &[u64],
        leaves: &[Parameter],
        config: Ask,
        materialize_row: bool,
    ) -> Result<(usize, f32), String> {
        let history_count = if self.state.is_some() {
            self.capacity
        } else {
            history.len()
        };
        self.ensure_candidates(seeds.len())?;
        let distance_groups = distance_groups(seeds.len(), history_count, self.tile_count)?;
        let steps = if self.region.is_some() {
            Vec::new()
        } else {
            make_steps(leaves, config.length)
        };
        self.sync_history(history)?;
        self.sync_steps(&steps);
        self.write_seeds(seeds);
        self.sync_draws(config);

        let params = Params {
            row_stride: to_u32(self.row_stride, "row stride")?,
            history: to_u32(history_count, "history length")?,
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
            num_pert: 0,
        };
        copy_to(&self.scratch.params, &[params]);
        let params = self.scratch.params.clone();

        let command = self.runtime.queue.new_command_buffer();
        self.encode_region(command);
        self.encode_state(command, &params);
        let encoder = command.new_compute_command_encoder();
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
        encoder.end_encoding();

        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.score);
        encoder.set_buffer(0, Some(&self.scratch.partials), 0);
        encoder.set_buffer(1, Some(&self.scratch.outcomes), 0);
        encoder.set_buffer(2, Some(&self.scratch.draws), 0);
        encoder.set_buffer(3, Some(&self.scratch.scores), 0);
        set_params(&encoder, 4, &params);
        encoder.set_buffer(5, Some(&self.scratch.history_slots), 0);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: seeds.len() as u64,
                height: 1,
                depth: 1,
            },
            thread_group(THREADS),
        );
        encoder.end_encoding();

        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pick);
        encoder.set_buffer(0, Some(&self.scratch.scores), 0);
        encoder.set_buffer(1, Some(&self.scratch.choice), 0);
        encoder.set_buffer(2, Some(&self.scratch.selected_scores), 0);
        set_params(&encoder, 3, &params);
        encoder.dispatch_thread_groups(
            thread_group(1),
            thread_group(selection_threads(seeds.len())),
        );
        encoder.end_encoding();

        if materialize_row {
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.write);
            encoder.set_buffer(0, Some(&self.rows), 0);
            encoder.set_buffer(1, Some(&self.scratch.seeds), 0);
            encoder.set_buffer(2, Some(&self.scratch.choice), 0);
            encoder.set_buffer(3, Some(&self.scratch.leaves), 0);
            encoder.set_buffer(4, Some(&self.scratch.tiles), 0);
            set_params(&encoder, 5, &params);
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: self.tile_count as u64,
                    height: 1,
                    depth: 1,
                },
                thread_group(THREADS),
            );
            encoder.end_encoding();
        }
        command.commit();
        command.wait_until_completed();
        let index = read_one::<u32>(&self.scratch.choice) as usize;
        if index == u32::MAX as usize {
            return Err("proposal rejected: insufficient history or unavailable slot".into());
        }
        let score = read_one::<f32>(&self.scratch.selected_scores);
        Ok((index, score))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn ask_stream(
        &mut self,
        base_slot: usize,
        history: &[(usize, f32)],
        trial_slot: usize,
        base_seed: u64,
        count: usize,
        leaves: &[Parameter],
        config: Ask,
        materialize_row: bool,
    ) -> Result<(usize, u64, f32), String> {
        let history_count = if self.state.is_some() {
            self.capacity
        } else {
            history.len()
        };
        self.ensure_candidates(count)?;
        let distance_groups = distance_groups(count, history_count, self.tile_count)?;
        let steps = if self.region.is_some() {
            Vec::new()
        } else {
            make_steps(leaves, config.length)
        };
        self.sync_history(history)?;
        self.sync_steps(&steps);
        self.fill_seeds(base_seed, count)?;
        self.sync_draws(config);

        let params = Params {
            row_stride: to_u32(self.row_stride, "row stride")?,
            history: to_u32(history_count, "history length")?,
            candidates: to_u32(count, "candidate count")?,
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
            num_pert: 0,
        };
        copy_to(&self.scratch.params, &[params]);
        let params = self.scratch.params.clone();

        let command = self.runtime.queue.new_command_buffer();
        self.encode_region(command);
        self.encode_state(command, &params);
        let encoder = command.new_compute_command_encoder();
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
        encoder.end_encoding();

        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.score);
        encoder.set_buffer(0, Some(&self.scratch.partials), 0);
        encoder.set_buffer(1, Some(&self.scratch.outcomes), 0);
        encoder.set_buffer(2, Some(&self.scratch.draws), 0);
        encoder.set_buffer(3, Some(&self.scratch.scores), 0);
        set_params(&encoder, 4, &params);
        encoder.set_buffer(5, Some(&self.scratch.history_slots), 0);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: count as u64,
                height: 1,
                depth: 1,
            },
            thread_group(THREADS),
        );
        encoder.end_encoding();

        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pick);
        encoder.set_buffer(0, Some(&self.scratch.scores), 0);
        encoder.set_buffer(1, Some(&self.scratch.choice), 0);
        encoder.set_buffer(2, Some(&self.scratch.selected_scores), 0);
        set_params(&encoder, 3, &params);
        encoder.dispatch_thread_groups(thread_group(1), thread_group(selection_threads(count)));
        encoder.end_encoding();

        if materialize_row {
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.write);
            encoder.set_buffer(0, Some(&self.rows), 0);
            encoder.set_buffer(1, Some(&self.scratch.seeds), 0);
            encoder.set_buffer(2, Some(&self.scratch.choice), 0);
            encoder.set_buffer(3, Some(&self.scratch.leaves), 0);
            encoder.set_buffer(4, Some(&self.scratch.tiles), 0);
            set_params(&encoder, 5, &params);
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: self.tile_count as u64,
                    height: 1,
                    depth: 1,
                },
                thread_group(THREADS),
            );
            encoder.end_encoding();
        }
        command.commit();
        command.wait_until_completed();
        let index = read_one::<u32>(&self.scratch.choice) as usize;
        if index == u32::MAX as usize {
            return Err("proposal rejected: insufficient history or unavailable slot".into());
        }
        let score = read_one::<f32>(&self.scratch.selected_scores);
        Ok((index, super::cpu::seed_at(base_seed, index as u32), score))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn ask_sparse(
        &mut self,
        base_slot: usize,
        history: &[(usize, f32)],
        trial_slot: usize,
        seeds: &[u64],
        edits: &[SparseEdit],
        num_pert: usize,
        leaves: &[Parameter],
        config: Ask,
    ) -> Result<(usize, f32), String> {
        let history_count = if self.state.is_some() {
            self.capacity
        } else {
            history.len()
        };
        if self.state.is_none() && base_slot == trial_slot {
            return Err("Metal sparse base and destination slots must differ".to_string());
        }
        if history_count == 0 || history_count > super::MAX_HISTORY {
            return Err(format!(
                "Metal sparse history must contain 1..={} rows",
                super::MAX_HISTORY
            ));
        }
        if seeds.is_empty() {
            return Err("Metal sparse trials require at least one candidate".to_string());
        }
        if num_pert == 0 {
            return Err("Metal sparse edit count must be positive".to_string());
        }
        if self.state.is_none() && edits.len() != seeds.len().saturating_mul(num_pert) {
            return Err("Metal sparse edit count does not match candidates".to_string());
        }
        if config.neighbors == 0 || config.neighbors > history_count {
            return Err("Metal sparse neighbor count exceeds resident history".to_string());
        }

        self.ensure_candidates(seeds.len())?;
        self.ensure_edits(seeds.len().saturating_mul(num_pert))?;
        let steps = if self.region.is_some() {
            Vec::new()
        } else {
            make_steps(leaves, config.length)
        };
        self.sync_history(history)?;
        self.sync_steps(&steps);
        self.write_seeds(seeds);
        self.sync_draws(config);
        if self.state.is_none() {
            copy_to(&self.scratch.edits, edits);
        }

        let params = Params {
            row_stride: to_u32(self.row_stride, "row stride")?,
            history: to_u32(history_count, "history length")?,
            candidates: to_u32(seeds.len(), "candidate count")?,
            leaves: to_u32(leaves.len(), "leaf count")?,
            tiles: to_u32(self.tile_count, "tile count")?,
            neighbors: to_u32(config.neighbors, "neighbor count")?,
            base_slot: to_u32(base_slot, "base slot")?,
            trial_slot: to_u32(trial_slot, "trial slot")?,
            center_count: to_u32(
                leaves.iter().map(|leaf| leaf.length).sum(),
                "dimension count",
            )?,
            acquisition: crate::weights::acquisition_code(config.acquisition),
            epistemic_scale: config.epistemic_scale,
            aleatoric_scale: config.aleatoric_scale,
            y_scale: config.y_scale,
            beta: config.beta,
            num_pert: to_u32(num_pert, "sparse edit count")?,
        };
        copy_to(&self.scratch.params, &[params]);
        let params = self.scratch.params.clone();

        let command = self.runtime.queue.new_command_buffer();
        self.encode_region(command);
        self.encode_state(command, &params);
        if self.state.is_some() {
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.fill_edits);
            encoder.set_buffer(0, Some(&self.scratch.edits), 0);
            encoder.set_buffer(1, Some(&self.scratch.seeds), 0);
            encoder.set_buffer(2, Some(&self.scratch.leaves), 0);
            set_params(&encoder, 3, &params);
            encoder.dispatch_thread_groups(thread_group(seeds.len() as u64), thread_group(1));
            encoder.end_encoding();
        }
        if self.state.is_none() {
            let blit = command.new_blit_command_encoder();
            blit.copy_from_buffer(
                &self.rows,
                (base_slot * self.row_stride) as u64,
                &self.rows,
                (trial_slot * self.row_stride) as u64,
                self.row_bytes as u64,
            );
            blit.end_encoding();
        }

        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.base_distance);
        encoder.set_buffer(0, Some(&self.rows), 0);
        encoder.set_buffer(1, Some(&self.scratch.history_slots), 0);
        encoder.set_buffer(2, Some(&self.scratch.leaves), 0);
        encoder.set_buffer(3, Some(&self.scratch.base_distances), 0);
        set_params(&encoder, 4, &params);
        encoder.dispatch_thread_groups(thread_group(history_count as u64), thread_group(THREADS));
        encoder.end_encoding();

        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.score_sparse);
        encoder.set_buffer(0, Some(&self.rows), 0);
        encoder.set_buffer(1, Some(&self.scratch.history_slots), 0);
        encoder.set_buffer(2, Some(&self.scratch.outcomes), 0);
        encoder.set_buffer(3, Some(&self.scratch.seeds), 0);
        encoder.set_buffer(4, Some(&self.scratch.draws), 0);
        encoder.set_buffer(5, Some(&self.scratch.leaves), 0);
        encoder.set_buffer(6, Some(&self.scratch.edits), 0);
        encoder.set_buffer(7, Some(&self.scratch.base_distances), 0);
        encoder.set_buffer(8, Some(&self.scratch.scores), 0);
        set_params(&encoder, 9, &params);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: seeds.len() as u64,
                height: 1,
                depth: 1,
            },
            thread_group(THREADS),
        );
        encoder.end_encoding();

        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pick);
        encoder.set_buffer(0, Some(&self.scratch.scores), 0);
        encoder.set_buffer(1, Some(&self.scratch.choice), 0);
        encoder.set_buffer(2, Some(&self.scratch.selected_scores), 0);
        set_params(&encoder, 3, &params);
        encoder.dispatch_thread_groups(
            thread_group(1),
            thread_group(selection_threads(seeds.len())),
        );
        encoder.end_encoding();

        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.write_sparse);
        encoder.set_buffer(0, Some(&self.rows), 0);
        encoder.set_buffer(1, Some(&self.scratch.seeds), 0);
        encoder.set_buffer(2, Some(&self.scratch.choice), 0);
        encoder.set_buffer(3, Some(&self.scratch.leaves), 0);
        encoder.set_buffer(4, Some(&self.scratch.edits), 0);
        set_params(&encoder, 5, &params);
        encoder.dispatch_thread_groups(thread_group(1), thread_group(1));
        encoder.end_encoding();

        command.commit();
        command.wait_until_completed();
        let index = read_one::<u32>(&self.scratch.choice) as usize;
        if index == u32::MAX as usize {
            return Err("proposal rejected: insufficient history or unavailable slot".into());
        }
        let score = read_one::<f32>(&self.scratch.selected_scores);
        Ok((index, score))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn sparse_stream(
        &mut self,
        base_slot: usize,
        history: &[(usize, f32)],
        trial_slot: usize,
        base_seed: u64,
        count: usize,
        num_pert: usize,
        leaves: &[Parameter],
        config: Ask,
    ) -> Result<(usize, u64, f32), String> {
        let history_count = if self.state.is_some() {
            self.capacity
        } else {
            history.len()
        };
        if self.state.is_none() && base_slot == trial_slot {
            return Err("Metal sparse base and destination slots must differ".to_string());
        }
        if history_count == 0 || history_count > super::MAX_HISTORY {
            return Err(format!(
                "Metal sparse history must contain 1..={} rows",
                super::MAX_HISTORY
            ));
        }
        if config.neighbors == 0 || config.neighbors > history_count {
            return Err("Metal sparse neighbor count exceeds resident history".to_string());
        }
        let dimensions = leaves.iter().map(|leaf| leaf.length).sum::<usize>();
        self.ensure_candidates(count)?;
        self.ensure_edits(count.saturating_mul(num_pert))?;
        let steps = if self.region.is_some() {
            Vec::new()
        } else {
            make_steps(leaves, config.length)
        };
        self.sync_history(history)?;
        self.sync_steps(&steps);
        self.fill_seeds(base_seed, count)?;
        self.sync_draws(config);

        let params = Params {
            row_stride: to_u32(self.row_stride, "row stride")?,
            history: to_u32(history_count, "history length")?,
            candidates: to_u32(count, "candidate count")?,
            leaves: to_u32(leaves.len(), "leaf count")?,
            tiles: to_u32(self.tile_count, "tile count")?,
            neighbors: to_u32(config.neighbors, "neighbor count")?,
            base_slot: to_u32(base_slot, "base slot")?,
            trial_slot: to_u32(trial_slot, "trial slot")?,
            center_count: to_u32(dimensions, "dimension count")?,
            acquisition: crate::weights::acquisition_code(config.acquisition),
            epistemic_scale: config.epistemic_scale,
            aleatoric_scale: config.aleatoric_scale,
            y_scale: config.y_scale,
            beta: config.beta,
            num_pert: to_u32(num_pert, "sparse edit count")?,
        };
        copy_to(&self.scratch.params, &[params]);
        let params = self.scratch.params.clone();

        let command = self.runtime.queue.new_command_buffer();
        self.encode_region(command);
        self.encode_state(command, &params);
        if self.state.is_none() {
            let blit = command.new_blit_command_encoder();
            blit.copy_from_buffer(
                &self.rows,
                (base_slot * self.row_stride) as u64,
                &self.rows,
                (trial_slot * self.row_stride) as u64,
                self.row_bytes as u64,
            );
            blit.end_encoding();
        }

        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.fill_edits);
        encoder.set_buffer(0, Some(&self.scratch.edits), 0);
        encoder.set_buffer(1, Some(&self.scratch.seeds), 0);
        encoder.set_buffer(2, Some(&self.scratch.leaves), 0);
        set_params(&encoder, 3, &params);
        encoder.dispatch_thread_groups(thread_group(count as u64), thread_group(1));
        encoder.end_encoding();

        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.base_distance);
        encoder.set_buffer(0, Some(&self.rows), 0);
        encoder.set_buffer(1, Some(&self.scratch.history_slots), 0);
        encoder.set_buffer(2, Some(&self.scratch.leaves), 0);
        encoder.set_buffer(3, Some(&self.scratch.base_distances), 0);
        set_params(&encoder, 4, &params);
        encoder.dispatch_thread_groups(thread_group(history_count as u64), thread_group(THREADS));
        encoder.end_encoding();

        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.score_sparse);
        encoder.set_buffer(0, Some(&self.rows), 0);
        encoder.set_buffer(1, Some(&self.scratch.history_slots), 0);
        encoder.set_buffer(2, Some(&self.scratch.outcomes), 0);
        encoder.set_buffer(3, Some(&self.scratch.seeds), 0);
        encoder.set_buffer(4, Some(&self.scratch.draws), 0);
        encoder.set_buffer(5, Some(&self.scratch.leaves), 0);
        encoder.set_buffer(6, Some(&self.scratch.edits), 0);
        encoder.set_buffer(7, Some(&self.scratch.base_distances), 0);
        encoder.set_buffer(8, Some(&self.scratch.scores), 0);
        set_params(&encoder, 9, &params);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: count as u64,
                height: 1,
                depth: 1,
            },
            thread_group(THREADS),
        );
        encoder.end_encoding();

        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pick);
        encoder.set_buffer(0, Some(&self.scratch.scores), 0);
        encoder.set_buffer(1, Some(&self.scratch.choice), 0);
        encoder.set_buffer(2, Some(&self.scratch.selected_scores), 0);
        set_params(&encoder, 3, &params);
        encoder.dispatch_thread_groups(thread_group(1), thread_group(selection_threads(count)));
        encoder.end_encoding();

        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.write_sparse);
        encoder.set_buffer(0, Some(&self.rows), 0);
        encoder.set_buffer(1, Some(&self.scratch.seeds), 0);
        encoder.set_buffer(2, Some(&self.scratch.choice), 0);
        encoder.set_buffer(3, Some(&self.scratch.leaves), 0);
        encoder.set_buffer(4, Some(&self.scratch.edits), 0);
        set_params(&encoder, 5, &params);
        encoder.dispatch_thread_groups(thread_group(1), thread_group(1));
        encoder.end_encoding();

        command.commit();
        command.wait_until_completed();
        let index = read_one::<u32>(&self.scratch.choice) as usize;
        if index == u32::MAX as usize {
            return Err("proposal rejected: insufficient history or unavailable slot".into());
        }
        let score = read_one::<f32>(&self.scratch.selected_scores);
        Ok((index, super::cpu::seed_at(base_seed, index as u32), score))
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
            num_pert: 0,
        };
        copy_to(&self.scratch.params, &[params]);
        let params = self.scratch.params.clone();

        let command = self.runtime.queue.new_command_buffer();
        self.encode_region(command);
        self.encode_state(command, &params);
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.write);
        encoder.set_buffer(0, Some(&self.rows), 0);
        encoder.set_buffer(1, Some(&self.scratch.seeds), 0);
        encoder.set_buffer(2, Some(&self.scratch.choice), 0);
        encoder.set_buffer(3, Some(&self.scratch.leaves), 0);
        encoder.set_buffer(4, Some(&self.scratch.tiles), 0);
        set_params(&encoder, 5, &params);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: self.tile_count as u64,
                height: 1,
                depth: 1,
            },
            thread_group(THREADS),
        );
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        Ok(())
    }

    #[allow(dead_code)]
    pub(super) fn ask_regions(
        &mut self,
        base_slot: usize,
        history: &[(usize, f32)],
        num_regions: usize,
        seeds_per_region: usize,
        seeds: &[u64],
        leaves: &[Parameter],
        config: Ask,
    ) -> Result<Vec<(usize, f32)>, String> {
        self.regions_core(
            base_slot,
            history,
            num_regions,
            seeds_per_region,
            None,
            seeds,
            None,
            leaves,
            config,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn regions_stream(
        &mut self,
        base_slot: usize,
        history: &[(usize, f32)],
        num_regions: usize,
        seeds_per_region: usize,
        base_seed: u64,
        leaves: &[Parameter],
        config: Ask,
    ) -> Result<Vec<(usize, f32)>, String> {
        self.regions_core(
            base_slot,
            history,
            num_regions,
            seeds_per_region,
            None,
            &[],
            Some(base_seed),
            leaves,
            config,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn ask_centers(
        &mut self,
        base_slot: usize,
        history: &[(usize, f32)],
        seeds_per_region: usize,
        centers: &[Center],
        region_centers: &[usize],
        seeds: &[u64],
        leaves: &[Parameter],
        config: Ask,
    ) -> Result<Vec<(usize, f32)>, String> {
        self.regions_core(
            base_slot,
            history,
            region_centers.len(),
            seeds_per_region,
            Some((centers, region_centers)),
            seeds,
            None,
            leaves,
            config,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn centers_stream(
        &mut self,
        base_slot: usize,
        history: &[(usize, f32)],
        seeds_per_region: usize,
        centers: &[Center],
        region_centers: &[usize],
        base_seed: u64,
        leaves: &[Parameter],
        config: Ask,
    ) -> Result<Vec<(usize, f32)>, String> {
        self.regions_core(
            base_slot,
            history,
            region_centers.len(),
            seeds_per_region,
            Some((centers, region_centers)),
            &[],
            Some(base_seed),
            leaves,
            config,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn regions_core(
        &mut self,
        base_slot: usize,
        history: &[(usize, f32)],
        num_regions: usize,
        seeds_per_region: usize,
        tree: Option<(&[Center], &[usize])>,
        seeds: &[u64],
        base_seed: Option<u64>,
        leaves: &[Parameter],
        config: Ask,
    ) -> Result<Vec<(usize, f32)>, String> {
        let history_count = if self.state.is_some() {
            self.capacity
        } else {
            history.len()
        };
        if num_regions == 0 || seeds_per_region == 0 {
            return Err("multi-TR search requires non-zero regions and candidates".to_string());
        }
        let total_candidates = num_regions
            .checked_mul(seeds_per_region)
            .ok_or("multi-TR candidate count overflow")?;
        if base_seed.is_none() && seeds.len() != total_candidates {
            return Err(format!(
                "expected {total_candidates} seeds for {num_regions} regions, got {}",
                seeds.len()
            ));
        }
        self.ensure_candidates(total_candidates)?;
        let center_count = self.write_centers(tree, seeds_per_region)?;
        let distance_groups = distance_groups(total_candidates, history_count, self.tile_count)?;
        let steps = if self.region.is_some() {
            Vec::new()
        } else {
            make_steps(leaves, config.length)
        };
        self.sync_history(history)?;
        self.sync_steps(&steps);
        self.sync_draws(config);
        if let Some(seed) = base_seed {
            self.fill_seeds(seed, total_candidates)?;
        } else {
            self.write_seeds(seeds);
        }

        let params = Params {
            row_stride: to_u32(self.row_stride, "row stride")?,
            history: to_u32(history_count, "history length")?,
            candidates: to_u32(total_candidates, "candidate count")?,
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
            num_pert: 0,
        };
        copy_to(&self.scratch.params, &[params]);
        let params = self.scratch.params.clone();

        let command = self.runtime.queue.new_command_buffer();
        self.encode_region(command);
        self.encode_state(command, &params);
        let encoder = command.new_compute_command_encoder();
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
        encoder.end_encoding();

        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.score);
        encoder.set_buffer(0, Some(&self.scratch.partials), 0);
        encoder.set_buffer(1, Some(&self.scratch.outcomes), 0);
        encoder.set_buffer(2, Some(&self.scratch.draws), 0);
        encoder.set_buffer(3, Some(&self.scratch.scores), 0);
        set_params(&encoder, 4, &params);
        encoder.set_buffer(5, Some(&self.scratch.history_slots), 0);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: total_candidates as u64,
                height: 1,
                depth: 1,
            },
            thread_group(THREADS),
        );
        encoder.end_encoding();

        let multi_tr_params = MultiTrParams {
            num_regions: to_u32(num_regions, "region count")?,
            candidates_per_region: to_u32(seeds_per_region, "candidates per region")?,
        };
        let encoder = command.new_compute_command_encoder();
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
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
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

    pub(super) fn byte_sum(&self, slot: usize) -> Result<u64, String> {
        self.row_buffer(slot)?;
        let params = RowSumParams {
            row_stride: to_u32(self.row_stride, "row stride")?,
            row_bytes: to_u32(self.row_bytes, "row bytes")?,
            slot: to_u32(slot, "row slot")?,
            pad: 0,
        };
        let command = self.runtime.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.row_sum);
        encoder.set_buffer(0, Some(&self.rows), 0);
        encoder.set_buffer(1, Some(&self.scratch.row_sum), 0);
        sum_params(&encoder, 2, &params);
        encoder.dispatch_thread_groups(thread_group(1), thread_group(THREADS));
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        Ok(read_one::<u64>(&self.scratch.row_sum))
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

    fn ensure_edits(&mut self, count: usize) -> Result<(), String> {
        if count == 0 {
            return Err("Metal sparse edit buffer cannot be empty".to_string());
        }
        if count <= self.scratch.edit_capacity {
            return Ok(());
        }
        let capacity = count.next_power_of_two();
        self.scratch.edits = shared(
            &self.runtime,
            capacity.saturating_mul(size_of::<SparseEdit>()),
            "sparse edits",
        )?;
        self.scratch.edit_capacity = capacity;
        Ok(())
    }

    fn sync_history(&mut self, history: &[(usize, f32)]) -> Result<(), String> {
        if self.state.is_some() {
            return Ok(());
        }
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

    pub(super) fn bind_region(&mut self, region: Buffer) {
        self.region = Some(region);
    }

    fn encode_region(&self, command: &::metal::CommandBufferRef) {
        if let Some(region) = &self.region {
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.region_steps);
            encoder.set_buffer(0, Some(region), 0);
            encoder.set_buffer(1, Some(&self.scratch.leaves), 0);
            encoder.set_buffer(2, Some(&self.radii), 0);
            let count = self.radii.length() / size_of::<f32>() as u64;
            encoder.dispatch_threads(thread_group(count), thread_group(1));
            encoder.end_encoding();
        }
    }

    fn sync_steps(&mut self, steps: &[LeafStep]) {
        if self.region.is_some() || self.resident.steps == steps {
            return;
        }
        copy_to(&self.scratch.leaves, steps);
        self.resident.steps.clear();
        self.resident.steps.extend_from_slice(steps);
    }

    fn sync_draws(&self, config: Ask) {
        if config.acquisition == crate::weights::AcquisitionKind::Thompson {
            // Cover every resident slot without reading device-managed history.
            let slots = self.scratch.draws.length() as usize / size_of::<f32>();
            let draws = crate::weights::thompson_draws(slots, config.seed);
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

    fn fill_seeds(&self, base_seed: u64, count: usize) -> Result<(), String> {
        let seed = Seed {
            low: base_seed as u32,
            high: (base_seed >> 32) as u32,
        };
        let candidates = to_u32(count, "candidate count")?;
        let command = self.runtime.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.fill_seeds);
        encoder.set_buffer(0, Some(&self.scratch.seeds), 0);
        encoder.set_bytes(
            1,
            size_of::<Seed>() as u64,
            (&seed as *const Seed).cast::<c_void>(),
        );
        encoder.set_bytes(
            2,
            size_of::<u32>() as u64,
            (&candidates as *const u32).cast::<c_void>(),
        );
        encoder.dispatch_thread_groups(thread_group(count as u64), thread_group(1));
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        Ok(())
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

fn set_params(encoder: &metal::ComputeCommandEncoderRef, index: u64, params: &Buffer) {
    encoder.set_buffer(index, Some(params), 0);
}

fn sum_params(encoder: &metal::ComputeCommandEncoderRef, index: u64, params: &RowSumParams) {
    encoder.set_bytes(
        index,
        size_of::<RowSumParams>() as u64,
        (params as *const RowSumParams).cast::<c_void>(),
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

impl Engine {
    pub(super) fn start_state(&mut self, capacity: usize, value: f32) -> Result<(), String> {
        let mut state = [u32::MAX; 36];
        state[..4].copy_from_slice(&[
            0,
            1,
            capacity as u32,
            (self.rows.length() as usize / self.row_stride) as u32,
        ]);
        self.state = Some(self.runtime.buffer_with(&state));
        self.capacity = capacity;
        copy_to(&self.scratch.history_slots, &[0u32]);
        copy_to(&self.scratch.outcomes, &[value]);
        Ok(())
    }

    fn encode_state(&self, command: &::metal::CommandBufferRef, params: &Buffer) {
        if let Some(state) = &self.state {
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.configure);
            encoder.set_buffer(0, Some(state), 0);
            encoder.set_buffer(1, Some(&self.scratch.history_slots), 0);
            encoder.set_buffer(2, Some(params), 0);
            encoder.set_buffer(3, Some(&self.scratch.choice), 0);
            encoder.dispatch_thread_groups(thread_group(1), thread_group(1));
            encoder.end_encoding();
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.copy_row);
            encoder.set_buffer(0, Some(&self.rows), 0);
            encoder.set_buffer(1, Some(params), 0);
            encoder.dispatch_threads(thread_group(self.row_stride as u64), thread_group(THREADS));
            encoder.end_encoding();
        }
    }

    pub(super) fn observe(&self, handle: usize, value: f32) -> Result<(), String> {
        let state = self.state.as_ref().ok_or("search state is not resident")?;
        let region = self.region.as_ref().ok_or("region is not resident")?;
        let handle = handle as u32;
        let value = f64::from(value).to_bits();
        let command = self.runtime.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.advance);
        encoder.set_buffer(0, Some(state), 0);
        encoder.set_buffer(1, Some(&self.scratch.history_slots), 0);
        encoder.set_buffer(2, Some(&self.scratch.outcomes), 0);
        encoder.set_buffer(3, Some(region), 0);
        encoder.set_buffer(4, Some(&self.shadow), 0);
        encoder.set_bytes(5, 4, (&handle as *const u32).cast());
        encoder.set_bytes(6, 8, (&value as *const u64).cast());
        encoder.dispatch_thread_groups(thread_group(1), thread_group(1));
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        if command.status() == ::metal::MTLCommandBufferStatus::Error {
            return Err("device search update failed".into());
        }
        Ok(())
    }

    pub(super) fn state_word(&self, index: usize) -> Result<u32, String> {
        let state = self.state.as_ref().ok_or("search state is not resident")?;
        Ok(unsafe { state.contents().cast::<u32>().add(index).read() })
    }
}

#[cfg(test)]
mod posterior_tests {
    use super::*;

    #[test]
    fn resident_noise_uses_slots_and_stays_finite() {
        let leaves = [Parameter::new(0, 1, 8, 1.0, 1.0, 1.0).unwrap()];
        let mut engine = match Engine::new(&[128], &leaves, 8) {
            Ok(engine) => engine,
            Err(error) if error.contains("no default Metal device found") => return,
            Err(error) => panic!("{error}"),
        };
        for (slot, value) in [(1, 129), (4, 131), (6, 132)] {
            engine.write(slot, &[value]);
        }
        let centers = [
            Center {
                parent: None,
                seed: 3,
            },
            Center {
                parent: Some(0),
                seed: 5,
            },
        ];
        let edits = [SparseEdit {
            leaf: 0,
            element: 0,
        }; 4];
        for aleatoric_scale in [0.05, 1.0e25] {
            let config = Ask {
                length: 0.0,
                neighbors: 2,
                acquisition: crate::weights::AcquisitionKind::Thompson,
                seed: 0x1234_5678_9abc_def0,
                aleatoric_scale,
                ..Ask::default()
            };
            // The nearest rows occupy slots 1 and 4, regardless of history order.
            // Compute the reference norm in f64 so tiny squared weights survive.
            let weights = [1.0f32, 9.0].map(|distance| {
                f64::from(1.0 / (1.0e-9 + config.epistemic_scale * distance + aleatoric_scale))
            });
            let noise = [1, 4]
                .map(|slot| f64::from(crate::hash::normal_metric(config.seed, slot, 0) as f32));
            let z = (weights[0] * noise[0] + weights[1] * noise[1])
                / (weights[0] * weights[0] + weights[1] * weights[1]).sqrt();
            let se = (1.0 / (weights[0] + weights[1]).max(1.0e-12)).sqrt();
            let expected = (se * z) as f32;
            let check = |score: f32| {
                assert!(score.is_finite());
                assert!(
                    (score - expected).abs() < 2.0e-5 * expected.abs().max(1.0),
                    "score={score}, expected={expected}"
                );
            };
            for history in [
                [(6, 0.0), (1, 0.0), (4, 0.0)],
                [(4, 0.0), (6, 0.0), (1, 0.0)],
            ] {
                for seeds in [&[7u64][..], &[13, 7, 13, 11][..]] {
                    let (index, score) = engine
                        .ask(0, &history, 7, seeds, &leaves, config, false)
                        .unwrap();
                    assert_eq!(index, 0);
                    check(score);
                    let (index, score) = engine
                        .ask_sparse(
                            0,
                            &history,
                            7,
                            seeds,
                            &edits[..seeds.len()],
                            1,
                            &leaves,
                            config,
                        )
                        .unwrap();
                    assert_eq!(index, 0);
                    check(score);
                }
                check(
                    engine
                        .ask_stream(0, &history, 7, 19, 2, &leaves, config, false)
                        .unwrap()
                        .2,
                );
                check(
                    engine
                        .sparse_stream(0, &history, 7, 19, 2, 1, &leaves, config)
                        .unwrap()
                        .2,
                );
                for (_, score) in engine
                    .regions_stream(0, &history, 2, 2, 19, &leaves, config)
                    .unwrap()
                {
                    check(score);
                }
                for (_, score) in engine
                    .centers_stream(0, &history, 2, &centers, &[0, 1], 19, &leaves, config)
                    .unwrap()
                {
                    check(score);
                }
            }
        }

        // A device-managed search supplies no host history to the draw upload.
        engine.start_state(3, 0.0).unwrap();
        copy_to(engine.state.as_ref().unwrap(), &[0u32, 3]);
        copy_to(&engine.scratch.history_slots, &[4u32, 6, 1]);
        copy_to(&engine.scratch.outcomes, &[0.0f32; 3]);
        let config = Ask {
            length: 0.0,
            neighbors: 2,
            acquisition: crate::weights::AcquisitionKind::Thompson,
            seed: 42,
            ..Ask::default()
        };
        let history = [(4, 0.0), (6, 0.0), (1, 0.0)];
        engine.sync_draws(config);
        let expected = crate::weights::thompson_history_draws(&history, config.seed);
        let draws = read_slice::<f32>(&engine.scratch.draws, 8);
        assert_eq!([draws[4], draws[6], draws[1]].as_slice(), expected);
    }
}
