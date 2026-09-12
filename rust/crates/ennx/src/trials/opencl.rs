use std::ptr;

use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_CPU, CL_DEVICE_TYPE_GPU};
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, ClMem, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE};
use opencl3::program::Program;
use opencl3::types::{cl_mem_flags, CL_BLOCKING};

use super::{make_steps, make_tiles, Ask, Center, LeafStep, Parameter, SparseEdit, Tile};

const THREADS: usize = 256;
const SOURCE: &str = include_str!("trials.cl");

#[repr(C)]
#[derive(Clone, Copy)]
struct Seed {
    low: u32,
    high: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Params {
    row_bytes: u32,
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
struct RowSumParams {
    row_bytes: u32,
    slot: u32,
    pad0: u32,
    pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MultiTrParams {
    num_regions: u32,
    candidates_per_region: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CenterStep {
    parent: u32,
    seed: Seed,
}

struct Scratch {
    params: std::sync::Arc<Buffer<Params>>,
    history_slots: Buffer<u32>,
    outcomes: Buffer<f32>,
    seeds: Buffer<Seed>,
    draws: Buffer<f32>,
    scores: Buffer<f32>,
    partials: Buffer<f32>,
    choice: Buffer<u32>,
    leaves: Buffer<LeafStep>,
    tiles: Buffer<Tile>,
    centers: Buffer<CenterStep>,
    candidate_centers: Buffer<u32>,
    edits: Buffer<SparseEdit>,
    base_distances: Buffer<f32>,
    row_sum: Buffer<u64>,
    selected_scores: Buffer<f32>,
    candidate_capacity: usize,
    center_capacity: usize,
    edit_capacity: usize,
}

/// Borrowed OpenCL row binding for a resident candidate.
///
/// The queue and context borrows keep the OpenCL runtime alive for as long as
/// the evaluator needs the row, while the cloned buffer handle keeps the device
/// allocation alive independently of the owning search.
pub(crate) struct ResidentRow<'a> {
    context: &'a Context,
    queue: &'a CommandQueue,
    rows: &'a Buffer<u8>,
    offset: usize,
    row_bytes: usize,
}

impl<'a> ResidentRow<'a> {
    pub fn context(&self) -> &'a Context {
        self.context
    }

    pub fn queue(&self) -> &'a CommandQueue {
        self.queue
    }

    pub fn buffer(&self) -> &'a Buffer<u8> {
        self.rows
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn row_bytes(&self) -> usize {
        self.row_bytes
    }

    #[allow(dead_code)]
    pub fn end(&self) -> usize {
        self.offset + self.row_bytes
    }
}

pub(super) struct Engine {
    pub(super) context: Context,
    queue: CommandQueue,
    rows: Buffer<u8>,
    row_bytes: usize,
    slots: usize,
    tile_count: usize,
    distance: Kernel,
    base_distance: Kernel,
    score: Kernel,
    score_sparse: Kernel,
    fill_seeds: Kernel,
    fill_edits: Kernel,
    pick: Kernel,
    multi_tr_pick: Kernel,
    write: Kernel,
    write_sparse: Kernel,
    row_sum: Kernel,
    scratch: Scratch,
    uploaded_history: Vec<(u32, u32)>,
    uploaded_leaves: Vec<LeafStep>,
    region: Option<std::sync::Arc<Buffer<u64>>>,
    radii: Buffer<f32>,
    leaf_count: usize,
    region_steps: Kernel,
    state: Option<Buffer<u32>>,
    capacity: usize,
    configure: Kernel,
    advance: Kernel,
    copy_row: Kernel,
    shadow: Buffer<u64>,
}

impl Engine {
    pub(super) fn new(base: &[u8], leaves: &[Parameter], slots: usize) -> Result<Self, String> {
        let device_id = get_all_devices(CL_DEVICE_TYPE_GPU)
            .map_err(|error| format!("failed to enumerate OpenCL GPU devices: {error}"))?
            .into_iter()
            .next()
            .or_else(|| {
                get_all_devices(CL_DEVICE_TYPE_CPU)
                    .ok()
                    .and_then(|devices| devices.into_iter().next())
            })
            .ok_or("no OpenCL GPU or CPU device found")?;
        let device = Device::new(device_id);
        let context = Context::from_device(&device)
            .map_err(|error| format!("failed to create OpenCL context: {error}"))?;
        let queue = CommandQueue::create_default(&context, 0)
            .map_err(|error| format!("failed to create OpenCL command queue: {error}"))?;
        let source = format!(
            "{SOURCE}\n{}\n{}",
            include_str!("../search/arithmetic.cl"),
            include_str!("region_steps.cl")
        );
        let source = format!(
            "{source}\n#define DEVICE __global\n{}\n{}",
            include_str!("../search/adaptation.cl"),
            include_str!("control.cl")
        );
        let program = Program::create_and_build_from_source(&context, &source, "")
            .map_err(|error| format!("failed to build OpenCL trial kernels: {error}"))?;
        let distance = Kernel::create(&program, "distance_trials")
            .map_err(|error| format!("missing OpenCL trial kernel distance_trials: {error}"))?;
        let base_distance = Kernel::create(&program, "base_distance")
            .map_err(|error| format!("missing OpenCL trial kernel base_distance: {error}"))?;
        let score = Kernel::create(&program, "score_trials")
            .map_err(|error| format!("missing OpenCL trial kernel score_trials: {error}"))?;
        let score_sparse = Kernel::create(&program, "score_sparse")
            .map_err(|error| format!("missing OpenCL trial kernel score_sparse: {error}"))?;
        let fill_seeds = Kernel::create(&program, "fill_seeds")
            .map_err(|error| format!("missing OpenCL trial kernel fill_seeds: {error}"))?;
        let fill_edits = Kernel::create(&program, "fill_edits")
            .map_err(|error| format!("missing OpenCL trial kernel fill_edits: {error}"))?;
        let pick = Kernel::create(&program, "pick_trial")
            .map_err(|error| format!("missing OpenCL trial kernel pick_trial: {error}"))?;
        let multi_tr_pick = Kernel::create(&program, "multi_tr_pick_trials").map_err(|error| {
            format!("missing OpenCL trial kernel multi_tr_pick_trials: {error}")
        })?;
        let write = Kernel::create(&program, "write_trial")
            .map_err(|error| format!("missing OpenCL trial kernel write_trial: {error}"))?;
        let write_sparse = Kernel::create(&program, "write_sparse")
            .map_err(|error| format!("missing OpenCL trial kernel write_sparse: {error}"))?;
        let row_sum = Kernel::create(&program, "row_sum")
            .map_err(|error| format!("missing OpenCL trial kernel row_sum: {error}"))?;
        let region_steps =
            Kernel::create(&program, "region_steps").map_err(|error| error.to_string())?;
        let mut radii = buffer(&context, leaves.len(), CL_MEM_READ_ONLY, "radii")?;
        unsafe {
            queue
                .enqueue_write_buffer(
                    &mut radii,
                    CL_BLOCKING,
                    0,
                    &leaves.iter().map(|leaf| leaf.radius).collect::<Vec<_>>(),
                    &[],
                )
                .map_err(|error| error.to_string())?;
        }
        let configure = Kernel::create(&program, "configure").map_err(|error| error.to_string())?;
        let advance = Kernel::create(&program, "advance").map_err(|error| error.to_string())?;
        let copy_row = Kernel::create(&program, "copy_row").map_err(|error| error.to_string())?;
        let shadow = buffer(&context, 16, CL_MEM_READ_WRITE, "region staging")?;
        let row_bytes = base.len();
        let tiles = make_tiles(leaves);
        let mut rows = buffer::<u8>(
            &context,
            slots.saturating_mul(row_bytes),
            CL_MEM_READ_WRITE,
            "model rows",
        )?;
        unsafe {
            queue
                .enqueue_write_buffer(&mut rows, CL_BLOCKING, 0, base, &[])
                .map_err(|error| format!("failed to write OpenCL base row: {error}"))?;
        }
        let mut scratch = Scratch {
            params: std::sync::Arc::new(buffer(
                &context,
                1,
                CL_MEM_READ_WRITE,
                "dispatch parameters",
            )?),
            history_slots: buffer(
                &context,
                super::MAX_HISTORY,
                CL_MEM_READ_WRITE,
                "history slots",
            )?,
            outcomes: buffer(&context, super::MAX_HISTORY, CL_MEM_READ_WRITE, "outcomes")?,
            seeds: buffer(&context, 1, CL_MEM_READ_ONLY, "seeds")?,
            draws: buffer(&context, slots, CL_MEM_READ_ONLY, "draws")?,
            scores: buffer(&context, 1, CL_MEM_READ_WRITE, "scores")?,
            partials: buffer(
                &context,
                super::MAX_HISTORY.saturating_mul(tiles.len()),
                CL_MEM_READ_WRITE,
                "partial distances",
            )?,
            choice: buffer(&context, 1, CL_MEM_READ_WRITE, "choice")?,
            leaves: buffer(&context, leaves.len(), CL_MEM_READ_WRITE, "leaves")?,
            tiles: buffer(&context, tiles.len(), CL_MEM_READ_ONLY, "tiles")?,
            centers: buffer(&context, 1, CL_MEM_READ_ONLY, "centers")?,
            candidate_centers: buffer(&context, 1, CL_MEM_READ_ONLY, "candidate centers")?,
            edits: buffer(&context, 1, CL_MEM_READ_ONLY, "sparse edits")?,
            base_distances: buffer(
                &context,
                super::MAX_HISTORY,
                CL_MEM_READ_WRITE,
                "base distances",
            )?,
            row_sum: buffer(&context, 1, CL_MEM_READ_WRITE, "row sum")?,
            selected_scores: buffer(&context, 1, CL_MEM_READ_WRITE, "selected scores")?,
            candidate_capacity: 1,
            center_capacity: 1,
            edit_capacity: 1,
        };
        unsafe {
            queue
                .enqueue_write_buffer(&mut scratch.tiles, CL_BLOCKING, 0, &tiles, &[])
                .map_err(|error| format!("failed to write OpenCL tiles: {error}"))?;
        }
        unsafe {
            queue
                .enqueue_write_buffer(
                    &mut scratch.leaves,
                    CL_BLOCKING,
                    0,
                    &make_steps(leaves, 0.0),
                    &[],
                )
                .map_err(|error| error.to_string())?;
        }
        Ok(Self {
            region: None,
            state: None,
            capacity: 0,
            configure,
            advance,
            copy_row,
            shadow,
            radii,
            leaf_count: leaves.len(),
            region_steps,
            context,
            queue,
            rows,
            row_bytes,
            slots,
            tile_count: tiles.len(),
            distance,
            base_distance,
            score,
            score_sparse,
            fill_seeds,
            fill_edits,
            pick,
            multi_tr_pick,
            write,
            write_sparse,
            row_sum,
            scratch,
            uploaded_history: Vec::new(),
            uploaded_leaves: Vec::new(),
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
        let distance_groups = seeds
            .len()
            .div_ceil(2)
            .checked_mul(self.tile_count)
            .ok_or("distance dispatch size overflow")?;
        let history_slots: Vec<u32> = history
            .iter()
            .map(|&(slot, _)| to_u32(slot, "history slot"))
            .collect::<Result<_, _>>()?;
        let outcomes: Vec<f32> = history.iter().map(|&(_, value)| value).collect();
        let seeds: Vec<Seed> = seeds
            .iter()
            .map(|&seed| Seed {
                low: seed as u32,
                high: (seed >> 32) as u32,
            })
            .collect();
        let steps = if self.region.is_some() {
            Vec::new()
        } else {
            make_steps(leaves, config.length)
        };
        self.write_inputs(&history_slots, &outcomes, &seeds, &steps)?;
        self.fill_draws(config)?;

        let params = Params {
            row_bytes: to_u32(self.row_bytes, "row bytes")?,
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
        let params = self.parameters(params)?;

        unsafe {
            ExecuteKernel::new(&self.distance)
                .set_arg(&self.rows)
                .set_arg(&self.scratch.history_slots)
                .set_arg(&self.scratch.seeds)
                .set_arg(&self.scratch.leaves)
                .set_arg(&self.scratch.tiles)
                .set_arg(&self.scratch.partials)
                .set_arg(&self.scratch.centers)
                .set_arg(&self.scratch.candidate_centers)
                .set_arg(params.as_ref())
                .set_global_work_size(
                    distance_groups
                        .checked_mul(THREADS)
                        .ok_or("distance work size overflow")?,
                )
                .set_local_work_size(THREADS)
                .enqueue_nd_range(&self.queue)
                .map_err(|error| format!("failed to launch OpenCL trial distances: {error}"))?;
            ExecuteKernel::new(&self.score)
                .set_arg(&self.scratch.partials)
                .set_arg(&self.scratch.outcomes)
                .set_arg(&self.scratch.draws)
                .set_arg(&self.scratch.scores)
                .set_arg(params.as_ref())
                .set_arg(&self.scratch.history_slots)
                .set_global_work_size(seeds.len() * THREADS)
                .set_local_work_size(THREADS)
                .enqueue_nd_range(&self.queue)
                .map_err(|error| format!("failed to launch OpenCL trial scoring: {error}"))?;
            ExecuteKernel::new(&self.pick)
                .set_arg(&self.scratch.scores)
                .set_arg(&self.scratch.choice)
                .set_arg(&self.scratch.selected_scores)
                .set_arg(params.as_ref())
                .set_global_work_size(THREADS)
                .set_local_work_size(THREADS)
                .enqueue_nd_range(&self.queue)
                .map_err(|error| format!("failed to launch OpenCL trial selection: {error}"))?;
            if materialize_row {
                ExecuteKernel::new(&self.write)
                    .set_arg(&self.rows)
                    .set_arg(&self.scratch.seeds)
                    .set_arg(&self.scratch.choice)
                    .set_arg(&self.scratch.leaves)
                    .set_arg(&self.scratch.tiles)
                    .set_arg(params.as_ref())
                    .set_global_work_size(self.tile_count * THREADS)
                    .set_local_work_size(THREADS)
                    .enqueue_nd_range(&self.queue)
                    .map_err(|error| format!("failed to launch OpenCL trial write: {error}"))?;
            }
        }

        let mut choice = [0u32];
        unsafe {
            self.queue
                .enqueue_read_buffer(&self.scratch.choice, CL_BLOCKING, 0, &mut choice, &[])
                .map_err(|error| format!("failed to read OpenCL trial choice: {error}"))?;
        }
        let index = choice[0] as usize;
        if index >= seeds.len() {
            return Err("OpenCL trial choice exceeds candidate count".into());
        }
        let mut score = [0.0f32];
        unsafe {
            self.queue
                .enqueue_read_buffer(
                    &self.scratch.selected_scores,
                    CL_BLOCKING,
                    0,
                    &mut score,
                    &[],
                )
                .map_err(|error| format!("failed to read OpenCL winning score: {error}"))?;
        }
        Ok((index, score[0]))
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
        let distance_groups = count
            .div_ceil(2)
            .checked_mul(self.tile_count)
            .ok_or("distance dispatch size overflow")?;
        let history_slots: Vec<u32> = history
            .iter()
            .map(|&(slot, _)| to_u32(slot, "history slot"))
            .collect::<Result<_, _>>()?;
        let outcomes: Vec<f32> = history.iter().map(|&(_, value)| value).collect();
        let steps = if self.region.is_some() {
            Vec::new()
        } else {
            make_steps(leaves, config.length)
        };
        self.write_static(&history_slots, &outcomes, &steps)?;
        self.fill_seeds(base_seed, count)?;
        self.fill_draws(config)?;

        let params = Params {
            row_bytes: to_u32(self.row_bytes, "row bytes")?,
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
        let params = self.parameters(params)?;

        unsafe {
            ExecuteKernel::new(&self.distance)
                .set_arg(&self.rows)
                .set_arg(&self.scratch.history_slots)
                .set_arg(&self.scratch.seeds)
                .set_arg(&self.scratch.leaves)
                .set_arg(&self.scratch.tiles)
                .set_arg(&self.scratch.partials)
                .set_arg(&self.scratch.centers)
                .set_arg(&self.scratch.candidate_centers)
                .set_arg(params.as_ref())
                .set_global_work_size(
                    distance_groups
                        .checked_mul(THREADS)
                        .ok_or("distance work size overflow")?,
                )
                .set_local_work_size(THREADS)
                .enqueue_nd_range(&self.queue)
                .map_err(|error| format!("failed to launch OpenCL trial distances: {error}"))?;
            ExecuteKernel::new(&self.score)
                .set_arg(&self.scratch.partials)
                .set_arg(&self.scratch.outcomes)
                .set_arg(&self.scratch.draws)
                .set_arg(&self.scratch.scores)
                .set_arg(params.as_ref())
                .set_arg(&self.scratch.history_slots)
                .set_global_work_size(count * THREADS)
                .set_local_work_size(THREADS)
                .enqueue_nd_range(&self.queue)
                .map_err(|error| format!("failed to launch OpenCL trial scoring: {error}"))?;
            ExecuteKernel::new(&self.pick)
                .set_arg(&self.scratch.scores)
                .set_arg(&self.scratch.choice)
                .set_arg(&self.scratch.selected_scores)
                .set_arg(params.as_ref())
                .set_global_work_size(THREADS)
                .set_local_work_size(THREADS)
                .enqueue_nd_range(&self.queue)
                .map_err(|error| format!("failed to launch OpenCL trial selection: {error}"))?;
            if materialize_row {
                ExecuteKernel::new(&self.write)
                    .set_arg(&self.rows)
                    .set_arg(&self.scratch.seeds)
                    .set_arg(&self.scratch.choice)
                    .set_arg(&self.scratch.leaves)
                    .set_arg(&self.scratch.tiles)
                    .set_arg(params.as_ref())
                    .set_global_work_size(self.tile_count * THREADS)
                    .set_local_work_size(THREADS)
                    .enqueue_nd_range(&self.queue)
                    .map_err(|error| format!("failed to launch OpenCL trial write: {error}"))?;
            }
        }

        let mut choice = [0u32];
        let mut score = [0.0f32];
        unsafe {
            self.queue
                .enqueue_read_buffer(&self.scratch.choice, CL_BLOCKING, 0, &mut choice, &[])
                .map_err(|error| format!("failed to read OpenCL trial choice: {error}"))?;
            self.queue
                .enqueue_read_buffer(
                    &self.scratch.selected_scores,
                    CL_BLOCKING,
                    0,
                    &mut score,
                    &[],
                )
                .map_err(|error| format!("failed to read OpenCL selected score: {error}"))?;
        }
        let index = choice[0] as usize;
        if index >= count {
            return Err("OpenCL trial choice exceeds candidate count".into());
        }
        Ok((index, super::cpu::seed_at(base_seed, choice[0]), score[0]))
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
            return Err("OpenCL sparse base and destination slots must differ".to_string());
        }
        if history_count == 0 || history_count > super::MAX_HISTORY {
            return Err(format!(
                "OpenCL sparse history must contain 1..={} rows",
                super::MAX_HISTORY
            ));
        }
        if seeds.is_empty() {
            return Err("OpenCL sparse trials require at least one candidate".to_string());
        }
        if num_pert == 0 {
            return Err("OpenCL sparse edit count must be positive".to_string());
        }
        if self.state.is_none() && edits.len() != seeds.len().saturating_mul(num_pert) {
            return Err("OpenCL sparse edit count does not match candidates".to_string());
        }
        if config.neighbors == 0 || config.neighbors > history_count {
            return Err("OpenCL sparse neighbor count exceeds resident history".to_string());
        }
        self.ensure_candidates(seeds.len())?;
        self.ensure_edits(seeds.len().saturating_mul(num_pert))?;
        let history_slots: Vec<u32> = history
            .iter()
            .map(|&(slot, _)| to_u32(slot, "history slot"))
            .collect::<Result<_, _>>()?;
        let outcomes: Vec<f32> = history.iter().map(|&(_, value)| value).collect();
        let seeds: Vec<Seed> = seeds
            .iter()
            .map(|&seed| Seed {
                low: seed as u32,
                high: (seed >> 32) as u32,
            })
            .collect();
        let steps = if self.region.is_some() {
            Vec::new()
        } else {
            make_steps(leaves, config.length)
        };
        self.write_inputs(&history_slots, &outcomes, &seeds, &steps)?;
        self.fill_draws(config)?;
        if self.state.is_none() {
            unsafe {
                self.queue
                    .enqueue_write_buffer(&mut self.scratch.edits, CL_BLOCKING, 0, edits, &[])
                    .map_err(|error| format!("failed to write OpenCL sparse edits: {error}"))?;
            }
        }

        let params = Params {
            row_bytes: to_u32(self.row_bytes, "row bytes")?,
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
        let params = self.parameters(params)?;
        if self.state.is_some() {
            unsafe {
                ExecuteKernel::new(&self.fill_edits)
                    .set_arg(&self.scratch.edits)
                    .set_arg(&self.scratch.seeds)
                    .set_arg(&self.scratch.leaves)
                    .set_arg(params.as_ref())
                    .set_global_work_size(seeds.len())
                    .set_local_work_size(1)
                    .enqueue_nd_range(&self.queue)
                    .map_err(|error| error.to_string())?;
            }
        }

        unsafe {
            ExecuteKernel::new(&self.base_distance)
                .set_arg(&self.rows)
                .set_arg(&self.scratch.history_slots)
                .set_arg(&self.scratch.leaves)
                .set_arg(&self.scratch.base_distances)
                .set_arg(params.as_ref())
                .set_global_work_size(history_count * THREADS)
                .set_local_work_size(THREADS)
                .enqueue_nd_range(&self.queue)
                .map_err(|error| {
                    format!("failed to launch OpenCL sparse base distances: {error}")
                })?;
            ExecuteKernel::new(&self.score_sparse)
                .set_arg(&self.rows)
                .set_arg(&self.scratch.history_slots)
                .set_arg(&self.scratch.outcomes)
                .set_arg(&self.scratch.seeds)
                .set_arg(&self.scratch.draws)
                .set_arg(&self.scratch.leaves)
                .set_arg(&self.scratch.edits)
                .set_arg(&self.scratch.base_distances)
                .set_arg(&self.scratch.scores)
                .set_arg(params.as_ref())
                .set_global_work_size(seeds.len() * THREADS)
                .set_local_work_size(THREADS)
                .enqueue_nd_range(&self.queue)
                .map_err(|error| format!("failed to launch OpenCL sparse scoring: {error}"))?;
            ExecuteKernel::new(&self.pick)
                .set_arg(&self.scratch.scores)
                .set_arg(&self.scratch.choice)
                .set_arg(&self.scratch.selected_scores)
                .set_arg(params.as_ref())
                .set_global_work_size(THREADS)
                .set_local_work_size(THREADS)
                .enqueue_nd_range(&self.queue)
                .map_err(|error| format!("failed to launch OpenCL sparse selection: {error}"))?;
        }

        if self.state.is_none() {
            let (source, mut destination) = unsafe {
                let source = self
                    .rows
                    .create_sub_buffer(CL_MEM_READ_ONLY, base_slot * self.row_bytes, self.row_bytes)
                    .map_err(|error| {
                        format!("failed to create OpenCL sparse source row: {error}")
                    })?;
                let destination = self
                    .rows
                    .create_sub_buffer(
                        CL_MEM_READ_WRITE,
                        trial_slot * self.row_bytes,
                        self.row_bytes,
                    )
                    .map_err(|error| {
                        format!("failed to create OpenCL sparse destination row: {error}")
                    })?;
                (source, destination)
            };
            unsafe {
                self.queue
                    .enqueue_copy_buffer(&source, &mut destination, 0, 0, self.row_bytes, &[])
                    .map_err(|error| format!("failed to copy OpenCL sparse trial row: {error}"))?;
            }
        }

        unsafe {
            ExecuteKernel::new(&self.write_sparse)
                .set_arg(&self.rows)
                .set_arg(&self.scratch.seeds)
                .set_arg(&self.scratch.choice)
                .set_arg(&self.scratch.leaves)
                .set_arg(&self.scratch.edits)
                .set_arg(params.as_ref())
                .set_global_work_size(1)
                .set_local_work_size(1)
                .enqueue_nd_range(&self.queue)
                .map_err(|error| format!("failed to launch OpenCL sparse write: {error}"))?;
        }

        let mut choice = [0u32];
        unsafe {
            self.queue
                .enqueue_read_buffer(&self.scratch.choice, CL_BLOCKING, 0, &mut choice, &[])
                .map_err(|error| format!("failed to read OpenCL sparse choice: {error}"))?;
        }
        let index = choice[0] as usize;
        if index >= seeds.len() {
            return Err("OpenCL sparse trial choice exceeds candidate count".into());
        }
        let mut score = [0.0f32];
        unsafe {
            self.queue
                .enqueue_read_buffer(
                    &self.scratch.selected_scores,
                    CL_BLOCKING,
                    0,
                    &mut score,
                    &[],
                )
                .map_err(|error| format!("failed to read OpenCL sparse winning score: {error}"))?;
        }
        Ok((index, score[0]))
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
            return Err("OpenCL sparse base and destination slots must differ".to_string());
        }
        if history_count == 0 || history_count > super::MAX_HISTORY {
            return Err(format!(
                "OpenCL sparse history must contain 1..={} rows",
                super::MAX_HISTORY
            ));
        }
        if config.neighbors == 0 || config.neighbors > history_count {
            return Err("OpenCL sparse neighbor count exceeds resident history".to_string());
        }
        let dimensions = leaves.iter().map(|leaf| leaf.length).sum::<usize>();
        self.ensure_candidates(count)?;
        self.ensure_edits(count.saturating_mul(num_pert))?;
        let history_slots: Vec<u32> = history
            .iter()
            .map(|&(slot, _)| to_u32(slot, "history slot"))
            .collect::<Result<_, _>>()?;
        let outcomes: Vec<f32> = history.iter().map(|&(_, value)| value).collect();
        let steps = if self.region.is_some() {
            Vec::new()
        } else {
            make_steps(leaves, config.length)
        };
        self.write_static(&history_slots, &outcomes, &steps)?;
        self.fill_seeds(base_seed, count)?;
        self.fill_draws(config)?;

        let params = Params {
            row_bytes: to_u32(self.row_bytes, "row bytes")?,
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
        let params = self.parameters(params)?;

        if self.state.is_none() {
            let (source, mut destination) = unsafe {
                let source = self
                    .rows
                    .create_sub_buffer(CL_MEM_READ_ONLY, base_slot * self.row_bytes, self.row_bytes)
                    .map_err(|error| {
                        format!("failed to create OpenCL sparse source row: {error}")
                    })?;
                let destination = self
                    .rows
                    .create_sub_buffer(
                        CL_MEM_READ_WRITE,
                        trial_slot * self.row_bytes,
                        self.row_bytes,
                    )
                    .map_err(|error| {
                        format!("failed to create OpenCL sparse destination row: {error}")
                    })?;
                (source, destination)
            };

            unsafe {
                self.queue
                    .enqueue_copy_buffer(&source, &mut destination, 0, 0, self.row_bytes, &[])
                    .map_err(|error| format!("failed to copy OpenCL sparse trial row: {error}"))?;
            }
        }
        unsafe {
            ExecuteKernel::new(&self.fill_edits)
                .set_arg(&self.scratch.edits)
                .set_arg(&self.scratch.seeds)
                .set_arg(&self.scratch.leaves)
                .set_arg(params.as_ref())
                .set_global_work_size(count)
                .set_local_work_size(1)
                .enqueue_nd_range(&self.queue)
                .map_err(|error| format!("failed to launch OpenCL sparse edit fill: {error}"))?;
            ExecuteKernel::new(&self.base_distance)
                .set_arg(&self.rows)
                .set_arg(&self.scratch.history_slots)
                .set_arg(&self.scratch.leaves)
                .set_arg(&self.scratch.base_distances)
                .set_arg(params.as_ref())
                .set_global_work_size(history_count * THREADS)
                .set_local_work_size(THREADS)
                .enqueue_nd_range(&self.queue)
                .map_err(|error| {
                    format!("failed to launch OpenCL sparse base distances: {error}")
                })?;
            ExecuteKernel::new(&self.score_sparse)
                .set_arg(&self.rows)
                .set_arg(&self.scratch.history_slots)
                .set_arg(&self.scratch.outcomes)
                .set_arg(&self.scratch.seeds)
                .set_arg(&self.scratch.draws)
                .set_arg(&self.scratch.leaves)
                .set_arg(&self.scratch.edits)
                .set_arg(&self.scratch.base_distances)
                .set_arg(&self.scratch.scores)
                .set_arg(params.as_ref())
                .set_global_work_size(count * THREADS)
                .set_local_work_size(THREADS)
                .enqueue_nd_range(&self.queue)
                .map_err(|error| format!("failed to launch OpenCL sparse scoring: {error}"))?;
            ExecuteKernel::new(&self.pick)
                .set_arg(&self.scratch.scores)
                .set_arg(&self.scratch.choice)
                .set_arg(&self.scratch.selected_scores)
                .set_arg(params.as_ref())
                .set_global_work_size(THREADS)
                .set_local_work_size(THREADS)
                .enqueue_nd_range(&self.queue)
                .map_err(|error| format!("failed to launch OpenCL sparse selection: {error}"))?;
            ExecuteKernel::new(&self.write_sparse)
                .set_arg(&self.rows)
                .set_arg(&self.scratch.seeds)
                .set_arg(&self.scratch.choice)
                .set_arg(&self.scratch.leaves)
                .set_arg(&self.scratch.edits)
                .set_arg(params.as_ref())
                .set_global_work_size(1)
                .set_local_work_size(1)
                .enqueue_nd_range(&self.queue)
                .map_err(|error| format!("failed to launch OpenCL sparse write: {error}"))?;
        }

        let mut choice = [0u32];
        let mut score = [0.0f32];
        unsafe {
            self.queue
                .enqueue_read_buffer(&self.scratch.choice, CL_BLOCKING, 0, &mut choice, &[])
                .map_err(|error| format!("failed to read OpenCL sparse choice: {error}"))?;
            self.queue
                .enqueue_read_buffer(
                    &self.scratch.selected_scores,
                    CL_BLOCKING,
                    0,
                    &mut score,
                    &[],
                )
                .map_err(|error| format!("failed to read OpenCL sparse winning score: {error}"))?;
        }
        let index = choice[0] as usize;
        if index >= count {
            return Err("OpenCL sparse trial choice exceeds candidate count".into());
        }
        Ok((index, super::cpu::seed_at(base_seed, choice[0]), score[0]))
    }

    pub(super) fn materialize(
        &mut self,
        base_slot: usize,
        trial_slot: usize,
        seed: u64,
        steps: &[LeafStep],
    ) -> Result<(), String> {
        self.ensure_candidates(1)?;
        let seeds = [Seed {
            low: seed as u32,
            high: (seed >> 32) as u32,
        }];
        let choice = [0_u32];
        unsafe {
            self.queue
                .enqueue_write_buffer(&mut self.scratch.seeds, CL_BLOCKING, 0, &seeds, &[])
                .map_err(|error| format!("failed to write OpenCL trial seed: {error}"))?;
            self.queue
                .enqueue_write_buffer(&mut self.scratch.choice, CL_BLOCKING, 0, &choice, &[])
                .map_err(|error| format!("failed to write OpenCL trial choice: {error}"))?;
        }
        self.sync_leaves(steps)?;

        let params = Params {
            row_bytes: to_u32(self.row_bytes, "row bytes")?,
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
        let params = self.parameters(params)?;
        unsafe {
            ExecuteKernel::new(&self.write)
                .set_arg(&self.rows)
                .set_arg(&self.scratch.seeds)
                .set_arg(&self.scratch.choice)
                .set_arg(&self.scratch.leaves)
                .set_arg(&self.scratch.tiles)
                .set_arg(params.as_ref())
                .set_global_work_size(self.tile_count * THREADS)
                .set_local_work_size(THREADS)
                .enqueue_nd_range(&self.queue)
                .map_err(|error| format!("failed to launch OpenCL trial write: {error}"))?;
        }
        self.queue
            .finish()
            .map_err(|error| format!("failed to finish OpenCL trial write: {error}"))
    }

    #[allow(clippy::too_many_arguments)]
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
        let distance_groups = total_candidates
            .div_ceil(2)
            .checked_mul(self.tile_count)
            .ok_or("distance dispatch size overflow")?;
        let history_slots: Vec<u32> = history
            .iter()
            .map(|&(slot, _)| to_u32(slot, "history slot"))
            .collect::<Result<_, _>>()?;
        let outcomes: Vec<f32> = history.iter().map(|&(_, value)| value).collect();
        let packed_seeds: Vec<Seed> = seeds
            .iter()
            .map(|&seed| Seed {
                low: seed as u32,
                high: (seed >> 32) as u32,
            })
            .collect();
        let steps = if self.region.is_some() {
            Vec::new()
        } else {
            make_steps(leaves, config.length)
        };
        if let Some(seed) = base_seed {
            self.write_static(&history_slots, &outcomes, &steps)?;
            self.fill_seeds(seed, total_candidates)?;
        } else {
            self.write_inputs(&history_slots, &outcomes, &packed_seeds, &steps)?;
        }
        self.fill_draws(config)?;
        let center_count = match tree {
            Some((centers, region_centers)) => {
                self.write_centers(centers, region_centers, seeds_per_region)?;
                centers.len()
            }
            None => 0,
        };
        let params = Params {
            row_bytes: to_u32(self.row_bytes, "row bytes")?,
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
        let params = self.parameters(params)?;
        self.run_scores(&params, distance_groups, total_candidates)?;
        self.pick_regions(num_regions, seeds_per_region)
    }

    pub(super) fn read(&self, slot: usize, row_bytes: usize) -> Result<Vec<u8>, String> {
        let mut row = vec![0u8; row_bytes];
        unsafe {
            self.queue
                .enqueue_read_buffer(&self.rows, CL_BLOCKING, slot * row_bytes, &mut row, &[])
                .map_err(|error| format!("failed to read OpenCL trial row: {error}"))?
                .wait()
                .map_err(|error| format!("failed waiting for OpenCL trial row: {error}"))?;
        }
        Ok(row)
    }

    pub(super) fn device_view(&self, slot: usize) -> Result<ResidentRow<'_>, String> {
        if self.row_bytes == 0 {
            return Err("model row bytes must be positive".to_string());
        }
        let allocation_len = self
            .rows
            .size()
            .map_err(|error| format!("failed to query OpenCL model row size: {error}"))?
            as usize;
        let offset = slot
            .checked_mul(self.row_bytes)
            .ok_or("model row offset overflow")?;
        let end = offset
            .checked_add(self.row_bytes)
            .ok_or("model row extent overflow")?;
        if end > allocation_len {
            return Err(format!("model row slot {slot} is out of range"));
        }
        Ok(ResidentRow {
            context: &self.context,
            queue: &self.queue,
            rows: &self.rows,
            offset,
            row_bytes: self.row_bytes,
        })
    }

    pub(super) fn byte_sum(&self, slot: usize) -> Result<u64, String> {
        self.device_view(slot)?;
        let params = RowSumParams {
            row_bytes: to_u32(self.row_bytes, "row bytes")?,
            slot: to_u32(slot, "row slot")?,
            pad0: 0,
            pad1: 0,
        };
        unsafe {
            ExecuteKernel::new(&self.row_sum)
                .set_arg(&self.rows)
                .set_arg(&self.scratch.row_sum)
                .set_arg(&params)
                .set_global_work_size(THREADS)
                .set_local_work_size(THREADS)
                .enqueue_nd_range(&self.queue)
                .map_err(|error| format!("failed to launch OpenCL resident row sum: {error}"))?;
        }
        let mut sum = [0u64];
        unsafe {
            self.queue
                .enqueue_read_buffer(&self.scratch.row_sum, CL_BLOCKING, 0, &mut sum, &[])
                .map_err(|error| format!("failed to read OpenCL resident row sum: {error}"))?;
        }
        Ok(sum[0])
    }

    pub(super) fn write(&mut self, slot: usize, row: &[u8]) -> Result<(), String> {
        unsafe {
            self.queue
                .enqueue_write_buffer(&mut self.rows, CL_BLOCKING, slot * self.row_bytes, row, &[])
                .map_err(|error| format!("failed to write OpenCL trial row: {error}"))?;
        }
        Ok(())
    }

    fn ensure_candidates(&mut self, count: usize) -> Result<(), String> {
        if count <= self.scratch.candidate_capacity {
            return Ok(());
        }
        let capacity = count.next_power_of_two();
        self.scratch.seeds = buffer(&self.context, capacity, CL_MEM_READ_ONLY, "seeds")?;
        self.scratch.scores = buffer(&self.context, capacity, CL_MEM_READ_WRITE, "scores")?;
        self.scratch.choice = buffer(&self.context, capacity, CL_MEM_READ_WRITE, "choices")?;
        self.scratch.selected_scores = buffer(
            &self.context,
            capacity,
            CL_MEM_READ_WRITE,
            "selected scores",
        )?;
        self.scratch.candidate_centers = buffer(
            &self.context,
            capacity,
            CL_MEM_READ_ONLY,
            "candidate centers",
        )?;
        let partial_count = capacity
            .checked_mul(super::MAX_HISTORY)
            .and_then(|value| value.checked_mul(self.tile_count))
            .ok_or("partial distance buffer size overflow")?;
        self.scratch.partials = buffer(
            &self.context,
            partial_count,
            CL_MEM_READ_WRITE,
            "partial distances",
        )?;
        self.scratch.candidate_capacity = capacity;
        Ok(())
    }

    fn ensure_edits(&mut self, count: usize) -> Result<(), String> {
        if count == 0 {
            return Err("sparse edit buffer cannot be empty".to_string());
        }
        if count <= self.scratch.edit_capacity {
            return Ok(());
        }
        let capacity = count.next_power_of_two();
        self.scratch.edits = buffer(&self.context, capacity, CL_MEM_READ_ONLY, "sparse edits")?;
        self.scratch.edit_capacity = capacity;
        Ok(())
    }

    fn run_scores(
        &self,
        params: &Buffer<Params>,
        distance_groups: usize,
        candidates: usize,
    ) -> Result<(), String> {
        unsafe {
            ExecuteKernel::new(&self.distance)
                .set_arg(&self.rows)
                .set_arg(&self.scratch.history_slots)
                .set_arg(&self.scratch.seeds)
                .set_arg(&self.scratch.leaves)
                .set_arg(&self.scratch.tiles)
                .set_arg(&self.scratch.partials)
                .set_arg(&self.scratch.centers)
                .set_arg(&self.scratch.candidate_centers)
                .set_arg(params)
                .set_global_work_size(
                    distance_groups
                        .checked_mul(THREADS)
                        .ok_or("distance work size overflow")?,
                )
                .set_local_work_size(THREADS)
                .enqueue_nd_range(&self.queue)
                .map_err(|error| format!("failed to launch OpenCL trial distances: {error}"))?;
            ExecuteKernel::new(&self.score)
                .set_arg(&self.scratch.partials)
                .set_arg(&self.scratch.outcomes)
                .set_arg(&self.scratch.draws)
                .set_arg(&self.scratch.scores)
                .set_arg(params)
                .set_arg(&self.scratch.history_slots)
                .set_global_work_size(candidates * THREADS)
                .set_local_work_size(THREADS)
                .enqueue_nd_range(&self.queue)
                .map_err(|error| format!("failed to launch OpenCL trial scoring: {error}"))?;
        }
        Ok(())
    }

    fn pick_regions(
        &self,
        num_regions: usize,
        seeds_per_region: usize,
    ) -> Result<Vec<(usize, f32)>, String> {
        let params = MultiTrParams {
            num_regions: to_u32(num_regions, "region count")?,
            candidates_per_region: to_u32(seeds_per_region, "candidates per region")?,
        };
        unsafe {
            ExecuteKernel::new(&self.multi_tr_pick)
                .set_arg(&self.scratch.scores)
                .set_arg(&self.scratch.choice)
                .set_arg(&self.scratch.selected_scores)
                .set_arg(&params)
                .set_global_work_size(
                    num_regions
                        .checked_mul(THREADS)
                        .ok_or("multi-TR selection work size overflow")?,
                )
                .set_local_work_size(THREADS)
                .enqueue_nd_range(&self.queue)
                .map_err(|error| format!("failed to launch OpenCL multi-TR selection: {error}"))?;
        }
        let mut choices = vec![0_u32; num_regions];
        let mut scores = vec![0.0_f32; num_regions];
        unsafe {
            self.queue
                .enqueue_read_buffer(&self.scratch.choice, CL_BLOCKING, 0, &mut choices, &[])
                .map_err(|error| format!("failed to read OpenCL multi-TR choices: {error}"))?;
            self.queue
                .enqueue_read_buffer(
                    &self.scratch.selected_scores,
                    CL_BLOCKING,
                    0,
                    &mut scores,
                    &[],
                )
                .map_err(|error| {
                    format!("failed to read OpenCL multi-TR selected scores: {error}")
                })?;
        }
        Ok(choices
            .into_iter()
            .zip(scores)
            .map(|(index, score)| (index as usize, score))
            .collect())
    }

    fn write_centers(
        &mut self,
        centers: &[Center],
        region_centers: &[usize],
        candidates_per_region: usize,
    ) -> Result<(), String> {
        if centers.len() > self.scratch.center_capacity {
            let capacity = centers.len().next_power_of_two();
            self.scratch.centers = buffer(&self.context, capacity, CL_MEM_READ_ONLY, "centers")?;
            self.scratch.center_capacity = capacity;
        }
        let steps: Vec<CenterStep> = centers
            .iter()
            .map(|center| {
                Ok(CenterStep {
                    parent: center
                        .parent
                        .map(|parent| to_u32(parent, "center parent"))
                        .transpose()?
                        .unwrap_or(u32::MAX),
                    seed: Seed {
                        low: center.seed as u32,
                        high: (center.seed >> 32) as u32,
                    },
                })
            })
            .collect::<Result<_, String>>()?;
        let region_centers: Vec<u32> = region_centers
            .iter()
            .map(|&center| to_u32(center, "region center"))
            .collect::<Result<_, _>>()?;
        let candidate_centers: Vec<u32> = region_centers
            .into_iter()
            .flat_map(|center| std::iter::repeat_n(center, candidates_per_region))
            .collect();
        unsafe {
            self.queue
                .enqueue_write_buffer(&mut self.scratch.centers, CL_BLOCKING, 0, &steps, &[])
                .map_err(|error| format!("failed to write OpenCL centers: {error}"))?;
            self.queue
                .enqueue_write_buffer(
                    &mut self.scratch.candidate_centers,
                    CL_BLOCKING,
                    0,
                    &candidate_centers,
                    &[],
                )
                .map_err(|error| format!("failed to write OpenCL candidate centers: {error}"))?;
        }
        Ok(())
    }

    fn write_inputs(
        &mut self,
        history_slots: &[u32],
        outcomes: &[f32],
        seeds: &[Seed],
        leaves: &[LeafStep],
    ) -> Result<(), String> {
        self.write_static(history_slots, outcomes, leaves)?;
        unsafe {
            self.queue
                .enqueue_write_buffer(&mut self.scratch.seeds, CL_BLOCKING, 0, seeds, &[])
                .map_err(|error| format!("failed to write OpenCL trial seeds: {error}"))?;
        }
        Ok(())
    }

    fn write_static(
        &mut self,
        history_slots: &[u32],
        outcomes: &[f32],
        leaves: &[LeafStep],
    ) -> Result<(), String> {
        // Borrowed host slices must remain alive until each upload completes,
        // including when a later enqueue fails. Keep blocking writes until an
        // event-owned staging abstraction can guarantee that on every exit.
        let prefix = self
            .uploaded_history
            .iter()
            .zip(history_slots.iter().zip(outcomes))
            .take_while(|((old_slot, old_value), (slot, value))| {
                old_slot == *slot && *old_value == value.to_bits()
            })
            .count();
        unsafe {
            if self.state.is_none() && prefix < history_slots.len() {
                self.queue
                    .enqueue_write_buffer(
                        &mut self.scratch.history_slots,
                        CL_BLOCKING,
                        prefix * std::mem::size_of::<u32>(),
                        &history_slots[prefix..],
                        &[],
                    )
                    .map_err(|error| format!("failed to write OpenCL history slots: {error}"))?;
                self.queue
                    .enqueue_write_buffer(
                        &mut self.scratch.outcomes,
                        CL_BLOCKING,
                        prefix * std::mem::size_of::<f32>(),
                        &outcomes[prefix..],
                        &[],
                    )
                    .map_err(|error| format!("failed to write OpenCL outcomes: {error}"))?;
            }
        }
        self.sync_leaves(leaves)?;
        self.uploaded_history = history_slots
            .iter()
            .zip(outcomes)
            .map(|(&slot, value)| (slot, value.to_bits()))
            .collect();
        Ok(())
    }

    fn fill_seeds(&self, base_seed: u64, count: usize) -> Result<(), String> {
        let seed = Seed {
            low: base_seed as u32,
            high: (base_seed >> 32) as u32,
        };
        unsafe {
            ExecuteKernel::new(&self.fill_seeds)
                .set_arg(&self.scratch.seeds)
                .set_arg(&seed)
                .set_arg(&to_u32(count, "candidate count")?)
                .set_global_work_size(count)
                .set_local_work_size(1)
                .enqueue_nd_range(&self.queue)
                .map_err(|error| format!("failed to launch OpenCL seed fill: {error}"))?;
        }
        Ok(())
    }

    fn fill_draws(&mut self, config: Ask) -> Result<(), String> {
        if config.acquisition != crate::weights::AcquisitionKind::Thompson {
            return Ok(());
        }
        // Cover every resident slot without reading device-managed history.
        let draws = crate::weights::thompson_draws(self.slots, config.seed);
        unsafe {
            self.queue
                .enqueue_write_buffer(&mut self.scratch.draws, CL_BLOCKING, 0, &draws, &[])
                .map_err(|error| format!("failed to write OpenCL Thompson draws: {error}"))?;
        }
        Ok(())
    }

    pub(super) fn bind_region(&mut self, region: std::sync::Arc<Buffer<u64>>) {
        self.region = Some(region);
    }

    fn sync_leaves(&mut self, leaves: &[LeafStep]) -> Result<(), String> {
        if let Some(region) = &self.region {
            unsafe {
                ExecuteKernel::new(&self.region_steps)
                    .set_arg(region.as_ref())
                    .set_arg(&self.scratch.leaves)
                    .set_arg(&self.radii)
                    .set_global_work_size(self.leaf_count)
                    .enqueue_nd_range(&self.queue)
                    .map_err(|error| error.to_string())?;
            }
            return Ok(());
        }
        if self.uploaded_leaves == leaves {
            return Ok(());
        }
        unsafe {
            self.queue
                .enqueue_write_buffer(&mut self.scratch.leaves, CL_BLOCKING, 0, leaves, &[])
                .map_err(|error| format!("failed to write OpenCL leaves: {error}"))?;
        }
        self.uploaded_leaves.clear();
        self.uploaded_leaves.extend_from_slice(leaves);
        Ok(())
    }
}

fn buffer<T>(
    context: &Context,
    len: usize,
    flags: cl_mem_flags,
    name: &str,
) -> Result<Buffer<T>, String> {
    if len == 0 {
        return Err(format!("{name} buffer cannot be empty"));
    }
    unsafe {
        Buffer::<T>::create(context, flags, len, ptr::null_mut())
            .map_err(|error| format!("failed to allocate OpenCL {name}: {error}"))
    }
}

fn to_u32(value: usize, name: &str) -> Result<u32, String> {
    u32::try_from(value).map_err(|_| format!("{name} exceeds u32 range"))
}

impl Engine {
    pub(super) fn start_state(&mut self, capacity: usize, value: f32) -> Result<(), String> {
        let mut values = [u32::MAX; 36];
        let total = self.rows.size().map_err(|error| error.to_string())? / self.row_bytes;
        values[..4].copy_from_slice(&[0, 1, capacity as u32, total as u32]);
        let mut state = buffer(&self.context, 36, CL_MEM_READ_WRITE, "search state")?;
        unsafe {
            self.queue
                .enqueue_write_buffer(&mut state, CL_BLOCKING, 0, &values, &[])
                .map_err(|error| error.to_string())?;
            self.queue
                .enqueue_write_buffer(
                    &mut self.scratch.history_slots,
                    CL_BLOCKING,
                    0,
                    &[0u32],
                    &[],
                )
                .map_err(|error| error.to_string())?;
            self.queue
                .enqueue_write_buffer(&mut self.scratch.outcomes, CL_BLOCKING, 0, &[value], &[])
                .map_err(|error| error.to_string())?;
        }
        self.state = Some(state);
        self.capacity = capacity;
        Ok(())
    }

    fn parameters(&mut self, value: Params) -> Result<std::sync::Arc<Buffer<Params>>, String> {
        let params = std::sync::Arc::get_mut(&mut self.scratch.params)
            .ok_or("dispatch parameters are still borrowed")?;
        unsafe {
            self.queue
                .enqueue_write_buffer(params, CL_BLOCKING, 0, &[value], &[])
                .map_err(|error| error.to_string())?;
            if let Some(state) = &self.state {
                ExecuteKernel::new(&self.configure)
                    .set_arg(state)
                    .set_arg(&self.scratch.history_slots)
                    .set_arg(params)
                    .set_arg(&self.scratch.choice)
                    .set_global_work_size(1)
                    .enqueue_nd_range(&self.queue)
                    .map_err(|error| error.to_string())?;
                ExecuteKernel::new(&self.copy_row)
                    .set_arg(&self.rows)
                    .set_arg(params)
                    .set_global_work_size(self.row_bytes)
                    .enqueue_nd_range(&self.queue)
                    .map_err(|error| error.to_string())?;
            }
        }
        Ok(self.scratch.params.clone())
    }

    pub(super) fn observe(&self, handle: usize, value: f32) -> Result<(), String> {
        let state = self.state.as_ref().ok_or("search state is not resident")?;
        let region = self.region.as_ref().ok_or("region is not resident")?;
        unsafe {
            ExecuteKernel::new(&self.advance)
                .set_arg(state)
                .set_arg(&self.scratch.history_slots)
                .set_arg(&self.scratch.outcomes)
                .set_arg(region.as_ref())
                .set_arg(&self.shadow)
                .set_arg(&(handle as u32))
                .set_arg(&f64::from(value).to_bits())
                .set_global_work_size(1)
                .enqueue_nd_range(&self.queue)
                .map_err(|error| error.to_string())?
                .wait()
                .map_err(|error| error.to_string())?;
        }
        Ok(())
    }

    pub(super) fn state_word(&self, index: usize) -> Result<u32, String> {
        let state = self.state.as_ref().ok_or("search state is not resident")?;
        let mut word = [0u32];
        unsafe {
            self.queue
                .enqueue_read_buffer(state, CL_BLOCKING, index * size_of::<u32>(), &mut word, &[])
                .map_err(|error| error.to_string())?;
        }
        Ok(word[0])
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
            Err(error)
                if error.contains("no OpenCL GPU or CPU device")
                    || error.contains("CL_PLATFORM_NOT_FOUND_KHR")
                    || error.contains("failed to enumerate OpenCL") =>
            {
                return
            }
            Err(error) => panic!("{error}"),
        };
        for (slot, value) in [(1, 129), (4, 131), (6, 132)] {
            engine.write(slot, &[value]).unwrap();
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
        unsafe {
            engine
                .queue
                .enqueue_write_buffer(
                    engine.state.as_mut().unwrap(),
                    CL_BLOCKING,
                    0,
                    &[0u32, 3],
                    &[],
                )
                .unwrap();
            engine
                .queue
                .enqueue_write_buffer(
                    &mut engine.scratch.history_slots,
                    CL_BLOCKING,
                    0,
                    &[4u32, 6, 1],
                    &[],
                )
                .unwrap();
            engine
                .queue
                .enqueue_write_buffer(
                    &mut engine.scratch.outcomes,
                    CL_BLOCKING,
                    0,
                    &[0.0f32; 3],
                    &[],
                )
                .unwrap();
        }
        let config = Ask {
            length: 0.0,
            neighbors: 2,
            acquisition: crate::weights::AcquisitionKind::Thompson,
            seed: 42,
            ..Ask::default()
        };
        let history = [(4, 0.0), (6, 0.0), (1, 0.0)];
        engine.fill_draws(config).unwrap();
        let expected = crate::weights::thompson_history_draws(&history, config.seed);
        let mut actual = [0.0f32; 8];
        unsafe {
            engine
                .queue
                .enqueue_read_buffer(&engine.scratch.draws, CL_BLOCKING, 0, &mut actual, &[])
                .unwrap();
        }
        assert_eq!([actual[4], actual[6], actual[1]].as_slice(), expected);
    }
}
