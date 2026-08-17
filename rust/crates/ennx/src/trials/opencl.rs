use std::ptr;

use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_CPU, CL_DEVICE_TYPE_GPU};
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE};
use opencl3::program::Program;
use opencl3::types::{cl_mem_flags, CL_BLOCKING, CL_NON_BLOCKING};

use super::{make_steps, make_tiles, Ask, Center, Leaf, LeafStep, Tile};

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
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CenterStep {
    parent: u32,
    seed: Seed,
}

struct Scratch {
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
    candidate_capacity: usize,
    center_capacity: usize,
}

pub(super) struct Engine {
    context: Context,
    queue: CommandQueue,
    rows: Buffer<u8>,
    row_bytes: usize,
    tile_count: usize,
    distance: Kernel,
    score: Kernel,
    pick: Kernel,
    write: Kernel,
    scratch: Scratch,
}

impl Engine {
    pub(super) fn new(base: &[u8], leaves: &[Leaf], slots: usize) -> Result<Self, String> {
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
        let program = Program::create_and_build_from_source(&context, SOURCE, "")
            .map_err(|error| format!("failed to build OpenCL trial kernels: {error}"))?;
        let distance = Kernel::create(&program, "distance_trials")
            .map_err(|error| format!("missing OpenCL trial kernel distance_trials: {error}"))?;
        let score = Kernel::create(&program, "score_trials")
            .map_err(|error| format!("missing OpenCL trial kernel score_trials: {error}"))?;
        let pick = Kernel::create(&program, "pick_trial")
            .map_err(|error| format!("missing OpenCL trial kernel pick_trial: {error}"))?;
        let write = Kernel::create(&program, "write_trial")
            .map_err(|error| format!("missing OpenCL trial kernel write_trial: {error}"))?;
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
            history_slots: buffer(
                &context,
                super::MAX_HISTORY,
                CL_MEM_READ_ONLY,
                "history slots",
            )?,
            outcomes: buffer(&context, super::MAX_HISTORY, CL_MEM_READ_ONLY, "outcomes")?,
            seeds: buffer(&context, 1, CL_MEM_READ_ONLY, "seeds")?,
            draws: buffer(&context, 1, CL_MEM_READ_ONLY, "draws")?,
            scores: buffer(&context, 1, CL_MEM_READ_WRITE, "scores")?,
            partials: buffer(
                &context,
                super::MAX_HISTORY.saturating_mul(tiles.len()),
                CL_MEM_READ_WRITE,
                "partial distances",
            )?,
            choice: buffer(&context, 1, CL_MEM_READ_WRITE, "choice")?,
            leaves: buffer(&context, leaves.len(), CL_MEM_READ_ONLY, "leaves")?,
            tiles: buffer(&context, tiles.len(), CL_MEM_READ_ONLY, "tiles")?,
            centers: buffer(&context, 1, CL_MEM_READ_ONLY, "centers")?,
            candidate_centers: buffer(&context, 1, CL_MEM_READ_ONLY, "candidate centers")?,
            candidate_capacity: 1,
            center_capacity: 1,
        };
        unsafe {
            queue
                .enqueue_write_buffer(&mut scratch.tiles, CL_BLOCKING, 0, &tiles, &[])
                .map_err(|error| format!("failed to write OpenCL tiles: {error}"))?;
        }
        Ok(Self {
            context,
            queue,
            rows,
            row_bytes,
            tile_count: tiles.len(),
            distance,
            score,
            pick,
            write,
            scratch,
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
        let draws = crate::weights::thompson_draws(seeds.len(), config.seed);
        let steps = make_steps(leaves, config.length);
        self.write_inputs(&history_slots, &outcomes, &seeds, &draws, &steps)?;

        let params = Params {
            row_bytes: to_u32(self.row_bytes, "row bytes")?,
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
                .set_arg(&params)
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
                .set_arg(&params)
                .set_global_work_size(seeds.len() * THREADS)
                .set_local_work_size(THREADS)
                .enqueue_nd_range(&self.queue)
                .map_err(|error| format!("failed to launch OpenCL trial scoring: {error}"))?;
            ExecuteKernel::new(&self.pick)
                .set_arg(&self.scratch.scores)
                .set_arg(&self.scratch.choice)
                .set_arg(&params)
                .set_global_work_size(1)
                .set_local_work_size(1)
                .enqueue_nd_range(&self.queue)
                .map_err(|error| format!("failed to launch OpenCL trial selection: {error}"))?;
            if materialize_row {
                ExecuteKernel::new(&self.write)
                    .set_arg(&self.rows)
                    .set_arg(&self.scratch.seeds)
                    .set_arg(&self.scratch.choice)
                    .set_arg(&self.scratch.leaves)
                    .set_arg(&self.scratch.tiles)
                    .set_arg(&params)
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
        let mut scores = vec![0.0f32; seeds.len()];
        unsafe {
            self.queue
                .enqueue_read_buffer(&self.scratch.scores, CL_BLOCKING, 0, &mut scores, &[])
                .map_err(|error| format!("failed to read OpenCL trial scores: {error}"))?;
        }
        let index = choice[0] as usize;
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
        let seeds = [Seed {
            low: seed as u32,
            high: (seed >> 32) as u32,
        }];
        let choice = [0_u32];
        unsafe {
            self.queue
                .enqueue_write_buffer(&mut self.scratch.seeds, CL_NON_BLOCKING, 0, &seeds, &[])
                .map_err(|error| format!("failed to write OpenCL trial seed: {error}"))?;
            self.queue
                .enqueue_write_buffer(&mut self.scratch.choice, CL_NON_BLOCKING, 0, &choice, &[])
                .map_err(|error| format!("failed to write OpenCL trial choice: {error}"))?;
            self.queue
                .enqueue_write_buffer(&mut self.scratch.leaves, CL_NON_BLOCKING, 0, steps, &[])
                .map_err(|error| format!("failed to write OpenCL trial leaves: {error}"))?;
        }

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
        };
        unsafe {
            ExecuteKernel::new(&self.write)
                .set_arg(&self.rows)
                .set_arg(&self.scratch.seeds)
                .set_arg(&self.scratch.choice)
                .set_arg(&self.scratch.leaves)
                .set_arg(&self.scratch.tiles)
                .set_arg(&params)
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
        let packed_seeds: Vec<Seed> = seeds
            .iter()
            .map(|&seed| Seed {
                low: seed as u32,
                high: (seed >> 32) as u32,
            })
            .collect();
        let draws = crate::weights::thompson_draws(seeds.len(), config.seed);
        let steps = make_steps(leaves, config.length);
        self.write_inputs(&history_slots, &outcomes, &packed_seeds, &draws, &steps)?;
        self.write_centers(centers, region_centers, seeds_per_region)?;
        let params = Params {
            row_bytes: to_u32(self.row_bytes, "row bytes")?,
            history: to_u32(history.len(), "history length")?,
            candidates: to_u32(seeds.len(), "candidate count")?,
            leaves: to_u32(leaves.len(), "leaf count")?,
            tiles: to_u32(self.tile_count, "tile count")?,
            neighbors: to_u32(config.neighbors, "neighbor count")?,
            base_slot: to_u32(base_slot, "base slot")?,
            trial_slot: to_u32(base_slot, "trial slot")?,
            center_count: to_u32(centers.len(), "center count")?,
            acquisition: crate::weights::acquisition_code(config.acquisition),
            epistemic_scale: config.epistemic_scale,
            aleatoric_scale: config.aleatoric_scale,
            y_scale: config.y_scale,
            beta: config.beta,
        };
        let scores = self.run_scores(&params, distance_groups, seeds.len())?;
        Ok(scores
            .chunks_exact(seeds_per_region)
            .enumerate()
            .map(|(region, scores)| {
                let (index, &score) = scores
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
                    .expect("regions have non-zero candidates");
                (region * seeds_per_region + index, score)
            })
            .collect())
    }

    pub(super) fn read(&self, slot: usize, row_bytes: usize) -> Result<Vec<u8>, String> {
        let mut row = vec![0u8; row_bytes];
        unsafe {
            self.queue
                .enqueue_read_buffer(&self.rows, CL_NON_BLOCKING, slot * row_bytes, &mut row, &[])
                .map_err(|error| format!("failed to read OpenCL trial row: {error}"))?
                .wait()
                .map_err(|error| format!("failed waiting for OpenCL trial row: {error}"))?;
        }
        Ok(row)
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
        self.scratch.draws = buffer(&self.context, capacity, CL_MEM_READ_ONLY, "draws")?;
        self.scratch.scores = buffer(&self.context, capacity, CL_MEM_READ_WRITE, "scores")?;
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

    fn run_scores(
        &self,
        params: &Params,
        distance_groups: usize,
        candidates: usize,
    ) -> Result<Vec<f32>, String> {
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
                .set_global_work_size(candidates * THREADS)
                .set_local_work_size(THREADS)
                .enqueue_nd_range(&self.queue)
                .map_err(|error| format!("failed to launch OpenCL trial scoring: {error}"))?;
        }
        let mut scores = vec![0.0; candidates];
        unsafe {
            self.queue
                .enqueue_read_buffer(&self.scratch.scores, CL_BLOCKING, 0, &mut scores, &[])
                .map_err(|error| format!("failed to read OpenCL trial scores: {error}"))?;
        }
        Ok(scores)
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
                .enqueue_write_buffer(&mut self.scratch.centers, CL_NON_BLOCKING, 0, &steps, &[])
                .map_err(|error| format!("failed to write OpenCL centers: {error}"))?;
            self.queue
                .enqueue_write_buffer(
                    &mut self.scratch.candidate_centers,
                    CL_NON_BLOCKING,
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
        draws: &[f32],
        leaves: &[LeafStep],
    ) -> Result<(), String> {
        unsafe {
            self.queue
                .enqueue_write_buffer(
                    &mut self.scratch.history_slots,
                    CL_NON_BLOCKING,
                    0,
                    history_slots,
                    &[],
                )
                .map_err(|error| format!("failed to write OpenCL history slots: {error}"))?;
            self.queue
                .enqueue_write_buffer(
                    &mut self.scratch.outcomes,
                    CL_NON_BLOCKING,
                    0,
                    outcomes,
                    &[],
                )
                .map_err(|error| format!("failed to write OpenCL outcomes: {error}"))?;
            self.queue
                .enqueue_write_buffer(&mut self.scratch.seeds, CL_NON_BLOCKING, 0, seeds, &[])
                .map_err(|error| format!("failed to write OpenCL trial seeds: {error}"))?;
            self.queue
                .enqueue_write_buffer(&mut self.scratch.draws, CL_NON_BLOCKING, 0, draws, &[])
                .map_err(|error| format!("failed to write OpenCL Thompson draws: {error}"))?;
            self.queue
                .enqueue_write_buffer(&mut self.scratch.leaves, CL_NON_BLOCKING, 0, leaves, &[])
                .map_err(|error| format!("failed to write OpenCL leaves: {error}"))?;
        }
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
