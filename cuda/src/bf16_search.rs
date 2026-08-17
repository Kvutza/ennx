use std::collections::BTreeSet;
use std::mem::size_of;

use cuda_core::{CudaStream, DeviceBuffer, LaunchConfig1D};

use super::*;

const BF16_MAX_PENDING: usize = 32;

struct Bf16Scratch {
    history_capacity: usize,
    candidate_capacity: usize,
    region_capacity: usize,
    partial_capacity: usize,
    status_capacity: usize,
    history_slots: DeviceBuffer<u32>,
    outcomes: DeviceBuffer<f32>,
    variances: DeviceBuffer<f32>,
    seeds: DeviceBuffer<Seed>,
    draws: DeviceBuffer<f32>,
    scores: DeviceBuffer<f32>,
    partials: DeviceBuffer<f32>,
    selection: DeviceBuffer<Selection>,
    trial_slots: DeviceBuffer<u32>,
    status: DeviceBuffer<u32>,
}

struct AskInput<'a> {
    base_slot: usize,
    history: usize,
    trial_slots: &'a [u32],
    seeds: &'a [u64],
    candidates_per_region: usize,
    coefficient: f32,
    draw_seed: u64,
    config: Ask,
}

#[derive(Debug, Clone)]
pub struct TellOutput {
    pub accepted: Vec<bool>,
    pub length: f64,
    pub best: f32,
    pub best_variance: f32,
    pub history: usize,
    pub restarts: usize,
    pub restarted: bool,
}

#[derive(Clone, Copy)]
struct SearchShape {
    regions: usize,
    candidates: usize,
    blocks: usize,
    partials: usize,
    status: usize,
}

impl Bf16Scratch {
    fn new(stream: &CudaStream) -> CudaResult<Self> {
        Ok(Self {
            history_capacity: 1,
            candidate_capacity: 1,
            region_capacity: 1,
            partial_capacity: 1,
            status_capacity: 1,
            history_slots: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            outcomes: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            variances: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            seeds: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            draws: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            scores: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            partials: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            selection: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            trial_slots: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            status: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
        })
    }

    fn ensure(
        &mut self,
        stream: &CudaStream,
        history: usize,
        candidates: usize,
        regions: usize,
        partials: usize,
        status: usize,
    ) -> CudaResult<()> {
        let history_capacity = next_capacity(history, "BF16 history")?;
        let candidate_capacity = next_capacity(candidates, "BF16 candidates")?;
        let region_capacity = next_capacity(regions, "BF16 regions")?;
        let partial_capacity = next_capacity(partials, "BF16 distance partials")?;
        let status_capacity = next_capacity(status, "BF16 status")?;
        if history_capacity > self.history_capacity {
            self.history_slots =
                DeviceBuffer::zeroed(stream, history_capacity).map_err(cuda_error)?;
            self.outcomes = DeviceBuffer::zeroed(stream, history_capacity).map_err(cuda_error)?;
            self.variances = DeviceBuffer::zeroed(stream, history_capacity).map_err(cuda_error)?;
            self.history_capacity = history_capacity;
        }
        if candidate_capacity > self.candidate_capacity {
            self.seeds = DeviceBuffer::zeroed(stream, candidate_capacity).map_err(cuda_error)?;
            self.draws = DeviceBuffer::zeroed(stream, candidate_capacity).map_err(cuda_error)?;
            self.scores = DeviceBuffer::zeroed(stream, candidate_capacity).map_err(cuda_error)?;
            self.candidate_capacity = candidate_capacity;
        }
        if partial_capacity > self.partial_capacity {
            self.partials = DeviceBuffer::zeroed(stream, partial_capacity).map_err(cuda_error)?;
            self.partial_capacity = partial_capacity;
        }
        if region_capacity > self.region_capacity {
            self.selection = DeviceBuffer::zeroed(stream, region_capacity).map_err(cuda_error)?;
            self.trial_slots = DeviceBuffer::zeroed(stream, region_capacity).map_err(cuda_error)?;
            self.region_capacity = region_capacity;
        }
        if status_capacity > self.status_capacity {
            self.status = DeviceBuffer::zeroed(stream, status_capacity).map_err(cuda_error)?;
            self.status_capacity = status_capacity;
        }
        Ok(())
    }
}

/// CUDA-resident BF16 candidate scoring and materialization.
pub struct Bf16SearchEngine {
    runtime: Runtime,
    rows: DeviceBuffer<u16>,
    leaves: DeviceBuffer<Bf16Leaf>,
    tiles: DeviceBuffer<DenseTile>,
    row_len: usize,
    row_stride: usize,
    slots: usize,
    tile_count: usize,
    scratch: Bf16Scratch,
    state: DeviceBuffer<SearchState>,
    summary: DeviceBuffer<TellSummary>,
    accepted: DeviceBuffer<u32>,
    tell_values: DeviceBuffer<f32>,
    tell_variances: DeviceBuffer<f32>,
    profiling: bool,
    last_profile: Option<AskProfile>,
}

impl Bf16SearchEngine {
    pub fn new(base: &[u16], leaves: &[Bf16Leaf], slots: usize) -> CudaResult<Self> {
        validate_bf16(base.len(), leaves)?;
        if base.iter().any(|value| !bf16_finite(*value)) {
            return Err("CUDA BF16 search base values must be finite".to_string());
        }
        let mut engine = Self::allocate(base.len(), leaves, slots)?;
        copy_prefix(&engine.rows, base, &engine.runtime.stream)?;
        engine.validate(0)?;
        Ok(engine)
    }

    /// Copy a contiguous BF16 CUDA allocation into persistent search storage.
    ///
    /// # Safety
    /// `pointer` must address at least `len * 2` readable bytes on CUDA device 0.
    pub unsafe fn from_device(
        pointer: u64,
        len: usize,
        leaves: &[Bf16Leaf],
        slots: usize,
    ) -> CudaResult<Self> {
        if pointer == 0 {
            return Err("CUDA BF16 search requires a device base".to_string());
        }
        validate_bf16(len, leaves)?;
        let mut engine = Self::allocate(len, leaves, slots)?;
        unsafe {
            cuda_core::memory::memcpy_dtod_async(
                engine.rows.cu_deviceptr(),
                pointer,
                row_bytes(len)?,
                engine.runtime.stream.cu_stream(),
            )
            .map_err(cuda_error)?;
        }
        engine.validate(0)?;
        Ok(engine)
    }

    fn allocate(len: usize, leaves: &[Bf16Leaf], slots: usize) -> CudaResult<Self> {
        if slots < 2 {
            return Err("CUDA BF16 search requires at least two row slots".to_string());
        }
        let tiles = bf16_tiles(leaves)?;
        let runtime = Runtime::new()?;
        let row_stride = len
            .checked_add(127)
            .ok_or("CUDA BF16 row stride overflow")?
            & !127;
        let row_count = slots
            .checked_mul(row_stride)
            .ok_or("CUDA BF16 resident row count overflow")?;
        let rows = DeviceBuffer::zeroed(&runtime.stream, row_count).map_err(cuda_error)?;
        let leaves = DeviceBuffer::from_host(&runtime.stream, leaves).map_err(cuda_error)?;
        let tile_count = tiles.len();
        let tiles = DeviceBuffer::from_host(&runtime.stream, &tiles).map_err(cuda_error)?;
        let scratch = Bf16Scratch::new(&runtime.stream)?;
        let state = DeviceBuffer::zeroed(&runtime.stream, 1).map_err(cuda_error)?;
        let summary = DeviceBuffer::zeroed(&runtime.stream, 1).map_err(cuda_error)?;
        let accepted =
            DeviceBuffer::zeroed(&runtime.stream, BF16_MAX_PENDING).map_err(cuda_error)?;
        let tell_values =
            DeviceBuffer::zeroed(&runtime.stream, BF16_MAX_PENDING).map_err(cuda_error)?;
        let tell_variances =
            DeviceBuffer::zeroed(&runtime.stream, BF16_MAX_PENDING).map_err(cuda_error)?;
        Ok(Self {
            runtime,
            rows,
            leaves,
            tiles,
            row_len: len,
            row_stride,
            slots,
            tile_count,
            scratch,
            state,
            summary,
            accepted,
            tell_values,
            tell_variances,
            profiling: false,
            last_profile: None,
        })
    }

    pub fn set_profiling(&mut self, enabled: bool) {
        set_profile(enabled, &mut self.profiling, &mut self.last_profile);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn init_search(
        &mut self,
        base_value: f32,
        base_variance: f32,
        capacity: usize,
        length_init: f64,
        length_min: f64,
        length_max: f64,
    ) -> CudaResult<()> {
        if capacity == 0
            || capacity > MAX_HISTORY
            || !base_value.is_finite()
            || !base_variance.is_finite()
            || base_variance < 0.0
            || !length_init.is_finite()
            || !length_min.is_finite()
            || !length_max.is_finite()
            || length_min <= 0.0
            || length_init < length_min
            || length_init > length_max
        {
            return Err("CUDA BF16 search state is invalid".to_string());
        }
        self.scratch
            .ensure(&self.runtime.stream, MAX_HISTORY, 1, 1, 1, self.tile_count)?;
        copy_prefix(&self.scratch.history_slots, &[1], &self.runtime.stream)?;
        copy_prefix(&self.scratch.outcomes, &[base_value], &self.runtime.stream)?;
        copy_prefix(
            &self.scratch.variances,
            &[base_variance],
            &self.runtime.stream,
        )?;
        let state = SearchState {
            length: length_init,
            length_init,
            length_min,
            length_max,
            best: base_value,
            best_variance: base_variance,
            trust_best: f64::from(base_value),
            hist_min: f64::from(base_value),
            hist_max: f64::from(base_value),
            prev_obs: 1,
            successes: 0,
            failures: 0,
            restarts: 0,
            history: 1,
            pad: 0,
        };
        copy_prefix(&self.state, &[state], &self.runtime.stream)
    }

    pub fn tell(
        &mut self,
        trial_slots: &[u32],
        values: &[f32],
        variances: &[f32],
        capacity: usize,
        failure_tolerance: usize,
    ) -> CudaResult<TellOutput> {
        self.check_tell(trial_slots, values.len(), variances.len(), capacity)?;
        copy_prefix(&self.tell_values, values, &self.runtime.stream)?;
        copy_prefix(&self.tell_variances, variances, &self.runtime.stream)?;
        self.launch_tell(trial_slots, values.len(), capacity, failure_tolerance)
    }

    /// Consume contiguous device-0 FP32 rewards without staging them through Python.
    ///
    /// # Safety
    /// The pointers must each address `count * 4` readable bytes on CUDA device 0.
    pub unsafe fn tell_device(
        &mut self,
        trial_slots: &[u32],
        values: u64,
        variances: Option<u64>,
        count: usize,
        capacity: usize,
        failure_tolerance: usize,
    ) -> CudaResult<TellOutput> {
        self.check_tell(trial_slots, count, count, capacity)?;
        if values == 0 || variances == Some(0) {
            return Err("CUDA BF16 tell requires valid device rewards".to_string());
        }
        let bytes = count
            .checked_mul(size_of::<f32>())
            .ok_or("CUDA BF16 tell byte count overflow")?;
        unsafe {
            cuda_core::memory::memcpy_dtod_async(
                self.tell_values.cu_deviceptr(),
                values,
                bytes,
                self.runtime.stream.cu_stream(),
            )
            .map_err(cuda_error)?;
            if let Some(pointer) = variances {
                cuda_core::memory::memcpy_dtod_async(
                    self.tell_variances.cu_deviceptr(),
                    pointer,
                    bytes,
                    self.runtime.stream.cu_stream(),
                )
                .map_err(cuda_error)?;
            } else {
                cuda_core::memory::memset_d8_async(
                    self.tell_variances.cu_deviceptr(),
                    0,
                    bytes,
                    self.runtime.stream.cu_stream(),
                )
                .map_err(cuda_error)?;
            }
        }
        self.launch_tell(trial_slots, count, capacity, failure_tolerance)
    }

    fn check_tell(
        &self,
        trial_slots: &[u32],
        values: usize,
        variances: usize,
        capacity: usize,
    ) -> CudaResult<()> {
        if trial_slots.is_empty()
            || trial_slots.len() > BF16_MAX_PENDING
            || values != trial_slots.len()
            || variances != trial_slots.len()
            || capacity == 0
            || capacity > MAX_HISTORY
        {
            return Err("CUDA BF16 tell shape is invalid".to_string());
        }
        for &slot in trial_slots {
            self.check_slot(slot as usize)?;
        }
        Ok(())
    }

    fn launch_tell(
        &mut self,
        trial_slots: &[u32],
        count: usize,
        capacity: usize,
        failure_tolerance: usize,
    ) -> CudaResult<TellOutput> {
        copy_prefix(&self.scratch.trial_slots, trial_slots, &self.runtime.stream)?;
        let launch = self
            .runtime
            .module
            .prepare_tell_bf16(LaunchConfig1D::new(1, THREADS, 0))
            .map_err(cuda_error)?;
        let params = TellParams {
            row_stride: self.row_stride as u64,
            row_len: self.row_len as u64,
            trials: to_u32(count, "BF16 tell count")?,
            capacity: to_u32(capacity, "BF16 history capacity")?,
            failure_tolerance: to_u32(failure_tolerance.max(1), "BF16 failure tolerance")?,
            pad: 0,
        };
        self.runtime
            .module
            .tell_bf16(
                &self.runtime.stream,
                &launch,
                &mut self.rows,
                &mut self.scratch.history_slots,
                &mut self.scratch.outcomes,
                &mut self.scratch.variances,
                &self.scratch.trial_slots,
                &self.tell_values,
                &self.tell_variances,
                &mut self.accepted,
                &mut self.state,
                &mut self.summary,
                params,
            )
            .map_err(cuda_error)?;
        self.runtime.context.check_err().map_err(cuda_error)?;
        let summary = read_prefix(&self.summary, &self.runtime.stream, 1)?[0];
        if summary.status != 0 {
            return Err("CUDA BF16 values and variances must be finite".to_string());
        }
        let accepted = read_prefix(&self.accepted, &self.runtime.stream, count)?
            .into_iter()
            .map(|value| value != 0)
            .collect();
        Ok(TellOutput {
            accepted,
            length: summary.length,
            best: summary.best,
            best_variance: summary.best_variance,
            history: summary.history as usize,
            restarts: summary.restarts as usize,
            restarted: summary.restarted != 0,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn ask(
        &mut self,
        base_slot: usize,
        history: usize,
        trial_slots: &[u32],
        seeds: &[u64],
        candidates_per_region: usize,
        coefficient: f32,
        draw_seed: u64,
        config: Ask,
    ) -> CudaResult<Vec<Selection>> {
        self.last_profile = None;
        let client = TRACY.get_or_init(tracy_client::Client::start);
        let _zone = client
            .clone()
            .span(tracy_client::span_location!("ennx.cuda.bf16.ask"), 0);
        let input = AskInput {
            base_slot,
            history,
            trial_slots,
            seeds,
            candidates_per_region,
            coefficient,
            draw_seed,
            config,
        };
        let shape = self.check_ask(&input)?;
        self.upload(&input, shape)?;
        let events = self.launch(&input, shape)?;
        let selections = self.collect(&input, shape)?;
        if let Some(events) = events {
            let profile = events.profile()?;
            publish_profile(client, profile);
            self.last_profile = Some(profile);
        }
        Ok(selections)
    }

    fn check_ask(&self, input: &AskInput<'_>) -> CudaResult<SearchShape> {
        self.check_slot(input.base_slot)?;
        let history = input.history;
        let regions = input.trial_slots.len();
        if history == 0 || history > MAX_HISTORY {
            return Err(format!(
                "CUDA BF16 history must contain 1..={MAX_HISTORY} rows"
            ));
        }
        if regions == 0 || input.candidates_per_region == 0 {
            return Err("CUDA BF16 search requires regions and candidates".to_string());
        }
        let candidates = regions
            .checked_mul(input.candidates_per_region)
            .ok_or("CUDA BF16 candidate count overflow")?;
        if input.seeds.len() != candidates {
            return Err("CUDA BF16 seeds do not match the search shape".to_string());
        }
        if input.config.neighbors == 0 || input.config.neighbors > history {
            return Err("CUDA BF16 neighbor count exceeds resident history".to_string());
        }
        if !input.coefficient.is_finite() || input.coefficient <= 0.0 {
            return Err("CUDA BF16 perturbation coefficient must be positive".to_string());
        }
        validate_ask(input)?;
        let mut destinations = BTreeSet::new();
        for &slot in input.trial_slots {
            self.check_slot(slot as usize)?;
            if slot as usize == input.base_slot || !destinations.insert(slot) {
                return Err("CUDA BF16 trial slots must be distinct from live rows".to_string());
            }
        }
        let status = regions
            .checked_mul(self.tile_count)
            .ok_or("CUDA BF16 status count overflow")?;
        let blocks = candidates
            .checked_mul(self.tile_count)
            .ok_or("CUDA BF16 distance block count overflow")?;
        let partials = blocks
            .checked_mul(history)
            .ok_or("CUDA BF16 distance partial count overflow")?;
        Ok(SearchShape {
            regions,
            candidates,
            blocks,
            partials,
            status,
        })
    }

    fn upload(&mut self, input: &AskInput<'_>, shape: SearchShape) -> CudaResult<()> {
        self.scratch.ensure(
            &self.runtime.stream,
            input.history,
            shape.candidates,
            shape.regions,
            shape.partials,
            shape.status,
        )?;
        let seeds = input
            .seeds
            .iter()
            .map(|seed| Seed {
                low: *seed as u32,
                high: (*seed >> 32) as u32,
            })
            .collect::<Vec<_>>();
        copy_prefix(&self.scratch.seeds, &seeds, &self.runtime.stream)?;
        copy_prefix(
            &self.scratch.trial_slots,
            input.trial_slots,
            &self.runtime.stream,
        )?;
        self.clear_status(shape.status)
    }

    fn launch(
        &mut self,
        input: &AskInput<'_>,
        shape: SearchShape,
    ) -> CudaResult<Option<AskEvents>> {
        let candidates = to_u32(shape.candidates, "BF16 candidate count")?;
        let regions = to_u32(shape.regions, "BF16 region count")?;
        let distance_launch = self
            .runtime
            .module
            .prepare_distance_bf16(LaunchConfig1D::new(
                to_u32(shape.blocks, "BF16 distance blocks")?,
                THREADS,
                0,
            ))
            .map_err(cuda_error)?;
        let score_launch = self
            .runtime
            .module
            .prepare_score_bf16(LaunchConfig1D::new(candidates, THREADS, 0))
            .map_err(cuda_error)?;
        let draw_launch = self
            .runtime
            .module
            .prepare_draw_bf16(LaunchConfig1D::new(
                to_u32(
                    shape.candidates.div_ceil(THREADS as usize),
                    "BF16 draw blocks",
                )?,
                THREADS,
                0,
            ))
            .map_err(cuda_error)?;
        let pick_launch = self
            .runtime
            .module
            .prepare_pick_trial(LaunchConfig1D::new(regions, THREADS, 0))
            .map_err(cuda_error)?;
        let write_launch = self
            .runtime
            .module
            .prepare_write_bf16(LaunchConfig1D::new(
                to_u32(shape.status, "BF16 write blocks")?,
                THREADS,
                0,
            ))
            .map_err(cuda_error)?;
        let profile = self.profiling
            || tracy_client::Client::is_connected()
            || std::env::var_os("ENNX_CUDA_PROFILE").is_some();
        let score_start = profile
            .then(|| timing_event(&self.runtime.stream))
            .transpose()?;
        let params = Bf16Score {
            row_stride: self.row_stride as u64,
            coefficient: input.coefficient,
            epistemic_scale: input.config.epistemic_scale,
            aleatoric_scale: input.config.aleatoric_scale,
            y_scale: input.config.y_scale,
            beta: input.config.beta,
            history: to_u32(input.history, "BF16 history rows")?,
            candidates,
            base_slot: to_u32(input.base_slot, "BF16 base slot")?,
            neighbors: to_u32(input.config.neighbors, "BF16 neighbors")?,
            acquisition: input.config.acquisition,
            tiles: to_u32(self.tile_count, "BF16 distance tile count")?,
            pad1: 0,
        };
        self.runtime
            .module
            .distance_bf16(
                &self.runtime.stream,
                &distance_launch,
                &self.rows,
                &self.scratch.history_slots,
                &self.scratch.seeds,
                &self.leaves,
                &self.tiles,
                &mut self.scratch.partials,
                params,
            )
            .map_err(cuda_error)?;
        self.runtime
            .module
            .draw_bf16(
                &self.runtime.stream,
                &draw_launch,
                &mut self.scratch.draws,
                input.draw_seed,
                candidates,
            )
            .map_err(cuda_error)?;
        self.runtime
            .module
            .score_bf16(
                &self.runtime.stream,
                &score_launch,
                &self.scratch.partials,
                &self.scratch.outcomes,
                &self.scratch.variances,
                &self.scratch.draws,
                &mut self.scratch.scores,
                params,
            )
            .map_err(cuda_error)?;
        let score_end = profile
            .then(|| timing_event(&self.runtime.stream))
            .transpose()?;
        self.runtime
            .module
            .pick_trial(
                &self.runtime.stream,
                &pick_launch,
                &self.scratch.scores,
                &mut self.scratch.selection,
                regions,
                to_u32(input.candidates_per_region, "BF16 candidates per region")?,
            )
            .map_err(cuda_error)?;
        let pick_end = profile
            .then(|| timing_event(&self.runtime.stream))
            .transpose()?;
        self.runtime
            .module
            .write_bf16(
                &self.runtime.stream,
                &write_launch,
                &mut self.rows,
                &self.scratch.seeds,
                &self.scratch.selection,
                &self.leaves,
                &self.tiles,
                &self.scratch.trial_slots,
                &mut self.scratch.status,
                self.row_stride as u64,
                to_u32(input.base_slot, "BF16 base slot")?,
                to_u32(self.tile_count, "BF16 tile count")?,
                input.coefficient,
            )
            .map_err(cuda_error)?;
        let materialize_end = profile
            .then(|| timing_event(&self.runtime.stream))
            .transpose()?;
        self.runtime.context.check_err().map_err(cuda_error)?;
        Ok(match (score_start, score_end, pick_end, materialize_end) {
            (Some(score_start), Some(score_end), Some(pick_end), Some(materialize_end)) => {
                Some(AskEvents {
                    score_start,
                    score_end,
                    pick_end,
                    materialize_end: Some(materialize_end),
                })
            }
            _ => None,
        })
    }

    fn collect(&self, input: &AskInput<'_>, shape: SearchShape) -> CudaResult<Vec<Selection>> {
        let selections = read_prefix(&self.scratch.selection, &self.runtime.stream, shape.regions)?;
        for (region, selection) in selections.iter().enumerate() {
            let first = region * input.candidates_per_region;
            let end = first + input.candidates_per_region;
            if !(first..end).contains(&(selection.index as usize)) {
                self.reset_trials(input.base_slot, input.trial_slots)?;
                return Err(format!(
                    "CUDA BF16 region {region} selected invalid trial index {}",
                    selection.index
                ));
            }
        }
        let status = read_prefix(&self.scratch.status, &self.runtime.stream, shape.status)?;
        if status.contains(&1) {
            self.reset_trials(input.base_slot, input.trial_slots)?;
            return Err("CUDA BF16 search perturbation overflowed FP32".to_string());
        }
        Ok(selections)
    }

    pub fn copy_row(&self, source: usize, destination: usize) -> CudaResult<()> {
        self.check_slot(source)?;
        self.check_slot(destination)?;
        if source == destination {
            return Ok(());
        }
        unsafe {
            cuda_core::memory::memcpy_dtod_async(
                self.row_pointer(destination)?,
                self.row_pointer(source)?,
                row_bytes(self.row_len)?,
                self.runtime.stream.cu_stream(),
            )
            .map_err(cuda_error)
        }
    }

    pub fn read(&self, slot: usize) -> CudaResult<Vec<u16>> {
        self.check_slot(slot)?;
        let mut output = Vec::<u16>::with_capacity(self.row_len);
        unsafe {
            cuda_core::memory::memcpy_dtoh_async(
                output.as_mut_ptr(),
                self.row_pointer(slot)?,
                row_bytes(self.row_len)?,
                self.runtime.stream.cu_stream(),
            )
            .map_err(cuda_error)?;
        }
        self.runtime.stream.synchronize().map_err(cuda_error)?;
        unsafe {
            output.set_len(self.row_len);
        }
        Ok(output)
    }

    pub fn device_row(&self, slot: usize, stream: Option<i64>) -> CudaResult<(u64, usize, usize)> {
        self.check_slot(slot)?;
        sync_stream(&self.runtime, stream)?;
        Ok((self.row_pointer(slot)?, row_bytes(self.row_len)?, 0))
    }

    pub fn last_profile(&self) -> Option<AskProfile> {
        self.last_profile
    }

    pub fn len(&self) -> usize {
        self.row_len
    }

    pub fn is_empty(&self) -> bool {
        self.row_len == 0
    }

    fn validate(&mut self, slot: usize) -> CudaResult<()> {
        self.check_slot(slot)?;
        self.scratch
            .ensure(&self.runtime.stream, 1, 1, 1, 1, self.tile_count)?;
        self.clear_status(self.tile_count)?;
        let launch = self
            .runtime
            .module
            .prepare_check_search(LaunchConfig1D::new(
                to_u32(self.tile_count, "BF16 tile count")?,
                THREADS,
                0,
            ))
            .map_err(cuda_error)?;
        self.runtime
            .module
            .check_search(
                &self.runtime.stream,
                &launch,
                &self.rows,
                &self.leaves,
                &self.tiles,
                &mut self.scratch.status,
                self.row_stride as u64,
                to_u32(slot, "BF16 row slot")?,
            )
            .map_err(cuda_error)?;
        self.runtime.context.check_err().map_err(cuda_error)?;
        let status = read_prefix(&self.scratch.status, &self.runtime.stream, self.tile_count)?;
        if status.contains(&1) {
            Err("CUDA BF16 search base values must be finite".to_string())
        } else {
            Ok(())
        }
    }

    fn clear_status(&self, count: usize) -> CudaResult<()> {
        unsafe {
            cuda_core::memory::memset_d8_async(
                self.scratch.status.cu_deviceptr(),
                0,
                count
                    .checked_mul(size_of::<u32>())
                    .ok_or("CUDA BF16 status byte count overflow")?,
                self.runtime.stream.cu_stream(),
            )
            .map_err(cuda_error)
        }
    }

    fn reset_trials(&self, base_slot: usize, slots: &[u32]) -> CudaResult<()> {
        for &slot in slots {
            self.copy_row(base_slot, slot as usize)?;
        }
        self.runtime.stream.synchronize().map_err(cuda_error)
    }

    fn row_pointer(&self, slot: usize) -> CudaResult<u64> {
        let offset = slot
            .checked_mul(self.row_stride)
            .and_then(|value| value.checked_mul(size_of::<u16>()))
            .ok_or("CUDA BF16 row pointer overflow")?;
        Ok(self.rows.cu_deviceptr() + offset as u64)
    }

    fn check_slot(&self, slot: usize) -> CudaResult<()> {
        if slot >= self.slots {
            Err(format!(
                "CUDA BF16 row slot {slot} exceeds capacity {}",
                self.slots
            ))
        } else {
            Ok(())
        }
    }
}

fn validate_ask(input: &AskInput<'_>) -> CudaResult<()> {
    let config = input.config;
    if config.acquisition > 2
        || !config.epistemic_scale.is_finite()
        || config.epistemic_scale < 0.0
        || !config.aleatoric_scale.is_finite()
        || config.aleatoric_scale < 0.0
        || !config.y_scale.is_finite()
        || config.y_scale < 0.0
        || !config.beta.is_finite()
    {
        return Err("CUDA BF16 acquisition configuration is invalid".to_string());
    }
    Ok(())
}

fn validate_bf16(len: usize, leaves: &[Bf16Leaf]) -> CudaResult<()> {
    if len == 0 || leaves.is_empty() {
        return Err("CUDA BF16 search requires base values and leaves".to_string());
    }
    let mut expected = 0u64;
    for leaf in leaves {
        if leaf.offset != expected || leaf.length == 0 {
            return Err("CUDA BF16 leaves must form a contiguous non-empty layout".to_string());
        }
        if !leaf.scale.is_finite()
            || leaf.scale <= 0.0
            || !leaf.weight.is_finite()
            || leaf.weight <= 0.0
        {
            return Err("CUDA BF16 leaf scales and weights must be positive".to_string());
        }
        u32::try_from(leaf.length).map_err(|_| "CUDA BF16 leaf length exceeds u32".to_string())?;
        expected = expected
            .checked_add(leaf.length)
            .ok_or("CUDA BF16 leaf layout overflow")?;
    }
    if expected != len as u64 {
        return Err(format!(
            "CUDA BF16 leaf layout covers {expected} values, expected {len}"
        ));
    }
    Ok(())
}

fn bf16_tiles(leaves: &[Bf16Leaf]) -> CudaResult<Vec<DenseTile>> {
    let mut tiles = Vec::new();
    for (leaf_index, leaf) in leaves.iter().enumerate() {
        let leaf_index = to_u32(leaf_index, "BF16 leaf count")?;
        let length = usize::try_from(leaf.length)
            .map_err(|_| "CUDA BF16 leaf length exceeds usize".to_string())?;
        let mut start = 0usize;
        while start < length {
            let tile_length = (length - start).min(DENSE_TILE_ELEMENTS);
            tiles.push(DenseTile {
                leaf: leaf_index,
                start: to_u32(start, "BF16 leaf offset")?,
                length: to_u32(tile_length, "BF16 tile length")?,
                pad: 0,
            });
            start += tile_length;
        }
    }
    Ok(tiles)
}

fn bf16_finite(value: u16) -> bool {
    f32::from_bits(u32::from(value) << 16).is_finite()
}

fn row_bytes(len: usize) -> CudaResult<usize> {
    len.checked_mul(size_of::<u16>())
        .ok_or("CUDA BF16 row byte count overflow".to_string())
}

fn next_capacity(value: usize, name: &str) -> CudaResult<usize> {
    value
        .max(1)
        .checked_next_power_of_two()
        .ok_or_else(|| format!("CUDA {name} capacity overflow"))
}
