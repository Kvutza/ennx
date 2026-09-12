use std::collections::BTreeSet;
use std::mem::size_of;

use cuda_core::{CudaStream, DeviceBuffer, LaunchConfig1D};

use super::*;

const BF16_PENDING: usize = 32;

/// Selected seed, acquisition score, nominal radius, and per-leaf (changed, squared L2).
pub type ProposalDescription = (u64, f32, f32, Vec<(u64, f64)>);

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
    tile_status: DeviceBuffer<u32>,
    selection: DeviceBuffer<Selection>,
    trial_slots: DeviceBuffer<u32>,
    destinations: DeviceBuffer<u32>,
    status: DeviceBuffer<u32>,
    changes: DeviceBuffer<Bf16Change>,
}

struct AskInput<'a> {
    base_slot: usize,
    history: usize,
    trial_slots: &'a [u32],
    seeds: &'a [u64],
    seed_root: Option<u64>,
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

#[derive(Clone, Copy)]
struct SingleRow {
    slot: usize,
    base_slot: usize,
    coefficient: f32,
    resident: bool,
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
            tile_status: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            selection: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            trial_slots: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            destinations: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            status: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            changes: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
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
            self.draws = DeviceBuffer::zeroed(stream, history_capacity).map_err(cuda_error)?;
            self.history_capacity = history_capacity;
        }
        if candidate_capacity > self.candidate_capacity {
            self.seeds = DeviceBuffer::zeroed(stream, candidate_capacity).map_err(cuda_error)?;
            self.scores = DeviceBuffer::zeroed(stream, candidate_capacity).map_err(cuda_error)?;
            self.candidate_capacity = candidate_capacity;
        }
        if partial_capacity > self.partial_capacity {
            self.partials = DeviceBuffer::zeroed(stream, partial_capacity).map_err(cuda_error)?;
            self.tile_status =
                DeviceBuffer::zeroed(stream, partial_capacity).map_err(cuda_error)?;
            self.partial_capacity = partial_capacity;
        }
        if region_capacity > self.region_capacity {
            self.selection = DeviceBuffer::zeroed(stream, region_capacity).map_err(cuda_error)?;
            self.trial_slots = DeviceBuffer::zeroed(stream, region_capacity).map_err(cuda_error)?;
            self.destinations =
                DeviceBuffer::zeroed(stream, region_capacity).map_err(cuda_error)?;
            self.region_capacity = region_capacity;
        }
        if status_capacity > self.status_capacity {
            self.status = DeviceBuffer::zeroed(stream, status_capacity).map_err(cuda_error)?;
            self.changes = DeviceBuffer::zeroed(stream, status_capacity).map_err(cuda_error)?;
            self.status_capacity = status_capacity;
        }
        Ok(())
    }
}

/// CUDA-resident BF16 candidate scoring and materialization.
pub struct Bf16SearchEngine {
    runtime: Runtime,
    rows: DeviceBuffer<u16>,
    batch: DeviceBuffer<u16>,
    reference: DeviceBuffer<u16>,
    reference_scales: DeviceBuffer<f32>,
    reference_partials: DeviceBuffer<f32>,
    tile_offsets: DeviceBuffer<u32>,
    sampler: u32,
    reference_seed: Option<u64>,
    paired_relative: Option<usize>,
    search_capacity: Option<usize>,
    started: bool,
    best_source: DeviceBuffer<u32>,
    single_slot: Option<SingleRow>,
    leaves: DeviceBuffer<Bf16Leaf>,
    tiles: DeviceBuffer<DenseTile>,
    row_len: usize,
    row_stride: usize,
    slots: usize,
    tile_count: usize,
    tile_leaves: Vec<usize>,
    leaf_count: usize,
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
            cuda_core::simt::memory::memcpy_dtod_async(
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
        let tile_leaves = tiles.iter().map(|tile| tile.leaf as usize).collect();
        let leaf_count = leaves.len();
        let runtime = Runtime::new()?;
        let row_stride = len
            .checked_add(127)
            .ok_or("CUDA BF16 row stride overflow")?
            & !127;
        let row_count = slots
            .checked_mul(row_stride)
            .ok_or("CUDA BF16 resident row count overflow")?;
        let rows = DeviceBuffer::zeroed(&runtime.stream, row_count).map_err(cuda_error)?;
        let batch = DeviceBuffer::zeroed(&runtime.stream, 1).map_err(cuda_error)?;
        let reference = DeviceBuffer::zeroed(&runtime.stream, 1).map_err(cuda_error)?;
        let reference_scales = DeviceBuffer::zeroed(&runtime.stream, 1).map_err(cuda_error)?;
        let reference_partials = DeviceBuffer::zeroed(&runtime.stream, 1).map_err(cuda_error)?;
        let offsets = (0..=leaves.len())
            .map(|leaf| {
                to_u32(
                    tiles.partition_point(|tile| (tile.leaf as usize) < leaf),
                    "BF16 leaf tiles",
                )
            })
            .collect::<CudaResult<Vec<_>>>()?;
        let tile_offsets =
            DeviceBuffer::from_host(&runtime.stream, &offsets).map_err(cuda_error)?;
        let best_source = DeviceBuffer::zeroed(&runtime.stream, 1).map_err(cuda_error)?;
        let leaves = DeviceBuffer::from_host(&runtime.stream, leaves).map_err(cuda_error)?;
        let tile_count = tiles.len();
        let tiles = DeviceBuffer::from_host(&runtime.stream, &tiles).map_err(cuda_error)?;
        let scratch = Bf16Scratch::new(&runtime.stream)?;
        let state = DeviceBuffer::zeroed(&runtime.stream, 1).map_err(cuda_error)?;
        let summary = DeviceBuffer::zeroed(&runtime.stream, 1).map_err(cuda_error)?;
        let accepted = DeviceBuffer::zeroed(&runtime.stream, BF16_PENDING).map_err(cuda_error)?;
        let tell_values =
            DeviceBuffer::zeroed(&runtime.stream, BF16_PENDING).map_err(cuda_error)?;
        let tell_variances =
            DeviceBuffer::zeroed(&runtime.stream, BF16_PENDING).map_err(cuda_error)?;
        Ok(Self {
            runtime,
            rows,
            batch,
            reference,
            reference_scales,
            reference_partials,
            tile_offsets,
            sampler: 0,
            reference_seed: None,
            paired_relative: None,
            search_capacity: None,
            started: false,
            best_source,
            single_slot: None,
            leaves,
            tiles,
            row_len: len,
            row_stride,
            slots,
            tile_count,
            tile_leaves,
            leaf_count,
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

    pub fn enable_correlated(&mut self, reference_seed: u64) -> CudaResult<()> {
        if self.sampler != 0 || self.single_slot.is_some() {
            return Err("Correlated BF16 sampling must be enabled before the first ask".into());
        }
        // Delay the model-sized reference until ask, after the caller can release its input copy.
        self.reference_seed = Some(reference_seed);
        self.sampler = 1;
        Ok(())
    }

    pub fn enable_gaussian(&mut self) -> CudaResult<()> {
        if self.sampler != 0 || self.single_slot.is_some() {
            return Err("Gaussian BF16 sampling must be enabled before the first ask".into());
        }
        self.sampler = 2;
        Ok(())
    }

    /// Initialize the pinned zero anchor before the first correlated round.
    pub fn enable_relative(&mut self, failure_tolerance: usize) -> CudaResult<()> {
        if self.sampler != 1
            || self.started
            || self.paired_relative.is_some()
            || !self.search_capacity.is_some_and(|capacity| capacity >= 2)
            || failure_tolerance == 0
            || u32::try_from(failure_tolerance).is_err()
        {
            return Err("Paired-relative mode requires a fresh initialized correlated search, capacity >= 2, and a positive u32 failure tolerance".into());
        }
        copy_prefix(&self.scratch.history_slots, &[1], &self.runtime.stream)?;
        copy_prefix(&self.scratch.outcomes, &[0.0], &self.runtime.stream)?;
        copy_prefix(&self.scratch.variances, &[0.0], &self.runtime.stream)?;
        self.paired_relative = Some(failure_tolerance);
        Ok(())
    }

    fn ensure_reference(&mut self) -> CudaResult<()> {
        let Some(seed) = self.reference_seed else {
            return Ok(());
        };
        self.reference =
            DeviceBuffer::zeroed(&self.runtime.stream, self.row_len).map_err(cuda_error)?;
        self.reference_scales =
            DeviceBuffer::zeroed(&self.runtime.stream, self.leaf_count).map_err(cuda_error)?;
        self.reference_partials =
            DeviceBuffer::zeroed(&self.runtime.stream, self.tile_count).map_err(cuda_error)?;
        self.launch_reference(seed, true)?;
        self.check_reference()?;
        self.reference_seed = None;
        Ok(())
    }

    fn check_reference(&self) -> CudaResult<()> {
        let scales = read_prefix(
            &self.reference_scales,
            &self.runtime.stream,
            self.leaf_count,
        )?;
        if scales
            .iter()
            .any(|scale| !scale.is_finite() || *scale <= 0.0)
        {
            return Err("BF16 Gaussian reference must have finite positive tensor RMS".into());
        }
        Ok(())
    }

    pub fn read_reference(&mut self) -> CudaResult<Vec<u16>> {
        if self.sampler != 1 {
            return Err("Only correlated sampling stores a reference".into());
        }
        self.ensure_reference()?;
        self.check_reference()?;
        read_prefix(&self.reference, &self.runtime.stream, self.row_len)
    }

    fn launch_reference(&mut self, seed: u64, initialize: bool) -> CudaResult<()> {
        let blocks = self.tile_count;
        let launch = self
            .runtime
            .module
            .prepare_reference_bf16(LaunchConfig1D::new(
                to_u32(blocks, "BF16 reference blocks")?,
                THREADS,
                0,
            ))
            .map_err(cuda_error)?;
        self.runtime
            .module
            .reference_bf16(
                &self.runtime.stream,
                &launch,
                &mut self.reference,
                &mut self.reference_partials,
                &self.reference_scales,
                &self.leaves,
                &self.tiles,
                &self.scratch.selection,
                &self.scratch.seeds,
                &self.accepted,
                &self.summary,
                seed,
                u32::from(initialize),
            )
            .map_err(cuda_error)?;
        let launch = self
            .runtime
            .module
            .prepare_reference_rms_bf16(LaunchConfig1D::new(
                to_u32(self.leaf_count, "BF16 reference tensors")?,
                THREADS,
                0,
            ))
            .map_err(cuda_error)?;
        self.runtime
            .module
            .reference_rmsbf16(
                &self.runtime.stream,
                &launch,
                &self.reference_partials,
                &self.leaves,
                &self.tile_offsets,
                &mut self.reference_scales,
                &self.accepted,
                &self.summary,
                u32::from(initialize),
            )
            .map_err(cuda_error)
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
        if self.paired_relative.is_some() {
            return Err("Cannot reinitialize paired-relative BF16 search".into());
        }
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
            .ensure(&self.runtime.stream, capacity, 1, 1, 1, self.tile_count)?;
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
            status: 0,
        };
        copy_prefix(&self.state, &[state], &self.runtime.stream)?;
        self.search_capacity = Some(capacity);
        Ok(())
    }

    pub fn tell(
        &mut self,
        trial_slots: &[u32],
        values: &[f32],
        variances: &[f32],
        capacity: usize,
        failure_tolerance: usize,
    ) -> CudaResult<TellOutput> {
        self.validate_tell(trial_slots, values.len(), variances.len(), capacity)?;
        copy_prefix(&self.tell_values, values, &self.runtime.stream)?;
        copy_prefix(&self.tell_variances, variances, &self.runtime.stream)?;
        self.launch_tell(trial_slots, values.len(), capacity, failure_tolerance)?;
        self.collect_tell(values.len())
    }

    pub fn queue_values(
        &mut self,
        trial_slots: &[u32],
        values: &[f32],
        variances: &[f32],
        capacity: usize,
        failure_tolerance: usize,
    ) -> CudaResult<()> {
        self.validate_tell(trial_slots, values.len(), variances.len(), capacity)?;
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
        self.validate_tell(trial_slots, count, count, capacity)?;
        if values == 0 || variances == Some(0) {
            return Err("CUDA BF16 tell requires valid device rewards".to_string());
        }
        let bytes = count
            .checked_mul(size_of::<f32>())
            .ok_or("CUDA BF16 tell byte count overflow")?;
        unsafe {
            cuda_core::simt::memory::memcpy_dtod_async(
                self.tell_values.cu_deviceptr(),
                values,
                bytes,
                self.runtime.stream.cu_stream(),
            )
            .map_err(cuda_error)?;
            if let Some(pointer) = variances {
                cuda_core::simt::memory::memcpy_dtod_async(
                    self.tell_variances.cu_deviceptr(),
                    pointer,
                    bytes,
                    self.runtime.stream.cu_stream(),
                )
                .map_err(cuda_error)?;
            } else {
                cuda_core::simt::memory::memset_d8_async(
                    self.tell_variances.cu_deviceptr(),
                    0,
                    bytes,
                    self.runtime.stream.cu_stream(),
                )
                .map_err(cuda_error)?;
            }
        }
        self.launch_tell(trial_slots, count, capacity, failure_tolerance)?;
        self.collect_tell(count)
    }

    /// Queue resident rewards and trust-region adaptation without a host readback.
    ///
    /// # Safety
    /// The pointers must each address `count * 4` readable bytes on CUDA device 0.
    pub unsafe fn queue_tell(
        &mut self,
        trial_slots: &[u32],
        values: u64,
        variances: Option<u64>,
        count: usize,
        capacity: usize,
        failure_tolerance: usize,
    ) -> CudaResult<()> {
        self.validate_tell(trial_slots, count, count, capacity)?;
        if values == 0 || variances == Some(0) {
            return Err("CUDA BF16 tell requires valid device rewards".to_string());
        }
        let bytes = count
            .checked_mul(size_of::<f32>())
            .ok_or("CUDA BF16 tell byte count overflow")?;
        unsafe {
            cuda_core::simt::memory::memcpy_dtod_async(
                self.tell_values.cu_deviceptr(),
                values,
                bytes,
                self.runtime.stream.cu_stream(),
            )
            .map_err(cuda_error)?;
            if let Some(pointer) = variances {
                cuda_core::simt::memory::memcpy_dtod_async(
                    self.tell_variances.cu_deviceptr(),
                    pointer,
                    bytes,
                    self.runtime.stream.cu_stream(),
                )
                .map_err(cuda_error)?;
            } else {
                cuda_core::simt::memory::memset_d8_async(
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

    fn validate_tell(
        &self,
        trial_slots: &[u32],
        values: usize,
        variances: usize,
        capacity: usize,
    ) -> CudaResult<()> {
        if self.sampler == 1 {
            self.check_reference()?;
        }
        if trial_slots.is_empty()
            || trial_slots.len() > BF16_PENDING
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
    ) -> CudaResult<()> {
        self.launch_tellupdate(trial_slots, count, capacity, failure_tolerance, None, None)
    }

    /// Queue one correlated observation and an explicit paired promotion decision.
    #[allow(clippy::too_many_arguments)]
    pub fn queue_paired(
        &mut self,
        trial_slot: u32,
        value: f32,
        variance: f32,
        incumbent_value: f32,
        incumbent_variance: f32,
        accept: bool,
        capacity: usize,
    ) -> CudaResult<()> {
        if self.sampler != 1
            || !value.is_finite()
            || !variance.is_finite()
            || variance < 0.0
            || !incumbent_value.is_finite()
            || !incumbent_variance.is_finite()
            || incumbent_variance < 0.0
        {
            return Err(
                "Paired tell requires correlated sampling and finite rewards/nonnegative variances"
                    .into(),
            );
        }
        self.validate_tell(&[trial_slot], 1, 1, capacity)?;
        copy_prefix(&self.tell_values, &[value], &self.runtime.stream)?;
        copy_prefix(&self.tell_variances, &[variance], &self.runtime.stream)?;
        self.launch_tellupdate(
            &[trial_slot],
            1,
            capacity,
            1,
            Some((incumbent_value, incumbent_variance, accept)),
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn queue_relative(
        &mut self,
        trial_slot: u32,
        value: f32,
        variance: f32,
        incumbent_value: f32,
        incumbent_variance: f32,
        improvement: f32,
        improvement_variance: f32,
        accept: bool,
        capacity: usize,
        reject_is_failure: bool,
    ) -> CudaResult<()> {
        let tolerance = self
            .paired_relative
            .ok_or("Enable paired-relative mode before telling relative observations")?;
        validate_tell(
            self.search_capacity,
            self.slots,
            capacity,
            trial_slot,
            self.single_slot,
        )?;
        if self.search_capacity != Some(capacity)
            || self.sampler != 1
            || [value, incumbent_value, improvement]
                .iter()
                .any(|value| !value.is_finite())
            || [variance, incumbent_variance, improvement_variance]
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err("Paired-relative tell requires the initialized capacity, finite rewards and improvement, and nonnegative finite variances".into());
        }
        self.validate_tell(&[trial_slot], 1, 1, capacity)?;
        copy_prefix(&self.tell_values, &[value], &self.runtime.stream)?;
        copy_prefix(&self.tell_variances, &[variance], &self.runtime.stream)?;
        self.launch_tellupdate(
            &[trial_slot],
            1,
            capacity,
            tolerance,
            Some((incumbent_value, incumbent_variance, accept)),
            Some((improvement, improvement_variance, reject_is_failure)),
        )?;
        self.single_slot = None;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_tellupdate(
        &mut self,
        trial_slots: &[u32],
        count: usize,
        capacity: usize,
        failure_tolerance: usize,
        paired: Option<(f32, f32, bool)>,
        relative: Option<(f32, f32, bool)>,
    ) -> CudaResult<()> {
        if self.paired_relative.is_some() != relative.is_some() {
            return Err("Cannot mix paired-relative and legacy BF16 tells".into());
        }
        self.started = true;
        // Legacy DLPack has no read-only flag. Reject consumer writes before
        // any reward or history update can accept the borrowed pending row.
        if let Some(single) = self.single_slot {
            if trial_slots == [single.slot as u32] {
                self.runtime.context.synchronize().map_err(cuda_error)?;
                let launch = self
                    .runtime
                    .module
                    .prepare_write_bf16(LaunchConfig1D::new(
                        to_u32(self.tile_count, "BF16 verification tiles")?,
                        THREADS,
                        0,
                    ))
                    .map_err(cuda_error)?;
                self.runtime
                    .module
                    .write_bf16(
                        &self.runtime.stream,
                        &launch,
                        &mut self.rows,
                        &mut self.batch,
                        &mut self.scratch.changes,
                        &self.state,
                        &self.scratch.seeds,
                        &self.reference,
                        &self.reference_scales,
                        &self.scratch.selection,
                        &self.leaves,
                        &self.tiles,
                        &self.scratch.trial_slots,
                        &mut self.scratch.status,
                        self.row_stride as u64,
                        self.row_len as u64,
                        to_u32(single.base_slot, "BF16 base slot")?,
                        to_u32(self.tile_count, "BF16 tile count")?,
                        single.coefficient,
                        u32::from(single.resident),
                        0,
                        1,
                        self.sampler,
                    )
                    .map_err(cuda_error)?;
            }
        }
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
            status_count: to_u32(
                count
                    .checked_mul(self.tile_count)
                    .ok_or("BF16 proposal status count overflow")?,
                "BF16 proposal status count",
            )?,
            correlated: self.sampler,
            paired: if relative.is_some() {
                2
            } else {
                u32::from(paired.is_some())
            },
            accept: u32::from(paired.is_some_and(|(_, _, accept)| accept)),
            incumbent_value: paired.map_or(0.0, |(value, _, _)| value),
            incumbent_variance: paired.map_or(0.0, |(_, variance, _)| variance),
            improvement: relative.map_or(0.0, |(value, _, _)| value),
            improvement_variance: relative.map_or(0.0, |(_, variance, _)| variance),
            reject_is_failure: u32::from(relative.is_none_or(|(_, _, failure)| failure)),
        };
        self.runtime
            .module
            .tell_bf16(
                &self.runtime.stream,
                &launch,
                &mut self.scratch.history_slots,
                &mut self.scratch.outcomes,
                &mut self.scratch.variances,
                &self.scratch.trial_slots,
                &mut self.scratch.destinations,
                &self.tell_values,
                &self.tell_variances,
                &self.scratch.status,
                &mut self.accepted,
                &mut self.state,
                &mut self.summary,
                &self.scratch.selection,
                &mut self.best_source,
                params,
            )
            .map_err(cuda_error)?;
        let blocks = self.row_len.div_ceil(THREADS as usize).min(65_535);
        let launch = self
            .runtime
            .module
            .prepare_copy_tell_bf16(LaunchConfig1D::new(
                to_u32(blocks, "BF16 copy blocks")?,
                THREADS,
                0,
            ))
            .map_err(cuda_error)?;
        self.runtime
            .module
            .copy_tellbf16(
                &self.runtime.stream,
                &launch,
                &mut self.rows,
                &self.scratch.trial_slots,
                &self.scratch.destinations,
                &self.summary,
                &self.best_source,
                params,
            )
            .map_err(cuda_error)?;
        if self.sampler == 1 {
            self.launch_reference(0, false)?;
        }
        self.runtime.context.check_err().map_err(cuda_error)
    }

    pub fn collect_tell(&self, count: usize) -> CudaResult<TellOutput> {
        if count == 0 || count > BF16_PENDING {
            return Err("CUDA BF16 summary count is invalid".to_string());
        }
        let summary = read_prefix(&self.summary, &self.runtime.stream, 1)?[0];
        if summary.status != 0 {
            return Err("CUDA BF16 proposal or rewards are invalid".to_string());
        }
        if self.sampler == 1 {
            self.check_reference()?;
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
        let input = AskInput {
            base_slot,
            history,
            trial_slots,
            seeds,
            seed_root: None,
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
            self.last_profile = Some(profile);
        }
        Ok(selections)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn ask_seeded(
        &mut self,
        base_slot: usize,
        history: usize,
        trial_slots: &[u32],
        candidates_per_region: usize,
        seed_root: u64,
        coefficient: f32,
        draw_seed: u64,
        config: Ask,
    ) -> CudaResult<()> {
        self.last_profile = None;
        let input = AskInput {
            base_slot,
            history,
            trial_slots,
            seeds: &[],
            seed_root: Some(seed_root),
            candidates_per_region,
            coefficient,
            draw_seed,
            config,
        };
        let shape = self.check_ask(&input)?;
        self.upload(&input, shape)?;
        let events = self.launch(&input, shape)?;
        if let Some(events) = events {
            let profile = events.profile()?;
            self.last_profile = Some(profile);
        }
        Ok(())
    }

    fn check_ask(&self, input: &AskInput<'_>) -> CudaResult<SearchShape> {
        if self.paired_relative.is_some() {
            validate_layout(
                self.search_capacity,
                self.slots,
                input.base_slot,
                input.history,
                input.trial_slots,
            )?;
            if self.single_slot.is_some() {
                return Err("Tell the pending paired-relative proposal before another ask".into());
            }
        }
        if self.sampler == 1 && self.reference_seed.is_none() {
            self.check_reference()?;
        }
        if self.sampler == 1
            && (input.seed_root.is_none()
                || input.trial_slots.len() != 1
                || input.candidates_per_region != 4)
        {
            return Err(
                "Correlated BF16 search requires one seeded arm with four candidates".into(),
            );
        }
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
        if input.seed_root.is_none() && input.seeds.len() != candidates {
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
        self.started = true;
        self.ensure_reference()?;
        self.scratch.ensure(
            &self.runtime.stream,
            input.history,
            shape.candidates,
            shape.regions,
            shape.partials,
            shape.status,
        )?;
        let batch_len = if shape.regions == 1 {
            1
        } else {
            shape
                .regions
                .checked_mul(self.row_len)
                .ok_or("CUDA BF16 batch size overflow")?
        };
        if batch_len > self.batch.len() {
            self.batch =
                DeviceBuffer::zeroed(&self.runtime.stream, batch_len).map_err(cuda_error)?;
        }
        if let Some(root) = input.seed_root {
            let launch = self
                .runtime
                .module
                .prepare_seed_bf16(LaunchConfig1D::new(
                    to_u32(
                        shape.candidates.div_ceil(THREADS as usize),
                        "BF16 seed blocks",
                    )?,
                    THREADS,
                    0,
                ))
                .map_err(cuda_error)?;
            self.runtime
                .module
                .seed_bf16(
                    &self.runtime.stream,
                    &launch,
                    &mut self.scratch.seeds,
                    root,
                    to_u32(shape.candidates, "BF16 seed count")?,
                    self.sampler,
                )
                .map_err(cuda_error)?;
        } else {
            let seeds = input
                .seeds
                .iter()
                .map(|seed| Seed {
                    low: *seed as u32,
                    high: (*seed >> 32) as u32,
                })
                .collect::<Vec<_>>();
            copy_prefix(&self.scratch.seeds, &seeds, &self.runtime.stream)?;
        }
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
                to_u32(input.history.div_ceil(THREADS as usize), "BF16 draw blocks")?,
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
        let profile = self.profiling || std::env::var_os("ENNX_CUDA_PROFILE").is_some();
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
            resident: u32::from(input.seed_root.is_some()),
            correlated: self.sampler,
        };
        self.runtime
            .module
            .distance_bf16(
                &self.runtime.stream,
                &distance_launch,
                &self.rows,
                &self.scratch.history_slots,
                &self.reference,
                &self.reference_scales,
                &self.state,
                &self.scratch.seeds,
                &self.leaves,
                &self.tiles,
                &mut self.scratch.partials,
                &mut self.scratch.tile_status,
                params,
            )
            .map_err(cuda_error)?;
        self.runtime
            .module
            .draw_bf16(
                &self.runtime.stream,
                &draw_launch,
                &mut self.scratch.draws,
                &self.scratch.history_slots,
                &self.state,
                input.draw_seed,
                params.history,
                params.resident,
            )
            .map_err(cuda_error)?;
        self.runtime
            .module
            .score_bf16(
                &self.runtime.stream,
                &score_launch,
                &self.scratch.partials,
                &self.scratch.tile_status,
                &self.scratch.outcomes,
                &self.scratch.variances,
                &self.state,
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
                &mut self.batch,
                &mut self.scratch.changes,
                &self.state,
                &self.scratch.seeds,
                &self.reference,
                &self.reference_scales,
                &self.scratch.selection,
                &self.leaves,
                &self.tiles,
                &self.scratch.trial_slots,
                &mut self.scratch.status,
                self.row_stride as u64,
                self.row_len as u64,
                to_u32(input.base_slot, "BF16 base slot")?,
                to_u32(self.tile_count, "BF16 tile count")?,
                input.coefficient,
                u32::from(input.seed_root.is_some()),
                u32::from(shape.regions > 1),
                0,
                self.sampler,
            )
            .map_err(cuda_error)?;
        let materialize_end = profile
            .then(|| timing_event(&self.runtime.stream))
            .transpose()?;
        self.runtime.context.check_err().map_err(cuda_error)?;
        self.single_slot = (shape.regions == 1).then_some(SingleRow {
            slot: input.trial_slots[0] as usize,
            base_slot: input.base_slot,
            coefficient: input.coefficient,
            resident: input.seed_root.is_some(),
        });
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
            cuda_core::simt::memory::memcpy_dtod_async(
                self.row_pointer(destination)?,
                self.row_pointer(source)?,
                row_bytes(self.row_len)?,
                self.runtime.stream.cu_stream(),
            )
            .map_err(cuda_error)
        }
    }

    /// Reuse an unused pending row as a disposable incumbent snapshot.
    pub fn snapshot_incumbent(&self, destination: usize) -> CudaResult<()> {
        if destination == 0 {
            return Err("The incumbent snapshot must not alias canonical row zero".into());
        }
        // A released consumer may have used a stream other than our producer stream.
        self.runtime.context.synchronize().map_err(cuda_error)?;
        self.copy_row(0, destination)
    }

    pub fn sync_consumers(&self) -> CudaResult<()> {
        self.runtime.context.synchronize().map_err(cuda_error)
    }

    pub fn read(&self, slot: usize) -> CudaResult<Vec<u16>> {
        self.check_slot(slot)?;
        let mut output = Vec::<u16>::with_capacity(self.row_len);
        unsafe {
            cuda_core::simt::memory::memcpy_dtoh_async(
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

    /// Gather resident trial slots into one contiguous device matrix.
    pub fn device_batch(
        &mut self,
        slots: &[u32],
        stream: Option<i64>,
    ) -> CudaResult<(u64, usize, usize)> {
        if slots.is_empty() || slots.len() > BF16_PENDING {
            return Err("CUDA BF16 batch shape is invalid".to_string());
        }
        if slots.len() == 1 {
            let (pointer, _, _) = self.device_row(slots[0] as usize, stream)?;
            return Ok((pointer, 1, self.row_len));
        }
        for &slot in slots {
            self.check_slot(slot as usize)?;
        }
        let elements = slots
            .len()
            .checked_mul(self.row_len)
            .ok_or("CUDA BF16 batch size overflow")?;
        if elements > self.batch.len() {
            self.batch =
                DeviceBuffer::zeroed(&self.runtime.stream, elements).map_err(cuda_error)?;
        }
        let bytes = row_bytes(self.row_len)?;
        for (row, &slot) in slots.iter().enumerate() {
            let offset = row
                .checked_mul(bytes)
                .ok_or("CUDA BF16 batch offset overflow")?;
            let destination = self
                .batch
                .cu_deviceptr()
                .checked_add(offset as u64)
                .ok_or("CUDA BF16 batch pointer overflow")?;
            unsafe {
                cuda_core::simt::memory::memcpy_dtod_async(
                    destination,
                    self.row_pointer(slot as usize)?,
                    bytes,
                    self.runtime.stream.cu_stream(),
                )
                .map_err(cuda_error)?;
            }
        }
        sync_stream(&self.runtime, stream)?;
        Ok((self.batch.cu_deviceptr(), slots.len(), self.row_len))
    }

    pub fn device_round(
        &self,
        rows: usize,
        stream: Option<i64>,
    ) -> CudaResult<(u64, usize, usize)> {
        if rows == 1 {
            if let Some(single) = self.single_slot {
                let (pointer, _, _) = self.device_row(single.slot, stream)?;
                return Ok((pointer, 1, self.row_len));
            }
        }
        let elements = rows
            .checked_mul(self.row_len)
            .ok_or("CUDA BF16 round size overflow")?;
        if rows == 0 || elements > self.batch.len() {
            return Err("CUDA BF16 round shape is invalid".to_string());
        }
        sync_stream(&self.runtime, stream)?;
        Ok((self.batch.cu_deviceptr(), rows, self.row_len))
    }

    pub fn last_profile(&self) -> Option<AskProfile> {
        self.last_profile
    }

    pub fn describe(&self, arms: usize) -> CudaResult<Vec<ProposalDescription>> {
        if arms == 0 || arms > self.scratch.region_capacity {
            return Err("Invalid BF16 proposal description count".into());
        }
        let count = arms
            .checked_mul(self.tile_count)
            .ok_or("BF16 diagnostics overflow")?;
        let status = read_prefix(&self.scratch.status, &self.runtime.stream, count)?;
        if status.iter().any(|&value| value != 0) {
            return Err("Cannot describe an invalid BF16 proposal".into());
        }
        let choices = read_prefix(&self.scratch.selection, &self.runtime.stream, arms)?;
        let seeds = read_prefix(
            &self.scratch.seeds,
            &self.runtime.stream,
            self.scratch.candidate_capacity,
        )?;
        let changes = read_prefix(&self.scratch.changes, &self.runtime.stream, count)?;
        let state = read_prefix(&self.state, &self.runtime.stream, 1)?;
        let mut descriptions = Vec::with_capacity(arms);
        for (arm, choice) in choices.iter().enumerate() {
            if !choice.score.is_finite() {
                return Err("BF16 acquisition score is nonfinite".into());
            }
            let seed = seeds
                .get(choice.index as usize)
                .ok_or("Invalid BF16 selected seed")?;
            let mut blocks = vec![(0_u64, 0.0_f64); self.leaf_count];
            for (tile, &leaf) in self.tile_leaves.iter().enumerate() {
                let change = changes[arm * self.tile_count + tile];
                if !change.squared.is_finite() || change.squared < 0.0 {
                    return Err("Nonfinite BF16 perturbation diagnostics".into());
                }
                blocks[leaf].0 += u64::from(change.changed);
                blocks[leaf].1 += f64::from(change.squared);
            }
            descriptions.push((
                u64::from(seed.low) | (u64::from(seed.high) << 32),
                choice.score,
                if self.sampler == 1 {
                    let factor = if choice.index & 1 == 0 { 0.5 } else { 2.0 };
                    (state[0].length * factor)
                        .max(state[0].length_min)
                        .min(state[0].length_max) as f32
                } else {
                    state[0].length as f32
                },
                blocks,
            ));
        }
        Ok(descriptions)
    }

    pub fn geometry(&self, arms: usize) -> CudaResult<Vec<(usize, f32)>> {
        if arms == 0 || arms > self.scratch.region_capacity {
            return Err("Invalid BF16 proposal geometry count".into());
        }
        let choices = read_prefix(&self.scratch.selection, &self.runtime.stream, arms)?;
        choices
            .into_iter()
            .map(|choice| {
                if !choice.score.is_finite()
                    || choice.index as usize >= self.scratch.candidate_capacity
                {
                    return Err("Invalid BF16 proposal geometry".into());
                }
                let persistence = if self.sampler == 1 && choice.index < 2 {
                    0.75
                } else {
                    0.0
                };
                Ok((choice.index as usize, persistence))
            })
            .collect()
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
            cuda_core::simt::memory::memset_d8_async(
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

fn validate_layout(
    configured_capacity: Option<usize>,
    slots: usize,
    base_slot: usize,
    capacity: usize,
    trial_slots: &[u32],
) -> CudaResult<()> {
    if configured_capacity != Some(capacity)
        || capacity < 2
        || base_slot != 0
        || trial_slots.len() != 1
        || trial_slots[0] as usize <= capacity
        || trial_slots[0] as usize >= slots
    {
        return Err("Paired-relative search requires base slot 0, the initialized history capacity, and one trial in pending storage beyond history".into());
    }
    Ok(())
}

fn validate_tell(
    configured_capacity: Option<usize>,
    slots: usize,
    capacity: usize,
    trial_slot: u32,
    pending: Option<SingleRow>,
) -> CudaResult<()> {
    validate_layout(configured_capacity, slots, 0, capacity, &[trial_slot])?;
    if !pending.is_some_and(|pending| {
        pending.slot == trial_slot as usize && pending.base_slot == 0 && pending.resident
    }) {
        return Err("Paired-relative tell must match the outstanding selected trial slot".into());
    }
    Ok(())
}

#[cfg(test)]
mod relative_validation_tests {
    use super::{SingleRow, validate_layout, validate_tell};

    #[test]
    fn layout_guards() {
        for capacity in [2, 3, 128] {
            let slots = capacity + 3;
            let pending = (capacity + 1) as u32;
            assert!(validate_layout(Some(capacity), slots, 0, capacity, &[pending]).is_ok());
            for slot in 0..=capacity {
                assert!(
                    validate_layout(Some(capacity), slots, 0, capacity, &[slot as u32]).is_err()
                );
            }
            for invalid in [slots as u32, u32::MAX] {
                assert!(validate_layout(Some(capacity), slots, 0, capacity, &[invalid]).is_err());
            }
            for history in [0, 1, capacity - 1, capacity + 1, 256] {
                assert!(validate_layout(Some(capacity), slots, 0, history, &[pending]).is_err());
            }
            assert!(validate_layout(None, slots, 0, capacity, &[pending]).is_err());
            assert!(validate_layout(Some(capacity), slots, 1, capacity, &[pending]).is_err());
            assert!(validate_layout(Some(capacity), slots, 0, capacity, &[]).is_err());
            assert!(
                validate_layout(Some(capacity), slots, 0, capacity, &[pending, pending + 1])
                    .is_err()
            );
        }
    }

    #[test]
    fn tell_selection() {
        let pending = SingleRow {
            slot: 3,
            base_slot: 0,
            coefficient: 1.0,
            resident: true,
        };
        assert!(validate_tell(Some(2), 5, 2, 3, Some(pending)).is_ok());
        assert!(validate_tell(Some(2), 5, 2, 3, None).is_err());
        assert!(validate_tell(Some(2), 5, 2, 4, Some(pending)).is_err());
        assert!(validate_tell(Some(2), 5, 3, 4, Some(pending)).is_err());
        for invalid in [
            SingleRow {
                base_slot: 1,
                ..pending
            },
            SingleRow {
                resident: false,
                ..pending
            },
        ] {
            assert!(validate_tell(Some(2), 5, 2, 3, Some(invalid)).is_err());
        }
        for live in 0..=2 {
            assert!(
                validate_tell(
                    Some(2),
                    5,
                    2,
                    live,
                    Some(SingleRow {
                        slot: live as usize,
                        ..pending
                    })
                )
                .is_err()
            );
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
            let tile_length = (length - start).min(DENSE_ELEMENTS);
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
