use std::collections::BTreeSet;
use std::mem::size_of;

use cuda_core::{CudaStream, DeviceBuffer, LaunchConfig1D};

use super::*;

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
    history_slots: &'a [u32],
    outcomes: &'a [f32],
    variances: &'a [f32],
    trial_slots: &'a [u32],
    seeds: &'a [u64],
    draws: &'a [f32],
    candidates_per_region: usize,
    coefficient: f32,
    config: Ask,
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
            profiling: false,
            last_profile: None,
        })
    }

    pub fn set_profiling(&mut self, enabled: bool) {
        set_profile(enabled, &mut self.profiling, &mut self.last_profile);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn ask(
        &mut self,
        base_slot: usize,
        history_slots: &[u32],
        outcomes: &[f32],
        variances: &[f32],
        trial_slots: &[u32],
        seeds: &[u64],
        draws: &[f32],
        candidates_per_region: usize,
        coefficient: f32,
        config: Ask,
    ) -> CudaResult<Vec<Selection>> {
        self.last_profile = None;
        let client = TRACY.get_or_init(tracy_client::Client::start);
        let _zone = client
            .clone()
            .span(tracy_client::span_location!("ennx.cuda.bf16.ask"), 0);
        let input = AskInput {
            base_slot,
            history_slots,
            outcomes,
            variances,
            trial_slots,
            seeds,
            draws,
            candidates_per_region,
            coefficient,
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
        let history = input.history_slots.len();
        let regions = input.trial_slots.len();
        if history == 0 || history > MAX_HISTORY {
            return Err(format!(
                "CUDA BF16 history must contain 1..={MAX_HISTORY} rows"
            ));
        }
        if input.outcomes.len() != history || input.variances.len() != history {
            return Err("CUDA BF16 history, outcomes, and variances differ in length".to_string());
        }
        if regions == 0 || input.candidates_per_region == 0 {
            return Err("CUDA BF16 search requires regions and candidates".to_string());
        }
        let candidates = regions
            .checked_mul(input.candidates_per_region)
            .ok_or("CUDA BF16 candidate count overflow")?;
        if input.seeds.len() != candidates || input.draws.len() != candidates {
            return Err("CUDA BF16 seeds and draws do not match the search shape".to_string());
        }
        if input.config.neighbors == 0 || input.config.neighbors > history {
            return Err("CUDA BF16 neighbor count exceeds resident history".to_string());
        }
        if !input.coefficient.is_finite() || input.coefficient <= 0.0 {
            return Err("CUDA BF16 perturbation coefficient must be positive".to_string());
        }
        validate_ask(input)?;
        for &slot in input.history_slots {
            self.check_slot(slot as usize)?;
        }
        let mut destinations = BTreeSet::new();
        for &slot in input.trial_slots {
            self.check_slot(slot as usize)?;
            if slot as usize == input.base_slot
                || input.history_slots.contains(&slot)
                || !destinations.insert(slot)
            {
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
            input.history_slots.len(),
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
        copy_prefix(
            &self.scratch.history_slots,
            input.history_slots,
            &self.runtime.stream,
        )?;
        copy_prefix(&self.scratch.outcomes, input.outcomes, &self.runtime.stream)?;
        copy_prefix(
            &self.scratch.variances,
            input.variances,
            &self.runtime.stream,
        )?;
        copy_prefix(&self.scratch.seeds, &seeds, &self.runtime.stream)?;
        copy_prefix(&self.scratch.draws, input.draws, &self.runtime.stream)?;
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
            history: to_u32(input.history_slots.len(), "BF16 history rows")?,
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
    if input.outcomes.iter().any(|value| !value.is_finite())
        || input
            .variances
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        || input.draws.iter().any(|value| !value.is_finite())
    {
        return Err("CUDA BF16 observations and draws must be finite".to_string());
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
