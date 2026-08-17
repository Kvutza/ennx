use std::collections::VecDeque;

use ndarray::ArrayView2;

use crate::weights::{AcquisitionKind, ComputeDevice};

#[cfg(all(target_os = "macos", feature = "metal"))]
mod metal;

#[cfg(feature = "opencl")]
mod opencl;

#[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
mod cuda;

mod bpann_history;
mod engine;
mod layout;
mod tree;

mod sparse;

pub use bpann_history::{BpannHistory, IndexedObservation, ObservationId};
pub(crate) use layout::{check_layout, make_steps, LeafStep};
#[cfg(any(
    all(feature = "cuda", target_os = "linux", target_arch = "x86_64"),
    all(feature = "metal", target_os = "macos"),
    feature = "opencl"
))]
pub(crate) use layout::{make_tiles, Tile};
pub use tree::Center;

const MAX_HISTORY: usize = 128;
const MAX_PENDING: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodingType {
    Int4 = 0,
    Int8 = 1,
    Fp4E2M1 = 2,
    Fp8E4M3 = 3,
    Fp8E5M2 = 4,
}

impl EncodingType {
    pub fn parse(bits: u8, mode: Option<&str>) -> Result<Self, String> {
        match (bits, mode.map(|s| s.trim().to_ascii_lowercase()).as_deref()) {
            (4, None | Some("int") | Some("int4")) => Ok(Self::Int4),
            (8, None | Some("int") | Some("int8")) => Ok(Self::Int8),
            (4, Some("fp4") | Some("fp4_e2m1") | Some("e2m1")) => Ok(Self::Fp4E2M1),
            (8, Some("fp8") | Some("fp8_e4m3") | Some("e4m3")) => Ok(Self::Fp8E4M3),
            (8, Some("fp8_e5m2") | Some("e5m2")) => Ok(Self::Fp8E5M2),
            _ => Err(format!("unsupported encoding bits={bits}, mode={mode:?}")),
        }
    }
}

pub static FP4_E2M1_LUT: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

pub fn decode_code(code: u32, encoding: EncodingType, scale: f32) -> f32 {
    match encoding {
        EncodingType::Int4 | EncodingType::Int8 => code as f32 * scale,
        EncodingType::Fp4E2M1 => FP4_E2M1_LUT[(code & 0x0f) as usize] * scale,
        EncodingType::Fp8E4M3 => decode_fp8_e4m3(code as u8) * scale,
        EncodingType::Fp8E5M2 => decode_fp8_e5m2(code as u8) * scale,
    }
}

pub fn decode_fp8_e4m3(byte: u8) -> f32 {
    let sign = if (byte & 0x80) != 0 { -1.0 } else { 1.0 };
    let exp = (byte >> 3) & 0x0f;
    let mant = byte & 0x07;
    if exp == 0 {
        sign * (mant as f32 / 8.0) * (2.0f32).powi(-6)
    } else if exp == 15 && mant == 7 {
        f32::NAN
    } else {
        sign * (1.0 + mant as f32 / 8.0) * (2.0f32).powi(exp as i32 - 7)
    }
}

pub fn decode_fp8_e5m2(byte: u8) -> f32 {
    let sign = if (byte & 0x80) != 0 { -1.0 } else { 1.0 };
    let exp = (byte >> 2) & 0x1f;
    let mant = byte & 0x03;
    if exp == 0 {
        sign * (mant as f32 / 4.0) * (2.0f32).powi(-14)
    } else if exp == 31 {
        if mant == 0 {
            sign * f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        sign * (1.0 + mant as f32 / 4.0) * (2.0f32).powi(exp as i32 - 15)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Leaf {
    pub offset: usize,
    pub length: usize,
    pub bits: u8,
    pub encoding: EncodingType,
    pub scale: f32,
    pub weight: f32,
    pub radius: f32,
}

impl Leaf {
    pub fn new(
        offset: usize,
        length: usize,
        bits: u8,
        scale: f32,
        weight: f32,
        radius: f32,
    ) -> Result<Self, String> {
        Self::new_with_encoding(
            offset,
            length,
            bits,
            EncodingType::parse(bits, None)?,
            scale,
            weight,
            radius,
        )
    }

    pub fn new_with_encoding(
        offset: usize,
        length: usize,
        bits: u8,
        encoding: EncodingType,
        scale: f32,
        weight: f32,
        radius: f32,
    ) -> Result<Self, String> {
        if length == 0 {
            return Err("leaf length must be positive".to_string());
        }
        if bits != 4 && bits != 8 {
            return Err(format!("leaf bits must be 4 or 8, got {bits}"));
        }
        for (name, value) in [("scale", scale), ("weight", weight), ("radius", radius)] {
            if !value.is_finite() || value <= 0.0 {
                return Err(format!("{name} must be finite and positive"));
            }
        }
        Ok(Self {
            offset,
            length,
            bits,
            encoding,
            scale,
            weight,
            radius,
        })
    }

    fn bytes(self) -> usize {
        match self.bits {
            4 => self.length.div_ceil(2),
            8 => self.length,
            _ => unreachable!("leaf width is checked at construction"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Ask {
    pub length: f32,
    pub neighbors: usize,
    pub epistemic_scale: f32,
    pub aleatoric_scale: f32,
    pub y_scale: f32,
    pub beta: f32,
    pub acquisition: AcquisitionKind,
    pub seed: u64,
}

impl Default for Ask {
    fn default() -> Self {
        Self {
            length: 0.8,
            neighbors: 10,
            epistemic_scale: 0.7,
            aleatoric_scale: 0.05,
            y_scale: 1.0,
            beta: 1.0,
            acquisition: AcquisitionKind::Ucb,
            seed: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Trial {
    id: u64,
    pub index: usize,
    pub seed: u64,
    pub score: f32,
}

#[derive(Debug, Clone, Copy)]
struct Record {
    slot: usize,
    value: f32,
}

#[derive(Debug, Clone, Copy)]
struct Pending {
    id: u64,
    slot: usize,
    seed: u64,
    length: f32,
    materialized: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SparseEdit {
    pub leaf: u32,
    pub element: u32,
}

enum Engine {
    Cpu(Cpu),
    #[cfg(all(target_os = "macos", feature = "metal"))]
    Metal(metal::Engine),
    #[cfg(feature = "opencl")]
    OpenCl(opencl::Engine),
    #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
    Cuda(cuda::Engine),
}

pub struct Search {
    leaves: Vec<Leaf>,
    row_bytes: usize,
    capacity: usize,
    pending_capacity: usize,
    slots: usize,
    base: usize,
    history: VecDeque<Record>,
    pending: Vec<Pending>,
    next_id: u64,
    engine: Engine,
}

impl Search {
    pub fn new(
        base: &[u8],
        base_value: f32,
        leaves: Vec<Leaf>,
        capacity: usize,
        device: ComputeDevice,
    ) -> Result<Self, String> {
        Self::new_batch(base, base_value, leaves, capacity, 1, device)
    }

    pub fn new_batch(
        base: &[u8],
        base_value: f32,
        leaves: Vec<Leaf>,
        capacity: usize,
        pending_capacity: usize,
        device: ComputeDevice,
    ) -> Result<Self, String> {
        if !base_value.is_finite() {
            return Err("base value must be finite".to_string());
        }
        if capacity == 0 || capacity > MAX_HISTORY {
            return Err(format!("history capacity must be in 1..={MAX_HISTORY}"));
        }
        if pending_capacity == 0 || pending_capacity > MAX_PENDING {
            return Err(format!("pending capacity must be in 1..={MAX_PENDING}"));
        }
        let row_bytes = check_layout(&leaves)?;
        if base.len() != row_bytes {
            return Err(format!(
                "base row has {} bytes, expected {row_bytes}",
                base.len()
            ));
        }
        let slots = capacity
            .checked_add(pending_capacity)
            .and_then(|slots| slots.checked_add(1))
            .ok_or("resident slot count overflow")?;
        let engine = Engine::new(base, &leaves, slots, device)?;
        Ok(Self {
            leaves,
            row_bytes,
            capacity,
            pending_capacity,
            slots,
            base: 0,
            history: VecDeque::from([Record {
                slot: 0,
                value: base_value,
            }]),
            pending: Vec::with_capacity(pending_capacity),
            next_id: 0,
            engine,
        })
    }

    pub fn ask(&mut self, seeds: &[u64], config: Ask) -> Result<Trial, String> {
        self.ask_with_materialization(seeds, config, true)
    }

    /// Select a seed without materializing its full weight row.
    ///
    /// This is the path used when the evaluator regenerates the perturbation
    /// from the returned seed.  Materializing a billion-parameter row during
    /// proposal would make `ask` scale with model size for no benefit.
    pub fn ask_lazy(&mut self, seeds: &[u64], config: Ask) -> Result<Trial, String> {
        self.ask_with_materialization(seeds, config, false)
    }

    pub(crate) fn ask_sparse(
        &mut self,
        seeds: &[u64],
        num_pert: usize,
        config: Ask,
    ) -> Result<Trial, String> {
        if !self.pending.is_empty() {
            return Err("tell must finish the pending trial before ask".to_string());
        }
        self.ask_sparse_open(seeds, num_pert, config)
    }

    pub(crate) fn ask_batch(
        &mut self,
        seeds: &[u64],
        arms: usize,
        num_pert: usize,
        config: Ask,
    ) -> Result<Vec<Trial>, String> {
        if !self.pending.is_empty() {
            return Err("tell must finish outstanding trials before batch ask".to_string());
        }
        if arms == 0 || arms > self.pending_capacity {
            return Err(format!(
                "batch arms must be in 1..={}, got {arms}",
                self.pending_capacity
            ));
        }
        if seeds.is_empty() || seeds.len() % arms != 0 {
            return Err("batch seeds must divide evenly into non-empty arms".to_string());
        }
        let candidates = seeds.len() / arms;
        if candidates == 0 {
            return Err("each batch arm requires candidates".to_string());
        }
        let mut trials = Vec::with_capacity(arms);
        for group in seeds.chunks_exact(candidates) {
            match self.ask_sparse_open(group, num_pert, config) {
                Ok(trial) => trials.push(trial),
                Err(error) => {
                    self.pending.clear();
                    return Err(error);
                }
            }
        }
        Ok(trials)
    }

    pub(crate) fn pending_len(&self) -> usize {
        self.pending.len()
    }

    pub(crate) fn check_pending(&self, trials: &[Trial]) -> Result<(), String> {
        for (index, trial) in trials.iter().enumerate() {
            if trials[..index].contains(trial) {
                return Err("batch contains a duplicate trial".to_string());
            }
            self.pending_for(*trial)?;
        }
        Ok(())
    }

    fn ask_sparse_open(
        &mut self,
        seeds: &[u64],
        num_pert: usize,
        config: Ask,
    ) -> Result<Trial, String> {
        check_ask(seeds, self.history.len(), config)?;
        let edits = sparse::make_edits(seeds, &self.leaves, num_pert)?;
        let slot = self.free_slot().ok_or("no free model slot")?;
        let history = self
            .history
            .iter()
            .map(|record| (record.slot, record.value))
            .collect::<Vec<_>>();
        let (index, score) = self.engine.ask_sparse(
            self.base,
            &history,
            slot,
            seeds,
            &edits,
            num_pert,
            &self.leaves,
            config,
        )?;
        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1);
        self.pending.push(Pending {
            id,
            slot,
            seed: seeds[index],
            length: config.length,
            materialized: true,
        });
        Ok(Trial {
            id,
            index,
            seed: seeds[index],
            score,
        })
    }

    fn ask_with_materialization(
        &mut self,
        seeds: &[u64],
        config: Ask,
        materialize_row: bool,
    ) -> Result<Trial, String> {
        let span = crate::tracy::zone(tracy_client::span_location!("trials.ask"));
        span.emit_value(seeds.len() as u64);
        if !self.pending.is_empty() {
            return Err("tell must finish the pending trial before ask".to_string());
        }
        check_ask(seeds, self.history.len(), config)?;
        let slot = self.free_slot().ok_or("no free model slot")?;
        let history: Vec<(usize, f32)> = self
            .history
            .iter()
            .map(|record| (record.slot, record.value))
            .collect();
        let (index, score) = self.engine.ask(
            self.base,
            &history,
            slot,
            seeds,
            &self.leaves,
            config,
            materialize_row,
        )?;
        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1);
        self.pending.push(Pending {
            id,
            slot,
            seed: seeds[index],
            length: config.length,
            materialized: materialize_row,
        });
        Ok(Trial {
            id,
            index,
            seed: seeds[index],
            score,
        })
    }

    /// Execute multi-region trial candidate evaluation on GPU.
    pub fn ask_multi_tr(
        &mut self,
        num_regions: usize,
        seeds_per_region: usize,
        seeds: &[u64],
        config: Ask,
    ) -> Result<Vec<(usize, f32)>, String> {
        let span = crate::tracy::zone(tracy_client::span_location!("trials.multi"));
        span.emit_value(seeds.len() as u64);
        if !self.pending.is_empty() {
            return Err("tell must finish the pending trial before ask".to_string());
        }
        if num_regions == 0 || seeds_per_region == 0 {
            return Err("multi-TR search requires non-zero regions and candidates".to_string());
        }
        let expected = num_regions
            .checked_mul(seeds_per_region)
            .ok_or("multi-TR candidate count overflow")?;
        if seeds.len() != expected {
            return Err(format!(
                "expected {expected} seeds for {num_regions} regions, got {}",
                seeds.len()
            ));
        }
        check_ask(seeds, self.history.len(), config)?;
        let history: Vec<(usize, f32)> = self
            .history
            .iter()
            .map(|record| (record.slot, record.value))
            .collect();
        self.engine.ask_multi(
            self.base,
            &history,
            num_regions,
            seeds_per_region,
            seeds,
            &self.leaves,
            config,
        )
    }

    /// Evaluate regions represented by compact perturbation chains.
    #[allow(clippy::too_many_arguments)]
    pub fn ask_multi_tr_tree(
        &mut self,
        num_regions: usize,
        seeds_per_region: usize,
        centers: &[Center],
        region_centers: &[usize],
        seeds: &[u64],
        config: Ask,
    ) -> Result<Vec<(usize, f32)>, String> {
        let span = crate::tracy::zone(tracy_client::span_location!("trials.tree"));
        span.emit_value(seeds.len() as u64);
        tree::check(centers, region_centers, num_regions)?;
        if !self.pending.is_empty() {
            return Err("tell must finish the pending trial before ask".to_string());
        }
        let expected = num_regions
            .checked_mul(seeds_per_region)
            .ok_or("multi-TR candidate count overflow")?;
        if seeds_per_region == 0 || seeds.len() != expected {
            return Err(format!(
                "expected {expected} seeds for {num_regions} regions, got {}",
                seeds.len()
            ));
        }
        check_ask(seeds, self.history.len(), config)?;
        let history: Vec<(usize, f32)> = self
            .history
            .iter()
            .map(|record| (record.slot, record.value))
            .collect();
        self.engine.ask_tree(
            self.base,
            &history,
            seeds_per_region,
            centers,
            region_centers,
            seeds,
            &self.leaves,
            config,
        )
    }

    /// Use BPANN to shortlist compact candidate descriptors, stream-resolve
    /// the matching full observations, and run the existing exact scorer.
    ///
    /// BPANN affects only shortlist retrieval. Candidate generation, exact
    /// squared distance, ENN prediction, and acquisition remain unchanged.
    pub fn ask_indexed<F>(
        &mut self,
        history: &BpannHistory,
        candidate_descriptors: &ArrayView2<'_, f64>,
        neighbors_per_candidate: usize,
        seeds: &[u64],
        config: Ask,
        resolve: F,
    ) -> Result<Trial, String>
    where
        F: FnMut(ObservationId) -> Result<Vec<u8>, String>,
    {
        if candidate_descriptors.nrows() != seeds.len() {
            return Err(format!(
                "candidate descriptor rows {} do not match seed count {}",
                candidate_descriptors.nrows(),
                seeds.len()
            ));
        }
        let shortlist = history.shortlist(
            candidate_descriptors,
            neighbors_per_candidate,
            self.capacity,
        )?;
        if shortlist.is_empty() {
            return Err("BPANN history returned an empty shortlist".to_string());
        }
        self.replace_indexed_history(&shortlist, resolve)?;
        self.ask(seeds, config)
    }

    pub fn row(&self, trial: Trial) -> Result<Vec<u8>, String> {
        let pending = self.pending_for(trial)?;
        if !pending.materialized {
            return Err(
                "lazy trial row is not materialized; call materialize_pending first".to_string(),
            );
        }
        self.engine.read(pending.slot, self.row_bytes)
    }

    /// Borrow the pending packed row through its CUDA device address.
    ///
    /// The returned address is owned by this search and becomes invalid when
    /// the search is dropped. The CUDA stream is synchronized before return.
    #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
    pub fn device_row(&self, trial: Trial) -> Result<(u64, usize, usize), String> {
        let pending = self.pending_for(trial)?;
        if !pending.materialized {
            return Err("lazy trial row must be materialized before CUDA export".to_string());
        }
        match &self.engine {
            Engine::Cuda(engine) => engine.device_row(pending.slot),
            _ => Err("pending row is not stored on CUDA".to_string()),
        }
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
    pub fn device_batch(&self, trials: &[Trial]) -> Result<Vec<(u64, usize, usize)>, String> {
        let mut slots = Vec::with_capacity(trials.len());
        for trial in trials {
            let pending = self.pending_for(*trial)?;
            if !pending.materialized {
                return Err("lazy trial row must be materialized before CUDA export".to_string());
            }
            slots.push(pending.slot);
        }
        match &self.engine {
            Engine::Cuda(engine) => engine.device_rows(&slots),
            _ => Err("pending rows are not stored on CUDA".to_string()),
        }
    }

    /// Materialize a lazily selected trial into its resident row slot.
    ///
    /// Calling this for a trial returned by [`Search::ask`] is a no-op. Lazy
    /// trials are also materialized automatically by [`Search::tell`] before
    /// they are added to history.
    pub fn materialize_pending(&mut self, trial: Trial) -> Result<(), String> {
        let pending = self.pending_for(trial)?;
        if pending.materialized {
            return Ok(());
        }
        self.engine.materialize(
            self.base,
            pending.slot,
            pending.seed,
            &self.leaves,
            pending.length,
        )?;
        self.pending
            .iter_mut()
            .find(|candidate| candidate.id == trial.id)
            .expect("the pending trial was validated above")
            .materialized = true;
        Ok(())
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(crate) fn pending_metal_row(
        &self,
        trial: Trial,
    ) -> Result<(::metal::Buffer, usize), String> {
        let pending = self.pending_for(trial)?;
        if !pending.materialized {
            return Err(
                "lazy trial row is not materialized; call materialize_pending first".to_string(),
            );
        }
        match &self.engine {
            Engine::Metal(engine) => engine.row_buffer(pending.slot),
            _ => Err("the resident search is not using Metal".to_string()),
        }
    }

    pub fn tell(&mut self, trial: Trial, value: f32, accept: bool) -> Result<(), String> {
        let _span = crate::tracy::zone(tracy_client::span_location!("trials.tell"));
        if !value.is_finite() {
            return Err("trial value must be finite".to_string());
        }
        self.materialize_pending(trial)?;
        let pending = self.pending_for(trial)?;
        if self.history.len() == self.capacity {
            self.history.pop_front();
        }
        self.history.push_back(Record {
            slot: pending.slot,
            value,
        });
        if accept {
            self.base = pending.slot;
        }
        self.pending.retain(|candidate| candidate.id != trial.id);
        Ok(())
    }

    pub fn history_len(&self) -> usize {
        self.history.len()
    }

    pub fn history_capacity(&self) -> usize {
        self.capacity
    }

    /// Begin a new trust-region generation around the current incumbent.
    pub(crate) fn restart(&mut self, value: f32) -> Result<(), String> {
        if !self.pending.is_empty() {
            return Err("tell must finish the pending trial before restart".to_string());
        }
        self.history.clear();
        self.history.push_back(Record {
            slot: self.base,
            value,
        });
        Ok(())
    }

    pub fn row_bytes(&self) -> usize {
        self.row_bytes
    }

    /// Replace the resident ENN history with a shortlist resolved by an
    /// external index such as [`BpannHistory`].
    ///
    /// `rows` is packed row-major using this search's quantized row layout.
    /// The shortlist is allowed to contain at most `history_capacity()` rows;
    /// one additional device slot remains free for the next generated trial.
    pub fn replace_history(&mut self, rows: &[u8], values: &[f32]) -> Result<(), String> {
        if !self.pending.is_empty() {
            return Err("tell must finish the pending trial before replacing history".to_string());
        }
        if values.is_empty() {
            return Err("replacement history requires at least one observation".to_string());
        }
        if values.len() > self.capacity {
            return Err(format!(
                "replacement history has {} rows, capacity is {}",
                values.len(),
                self.capacity
            ));
        }
        let expected = values
            .len()
            .checked_mul(self.row_bytes)
            .ok_or("replacement history byte count overflow")?;
        if rows.len() != expected {
            return Err(format!(
                "replacement history has {} bytes, expected {expected}",
                rows.len()
            ));
        }

        let slots: Vec<usize> = (0..self.slots)
            .filter(|slot| *slot != self.base)
            .take(values.len())
            .collect();
        if slots.len() != values.len() {
            return Err("not enough free model slots for replacement history".to_string());
        }
        for (row_index, &slot) in slots.iter().enumerate() {
            let start = row_index * self.row_bytes;
            self.engine
                .write(slot, &rows[start..start + self.row_bytes])?;
        }
        self.history = slots
            .into_iter()
            .zip(values.iter().copied())
            .map(|(slot, value)| Record { slot, value })
            .collect();
        Ok(())
    }

    /// Resolve a BPANN shortlist one observation at a time and load it into the
    /// exact scorer without building a `neighbors × row_bytes` host matrix.
    ///
    /// The resolver may regenerate a row from a seed/checkpoint archive. Rows
    /// are released after being copied into their device slots.
    pub fn replace_indexed_history<F>(
        &mut self,
        observations: &[IndexedObservation],
        mut resolve: F,
    ) -> Result<(), String>
    where
        F: FnMut(ObservationId) -> Result<Vec<u8>, String>,
    {
        if !self.pending.is_empty() {
            return Err("tell must finish the pending trial before replacing history".to_string());
        }
        if observations.is_empty() {
            return Err("replacement history requires at least one observation".to_string());
        }
        if observations.len() > self.capacity {
            return Err(format!(
                "replacement history has {} rows, capacity is {}",
                observations.len(),
                self.capacity
            ));
        }
        let slots: Vec<usize> = (0..self.slots)
            .filter(|slot| *slot != self.base)
            .take(observations.len())
            .collect();
        if slots.len() != observations.len() {
            return Err("not enough free model slots for replacement history".to_string());
        }

        self.history.clear();
        for (&slot, observation) in slots.iter().zip(observations) {
            let row = resolve(observation.id)?;
            if row.len() != self.row_bytes {
                return Err(format!(
                    "resolved observation {} has {} bytes, expected {}",
                    observation.id.0,
                    row.len(),
                    self.row_bytes
                ));
            }
            self.engine.write(slot, &row)?;
            self.history.push_back(Record {
                slot,
                value: observation.value,
            });
        }
        Ok(())
    }

    fn pending_for(&self, trial: Trial) -> Result<Pending, String> {
        self.pending
            .iter()
            .copied()
            .find(|pending| pending.id == trial.id)
            .ok_or_else(|| "trial does not match an outstanding ask".to_string())
    }

    fn free_slot(&self) -> Option<usize> {
        (0..self.slots).find(|slot| {
            *slot != self.base
                && self.history.iter().all(|record| record.slot != *slot)
                && self.pending.iter().all(|pending| pending.slot != *slot)
        })
    }
}

mod cpu;
use cpu::{check_ask, hash, materialize, perturb, score, trial_distance, Cpu};
#[cfg(test)]
#[path = "trials/tests.rs"]
mod tests;
