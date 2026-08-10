use std::collections::VecDeque;

use ndarray::ArrayView2;

use crate::util::insert_neighbor;
use crate::weights::{AcquisitionKind, ComputeBackend};

#[cfg(all(target_os = "macos", feature = "metal"))]
mod metal;

#[cfg(feature = "opencl")]
mod opencl;

mod bpann_history;
mod layout;
mod tree;

pub use bpann_history::{BpannHistory, IndexedObservation, ObservationId};
pub(crate) use layout::{check_layout, make_steps, make_tiles, Step, Tile};
pub use tree::Center;

const MAX_HISTORY: usize = 128;

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

enum Engine {
    Cpu(Cpu),
    #[cfg(all(target_os = "macos", feature = "metal"))]
    Metal(metal::Engine),
    #[cfg(feature = "opencl")]
    OpenCl(opencl::Engine),
}

pub struct Search {
    leaves: Vec<Leaf>,
    row_bytes: usize,
    capacity: usize,
    slots: usize,
    base: usize,
    history: VecDeque<Record>,
    pending: Option<Pending>,
    next_id: u64,
    engine: Engine,
}

impl Search {
    pub fn new(
        base: &[u8],
        base_value: f32,
        leaves: Vec<Leaf>,
        capacity: usize,
        backend: ComputeBackend,
    ) -> Result<Self, String> {
        if !base_value.is_finite() {
            return Err("base value must be finite".to_string());
        }
        if capacity == 0 || capacity > MAX_HISTORY {
            return Err(format!("history capacity must be in 1..={MAX_HISTORY}"));
        }
        let row_bytes = check_layout(&leaves)?;
        if base.len() != row_bytes {
            return Err(format!(
                "base row has {} bytes, expected {row_bytes}",
                base.len()
            ));
        }
        let slots = capacity + 2;
        let engine = Engine::new(base, &leaves, slots, backend)?;
        Ok(Self {
            leaves,
            row_bytes,
            capacity,
            slots,
            base: 0,
            history: VecDeque::from([Record {
                slot: 0,
                value: base_value,
            }]),
            pending: None,
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

    fn ask_with_materialization(
        &mut self,
        seeds: &[u64],
        config: Ask,
        materialize_row: bool,
    ) -> Result<Trial, String> {
        let span = crate::tracy::zone(tracy_client::span_location!("trials.ask"));
        span.emit_value(seeds.len() as u64);
        if self.pending.is_some() {
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
        self.pending = Some(Pending {
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
        if self.pending.is_some() {
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
        match &mut self.engine {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Engine::Metal(engine) => engine.ask_multi_tr(
                self.base,
                &history,
                num_regions,
                seeds_per_region,
                seeds,
                &self.leaves,
                config,
            ),
            _ => {
                let mut results = Vec::with_capacity(num_regions);
                for r in 0..num_regions {
                    let start = r * seeds_per_region;
                    let end = start + seeds_per_region;
                    let (index, score) = self.engine.ask(
                        self.base,
                        &history,
                        0,
                        &seeds[start..end],
                        &self.leaves,
                        config,
                        true,
                    )?;
                    results.push((start + index, score));
                }
                Ok(results)
            }
        }
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
        if self.pending.is_some() {
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
        match &mut self.engine {
            Engine::Cpu(engine) => engine.ask_multi_tr_tree(
                self.base,
                &history,
                seeds_per_region,
                centers,
                region_centers,
                seeds,
                &self.leaves,
                config,
            ),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Engine::Metal(engine) => engine.ask_multi_tr_tree(
                self.base,
                &history,
                seeds_per_region,
                centers,
                region_centers,
                seeds,
                &self.leaves,
                config,
            ),
            #[cfg(feature = "opencl")]
            Engine::OpenCl(engine) => engine.ask_multi_tr_tree(
                self.base,
                &history,
                seeds_per_region,
                centers,
                region_centers,
                seeds,
                &self.leaves,
                config,
            ),
        }
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
            .as_mut()
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
        self.pending = None;
        Ok(())
    }

    pub fn history_len(&self) -> usize {
        self.history.len()
    }

    pub fn history_capacity(&self) -> usize {
        self.capacity
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
        if self.pending.is_some() {
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
        if self.pending.is_some() {
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
        match self.pending {
            Some(pending) if pending.id == trial.id => Ok(pending),
            Some(_) => Err("trial does not match the pending ask".to_string()),
            None => Err("there is no pending trial".to_string()),
        }
    }

    fn free_slot(&self) -> Option<usize> {
        (0..self.slots).find(|slot| {
            *slot != self.base
                && self.history.iter().all(|record| record.slot != *slot)
                && self.pending.map(|pending| pending.slot) != Some(*slot)
        })
    }
}

impl Engine {
    #[allow(unused_variables)]
    fn new(
        base: &[u8],
        leaves: &[Leaf],
        slots: usize,
        backend: ComputeBackend,
    ) -> Result<Self, String> {
        match backend {
            ComputeBackend::Cpu => Ok(Self::Cpu(Cpu::new(base, slots))),
            ComputeBackend::Metal => {
                #[cfg(all(target_os = "macos", feature = "metal"))]
                {
                    Ok(Self::Metal(metal::Engine::new(base, leaves, slots)?))
                }
                #[cfg(not(all(target_os = "macos", feature = "metal")))]
                {
                    Err("Metal trial search is not available in this build".to_string())
                }
            }
            ComputeBackend::Agx => {
                #[cfg(all(target_os = "macos", feature = "metal"))]
                {
                    Ok(Self::Metal(metal::Engine::new_agx(base, leaves, slots)?))
                }
                #[cfg(not(all(target_os = "macos", feature = "metal")))]
                {
                    Err("AGX trial search is not available in this build".to_string())
                }
            }
            ComputeBackend::OpenCl => {
                #[cfg(feature = "opencl")]
                {
                    Ok(Self::OpenCl(opencl::Engine::new(base, leaves, slots)?))
                }
                #[cfg(not(feature = "opencl"))]
                {
                    Err("OpenCL trial search is not available in this build".to_string())
                }
            }
            ComputeBackend::Auto => {
                #[cfg(all(target_os = "macos", feature = "metal"))]
                {
                    return Ok(Self::Metal(
                        metal::Engine::new_agx(base, leaves, slots)
                            .or_else(|_| metal::Engine::new(base, leaves, slots))?,
                    ));
                }
                #[cfg(all(feature = "opencl", not(all(target_os = "macos", feature = "metal"))))]
                {
                    return Ok(Self::OpenCl(opencl::Engine::new(base, leaves, slots)?));
                }
                #[allow(unreachable_code)]
                Ok(Self::Cpu(Cpu::new(base, slots)))
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn ask(
        &mut self,
        base: usize,
        history: &[(usize, f32)],
        trial: usize,
        seeds: &[u64],
        leaves: &[Leaf],
        config: Ask,
        materialize_row: bool,
    ) -> Result<(usize, f32), String> {
        match self {
            Self::Cpu(engine) => {
                engine.ask(base, history, trial, seeds, leaves, config, materialize_row)
            }
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Self::Metal(engine) => {
                engine.ask(base, history, trial, seeds, leaves, config, materialize_row)
            }
            #[cfg(feature = "opencl")]
            Self::OpenCl(engine) => {
                engine.ask(base, history, trial, seeds, leaves, config, materialize_row)
            }
        }
    }

    #[allow(unused_variables)]
    fn read(&self, slot: usize, row_bytes: usize) -> Result<Vec<u8>, String> {
        match self {
            Self::Cpu(engine) => Ok(engine.read(slot).to_vec()),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Self::Metal(engine) => Ok(engine.read(slot, row_bytes)),
            #[cfg(feature = "opencl")]
            Self::OpenCl(engine) => engine.read(slot, row_bytes),
        }
    }

    #[allow(unused_variables)]
    fn write(&mut self, slot: usize, row: &[u8]) -> Result<(), String> {
        match self {
            Self::Cpu(engine) => {
                engine.read_mut(slot).copy_from_slice(row);
                Ok(())
            }
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Self::Metal(engine) => {
                engine.write(slot, row);
                Ok(())
            }
            #[cfg(feature = "opencl")]
            Self::OpenCl(engine) => engine.write(slot, row),
        }
    }

    #[allow(unused_variables)]
    fn materialize(
        &mut self,
        base_slot: usize,
        trial_slot: usize,
        seed: u64,
        leaves: &[Leaf],
        length: f32,
    ) -> Result<(), String> {
        let steps = make_steps(leaves, length);
        match self {
            Self::Cpu(engine) => {
                let base = engine.read(base_slot).to_vec();
                let row = materialize(&base, leaves, &steps, seed);
                engine.read_mut(trial_slot).copy_from_slice(&row);
                Ok(())
            }
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Self::Metal(engine) => engine.materialize(base_slot, trial_slot, seed, &steps),
            #[cfg(feature = "opencl")]
            Self::OpenCl(engine) => engine.materialize(base_slot, trial_slot, seed, &steps),
        }
    }
}

struct Cpu {
    rows: Vec<u8>,
    row_bytes: usize,
}

impl Cpu {
    fn new(base: &[u8], slots: usize) -> Self {
        let mut rows = vec![0; slots * base.len()];
        rows[..base.len()].copy_from_slice(base);
        Self {
            rows,
            row_bytes: base.len(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn ask(
        &mut self,
        base_slot: usize,
        history: &[(usize, f32)],
        trial_slot: usize,
        seeds: &[u64],
        leaves: &[Leaf],
        config: Ask,
        materialize_row: bool,
    ) -> Result<(usize, f32), String> {
        let steps = make_steps(leaves, config.length);
        let base = self.read(base_slot).to_vec();
        let draws = crate::weights::thompson_draws(seeds.len(), config.seed);
        let mut best_index = 0;
        let mut best_score = f32::NEG_INFINITY;
        let mut nearest = vec![(f32::INFINITY, 0usize); config.neighbors];
        for (index, &seed) in seeds.iter().enumerate() {
            nearest.fill((f32::INFINITY, 0usize));
            for (observation_index, &(slot, _)) in history.iter().enumerate() {
                let distance = trial_distance(&base, self.read(slot), leaves, &steps, seed);
                insert_neighbor(&mut nearest, distance, observation_index);
            }
            let score = score(&nearest, history, draws[index], config);
            if score > best_score || (score == best_score && index < best_index) {
                best_index = index;
                best_score = score;
            }
        }

        if materialize_row {
            let row = materialize(&base, leaves, &steps, seeds[best_index]);
            self.read_mut(trial_slot).copy_from_slice(&row);
        }
        Ok((best_index, best_score))
    }

    #[allow(clippy::too_many_arguments)]
    fn ask_multi_tr_tree(
        &self,
        base_slot: usize,
        history: &[(usize, f32)],
        seeds_per_region: usize,
        centers: &[Center],
        region_centers: &[usize],
        seeds: &[u64],
        leaves: &[Leaf],
        config: Ask,
    ) -> Result<Vec<(usize, f32)>, String> {
        tree::cpu_ask(
            self.read(base_slot),
            &self.rows,
            self.row_bytes,
            history,
            seeds_per_region,
            centers,
            region_centers,
            seeds,
            leaves,
            config,
        )
    }

    fn read(&self, slot: usize) -> &[u8] {
        &self.rows[slot * self.row_bytes..(slot + 1) * self.row_bytes]
    }

    fn read_mut(&mut self, slot: usize) -> &mut [u8] {
        &mut self.rows[slot * self.row_bytes..(slot + 1) * self.row_bytes]
    }
}

fn check_ask(seeds: &[u64], observations: usize, config: Ask) -> Result<(), String> {
    if seeds.is_empty() {
        return Err("ask requires at least one seed".to_string());
    }
    if config.neighbors == 0 || config.neighbors > observations {
        return Err(format!(
            "neighbor count must be between one and {observations}"
        ));
    }
    for (name, value) in [
        ("length", config.length),
        ("epistemic_scale", config.epistemic_scale),
        ("aleatoric_scale", config.aleatoric_scale),
        ("y_scale", config.y_scale),
        ("beta", config.beta),
    ] {
        if !value.is_finite() || value < 0.0 {
            return Err(format!("{name} must be finite and nonnegative"));
        }
    }
    Ok(())
}

pub(crate) fn perturb(code: u32, seed: u64, element: u32, step: Step) -> u32 {
    let random = hash(seed, element);
    let sign = random & 1;
    let extra = u32::from((random >> 1) < (step.threshold >> 1));
    let amount = step.whole + extra;
    if amount == 0 {
        return code;
    }
    let max_code = (1u32 << step.bits) - 1;
    if sign == 0 {
        if code >= amount {
            code - amount
        } else {
            (code + amount).min(max_code)
        }
    } else if code + amount <= max_code {
        code + amount
    } else {
        code.saturating_sub(amount)
    }
}

fn hash(seed: u64, element: u32) -> u32 {
    let mut value = (seed as u32) ^ element.wrapping_mul(0x9e37_79b9);
    value ^= value >> 16;
    value = value.wrapping_mul(0x7feb_352d);
    value ^= (seed >> 32) as u32;
    value = value.wrapping_mul(0x846c_a68b);
    value ^ (value >> 15)
}

fn materialize(base: &[u8], leaves: &[Leaf], steps: &[Step], seed: u64) -> Vec<u8> {
    let mut row = vec![0u8; base.len()];
    for (&leaf, &step) in leaves.iter().zip(steps) {
        match leaf.bits {
            4 => {
                for element in 0..leaf.length {
                    let byte = step.byte_offset as usize + element / 2;
                    let shift = (element & 1) * 4;
                    let code = u32::from((base[byte] >> shift) & 0x0f);
                    let value = perturb(code, seed, leaf.offset as u32 + element as u32, step);
                    row[byte] |= (value as u8) << shift;
                }
            }
            8 => {
                for element in 0..leaf.length {
                    let byte = step.byte_offset as usize + element;
                    let code = u32::from(base[byte]);
                    row[byte] =
                        perturb(code, seed, leaf.offset as u32 + element as u32, step) as u8;
                }
            }
            _ => unreachable!("leaf width is checked at construction"),
        }
    }
    row
}

fn trial_distance(
    base: &[u8],
    observation: &[u8],
    leaves: &[Leaf],
    steps: &[Step],
    seed: u64,
) -> f32 {
    let mut distance = 0.0f32;
    for (&leaf, &step) in leaves.iter().zip(steps) {
        let byte_offset = step.byte_offset as usize;
        let element_offset = leaf.offset as u32;
        if leaf.bits == 4 {
            for element in 0..leaf.length {
                let byte = byte_offset + element / 2;
                let shift = (element & 1) * 4;
                let code = u32::from((base[byte] >> shift) & 0x0f);
                let candidate_code = perturb(code, seed, element_offset + element as u32, step);
                let observed_code = u32::from((observation[byte] >> shift) & 0x0f);
                let candidate_val = decode_code(candidate_code, leaf.encoding, leaf.scale);
                let observed_val = decode_code(observed_code, leaf.encoding, leaf.scale);
                let delta = candidate_val - observed_val;
                distance = delta.mul_add(delta * leaf.weight, distance);
            }
        } else {
            for element in 0..leaf.length {
                let byte = byte_offset + element;
                let code = u32::from(base[byte]);
                let candidate_code = perturb(code, seed, element_offset + element as u32, step);
                let observed_code = u32::from(observation[byte]);
                let candidate_val = decode_code(candidate_code, leaf.encoding, leaf.scale);
                let observed_val = decode_code(observed_code, leaf.encoding, leaf.scale);
                let delta = candidate_val - observed_val;
                distance = delta.mul_add(delta * leaf.weight, distance);
            }
        }
    }
    distance
}

fn score(nearest: &[(f32, usize)], history: &[(usize, f32)], draw: f32, config: Ask) -> f32 {
    let mut weight_sum = 0.0;
    let mut weighted_value = 0.0;
    for &(distance, index) in nearest {
        let variance = 1.0e-9 + config.epistemic_scale * distance + config.aleatoric_scale;
        let weight = 1.0 / variance.max(1.0e-12);
        weight_sum += weight;
        weighted_value += weight * history[index].1;
    }
    let mean = weighted_value / weight_sum.max(1.0e-12);
    let se = (1.0 / weight_sum.max(1.0e-12)).sqrt() * config.y_scale;
    match config.acquisition {
        AcquisitionKind::Ucb => mean + config.beta * se,
        AcquisitionKind::Thompson => mean + se * draw,
        AcquisitionKind::Pareto => mean + se,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Axis};
    use tempfile::TempDir;

    fn leaves() -> Vec<Leaf> {
        vec![
            Leaf::new(0, 5, 4, 0.25, 1.0, 0.75).unwrap(),
            Leaf::new(5, 4, 8, 0.5, 0.5, 1.0).unwrap(),
        ]
    }

    #[test]
    fn cpu_search_is_deterministic_and_updates_every_leaf() {
        let base = [0x76, 0x98, 0x0a, 100, 120, 140, 160];
        let mut left = Search::new(&base, 1.0, leaves(), 4, ComputeBackend::Cpu).unwrap();
        let mut right = Search::new(&base, 1.0, leaves(), 4, ComputeBackend::Cpu).unwrap();
        let config = Ask {
            neighbors: 1,
            length: 1.0,
            ..Ask::default()
        };
        let a = left.ask(&[7, 11, 13], config).unwrap();
        let b = right.ask(&[7, 11, 13], config).unwrap();
        assert_eq!(a, b);
        let row = left.row(a).unwrap();
        assert_eq!(row, right.row(b).unwrap());
        assert_ne!(&row[..3], &base[..3]);
        assert_ne!(&row[3..], &base[3..]);
    }

    #[test]
    fn accepted_trial_becomes_the_next_center() {
        let base = [0x76, 0x98, 0x0a, 100, 120, 140, 160];
        let mut search = Search::new(&base, 0.0, leaves(), 2, ComputeBackend::Cpu).unwrap();
        let config = Ask {
            neighbors: 1,
            length: 1.0,
            ..Ask::default()
        };
        let first = search.ask(&[5], config).unwrap();
        let first_row = search.row(first).unwrap();
        search.tell(first, 1.0, true).unwrap();
        let second = search.ask(&[9], config).unwrap();
        let second_row = search.row(second).unwrap();
        assert_ne!(first_row, second_row);
        assert_eq!(search.history_len(), 2);
    }

    fn assert_lazy_trial_matches_eager(backend: ComputeBackend) {
        let base = [0x76, 0x98, 0x0a, 100, 120, 140, 160];
        let mut eager = Search::new(&base, 0.0, leaves(), 3, backend).unwrap();
        let mut lazy = Search::new(&base, 0.0, leaves(), 3, backend).unwrap();
        let config = Ask {
            neighbors: 1,
            length: 0.65,
            ..Ask::default()
        };

        let eager_trial = eager.ask(&[5, 7, 11], config).unwrap();
        let lazy_trial = lazy.ask_lazy(&[5, 7, 11], config).unwrap();
        assert_eq!(lazy_trial, eager_trial);
        assert!(lazy.row(lazy_trial).is_err());

        eager.tell(eager_trial, 1.0, true).unwrap();
        lazy.tell(lazy_trial, 1.0, true).unwrap();

        let eager_next = eager.ask(&[13, 17], config).unwrap();
        let lazy_next = lazy.ask(&[13, 17], config).unwrap();
        assert_eq!(lazy_next, eager_next);
        assert_eq!(lazy.row(lazy_next).unwrap(), eager.row(eager_next).unwrap());
    }

    #[test]
    fn lazy_trial_is_materialized_before_it_enters_history() {
        assert_lazy_trial_matches_eager(ComputeBackend::Cpu);
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn metal_lazy_trial_is_materialized_before_it_enters_history() {
        assert_lazy_trial_matches_eager(ComputeBackend::Metal);
        assert_lazy_trial_matches_eager(ComputeBackend::Agx);
    }

    #[cfg(feature = "opencl")]
    #[test]
    fn opencl_lazy_trial_is_materialized_before_it_enters_history() {
        let base = [0x76, 0x98, 0x0a, 100, 120, 140, 160];
        match Search::new(&base, 0.0, leaves(), 3, ComputeBackend::OpenCl) {
            Ok(_) => assert_lazy_trial_matches_eager(ComputeBackend::OpenCl),
            Err(error) if error.contains("no OpenCL GPU or CPU device") => {}
            Err(error) => panic!("{error}"),
        }
    }

    #[test]
    fn rejected_trial_does_not_replace_the_center() {
        let base = [0x76, 0x98, 0x0a, 100, 120, 140, 160];
        let mut search = Search::new(&base, 0.0, leaves(), 2, ComputeBackend::Cpu).unwrap();
        let mut control = Search::new(&base, 0.0, leaves(), 2, ComputeBackend::Cpu).unwrap();
        let config = Ask {
            neighbors: 1,
            length: 1.0,
            ..Ask::default()
        };
        let rejected = search.ask(&[5], config).unwrap();
        search.tell(rejected, -1.0, false).unwrap();
        let next = search.ask(&[5], config).unwrap();
        let expected = control.ask(&[5], config).unwrap();
        assert_eq!(search.row(next).unwrap(), control.row(expected).unwrap());
    }

    #[test]
    fn replacement_history_drives_exact_scoring_and_preserves_base() {
        let base = [0x76, 0x98, 0x0a, 100, 120, 140, 160];
        let mut search = Search::new(&base, 0.0, leaves(), 3, ComputeBackend::Cpu).unwrap();
        let rows = [
            0x11, 0x22, 0x03, 10, 20, 30, 40, 0x44, 0x55, 0x06, 70, 80, 90, 100,
        ];
        search.replace_history(&rows, &[3.0, 7.0]).unwrap();
        assert_eq!(search.history_len(), 2);
        assert_eq!(search.history_capacity(), 3);

        let trial = search
            .ask(
                &[17, 23],
                Ask {
                    neighbors: 1,
                    length: 1.0,
                    ..Ask::default()
                },
            )
            .unwrap();
        assert_eq!(search.row(trial).unwrap().len(), base.len());
        search.tell(trial, 9.0, false).unwrap();

        let next = search
            .ask(
                &[17],
                Ask {
                    neighbors: 1,
                    length: 1.0,
                    ..Ask::default()
                },
            )
            .unwrap();
        let mut control = Search::new(&base, 0.0, leaves(), 3, ComputeBackend::Cpu).unwrap();
        control.replace_history(&rows, &[3.0, 7.0]).unwrap();
        let expected = control
            .ask(
                &[17],
                Ask {
                    neighbors: 1,
                    length: 1.0,
                    ..Ask::default()
                },
            )
            .unwrap();
        assert_eq!(search.row(next).unwrap(), control.row(expected).unwrap());
    }

    #[test]
    fn replacement_history_validates_shape_and_pending_state() {
        let base = [0x76, 0x98, 0x0a, 100, 120, 140, 160];
        let mut search = Search::new(&base, 0.0, leaves(), 2, ComputeBackend::Cpu).unwrap();
        assert!(search.replace_history(&[], &[]).is_err());
        assert!(search.replace_history(&base, &[1.0, 2.0]).is_err());
        let trial = search
            .ask(
                &[7],
                Ask {
                    neighbors: 1,
                    ..Ask::default()
                },
            )
            .unwrap();
        assert!(search.replace_history(&base, &[1.0]).is_err());
        search.tell(trial, 1.0, false).unwrap();
    }

    #[test]
    fn indexed_history_resolves_one_row_at_a_time() {
        let base = [0x76, 0x98, 0x0a, 100, 120, 140, 160];
        let rows = [
            [0x11, 0x22, 0x03, 10, 20, 30, 40],
            [0x44, 0x55, 0x06, 70, 80, 90, 100],
        ];
        let observations = [
            IndexedObservation {
                id: ObservationId(1),
                value: 3.0,
            },
            IndexedObservation {
                id: ObservationId(0),
                value: 7.0,
            },
        ];
        let mut resolved = Vec::new();
        let mut search = Search::new(&base, 0.0, leaves(), 2, ComputeBackend::Cpu).unwrap();
        search
            .replace_indexed_history(&observations, |id| {
                resolved.push(id);
                Ok(rows[id.0 as usize].to_vec())
            })
            .unwrap();
        assert_eq!(resolved, vec![ObservationId(1), ObservationId(0)]);
        assert_eq!(search.history_len(), 2);
        assert!(search
            .ask(
                &[31],
                Ask {
                    neighbors: 2,
                    ..Ask::default()
                }
            )
            .is_ok());
    }

    #[test]
    fn indexed_ask_connects_bpann_shortlist_to_exact_search() {
        let base = [0x76, 0x98, 0x0a, 100, 120, 140, 160];
        let archive = [
            [0x11, 0x22, 0x03, 10, 20, 30, 40],
            [0x44, 0x55, 0x06, 70, 80, 90, 100],
            [0x77, 0x88, 0x09, 110, 120, 130, 140],
        ];
        let descriptors = array![[0.0, 0.0], [1.0, 0.0], [4.0, 0.0]];
        let dir = TempDir::new().unwrap();
        let mut history = BpannHistory::new(dir.path().to_path_buf(), 2).unwrap();
        for (index, descriptor) in descriptors.axis_iter(Axis(0)).enumerate() {
            history
                .append(&descriptor, (index as f32 + 1.0) * 10.0)
                .unwrap();
        }

        let candidate_descriptors = array![[0.1, 0.0], [3.9, 0.0]];
        let mut resolved = Vec::new();
        let mut search = Search::new(&base, 0.0, leaves(), 3, ComputeBackend::Cpu).unwrap();
        let trial = search
            .ask_indexed(
                &history,
                &candidate_descriptors.view(),
                1,
                &[17, 23],
                Ask {
                    neighbors: 1,
                    length: 1.0,
                    ..Ask::default()
                },
                |id| {
                    resolved.push(id);
                    Ok(archive[id.0 as usize].to_vec())
                },
            )
            .unwrap();
        assert_eq!(resolved, vec![ObservationId(0), ObservationId(2)]);
        assert_eq!(search.history_len(), 2);
        assert!(trial.index < 2);
        assert_eq!(search.row(trial).unwrap().len(), base.len());
    }
}
