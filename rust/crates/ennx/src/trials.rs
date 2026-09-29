use ndarray::ArrayView2;
use std::collections::VecDeque;

use crate::weights::{AcquisitionKind, ComputeDevice};

#[cfg(all(target_os = "macos", feature = "metal"))]
mod metal;

#[cfg(feature = "opencl")]
mod opencl;

#[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
mod cuda;

mod bindings;
mod bpann_history;
mod engine;
mod identity;
mod views;
pub use identity::Trial;
use identity::trial_id;
mod layout;
mod resident;
mod stream;
mod tree;

mod sparse;

pub use bpann_history::{BpannHistory, IndexedObservation, ObservationId};
pub(crate) use layout::{LeafStep, check_layout, make_steps};
#[cfg(any(
    all(feature = "cuda", target_os = "linux", target_arch = "x86_64"),
    all(feature = "metal", target_os = "macos"),
    feature = "opencl"
))]
pub(crate) use layout::{Tile, make_tiles};
#[cfg(feature = "opencl")]
pub(crate) use opencl::ResidentRow as OpenClResidentRow;
pub use resident::{DeviceView, device_views};
pub use tree::Center;

const MAX_HISTORY: usize = 128;
const MAX_PENDING: usize = 32;

mod encoding;
pub use encoding::{EncodingType, Parameter, decode_code};

#[derive(Debug, Clone, Copy)]
pub struct Ask {
    pub length: f32,
    pub neighbors: usize,
    pub epistemic_scale: f32,
    pub aleatoric_scale: f32,
    pub y_scale: f32,
    pub beta: f32,
    pub acquisition: AcquisitionKind,
    /// Sampled-function seed; resident observation slots identify shared Thompson noise.
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

#[repr(C)]
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
    leaves: Vec<Parameter>,
    row_bytes: usize,
    capacity: usize,
    pending_capacity: usize,
    slots: usize,
    base: usize,
    history: VecDeque<Record>,
    pending: Vec<Pending>,
    engine: Engine,
    device_state: bool,
}

impl Search {
    pub fn new(
        base: &[u8],
        base_value: f32,
        leaves: Vec<Parameter>,
        capacity: usize,
        device: ComputeDevice,
    ) -> Result<Self, String> {
        Self::new_batch(base, base_value, leaves, capacity, 1, device)
    }

    pub fn new_batch(
        base: &[u8],
        base_value: f32,
        leaves: Vec<Parameter>,
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
            engine,
            device_state: false,
        })
    }

    pub fn ask(&mut self, seeds: &[u64], config: Ask) -> Result<Trial, String> {
        self.ask_materialization(seeds, config, true)
    }

    /// Select a seed without materializing its full weight row.
    ///
    /// This is the path used when the evaluator regenerates the perturbation
    /// from the returned seed.  Materializing a billion-parameter row during
    /// proposal would make `ask` scale with model size for no benefit.
    pub fn ask_lazy(&mut self, seeds: &[u64], config: Ask) -> Result<Trial, String> {
        self.ask_materialization(seeds, config, false)
    }

    pub fn ask_sparse(
        &mut self,
        seeds: &[u64],
        num_pert: usize,
        config: Ask,
    ) -> Result<Trial, String> {
        if !self.pending.is_empty() {
            return Err("tell must finish the pending trial before ask".to_string());
        }
        self.ask_open(seeds, num_pert, config)
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
            match self.ask_open(group, num_pert, config) {
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

    fn ask_open(&mut self, seeds: &[u64], num_pert: usize, config: Ask) -> Result<Trial, String> {
        check_ask(seeds, self.history_bound(), config)?;
        let edits = if self.device_state {
            let dimensions = self.leaves.iter().map(|leaf| leaf.length).sum::<usize>();
            if num_pert == 0 || num_pert > dimensions {
                return Err(
                    "perturbation count must be positive and no larger than the parameter count"
                        .into(),
                );
            }
            Vec::new()
        } else {
            sparse::make_edits(seeds, &self.leaves, num_pert)?
        };
        let slot = self.free_slot().ok_or("no free model slot")?;
        let history = self.history_view();
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
        let id = trial_id()?;
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

    fn ask_materialization(
        &mut self,
        seeds: &[u64],
        config: Ask,
        materialize_row: bool,
    ) -> Result<Trial, String> {
        if !self.pending.is_empty() {
            return Err("tell must finish the pending trial before ask".to_string());
        }
        check_ask(seeds, self.history_bound(), config)?;
        let slot = self.free_slot().ok_or("no free model slot")?;
        let history: Vec<(usize, f32)> = self.history_view();
        let (index, score) = self.engine.ask(
            self.base,
            &history,
            slot,
            seeds,
            &self.leaves,
            config,
            materialize_row,
        )?;
        let id = trial_id()?;
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
    pub fn ask_regions(
        &mut self,
        num_regions: usize,
        seeds_per_region: usize,
        seeds: &[u64],
        config: Ask,
    ) -> Result<Vec<(usize, f32)>, String> {
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
        check_ask(seeds, self.history_bound(), config)?;
        let history: Vec<(usize, f32)> = self.history_view();
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
    pub fn ask_centers(
        &mut self,
        num_regions: usize,
        seeds_per_region: usize,
        centers: &[Center],
        region_centers: &[usize],
        seeds: &[u64],
        config: Ask,
    ) -> Result<Vec<(usize, f32)>, String> {
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
        check_ask(seeds, self.history_bound(), config)?;
        let history: Vec<(usize, f32)> = self.history_view();
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
    /// Each call replaces resident history, so Thompson draws need not agree
    /// across calls with different candidate-dependent shortlists.
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
        self.indexed_history(&shortlist, resolve)?;
        self.ask(seeds, config)
    }

    pub fn tell(&mut self, trial: Trial, value: f32, accept: bool) -> Result<(), String> {
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

    pub fn device(&self) -> ComputeDevice {
        self.engine.device()
    }

    /// Replace the resident ENN history with a shortlist resolved by an
    /// external index such as [`BpannHistory`].
    ///
    /// `rows` is packed row-major using this search's quantized row layout.
    /// The shortlist is allowed to contain at most `history_capacity()` rows;
    /// one additional device slot remains free for the next generated trial.
    /// Replacement assigns new row identities for Thompson draws; keeping the
    /// function seed does not preserve samples across history replacements.
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
    /// Reloading assigns row identities by resident slot, not `ObservationId`.
    pub fn indexed_history<F>(
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
        if self.device_state {
            return (0..self.pending_capacity)
                .find(|slot| self.pending.iter().all(|item| item.slot != *slot));
        }
        (0..self.slots).find(|slot| {
            *slot != self.base
                && self.history.iter().all(|record| record.slot != *slot)
                && self.pending.iter().all(|pending| pending.slot != *slot)
        })
    }
}

mod cpu;
use cpu::{Cpu, check_ask, check_count, hash, materialize, perturb, score, trial_distance};
#[cfg(test)]
#[path = "trials/tests.rs"]
mod tests;
