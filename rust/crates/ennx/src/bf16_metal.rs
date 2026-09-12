//! Dense FP32-compute/BF16-storage correlated search on the shared Apple GPU runtime.
//!
//! Integration: expose this module under `cfg(all(target_os = "macos", feature = "metal"))`.
//! The binding must lease every exported Buffer, forbid mutation while leases exist,
//! and expose weights read-only. All commands here complete before returning. Buffer
//! clones keep allocations alive but do not prevent writes by another Metal consumer.
//! Absolute observations use bounded FIFO history and the shared CPU TuRBO controller.
//! The old two-row paired-relative experiment is available only by explicit opt-in.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use metal::objc::rc::autoreleasepool;
use metal::{
    Buffer, CommandBuffer, CommandBufferRef, ComputePipelineState, MTLCommandBufferStatus,
};
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::Perturbation;
use crate::apple_gpu::{Runtime, gpu_interval, thread_group};
use crate::fitter::ENNFitter;
use crate::params::ENNParams;
use crate::reliability_region;
use crate::trials::Ask;
use crate::trust_region::{TRLengthConfig, TrustRegionOutcome, TurboTrustRegion};
use crate::weights::AcquisitionKind;

const SOURCE: &str = include_str!("bf16_search.metal");
const TILE_ELEMENTS: usize = 65_536;
const MAX_HISTORY: usize = 128;
#[path = "bf16_family.rs"]
mod family;
use family::{FAMILIES, FamilyHistory};
const POOL_METRICS: usize = 15;
const POOL_PAIRS: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
static NEXT_OWNER: AtomicU64 = AtomicU64::new(1);

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
struct Leaf {
    key: u64,
    offset: u64,
    length: u64,
    scale: f32,
    weight: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParamBlock {
    pub key: u64,
    pub offset: usize,
    pub len: usize,
    pub scale: f32,
    pub weight: f32,
}

impl ParamBlock {
    pub fn new(
        key: u64,
        offset: usize,
        len: usize,
        scale: f32,
        weight: f32,
    ) -> Result<Self, String> {
        if len == 0 || !scale.is_finite() || scale <= 0.0 || !weight.is_finite() || weight <= 0.0 {
            return Err("BF16 block length, scale and weight must be positive and finite".into());
        }
        offset.checked_add(len).ok_or("BF16 block range overflow")?;
        Ok(Self {
            key,
            offset,
            len,
            scale,
            weight,
        })
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Tile {
    leaf: u32,
    start: u32,
    length: u32,
    pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Params {
    seed: u64,
    stream_seed: u64,
    radius: f32,
    alternate_radius: f32,
    candidate: u32,
    tiles: u32,
    history: u32,
    initialize: u32,
    mode: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct SelectionParams {
    root_seed: u64,
    outcomes: [f32; MAX_HISTORY],
    variances: [f32; MAX_HISTORY],
    draws: [f32; MAX_HISTORY],
    base_distances: [f32; MAX_HISTORY],
    local_scales: [f32; MAX_HISTORY],
    epistemic_scale: f32,
    aleatoric_scale: f32,
    y_scale: f32,
    beta: f32,
    radius: f32,
    alternate_radius: f32,
    neighbors: u32,
    history: u32,
    acquisition: u32,
    tiles: u32,
    mode: u32,
    resident_history: u32,
    resident_indices: [u32; 2],
    implicit_history: u32,
    forced_candidate: u32,
    distance_scaling: u32,
    local_scale_neighbors: u32,
    incumbent_index: u32,
    candidate_floor: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Decision {
    index: u32,
    valid: u32,
    root_seed: u64,
    seed: u64,
    radius: f32,
    score: f32,
    mode: u32,
    pad: u32,
    predicted_mean: f32,
    predicted_standard_error: f32,
    incumbent_mean: f32,
    incumbent_standard_error: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct Partial {
    anchor: f32,
    rejected: f32,
    squared: f32,
    changed: u32,
    invalid: u32,
}

pub type ProposalDescription = (u64, f32, f32, Vec<(u64, f64)>);
pub type PoolDescription = (usize, u64, f32, f32, Vec<(i64, f32)>);
pub type PoolGeometry = (Vec<f32>, Vec<(usize, usize, Option<f32>)>, Vec<Option<f32>>);

/// Immutable selected metadata, independent of reusable GPU scratch storage.
#[derive(Clone, Debug)]
pub struct Proposals {
    owner: u64,
    id: u64,
    base_id: i64,
    pub index: usize,
    pub seed: u64,
    pub score: f32,
    pub length: f32,
    pub predicted_mean: f32,
    pub predicted_standard_error: f32,
    pub incumbent_mean: f32,
    pub incumbent_standard_error: f32,
    changes: Vec<(u64, f64)>,
    history_distances: Vec<(i64, f32)>,
    family_distances: Option<Vec<[f32; FAMILIES]>>,
    pub(crate) block_scales: Vec<f32>,
    pool: Vec<PoolDescription>,
    pool_radii: [f32; 4],
    pool_cosines: [Option<f32>; 6],
    reference_cosines: [Option<f32>; 4],
}

impl Proposals {
    pub fn arms(&self) -> usize {
        1
    }

    #[cfg(test)]
    pub(crate) fn pool_keys(&self) -> Vec<(usize, u64, f32)> {
        self.pool
            .iter()
            .map(|candidate| (candidate.0, candidate.1, candidate.2))
            .collect()
    }
}

/// Host wall time including GPU completion; these are not CUDA event timings.
/// The GPU-resident path records the complete pool/select/materialize command in
/// `score_ms`; the other two fields remain zero for tuple compatibility.
#[derive(Clone, Copy, Debug)]
pub struct AskProfile {
    pub score_ms: f32,
    pub pick_ms: f32,
    pub materialize_ms: f32,
    pub total_ms: f32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TellProfile {
    pub reference_ms: f32,
    pub history_copy_ms: f32,
    pub total_ms: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct MemoryInfo {
    pub row_bytes: u64,
    pub resident_bytes: u64,
    pub max_buffer_length: u64,
    pub recommended_max_working_set_size: u64,
    pub current_allocated_size: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct ControllerInfo {
    pub dimensions: usize,
    pub evaluated_arms: usize,
    pub length: f64,
    pub length_min: f64,
    pub length_max: f64,
    pub success_tolerance: i32,
    pub failure_tolerance: i32,
    pub success_counter: i32,
    pub failure_counter: i32,
    pub restarts: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NoisyDecision {
    pub accepted: bool,
    pub incumbent_value: f32,
    pub incumbent_variance: f32,
    pub improvement: f64,
    pub threshold: f64,
    pub predicted_improvement: Option<f64>,
    pub agreement_ratio: Option<f64>,
    pub trust_outcome: TrustRegionOutcome,
}

pub struct SearchState {
    independent_fp16: bool,
    perturbation: Perturbation,
    runtime: Arc<Runtime>,
    blocks: Vec<ParamBlock>,
    tiles: Vec<Tile>,
    leaves_gpu: Buffer,
    tiles_gpu: Buffer,
    offsets_gpu: Buffer,
    base: Buffer,
    anchor: Buffer,
    rejected: Buffer,
    history_rows: Vec<Buffer>,
    proposal: Buffer,
    reference: Option<Buffer>,
    reference_scales: Buffer,
    reference_partials: Buffer,
    partials: Buffer,
    decision: Buffer,
    pool_distances: Buffer,
    pool_geometry: Buffer,
    #[cfg(test)]
    propose_pipeline: ComputePipelineState,
    pool_pipeline: ComputePipelineState,
    selection_pipeline: ComputePipelineState,
    materialize_pipeline: ComputePipelineState,
    reference_pipeline: ComputePipelineState,
    rms_pipeline: ComputePipelineState,
    dimensions: usize,
    resident_bytes: u64,
    length_config: TRLengthConfig,
    length: f64,
    best: f32,
    best_variance: f32,
    outcomes: Vec<f32>,
    variances: Vec<f32>,
    identities: Vec<i64>,
    observation: i64,
    base_id: i64,
    trust: TurboTrustRegion,
    reliability: Option<reliability_region::ReliabilityController>,
    observed: Vec<f64>,
    restart_count: usize,
    history: usize,
    resident_history: usize,
    resident_identities: Vec<i64>,
    pairwise_distances: Vec<f32>,
    distance_scaling: crate::config::DistanceScaling,
    local_scale_neighbors: usize,
    family: Option<FamilyHistory>,
    implicit_history: bool,
    initial_observations: usize,
    fit_candidates: usize,
    fit_samples: usize,
    fit_neighbors: bool,
    fit_seed: u64,
    fitter: Option<ENNFitter>,
    fitted_enn: Option<(usize, f32, f32, f32)>,
    failures: usize,
    owner: u64,
    next_id: u64,
    pending: Option<Proposals>,
    queued: Option<bool>,
    reference_seed: Option<u64>,
    relative: bool,
    started: bool,
    poisoned: bool,
    profiling: bool,
    last_profile: Option<AskProfile>,
    async_command: Option<CommandBuffer>,
    profile_commands: Vec<CommandBuffer>,
    last_tell_profile: Option<TellProfile>,
}

impl SearchState {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        base: &[u16],
        base_value: f32,
        base_variance: f32,
        blocks: Vec<ParamBlock>,
        capacity: usize,
        pending_capacity: usize,
        length: TRLengthConfig,
    ) -> Result<Self, String> {
        autoreleasepool(|| {
            Self::new_inner(
                base,
                Some((base_value, base_variance)),
                blocks,
                capacity,
                pending_capacity,
                length,
                false,
                false,
                Perturbation::Gaussian,
            )
        })
    }

    pub(crate) fn new_unscored(
        base: &[u16],
        blocks: Vec<ParamBlock>,
        capacity: usize,
        pending_capacity: usize,
        length: TRLengthConfig,
    ) -> Result<Self, String> {
        autoreleasepool(|| {
            Self::new_inner(
                base,
                None,
                blocks,
                capacity,
                pending_capacity,
                length,
                false,
                false,
                Perturbation::Gaussian,
            )
        })
    }

    /// Full FP16 weight rows with independent per-coordinate innovations.
    pub(crate) fn new_fp16(
        base: &[u16],
        blocks: Vec<ParamBlock>,
        capacity: usize,
        length: TRLengthConfig,
        perturbation: Perturbation,
    ) -> Result<Self, String> {
        autoreleasepool(|| {
            Self::new_inner(
                base,
                None,
                blocks,
                capacity,
                1,
                length,
                true,
                false,
                perturbation,
            )
        })
    }

    pub(crate) fn new_fp16_implicit(
        base: &[u16],
        blocks: Vec<ParamBlock>,
        resident_capacity: usize,
        length: TRLengthConfig,
        perturbation: Perturbation,
    ) -> Result<Self, String> {
        autoreleasepool(|| {
            Self::new_inner(
                base,
                None,
                blocks,
                resident_capacity,
                1,
                length,
                true,
                true,
                perturbation,
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn new_inner(
        base: &[u16],
        initial: Option<(f32, f32)>,
        blocks: Vec<ParamBlock>,
        capacity: usize,
        pending_capacity: usize,
        length: TRLengthConfig,
        independent_fp16: bool,
        implicit_history: bool,
        perturbation: Perturbation,
    ) -> Result<Self, String> {
        let length = checked_length(length)?;
        if capacity == 0 || capacity > MAX_HISTORY || pending_capacity != 1 {
            return Err(
                "Metal correlated search requires capacity in 1..=128 and max_pending=1".into(),
            );
        }
        if let Some((value, variance)) = initial {
            check_scores(&[value], &[variance])?;
        }
        let (tiles, offsets, leaves) = make_layout(&blocks, base.len())?;
        let exponent_mask = if independent_fp16 { 0x7c00 } else { 0x7f80 };
        if base
            .iter()
            .any(|bits| bits & exponent_mask == exponent_mask)
        {
            return Err("Base weights must be finite".into());
        }
        let row_bytes = bytes::<u16>(base.len())?;
        let partial_count = tiles
            .len()
            .checked_mul(4)
            .and_then(|n| n.checked_mul(capacity.div_ceil(2)))
            .ok_or("BF16 partial count overflow")?;
        let geometry_count = tiles
            .len()
            .checked_mul(POOL_METRICS)
            .ok_or("BF16 geometry count overflow")?;
        let sizes = [
            row_bytes,
            bytes::<Leaf>(blocks.len())?,
            bytes::<Tile>(tiles.len())?,
            bytes::<u32>(offsets.len())?,
            bytes::<f32>(blocks.len())?,
            bytes::<f32>(tiles.len())?,
            bytes::<Partial>(partial_count)?,
            bytes::<Decision>(1)?,
            bytes::<f32>(4 * MAX_HISTORY)?,
            bytes::<f32>(geometry_count)?,
        ];
        let resident_bytes = sizes.iter().try_fold(
            row_bytes
                .checked_mul(capacity as u64 + 2)
                .ok_or("BF16 memory overflow")?,
            |sum, size| sum.checked_add(*size).ok_or("BF16 memory overflow"),
        )?;
        let runtime = Runtime::shared()?;
        preflight(&runtime, &sizes, resident_bytes)?;
        let mut defines = String::new();
        if independent_fp16 {
            defines.push_str("#define FP16_INDEPENDENT\n");
        }
        if perturbation == Perturbation::Rademacher {
            defines.push_str("#define RADEMACHER_ONLY\n");
        }
        let source = format!("{defines}{SOURCE}");
        #[cfg(test)]
        let propose_pipeline = runtime.precise(&source, "Dense search", "bf16_propose")?;
        let pool_pipeline = runtime.precise(&source, "Dense search", "bf16_propose_pool")?;
        let selection_pipeline = runtime.precise(&source, "Dense search", "bf16_select")?;
        let materialize_pipeline = runtime.precise(&source, "Dense search", "bf16_materialize")?;
        let reference_pipeline = runtime.precise(&source, "Dense search", "bf16_reference")?;
        let rms_pipeline = runtime.precise(&source, "Dense search", "bf16_reference_rms")?;
        for pipeline in [
            &pool_pipeline,
            &selection_pipeline,
            &materialize_pipeline,
            &reference_pipeline,
            &rms_pipeline,
        ] {
            if pipeline.max_total_threads_per_threadgroup() < 256 {
                return Err("BF16 Metal kernels require 256 threads per group".into());
            }
        }
        let history_rows: Vec<_> = (0..capacity)
            .map(|_| runtime.buffer::<u16>(base.len()))
            .collect();
        let mut trust = TurboTrustRegion::new(base.len(), length);
        trust.set_arms(1);
        let mut state = Self {
            independent_fp16,
            perturbation,
            leaves_gpu: runtime.buffer_with(&leaves),
            tiles_gpu: runtime.buffer_with(&tiles),
            offsets_gpu: runtime.buffer_with(&offsets),
            base: runtime.buffer_with(base),
            anchor: history_rows[0].clone(),
            rejected: history_rows[capacity.min(2) - 1].clone(),
            history_rows,
            proposal: runtime.buffer::<u16>(base.len()),
            reference: None,
            reference_scales: runtime.buffer_with(&vec![1.0f32; blocks.len()]),
            reference_partials: runtime.buffer::<f32>(tiles.len()),
            partials: runtime.buffer::<Partial>(partial_count),
            decision: runtime.buffer::<Decision>(1),
            pool_distances: runtime.buffer::<f32>(4 * MAX_HISTORY),
            pool_geometry: runtime.buffer::<f32>(geometry_count),
            runtime,
            blocks,
            tiles,
            #[cfg(test)]
            propose_pipeline,
            pool_pipeline,
            selection_pipeline,
            materialize_pipeline,
            reference_pipeline,
            rms_pipeline,
            dimensions: base.len(),
            resident_bytes,
            length_config: length,
            length: length.length_init,
            best: f32::NEG_INFINITY,
            best_variance: 0.0,
            outcomes: vec![
                0.0;
                if implicit_history {
                    MAX_HISTORY
                } else {
                    capacity
                }
            ],
            variances: vec![
                0.0;
                if implicit_history {
                    MAX_HISTORY
                } else {
                    capacity
                }
            ],
            identities: vec![
                1;
                if implicit_history {
                    MAX_HISTORY
                } else {
                    capacity
                }
            ],
            observation: 1,
            base_id: 1,
            trust,
            reliability: None,
            observed: Vec::new(),
            restart_count: 0,
            history: 0,
            resident_history: 0,
            resident_identities: vec![1; capacity],
            pairwise_distances: vec![0.0; MAX_HISTORY * MAX_HISTORY],
            distance_scaling: crate::config::DistanceScaling::Global,
            local_scale_neighbors: 8,
            family: None,
            implicit_history,
            initial_observations: 0,
            fit_candidates: 0,
            fit_samples: 0,
            fit_neighbors: false,
            fit_seed: 0,
            fitter: None,
            fitted_enn: None,
            failures: 0,
            owner: NEXT_OWNER.fetch_add(1, Ordering::Relaxed),
            next_id: 0,
            pending: None,
            queued: None,
            reference_seed: None,
            relative: false,
            started: false,
            poisoned: false,
            profiling: false,
            last_profile: None,
            async_command: None,
            profile_commands: Vec::new(),
            last_tell_profile: None,
        };
        state.copy(&state.base, &state.anchor)?;
        if let Some((value, variance)) = initial {
            state.observe_initial(value, variance)?;
        }
        Ok(state)
    }

    /// Supply the measured incumbent before the first proposal or controller update.
    pub(crate) fn observe_initial(&mut self, value: f32, variance: f32) -> Result<(), String> {
        self.check_idle()?;
        if self.started || self.history != 0 {
            return Err("Initial Metal BF16 observation is already set".into());
        }
        check_scores(&[value], &[variance])?;
        self.observed.push(f64::from(value));
        self.trust
            .update(&ndarray::ArrayView1::from(&self.observed), 1)
            .map_err(|error| error.to_string())?;
        self.outcomes[0] = value;
        self.variances[0] = variance;
        self.best = value;
        self.best_variance = variance;
        self.history = 1;
        self.resident_history = 1;
        self.resident_identities[0] = self.base_id;
        Ok(())
    }

    pub(crate) fn set_failure_tolerance(&mut self, failures: usize) -> Result<(), String> {
        self.check_idle()?;
        if self.started {
            return Err("Set Metal BF16 failure tolerance before any asks".into());
        }
        self.trust
            .set_failure_tolerance(failures)
            .map_err(|error| error.to_string())
    }

    pub(crate) fn configure_implicit_enn(
        &mut self,
        config: crate::config::ResidentEnnConfig,
    ) -> Result<(), String> {
        self.check_idle()?;
        check_ask(config.ask)?;
        if !self.implicit_history || self.history != 0 {
            return Err("Configure implicit ENN before its first observation".into());
        }
        if config.ask.neighbors > MAX_HISTORY {
            return Err(format!(
                "Implicit ENN k={} exceeds logical history capacity {MAX_HISTORY}",
                config.ask.neighbors
            ));
        }
        let k =
            i32::try_from(config.ask.neighbors).map_err(|_| "Implicit ENN k does not fit i32")?;
        let mut fitter = ENNFitter::new(k, true);
        fitter.set_params(
            ENNParams::new(
                k,
                f64::from(config.ask.epistemic_scale),
                f64::from(config.ask.aleatoric_scale),
            )
            .map_err(|error| error.to_string())?,
        );
        self.initial_observations = config.ask.neighbors;
        self.fit_candidates = config.num_candidates;
        self.fit_samples = config.num_samples;
        self.fit_neighbors = config.fit_neighbors;
        self.fit_seed = config.ask.seed;
        self.distance_scaling = config.distance_scaling;
        self.local_scale_neighbors = config.local_scale_neighbors;
        self.fitter = Some(fitter);
        Ok(())
    }

    pub(crate) fn configure_reliability_controller(
        &mut self,
        config: reliability_region::ReliabilityControllerConfig,
    ) -> Result<(), String> {
        self.check_idle()?;
        if !self.implicit_history || self.started || self.history != 0 {
            return Err("Configure reliability control before the initial observation".into());
        }
        if self.fitter.is_none() {
            return Err("Configure the resident ENN before reliability control".into());
        }
        self.reliability = Some(reliability_region::ReliabilityController::new(
            config,
            self.length_config,
        )?);
        Ok(())
    }

    pub(crate) fn fitted_enn(&self) -> Option<(usize, f32, f32, f32)> {
        self.fitted_enn
    }

    pub(crate) fn enable_family_shape(&mut self, groups: Vec<usize>) -> Result<(), String> {
        self.check_idle()?;
        if !self.independent_fp16
            || !self.implicit_history
            || self.history != 0
            || self.family.is_some()
        {
            return Err("Configure learned family shape once before initial observation".into());
        }
        self.family = Some(FamilyHistory::new(groups, &self.blocks)?);
        Ok(())
    }

    pub(crate) fn family_shape(&self) -> Option<([f32; FAMILIES], [f32; FAMILIES])> {
        self.family.as_ref().map(|f| (f.weights, f.scales()))
    }

    fn apply_family_shape(&mut self) -> Result<(), String> {
        let Some(family) = &self.family else {
            return Ok(());
        };
        let scales = family.scales();
        for (i, block) in self.blocks.iter_mut().enumerate() {
            let g = family.groups[i];
            block.scale = family.original[i].scale * scales[g];
            block.weight = family.original[i].weight * family.weights[g];
        }
        let (_, _, leaves) = make_layout(&self.blocks, self.dimensions)?;
        // No ask is in flight: fitting occurs after completed objective/tell work.
        unsafe {
            std::ptr::copy_nonoverlapping(
                leaves.as_ptr(),
                self.leaves_gpu.contents().cast::<Leaf>(),
                leaves.len(),
            );
        }
        let distances = family.aggregate(family.weights, self.history);
        for i in 0..self.history {
            for j in 0..self.history {
                self.pairwise_distances[i * MAX_HISTORY + j] = distances[[i, j]] as f32;
            }
        }
        Ok(())
    }

    /// Records the seed; the model-sized reference allocation waits until first ask.
    pub fn correlate(&mut self, reference_seed: u64) -> Result<(), String> {
        self.check_idle()?;
        if self.independent_fp16 || self.started || self.reference_seed.is_some() {
            return Err("Enable correlated Metal sampling once before any asks".into());
        }
        self.reference_seed = Some(reference_seed);
        Ok(())
    }

    /// Build the correlated direction before a latency-sensitive ask.
    pub(crate) fn prepare_reference(&mut self) -> Result<(), String> {
        self.check_idle()?;
        self.ensure_ref()
    }

    pub fn enable_relative(&mut self, failure_tolerance: usize) -> Result<(), String> {
        self.check_idle()?;
        if self.started
            || self.relative
            || self.history == 0
            || self.reference_seed.is_none()
            || failure_tolerance != 4
            || self.history_rows.len() != 2
        {
            return Err("Metal paired-relative mode requires fresh correlated search and failure_tolerance=4".into());
        }
        self.relative = true;
        self.outcomes.fill(0.0);
        self.variances.fill(0.0);
        Ok(())
    }

    /// Score, select and materialize four correlated BF16 candidates on the GPU.
    pub fn ask_round(
        &mut self,
        arms: usize,
        candidates: usize,
        seed: u64,
        config: Ask,
    ) -> Result<Proposals, String> {
        autoreleasepool(|| self.ask_inner(arms, candidates, seed, config))
    }

    /// Enqueue candidate selection and materialization without waiting. The
    /// caller may append work on the shared Metal queue before finalizing the
    /// proposal. Queue ordering makes the selected row visible to that work.
    pub fn begin_ask(
        &mut self,
        arms: usize,
        candidates: usize,
        seed: u64,
        config: Ask,
    ) -> Result<Buffer, String> {
        self.begin_ask_mode(arms, candidates, seed, config, None)
    }

    /// Diagnostic selection ablation: preserve the pool, posterior and tell
    /// policy, but choose a precommitted pool index instead of maximizing UCB.
    #[cfg(test)]
    pub(crate) fn begin_forced(
        &mut self,
        seed: u64,
        config: Ask,
        candidate: usize,
    ) -> Result<Buffer, String> {
        if candidate >= 4 {
            return Err("Diagnostic candidate must be below four".into());
        }
        self.begin_ask_mode(1, 4, seed, config, Some(candidate))
    }

    /// Materialize a counterfactual from an already-scored pool. Diagnostics
    /// must restore round.index before tell; no observation is added here.
    #[cfg(test)]
    pub(crate) fn diagnostic_row(
        &self,
        round: &Proposals,
        root: u64,
        config: Ask,
        candidate: usize,
    ) -> Result<Buffer, String> {
        self.check_round(round)?;
        let decision = read::<Decision>(&self.decision, 1)[0];
        if candidate >= 4 || root != decision.root_seed {
            return Err("Diagnostic pool candidate or root mismatch".into());
        }
        let command = self.runtime.queue.new_command_buffer();
        self.encode_select(command, root, config, Some(candidate));
        self.encode_row(command);
        finish(command)?;
        Ok(self.proposal.clone())
    }

    pub(crate) fn begin_initial(&mut self, seed: u64, candidate: usize) -> Result<Buffer, String> {
        if !self.implicit_history
            || self.initial_observations == 0
            || self.history >= self.initial_observations
        {
            return Err("Metal ENN initialization is complete or not configured".into());
        }
        if candidate >= 4 {
            return Err("Metal initialization candidate must be below four".into());
        }
        self.begin_ask_mode(1, 4, seed, Ask::default(), Some(candidate))
    }

    fn begin_ask_mode(
        &mut self,
        arms: usize,
        candidates: usize,
        seed: u64,
        config: Ask,
        forced_candidate: Option<usize>,
    ) -> Result<Buffer, String> {
        self.check_idle()?;
        if self.history == 0 {
            return Err("Measure the initial Metal BF16 incumbent before asking".into());
        }
        if arms != 1 || candidates != 4 {
            return Err("Metal search requires arms=1 and candidates=4".into());
        }
        check_ask(config)?;
        self.last_profile = None;
        self.ensure_ref()?;
        if self.profiling {
            self.profile_commands.clear();
            let command = self.runtime.queue.new_command_buffer().to_owned();
            self.encode_pool(&command, self.pool_params(seed));
            command.commit();
            self.profile_commands.push(command);
            let command = self.runtime.queue.new_command_buffer().to_owned();
            self.encode_select(&command, seed, config, forced_candidate);
            command.commit();
            self.profile_commands.push(command);
            let command = self.runtime.queue.new_command_buffer().to_owned();
            self.encode_row(&command);
            command.commit();
            self.profile_commands.push(command);
            self.async_command = Some(self.profile_commands[2].to_owned());
            return Ok(self.proposal.clone());
        }
        let command = self.runtime.queue.new_command_buffer();
        self.encode_pool(command, self.pool_params(seed));
        self.encode_select(command, seed, config, forced_candidate);
        self.encode_row(command);
        command.commit();
        self.async_command = Some(command.to_owned());
        Ok(self.proposal.clone())
    }

    /// Wait for a previously enqueued ask and publish its immutable handle.
    pub fn finish_ask(&mut self) -> Result<Proposals, String> {
        let command = self
            .async_command
            .take()
            .ok_or_else(|| "No asynchronous Metal BF16 ask is pending".to_string())?;
        command.wait_until_completed();
        if command.status() != MTLCommandBufferStatus::Completed {
            self.poisoned = true;
            return Err(format!(
                "Metal BF16 asynchronous ask failed: {:?}",
                command.status()
            ));
        }
        self.finish_profile()?;
        self.publish_ask()
    }

    /// Abort an asynchronous ask after a downstream evaluator failure.
    pub fn abort_ask(&mut self) {
        if let Some(command) = self.async_command.take() {
            command.wait_until_completed();
        }
        self.poisoned = true;
    }

    fn ask_inner(
        &mut self,
        arms: usize,
        candidates: usize,
        seed: u64,
        config: Ask,
    ) -> Result<Proposals, String> {
        let start = Instant::now();
        self.begin_ask(arms, candidates, seed, config)?;
        self.finish_askat(start)
    }

    fn finish_askat(&mut self, start: Instant) -> Result<Proposals, String> {
        let command = self
            .async_command
            .take()
            .ok_or_else(|| "No asynchronous Metal BF16 ask is pending".to_string())?;
        command.wait_until_completed();
        if command.status() != MTLCommandBufferStatus::Completed {
            self.poisoned = true;
            return Err(format!(
                "Metal BF16 asynchronous ask failed: {:?}",
                command.status()
            ));
        }
        self.finish_profile()?;
        self.publish_start(start)
    }

    fn finish_profile(&mut self) -> Result<(), String> {
        if !self.profiling {
            self.last_profile = None;
            return Ok(());
        }
        if self.profile_commands.len() != 3 {
            return Err("Metal BF16 controller trace is incomplete".into());
        }
        let intervals: Vec<_> = self
            .profile_commands
            .iter()
            .map(|command| {
                if command.status() != MTLCommandBufferStatus::Completed {
                    return Err(format!(
                        "Metal BF16 traced ask failed: {:?}",
                        command.status()
                    ));
                }
                gpu_interval(command)
                    .ok_or_else(|| "Metal did not expose a controller GPU interval".to_string())
            })
            .collect::<Result<_, _>>()?;
        let milliseconds =
            |index: usize| ((intervals[index].1 - intervals[index].0) * 1000.0) as f32;
        self.last_profile = Some(AskProfile {
            score_ms: milliseconds(0),
            pick_ms: milliseconds(1),
            materialize_ms: milliseconds(2),
            total_ms: ((intervals[2].1 - intervals[0].0) * 1000.0) as f32,
        });
        self.profile_commands.clear();
        Ok(())
    }

    fn publish_ask(&mut self) -> Result<Proposals, String> {
        self.publish_start(Instant::now())
    }

    fn publish_start(&mut self, start: Instant) -> Result<Proposals, String> {
        let total_ms = start.elapsed().as_secs_f32() * 1000.0;
        let decision = read::<Decision>(&self.decision, 1)[0];
        if decision.valid == 0
            || decision.index >= 4
            || !decision.score.is_finite()
            || !decision.predicted_mean.is_finite()
            || !decision.predicted_standard_error.is_finite()
            || decision.predicted_standard_error < 0.0
            || !decision.incumbent_mean.is_finite()
            || !decision.incumbent_standard_error.is_finite()
            || decision.incumbent_standard_error < 0.0
        {
            return Err("No finite Metal BF16 acquisition candidate".into());
        }
        if decision.mode != self.perturbation.shader() {
            return Err("Metal BF16 selected perturbation mode changed".into());
        }
        let index = decision.index as usize;
        let distances = read::<f32>(&self.pool_distances, 4 * self.history);
        if distances.iter().any(|distance| !distance.is_finite()) {
            return Err("Metal BF16 candidate-pool history distance is nonfinite".into());
        }
        let pool = distances
            .chunks_exact(self.history)
            .enumerate()
            .map(|(candidate, row)| {
                (
                    candidate,
                    candidate_seed(decision.root_seed, candidate),
                    self.radius(candidate),
                    if !self.independent_fp16 && candidate < 2 {
                        0.75
                    } else {
                        0.0
                    },
                    self.identities[..self.history]
                        .iter()
                        .copied()
                        .zip(row.iter().copied())
                        .collect(),
                )
            })
            .collect::<Vec<PoolDescription>>();
        if pool[index].1 != decision.seed || pool[index].2 != decision.radius {
            return Err("Metal BF16 selected descriptor disagrees with its pool".into());
        }
        let geometry = read::<f32>(&self.pool_geometry, POOL_METRICS * self.tiles.len());
        if geometry.iter().any(|value| !value.is_finite()) {
            return Err("Metal BF16 candidate-pool geometry is nonfinite".into());
        }
        let mut totals = [0.0f64; POOL_METRICS];
        for (metric, total) in totals.iter_mut().enumerate() {
            *total = geometry[metric * self.tiles.len()..(metric + 1) * self.tiles.len()]
                .iter()
                .map(|&value| f64::from(value))
                .sum();
        }
        if totals[..4].iter().any(|&norm| norm < 0.0) {
            return Err("Metal BF16 candidate pool has negative squared radius".into());
        }
        let pool_radii = std::array::from_fn(|candidate| totals[candidate].sqrt() as f32);
        let mut pool_cosines = [None; 6];
        for (pair, &(left, right)) in POOL_PAIRS.iter().enumerate() {
            let denominator = f64::from(pool_radii[left]) * f64::from(pool_radii[right]);
            if denominator == 0.0 {
                continue;
            }
            let cosine = totals[pair + 4] / denominator;
            if !cosine.is_finite() || cosine.abs() > 1.001 {
                return Err("Metal BF16 candidate-pool cosine is invalid".into());
            }
            pool_cosines[pair] = Some(cosine.clamp(-1.0, 1.0) as f32);
        }
        if totals[10] < 0.0 {
            return Err("Metal BF16 reference has negative squared radius".into());
        }
        let reference_radius = totals[10].sqrt();
        let mut reference_cosines = [None; 4];
        for candidate in 0..4 {
            let denominator = f64::from(pool_radii[candidate]) * reference_radius;
            if denominator == 0.0 {
                continue;
            }
            let cosine = totals[candidate + 11] / denominator;
            if !cosine.is_finite() || cosine.abs() > 1.001 {
                return Err("Metal BF16 candidate-reference cosine is invalid".into());
            }
            reference_cosines[candidate] = Some(cosine.clamp(-1.0, 1.0) as f32);
        }
        let history_distances = pool[index].4.clone();
        let partials = read::<Partial>(&self.partials, self.tiles.len() * 4);
        let mut changes = vec![(0u64, 0.0f64); self.blocks.len()];
        for (tile_index, tile) in self.tiles.iter().enumerate() {
            let partial = partials[index * self.tiles.len() + tile_index];
            changes[tile.leaf as usize].0 += u64::from(partial.changed);
            changes[tile.leaf as usize].1 += f64::from(partial.squared);
        }
        let family_distances = if let Some(family) = &self.family {
            let base = self.identities[..self.history]
                .iter()
                .position(|&id| id == self.base_id)
                .ok_or("Family history lost its incumbent")?;
            let mut norms = [0.0; FAMILIES];
            let mut resident = [[0.0; FAMILIES]; 2];
            for (t, tile) in self.tiles.iter().enumerate() {
                let leaf = tile.leaf as usize;
                let g = family.groups[leaf];
                let p = partials[index * self.tiles.len() + t];
                norms[g] += p.squared * family.original[leaf].weight;
                resident[0][g] += p.anchor / family.weights[g];
                resident[1][g] += p.rejected / family.weights[g];
            }
            let mut rows = (0..self.history)
                .map(|i| {
                    std::array::from_fn(|g| family.components[base * MAX_HISTORY + i][g] + norms[g])
                })
                .collect::<Vec<_>>();
            for (slot, values) in resident.iter().enumerate().take(self.resident_history) {
                let row = self.identities[..self.history]
                    .iter()
                    .position(|&id| id == self.resident_identities[slot])
                    .ok_or("Family history lost resident identity")?;
                rows[row] = *values;
            }
            if rows.iter().flatten().any(|v| !v.is_finite() || *v < 0.0) {
                return Err("Invalid family distance component".into());
            }
            Some(rows)
        } else {
            None
        };
        let round = Proposals {
            owner: self.owner,
            id: self.next_id,
            base_id: self.base_id,
            index,
            seed: decision.seed,
            score: decision.score,
            length: decision.radius,
            predicted_mean: decision.predicted_mean,
            predicted_standard_error: decision.predicted_standard_error,
            incumbent_mean: decision.incumbent_mean,
            incumbent_standard_error: decision.incumbent_standard_error,
            changes,
            history_distances,
            family_distances,
            block_scales: self.blocks.iter().map(|b| b.scale).collect(),
            pool,
            pool_radii,
            pool_cosines,
            reference_cosines,
        };
        self.pending = Some(round.clone());
        self.next_id = self.next_id.wrapping_add(1);
        self.started = true;
        if self.profiling && self.last_profile.is_none() {
            self.last_profile = Some(AskProfile {
                score_ms: total_ms,
                pick_ms: 0.0,
                materialize_ms: 0.0,
                total_ms,
            });
        }
        Ok(round)
    }

    fn local_history_scales(&self) -> Vec<f64> {
        (0..self.history)
            .map(|row| {
                let mut values = (0..self.history)
                    .filter(|&column| column != row)
                    .map(|column| f64::from(self.pairwise_distances[row * MAX_HISTORY + column]))
                    .collect::<Vec<_>>();
                values.sort_by(f64::total_cmp);
                let k = self.local_scale_neighbors.min(values.len());
                values.get(k.saturating_sub(1)).copied().unwrap_or(1.0)
            })
            .collect()
    }

    fn ranked_neighbors(&self, round: &Proposals) -> Vec<usize> {
        let scales = (self.distance_scaling == crate::config::DistanceScaling::SelfTuning)
            .then(|| self.local_history_scales());
        let mut ranked = round
            .history_distances
            .iter()
            .enumerate()
            .map(|(index, &(_, distance))| {
                let scaled = scales.as_ref().map_or(f64::from(distance), |values| {
                    f64::from(distance) / values[index].max(1.0e-12).sqrt()
                });
                (index, scaled)
            })
            .collect::<Vec<_>>();
        ranked.sort_by(|left, right| left.1.total_cmp(&right.1));
        ranked
            .into_iter()
            .take(self.initial_observations.min(self.history))
            .map(|(index, _)| index)
            .collect()
    }

    fn rank_concordance(
        &self,
        round: &Proposals,
        value: f32,
        variance: f32,
    ) -> (Option<f64>, f64, f64) {
        let Some(controller) = &self.reliability else {
            return (None, 0.0, 0.0);
        };
        let z = controller.config().confidence_z;
        let predicted_variance = f64::from(round.predicted_standard_error).powi(2);
        let mut decisive = 0usize;
        let mut correct = 0usize;
        let neighbors = self.ranked_neighbors(round);
        for index in neighbors.iter().copied() {
            let outcome = f64::from(self.outcomes[index]);
            let history_variance = f64::from(self.variances[index]);
            let predicted_delta = f64::from(round.predicted_mean) - outcome;
            let observed_delta = f64::from(value) - outcome;
            let predicted_threshold = z * (predicted_variance + history_variance).sqrt();
            let observed_threshold = z * (f64::from(variance) + history_variance).sqrt();
            if predicted_delta.abs() <= predicted_threshold
                || observed_delta.abs() <= observed_threshold
            {
                continue;
            }
            decisive += 1;
            correct += usize::from(predicted_delta.signum() == observed_delta.signum());
        }
        let coverage = decisive as f64 / neighbors.len().max(1) as f64;
        let concordance = (decisive > 0).then_some(correct as f64 / decisive as f64);
        (concordance, coverage, decisive as f64)
    }

    fn reliability_evidence(
        &self,
        round: &Proposals,
        value: f32,
        variance: f32,
        improvement: f64,
        improvement_variance: f64,
    ) -> Result<Option<reliability_region::ReliabilityEvidence>, String> {
        if self.reliability.is_none() {
            return Ok(None);
        }
        let realized_radius = f64::from(
            *round
                .pool_radii
                .get(round.index)
                .ok_or("Selected pool radius is missing")?,
        );
        let (concordance, coverage, rank_evidence) = self.rank_concordance(round, value, variance);
        Ok(Some(reliability_region::ReliabilityEvidence {
            concordance,
            coverage,
            rank_evidence,
            improvement,
            improvement_variance,
            nominal_radius: f64::from(round.length),
            realized_radius,
            center_radius: None,
        }))
    }

    fn center_radius(&self) -> Option<f64> {
        let config = self.reliability.as_ref()?.config();
        let center = self.identities[..self.history]
            .iter()
            .position(|&identity| identity == self.base_id)?;
        let mut distances = (0..self.history)
            .filter(|&column| column != center)
            .map(|column| f64::from(self.pairwise_distances[center * MAX_HISTORY + column]))
            .filter(|distance| distance.is_finite() && *distance >= 0.0)
            .collect::<Vec<_>>();
        distances.sort_by(f64::total_cmp);
        let k = config.local_scale_neighbors.min(distances.len());
        distances
            .get(k.checked_sub(1)?)
            .copied()
            .map(|squared| squared.max(1.0e-12).sqrt())
    }

    /// Keep absolute measurements across incumbent changes. The paired decision
    /// controls weight acceptance; radius adaptation uses the shared CPU controller.
    pub fn tell_paired(
        &mut self,
        round: &Proposals,
        value: f32,
        variance: f32,
        incumbent_value: f32,
        incumbent_variance: f32,
        accept: bool,
    ) -> Result<(), String> {
        autoreleasepool(|| {
            self.tell_absolute(
                round,
                value,
                variance,
                incumbent_value,
                incumbent_variance,
                accept,
                None,
                None,
                None,
                true,
            )
        })
    }

    pub(crate) fn tell_initial(
        &mut self,
        round: &Proposals,
        value: f32,
        variance: f32,
    ) -> Result<NoisyDecision, String> {
        autoreleasepool(|| {
            if !self.implicit_history
                || self.initial_observations == 0
                || self.history >= self.initial_observations
            {
                return Err("Metal ENN initialization is complete or not configured".into());
            }
            let incumbent_value = self.best;
            let incumbent_variance = self.best_variance;
            check_scores(&[value, incumbent_value], &[variance, incumbent_variance])?;
            let improvement = f64::from(value) - f64::from(incumbent_value);
            let accepted = improvement > 0.0;
            self.tell_absolute(
                round,
                value,
                variance,
                incumbent_value,
                incumbent_variance,
                accepted,
                None,
                None,
                None,
                false,
            )?;
            if self.history == self.initial_observations {
                if let Some(controller) = &self.reliability {
                    self.length = controller.length();
                } else {
                    let failure_tolerance = self.trust.failure_tolerance() as usize;
                    self.trust = TurboTrustRegion::new(self.dimensions, self.length_config);
                    self.trust.set_arms(1);
                    self.trust
                        .set_failure_tolerance(failure_tolerance)
                        .map_err(|error| error.to_string())?;
                    self.trust
                        .update(
                            &ndarray::ArrayView1::from(&self.observed),
                            self.observed.len(),
                        )
                        .map_err(|error| error.to_string())?;
                    self.length = self.trust.length();
                }
            }
            Ok(NoisyDecision {
                accepted,
                incumbent_value,
                incumbent_variance,
                improvement,
                threshold: 0.0,
                predicted_improvement: None,
                agreement_ratio: None,
                trust_outcome: TrustRegionOutcome::Inconclusive,
            })
        })
    }

    /// Add one independent noisy observation. The candidate becomes the new
    /// incumbent only when its reward exceeds the stored incumbent by two
    /// combined standard errors.
    pub fn tell_noisy(
        &mut self,
        round: &Proposals,
        value: f32,
        variance: f32,
    ) -> Result<NoisyDecision, String> {
        autoreleasepool(|| {
            self.check_round(round)?;
            if self.relative {
                return Err("Cannot mix absolute and paired-relative observations".into());
            }
            let incumbent_value = self.best;
            let incumbent_variance = self.best_variance;
            check_scores(&[value, incumbent_value], &[variance, incumbent_variance])?;
            let improvement = f64::from(value) - f64::from(incumbent_value);
            let combined_variance = f64::from(variance) + f64::from(incumbent_variance);
            if !combined_variance.is_finite() {
                return Err("Combined noisy-observation variance is not finite".into());
            }
            let threshold = 2.0 * combined_variance.sqrt();
            let accepted = improvement > threshold;
            let next_incumbent = if accepted { value } else { incumbent_value };
            self.tell_absolute(
                round,
                value,
                variance,
                incumbent_value,
                incumbent_variance,
                accepted,
                Some(next_incumbent),
                None,
                None,
                true,
            )?;
            Ok(NoisyDecision {
                accepted,
                incumbent_value,
                incumbent_variance,
                improvement,
                threshold,
                predicted_improvement: None,
                agreement_ratio: None,
                trust_outcome: if accepted {
                    TrustRegionOutcome::Success
                } else {
                    TrustRegionOutcome::Failure
                },
            })
        })
    }

    /// Add one stochastic-objective observation using the local ENN posterior
    /// for the incumbent baseline and trust-region model agreement.
    pub fn tell_model_aware(
        &mut self,
        round: &Proposals,
        value: f32,
        variance: f32,
    ) -> Result<NoisyDecision, String> {
        autoreleasepool(|| {
            self.check_round(round)?;
            if self.relative {
                return Err("Cannot mix absolute and paired-relative observations".into());
            }
            check_scores(
                &[value, round.predicted_mean, round.incumbent_mean],
                &[
                    variance,
                    round.predicted_standard_error.powi(2),
                    round.incumbent_standard_error.powi(2),
                ],
            )?;
            let incumbent_value = round.incumbent_mean;
            let incumbent_variance = round.incumbent_standard_error.powi(2);
            let improvement = f64::from(value) - f64::from(incumbent_value);
            let threshold = 2.0 * (f64::from(variance) + f64::from(incumbent_variance)).sqrt();
            let accepted = improvement > threshold;
            let predicted_improvement =
                f64::from(round.predicted_mean) - f64::from(incumbent_value);
            let agreement_ratio = (predicted_improvement > f64::EPSILON)
                .then_some(improvement / predicted_improvement);
            let trust_outcome = if accepted {
                TrustRegionOutcome::Success
            } else if improvement < -threshold {
                TrustRegionOutcome::Failure
            } else {
                TrustRegionOutcome::Inconclusive
            };
            let next_incumbent = if accepted { value } else { incumbent_value };
            let reliability = self.reliability_evidence(
                round,
                value,
                variance,
                improvement,
                f64::from(variance) + f64::from(incumbent_variance),
            )?;
            self.tell_absolute(
                round,
                value,
                variance,
                incumbent_value,
                incumbent_variance,
                accepted,
                Some(next_incumbent),
                Some(trust_outcome),
                reliability,
                true,
            )?;
            Ok(NoisyDecision {
                accepted,
                incumbent_value,
                incumbent_variance,
                improvement,
                threshold,
                predicted_improvement: Some(predicted_improvement),
                agreement_ratio,
                trust_outcome,
            })
        })
    }

    /// Add an absolute cumulative outcome measured through a paired
    /// candidate/incumbent comparison on the same stochastic objective.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn tell_paired_model_aware(
        &mut self,
        round: &Proposals,
        value: f32,
        variance: f32,
        incumbent_value: f32,
        incumbent_variance: f32,
        improvement_variance: f32,
    ) -> Result<NoisyDecision, String> {
        autoreleasepool(|| {
            self.check_round(round)?;
            check_scores(
                &[
                    value,
                    incumbent_value,
                    round.predicted_mean,
                    round.incumbent_mean,
                ],
                &[
                    variance,
                    incumbent_variance,
                    round.predicted_standard_error.powi(2),
                    round.incumbent_standard_error.powi(2),
                ],
            )?;
            if !improvement_variance.is_finite() || improvement_variance < 0.0 {
                return Err("Paired improvement variance must be finite and nonnegative".into());
            }
            let improvement = f64::from(value) - f64::from(incumbent_value);
            let threshold = 2.0 * f64::from(improvement_variance).sqrt();
            let accepted = improvement > threshold;
            let predicted_improvement =
                f64::from(round.predicted_mean) - f64::from(round.incumbent_mean);
            let agreement_ratio = (predicted_improvement > f64::EPSILON)
                .then_some(improvement / predicted_improvement);
            let trust_outcome = if accepted {
                TrustRegionOutcome::Success
            } else if improvement < -threshold {
                TrustRegionOutcome::Failure
            } else {
                TrustRegionOutcome::Inconclusive
            };
            let reliability = self.reliability_evidence(
                round,
                value,
                variance,
                improvement,
                f64::from(improvement_variance),
            )?;
            self.tell_absolute(
                round,
                value,
                variance,
                incumbent_value,
                incumbent_variance,
                accepted,
                Some(incumbent_value),
                Some(trust_outcome),
                reliability,
                true,
            )?;
            Ok(NoisyDecision {
                accepted,
                incumbent_value,
                incumbent_variance,
                improvement,
                threshold,
                predicted_improvement: Some(predicted_improvement),
                agreement_ratio,
                trust_outcome,
            })
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn tell_absolute(
        &mut self,
        round: &Proposals,
        value: f32,
        variance: f32,
        incumbent_value: f32,
        incumbent_variance: f32,
        accept: bool,
        trust_incumbent: Option<f32>,
        trust_outcome: Option<TrustRegionOutcome>,
        reliability_evidence: Option<reliability_region::ReliabilityEvidence>,
        adapt_region: bool,
    ) -> Result<(), String> {
        self.last_tell_profile = None;
        self.check_round(round)?;
        if self.relative {
            return Err("Cannot mix absolute and paired-relative observations".into());
        }
        check_scores(&[value, incumbent_value], &[variance, incumbent_variance])?;
        let identity = self
            .observation
            .checked_add(1)
            .ok_or("Observation ID overflow")?;
        if self.implicit_history && self.history == MAX_HISTORY {
            return Err("Metal implicit history reached its 128-observation capacity".into());
        }
        let implicit_distances = if self.implicit_history {
            if round.history_distances.len() != self.history {
                return Err("Metal implicit history distance row has the wrong length".into());
            }
            Some(
                round
                    .history_distances
                    .iter()
                    .map(|&(previous_identity, distance)| {
                        self.identities[..self.history]
                            .iter()
                            .position(|&candidate| candidate == previous_identity)
                            .map(|previous| (previous, distance))
                            .ok_or_else(|| {
                                "Metal implicit history distance has an unknown identity"
                                    .to_string()
                            })
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            )
        } else {
            None
        };
        let slot = if self.resident_history == self.history_rows.len() {
            0
        } else {
            self.resident_history
        };
        let command = self.runtime.queue.new_command_buffer().to_owned();
        if accept {
            if !self.independent_fp16 {
                self.encode_ref(
                    &command,
                    Params {
                        seed: round.seed,
                        candidate: round.index as u32,
                        ..Params::default()
                    },
                );
            }
            let blit = command.new_blit_command_encoder();
            blit.copy_from_buffer(&self.proposal, 0, &self.base, 0, self.row_bytes());
            blit.end_encoding();
            if let Err(error) = finish(&command) {
                self.poisoned = true;
                return Err(error);
            }
            if let Err(error) = self.check_reference() {
                self.poisoned = true;
                return Err(error);
            }
        }
        std::mem::swap(&mut self.proposal, &mut self.history_rows[slot]);
        if self.profiling {
            let reference_ms = if accept {
                gpu_interval(&command)
                    .map(|(start, end)| ((end - start) * 1000.0) as f32)
                    .ok_or_else(|| "Metal did not expose a tell GPU interval".to_string())?
            } else {
                0.0
            };
            self.last_tell_profile = Some(TellProfile {
                reference_ms,
                history_copy_ms: 0.0,
                total_ms: reference_ms,
            });
        }
        self.observation = identity;
        if accept {
            self.base_id = identity;
        }
        self.resident_identities[slot] = identity;
        if self.resident_history == self.history_rows.len() {
            self.history_rows.rotate_left(1);
            self.resident_identities.rotate_left(1);
        } else {
            self.resident_history += 1;
        }
        if self.implicit_history {
            let next = self.history;
            self.outcomes[next] = value;
            self.variances[next] = variance;
            self.identities[next] = identity;
            for &(previous, distance) in implicit_distances.as_ref().unwrap() {
                self.pairwise_distances[next * MAX_HISTORY + previous] = distance;
                self.pairwise_distances[previous * MAX_HISTORY + next] = distance;
            }
            self.history += 1;
            if let Some(family) = &mut self.family {
                let components = round
                    .family_distances
                    .as_ref()
                    .ok_or("Missing family components")?;
                for (previous, &values) in components.iter().enumerate() {
                    family.components[next * MAX_HISTORY + previous] = values;
                    family.components[previous * MAX_HISTORY + next] = values;
                }
            }
            if self.history >= self.initial_observations {
                self.fit_implicit_enn()?;
            }
        } else {
            self.outcomes[slot] = value;
            self.variances[slot] = variance;
            self.identities[slot] = identity;
            if self.history == self.history_rows.len() {
                self.outcomes.rotate_left(1);
                self.variances.rotate_left(1);
                self.identities.rotate_left(1);
            } else {
                self.history += 1;
            }
        }
        self.best = if accept { value } else { incumbent_value };
        self.best_variance = if accept { variance } else { incumbent_variance };
        let update = if adapt_region && self.reliability.is_some() {
            reliability_evidence
                .ok_or_else(|| "Reliability control requires a model-aware tell".to_string())
                .and_then(|evidence| self.update_reliability(value, evidence))
        } else if adapt_region {
            match (trust_incumbent, trust_outcome) {
                (Some(incumbent), Some(outcome)) => {
                    self.update_region_outcome(value, incumbent, outcome)
                }
                (Some(incumbent), None) => self.update_region_noisy(value, incumbent),
                (None, None) => self.update_region(value),
                (None, Some(_)) => Err("Trust outcome requires an incumbent".into()),
            }
        } else {
            self.observed.push(f64::from(value));
            Ok(())
        };
        if let Err(error) = update {
            self.poisoned = true;
            return Err(error);
        }
        self.pending = None;
        self.queued = Some(accept);
        Ok(())
    }

    fn update_reliability(
        &mut self,
        value: f32,
        mut evidence: reliability_region::ReliabilityEvidence,
    ) -> Result<(), String> {
        self.observed.push(f64::from(value));
        evidence.center_radius = self.center_radius();
        let controller = self
            .reliability
            .as_mut()
            .ok_or("Reliability controller is not configured")?;
        controller.update(evidence)?;
        self.length = controller.length();
        Ok(())
    }

    fn fit_implicit_enn(&mut self) -> Result<(), String> {
        let n = self.history;
        let fit_family = n == self.initial_observations || (n - self.initial_observations) % 4 == 0;
        if fit_family {
            if let Some(family) = &mut self.family {
                let outcomes =
                    ndarray::Array2::from_shape_fn((n, 1), |(i, _)| f64::from(self.outcomes[i]));
                let variances =
                    ndarray::Array2::from_shape_fn((n, 1), |(i, _)| f64::from(self.variances[i]));
                let params = *self
                    .fitter
                    .as_ref()
                    .and_then(|f| f.params())
                    .ok_or("Family fit needs ENN parameters")?;
                family.fit(
                    &outcomes.view(),
                    &variances.view(),
                    params,
                    self.fit_samples,
                    self.fit_seed.wrapping_add(n as u64),
                    (self.distance_scaling == crate::config::DistanceScaling::SelfTuning)
                        .then_some(self.local_scale_neighbors),
                )?;
                self.apply_family_shape()?;
            }
        }
        let raw_distances = ndarray::Array2::from_shape_fn((n, n), |(row, column)| {
            f64::from(self.pairwise_distances[row * MAX_HISTORY + column])
        });
        let distances = if self.distance_scaling == crate::config::DistanceScaling::SelfTuning {
            crate::fit::self_tuned_distances(&raw_distances.view(), self.local_scale_neighbors)
                .map_err(|error| error.to_string())?
        } else {
            raw_distances
        };
        let outcomes =
            ndarray::Array2::from_shape_fn((n, 1), |(row, _)| f64::from(self.outcomes[row]));
        let variances =
            ndarray::Array2::from_shape_fn((n, 1), |(row, _)| f64::from(self.variances[row]));
        let mut rng = StdRng::seed_from_u64(self.fit_seed.wrapping_add(n as u64));
        let fitter = self
            .fitter
            .as_mut()
            .ok_or("Metal implicit ENN is not configured")?;
        let params = if self.fit_neighbors {
            fitter.ask_distances_adaptive_k(
                &distances.view(),
                &outcomes.view(),
                Some(&variances.view()),
                i32::try_from(self.initial_observations)
                    .map_err(|_| "Initial ENN neighbor bound does not fit i32")?,
                self.fit_candidates,
                self.fit_samples,
                None,
                &mut rng,
            )
        } else {
            fitter.ask_distances(
                &distances.view(),
                &outcomes.view(),
                Some(&variances.view()),
                self.fit_candidates,
                self.fit_samples,
                None,
                &mut rng,
            )
        }
        .map_err(|error| error.to_string())?;
        let y_scale = fitter.y_std()[0];
        self.fitted_enn = Some((
            usize::try_from(params.k_neighbors)
                .map_err(|_| "Fitted ENN neighbor count does not fit usize")?,
            params.epistemic_scale as f32,
            params.aleatoric_scale as f32,
            y_scale as f32,
        ));
        Ok(())
    }

    fn update_region(&mut self, value: f32) -> Result<(), String> {
        self.observed.push(f64::from(value));
        self.trust
            .update(
                &ndarray::ArrayView1::from(&self.observed),
                self.observed.len(),
            )
            .map_err(|error| error.to_string())?;
        if self.trust.needs_restart() {
            self.trust.restart();
            self.restart_count += 1;
            self.copy(&self.base, &self.history_rows[0])?;
            self.resident_history = 1;
            self.resident_identities[0] = self.base_id;
            if !self.implicit_history {
                self.outcomes[0] = self.best;
                self.variances[0] = self.best_variance;
                self.identities[0] = self.base_id;
                self.history = 1;
            }
            self.observed.clear();
            self.observed.push(f64::from(self.best));
            self.trust.set_watermark(0);
            self.trust
                .update(&ndarray::ArrayView1::from(&self.observed), 1)
                .map_err(|error| error.to_string())?;
        }
        self.length = self.trust.length();
        Ok(())
    }

    fn update_region_noisy(&mut self, value: f32, incumbent: f32) -> Result<(), String> {
        self.observed.push(f64::from(value));
        self.trust
            .update_history(
                &ndarray::ArrayView1::from(&self.observed),
                self.observed.len(),
                f64::from(incumbent),
            )
            .map_err(|error| error.to_string())?;
        if self.trust.needs_restart() {
            self.trust.restart();
            self.restart_count += 1;
            self.copy(&self.base, &self.history_rows[0])?;
            self.resident_history = 1;
            self.resident_identities[0] = self.base_id;
            if !self.implicit_history {
                self.outcomes[0] = self.best;
                self.variances[0] = self.best_variance;
                self.identities[0] = self.base_id;
                self.history = 1;
            }
            self.observed.clear();
            self.observed.push(f64::from(self.best));
            self.trust.set_watermark(0);
            self.trust
                .update_history(
                    &ndarray::ArrayView1::from(&self.observed),
                    1,
                    f64::from(self.best),
                )
                .map_err(|error| error.to_string())?;
        }
        self.length = self.trust.length();
        Ok(())
    }

    fn update_region_outcome(
        &mut self,
        value: f32,
        incumbent: f32,
        outcome: TrustRegionOutcome,
    ) -> Result<(), String> {
        self.observed.push(f64::from(value));
        let y_new = [f64::from(value)];
        self.trust
            .update_outcome(
                &ndarray::ArrayView1::from(&y_new),
                self.observed.len(),
                f64::from(incumbent),
                outcome,
            )
            .map_err(|error| error.to_string())?;
        if self.trust.needs_restart() {
            self.trust.restart();
            self.restart_count += 1;
            self.copy(&self.base, &self.history_rows[0])?;
            self.resident_history = 1;
            self.resident_identities[0] = self.base_id;
            if !self.implicit_history {
                self.outcomes[0] = self.best;
                self.variances[0] = self.best_variance;
                self.identities[0] = self.base_id;
                self.history = 1;
            }
            self.observed.clear();
            self.observed.push(f64::from(self.best));
            self.trust.set_watermark(0);
            let seed = [f64::from(self.best)];
            self.trust
                .update_outcome(
                    &ndarray::ArrayView1::from(&seed),
                    1,
                    f64::from(self.best),
                    TrustRegionOutcome::Inconclusive,
                )
                .map_err(|error| error.to_string())?;
        }
        self.length = self.trust.length();
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    /// Inconclusive rejections still update history. Set `reject_is_failure` only
    /// when the external screening rule counts a rejection toward contraction.
    pub fn tell_relative(
        &mut self,
        round: &Proposals,
        value: f32,
        variance: f32,
        incumbent_value: f32,
        incumbent_variance: f32,
        improvement: f32,
        improvement_variance: f32,
        accept: bool,
        reject_is_failure: bool,
    ) -> Result<(), String> {
        autoreleasepool(|| {
            self.tell_inner(
                round,
                value,
                variance,
                incumbent_value,
                incumbent_variance,
                improvement,
                improvement_variance,
                accept,
                reject_is_failure,
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn tell_inner(
        &mut self,
        round: &Proposals,
        value: f32,
        variance: f32,
        incumbent_value: f32,
        incumbent_variance: f32,
        improvement: f32,
        improvement_variance: f32,
        accept: bool,
        reject_is_failure: bool,
    ) -> Result<(), String> {
        self.check_round(round)?;
        if !self.relative {
            return Err("Enable legacy paired-relative mode before relative tells".into());
        }
        check_scores(
            &[value, incumbent_value, improvement],
            &[variance, incumbent_variance, improvement_variance],
        )?;
        let selected = self.pending.as_ref().unwrap();
        let radius = selected.length;
        let command = self.runtime.queue.new_command_buffer();
        if accept {
            self.encode_ref(
                command,
                Params {
                    seed: selected.seed,
                    candidate: selected.index as u32,
                    ..Params::default()
                },
            );
            let blit = command.new_blit_command_encoder();
            blit.copy_from_buffer(&self.proposal, 0, &self.base, 0, self.row_bytes());
            blit.copy_from_buffer(&self.proposal, 0, &self.anchor, 0, self.row_bytes());
            blit.end_encoding();
            if let Err(error) = finish(command) {
                self.poisoned = true;
                return Err(error);
            }
            if let Err(error) = self.check_reference() {
                self.poisoned = true;
                return Err(error);
            }
            self.best = value;
            self.best_variance = variance;
            self.outcomes.fill(0.0);
            self.variances.fill(0.0);
            self.history = 1;
            self.length = f64::from(radius);
            self.failures = 0;
        } else {
            let blit = command.new_blit_command_encoder();
            blit.copy_from_buffer(&self.proposal, 0, &self.rejected, 0, self.row_bytes());
            blit.end_encoding();
            if let Err(error) = finish(command) {
                self.poisoned = true;
                return Err(error);
            }
            self.best = incumbent_value;
            self.best_variance = incumbent_variance;
            self.outcomes[1] = improvement;
            self.variances[1] = improvement_variance;
            self.history = 2;
            if reject_is_failure {
                self.failures += 1;
                if self.failures == 4 {
                    self.length = (self.length * 0.5).max(self.length_config.length_min);
                    self.failures = 0;
                }
            }
        }
        self.pending = None;
        self.queued = Some(accept);
        Ok(())
    }

    pub fn sync(&mut self) -> Result<Vec<bool>, String> {
        self.check_healthy()?;
        Ok(self.queued.take().into_iter().collect())
    }

    pub fn describe(&self, round: &Proposals) -> Result<Vec<ProposalDescription>, String> {
        self.check_round(round)?;
        let p = self.pending.as_ref().unwrap();
        Ok(vec![(p.seed, p.score, p.length, p.changes.clone())])
    }

    pub fn geometry(&self, round: &Proposals) -> Result<Vec<(usize, f32)>, String> {
        self.check_round(round)?;
        let index = self.pending.as_ref().unwrap().index;
        Ok(vec![(
            index,
            if !self.independent_fp16 && index < 2 {
                0.75
            } else {
                0.0
            },
        )])
    }

    pub fn base_id(&self, round: &Proposals) -> Result<i64, String> {
        self.check_round(round)?;
        Ok(round.base_id)
    }

    pub fn history_dists(&self, round: &Proposals) -> Result<Vec<(i64, f32)>, String> {
        self.check_round(round)?;
        Ok(round.history_distances.clone())
    }

    pub fn pool(&self, round: &Proposals) -> Result<Vec<PoolDescription>, String> {
        self.check_round(round)?;
        Ok(round.pool.clone())
    }

    pub fn pool_geometry(&self, round: &Proposals) -> Result<PoolGeometry, String> {
        self.check_round(round)?;
        Ok((
            round.pool_radii.to_vec(),
            POOL_PAIRS
                .iter()
                .zip(round.pool_cosines)
                .map(|(&(left, right), cosine)| (left, right, cosine))
                .collect(),
            round.reference_cosines.to_vec(),
        ))
    }

    /// Current accepted BF16 weights, one contiguous row. Read-only lease required.
    pub fn base_buffer(&self) -> Buffer {
        self.base.clone()
    }
    pub fn best_buffer(&self) -> Buffer {
        self.base_buffer()
    }

    /// Selected BF16 row. A binding must keep its lease alive until GPU consumers finish.
    pub fn proposal_buffer(&self) -> Result<Buffer, String> {
        self.check_healthy()?;
        if self.pending.is_none() {
            return Err("No pending Metal BF16 proposal".into());
        }
        Ok(self.proposal.clone())
    }

    pub fn propose_buffer(&self, round: &Proposals) -> Result<Buffer, String> {
        self.check_round(round)?;
        self.proposal_buffer()
    }

    #[cfg(test)]
    pub(crate) fn test_candidate(&mut self, root: u64, index: usize) -> Result<Buffer, String> {
        self.check_idle()?;
        if index >= 4 {
            return Err("Metal BF16 candidate index must be below four".into());
        }
        self.ensure_ref()?;
        let command = self.runtime.queue.new_command_buffer();
        self.encode_proposal(command, self.params(root, index));
        finish(command)?;
        Ok(self.proposal.clone())
    }

    #[cfg(test)]
    pub(crate) fn test_profile(&mut self, root: u64, config: Ask) -> Result<[f32; 3], String> {
        self.check_idle()?;
        check_ask(config)?;
        self.ensure_ref()?;

        let start = Instant::now();
        let command = self.runtime.queue.new_command_buffer();
        self.encode_pool(command, self.pool_params(root));
        finish(command)?;
        let pool_ms = start.elapsed().as_secs_f32() * 1000.0;

        let start = Instant::now();
        let command = self.runtime.queue.new_command_buffer();
        self.encode_select(command, root, config, None);
        finish(command)?;
        let select_ms = start.elapsed().as_secs_f32() * 1000.0;

        let start = Instant::now();
        let command = self.runtime.queue.new_command_buffer();
        self.encode_row(command);
        finish(command)?;
        let row_ms = start.elapsed().as_secs_f32() * 1000.0;
        Ok([pool_ms, select_ms, row_ms])
    }

    /// Validates a proposal handle before the binding exports its row.
    pub fn check_round(&self, round: &Proposals) -> Result<(), String> {
        self.check_healthy()?;
        if !self
            .pending
            .as_ref()
            .is_some_and(|p| p.owner == round.owner && p.id == round.id)
        {
            return Err("Stale or foreign Metal BF16 proposal".into());
        }
        Ok(())
    }

    /// Optional writable-consumer isolation using existing pending storage, no sixth row.
    /// The binding must invalidate this snapshot before ask and lease it while exported.
    pub fn snapshot(&self) -> Result<Buffer, String> {
        self.check_idle()?;
        self.copy(&self.base, &self.proposal)?;
        Ok(self.proposal.clone())
    }

    pub fn read_best(&self) -> Result<Vec<u16>, String> {
        self.check_healthy()?;
        Ok(read(&self.base, self.dimensions))
    }

    /// Explicit model-sized validation copy; also triggers lazy reference initialization.
    pub fn read_reference(&mut self) -> Result<Vec<u16>, String> {
        self.check_healthy()?;
        self.ensure_ref()?;
        Ok(read(self.reference.as_ref().unwrap(), self.dimensions))
    }

    pub fn len(&self) -> usize {
        self.dimensions
    }
    pub fn is_empty(&self) -> bool {
        false
    }
    pub fn length(&self) -> Result<f64, String> {
        self.check_healthy()?;
        Ok(self.length)
    }
    pub fn best(&self) -> Result<f32, String> {
        self.check_healthy()?;
        if self.history == 0 {
            return Err("Initial Metal BF16 incumbent has not been measured".into());
        }
        Ok(self.best)
    }
    pub fn best_variance(&self) -> Result<f32, String> {
        self.check_healthy()?;
        if self.history == 0 {
            return Err("Initial Metal BF16 incumbent has not been measured".into());
        }
        Ok(self.best_variance)
    }
    pub fn history_len(&self) -> Result<usize, String> {
        self.check_healthy()?;
        Ok(self.history)
    }

    /// Start a fresh bounded ENN window from the current incumbent.
    pub(crate) fn compact_implicit_history(&mut self) -> Result<(), String> {
        self.check_idle()?;
        if !self.implicit_history || self.history < MAX_HISTORY {
            return Ok(());
        }
        self.copy(&self.base, &self.history_rows[0])?;
        self.resident_history = 1;
        self.resident_identities[0] = self.base_id;
        self.history = 1;
        self.outcomes[0] = self.best;
        self.variances[0] = self.best_variance;
        self.identities[0] = self.base_id;
        self.pairwise_distances.fill(0.0);
        if let Some(family) = &mut self.family {
            family.components.fill([0.0; FAMILIES]);
            family.weights = [1.0; FAMILIES];
        }
        self.apply_family_shape()?;
        self.fitted_enn = None;
        Ok(())
    }
    pub fn restarts(&self) -> Result<usize, String> {
        self.check_healthy()?;
        Ok(self.restart_count)
    }
    pub fn set_profiling(&mut self, enabled: bool) {
        self.profiling = enabled;
        if !enabled {
            self.last_profile = None;
        }
    }
    pub fn last_profile(&self) -> Option<AskProfile> {
        self.last_profile
    }
    pub fn last_tell_profile(&self) -> Option<TellProfile> {
        self.last_tell_profile
    }
    pub fn memory_info(&self) -> MemoryInfo {
        MemoryInfo {
            row_bytes: self.row_bytes(),
            resident_bytes: self.resident_bytes,
            max_buffer_length: self.runtime.device.max_buffer_length(),
            recommended_max_working_set_size: self
                .runtime
                .device
                .recommended_max_working_set_size(),
            current_allocated_size: self.runtime.device.current_allocated_size(),
        }
    }

    pub fn controller_info(&self) -> Result<ControllerInfo, String> {
        self.check_healthy()?;
        Ok(ControllerInfo {
            dimensions: self.dimensions,
            evaluated_arms: 1,
            length: self.length,
            length_min: self.length_config.length_min,
            length_max: self.length_config.length_max,
            success_tolerance: self.trust.success_tolerance(),
            failure_tolerance: self.trust.failure_tolerance(),
            success_counter: self.trust.success_counter(),
            failure_counter: self.trust.failure_counter(),
            restarts: self.restart_count,
        })
    }

    pub fn reliability_info(
        &self,
    ) -> Result<Option<reliability_region::ReliabilityTelemetry>, String> {
        self.check_healthy()?;
        Ok(self
            .reliability
            .as_ref()
            .map(reliability_region::ReliabilityController::telemetry))
    }

    fn row_bytes(&self) -> u64 {
        self.dimensions as u64 * 2
    }
    fn check_healthy(&self) -> Result<(), String> {
        if self.poisoned {
            Err("Metal BF16 state is unusable after a failed GPU update".into())
        } else {
            Ok(())
        }
    }
    fn check_idle(&self) -> Result<(), String> {
        self.check_healthy()?;
        if self.pending.is_some() || self.queued.is_some() || self.async_command.is_some() {
            Err("Tell the pending Metal BF16 proposal and sync before another mutation".into())
        } else {
            Ok(())
        }
    }
    #[cfg(test)]
    fn params(&self, root: u64, candidate: usize) -> Params {
        let radius = self.radius(candidate);
        Params {
            seed: candidate_seed(root, candidate),
            radius,
            alternate_radius: radius,
            candidate: candidate as u32,
            tiles: self.tiles.len() as u32,
            history: self.physical_history() as u32,
            mode: self.perturbation.shader(),
            ..Params::default()
        }
    }
    fn pool_params(&self, root: u64) -> Params {
        Params {
            seed: candidate_seed(root, 0),
            stream_seed: candidate_seed(root, 2),
            radius: self.radius(0),
            alternate_radius: self.radius(1),
            tiles: self.tiles.len() as u32,
            history: self.physical_history() as u32,
            mode: self.perturbation.shader(),
            ..Params::default()
        }
    }
    fn select_params(
        &self,
        root: u64,
        mut config: Ask,
        forced_candidate: Option<usize>,
    ) -> SelectionParams {
        if let Some((neighbors, epistemic, aleatoric, y_scale)) = self.fitted_enn {
            config.neighbors = neighbors.min(self.history);
            config.epistemic_scale = epistemic;
            config.aleatoric_scale = aleatoric;
            config.y_scale = y_scale;
        }
        let base_index = self.identities[..self.history]
            .iter()
            .position(|&identity| identity == self.base_id)
            .unwrap_or(0);
        let base_distances = std::array::from_fn(|i| {
            if self.implicit_history && i < self.history {
                self.pairwise_distances[base_index * MAX_HISTORY + i]
            } else {
                0.0
            }
        });
        let mut local_scales = [1.0; MAX_HISTORY];
        if self.distance_scaling == crate::config::DistanceScaling::SelfTuning && self.history > 1 {
            let mut values = Vec::with_capacity(self.history - 1);
            for (row, scale) in local_scales[..self.history].iter_mut().enumerate() {
                values.clear();
                values.extend(
                    (0..self.history)
                        .filter(|&column| column != row)
                        .map(|column| self.pairwise_distances[row * MAX_HISTORY + column]),
                );
                values.sort_by(f32::total_cmp);
                *scale = values[self.local_scale_neighbors.min(values.len()) - 1].max(1.0e-12);
            }
        }
        let resident_indices = std::array::from_fn(|i| {
            self.identities[..self.history]
                .iter()
                .position(|identity| {
                    i < self.resident_history && *identity == self.resident_identities[i]
                })
                .unwrap_or(0) as u32
        });
        SelectionParams {
            root_seed: root,
            outcomes: std::array::from_fn(|i| self.outcomes.get(i).copied().unwrap_or(0.0)),
            variances: std::array::from_fn(|i| self.variances.get(i).copied().unwrap_or(0.0)),
            draws: std::array::from_fn(|i| {
                let identity = if self.relative {
                    i as i64 + 1
                } else {
                    self.identities.get(i).copied().unwrap_or(0)
                };
                crate::hash::normal_metric(config.seed, identity, 0) as f32
            }),
            base_distances,
            local_scales,
            epistemic_scale: config.epistemic_scale,
            aleatoric_scale: config.aleatoric_scale,
            y_scale: config.y_scale,
            beta: config.beta,
            radius: self.radius(0),
            alternate_radius: self.radius(1),
            neighbors: config.neighbors as u32,
            history: self.history as u32,
            acquisition: match config.acquisition {
                AcquisitionKind::Ucb => 0,
                AcquisitionKind::Thompson => 1,
                AcquisitionKind::Pareto => 2,
            },
            tiles: self.tiles.len() as u32,
            mode: self.perturbation.shader(),
            resident_history: self.physical_history() as u32,
            resident_indices,
            implicit_history: u32::from(self.implicit_history),
            forced_candidate: forced_candidate.unwrap_or(4) as u32,
            distance_scaling: u32::from(
                self.distance_scaling == crate::config::DistanceScaling::SelfTuning,
            ),
            local_scale_neighbors: self.local_scale_neighbors as u32,
            incumbent_index: base_index as u32,
            candidate_floor: self
                .reliability
                .as_ref()
                .is_some_and(reliability_region::ReliabilityController::force_fresh)
                .then_some(2)
                .unwrap_or(0),
        }
    }

    fn physical_history(&self) -> usize {
        if self.implicit_history {
            self.resident_history
        } else {
            self.history
        }
    }
    fn radius(&self, candidate: usize) -> f32 {
        (self.length * if candidate & 1 == 0 { 0.5 } else { 2.0 })
            .clamp(self.length_config.length_min, self.length_config.length_max) as f32
    }
    fn ensure_ref(&mut self) -> Result<(), String> {
        autoreleasepool(|| self.init_reference())
    }

    fn init_reference(&mut self) -> Result<(), String> {
        if self.independent_fp16 {
            return Ok(());
        }
        let seed = self
            .reference_seed
            .ok_or("Enable correlated Metal sampling first")?;
        if self.reference.is_none() {
            preflight(&self.runtime, &[self.row_bytes()], self.row_bytes())?;
            self.reference = Some(self.runtime.buffer::<u16>(self.dimensions));
            let command = self.runtime.queue.new_command_buffer();
            self.encode_ref(
                command,
                Params {
                    seed,
                    initialize: 1,
                    ..Params::default()
                },
            );
            if let Err(error) = finish(command).and_then(|()| self.check_reference()) {
                self.reference = None;
                return Err(error);
            }
        }
        Ok(())
    }
    fn check_reference(&self) -> Result<(), String> {
        if read::<f32>(&self.reference_scales, self.blocks.len())
            .iter()
            .any(|x| !x.is_finite() || *x <= 0.0)
        {
            Err("Invalid Metal BF16 reference RMS".into())
        } else {
            Ok(())
        }
    }
    #[cfg(test)]
    fn encode_proposal(&self, command: &CommandBufferRef, params: Params) {
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.propose_pipeline);
        for (index, buffer) in [
            &self.base,
            &self.anchor,
            &self.rejected,
            self.reference.as_ref().unwrap_or(&self.base),
            &self.reference_scales,
            &self.leaves_gpu,
            &self.tiles_gpu,
            &self.proposal,
            &self.partials,
        ]
        .iter()
        .enumerate()
        {
            encoder.set_buffer(index as u64, Some(buffer), 0);
        }
        encoder.set_bytes(
            9,
            size_of::<Params>() as u64,
            (&params as *const Params).cast(),
        );
        encoder.dispatch_thread_groups(thread_group(self.tiles.len() as u64), thread_group(256));
        encoder.end_encoding();
    }
    fn encode_pool(&self, command: &CommandBufferRef, params: Params) {
        for pair in 0..self.resident_history.div_ceil(2) {
            let first = pair * 2;
            let params = Params {
                history: (self.resident_history - first).min(2) as u32,
                initialize: pair as u32,
                ..params
            };
            self.encode_pair(command, params, first);
        }
    }
    fn encode_pair(&self, command: &CommandBufferRef, params: Params, first: usize) {
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pool_pipeline);
        for (index, buffer) in [
            &self.base,
            &self.history_rows[first],
            &self.history_rows[(first + 1).min(self.resident_history - 1)],
            self.reference.as_ref().unwrap_or(&self.base),
            &self.reference_scales,
            &self.leaves_gpu,
            &self.tiles_gpu,
            &self.proposal,
            &self.partials,
            &self.pool_geometry,
        ]
        .iter()
        .enumerate()
        {
            encoder.set_buffer(index as u64, Some(buffer), 0);
        }
        encoder.set_bytes(
            10,
            size_of::<Params>() as u64,
            (&params as *const Params).cast(),
        );
        encoder.dispatch_thread_groups(thread_group(self.tiles.len() as u64), thread_group(256));
        encoder.end_encoding();
    }
    fn encode_select(
        &self,
        command: &CommandBufferRef,
        root: u64,
        config: Ask,
        forced_candidate: Option<usize>,
    ) {
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.selection_pipeline);
        for (index, buffer) in [
            &self.partials,
            &self.decision,
            &self.pool_distances,
            &self.pool_geometry,
        ]
        .iter()
        .enumerate()
        {
            encoder.set_buffer(index as u64, Some(buffer), 0);
        }
        let params = self.select_params(root, config, forced_candidate);
        encoder.set_bytes(
            4,
            size_of::<SelectionParams>() as u64,
            (&params as *const SelectionParams).cast(),
        );
        encoder.dispatch_thread_groups(thread_group(1), thread_group(1));
        encoder.end_encoding();
    }
    fn encode_row(&self, command: &CommandBufferRef) {
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.materialize_pipeline);
        for (index, buffer) in [
            &self.base,
            self.reference.as_ref().unwrap_or(&self.base),
            &self.reference_scales,
            &self.leaves_gpu,
            &self.tiles_gpu,
            &self.proposal,
            &self.decision,
        ]
        .iter()
        .enumerate()
        {
            encoder.set_buffer(index as u64, Some(buffer), 0);
        }
        encoder.dispatch_thread_groups(thread_group(self.tiles.len() as u64), thread_group(256));
        encoder.end_encoding();
    }
    fn encode_ref(&self, command: &CommandBufferRef, params: Params) {
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.reference_pipeline);
        for (index, buffer) in [
            self.reference.as_ref().unwrap(),
            &self.reference_scales,
            &self.leaves_gpu,
            &self.tiles_gpu,
            &self.reference_partials,
        ]
        .iter()
        .enumerate()
        {
            encoder.set_buffer(index as u64, Some(buffer), 0);
        }
        encoder.set_bytes(
            5,
            size_of::<Params>() as u64,
            (&params as *const Params).cast(),
        );
        encoder.dispatch_thread_groups(thread_group(self.tiles.len() as u64), thread_group(256));
        encoder.end_encoding();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.rms_pipeline);
        for (index, buffer) in [
            &self.reference_partials,
            &self.leaves_gpu,
            &self.offsets_gpu,
            &self.reference_scales,
        ]
        .iter()
        .enumerate()
        {
            encoder.set_buffer(index as u64, Some(buffer), 0);
        }
        encoder.dispatch_thread_groups(thread_group(self.blocks.len() as u64), thread_group(256));
        encoder.end_encoding();
    }
    fn copy(&self, source: &Buffer, destination: &Buffer) -> Result<(), String> {
        autoreleasepool(|| self.copy_inner(source, destination))
    }

    fn copy_inner(&self, source: &Buffer, destination: &Buffer) -> Result<(), String> {
        let command = self.runtime.queue.new_command_buffer();
        let blit = command.new_blit_command_encoder();
        blit.copy_from_buffer(source, 0, destination, 0, self.row_bytes());
        blit.end_encoding();
        finish(command)
    }
}

fn finish(command: &CommandBufferRef) -> Result<(), String> {
    command.commit();
    command.wait_until_completed();
    if command.status() != MTLCommandBufferStatus::Completed {
        Err(format!("Metal BF16 command failed: {:?}", command.status()))
    } else {
        Ok(())
    }
}

fn read<T: Copy>(buffer: &Buffer, count: usize) -> Vec<T> {
    assert!(std::mem::size_of::<T>() * count <= buffer.length() as usize);
    // All callers use initialized shared buffers after command completion.
    unsafe { std::slice::from_raw_parts(buffer.contents().cast::<T>(), count).to_vec() }
}

fn make_layout(
    blocks: &[ParamBlock],
    elements: usize,
) -> Result<(Vec<Tile>, Vec<u32>, Vec<Leaf>), String> {
    let mut covered = 0usize;
    let mut tiles = Vec::new();
    let mut offsets = vec![0u32];
    let mut leaves = Vec::new();
    for (index, block) in blocks.iter().enumerate() {
        ParamBlock::new(
            block.key,
            block.offset,
            block.len,
            block.scale,
            block.weight,
        )?;
        if block.offset != covered || block.len > u32::MAX as usize {
            return Err("BF16 blocks must be contiguous with each length fitting u32".into());
        }
        covered = covered
            .checked_add(block.len)
            .ok_or("BF16 layout overflow")?;
        let leaf = u32::try_from(index).map_err(|_| "too many BF16 blocks")?;
        for start in (0..block.len).step_by(TILE_ELEMENTS) {
            tiles.push(Tile {
                leaf,
                start: start as u32,
                length: (block.len - start).min(TILE_ELEMENTS) as u32,
                pad: 0,
            });
        }
        offsets.push(u32::try_from(tiles.len()).map_err(|_| "too many BF16 tiles")?);
        leaves.push(Leaf {
            key: block.key,
            offset: block.offset as u64,
            length: block.len as u64,
            scale: block.scale,
            weight: block.weight,
        });
    }
    if covered == 0 || covered != elements {
        return Err("BF16 blocks must cover the complete nonempty base".into());
    }
    Ok((tiles, offsets, leaves))
}

fn bytes<T>(count: usize) -> Result<u64, String> {
    count
        .checked_mul(size_of::<T>())
        .and_then(|x| u64::try_from(x).ok())
        .ok_or_else(|| "Metal BF16 allocation size overflow".into())
}

fn preflight(runtime: &Runtime, sizes: &[u64], additional: u64) -> Result<(), String> {
    check_memory(
        sizes,
        additional,
        runtime.device.max_buffer_length(),
        runtime.device.current_allocated_size(),
        runtime.device.recommended_max_working_set_size(),
    )
}

fn check_memory(
    sizes: &[u64],
    additional: u64,
    max_buffer: u64,
    current: u64,
    recommended: u64,
) -> Result<(), String> {
    if sizes.iter().any(|size| *size > max_buffer) {
        return Err(format!(
            "Metal BF16 allocation exceeds maxBufferLength={max_buffer}; each row must fit one buffer"
        ));
    }
    let total = current
        .checked_add(additional)
        .ok_or("Metal BF16 memory estimate overflow")?;
    if total > recommended {
        return Err(format!(
            "Metal BF16 requires {additional} additional bytes; currentAllocatedSize={current}, recommendedMaxWorkingSetSize={recommended}"
        ));
    }
    Ok(())
}

fn checked_length(length: TRLengthConfig) -> Result<TRLengthConfig, String> {
    if [length.length_min, length.length_init, length.length_max]
        .iter()
        .any(|value| !value.is_finite() || !(*value as f32).is_finite() || *value as f32 <= 0.0)
        || length.length_init < length.length_min
        || length.length_init > length.length_max
    {
        return Err(
            "trust-region radii must be ordered, positive and representable as FP32".into(),
        );
    }
    let mut min = length.length_min as f32;
    let mut max = length.length_max as f32;
    if f64::from(min) < length.length_min {
        min = min.next_up();
    }
    if f64::from(max) > length.length_max {
        max = max.next_down();
    }
    if !min.is_finite() || min >= max {
        return Err("trust-region bounds must contain two distinct FP32 radii".into());
    }
    let (min, max) = (f64::from(min), f64::from(max));
    let initial = length.length_init.clamp(min, max);
    Ok(TRLengthConfig::new(initial, min, max))
}

fn check_scores(values: &[f32], variances: &[f32]) -> Result<(), String> {
    if values.iter().any(|x| !x.is_finite()) || variances.iter().any(|x| !x.is_finite() || *x < 0.0)
    {
        Err("BF16 measurements must be finite with nonnegative finite variances".into())
    } else {
        Ok(())
    }
}

fn check_ask(config: Ask) -> Result<(), String> {
    if config.neighbors == 0
        || config.neighbors > MAX_HISTORY
        || [
            config.epistemic_scale,
            config.aleatoric_scale,
            config.y_scale,
        ]
        .iter()
        .any(|x| !x.is_finite() || *x < 0.0)
        || !config.beta.is_finite()
    {
        return Err("Metal BF16 acquisition requires neighbors in 1..=128, finite nonnegative scales and finite beta".into());
    }
    Ok(())
}

fn trial_hash(low: u32, high: u32, element: u32) -> u32 {
    let mut value = low ^ element.wrapping_mul(0x9e37_79b9);
    value ^= value >> 16;
    value = value.wrapping_mul(0x7feb_352d);
    value ^= high;
    value = value.wrapping_mul(0x846c_a68b);
    value ^ (value >> 15)
}

fn candidate_seed(root: u64, candidate: usize) -> u64 {
    let stream = (candidate / 2) as u32;
    let (low, high) = (root as u32, (root >> 32) as u32);
    u64::from(trial_hash(low, high, stream))
        | (u64::from(trial_hash(high, low, stream ^ 0x9e37_79b9)) << 32)
}

/// The same FP32 ENNX weighting and shared-per-observation Thompson draw as CUDA.
#[cfg(test)]
fn acquisition(distances: &[f32], outcomes: &[f32], variances: &[f32], config: Ask) -> f32 {
    let mut indices = (0..distances.len()).collect::<Vec<_>>();
    indices.sort_by(|a, b| distances[*a].total_cmp(&distances[*b]).then(a.cmp(b)));
    indices.truncate(config.neighbors.min(distances.len()));
    let y_scale_sq = (config.y_scale * config.y_scale).max(1.0e-12);
    let weight = |index: usize| {
        let variance = 1.0e-9
            + config.epistemic_scale * distances[index]
            + config.aleatoric_scale
            + variances[index] / y_scale_sq;
        1.0 / variance.max(1.0e-12)
    };
    let (mut sum, mut value, mut reference) = (0.0f32, 0.0f32, f32::MIN_POSITIVE);
    for &index in &indices {
        let w = weight(index);
        sum += w;
        value += w * outcomes[index];
        reference = reference.max(w);
    }
    let mean = value / sum.max(1.0e-12);
    let aleatoric = indices
        .iter()
        .map(|&index| {
            (weight(index) / sum.max(1.0e-12))
                * (config.aleatoric_scale + variances[index] / y_scale_sq)
        })
        .sum::<f32>();
    let se = (1.0 / sum.max(1.0e-12) + aleatoric).sqrt() * config.y_scale;
    match config.acquisition {
        AcquisitionKind::Thompson => {
            let (mut noise, mut squared) = (0.0f32, 0.0f32);
            for &index in &indices {
                let w = weight(index) / reference;
                // Capacity two pins anchor slot 1 and reuses rejected slot 2.
                noise += w * crate::hash::normal_metric(config.seed, (index + 1) as i64, 0) as f32;
                squared += w * w;
            }
            mean + se * (noise / squared.sqrt().max(1.0e-12))
        }
        AcquisitionKind::Pareto => mean + se,
        AcquisitionKind::Ucb => mean + config.beta * se,
    }
}

#[cfg(test)]
#[path = "bf16_audit.rs"]
mod audit;

#[cfg(test)]
mod tests {
    use super::*;

    fn decode_fp16(bits: u16) -> f64 {
        let exponent = (bits >> 10) & 31;
        let fraction = f64::from(bits & 1023);
        let value = match exponent {
            0 => fraction * 2.0f64.powi(-24),
            31 => f64::NAN,
            _ => (1024.0 + fraction) * 2.0f64.powi(i32::from(exponent) - 25),
        };
        if bits & 0x8000 == 0 { value } else { -value }
    }

    #[test]
    fn fp16_full_weights() {
        let first = TILE_ELEMENTS + 17;
        let base = vec![0x3400; first + 19];
        let blocks = vec![
            ParamBlock::new(71, 0, first, 0.25, 16.0 / first as f32).unwrap(),
            ParamBlock::new(93, first, 19, 0.5, 4.0 / 19.0).unwrap(),
        ];
        let mut search = SearchState::new_fp16(
            &base,
            blocks.clone(),
            2,
            TRLengthConfig::new(0.1, 0.001, 0.4),
            Perturbation::Gaussian,
        )
        .unwrap();
        assert!(search.correlate(7).is_err());
        search.observe_initial(-3.0, 0.0).unwrap();
        assert_eq!(search.controller_info().unwrap().dimensions, base.len());
        for step in 0..4 {
            let root = 12345 + step;
            let before = read::<u16>(&search.base, base.len()).to_vec();
            let mut candidates = Vec::new();
            for index in 0..4 {
                let row = search.test_candidate(root, index).unwrap();
                let values = read::<u16>(&row, base.len()).to_vec();
                for block in &blocks {
                    for j in 0..block.len {
                        let noise = normal(
                            candidate_seed(root, index) ^ 0x8ebc_6af0_9c88_c6e3,
                            block.key,
                            j,
                        );
                        let expected = decode_fp16(before[block.offset + j])
                            + f64::from((block.scale * search.radius(index)) * noise);
                        let actual = decode_fp16(values[block.offset + j]);
                        assert!(actual.is_finite());
                        assert!(
                            (actual - expected).abs() <= expected.abs() * 0.0005 + 2e-6,
                            "candidate={index} key={} coordinate={j} actual={actual} expected={expected}",
                            block.key
                        );
                    }
                }
                candidates.push(values);
            }
            let proposal = search.ask_round(1, 4, root, config()).unwrap();
            let selected = read::<u16>(&search.proposal, base.len()).to_vec();
            assert_eq!(selected, candidates[proposal.index]);
            assert!(search.reference.is_none());
            for (index, _, _, correlation, distances) in search.pool(&proposal).unwrap() {
                assert_eq!(correlation, 0.0);
                for (slot, (_, distance)) in distances.iter().enumerate() {
                    let history = read::<u16>(&search.history_rows[slot], base.len());
                    let exact: f64 = blocks
                        .iter()
                        .map(|block| {
                            (block.offset..block.offset + block.len)
                                .map(|j| {
                                    (decode_fp16(candidates[index][j]) - decode_fp16(history[j]))
                                        .powi(2)
                                        * f64::from(block.weight)
                                })
                                .sum::<f64>()
                        })
                        .sum();
                    close(f64::from(*distance), exact, 2e-6);
                }
            }
            let accept = step % 2 == 0;
            let reward = search.best().unwrap() + if accept { 1.0 } else { -1.0 };
            let decision = search.tell_noisy(&proposal, reward, 0.0).unwrap();
            assert_eq!(decision.accepted, accept);
            assert_eq!(
                &read::<u16>(&search.base, base.len()),
                if accept { &selected } else { &before }
            );
            assert_eq!(search.sync().unwrap(), vec![accept]);
        }
        let invalid = [0x7c00];
        assert!(
            SearchState::new_fp16(
                &invalid,
                vec![ParamBlock::new(0, 0, 1, 1.0, 1.0).unwrap()],
                2,
                TRLengthConfig::new(0.1, 0.001, 0.4),
                Perturbation::Gaussian,
            )
            .is_err()
        );
    }

    #[test]
    fn rademacher() {
        let base = vec![0x3400; 257];
        let blocks =
            vec![ParamBlock::new(71, 0, base.len(), 0.25, 16.0 / base.len() as f32).unwrap()];
        let mut search = SearchState::new_fp16(
            &base,
            blocks.clone(),
            2,
            TRLengthConfig::new(0.1, 0.001, 0.4),
            Perturbation::Rademacher,
        )
        .unwrap();
        search.observe_initial(-3.0, 0.0).unwrap();
        let root = 12345;
        let mut candidates = Vec::new();
        for index in 0..4 {
            let row = search.test_candidate(root, index).unwrap();
            let values = read::<u16>(&row, base.len()).to_vec();
            for (element, &value) in values.iter().enumerate() {
                let noise = sign(
                    candidate_seed(root, index) ^ 0x8ebc_6af0_9c88_c6e3,
                    blocks[0].key,
                    element,
                );
                let expected = decode_fp16(base[element])
                    + f64::from(blocks[0].scale * search.radius(index) * noise);
                let actual = decode_fp16(value);
                assert!((actual - expected).abs() <= expected.abs() * 0.0005 + 2e-6);
            }
            candidates.push(values);
        }
        let proposal = search.ask_round(1, 4, root, config()).unwrap();
        assert_eq!(
            read::<u16>(&search.proposal, base.len()),
            candidates[proposal.index]
        );
        assert_eq!(
            candidates[0]
                .iter()
                .zip(&base)
                .filter(|(left, right)| left != right)
                .count(),
            base.len()
        );
    }

    fn decode(bits: u16) -> f32 {
        f32::from_bits(u32::from(bits) << 16)
    }
    fn encode(value: f32) -> u16 {
        let bits = value.to_bits();
        (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
    }

    fn normal(seed: u64, key: u64, element: usize) -> f32 {
        let mix = crate::hash::splitmix64;
        let first = mix(seed
            ^ mix(key ^ 0xd6e8_feb8_6659_fd93)
            ^ mix((element as u64 / 2) ^ 0xa076_1d64_78bd_642f));
        let second = mix(first ^ 0xd2b7_4407_b1ce_6e93);
        let u1 = (((first >> 11) as f64 + 1.0) / 9007199254740992.0) as f32;
        let u2 = ((second >> 11) as f64 / 9007199254740992.0) as f32;
        let radius = (-2.0 * u1.clamp(1e-12, 0.99999994).ln()).sqrt();
        let angle = std::f32::consts::TAU * u2;
        radius
            * if element & 1 == 0 {
                angle.cos()
            } else {
                angle.sin()
            }
    }

    fn sign(seed: u64, key: u64, element: usize) -> f32 {
        let mix = crate::hash::splitmix64;
        let first = mix(seed
            ^ mix(key ^ 0xd6e8_feb8_6659_fd93)
            ^ mix((element as u64 / 2) ^ 0xa076_1d64_78bd_642f));
        if first & (1 << (element & 1)) == 0 {
            -1.0
        } else {
            1.0
        }
    }

    fn close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance * (1.0 + expected.abs()),
            "actual={actual}, expected={expected}, tolerance={tolerance}"
        );
    }

    fn rounding(actual: u16, expected: f32, allowance: f32) {
        let low = decode(encode(expected - allowance));
        let high = decode(encode(expected + allowance));
        let observed = decode(actual);
        assert!(
            observed.is_finite() && observed >= low && observed <= high,
            "BF16 {actual:04x} ({observed}) outside [{low}, {high}], raw={expected}"
        );
    }

    fn config() -> Ask {
        Ask {
            neighbors: 2,
            epistemic_scale: 0.7,
            aleatoric_scale: 0.05,
            acquisition: AcquisitionKind::Thompson,
            seed: 0xa817_9b4c_8820_9132,
            ..Ask::default()
        }
    }

    fn fixture(n: usize) -> (Vec<u16>, Vec<ParamBlock>) {
        let base = (0..n)
            .map(|i| encode(((i % 29) as f32 - 14.0) * 0.015625))
            .collect();
        let first = n / 3;
        let blocks = vec![
            ParamBlock::new(
                0x1234_5678_abcd_ef12,
                0,
                first,
                0.125,
                1.0 / (first as f32 * 0.125f32.powi(2)),
            )
            .unwrap(),
            ParamBlock::new(
                7,
                first,
                n - first,
                0.25,
                1.0 / ((n - first) as f32 * 0.25f32.powi(2)),
            )
            .unwrap(),
        ];
        (base, blocks)
    }

    fn state(base: &[u16], blocks: Vec<ParamBlock>) -> SearchState {
        let mut s = SearchState::new(
            base,
            -3.0,
            0.5,
            blocks,
            2,
            1,
            TRLengthConfig::new(0.03125, 0.001953125, 0.125),
        )
        .unwrap();
        assert!(s.reference.is_none());
        s.correlate(1234).unwrap();
        s.enable_relative(4).unwrap();
        s
    }

    fn scalar_dirs(s: &SearchState, reference: &[u16], seed: u64, index: usize) -> Vec<f32> {
        let mut result = vec![0.0; s.len()];
        for block in &s.blocks {
            let squared = reference[block.offset..block.offset + block.len]
                .iter()
                .map(|x| f64::from(decode(*x)).powi(2))
                .sum::<f64>();
            let inverse = (block.len as f64 / squared).sqrt() as f32;
            for element in 0..block.len {
                let noise = normal(seed ^ 0x8ebc_6af0_9c88_c6e3, block.key, element);
                let direction = if index < 2 {
                    0.75 * (decode(reference[block.offset + element]) * inverse)
                        + 0.4375f32.sqrt() * noise
                } else {
                    noise
                };
                result[block.offset + element] = direction;
            }
        }
        result
    }

    fn scalar_score(distances: &[f64], outcomes: &[f32], variances: &[f32], c: Ask) -> f64 {
        let ids: Vec<_> = (1..=distances.len() as i64).collect();
        score_ids(distances, outcomes, variances, &ids, c)
    }

    fn score_ids(
        distances: &[f64],
        outcomes: &[f32],
        variances: &[f32],
        ids: &[i64],
        c: Ask,
    ) -> f64 {
        let mut indices = (0..distances.len()).collect::<Vec<_>>();
        indices.sort_by(|a, b| distances[*a].total_cmp(&distances[*b]).then(a.cmp(b)));
        indices.truncate(c.neighbors.min(distances.len()));
        let y_scale = f64::from(c.y_scale);
        let y_scale_sq = (y_scale * y_scale).max(1e-12);
        let weights = indices
            .iter()
            .map(|&i| {
                1.0 / (1e-9
                    + f64::from(c.epistemic_scale) * distances[i]
                    + f64::from(c.aleatoric_scale)
                    + f64::from(variances[i]) / y_scale_sq)
                    .max(1e-12)
            })
            .collect::<Vec<_>>();
        let total = weights.iter().sum::<f64>();
        let mean = indices
            .iter()
            .zip(&weights)
            .map(|(&i, w)| w * f64::from(outcomes[i]))
            .sum::<f64>()
            / total;
        let aleatoric = indices
            .iter()
            .zip(&weights)
            .map(|(&i, weight)| {
                (weight / total)
                    * (f64::from(c.aleatoric_scale) + f64::from(variances[i]) / y_scale_sq)
            })
            .sum::<f64>();
        let se = (1.0 / total + aleatoric).sqrt() * y_scale;
        let multiple = match c.acquisition {
            AcquisitionKind::Ucb => f64::from(c.beta),
            AcquisitionKind::Pareto => 1.0,
            AcquisitionKind::Thompson => {
                indices
                    .iter()
                    .zip(&weights)
                    .map(|(&i, w)| {
                        w * f64::from(crate::hash::normal_metric(c.seed, ids[i], 0) as f32)
                    })
                    .sum::<f64>()
                    / weights.iter().map(|w| w * w).sum::<f64>().sqrt()
            }
        };
        mean + se * multiple
    }

    fn history_scores(s: &SearchState, root: u64, config: Ask) -> Vec<(f64, Vec<u16>)> {
        let rows: Vec<Vec<u16>> = s.history_rows[..s.history]
            .iter()
            .map(|row| read(row, s.len()))
            .collect();
        (0..4)
            .map(|candidate| {
                let command = s.runtime.queue.new_command_buffer();
                s.encode_proposal(command, s.params(root, candidate));
                finish(command).unwrap();
                let bits = read::<u16>(&s.proposal, s.len());
                let distances: Vec<f64> = rows
                    .iter()
                    .map(|row| {
                        s.blocks
                            .iter()
                            .map(|block| {
                                (block.offset..block.offset + block.len)
                                    .map(|i| {
                                        (f64::from(decode(bits[i])) - f64::from(decode(row[i])))
                                            .powi(2)
                                            * f64::from(block.weight)
                                    })
                                    .sum::<f64>()
                            })
                            .sum()
                    })
                    .collect();
                (
                    score_ids(
                        &distances,
                        &s.outcomes[..s.history],
                        &s.variances[..s.history],
                        &s.identities[..s.history],
                        config,
                    ),
                    bits,
                )
            })
            .collect()
    }

    #[test]
    fn fifo_parity() {
        autoreleasepool(|| {
            for (n, capacity) in [(17, 1), (17, 3), (65553, 5)] {
                let (base, blocks) = fixture(n);
                let bounds = TRLengthConfig::new(0.03125, 0.001953125, 0.125);
                let mut s =
                    SearchState::new(&base, -3.0, 0.5, blocks, capacity, 1, bounds).unwrap();
                s.correlate(1234).unwrap();
                s.ensure_ref().unwrap();
                let mut trust = TurboTrustRegion::new(n, bounds);
                trust.set_arms(1);
                let mut values = vec![-3.0];
                trust
                    .update(&ndarray::ArrayView1::from(&values), values.len())
                    .unwrap();
                let mut rows = std::collections::VecDeque::from([(base, -3.0f32, 0.5f32, 1i64)]);
                for step in 0..10 {
                    let config = Ask {
                        neighbors: 1 + step % capacity,
                        acquisition: [
                            AcquisitionKind::Ucb,
                            AcquisitionKind::Thompson,
                            AcquisitionKind::Pareto,
                        ][step % 3],
                        seed: 91,
                        ..Ask::default()
                    };
                    let expected = history_scores(&s, step as u64, config);
                    let round = s.ask_round(1, 4, step as u64, config).unwrap();
                    let best = expected
                        .iter()
                        .map(|v| v.0)
                        .fold(f64::NEG_INFINITY, f64::max);
                    close(f64::from(round.score), best, 5e-5);
                    let bits = read::<u16>(&s.proposal, n);
                    assert_eq!(bits, expected[round.index].1);
                    let value = if step < 4 { step as f32 } else { -4.0 };
                    let accept = step < 4;
                    s.tell_paired(&round, value, 0.25, s.best, s.best_variance, accept)
                        .unwrap();
                    assert_eq!(s.sync().unwrap(), vec![accept]);
                    values.push(f64::from(value));
                    trust
                        .update(&ndarray::ArrayView1::from(&values), values.len())
                        .unwrap();
                    assert_eq!(s.length().unwrap(), trust.length());
                    if rows.len() == capacity {
                        rows.pop_front();
                    }
                    rows.push_back((bits, value, 0.25, step as i64 + 2));
                    assert_eq!(s.history_len().unwrap(), rows.len());
                    for (i, (bits, value, variance, id)) in rows.iter().enumerate() {
                        assert_eq!(read::<u16>(&s.history_rows[i], n), *bits);
                        assert_eq!(
                            (s.outcomes[i], s.variances[i], s.identities[i]),
                            (*value, *variance, *id)
                        );
                    }
                }
            }
        });
    }

    #[test]
    fn restart_parity() {
        autoreleasepool(|| {
            let (base, blocks) = fixture(3);
            let bounds = TRLengthConfig::new(0.03125, 0.015625, 0.125);
            let mut s = SearchState::new(&base, 0.0, 0.1, blocks, 3, 1, bounds).unwrap();
            s.correlate(41).unwrap();
            let mut trust = TurboTrustRegion::new(3, bounds);
            trust.set_arms(1);
            let mut values = vec![0.0];
            trust
                .update(&ndarray::ArrayView1::from(&values), values.len())
                .unwrap();
            let mut restarts = 0;
            for step in 0..14 {
                let round = s.ask_round(1, 4, step, Ask::default()).unwrap();
                s.tell_paired(&round, -1.0, 0.2, 0.0, 0.1, false).unwrap();
                s.sync().unwrap();
                values.push(-1.0);
                trust
                    .update(&ndarray::ArrayView1::from(&values), values.len())
                    .unwrap();
                if trust.needs_restart() {
                    trust.restart();
                    trust.set_watermark(0);
                    values = vec![0.0];
                    trust
                        .update(&ndarray::ArrayView1::from(&values), values.len())
                        .unwrap();
                    restarts += 1;
                    assert_eq!(s.history_len().unwrap(), 1);
                    assert_eq!(read::<u16>(&s.history_rows[0], 3), base);
                    assert_eq!(s.identities[0], 1);
                    assert_eq!((s.outcomes[0], s.variances[0]), (0.0, 0.1));
                }
                assert_eq!(s.length().unwrap(), trust.length());
                assert_eq!(s.restarts().unwrap(), restarts);
            }
            assert!(restarts > 0);
        });
    }

    #[test]
    fn measured_startup_and_adaptation() {
        autoreleasepool(|| {
            let (base, blocks) = fixture(17);
            let bounds = TRLengthConfig::new(0.03125, 0.015625, 0.125);
            let mut s = SearchState::new_unscored(&base, blocks, 2, 1, bounds).unwrap();
            s.correlate(41).unwrap();
            s.set_failure_tolerance(4).unwrap();
            assert_eq!(s.history_len().unwrap(), 0);
            assert!(s.observed.is_empty());
            assert!(s.best().is_err());
            assert!(s.best_variance().is_err());
            assert!(s.begin_ask(1, 4, 123, Ask::default()).is_err());
            assert!(s.reference.is_none());
            assert!(s.async_command.is_none());
            assert!(s.observe_initial(f32::NAN, 0.0).is_err());
            assert!(s.observe_initial(-12.0, -1.0).is_err());
            assert!(s.observed.is_empty());

            s.observe_initial(-12.0, 0.125).unwrap();
            assert_eq!(s.history_len().unwrap(), 1);
            assert_eq!(s.best().unwrap(), -12.0);
            assert_eq!(s.best_variance().unwrap(), 0.125);
            assert_eq!(s.outcomes[0], -12.0);
            assert_eq!(s.variances[0], 0.125);
            assert_eq!(s.observed, [-12.0]);
            assert_eq!(s.trust.prev_obs(), 1);
            assert!(s.observe_initial(0.0, 0.0).is_err());

            for (step, value) in [-11.0, -10.0, -9.0].into_iter().enumerate() {
                let round = s.ask_round(1, 4, step as u64, Ask::default()).unwrap();
                s.tell_paired(&round, value, 0.0, s.best, s.best_variance, true)
                    .unwrap();
                assert_eq!(s.sync().unwrap(), [true]);
                assert_eq!(s.length().unwrap(), if step < 2 { 0.03125 } else { 0.0625 });
            }
            assert!(s.set_failure_tolerance(5).is_err());
            let accepted_weights = s.read_best().unwrap();
            for step in 0..16 {
                let round = s.ask_round(1, 4, 100 + step, Ask::default()).unwrap();
                s.tell_paired(&round, -13.0, 0.0, -9.0, 0.0, false).unwrap();
                assert_eq!(s.sync().unwrap(), [false]);
                assert_eq!(s.read_best().unwrap(), accepted_weights);
                assert_eq!(s.controller_info().unwrap().failure_tolerance, 4);
                assert_eq!(
                    s.controller_info().unwrap().failure_counter,
                    (step as i32 + 1) % 4
                );
                let expected = match step {
                    0..=2 => 0.0625,
                    3..=6 | 11..=14 => 0.03125,
                    _ => 0.015625,
                };
                assert_eq!(s.length().unwrap(), expected);
                if step == 11 {
                    assert_eq!(s.restarts().unwrap(), 1);
                    assert_eq!(s.history_len().unwrap(), 1);
                    assert_eq!(s.observed, [-9.0]);
                    assert_eq!(s.trust.prev_obs(), 1);
                    assert_eq!(s.outcomes[0], -9.0);
                }
            }
        });
    }

    #[test]
    fn one_observation_noisy_decision() {
        autoreleasepool(|| {
            let (base, blocks) = fixture(17);
            let bounds = TRLengthConfig::new(0.03125, 0.001953125, 0.125);
            let mut state = SearchState::new(&base, -3.0, 0.25, blocks, 2, 1, bounds).unwrap();
            state.correlate(41).unwrap();
            state.set_failure_tolerance(4).unwrap();

            let rejected_round = state.ask_round(1, 4, 100, Ask::default()).unwrap();
            let rejected_bits = read::<u16>(&state.proposal, state.len());
            let rejected = state.tell_noisy(&rejected_round, -1.0, 0.75).unwrap();
            assert_eq!(rejected.improvement, 2.0);
            assert_eq!(rejected.threshold, 2.0);
            assert!(!rejected.accepted);
            assert_eq!(state.sync().unwrap(), [false]);
            assert_eq!(state.best().unwrap(), -3.0);
            assert_eq!(state.best_variance().unwrap(), 0.25);
            assert_eq!(state.read_best().unwrap(), base);
            assert_eq!(state.history_len().unwrap(), 2);
            assert_eq!(
                read::<u16>(&state.history_rows[1], state.len()),
                rejected_bits
            );
            assert_eq!(state.controller_info().unwrap().failure_counter, 1);

            let accepted_round = state.ask_round(1, 4, 101, Ask::default()).unwrap();
            let accepted_bits = read::<u16>(&state.proposal, state.len());
            let accepted = state.tell_noisy(&accepted_round, 0.0, 0.0).unwrap();
            assert_eq!(accepted.improvement, 3.0);
            assert_eq!(accepted.threshold, 1.0);
            assert!(accepted.accepted);
            assert_eq!(state.sync().unwrap(), [true]);
            assert_eq!(state.best().unwrap(), 0.0);
            assert_eq!(state.best_variance().unwrap(), 0.0);
            assert_eq!(state.read_best().unwrap(), accepted_bits);
            assert_eq!(state.controller_info().unwrap().success_counter, 1);
            assert_eq!(state.observed, [-3.0, -1.0, 0.0]);
        });
    }

    #[test]
    fn model_aware_decision_preserves_radius_without_evidence() {
        autoreleasepool(|| {
            let (base, blocks) = fixture(17);
            let bounds = TRLengthConfig::new(0.03125, 0.001953125, 0.125);
            let mut state = SearchState::new(&base, -3.0, 0.25, blocks, 2, 1, bounds).unwrap();
            state.correlate(41).unwrap();
            state.set_failure_tolerance(4).unwrap();

            let mut round = state.ask_round(1, 4, 100, Ask::default()).unwrap();
            assert!(round.predicted_mean.is_finite());
            assert!(round.predicted_standard_error.is_finite());
            assert!(round.predicted_standard_error >= 0.0);
            assert!(round.incumbent_mean.is_finite());
            assert!(round.incumbent_standard_error.is_finite());
            assert!(round.incumbent_standard_error >= 0.0);

            round.predicted_mean = 0.0;
            round.predicted_standard_error = 1.0;
            round.incumbent_mean = 0.0;
            round.incumbent_standard_error = 1.0;
            let decision = state.tell_model_aware(&round, 0.0, 1.0).unwrap();
            assert!(!decision.accepted);
            assert_eq!(decision.trust_outcome, TrustRegionOutcome::Inconclusive);
            assert_eq!(decision.predicted_improvement, Some(0.0));
            assert_eq!(decision.agreement_ratio, None);
            assert_eq!(state.sync().unwrap(), [false]);
            assert_eq!(state.length().unwrap(), 0.03125);
            assert_eq!(state.controller_info().unwrap().success_counter, 0);
            assert_eq!(state.controller_info().unwrap().failure_counter, 0);
        });
    }

    #[test]
    fn wide_history() {
        autoreleasepool(|| {
            let (base, blocks) = fixture(17);
            let mut s = SearchState::new(
                &base,
                0.0,
                0.0,
                blocks,
                MAX_HISTORY,
                1,
                TRLengthConfig::new(0.01, 0.0001, 0.08),
            )
            .unwrap();
            s.correlate(41).unwrap();
            s.ensure_ref().unwrap();
            s.history = MAX_HISTORY;
            s.resident_history = MAX_HISTORY;
            for i in 0..MAX_HISTORY {
                let row: Vec<_> = base
                    .iter()
                    .map(|v| encode(decode(*v) + i as f32 / 512.0))
                    .collect();
                s.history_rows[i] = s.runtime.buffer_with(&row);
                s.outcomes[i] = (i % 7) as f32;
                s.variances[i] = (i % 3) as f32 / 10.0;
                s.identities[i] = i as i64 + 17;
            }
            let config = Ask {
                neighbors: MAX_HISTORY,
                acquisition: AcquisitionKind::Thompson,
                seed: 71,
                ..Ask::default()
            };
            let expected = history_scores(&s, 19, config);
            let round = s.ask_round(1, 4, 19, config).unwrap();
            let best = expected
                .iter()
                .map(|v| v.0)
                .fold(f64::NEG_INFINITY, f64::max);
            close(f64::from(round.score), best, 5e-5);
            assert_eq!(read::<u16>(&s.proposal, 17), expected[round.index].1);
        });
    }

    #[test]
    fn host_guards() {
        for capacity in [0, MAX_HISTORY + 1] {
            let (base, blocks) = fixture(3);
            assert!(
                SearchState::new(
                    &base,
                    0.0,
                    0.0,
                    blocks,
                    capacity,
                    1,
                    TRLengthConfig::new(0.01, 0.001, 0.1)
                )
                .is_err()
            );
        }
        for neighbors in [0, MAX_HISTORY + 1] {
            assert!(
                check_ask(Ask {
                    neighbors,
                    ..Ask::default()
                })
                .is_err()
            );
        }
        assert_eq!(size_of::<Leaf>(), 32);
        assert_eq!(size_of::<Tile>(), 16);
        assert_eq!(size_of::<Params>(), 48);
        assert_eq!(size_of::<SelectionParams>(), 2648);
        assert_eq!(size_of::<Partial>(), 20);
        assert_eq!(size_of::<Decision>(), 56);
        // Five 2.6 GB rows fit as separate buffers; a combined allocation does not.
        assert!(
            check_memory(
                &[2_600_000_000],
                13_000_000_000,
                3_000_000_000,
                1_000_000_000,
                18_000_000_000
            )
            .is_ok()
        );
        assert!(
            check_memory(
                &[13_000_000_000],
                13_000_000_000,
                3_000_000_000,
                0,
                18_000_000_000
            )
            .is_err()
        );
        assert!(
            check_memory(
                &[2_600_000_000],
                13_000_000_000,
                3_000_000_000,
                6_000_000_000,
                18_000_000_000
            )
            .is_err()
        );
        assert!(check_memory(&[1], u64::MAX, 100, 1, u64::MAX).is_err());
        for length in [
            TRLengthConfig::new(0.0, 0.0, 1.0),
            TRLengthConfig::new(1.0, 2.0, 3.0),
            TRLengthConfig::new(1.0, 1.0, 1.0),
            TRLengthConfig::new(1.0, 1e-100, 2.0),
        ] {
            assert!(checked_length(length).is_err());
        }
        let bounds = checked_length(TRLengthConfig::new(0.03, 0.001, 0.1)).unwrap();
        assert!(bounds.length_min >= 0.001 && bounds.length_max <= 0.1);
        for root in [0, 1, u64::MAX, 0x1234_5678_9abc_def0] {
            assert_eq!(candidate_seed(root, 0), candidate_seed(root, 1));
            assert_eq!(candidate_seed(root, 2), candidate_seed(root, 3));
            assert_ne!(candidate_seed(root, 0), candidate_seed(root, 2));
        }
        for acquisition_kind in [
            AcquisitionKind::Thompson,
            AcquisitionKind::Ucb,
            AcquisitionKind::Pareto,
        ] {
            for neighbors in [1, 2] {
                let c = Ask {
                    acquisition: acquisition_kind,
                    neighbors,
                    ..config()
                };
                for distances in [[0.0, 0.2], [1.0, 1.0], [0.01, 2.0]] {
                    // The farther observation has larger precision despite neighbor order.
                    let variances = [50.0, 0.0001];
                    let actual = acquisition(&distances, &[0.0, -0.125], &variances, c);
                    let expected =
                        scalar_score(&distances.map(f64::from), &[0.0, -0.125], &variances, c);
                    close(f64::from(actual), expected, 1e-6);
                }
            }
        }
    }

    #[test]
    fn controller_info() {
        let (base, blocks) = fixture(10);
        let mut state = SearchState::new(
            &base,
            -3.0,
            0.0,
            blocks,
            2,
            1,
            TRLengthConfig::new(0.01, 0.0001, 0.08),
        )
        .unwrap();
        state.correlate(41).unwrap();

        let info = state.controller_info().unwrap();
        assert_eq!(info.dimensions, 10);
        assert_eq!(info.evaluated_arms, 1);
        close(info.length, 0.01, f64::from(f32::EPSILON));
        close(info.length_min, 0.0001, f64::from(f32::EPSILON));
        close(info.length_max, 0.08, f64::from(f32::EPSILON));
        assert_eq!(info.success_tolerance, 3);
        assert_eq!(info.failure_tolerance, 10);
        assert_eq!(info.success_counter, 0);
        assert_eq!(info.failure_counter, 0);
        assert_eq!(info.restarts, 0);
    }

    #[test]
    fn pool_geometry() {
        autoreleasepool(|| {
            let (base, blocks) = fixture(771);
            let mut s = state(&base, blocks);
            let reference = s.read_reference().unwrap();
            for block in &s.blocks {
                for element in 0..block.len {
                    let raw = normal(1234 ^ 0xe703_7ed1_a0b4_28db, block.key, element);
                    rounding(
                        reference[block.offset + element],
                        raw,
                        16.0 * f32::EPSILON * (1.0 + raw.abs()),
                    );
                }
            }
            let scales = read::<f32>(&s.reference_scales, s.blocks.len());
            for (block, &scale) in s.blocks.iter().zip(&scales) {
                let square = reference[block.offset..block.offset + block.len]
                    .iter()
                    .map(|x| f64::from(decode(*x)).powi(2))
                    .sum::<f64>();
                close(f64::from(scale), (block.len as f64 / square).sqrt(), 2e-6);
            }
            let root = 0x8772_5981_91f0_cdd8;
            let mut pool = Vec::new();
            let mut scores = Vec::new();
            let mut candidate_distances = Vec::new();
            let mut expected_partials = Vec::new();
            for candidate in 0..4 {
                let p = s.params(root, candidate);
                let command = s.runtime.queue.new_command_buffer();
                s.encode_proposal(command, p);
                finish(command).unwrap();
                let actual = read::<u16>(&s.proposal, s.len());
                let direction = scalar_dirs(&s, &reference, p.seed, candidate);
                let mut distance = 0.0;
                for block in &s.blocks {
                    for element in block.offset..block.offset + block.len {
                        let delta = (block.scale * p.radius) * direction[element];
                        let raw = decode(base[element]) + delta;
                        rounding(
                            actual[element],
                            raw,
                            32.0 * f32::EPSILON
                                * (block.scale * p.radius * (1.0 + direction[element].abs())
                                    + decode(base[element]).abs()
                                    + delta.abs()),
                        );
                        distance += (f64::from(decode(actual[element]))
                            - f64::from(decode(base[element])))
                        .powi(2)
                            * f64::from(block.weight);
                    }
                }
                let partial = read::<Partial>(&s.partials, s.tiles.len() * 4);
                expected_partials.extend_from_slice(
                    &partial[candidate * s.tiles.len()..(candidate + 1) * s.tiles.len()],
                );
                let gpu_distance = partial
                    [candidate * s.tiles.len()..(candidate + 1) * s.tiles.len()]
                    .iter()
                    .map(|x| f64::from(x.anchor))
                    .sum::<f64>();
                close(gpu_distance, distance, 2e-6);
                candidate_distances.push(gpu_distance);
                scores.push(scalar_score(&[distance], &[0.0], &[0.0], config()));
                pool.push(actual);
            }
            let command = s.runtime.queue.new_command_buffer();
            s.encode_pool(command, s.pool_params(root));
            finish(command).unwrap();
            let actual_partials = read::<Partial>(&s.partials, s.tiles.len() * 4);
            for (actual, expected) in actual_partials.iter().zip(&expected_partials) {
                close(f64::from(actual.anchor), f64::from(expected.anchor), 2e-6);
                close(
                    f64::from(actual.rejected),
                    f64::from(expected.rejected),
                    2e-6,
                );
                close(f64::from(actual.squared), f64::from(expected.squared), 2e-6);
                assert_eq!(actual.changed, expected.changed);
                assert_eq!(actual.invalid, expected.invalid);
            }
            let selected = scores
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0;
            let round = s.ask_round(1, 4, root, config()).unwrap();
            assert_eq!(round.index, selected);
            assert_eq!(s.base_id(&round).unwrap(), 1);
            close(f64::from(round.score), scores[selected], 2e-6);
            let distances = s.history_dists(&round).unwrap();
            assert_eq!(distances.len(), 1);
            assert_eq!(distances[0].0, 1);
            close(
                f64::from(distances[0].1),
                candidate_distances[selected],
                2e-6,
            );
            let pool_descriptions = s.pool(&round).unwrap();
            assert_eq!(pool_descriptions.len(), 4);
            for (candidate, description) in pool_descriptions.iter().enumerate() {
                assert_eq!(description.0, candidate);
                assert_eq!(description.1, candidate_seed(root, candidate));
                assert_eq!(description.2, s.radius(candidate));
                assert_eq!(description.3, if candidate < 2 { 0.75 } else { 0.0 });
                assert_eq!(description.4.len(), 1);
                assert_eq!(description.4[0].0, 1);
                close(
                    f64::from(description.4[0].1),
                    candidate_distances[candidate],
                    2e-6,
                );
            }
            let (radii, cosines, reference_cosines) = s.pool_geometry(&round).unwrap();
            let deltas = pool
                .iter()
                .map(|candidate| {
                    s.blocks
                        .iter()
                        .flat_map(|block| {
                            (block.offset..block.offset + block.len).map(|element| {
                                (f64::from(decode(candidate[element]))
                                    - f64::from(decode(base[element])))
                                    * f64::from(block.weight).sqrt()
                            })
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            for (candidate, delta) in deltas.iter().enumerate() {
                let expected = delta.iter().map(|value| value * value).sum::<f64>().sqrt();
                close(f64::from(radii[candidate]), expected, 2e-6);
            }
            for (&(left, right), &(actual_left, actual_right, cosine)) in
                POOL_PAIRS.iter().zip(&cosines)
            {
                assert_eq!((actual_left, actual_right), (left, right));
                let dot = deltas[left]
                    .iter()
                    .zip(&deltas[right])
                    .map(|(a, b)| a * b)
                    .sum::<f64>();
                let expected = dot / (f64::from(radii[left]) * f64::from(radii[right]));
                close(f64::from(cosine.unwrap()), expected, 2e-6);
            }
            let reference_delta = s
                .blocks
                .iter()
                .zip(&scales)
                .flat_map(|(block, &scale)| {
                    let reference = &reference;
                    (block.offset..block.offset + block.len).map(move |element| {
                        f64::from(block.scale)
                            * f64::from(decode(reference[element]))
                            * f64::from(scale)
                            * f64::from(block.weight).sqrt()
                    })
                })
                .collect::<Vec<_>>();
            let reference_radius = reference_delta
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                .sqrt();
            for (candidate, delta) in deltas.iter().enumerate() {
                let dot = delta
                    .iter()
                    .zip(&reference_delta)
                    .map(|(left, right)| left * right)
                    .sum::<f64>();
                let expected = dot / (f64::from(radii[candidate]) * reference_radius);
                close(
                    f64::from(reference_cosines[candidate].unwrap()),
                    expected,
                    2e-6,
                );
            }
            assert_eq!(
                read::<u16>(&s.propose_buffer(&round).unwrap(), s.len()),
                pool[selected]
            );
            assert!(s.ask_round(1, 4, 0, config()).is_err());
            let description = s.describe(&round).unwrap().remove(0);
            for (block, &(changed, squared)) in s.blocks.iter().zip(&description.3) {
                let range = block.offset..block.offset + block.len;
                assert_eq!(
                    changed,
                    range
                        .clone()
                        .filter(|&i| base[i] != pool[selected][i])
                        .count() as u64
                );
                let expected = range
                    .map(|i| {
                        (f64::from(decode(base[i])) - f64::from(decode(pool[selected][i]))).powi(2)
                    })
                    .sum();
                close(squared, expected, 2e-6);
            }
            assert!(
                s.tell_relative(&round, f32::NAN, 0.0, 0.0, 0.0, 0.0, 0.0, false, true)
                    .is_err()
            );
            s.tell_relative(&round, 1e8, 0.1, 1e8, 0.2, -1e-6, 0.03125, false, true)
                .unwrap();
            assert_eq!(s.sync().unwrap(), vec![false]);
            assert_eq!(s.outcomes, [0.0, -1e-6]);
            assert_eq!(s.variances, [0.0, 0.03125]);
            assert_eq!(s.read_best().unwrap(), base);
            assert_eq!(s.read_reference().unwrap(), reference);
            assert!(s.check_round(&round).is_err());
            assert!(s.propose_buffer(&round).is_err());
            assert_eq!(s.length, 0.03125);
            for rejection in 2..=4 {
                let trial = s.ask_round(1, 4, root + rejection, config()).unwrap();
                let bits = read::<u16>(&s.proposal, s.len());
                s.tell_relative(&trial, -5.0, 0.2, -3.0, 0.1, -0.125, 0.01, false, true)
                    .unwrap();
                assert_eq!(s.sync().unwrap(), vec![false]);
                assert_eq!(s.history, 2);
                assert_eq!(read::<u16>(&s.rejected, s.len()), bits);
                assert_eq!(s.outcomes, [0.0, -0.125]);
                assert_eq!(s.length, if rejection == 4 { 0.015625 } else { 0.03125 });
            }
            assert_eq!(s.read_reference().unwrap(), reference);
            let trial = s.ask_round(1, 4, root + 10, config()).unwrap();
            let accepted = read::<u16>(&s.proposal, s.len());
            let direction = scalar_dirs(&s, &reference, trial.seed, trial.index);
            s.tell_relative(&trial, -7.0, 0.0625, -1.0, 0.5, 0.125, 0.02, true, true)
                .unwrap();
            assert_eq!(s.sync().unwrap(), vec![true]);
            assert_eq!(s.sync().unwrap(), Vec::<bool>::new());
            assert_eq!(s.best().unwrap(), -7.0); // Explicit decision, not cached absolute reward.
            assert_eq!(s.best_variance().unwrap(), 0.0625);
            assert_eq!(s.length().unwrap(), f64::from(trial.length));
            assert_eq!(s.history_len().unwrap(), 1);
            assert_eq!(s.failures, 0);
            assert_eq!(s.outcomes, [0.0; 2]);
            assert_eq!(s.variances, [0.0; 2]);
            assert_eq!(s.read_best().unwrap(), accepted);
            assert_eq!(read::<u16>(&s.anchor, s.len()), accepted);
            for (actual, raw) in s.read_reference().unwrap().into_iter().zip(direction) {
                rounding(actual, raw, 32.0 * f32::EPSILON * (1.0 + raw.abs()));
            }
            assert_eq!(s.restarts().unwrap(), 0);
        });
    }

    #[test]
    fn unclear_history() {
        let (base, blocks) = fixture(257);
        let mut s = state(&base, blocks);
        let reference = s.read_reference().unwrap();
        let initial_radius = s.length;
        // Inconclusive observations between harmful ones preserve, but do not add to,
        // the accumulated evidence for contraction.
        for i in 0..8 {
            let harmful = i % 2 == 0;
            let round = s.ask_round(1, 4, 100 + i, config()).unwrap();
            let bits = read::<u16>(&s.proposal, s.len());
            let improvement = if harmful { -0.5 } else { 0.01 };
            s.tell_relative(
                &round,
                -3.0 + improvement,
                0.1,
                -3.0,
                0.1,
                improvement,
                0.1,
                false,
                harmful,
            )
            .unwrap();
            assert_eq!(s.sync().unwrap(), vec![false]);
            assert_eq!(s.history, 2);
            assert_eq!(s.outcomes, [0.0, improvement]);
            assert_eq!(s.variances, [0.0, 0.1]);
            assert_eq!(read::<u16>(&s.rejected, s.len()), bits);
            assert_eq!(s.read_best().unwrap(), base);
            assert_eq!(s.read_reference().unwrap(), reference);
            assert_eq!(s.best().unwrap(), -3.0);
            assert_eq!(s.failures, ((i + 2) / 2) as usize % 4);
            assert_eq!(
                s.length,
                if i < 6 {
                    initial_radius
                } else {
                    initial_radius * 0.5
                }
            );
        }
        let round = s.ask_round(1, 4, 200, config()).unwrap();
        s.tell_relative(&round, -4.0, 0.0, -3.0, 0.0, -1.0, 0.0, false, true)
            .unwrap();
        s.sync().unwrap();
        assert_eq!(s.failures, 1);
        let round = s.ask_round(1, 4, 201, config()).unwrap();
        let selected_radius = f64::from(round.length);
        s.tell_relative(&round, -2.0, 0.0, -3.0, 0.0, 1.0, 0.0, true, false)
            .unwrap();
        assert_eq!(s.sync().unwrap(), vec![true]);
        assert_eq!(s.failures, 0);
        assert_eq!(s.history, 1);
        assert_eq!(s.length, selected_radius);
    }

    #[test]
    fn round_memory() {
        let base = vec![encode(1.0); 33];
        let blocks = vec![ParamBlock::new(7, 0, 33, 1e-20, 1.0).unwrap()];
        let mut s = state(&base, blocks);
        assert!(s.ask_round(2, 4, 0, config()).is_err());
        assert!(s.ask_round(1, 5, 0, config()).is_err());
        assert!(s.enable_relative(4).is_err());
        assert!(s.correlate(5).is_err());
        let mut warmed = 0;
        for i in 0..32 {
            let round = s.ask_round(1, 4, i, config()).unwrap();
            assert_eq!(round.index, 0);
            assert!(round.score.is_finite());
            assert_eq!(round.changes, vec![(0, 0.0)]);
            let (radii, cosines, reference_cosines) = s.pool_geometry(&round).unwrap();
            assert_eq!(radii, vec![0.0; 4]);
            assert!(cosines.iter().all(|entry| entry.2.is_none()));
            assert!(reference_cosines.iter().all(Option::is_none));
            assert_eq!(read::<u16>(&s.proposal, s.len()), base);
            s.tell_relative(&round, -3.0, 0.0, -3.0, 0.0, 0.0, 0.0, false, true)
                .unwrap();
            assert!(s.ask_round(1, 4, i, config()).is_err()); // sync consumes the event.
            assert_eq!(s.sync().unwrap(), vec![false]);
            if i == 3 {
                warmed = s.memory_info().current_allocated_size;
            }
        }
        let final_bytes = s.memory_info().current_allocated_size;
        eprintln!("Metal tiny rounds allocated bytes: warm={warmed}, final={final_bytes}");
        assert!(
            final_bytes <= warmed + 1024 * 1024,
            "per-round Metal resources retained"
        );
        assert_eq!(s.length, s.length_config.length_min);
        assert_eq!(s.history, 2);
    }

    #[test]
    fn tile_reference() {
        let n = TILE_ELEMENTS + 17;
        let base = vec![encode(0.125); n];
        let blocks = vec![ParamBlock::new(99, 0, n, 0.25, 1.0 / n as f32).unwrap()];
        let mut s = state(&base, blocks);
        let round = s.ask_round(1, 4, 4521, config()).unwrap();
        let actual = read::<u16>(&s.proposal, n);
        let expected = actual
            .iter()
            .map(|x| (f64::from(decode(*x)) - 0.125).powi(2))
            .sum::<f64>();
        close(round.changes[0].1, expected, 3e-6);
        assert_eq!(
            round.changes[0].0,
            actual.iter().filter(|x| **x != base[0]).count() as u64
        );
        let reference = s.read_reference().unwrap();
        let square = reference
            .iter()
            .map(|x| f64::from(decode(*x)).powi(2))
            .sum::<f64>();
        close(
            f64::from(read::<f32>(&s.reference_scales, 1)[0]),
            (n as f64 / square).sqrt(),
            2e-6,
        );
    }

    #[test]
    fn radial_select() {
        // Characterization of the current surrogate limitation, not a desired
        // invariant for future search algorithms. These run the actual kernels.
        let positive = (0..1000u64)
            .find(|&seed| crate::hash::normal_metric(seed, 1, 0) > 0.5)
            .unwrap();
        let negative = (0..1000u64)
            .find(|&seed| crate::hash::normal_metric(seed, 1, 0) < -0.5)
            .unwrap();
        for n in [1024, TILE_ELEMENTS + 17] {
            let (base, blocks) = fixture(n);
            let mut previous_distances = None;
            for (name, kind, seed, seek_farthest) in [
                ("ucb", AcquisitionKind::Ucb, positive, true),
                ("legacy-pareto", AcquisitionKind::Pareto, positive, true),
                (
                    "thompson-positive",
                    AcquisitionKind::Thompson,
                    positive,
                    true,
                ),
                (
                    "thompson-negative",
                    AcquisitionKind::Thompson,
                    negative,
                    false,
                ),
            ] {
                let mut s = state(&base, blocks.clone());
                let c = Ask {
                    acquisition: kind,
                    seed,
                    ..config()
                };
                let round = s.ask_round(1, 4, 42, c).unwrap();
                assert_eq!(s.history, 1);
                let partials = read::<Partial>(&s.partials, s.tiles.len() * 4);
                let distances: Vec<f32> = partials
                    .chunks_exact(s.tiles.len())
                    .map(|tiles| {
                        assert!(tiles.iter().all(|p| p.invalid == 0));
                        assert!(tiles.iter().any(|p| p.changed > 0));
                        tiles.iter().map(|p| p.anchor).sum()
                    })
                    .collect();
                if let Some(previous) = &previous_distances {
                    assert_eq!(&distances, previous);
                }
                previous_distances = Some(distances.clone());
                let target = distances
                    .iter()
                    .copied()
                    .reduce(|a, b| if seek_farthest { a.max(b) } else { a.min(b) })
                    .unwrap();
                let selected = distances[round.index];
                assert!((selected - target).abs() <= 1e-6 * target);
                let se: Vec<f32> = distances
                    .iter()
                    .map(|distance| {
                        (1e-9 + c.epistemic_scale * distance + c.aleatoric_scale).sqrt() * c.y_scale
                    })
                    .collect();
                eprintln!(
                    "resident audit: n={n}, {name}, selected={}, distances={distances:?}, SE={se:?}",
                    round.index
                );
            }
        }
    }
}
