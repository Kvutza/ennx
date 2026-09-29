//! Dense FP32-compute/BF16-storage correlated search on the shared Apple GPU runtime.
//!
//! Integration: expose this module under `cfg(all(target_os = "macos", feature = "metal"))`.
//! The binding must lease every exported Buffer, forbid mutation while leases exist,
//! and expose weights read-only. All commands here complete before returning. Buffer
//! clones keep allocations alive but do not prevent writes by another Metal consumer.
//! Absolute observations use bounded FIFO history and the shared CPU TuRBO controller.
//! The old two-row paired-relative experiment is available only by explicit opt-in.

use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::time::Instant;

use metal::objc::rc::autoreleasepool;
use metal::{
    Buffer, CommandBuffer, CommandBufferRef, ComputePipelineState, MTLCommandBufferStatus,
};

use crate::Perturbation;
use crate::apple_gpu::Runtime;
use crate::fitter::ENNFitter;
use crate::reliability_region;
use crate::trials::Ask;
use crate::trust_region::{TRLengthConfig, TrustRegionOutcome, TurboTrustRegion};
use crate::weights::AcquisitionKind;

const SOURCE: &str = include_str!("bf16_search.metal");
#[path = "bf16_metal/configuration.rs"]
mod configuration;
#[path = "bf16_metal/controller.rs"]
mod controller;
#[path = "bf16_metal/dispatch.rs"]
mod dispatch;
#[path = "bf16_metal/history.rs"]
mod history;
#[path = "bf16_metal/initialization.rs"]
mod initialization;
#[path = "bf16_metal/inspection.rs"]
mod inspection;
#[path = "bf16_metal/layout.rs"]
mod layout;
#[path = "bf16_metal/leaf.rs"]
mod leaf;
#[path = "bf16_metal/objective_acquisition.rs"]
mod objective_acquisition;
#[path = "bf16_metal/objectives.rs"]
mod objectives;
#[path = "bf16_metal/procedural.rs"]
mod procedural;
pub use objective_acquisition::{ObjectiveAcquisition, ObjectivePolicy};
#[path = "bf16_metal/bounds.rs"]
mod bounds;
#[path = "bf16_metal/observations.rs"]
mod observations;
#[path = "bf16_metal/propose.rs"]
mod propose;
#[path = "bf16_metal/publication.rs"]
mod publication;
pub use bounds::BoundReport;
#[path = "bf16_metal/axis.rs"]
mod axis;
#[path = "bf16_metal/relative.rs"]
mod relative;
#[path = "bf16_metal/reliability.rs"]
mod reliability;
#[path = "bf16_metal/selection.rs"]
mod selection;
#[path = "bf16_metal/shapes.rs"]
mod shapes;
#[path = "bf16_metal/state.rs"]
mod state;
#[path = "bf16_metal/threshold.rs"]
mod threshold;
#[path = "bf16_metal/ziggurat.rs"]
mod ziggurat;
use layout::*;
use leaf::*;
use shapes::*;
use ziggurat::*;
const TILE_ELEMENTS: usize = 65_536;
const MAX_HISTORY: usize = crate::objective_observation::OBJECTIVE_CAPACITY;
#[path = "bf16_family.rs"]
mod family;
use family::{FAMILIES, FamilyHistory};
#[path = "bf16_metric.rs"]
mod metric;
// Production decisions consume only the four proposal norms.  The remaining
// eleven full-vector dot products are retained in test builds as an oracle for
// the geometry tests, but must not tax every optimizer round.
const POOL_METRICS: usize = if cfg!(test) { 15 } else { 4 };
const POOL_PAIRS: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
static NEXT_OWNER: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProgramAddress(u16);

impl ProgramAddress {
    pub fn new(family: usize, layer: Option<usize>, expert: Option<usize>) -> Result<Self, String> {
        if family >= 32 || layer.is_some_and(|value| value >= 7) {
            return Err("program address exceeds its family or layer code".into());
        }
        let gray = |value: usize| value ^ (value >> 1);
        let family = gray(family) as u16;
        let layer = layer.map_or(7, gray) as u16;
        let expert = expert.map_or(15, |value| gray(value) % 15) as u16;
        Ok(Self(family | (layer << 5) | (expert << 8)))
    }

    fn opaque(key: u64) -> Self {
        Self((crate::hash::splitmix64(key) as u16) & 0x0fff)
    }

    fn bits(self) -> u16 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParamBlock {
    pub key: u64,
    pub tensor_id: u64,
    pub logical_start: u64,
    pub offset: usize,
    pub len: usize,
    pub scale: f32,
    pub weight: f32,
    pub address: ProgramAddress,
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
            tensor_id: key,
            logical_start: 0,
            offset,
            len,
            scale,
            weight,
            address: ProgramAddress::opaque(key),
        })
    }

    pub fn new_logical(
        tensor_id: u64,
        logical_start: u64,
        offset: usize,
        len: usize,
        scale: f32,
        weight: f32,
    ) -> Result<Self, String> {
        let key = crate::tensor_store::block_key(tensor_id, logical_start, len as u64)?;
        let mut block = Self::new(key, offset, len, scale, weight)?;
        block.tensor_id = tensor_id;
        block.logical_start = logical_start;
        Ok(block)
    }

    pub fn with_address(mut self, address: ProgramAddress) -> Self {
        self.address = address;
        self
    }
}

pub type ProposalDescription = (u64, f32, f32, Vec<(u64, f64)>);
pub type PoolDescription = (usize, u64, f32, f32, Vec<(i64, f32)>);
pub type PoolGeometry = (Vec<f32>, Vec<(usize, usize, Option<f32>)>, Vec<Option<f32>>);

/// Immutable selected metadata, independent of reusable GPU scratch storage.
#[derive(Clone, Debug)]
pub struct Proposals {
    pool_layout: crate::procedural_pool::ProceduralPool,
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
    bounds: Option<BoundReport>,
    axis_coordinate: Option<f32>,
    axis_value: Option<f32>,
    basis_seed: u64,
    direction_norm: f32,
    threshold_table: Option<crate::threshold::ThresholdTable>,
}

impl Proposals {
    pub fn arms(&self) -> usize {
        self.pool_layout.arms() as usize
    }

    pub fn bound_report(&self) -> Option<BoundReport> {
        self.bounds
    }

    /// Selected value for an optional scalar search axis.
    pub fn axis_value(&self) -> Option<f32> {
        self.axis_value
    }

    /// Logical perturbation identity. This does not identify a resident model
    /// buffer; only the selected proposal is materialized by the backend.
    pub fn identity(&self) -> crate::procedural_pool::CandidateIdentity {
        self.pool_layout.identity(self.index as u32).unwrap()
    }

    /// Borrow the logical pool metadata without allocating additional models
    /// or duplicating its history-distance vectors.
    pub fn procedural_candidates(
        &self,
    ) -> impl Iterator<Item = crate::procedural_pool::ProceduralCandidate<'_>> {
        self.pool
            .iter()
            .map(|candidate| crate::procedural_pool::ProceduralCandidate {
                identity: self.pool_layout.identity(candidate.0 as u32).unwrap(),
                seed: candidate.1,
                radius: candidate.2,
                reference_correlation: candidate.3,
                history_distances: &candidate.4,
            })
    }

    #[cfg(test)]
    pub(crate) fn pool_keys(&self) -> Vec<(usize, u64, f32)> {
        self.pool
            .iter()
            .map(|candidate| (candidate.0, candidate.1, candidate.2))
            .collect()
    }
}

impl SearchState {
    pub(crate) fn tensor_version(
        &self,
        parent: u64,
        proposal: &Proposals,
    ) -> Result<crate::tensor_store::TensorVersion, String> {
        if proposal.owner != self.owner || proposal.block_scales.len() != self.blocks.len() {
            return Err("proposal does not belong to this tensor search state".into());
        }
        let blocks = self
            .blocks
            .iter()
            .zip(&proposal.block_scales)
            .map(|(block, &scale)| {
                crate::tensor_store::TensorBlock::new(
                    block.tensor_id,
                    block.logical_start,
                    block.len as u64,
                    scale,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        if let Some(table) = &proposal.threshold_table {
            crate::tensor_store::TensorVersion::threshold(
                parent,
                proposal.seed,
                proposal.basis_seed,
                proposal.length,
                blocks,
                table.words(),
            )
        } else {
            crate::tensor_store::TensorVersion::perturb(
                parent,
                proposal.seed,
                proposal.length,
                self.perturbation.name(),
                blocks,
            )
        }
    }
}

/// Metal command-buffer GPU intervals for one resident proposal transaction.
///
/// The intervals cover procedural pool scoring, device-side selection, and
/// generation of the selected row from its compressed seed. They exclude host
/// waiting and never imply that all candidate weight vectors were materialized.
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
    replay_origin: Option<Buffer>,
    anchor: Buffer,
    rejected: Buffer,
    history_rows: Vec<Buffer>,
    proposal: Buffer,
    reference: Option<Buffer>,
    reference_scales: Buffer,
    reference_partials: Buffer,
    partials: Buffer,
    pool_aggregates: Buffer,
    decision: Buffer,
    pool_distances: Buffer,
    replay_steps: Buffer,
    replay_scales: Buffer,
    replay_partials: Buffer,
    replay_components: Buffer,
    pool_family_distances: Buffer,
    family_groups: Buffer,
    family_base_weights: Buffer,
    pool_geometry: Buffer,
    threshold_tables: Buffer,
    propose_pipeline: ComputePipelineState,
    initial_pipeline: ComputePipelineState,
    pool_pipeline: ComputePipelineState,
    reduction_pipeline: ComputePipelineState,
    replay_short_pipeline: ComputePipelineState,
    replay_pipeline: ComputePipelineState,
    replay_reduction_pipeline: ComputePipelineState,
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
    objective_history: crate::objective_observation::ObjectiveWindow,
    objective_selector: Option<objective_acquisition::ObjectiveSelector>,
    pool_layout: crate::procedural_pool::ProceduralPool,
    proposal_method: crate::procedural_pool::ProposalMethod,
    threshold: Option<threshold::ThresholdSearch>,
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
    metric_gpu: Option<metric::MetricGpu>,
    implicit_history: bool,
    exact_history: bool,
    latent_history: bool,
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
    tell_profile: Option<TellProfile>,
    axis: Option<axis::SearchAxis>,
}

#[cfg(test)]
#[path = "bf16_audit.rs"]
mod audit;
#[cfg(test)]
#[path = "bf16_metal/objective_tests.rs"]
mod objective_tests;

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
    fn fp16_weights() {
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

    #[test]
    fn rademacher_initial() {
        let base = vec![0x3400; TILE_ELEMENTS + 17];
        let blocks =
            vec![ParamBlock::new(71, 0, base.len(), 0.25, 16.0 / base.len() as f32).unwrap()];
        let mut search = SearchState::new_implicit(
            &base,
            blocks.clone(),
            1,
            TRLengthConfig::new(0.1, 0.001, 0.4),
            Perturbation::Rademacher,
        )
        .unwrap();
        search
            .configure_enn(crate::config::ResidentEnnConfig {
                pool: crate::procedural_pool::ProceduralPool::legacy(),
                proposal_method: crate::procedural_pool::ProposalMethod::Independent,
                ask: Ask {
                    neighbors: 2,
                    ..config()
                },
                num_candidates: 4,
                num_samples: 2,
                fit_neighbors: false,
                distance_scaling: crate::config::DistanceScaling::Global,
                history_geometry: crate::config::HistoryGeometry::Realized,
                local_scale_neighbors: 2,
            })
            .unwrap();
        search.observe_initial(-3.0, 0.0).unwrap();
        let root = 12_345;
        let candidate = 2;
        let row = search.begin_initial(root, candidate).unwrap();
        let proposal = search.finish_ask().unwrap();
        let values = read::<u16>(&row, base.len());
        assert_eq!(proposal.index, candidate);
        assert_eq!(proposal.seed, candidate_seed(root, candidate));
        for (element, &value) in values.iter().enumerate() {
            let noise = sign(
                candidate_seed(root, candidate) ^ 0x8ebc_6af0_9c88_c6e3,
                blocks[0].key,
                element,
            );
            let expected =
                decode_fp16(base[element]) + f64::from(blocks[0].scale * proposal.length * noise);
            let actual = decode_fp16(value);
            assert!((actual - expected).abs() <= expected.abs() * 0.0005 + 2e-6);
        }
        assert_eq!(proposal.changes[0].0, base.len() as u64);
    }

    #[test]
    fn latent_history() {
        let base = vec![0x3400; 4096];
        let blocks =
            vec![ParamBlock::new(71, 0, base.len(), 0.25, 16.0 / base.len() as f32).unwrap()];
        let mut search = SearchState::new_implicit(
            &base,
            blocks,
            1,
            TRLengthConfig::new(0.1, 0.001, 0.4),
            Perturbation::Rademacher,
        )
        .unwrap();
        search
            .configure_enn(crate::config::ResidentEnnConfig {
                pool: crate::procedural_pool::ProceduralPool::legacy(),
                proposal_method: crate::procedural_pool::ProposalMethod::Independent,
                ask: Ask {
                    neighbors: 3,
                    ..config()
                },
                num_candidates: 4,
                num_samples: 2,
                fit_neighbors: false,
                distance_scaling: crate::config::DistanceScaling::Global,
                history_geometry: crate::config::HistoryGeometry::Latent,
                local_scale_neighbors: 2,
            })
            .unwrap();
        search.observe_initial(0.0, 0.0).unwrap();

        search.begin_initial(12_345, 0).unwrap();
        let first = search.finish_ask().unwrap();
        let first_norm = search.latent_norm(first.index);
        close(
            f64::from(first.history_distances[0].1),
            f64::from(first_norm),
            1e-6,
        );
        assert!(!search.tell_initial(&first, -1.0, 0.0).unwrap().accepted);
        assert_eq!(search.sync().unwrap(), [false]);

        search.begin_initial(12_346, 1).unwrap();
        let second = search.finish_ask().unwrap();
        let second_norm = search.latent_norm(second.index);
        close(
            f64::from(second.history_distances[0].1),
            f64::from(second_norm),
            1e-6,
        );
        close(
            f64::from(second.history_distances[1].1),
            f64::from(first_norm + second_norm),
            1e-6,
        );
        assert!(second.changes.iter().map(|change| change.0).sum::<u64>() > 0);
    }

    #[test]
    fn rewarm_threshold() {
        let base = vec![0x3400; 4096];
        let blocks =
            vec![ParamBlock::new(71, 0, base.len(), 0.25, 16.0 / base.len() as f32).unwrap()];
        let mut search = SearchState::new_implicit(
            &base,
            blocks,
            1,
            TRLengthConfig::new(0.1, 0.001, 0.4),
            Perturbation::Rademacher,
        )
        .unwrap();
        search
            .configure_enn(crate::config::ResidentEnnConfig {
                pool: crate::procedural_pool::ProceduralPool::legacy(),
                proposal_method: crate::procedural_pool::ProposalMethod::Independent,
                ask: Ask {
                    neighbors: 3,
                    ..config()
                },
                num_candidates: 4,
                num_samples: 2,
                fit_neighbors: false,
                distance_scaling: crate::config::DistanceScaling::Global,
                history_geometry: crate::config::HistoryGeometry::Latent,
                local_scale_neighbors: 2,
            })
            .unwrap();
        search.observe_initial(0.0, 0.01).unwrap();
        search.observation = MAX_HISTORY as i64;

        search.begin_initial(12_345, 0).unwrap();
        let proposal = search.finish_ask().unwrap();
        let decision = search.tell_initial(&proposal, 0.1, 0.01).unwrap();
        assert!(!decision.accepted);
        close(
            decision.threshold,
            2.0 * (2.0 * f64::from(0.01f32)).sqrt(),
            1e-12,
        );
        assert_eq!(search.sync().unwrap(), [false]);
    }

    #[test]
    fn gaussian_initial() {
        let base = vec![0x3400; TILE_ELEMENTS + 17];
        let blocks =
            vec![ParamBlock::new(71, 0, base.len(), 0.25, 16.0 / base.len() as f32).unwrap()];
        let mut search = SearchState::new_implicit(
            &base,
            blocks.clone(),
            1,
            TRLengthConfig::new(0.1, 0.001, 0.4),
            Perturbation::Gaussian,
        )
        .unwrap();
        search
            .configure_enn(crate::config::ResidentEnnConfig {
                pool: crate::procedural_pool::ProceduralPool::legacy(),
                proposal_method: crate::procedural_pool::ProposalMethod::Independent,
                ask: Ask {
                    neighbors: 2,
                    ..config()
                },
                num_candidates: 4,
                num_samples: 2,
                fit_neighbors: false,
                distance_scaling: crate::config::DistanceScaling::Global,
                history_geometry: crate::config::HistoryGeometry::Realized,
                local_scale_neighbors: 2,
            })
            .unwrap();
        search.observe_initial(-3.0, 0.0).unwrap();
        let root = rand::random();
        let candidate = root as usize & 3;
        let row = search.begin_initial(root, candidate).unwrap();
        let proposal = search.finish_ask().unwrap();
        let values = read::<u16>(&row, base.len());
        assert_eq!(proposal.index, candidate);
        assert_eq!(proposal.seed, candidate_seed(root, candidate));
        for (element, &value) in values.iter().enumerate() {
            let noise = normal(
                candidate_seed(root, candidate) ^ 0x8ebc_6af0_9c88_c6e3,
                blocks[0].key,
                element,
            );
            let expected =
                decode_fp16(base[element]) + f64::from(blocks[0].scale * proposal.length * noise);
            let actual = decode_fp16(value);
            assert!(
                (actual - expected).abs() <= expected.abs() * 0.0005 + 2e-6,
                "coordinate={element} actual={actual} expected={expected}"
            );
        }
    }

    #[test]
    fn replay_archived() {
        let base: Vec<u16> = (0..4096)
            .map(|index| 0x3000 + index as u16 % 1024)
            .collect();
        let blocks = (0..4)
            .map(|group| {
                let scale = 0.25 * (group + 1) as f32;
                ParamBlock::new(
                    71 + group as u64,
                    group * 1024,
                    1024,
                    scale,
                    1.0 / (1024.0 * scale * scale),
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let mut search = SearchState::new_implicit(
            &base,
            blocks.clone(),
            1,
            TRLengthConfig::new(0.1, 0.001, 0.4),
            Perturbation::Rademacher,
        )
        .unwrap();
        search.enable_family(vec![0, 1, 2, 3]).unwrap();
        let ask = Ask {
            neighbors: 2,
            ..config()
        };
        search
            .configure_enn(crate::config::ResidentEnnConfig {
                pool: crate::procedural_pool::ProceduralPool::legacy(),
                proposal_method: crate::procedural_pool::ProposalMethod::Independent,
                ask,
                num_candidates: 4,
                num_samples: 2,
                fit_neighbors: false,
                distance_scaling: crate::config::DistanceScaling::Global,
                history_geometry: crate::config::HistoryGeometry::Realized,
                local_scale_neighbors: 2,
            })
            .unwrap();
        search.observe_initial(-3.0, 0.0).unwrap();
        let mut archive = vec![(1i64, base)];
        for step in 0..4u64 {
            let root = 12_345 + step;
            if step != 0 {
                let recipes = read::<ReplayStep>(&search.replay_steps, search.history);
                assert_eq!(
                    recipes[step as usize].accepted,
                    u32::from((step - 1) % 2 == 0)
                );
                assert!(search.exact_history);
                assert!(search.history > search.resident_history);
            }
            search.begin_ask(1, 4, root, ask).unwrap();
            let round = search.finish_ask().unwrap();
            let replayed = read::<u16>(&search.history_rows[0], search.len());
            let incumbent = read::<u16>(&search.base, search.len());
            if let Some(index) = replayed
                .iter()
                .zip(&incumbent)
                .position(|(actual, expected)| actual != expected)
            {
                panic!(
                    "step={step} replay lineage differs at {index}: actual={} expected={}",
                    replayed[index], incumbent[index]
                );
            }
            let selected = read::<u16>(&search.proposal, search.len());
            for candidate in 0..4 {
                search.diagnostic_row(&round, root, ask, candidate).unwrap();
                let row = read::<u16>(&search.proposal, search.len());
                for &(identity, actual) in &round.pool[candidate].4 {
                    let historical = &archive
                        .iter()
                        .find(|(stored, _)| *stored == identity)
                        .unwrap()
                        .1;
                    let expected = blocks
                        .iter()
                        .map(|block| {
                            (block.offset..block.offset + block.len)
                                .map(|index| {
                                    let delta =
                                        decode_fp16(row[index]) - decode_fp16(historical[index]);
                                    delta * delta * f64::from(block.weight)
                                })
                                .sum::<f64>()
                        })
                        .sum::<f64>() as f32;
                    assert!(
                        (actual - expected).abs() <= 1.0e-5 + 3.0e-3 * expected.abs(),
                        "step={step} candidate={candidate} identity={identity} actual={actual} expected={expected}"
                    );
                    if candidate == round.index {
                        let components = round.family_distances.as_ref().unwrap();
                        let history_index = round.pool[candidate]
                            .4
                            .iter()
                            .position(|&(stored, _)| stored == identity)
                            .unwrap();
                        for (group, block) in blocks.iter().enumerate() {
                            let expected = (block.offset..block.offset + block.len)
                                .map(|index| {
                                    let delta =
                                        decode_fp16(row[index]) - decode_fp16(historical[index]);
                                    delta * delta * f64::from(block.weight)
                                })
                                .sum::<f64>() as f32;
                            let actual = components[history_index][group];
                            assert!(
                                (actual - expected).abs() <= 1.0e-5 + 3.0e-3 * expected.abs(),
                                "family step={step} candidate={candidate} identity={identity} group={group} actual={actual} expected={expected}"
                            );
                        }
                    }
                }
            }
            search
                .diagnostic_row(&round, root, ask, round.index)
                .unwrap();
            assert_eq!(read::<u16>(&search.proposal, search.len()), selected);
            let accepted = step % 2 == 0;
            search
                .tell_paired(&round, -2.0 + step as f32, 0.0, search.best, 0.0, accepted)
                .unwrap();
            search.sync().unwrap();
            archive.push((search.observation, selected));
        }
    }

    #[test]
    fn ziggurat_moments() {
        let seed = rand::random();
        let count = 1usize << 18;
        let mut sum = 0.0f64;
        let mut squared = 0.0f64;
        let mut fourth = 0.0f64;
        for element in 0..count {
            let sample = f64::from(normal(seed, 71, element));
            sum += sample;
            squared += sample * sample;
            fourth += sample.powi(4);
        }
        let mean = sum / count as f64;
        let variance = squared / count as f64 - mean * mean;
        let fourth_moment = fourth / count as f64;
        assert!(mean.abs() < 0.01, "mean={mean}");
        assert!((variance - 1.0).abs() < 0.02, "variance={variance}");
        assert!(
            (fourth_moment - 3.0).abs() < 0.15,
            "fourth_moment={fourth_moment}"
        );
    }

    fn decode(bits: u16) -> f32 {
        f32::from_bits(u32::from(bits) << 16)
    }
    fn encode(value: f32) -> u16 {
        let bits = value.to_bits();
        (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
    }

    fn ziggurat_bits(seed: u64, key: u64, element: u64, attempt: u32, stream: u32) -> u32 {
        fn mix32(mut value: u32) -> u32 {
            value ^= value >> 16;
            value = value.wrapping_mul(0x7feb_352d);
            value ^= value >> 15;
            value = value.wrapping_mul(0x846c_a68b);
            value ^ (value >> 16)
        }
        let counter = (element as u32).wrapping_mul(0x9e37_79b9)
            ^ ((element >> 32) as u32).wrapping_mul(0x85eb_ca6b);
        let domain = seed as u32
            ^ ((seed >> 32) as u32).wrapping_mul(0xc2b2_ae35)
            ^ (key as u32).wrapping_mul(0x27d4_eb2f)
            ^ ((key >> 32) as u32).wrapping_mul(0x1656_67b1);
        mix32(
            domain ^ counter ^ attempt.wrapping_mul(0xd3a2_646c) ^ stream.wrapping_mul(0xfd70_46c5),
        )
    }

    fn ziggurat_uniform(bits: u32) -> f32 {
        (((bits >> 8) + 1) as f32 * 2.0f32.powi(-24)).min(0.999_999_94)
    }

    fn normal(seed: u64, key: u64, element: usize) -> f32 {
        let element = element as u64;
        for attempt in 0..u32::MAX {
            let bits = ziggurat_bits(seed, key, element, attempt, 0);
            let signed_sample = bits as i32;
            let layer = signed_sample as u32 as usize & (ZIGGURAT_LAYERS - 1);
            let sample = signed_sample as f32 * ZIGGURAT.widths[layer];
            if signed_sample.unsigned_abs() < ZIGGURAT.thresholds[layer] {
                return sample;
            }
            if layer == 0 {
                for tail in 0..u32::MAX - attempt {
                    let x = -ziggurat_uniform(ziggurat_bits(seed, key, element, attempt + tail, 1))
                        .ln()
                        / ZIGGURAT_R as f32;
                    let y = -ziggurat_uniform(ziggurat_bits(seed, key, element, attempt + tail, 2))
                        .ln();
                    if 2.0 * y >= x * x {
                        return if signed_sample < 0 {
                            -(ZIGGURAT_R as f32) - x
                        } else {
                            ZIGGURAT_R as f32 + x
                        };
                    }
                }
            } else {
                let uniform = ziggurat_uniform(ziggurat_bits(seed, key, element, attempt, 1));
                let density = ZIGGURAT.densities[layer]
                    + uniform * (ZIGGURAT.densities[layer - 1] - ZIGGURAT.densities[layer]);
                if density < (-0.5 * sample * sample).exp() {
                    return sample;
                }
            }
        }
        unreachable!("Ziggurat retry counter exhausted")
    }

    fn sign(seed: u64, key: u64, element: usize) -> f32 {
        fn mix32(mut value: u32) -> u32 {
            value ^= value >> 16;
            value = value.wrapping_mul(0x7feb_352d);
            value ^= value >> 15;
            value = value.wrapping_mul(0x846c_a68b);
            value ^ (value >> 16)
        }
        let pair = element as u64 / 2;
        let first = mix32(
            seed as u32
                ^ (seed >> 32) as u32
                ^ mix32(key as u32 ^ (key >> 32) as u32)
                ^ mix32(pair as u32),
        );
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
    fn family_recovers() {
        autoreleasepool(|| {
            let base = vec![0x3c00; 4096];
            let blocks = (0..4)
                .map(|group| {
                    ParamBlock::new(71 + group as u64, group * 1024, 1024, 0.25, 1.0 / 64.0)
                        .unwrap()
                })
                .collect();
            let groups = vec![0, 1, 2, 3];
            let bounds = TRLengthConfig::new(0.03125, 0.001953125, 0.125);
            let mut search =
                SearchState::new_implicit(&base, blocks, 1, bounds, Perturbation::Rademacher)
                    .unwrap();
            search.enable_family(groups).unwrap();
            search.observe_initial(-3.0, 0.5).unwrap();
            search.base_id = 73;
            search.compact_history().unwrap();
            assert_eq!(search.history, 1);
            assert_eq!(search.identities[0], 73);
            assert_eq!(search.resident_identities[0], 73);
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
    fn startup_adaptation() {
        autoreleasepool(|| {
            let (base, blocks) = fixture(17);
            let bounds = TRLengthConfig::new(0.03125, 0.015625, 0.125);
            let mut s = SearchState::new_unscored(&base, blocks, 2, 1, bounds).unwrap();
            s.correlate(41).unwrap();
            s.set_tolerance(4).unwrap();
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
            assert!(s.set_tolerance(5).is_err());
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
    fn single_decision() {
        autoreleasepool(|| {
            let (base, blocks) = fixture(17);
            let bounds = TRLengthConfig::new(0.03125, 0.001953125, 0.125);
            let mut state = SearchState::new(&base, -3.0, 0.25, blocks, 2, 1, bounds).unwrap();
            state.correlate(41).unwrap();
            state.set_tolerance(4).unwrap();

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
            let accepted_buffer: *const metal::BufferRef = &*state.proposal;
            let accepted = state.tell_noisy(&accepted_round, 0.0, 0.0).unwrap();
            assert_eq!(accepted.improvement, 3.0);
            assert_eq!(accepted.threshold, 1.0);
            assert!(accepted.accepted);
            assert_eq!(state.sync().unwrap(), [true]);
            assert_eq!(state.best().unwrap(), 0.0);
            assert_eq!(state.best_variance().unwrap(), 0.0);
            assert_eq!(state.read_best().unwrap(), accepted_bits);
            assert!(std::ptr::eq(accepted_buffer, &*state.base));
            assert_eq!(state.controller_info().unwrap().success_counter, 1);
            assert_eq!(state.observed, [-3.0, -1.0, 0.0]);
        });
    }

    #[test]
    fn unevidenced_radius() {
        autoreleasepool(|| {
            let (base, blocks) = fixture(17);
            let bounds = TRLengthConfig::new(0.03125, 0.001953125, 0.125);
            let mut state = SearchState::new(&base, -3.0, 0.25, blocks, 2, 1, bounds).unwrap();
            state.correlate(41).unwrap();
            state.set_tolerance(4).unwrap();

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
            let decision = state.tell_modeled(&round, 0.0, 1.0).unwrap();
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
    fn paired_gain() {
        autoreleasepool(|| {
            let (base, blocks) = fixture(17);
            let bounds = TRLengthConfig::new(0.03125, 0.015625, 0.125);
            let mut state = SearchState::new(&base, -3.0, 0.01, blocks, 2, 1, bounds).unwrap();
            state.correlate(41).unwrap();
            let round = state.ask_round(1, 4, 100, Ask::default()).unwrap();
            let decision = state
                .paired_modeled(&round, -2.9, 0.01, -3.0, 0.01, 0.5, 0.01)
                .unwrap();
            assert!(decision.accepted);
            assert_eq!(decision.improvement, 0.5);
            assert_eq!(state.sync().unwrap(), [true]);
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
        // Three 64-bit coordinates, two floats, and the typed program address.
        assert_eq!(size_of::<Leaf>(), 40);
        assert_eq!(size_of::<Tile>(), 16);
        assert_eq!(size_of::<Params>(), 64);
        assert_eq!(size_of::<ReplayStep>(), 24);
        // Includes the bounded scalar-axis history shared with the Metal ABI.
        assert_eq!(size_of::<SelectionParams>(), 3224);
        assert_eq!(size_of::<Partial>(), 20);
        assert_eq!(size_of::<Decision>(), 64);
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
            assert_eq!(
                round.identity(),
                crate::procedural_pool::CandidateIdentity {
                    arm: 0,
                    slot: selected as u32
                }
            );
            for (typed, legacy) in round.procedural_candidates().zip(&pool_descriptions) {
                assert_eq!(
                    typed.identity,
                    crate::procedural_pool::CandidateIdentity {
                        arm: 0,
                        slot: legacy.0 as u32
                    }
                );
                assert_eq!(typed.seed, legacy.1);
                assert_eq!(typed.radius, legacy.2);
                assert_eq!(typed.reference_correlation, legacy.3);
                assert_eq!(typed.history_distances, legacy.4);
            }
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
