//! Experimental ENNX APIs.
//!
//! This module is the staging area for unstable lower-level surface area.
//! Keep stable user-facing Rust entry points in [`crate::prelude`].

#[cfg(all(target_os = "macos", feature = "metal"))]
pub use crate::apple_gpu::{device_info as apple_gpu_info, DeviceInfo as AppleGpuInfo};
#[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
pub use crate::bf16_search::{ParamBlock, Proposal, Proposals, SearchState};
pub use crate::dense::{
    apply as apply_dense, dist2 as dense_dist2, linear as dense_linear,
    tensor_key as dense_tensor_key, DenseLeaf, DenseLinear, DenseResult, DenseTerm, DenseView,
    ParamBuffer, METAL_OPS, OPENCL_OPS,
};
#[cfg(all(target_os = "macos", feature = "metal"))]
pub use crate::forward_metal::{
    KdaMoeMetalArena, KdaMoeMetalExecutor, KdaMoeMetalKdaVectors, KdaMoeMetalMemory,
    KdaMoeMetalModel, KdaMoeMetalWeights,
};
pub use crate::forward_program::{
    ForwardEvaluator, ForwardOp, ForwardProgram, KdaControlRequest, KdaDispatch, KdaEncoder,
    KdaForwardRequest, KdaMoeDispatch, KdaMoeLayerRequest, KdaPackedLinear, KdaTensorLayout,
    KernelPlan, PackedAffinePlan, ResidentBoState, ResidentRound, WorkAxis, WorkGrid, WorkTile,
};
pub use crate::forward_weights::PackedModel;
pub use crate::knn::{KnnIndex, KnnPlan, KnnProfile};
pub use crate::optimizer::{
    MultiTrustRegionConfig, MultiTrustRegionState, ObservationDelta, Optimizer, RegionBatch,
    RegionCandidate, SharingPolicy, Telemetry,
};
pub use crate::optimizer_factory::create_optimizer_enn_multi_tr;
pub use crate::quantization::{quantize_fp4_e2m1, quantize_int4, FP4_E2M1_LUT};
pub use crate::trials::{
    Ask as SearchConfig, BpannHistory, Center as SearchCenter, EncodingType, IndexedObservation,
    Leaf as PackedLeaf, ObservationId, Search as PackedSearch, Trial as PackedTrial,
};
pub use crate::turbo_search::{PackedTurbo, TurboTrial};
pub use crate::weights::{
    apply_sparse, blocks_for_words, draw_sparse, merge_values, missing_words, select_weights,
    sparse_union, sparse_xor, take_words, AcquisitionKind, ComputeDevice, WeightBlock,
    WeightSelectConfig, WeightSelectResult,
};
