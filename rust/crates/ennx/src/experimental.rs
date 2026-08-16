//! Experimental ENNX APIs.
//!
//! This module is the staging area for unstable lower-level surface area.
//! Keep stable user-facing Rust entry points in [`crate::prelude`].

#[cfg(all(target_os = "macos", feature = "metal"))]
pub use crate::apple_gpu::{device_info as apple_gpu_info, DeviceInfo as AppleGpuInfo};
pub use crate::dense::{
    apply as apply_dense, dist2 as dense_dist2, linear as dense_linear,
    tensor_key as dense_tensor_key, Bf16Tree, DenseLeaf, DenseLinear, DenseResult, DenseTerm,
    DenseView, METAL_OPS, OPENCL_OPS,
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
    MultiTrustRegionConfig, MultiTrustRegionState, ObservationDelta, Optimizer, ProgramTrial,
    RegionBatch, RegionCandidate, SharingPolicy, Telemetry,
};
pub use crate::optimizer_factory::create_optimizer_enn_multi_tr;
pub use crate::quantization::{quantize_fp4_e2m1, quantize_int4, FP4_E2M1_LUT};
pub use crate::trials::{
    Ask as WeightAsk, BpannHistory, Center as WeightCenter, EncodingType, IndexedObservation,
    Leaf as WeightLeaf, ObservationId, Search as WeightSearch, Trial as WeightTrial,
};
pub use crate::turbo_search::{TurboSearch, TurboTrial};
pub use crate::weights::{
    apply_sparse, blocks_for_words, draw_sparse, merge_values, missing_words, select_weights,
    sparse_union, sparse_xor, take_words, AcquisitionKind, ComputeBackend, WeightBlock,
    WeightSelectConfig, WeightSelectResult,
};
