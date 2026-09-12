//! Experimental ENNX APIs.
//!
//! This module is the staging area for unstable lower-level surface area.
//! Keep stable user-facing Rust entry points in [`crate::prelude`].

#[cfg(all(target_os = "macos", feature = "metal"))]
pub use crate::apple_gpu::{DeviceInfo as AppleGpuInfo, Target, device_info as apple_gpu_info};
#[cfg(all(target_os = "macos", feature = "metal"))]
pub use crate::bf16_metal::{
    ParamBlock as MetalParamBlock, Proposals as MetalProposals, SearchState as MetalSearchState,
};
#[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
pub use crate::bf16_search::{ParamBlock, Proposal, Proposals, SearchState};
pub use crate::dense::{
    DenseLeaf, DenseLinear, DenseResult, DenseTerm, DenseView, METAL_OPS, OPENCL_OPS, ParamBuffer,
    apply as apply_dense, dist2 as dense_dist2, linear as dense_linear,
    tensor_key as dense_tensor_key,
};
#[cfg(all(target_os = "macos", feature = "metal"))]
pub use crate::flame_metal::{
    FlameConfig as MetalFlameConfig, FlameEvaluator as MetalFlameEvaluator,
    memory_info as metal_flame_memory_info, upload as metal_flame_upload,
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
pub use crate::optimizer_factory::enn_tr;
pub use crate::quantization::{FP4_LUT, quantize_e2m1, quantize_int4};
#[cfg(all(target_os = "macos", feature = "metal"))]
pub use crate::qwen_metal::QwenEvaluator as MetalQwenEvaluator;
pub use crate::trials::{
    Ask as SearchConfig, BpannHistory, Center as SearchCenter, EncodingType, IndexedObservation,
    ObservationId,
};
pub use crate::weights::{
    AcquisitionKind, ComputeDevice, WeightBlock, WeightSelectConfig, WeightSelectResult,
    apply_sparse, blocks_words, draw_sparse, merge_values, missing_words, select_weights,
    sparse_union, sparse_xor, take_words,
};
#[cfg(all(feature = "native-flame", target_os = "linux", target_arch = "x86_64"))]
pub use ennx_cuda::flame::{FlameConfig, FlameEvaluator};

#[cfg(all(target_os = "macos", feature = "metal"))]
pub use crate::fbt_model::{
    GateUpProbe, RoundLatencyRecord, RoundLatencyStudy, run_gate_up_probe, run_round_study,
};
#[cfg(all(target_os = "macos", feature = "metal"))]
pub use crate::fbt_moe::{
    ActualBoResult, GroupedMoeProbe, run_grouped_moe_probe, run_grouped_moe_probe_with_dataset,
    run_pretrain,
};
