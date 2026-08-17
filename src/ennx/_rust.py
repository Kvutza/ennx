from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

try:
    from . import ennx_rust as _ext
except ImportError as exc:  # pragma: no cover - exercised when extension unavailable
    raise ImportError(
        "Rust extension submodule `ennx.ennx_rust` is not available"
    ) from exc


hypervolume_2d_max = _ext.hypervolume.hypervolume_2d_max
normal_hash_batch_multi_seed_fast = _ext.hash.normal_hash_batch_multi_seed_fast
standardize_y = _ext.util.standardize_y
pareto_front_2d_maximize = _ext.util.pareto_front_2d_maximize
calculate_sobol_indices = _ext.util.calculate_sobol_indices
sobol_sequence = _ext.util.sobol_sequence
arms_from_pareto_fronts = _ext.util.arms_from_pareto_fronts
quantize_int4 = _ext.util.quantize_int4
quantize_fp4_e2m1 = _ext.util.quantize_fp4_e2m1
set_config_path = _ext.util.set_config_path
ensure_config_file = _ext.util.ensure_config_file
EpistemicNearestNeighbors = _ext.model.EpistemicNearestNeighbors
ENNParams = _ext.model.ENNParams
ENNStatefulFitter = _ext.fit.ENNStatefulFitter
subsample_loglik = _ext.fit.subsample_loglik
Optimizer = _ext.optimizer.Optimizer
Telemetry = _ext.optimizer.Telemetry
MultiTrustRegion = _ext.optimizer.MultiTrustRegion
PackedSearch = _ext.optimizer.PackedSearch
PackedTurbo = _ext.optimizer.PackedTurbo
TurboTrial = _ext.optimizer.TurboTrial
BpannHistory = _ext.optimizer.BpannHistory
create_optimizer = _ext.optimizer.create_optimizer
create_optimizer_enn = _ext.optimizer.create_optimizer_enn
create_optimizer_enn_multi_tr = _ext.optimizer.create_optimizer_enn_multi_tr
create_optimizer_zero = _ext.optimizer.create_optimizer_zero
create_optimizer_lhd = _ext.optimizer.create_optimizer_lhd
dense_apply = _ext.optimizer.dense_apply
dense_dist2 = _ext.optimizer.dense_dist2
dense_linear = _ext.optimizer.dense_linear
DenseLinear = _ext.optimizer.DenseLinear
Bf16Tree = getattr(_ext.optimizer, "Bf16Tree", None)
Bf16Search = getattr(_ext.optimizer, "Bf16Search", None)
Bf16Trial = getattr(_ext.optimizer, "Bf16Trial", None)
Bf16View = getattr(_ext.optimizer, "Bf16View", None)
weight_int4_select_ucb = _ext.optimizer.weight_int4_select_ucb
weight_select_ucb = _ext.optimizer.weight_select_ucb
ModelPackage = _ext.experimental.ModelPackage
NativeKdaModel = getattr(_ext.experimental, "NativeKdaModel", None)
ResidentBoSession = _ext.experimental.ResidentBoSession


__all__ = [
    "BpannHistory",
    "Bf16Tree",
    "Bf16Search",
    "Bf16Trial",
    "Bf16View",
    "DenseLinear",
    "ENNParams",
    "ENNStatefulFitter",
    "EpistemicNearestNeighbors",
    "ModelPackage",
    "MultiTrustRegion",
    "NativeKdaModel",
    "Optimizer",
    "ResidentBoSession",
    "Telemetry",
    "PackedTurbo",
    "TurboTrial",
    "PackedSearch",
    "arms_from_pareto_fronts",
    "calculate_sobol_indices",
    "create_optimizer",
    "create_optimizer_enn",
    "create_optimizer_enn_multi_tr",
    "create_optimizer_lhd",
    "create_optimizer_zero",
    "dense_apply",
    "dense_dist2",
    "dense_linear",
    "ensure_config_file",
    "hypervolume_2d_max",
    "normal_hash_batch_multi_seed_fast",
    "pareto_front_2d_maximize",
    "quantize_fp4_e2m1",
    "quantize_int4",
    "set_config_path",
    "sobol_sequence",
    "standardize_y",
    "subsample_loglik",
    "weight_int4_select_ucb",
    "weight_select_ucb",
]
