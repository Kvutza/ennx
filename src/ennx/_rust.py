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


hypervolume2d_max = _ext.hypervolume.hypervolume2d_max
normal_hash = _ext.hash.normal_hash
standardize_y = _ext.util.standardize_y
pareto2d_max = _ext.util.pareto2d_max
sobol_indices = _ext.util.sobol_indices
sobol_sequence = _ext.util.sobol_sequence
pareto_arms = _ext.util.pareto_arms
quantize_int4 = _ext.util.quantize_int4
quantize_e2m1 = _ext.util.quantize_e2m1
set_path = _ext.util.set_path
ensure_config_file = _ext.util.ensure_config_file
ENN = _ext.model.ENN
ENNParams = _ext.model.ENNParams
ENNStatefulFitter = _ext.fit.ENNStatefulFitter
subsample_loglik = _ext.fit.subsample_loglik
Optimizer = _ext.optimizer.Optimizer
Telemetry = _ext.optimizer.Telemetry
MultiTrustRegion = _ext.optimizer.MultiTrustRegion
BpannHistory = _ext.optimizer.BpannHistory
create_optimizer = _ext.optimizer.create_optimizer
enn_optimizer = _ext.optimizer.enn_optimizer
enn_tr = _ext.optimizer.enn_tr
create_zero = _ext.optimizer.create_zero
create_lhd = _ext.optimizer.create_lhd
dense_apply = _ext.optimizer.dense_apply
dense_dist2 = _ext.optimizer.dense_dist2
dense_linear = _ext.optimizer.dense_linear
DenseLinear = _ext.optimizer.DenseLinear
ParamBuffer = getattr(_ext.optimizer, "ParamBuffer", None)
ParamBlock = getattr(_ext.optimizer, "ParamBlock", None)
SearchState = getattr(_ext.optimizer, "SearchState", None)
Proposals = getattr(_ext.optimizer, "Proposals", None)
weight_int4_select_ucb = _ext.optimizer.weight_int4_select_ucb
weight_select_ucb = _ext.optimizer.weight_select_ucb
ModelPackage = _ext.experimental.ModelPackage
NativeKdaModel = getattr(_ext.experimental, "NativeKdaModel", None)
ResidentBoSession = _ext.experimental.ResidentBoSession


__all__ = [
    "BpannHistory",
    "ParamBuffer",
    "ParamBlock",
    "SearchState",
    "Proposals",
    "DenseLinear",
    "ENNParams",
    "ENNStatefulFitter",
    "ENN",
    "ModelPackage",
    "MultiTrustRegion",
    "NativeKdaModel",
    "Optimizer",
    "ResidentBoSession",
    "Telemetry",
    "pareto_arms",
    "sobol_indices",
    "create_optimizer",
    "enn_optimizer",
    "enn_tr",
    "create_zero",
    "create_lhd",
    "dense_apply",
    "dense_dist2",
    "dense_linear",
    "ensure_config_file",
    "hypervolume2d_max",
    "normal_hash",
    "pareto2d_max",
    "quantize_e2m1",
    "quantize_int4",
    "set_path",
    "sobol_sequence",
    "standardize_y",
    "subsample_loglik",
    "weight_int4_select_ucb",
    "weight_select_ucb",
]
