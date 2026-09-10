from __future__ import annotations

import sys

from .._lazy import module_attr

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "ModelPackage": (".._rust", "ModelPackage"),
    "NativeKdaModel": (".._rust", "NativeKdaModel"),
    "ResidentBoSession": (".._rust", "ResidentBoSession"),
    "Optimizer": (".._rust", "Optimizer"),
    "Telemetry": (".._rust", "Telemetry"),
    "MultiTrustRegion": (".._rust", "MultiTrustRegion"),
    "BpannHistory": (".._rust", "BpannHistory"),
    "ParamBuffer": (".._rust", "ParamBuffer"),
    "ParamBlock": (".._rust", "ParamBlock"),
    "SearchState": (".._rust", "SearchState"),
    "Proposals": (".._rust", "Proposals"),
    "SharingPolicy": (".mtrregn", "SharingPolicy"),
    "RegionBatch": (".mtrregn", "RegionBatch"),
    "RegionCandidate": (".mtrregn", "RegionCandidate"),
    "CandidateProposal": (".mtrregn", "CandidateProposal"),
    "RegionRound": (".mtrregn", "RegionRound"),
    "MultiTrustRegionLoop": (".mtrregn", "MultiTrustRegionLoop"),
    "make_region": (".mtrregn", "make_region"),
    "allocate_batches": (".mtrregn", "allocate_batches"),
    "select_candidates": (".mtrregn", "select_candidates"),
    "mtrregn": (".mtrregn", "mtrregn"),
    "enn_optimizer": (".._rust", "enn_optimizer"),
    "enn_tr": (".._rust", "enn_tr"),
    "create_optimizer": (".._rust", "create_optimizer"),
    "create_zero": (".._rust", "create_zero"),
    "create_lhd": (".._rust", "create_lhd"),
    "weight_int4_select_ucb": (".._rust", "weight_int4_select_ucb"),
    "weight_select_ucb": (".._rust", "weight_select_ucb"),
    "dense_apply": (".._rust", "dense_apply"),
    "dense_dist2": (".._rust", "dense_dist2"),
    "dense_linear": (".._rust", "dense_linear"),
    "DenseLinear": (".._rust", "DenseLinear"),
    "quantize_int4": ("..quantization", "quantize_int4"),
    "quantize_e2m1": ("..quantization", "quantize_e2m1"),
}

experimental = sys.modules[__name__]


def turbo_enn(
    base: object,
    base_value: float,
    blocks: list[object],
    capacity: int,
    *,
    max_pending: int = 1,
    base_variance: float = 0.0,
    length_init: float = 0.8,
    length_min: float = 0.0078125,
    length_max: float = 1.6,
) -> object:
    search_type = __getattr__("SearchState")
    if search_type is None:
        raise RuntimeError("turbo_enn requires the CUDA wheel")
    return search_type(
        base,
        base_value,
        blocks,
        capacity,
        max_pending=max_pending,
        base_variance=base_variance,
        length_init=length_init,
        length_min=length_min,
        length_max=length_max,
    )


def __getattr__(name: str):
    return module_attr(name, globals())


__all__: list[str] = [
    "BpannHistory",
    "CandidateProposal",
    "DenseLinear",
    "ModelPackage",
    "MultiTrustRegion",
    "MultiTrustRegionLoop",
    "NativeKdaModel",
    "Optimizer",
    "ParamBlock",
    "ParamBuffer",
    "Proposals",
    "RegionBatch",
    "RegionCandidate",
    "RegionRound",
    "ResidentBoSession",
    "SearchState",
    "SharingPolicy",
    "Telemetry",
    "allocate_batches",
    "enn_optimizer",
    "enn_tr",
    "create_zero",
    "create_lhd",
    "create_optimizer",
    "dense_apply",
    "dense_dist2",
    "dense_linear",
    "experimental",
    "make_region",
    "mtrregn",
    "quantize_e2m1",
    "quantize_int4",
    "select_candidates",
    "turbo_enn",
    "weight_int4_select_ucb",
    "weight_select_ucb",
]
