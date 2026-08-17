from __future__ import annotations

import sys

from .._lazy import lazy_getattr

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "ModelPackage": (".._rust", "ModelPackage"),
    "NativeKdaModel": (".._rust", "NativeKdaModel"),
    "ResidentBoSession": (".._rust", "ResidentBoSession"),
    "Optimizer": (".._rust", "Optimizer"),
    "Telemetry": (".._rust", "Telemetry"),
    "PackedTurbo": (".._rust", "PackedTurbo"),
    "TurboTrial": (".._rust", "TurboTrial"),
    "MultiTrustRegion": (".._rust", "MultiTrustRegion"),
    "PackedSearch": (".._rust", "PackedSearch"),
    "BpannHistory": (".._rust", "BpannHistory"),
    "ParamBuffer": (".._rust", "ParamBuffer"),
    "ParamBlock": (".._rust", "ParamBlock"),
    "SearchState": (".._rust", "SearchState"),
    "Proposals": (".._rust", "Proposals"),
    "SharingPolicy": (".multi_trust_region", "SharingPolicy"),
    "RegionBatch": (".multi_trust_region", "RegionBatch"),
    "RegionCandidate": (".multi_trust_region", "RegionCandidate"),
    "CandidateProposal": (".multi_trust_region", "CandidateProposal"),
    "RegionRound": (".multi_trust_region", "RegionRound"),
    "MultiTrustRegionLoop": (".multi_trust_region", "MultiTrustRegionLoop"),
    "make_multi_trust_region": (".multi_trust_region", "make_multi_trust_region"),
    "allocate_region_batches": (".multi_trust_region", "allocate_region_batches"),
    "select_region_candidates": (".multi_trust_region", "select_region_candidates"),
    "multi_trust_region": (".multi_trust_region", "multi_trust_region"),
    "create_optimizer_enn": (".._rust", "create_optimizer_enn"),
    "create_optimizer_enn_multi_tr": (".._rust", "create_optimizer_enn_multi_tr"),
    "create_optimizer_zero": (".._rust", "create_optimizer_zero"),
    "create_optimizer_lhd": (".._rust", "create_optimizer_lhd"),
    "weight_int4_select_ucb": (".._rust", "weight_int4_select_ucb"),
    "weight_select_ucb": (".._rust", "weight_select_ucb"),
    "dense_apply": (".._rust", "dense_apply"),
    "dense_dist2": (".._rust", "dense_dist2"),
    "dense_linear": (".._rust", "dense_linear"),
    "DenseLinear": (".._rust", "DenseLinear"),
    "quantize_int4": ("..quantization", "quantize_int4"),
    "quantize_fp4_e2m1": ("..quantization", "quantize_fp4_e2m1"),
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
    return lazy_getattr(
        name=name,
        module_name=__name__,
        package=__package__,
        mapping=_LAZY_ATTRS,
        extra="`pip install 'ennx[with-deps]'`",
    )


__all__: list[str] = [
    "BpannHistory",
    "CandidateProposal",
    "DenseLinear",
    "ModelPackage",
    "MultiTrustRegion",
    "MultiTrustRegionLoop",
    "NativeKdaModel",
    "Optimizer",
    "PackedSearch",
    "PackedTurbo",
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
    "TurboTrial",
    "allocate_region_batches",
    "create_optimizer_enn",
    "create_optimizer_enn_multi_tr",
    "create_optimizer_lhd",
    "create_optimizer_zero",
    "dense_apply",
    "dense_dist2",
    "dense_linear",
    "experimental",
    "make_multi_trust_region",
    "multi_trust_region",
    "quantize_fp4_e2m1",
    "quantize_int4",
    "select_region_candidates",
    "turbo_enn",
    "weight_int4_select_ucb",
    "weight_select_ucb",
]
