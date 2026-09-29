from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, replace

from .._lazy import module_attr

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "ModelPackage": (".._rust", "ModelPackage"),
    "NativeKdaModel": (".._rust", "NativeKdaModel"),
    "FlameEvaluator": (".._rust", "FlameEvaluator"),
    "MetalFlameEvaluator": (".._rust", "MetalFlameEvaluator"),
    "MetalQwenEvaluator": (".._rust", "MetalQwenEvaluator"),
    "MetalWeights": (".._rust", "MetalWeights"),
    "MetalParamBlock": (".._rust", "MetalParamBlock"),
    "MetalSearchState": (".._rust", "MetalSearchState"),
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


@dataclass(frozen=True)
class TurboEnnConfig:
    """Reusable resident-search settings; CUDA validates their numeric domains."""

    max_pending: int = 1
    base_variance: float = 0.0
    length_init: float = 0.8
    length_min: float = 0.0078125
    length_max: float = 1.6
    failure_tolerance: int | None = None
    sampler: str = "independent"
    reference_seed: int = 0


def turbo_enn(
    base: object,
    base_value: float,
    blocks: list[object],
    capacity: int,
    *,
    config: TurboEnnConfig = TurboEnnConfig(),
    **overrides: object,
) -> object:
    """Create resident CUDA BF16 search with independent sign proposals by default.

    ``sampler="independent"`` preserves legacy sign noise with TuRBO.
    ``sampler="gaussian"`` uses independent dense Gaussian noise with TuRBO and
    no reference direction. ``reference_seed`` is unused in these two modes.
    ``sampler="correlated"`` uses dense Gaussian noise and a reference direction
    initialized by ``reference_seed`` (u64), updated on acceptance. It requires
    ``max_pending=1``,
    ``failure_tolerance=None``, and ``ask(arms=1, candidates=4, ...)``. The GPU
    accepted-radius controller replaces TuRBO counters in this mode.
    ``proposals.geometry()`` reports each selected candidate index and Gaussian
    persistence (0.75 for correlated indices 0/1, otherwise 0); ``describe()``
    reports its actual selected radius. Persistence is the noise mixture
    coefficient, not measured correlation after BF16 rounding or selection.
    ``search.read_reference()`` explicitly copies the stored correlated reference
    to a host NumPy uint16 array of raw BF16 bits for validation. This is a
    model-sized copy, requires a completed round and released DLPack views, and
    errors in other modes.

    Pass a reusable ``config`` or the existing individual setting keywords.
    Individual keywords override the corresponding fields in ``config``.
    """
    settings = replace(config, **overrides) if overrides else config
    search_type = __getattr__("SearchState")
    if search_type is None:
        raise RuntimeError("turbo_enn requires the CUDA wheel")
    return search_type(
        base,
        base_value,
        blocks,
        capacity,
        **asdict(settings),
    )


def __getattr__(name: str):
    return module_attr(name, globals())


__all__: list[str] = [
    "BpannHistory",
    "CandidateProposal",
    "DenseLinear",
    "FlameEvaluator",
    "MetalFlameEvaluator",
    "MetalQwenEvaluator",
    "MetalParamBlock",
    "MetalSearchState",
    "MetalWeights",
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
    "TurboEnnConfig",
    "allocate_batches",
    "create_lhd",
    "create_optimizer",
    "create_zero",
    "dense_apply",
    "dense_dist2",
    "dense_linear",
    "enn_optimizer",
    "enn_tr",
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
