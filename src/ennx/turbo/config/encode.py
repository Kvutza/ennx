from __future__ import annotations

import os
from typing import Any

import numpy as np

from .candidate_rv import CandidateRV
from .eidxdrv import ENN_INDEX_DRIVER_TO_RUST
from .init_config import LHDOnlyInit
from .model import (
    DrawAcquisitionConfig,
    GPSurrogateConfig,
    MorboTRConfig,
    NoSurrogateConfig,
    NoTRConfig,
    ParetoAcquisitionConfig,
    RandomAcquisitionConfig,
    TurboTRConfig,
    UCBAcquisitionConfig,
)
from .optimizer_config import OptimizerConfig
from .surrogate import ENNSurrogateConfig

ENN_K = 10
_CAND_FACTOR = 100.0
_CAND_MAX = 5000


def enn_k(cfg: OptimizerConfig) -> int:
    sur = cfg.surrogate
    if not isinstance(sur, ENNSurrogateConfig):
        raise TypeError(f"expected ENNSurrogateConfig, got {type(sur)!r}")
    return ENN_K if sur.k is None else int(sur.k)


def _acq(cfg: OptimizerConfig) -> dict[str, Any]:
    acq = cfg.acquisition
    if isinstance(acq, UCBAcquisitionConfig):
        return {"acquisition": "ucb", "acquisition_beta": float(acq.beta)}
    if isinstance(acq, DrawAcquisitionConfig):
        return {"acquisition": "thompson"}
    if isinstance(acq, RandomAcquisitionConfig):
        return {"acquisition": "random"}
    if isinstance(acq, ParetoAcquisitionConfig):
        return {"acquisition": "pareto"}
    return {}


def _cand(cfg: OptimizerConfig) -> dict[str, Any]:
    c = cfg.candidates
    rv = {
        CandidateRV.SOBOL: "sobol",
        CandidateRV.UNIFORM: "uniform",
        CandidateRV.RAASP: "raasp",
    }.get(c.candidate_rv)
    out: dict[str, Any] = {"candidate_rv": rv} if rv else {}
    if c.num_candidates is None:
        out.update(num_candidates_factor=_CAND_FACTOR, max_candidates=_CAND_MAX)
    else:
        n = int(c.num_candidates)
        out.update(num_candidates_factor=1.0, min_candidates=n)
        if c.num_candidates_per_arm is None:
            out["max_candidates"] = n
    if c.num_candidates_per_arm is not None:
        out["num_candidates_per_arm"] = int(c.num_candidates_per_arm)
    if c.num_pert is not None:
        out["num_pert"] = int(c.num_pert)
    return out


def _tr(cfg: OptimizerConfig) -> dict[str, Any]:
    tr = cfg.trust_region
    if isinstance(tr, MorboTRConfig):
        out: dict[str, Any] = {
            "trust_region": "morbo",
            "num_metrics": int(tr.num_metrics),
            "alpha": float(tr.alpha),
            "length_init": float(tr.length_init),
            "length_min": float(tr.length_min),
            "length_max": float(tr.length_max),
            "rescalarize": tr.rescalarize.value,
        }
        if tr.noise_aware:
            out["noise_aware"] = True
        return out
    if not isinstance(tr, TurboTRConfig):
        return {}

    out = {}
    if tr.length_init != 0.8:
        out["length_init"] = float(tr.length_init)
    if abs(tr.length_min - 0.5**7) > 1e-12:
        out["length_min"] = float(tr.length_min)
    if tr.length_max != 1.6:
        out["length_max"] = float(tr.length_max)
    if tr.noise_aware:
        out["noise_aware"] = True
    return out


def _enn(cfg: OptimizerConfig) -> dict[str, Any]:
    sur = cfg.surrogate
    if not isinstance(sur, ENNSurrogateConfig):
        return {}

    out: dict[str, Any] = {}
    if sur.index_driver in ENN_INDEX_DRIVER_TO_RUST:
        out["index_driver"] = ENN_INDEX_DRIVER_TO_RUST[sur.index_driver]
    if sur.num_samples is not None:
        out["num_samples"] = int(sur.num_samples)
    if sur.num_candidates is not None:
        out["num_candidates"] = int(sur.num_candidates)
    if sur.scale_x:
        out["scale_x"] = True
    if sur.enn_storage is not None:
        out["enn_storage"] = sur.enn_storage
    if sur.work_dir is not None:
        out["work_dir"] = os.fspath(sur.work_dir)
    if sur.y_bounds is not None:
        out["y_bounds"] = np.asarray(sur.y_bounds, dtype=float)
    return out


def encode(cfg: OptimizerConfig) -> dict[str, Any] | None:
    out = _acq(cfg) | _cand(cfg) | _tr(cfg) | _enn(cfg)
    return out or None


def supports(cfg: OptimizerConfig) -> bool:
    return isinstance(
        cfg.surrogate,
        (ENNSurrogateConfig, GPSurrogateConfig, NoSurrogateConfig),
    )


def is_lhd(cfg: OptimizerConfig) -> bool:
    return (
        isinstance(cfg.trust_region, NoTRConfig)
        and isinstance(cfg.init.init_strategy, LHDOnlyInit)
        and isinstance(cfg.surrogate, NoSurrogateConfig)
    )
