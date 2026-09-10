from __future__ import annotations

import numpy as np

from ennx.ennx.posterior_flags import PosteriorFlags
from ennx.turbo.config.eidxdrv import ENNIndexDriver


def _rustseeds(function_seeds: np.ndarray | list[int]) -> list[int]:
    if hasattr(function_seeds, "__iter__"):
        return np.asarray(function_seeds, dtype=np.int64).tolist()
    return list(function_seeds)


def posterior_neighbors(
    rust_model,
    x: np.ndarray,
    *,
    search_k: int,
    flags: PosteriorFlags = PosteriorFlags(),
) -> tuple[np.ndarray, np.ndarray]:
    dist2s, idx = rust_model.posterior_neighbors(
        np.asarray(x, dtype=float),
        int(search_k),
        bool(flags.exclude_nearest),
        bool(flags.tie_neighbors),
    )
    return np.asarray(dist2s, dtype=float), np.asarray(idx, dtype=int)


def nearest_neighbors(
    rust_model,
    x: np.ndarray,
    *,
    search_k: int,
    exclude_nearest: bool,
) -> tuple[np.ndarray, np.ndarray]:
    dist2s, idx = rust_model.nearest_neighbors(
        np.asarray(x, dtype=float),
        int(search_k),
        bool(exclude_nearest),
    )
    return np.asarray(dist2s, dtype=float), np.asarray(idx, dtype=int)


def _rustname(index_driver: ENNIndexDriver) -> str:
    from ennx.turbo.config.eidxdrv import ENN_INDEX_DRIVER_TO_RUST

    if index_driver not in ENN_INDEX_DRIVER_TO_RUST:
        raise ValueError(f"Unsupported index driver: {index_driver}")
    return ENN_INDEX_DRIVER_TO_RUST[index_driver]
