from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from .enn_distance_metric import ENNDistanceMetric
from .enn_fit_config import ENNFitConfig
from .enn_index_driver import ENNIndexDriver


@dataclass(frozen=True)
class ENNSurrogateConfig:
    k: int | None = None
    fit: ENNFitConfig = field(default_factory=ENNFitConfig)
    scale_x: bool = False
    index_driver: ENNIndexDriver = ENNIndexDriver.FLAT
    distance_metric: ENNDistanceMetric = ENNDistanceMetric.SQUARED_L2
    enn_storage: str | None = None
    work_dir: str | os.PathLike[str] | None = None
    y_bounds: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.scale_x and self.index_driver == ENNIndexDriver.BPANN_DISK:
            raise ValueError("scale_x=True is not compatible with BPANN_DISK")
        if self.y_bounds is not None:
            bounds = np.asarray(self.y_bounds, dtype=float)
            if bounds.ndim != 2 or bounds.shape[1] != 2:
                raise ValueError(
                    f"y_bounds must have shape (metrics, 2), got {bounds.shape}"
                )
            object.__setattr__(self, "y_bounds", bounds)

    @property
    def num_fit_samples(self) -> int | None:
        return self.fit.num_fit_samples

    @property
    def num_fit_candidates(self) -> int | None:
        return self.fit.num_fit_candidates
