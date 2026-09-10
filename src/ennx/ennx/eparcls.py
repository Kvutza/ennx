from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ENNParams:
    k_neighbors: int
    epistemic_scale: float
    aleatoric_scale: float

    def __post_init__(self) -> None:
        import numpy as np

        k = int(self.k_neighbors)
        if k <= 0:
            raise ValueError(f"k_neighbors must be > 0, got {k}")
        epi_var_scale = float(self.epistemic_scale)
        if not np.isfinite(epi_var_scale) or epi_var_scale < 0.0:
            raise ValueError(f"epistemic_scale must be >= 0, got {epi_var_scale}")
        ale_scale = float(self.aleatoric_scale)
        if not np.isfinite(ale_scale) or ale_scale < 0.0:
            raise ValueError(f"aleatoric_scale must be >= 0, got {ale_scale}")
