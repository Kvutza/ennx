from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ENNFitConfig:
    num_samples: int | None = None
    num_candidates: int | None = None
    infer_aleatoric_variance_scale: bool = True

    def __post_init__(self) -> None:
        if self.num_samples is not None and self.num_samples <= 0:
            raise ValueError(f"num_samples must be > 0, got {self.num_samples}")
        if self.num_candidates is not None and self.num_candidates <= 0:
            raise ValueError(f"num_candidates must be > 0, got {self.num_candidates}")
