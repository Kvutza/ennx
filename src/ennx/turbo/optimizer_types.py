from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


class _View:
    def __init__(self, x: np.ndarray) -> None:
        self._x = np.asarray(x, dtype=float)

    def view(self) -> np.ndarray:
        return self._x


@dataclass(frozen=True, slots=True)
class _Obs:
    x: np.ndarray
    y: np.ndarray
    yvar: np.ndarray | None


@dataclass(frozen=True, slots=True)
class Telemetry:
    dt_fit: float
    dt_sel: float
    dt_gen: float = 0.0
    dt_tell: float = 0.0
    num_candidates: int = 0


@dataclass(frozen=True, slots=True)
class _Plan:
    kind: str
    k: int = 10
    gp: Any | None = None
