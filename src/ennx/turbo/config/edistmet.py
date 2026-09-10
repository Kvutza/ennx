from __future__ import annotations

from enum import Enum


class ENNDistanceMetric(Enum):
    SQUARED_L2 = "squared_l2"
    COSINE = "cosine"
