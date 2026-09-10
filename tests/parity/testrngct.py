from __future__ import annotations

import numpy as np
import pytest

from ennx.turbo.config import (
    AcqType,
    ENNFitConfig,
    ENNSurrogateConfig,
    turbo_enn,
)

pytest.importorskip("ennx._rust")

EXACT_RTOL = 1e-14
EXACT_ATOL = 1e-14


def test_001():
    from .optimizer_checks import make_optimizer

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    config = turbo_enn(
        acq_type=AcqType.UCB,
        enn=ENNSurrogateConfig(k=4, fit=ENNFitConfig(num_samples=10)),
        num_init=6,
    )
    opt_a = make_optimizer(bounds, config, seed=99)
    opt_b = make_optimizer(bounds, config, seed=99)
    xa = opt_a.ask(num_arms=3)
    xb = opt_b.ask(num_arms=3)
    np.testing.assert_allclose(xa, xb, rtol=EXACT_RTOL, atol=EXACT_ATOL)
