from __future__ import annotations

import numpy as np
import pytest

from ennx import create_optimizer, turbo_enn, turbo_one, turbo_zero
from ennx.turbo.config import ENNSurrogateConfig, lhd_only
from ennx.turbo.config.encode import ENN_K, enn_k, supports
from ennx.turbo.optimizer import Optimizer

pytest.importorskip("ennx._rust")

BOUNDS = np.array([[0.0, 1.0], [0.0, 1.0]])


@pytest.mark.parametrize(
    "cfg",
    [turbo_enn(), turbo_zero(), lhd_only(), turbo_one()],
)
def test_001(cfg):
    assert supports(cfg)
    opt = create_optimizer(bounds=BOUNDS, config=cfg, rng=np.random.default_rng(0))
    assert isinstance(opt, Optimizer)


def test_defaultennk():
    cfg = turbo_enn(enn=ENNSurrogateConfig(k=None))
    assert enn_k(cfg) == ENN_K
