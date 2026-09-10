from __future__ import annotations

import numpy as np
import pytest

from ennx.turbo.config import (
    AcqType,
    ENNFitConfig,
    ENNSurrogateConfig,
    turbo_enn,
    turbo_zero,
)
from ennx.turbo.config.candgencfg import CandidateGenConfig
from ennx.turbo.config.candidate_rv import CandidateRV

pytest.importorskip("ennx._rust")


def _obj(x):
    return -np.sum((x - 0.5) ** 2, axis=1)


def test_001():
    from .optimizer_checks import check_opt, make_optimizer

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    config = turbo_enn(
        acq_type=AcqType.UCB,
        enn=ENNSurrogateConfig(k=3, fit=ENNFitConfig(num_samples=10)),
        num_init=4,
        candidates=CandidateGenConfig(candidate_rv=CandidateRV.RAASP),
    )
    opt = make_optimizer(bounds, config, seed=41)
    check_opt(opt, bounds)


def test_raaspdistribution():
    from .optimizer_checks import make_optimizer

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    num_arms = 4
    config = turbo_zero(num_init=8, candidate_rv=CandidateRV.RAASP)
    opt = make_optimizer(bounds, config, seed=77)
    cands = []
    for _ in range(6):
        x = opt.ask(num_arms=num_arms)
        cands.append(x)
        opt.tell(x, _obj(x).reshape(-1, 1))

    x = np.concatenate(cands, axis=0)
    assert x.shape[1] == 2
    assert np.all((0 <= x) & (x <= 1))
