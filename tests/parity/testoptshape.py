from __future__ import annotations

import numpy as np
import pytest

from ennx import create_optimizer, turbo_zero
from ennx.turbo.config.encode import supports

pytest.importorskip("ennx._rust")


def test_fixedcandidatecount():
    config = turbo_zero(num_candidates=500, num_init=4)
    assert supports(config)
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    rng = np.random.default_rng(42)
    opt = create_optimizer(bounds=bounds, config=config, rng=rng)
    assert opt.ask(1).shape == (1, 2)


def test_001():
    config = turbo_zero(num_init=4)
    assert supports(config)
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    rng = np.random.default_rng(42)
    opt = create_optimizer(bounds=bounds, config=config, rng=rng)
    assert opt.ask(1).shape == (1, 2)


def test_002():
    num_arms = 8
    config = turbo_zero(num_init=4)
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    opt = create_optimizer(bounds=bounds, config=config, rng=np.random.default_rng(44))
    expected = config.candidates.resolve_candidates(num_dim=2, num_arms=num_arms)
    while opt.init_progress is not None:
        x = opt.ask(num_arms=num_arms)
        y = -np.sum((x - 0.5) ** 2, axis=1).reshape(-1, 1)
        opt.tell(x, y)
    opt.ask(num_arms=num_arms)
    assert opt.telemetry().num_candidates == expected


def test_badrowcount():
    config = turbo_zero(num_init=4)
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    rng = np.random.default_rng(42)
    opt = create_optimizer(bounds=bounds, config=config, rng=rng)

    x = opt.ask(num_arms=3)
    y = np.array([[1.0], [2.0]])
    with pytest.raises(ValueError, match="shape"):
        opt.tell(x, y)
