from __future__ import annotations

import numpy as np
import pytest

from ennx import create_optimizer, turbo_enn, turbo_zero
from ennx.turbo.config import CandidateGenConfig, ENNFitConfig, ENNSurrogateConfig
from ennx.turbo.config.encode import encode

pytest.importorskip("ennx._rust")


def test_perarmpool():
    config = turbo_zero(num_candidates_per_arm=1500, num_init=4)
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    rng = np.random.default_rng(99)
    opt = create_optimizer(bounds=bounds, config=config, rng=rng)
    assert config.candidates.resolve_candidates(num_dim=2, num_arms=3) == 4500
    while opt.init_progress is not None:
        x = opt.ask(num_arms=3)
        y = -np.sum((x - 0.5) ** 2, axis=1).reshape(-1, 1)
        opt.tell(x, y)
    x = opt.ask(num_arms=3)
    assert opt.telemetry().num_candidates == 4500
    y = -np.sum((x - 0.5) ** 2, axis=1).reshape(-1, 1)
    opt.tell(x, y)
    x2 = opt.ask(num_arms=8)
    assert x2.shape == (8, 2)
    assert opt.telemetry().num_candidates == 12000


def test_perarmisencoded():
    cfg_enn = turbo_enn(
        candidates=CandidateGenConfig(num_candidates_per_arm=25),
        num_init=4,
    )
    cfg_zero = turbo_zero(num_candidates_per_arm=25, num_init=4)
    for cfg in (cfg_enn, cfg_zero):
        overrides = encode(cfg)
        assert overrides is not None
        assert overrides.get("num_candidates_per_arm") == 25


def test_ennperarmpool():
    config = turbo_enn(
        candidates=CandidateGenConfig(num_candidates_per_arm=40),
        enn=ENNSurrogateConfig(k=3, fit=ENNFitConfig(num_samples=8)),
        num_init=4,
    )
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    rng = np.random.default_rng(101)
    opt = create_optimizer(bounds=bounds, config=config, rng=rng)
    expected = config.candidates.resolve_candidates(num_dim=2, num_arms=3)
    assert expected == 200
    while opt.init_progress is not None:
        x = opt.ask(num_arms=3)
        y = -np.sum((x - 0.5) ** 2, axis=1).reshape(-1, 1)
        opt.tell(x, y)
    opt.ask(num_arms=3)
    assert opt.telemetry().num_candidates == expected


def test_fixedandperarm():
    config = turbo_zero(
        num_candidates=100,
        num_candidates_per_arm=50,
        num_init=4,
    )
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    rng = np.random.default_rng(102)
    opt = create_optimizer(bounds=bounds, config=config, rng=rng)
    assert config.candidates.resolve_candidates(num_dim=2, num_arms=4) == 200
    while opt.init_progress is not None:
        x = opt.ask(num_arms=4)
        y = -np.sum((x - 0.5) ** 2, axis=1).reshape(-1, 1)
        opt.tell(x, y)
    opt.ask(num_arms=4)
    assert opt.telemetry().num_candidates == 200


def test_perarmformula():
    cfg = turbo_zero(num_candidates_per_arm=40, num_init=4)
    assert cfg.candidates.resolve_candidates(num_dim=2, num_arms=3) == 200
    assert cfg.candidates.resolve_candidates(num_dim=2, num_arms=8) == 320
    overrides = encode(cfg)
    assert overrides.get("num_candidates_per_arm") == 40
    assert overrides.get("num_candidates_factor") == 100.0
    assert overrides.get("max_candidates") == 5000


def _finishinit(opt, num_arms: int) -> None:
    while opt.init_progress is not None:
        x = opt.ask(num_arms=num_arms)
        y = -np.sum((x - 0.5) ** 2, axis=1).reshape(-1, 1)
        opt.tell(x, y)


def test_defaultpoolmanyarms():
    num_arms = 25
    config = turbo_zero(num_init=4)
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    opt = create_optimizer(bounds=bounds, config=config, rng=np.random.default_rng(103))
    expected = config.candidates.resolve_candidates(num_dim=2, num_arms=num_arms)
    assert expected == 200
    _finishinit(opt, num_arms)
    opt.ask(num_arms=num_arms)
    assert opt.telemetry().num_candidates == expected


def test_highdimpoolcap():
    num_dim = 60
    num_arms = 1
    config = turbo_zero(num_candidates_per_arm=40, num_init=4)
    bounds = np.tile(np.array([[0.0, 1.0]], dtype=float), (num_dim, 1))
    opt = create_optimizer(bounds=bounds, config=config, rng=np.random.default_rng(104))
    expected = config.candidates.resolve_candidates(num_dim=num_dim, num_arms=num_arms)
    assert expected == 5000
    _finishinit(opt, num_arms)
    opt.ask(num_arms=num_arms)
    assert opt.telemetry().num_candidates == expected
