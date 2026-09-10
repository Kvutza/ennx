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

pytest.importorskip("ennx._rust")


def _obj(x):
    return -np.sum((x - 0.5) ** 2, axis=1)


def _trlengths(opt, num_arms: int, num_cycles: int):
    lengths = []
    for _ in range(num_cycles):
        x = opt.ask(num_arms=num_arms)
        y = _obj(x)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        opt.tell(x, y)
        lengths.append(opt.tr_length)
    return lengths


def test_001():
    from .optimizer_checks import make_optimizer

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    num_arms = 4
    config = turbo_enn()
    opt = make_optimizer(bounds, config, seed=41)

    lengths = _trlengths(opt, num_arms, num_cycles=8)
    assert len(lengths) == 8
    for length in lengths:
        assert 0.0 < length <= 2.0


def test_002():
    from .optimizer_checks import make_optimizer

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    num_arms = 3
    config = turbo_enn(
        acq_type=AcqType.PARETO,
        enn=ENNSurrogateConfig(k=3, fit=ENNFitConfig(num_samples=10)),
        num_init=6,
    )
    opt = make_optimizer(bounds, config, seed=47)
    lengths = _trlengths(opt, num_arms, num_cycles=6)
    assert len(lengths) == 6
    for length in lengths:
        assert 0.0 < length <= 2.0


def test_003():
    from .optimizer_checks import check_opt, make_optimizer

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    config = turbo_enn(
        acq_type=AcqType.UCB,
        enn=ENNSurrogateConfig(k=3, fit=ENNFitConfig(num_samples=10)),
        num_init=4,
    )
    opt = make_optimizer(bounds, config, seed=19)
    check_opt(opt, bounds)


def test_004():
    from .optimizer_checks import check_opt, make_optimizer

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    config = turbo_enn(
        acq_type=AcqType.THOMPSON,
        enn=ENNSurrogateConfig(k=3, fit=ENNFitConfig(num_samples=10)),
        num_init=4,
    )
    opt = make_optimizer(bounds, config, seed=23)
    check_opt(opt, bounds)


def test_multiobjectiveroute():
    from ennx import create_optimizer

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    config = turbo_enn(
        acq_type=AcqType.PARETO,
        enn=ENNSurrogateConfig(k=4, fit=ENNFitConfig(num_samples=10)),
        num_init=4,
    )
    rng = np.random.default_rng(3)
    opt = create_optimizer(bounds=bounds, config=config, rng=rng)
    x = opt.ask(num_arms=2)
    assert x.shape == (2, 2)


def test_multiobjectivewidth():
    from ennx import create_optimizer

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    config = turbo_enn(
        acq_type=AcqType.PARETO,
        enn=ENNSurrogateConfig(k=3, fit=ENNFitConfig(num_samples=10)),
        num_init=4,
    )
    rng = np.random.default_rng(5)
    opt = create_optimizer(bounds=bounds, config=config, rng=rng)
    x = opt.ask(num_arms=2)
    y = _obj(x).reshape(-1, 1)
    opt.tell(x, y)
    assert x.shape == (2, 2)
    x2 = opt.ask(num_arms=2)
    y2 = np.column_stack([_obj(x2), -_obj(x2)])
    with pytest.raises(ValueError, match="unsupported"):
        opt.tell(x2, y2)


def test_005():
    from ennx import create_optimizer

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    config = turbo_enn(
        acq_type=AcqType.PARETO,
        enn=ENNSurrogateConfig(k=3, fit=ENNFitConfig(num_samples=10)),
        num_init=4,
    )
    rng = np.random.default_rng(11)
    opt = create_optimizer(bounds=bounds, config=config, rng=rng)
    x = opt.ask(num_arms=2)
    y = np.column_stack([_obj(x), -_obj(x)])
    opt.tell(x, y)
    x2 = opt.ask(num_arms=2)
    y2 = np.column_stack([_obj(x2), -_obj(x2)])
    opt.tell(x2, y2)
    assert x2.shape == (2, 2)
    assert y2.shape == (2, 2)


def test_006():
    from ennx import create_optimizer

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    config = turbo_enn(
        acq_type=AcqType.PARETO,
        enn=ENNSurrogateConfig(k=3, fit=ENNFitConfig(num_samples=10)),
        num_init=4,
    )
    rng = np.random.default_rng(7)
    opt = create_optimizer(bounds=bounds, config=config, rng=rng)
    x = opt.ask(num_arms=4)
    y = np.column_stack([_obj(x), -_obj(x)])
    opt.tell(x, y)
    y_obs = opt._yobs.view()
    assert y_obs.shape[1] == 2


def test_007():
    from .optimizer_checks import make_optimizer

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    config = turbo_zero(num_init=4)
    opt = make_optimizer(bounds, config, seed=7)

    for _ in range(25):
        x = opt.ask(num_arms=1)
        y = _obj(x).reshape(-1, 1)
        opt.tell(x, y)

    assert opt.region_count == 25
