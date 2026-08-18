from __future__ import annotations

import numpy as np
import pytest

from ennx.turbo.config import (
    AcqType,
    ENNFitConfig,
    ENNSurrogateConfig,
    turbo_enn_config,
    turbo_zero_config,
)

pytest.importorskip("ennx._rust")


def _obj(x):
    return -np.sum((x - 0.5) ** 2, axis=1)


def _tr_lengths(opt, num_arms: int, num_cycles: int):
    lengths = []
    for _ in range(num_cycles):
        x = opt.ask(num_arms=num_arms)
        y = _obj(x)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        opt.tell(x, y)
        lengths.append(opt.tr_length)
    return lengths


def test_optimizer_tr_length_trajectory_contract():
    from .optimizer_checks import make_optimizer

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    num_arms = 4
    config = turbo_enn_config(
        bounds=bounds,
        batch_size=num_arms,
        max_evals=30,
        device="cpu",
    )
    opt = make_optimizer(config)

    lengths = _tr_lengths(opt, num_arms, num_cycles=8)
    assert len(lengths) == 8
    for length in lengths:
        assert 0.0 < length <= 2.0


def test_optimizer_pareto_tr_length_contract():
    from .optimizer_checks import make_optimizer

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    num_arms = 3
    config = turbo_enn_config(
        acq_type=AcqType.PARETO,
        enn=ENNSurrogateConfig(k=3, fit=ENNFitConfig(num_fit_samples=10)),
        num_init=6,
    )
    opt = make_optimizer(bounds, config, seed=47)
    lengths = _tr_lengths(opt, num_arms, num_cycles=6)
    assert len(lengths) == 6
    for length in lengths:
        assert 0.0 < length <= 2.0


def test_acquisition_ucb_produces_valid_candidates():
    from .optimizer_checks import check_opt_contract, make_optimizer

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    config = turbo_enn_config(
        acq_type=AcqType.UCB,
        enn=ENNSurrogateConfig(k=3, fit=ENNFitConfig(num_fit_samples=10)),
        num_init=4,
    )
    opt = make_optimizer(bounds, config, seed=19)
    check_opt_contract(opt, bounds)


def test_acquisition_thompson_config_passthrough():
    from .optimizer_checks import check_opt_contract, make_optimizer

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    config = turbo_enn_config(
        acq_type=AcqType.THOMPSON,
        enn=ENNSurrogateConfig(k=3, fit=ENNFitConfig(num_fit_samples=10)),
        num_init=4,
    )
    opt = make_optimizer(bounds, config, seed=23)
    check_opt_contract(opt, bounds)


def test_multi_objective_route():
    from ennx import create_optimizer

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    config = turbo_enn_config(
        acq_type=AcqType.PARETO,
        enn=ENNSurrogateConfig(k=4, fit=ENNFitConfig(num_fit_samples=10)),
        num_init=4,
    )
    rng = np.random.default_rng(3)
    opt = create_optimizer(bounds=bounds, config=config, rng=rng)
    x = opt.ask(num_arms=2)
    assert x.shape == (2, 2)


def test_multi_objective_width():
    from ennx import create_optimizer

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    config = turbo_enn_config(
        acq_type=AcqType.PARETO,
        enn=ENNSurrogateConfig(k=3, fit=ENNFitConfig(num_fit_samples=10)),
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


def test_multi_objective_two_metrics():
    from ennx import create_optimizer

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    config = turbo_enn_config(
        acq_type=AcqType.PARETO,
        enn=ENNSurrogateConfig(k=3, fit=ENNFitConfig(num_fit_samples=10)),
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


def test_multi_objective_y_obs_shape():
    from ennx import create_optimizer

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    config = turbo_enn_config(
        acq_type=AcqType.PARETO,
        enn=ENNSurrogateConfig(k=3, fit=ENNFitConfig(num_fit_samples=10)),
        num_init=4,
    )
    rng = np.random.default_rng(7)
    opt = create_optimizer(bounds=bounds, config=config, rng=rng)
    x = opt.ask(num_arms=4)
    y = np.column_stack([_obj(x), -_obj(x)])
    opt.tell(x, y)
    y_obs = opt._y_obs.view()
    assert y_obs.shape[1] == 2


def test_keeps_all_observations():
    from .optimizer_checks import make_optimizer

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    config = turbo_zero_config(num_init=4)
    opt = make_optimizer(bounds, config, seed=7)

    for _ in range(25):
        x = opt.ask(num_arms=1)
        y = _obj(x).reshape(-1, 1)
        opt.tell(x, y)

    assert opt.tr_obs_count == 25
