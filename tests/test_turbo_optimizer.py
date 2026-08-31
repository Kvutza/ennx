from __future__ import annotations

import numpy as np
import pytest

from ennx import create_optimizer, turbo_enn_config, turbo_one_config, turbo_zero_config


def _score(x: np.ndarray) -> np.ndarray:
    return -np.sum(x**2, axis=1)


@pytest.mark.parametrize("cfg", [turbo_zero_config(), turbo_enn_config()])
def test_ask_tell_cycle(cfg):
    bounds = np.array([[-1.0, 1.0], [-1.0, 1.0]])
    opt = create_optimizer(bounds=bounds, config=cfg, rng=np.random.default_rng(0))
    x = opt.ask(4)
    y = opt.tell(x, _score(x))

    assert x.shape == (4, 2)
    assert y.shape == (4, 1)
    assert opt.ask(2).shape == (2, 2)


def test_tell_write_protects_arrays():
    bounds = np.array([[-1.0, 1.0], [-1.0, 1.0]])
    opt = create_optimizer(
        bounds=bounds, config=turbo_zero_config(), rng=np.random.default_rng(0)
    )
    x = opt.ask(4)
    y = _score(x)

    assert x.flags.writeable
    assert y.flags.writeable

    opt.tell(x, y)

    # Verify that the arrays were marked read-only
    assert not x.flags.writeable
    assert not y.flags.writeable

    with pytest.raises(ValueError):
        x[0, 0] = 999.0
    with pytest.raises(ValueError):
        y[0] = 999.0


def test_list_bounds():
    opt = create_optimizer(
        bounds=[[0.0, 1.0], [0.0, 1.0]],
        config=turbo_zero_config(),
        rng=np.random.default_rng(1),
    )
    assert opt.ask(2).shape == (2, 2)


def test_bad_batch_is_rejected():
    opt = create_optimizer(
        bounds=np.array([[0.0, 1.0], [0.0, 1.0]]),
        config=turbo_zero_config(),
        rng=np.random.default_rng(2),
    )
    with pytest.raises(ValueError, match="shape"):
        opt.tell(opt.ask(3), np.ones(2))


@pytest.mark.gp
def test_gp_noise_cycle():
    bounds = np.array([[-1.0, 1.0], [-1.0, 1.0]])
    rng = np.random.default_rng(3)
    opt = create_optimizer(
        bounds=bounds,
        config=turbo_one_config(num_init=2),
        rng=rng,
    )
    x = opt.ask(2)
    opt.tell(x, _score(x), np.full(2, 0.05))
    assert opt.ask(2).shape == (2, 2)
