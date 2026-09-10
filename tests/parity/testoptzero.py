from __future__ import annotations

import numpy as np
import pytest

from ennx.turbo.config import turbo_zero

pytest.importorskip("ennx._rust")


def _obj(x):
    return -np.sum((x - 0.5) ** 2, axis=1)


def test_001():
    from .optimizer_checks import check_opt, make_optimizer

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    config = turbo_zero(num_init=4)
    opt = make_optimizer(bounds, config, seed=13)
    check_opt(opt, bounds)


def test_002():
    from .optimizer_checks import make_optimizer, run_cycle

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    config = turbo_zero(num_init=4)
    rng = np.random.default_rng(17)
    opt = make_optimizer(bounds, config, seed=17)
    x0 = opt.ask(num_arms=2)
    assert opt.region_count == 0
    y0 = _obj(x0).reshape(-1, 1)
    opt.tell(x0, y0)
    assert opt.region_count == 2
    _, _, best = run_cycle(opt, rng, num_arms=2, obj_fn=_obj, num_cycles=3)
    assert opt.region_count == 8
    assert best >= -1.0
