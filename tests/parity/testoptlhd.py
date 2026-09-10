from __future__ import annotations

import numpy as np
import pytest

from ennx.turbo.config import lhd_only

pytest.importorskip("ennx._rust")


def _obj(x):
    return -np.sum((x - 0.5) ** 2, axis=1)


def test_001():
    from .optimizer_checks import check_opt, make_optimizer

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    config = lhd_only(num_init=5)
    opt = make_optimizer(bounds, config, seed=31)
    check_opt(opt, bounds)


def test_002():
    from .optimizer_checks import tr_cycles

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    config = lhd_only(num_init=5)
    tr_cycles(bounds, config, opt_seed=37, cycle_rng_seed=37, obj_fn=_obj)
