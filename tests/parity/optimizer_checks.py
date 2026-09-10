from __future__ import annotations

import numpy as np

from ennx import create_optimizer
from ennx.turbo.config.encode import supports


def tr_cycles(
    bounds: np.ndarray,
    config,
    *,
    opt_seed: int,
    cycle_rng_seed: int,
    obj_fn,
) -> None:
    rng = np.random.default_rng(cycle_rng_seed)
    opt = make_optimizer(bounds, config, seed=opt_seed)
    x0 = opt.ask(num_arms=2)
    assert opt.region_count == 0
    y0 = obj_fn(x0).reshape(-1, 1)
    opt.tell(x0, y0)
    assert opt.region_count == 2
    _, _, best = run_cycle(opt, rng, num_arms=2, obj_fn=obj_fn, num_cycles=3)
    assert opt.region_count == 8
    assert best >= -1.0


def run_cycle(opt, rng, num_arms: int, obj_fn, num_cycles: int):
    xs, ys = [], []
    for _ in range(num_cycles):
        x = opt.ask(num_arms=num_arms)
        y = obj_fn(x)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        opt.tell(x, y)
        xs.append(x)
        ys.append(y)
    all_y = np.concatenate(ys, axis=0)
    best_y = float(np.max(all_y))
    return xs, ys, best_y


def check_opt(opt, bounds: np.ndarray):
    x = opt.ask(num_arms=3)
    assert isinstance(x, np.ndarray)
    assert x.shape == (3, bounds.shape[0])
    assert np.all(x >= bounds[:, 0])
    assert np.all(x <= bounds[:, 1])
    t = opt.telemetry()
    assert hasattr(t, "dt_fit") and hasattr(t, "dt_gen")
    assert hasattr(t, "dt_sel") and hasattr(t, "dt_tell")
    return x


def make_optimizer(bounds, config, seed: int):
    rng = np.random.default_rng(seed)
    assert supports(config)
    return create_optimizer(bounds=bounds, config=config, rng=rng)
