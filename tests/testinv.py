from __future__ import annotations

import numpy as np
import pytest
from enn_helpers import sphere_objective

from ennx.turbo.config import (
    OptimizerConfig,
    turbo_enn,
    turbo_one,
    turbo_zero,
)


def _unit(x: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    return (x - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])


@pytest.mark.parametrize(
    "config",
    [
        turbo_zero(),
        turbo_enn(),
        pytest.param(turbo_one(), marks=pytest.mark.slow),
    ],
    ids=["TURBO_ZERO", "TURBO_ENN", "TURBO_ONE"],
)
def test_001(config: OptimizerConfig) -> None:
    from ennx import create_optimizer

    bounds1 = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    bounds2 = np.array([[2.0, 4.0], [-3.0, 1.0]], dtype=float)
    num_arms, num_steps = 2, 2
    rng1, rng2 = np.random.default_rng(0), np.random.default_rng(0)
    opt1 = create_optimizer(bounds=bounds1, config=config, rng=rng1)
    opt2 = create_optimizer(bounds=bounds2, config=config, rng=rng2)
    for _ in range(num_steps):
        x1, x2 = opt1.ask(num_arms=num_arms), opt2.ask(num_arms=num_arms)
        u1, u2 = _unit(x1, bounds1), _unit(x2, bounds2)
        assert np.allclose(u1, u2)
        y1 = sphere_objective(2.0 * u1 - 1.0)
        y2 = sphere_objective(2.0 * u2 - 1.0)
        assert np.allclose(y1, y2)
        opt1.tell(x1, y1)
        opt2.tell(x2, y2)


@pytest.mark.parametrize(
    "config",
    [turbo_zero(), turbo_enn()],
    ids=["TURBO_ZERO", "TURBO_ENN"],
)
def test_002(config: OptimizerConfig) -> None:
    from ennx import create_optimizer

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    num_arms, num_steps = 2, 2

    def run_transform(scale: float, shift: float) -> np.ndarray:
        rng = np.random.default_rng(0)
        opt = create_optimizer(bounds=bounds, config=config, rng=rng)
        unit_trajectory = []
        for _ in range(num_steps):
            x = opt.ask(num_arms=num_arms)
            u = x.copy()
            base_y = sphere_objective(2.0 * u - 1.0)
            opt.tell(x, scale * base_y + shift)
            unit_trajectory.append(u)
        return np.stack(unit_trajectory, axis=0)

    traj_base = run_transform(scale=1.0, shift=0.0)
    traj_affine = run_transform(scale=2.0, shift=0.5)
    assert np.allclose(traj_base, traj_affine)
