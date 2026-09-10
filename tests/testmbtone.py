import numpy as np
import pytest

from ennx import create_optimizer
from ennx.benchmarks import DoubleAckley
from ennx.turbo.config import MorboTRConfig, MultiObjectiveConfig, turbo_one


@pytest.mark.slow
def test_001():
    num_dim = 6
    num_arms = 3
    noise = 0.1
    num_metrics = 2
    rng = np.random.default_rng(42)
    objective = DoubleAckley(noise=noise, rng=rng)
    bounds = np.array([objective.bounds] * num_dim, dtype=float)
    config = turbo_one(
        trust_region=MorboTRConfig(
            multi_objective=MultiObjectiveConfig(num_metrics=num_metrics)
        )
    )
    optimizer = create_optimizer(bounds=bounds, config=config, rng=rng)
    for iteration in range(2):
        x_arms = optimizer.ask(num_arms=num_arms)
        y_obs = objective(x_arms)
        optimizer.tell(x_arms, y_obs)
        print(
            f"Iteration {iteration}: x_arms shape = {x_arms.shape}, y_obs shape = {y_obs.shape}"
        )
