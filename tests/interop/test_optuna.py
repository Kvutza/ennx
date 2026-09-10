# ruff: noqa: E402
import numpy as np
import pytest

optuna = pytest.importorskip("optuna")
from optuna.distributions import FloatDistribution, IntDistribution
from optuna.trial import TrialState

from ennx import create_optimizer
from ennx.optuna import Sampler


def space():
    return {"x": FloatDistribution(-2, 3), "rate": FloatDistribution(0.01, 1, log=True)}


def suggest(trial):
    return {
        "x": trial.suggest_float("x", -2, 3),
        "rate": trial.suggest_float("rate", 0.01, 1, log=True),
    }


@pytest.mark.parametrize("direction", ["minimize", "maximize"])
def test_trajectory(config, direction):
    sampler = Sampler(space(), config=config, seed=19)
    study = optuna.create_study(sampler=sampler, direction=direction)
    reference = create_optimizer(
        bounds=sampler._loop.space.bounds, config=config, rng=np.random.default_rng(19)
    )
    sign = -1 if direction == "minimize" else 1
    for _ in range(6):
        point = suggest(trial := study.ask())
        expected = sampler._loop.space.decode(reference.ask(1)[0])
        assert point == expected
        score = point["x"] ** 2 + point["rate"]
        study.tell(trial, score)
        reference.tell(
            sampler._loop.space.encode(point)[None], np.array([[sign * score]])
        )
    assert len(study.trials) == 6


def test_optimize(config):
    sampler = Sampler(space(), config=config, seed=31)
    study = optuna.create_study(sampler=sampler)
    study.optimize(lambda trial: sum(suggest(trial).values()), n_trials=5)
    assert len(study.trials) == 5
    assert np.isfinite(study.best_value)


def test_pending(config):
    sampler = Sampler(space(), config=config)
    study = optuna.create_study(sampler=sampler)
    first = study.ask()
    second = study.ask()
    with pytest.raises(RuntimeError, match="pending trial"):
        suggest(second)
    study.tell(second, state=TrialState.FAIL)
    a = suggest(first)
    second = study.ask()
    b = suggest(second)
    assert a != b
    study.tell(first, state=TrialState.FAIL)
    study.tell(second, state=TrialState.PRUNED)
    suggest(study.ask())
    assert not sampler._loop._seen


def test_restart(config, tmp_path):
    storage = f"sqlite:///{tmp_path / 'study.db'}"
    study = optuna.create_study(
        study_name="ennx",
        storage=storage,
        sampler=Sampler(space(), config=config, seed=3),
    )
    study.optimize(lambda trial: sum(suggest(trial).values()), n_trials=4)
    sampler = Sampler(space(), config=config, seed=3)
    restored = optuna.load_study(study_name="ennx", storage=storage, sampler=sampler)
    suggest(restored.ask())
    assert len(sampler._loop._seen) == 4
    suggest(restored.ask())
    assert len(sampler._loop._seen) == 4


def test_spaces(config):
    for bad in [
        {},
        {"x": IntDistribution(0, 3)},
        {"x": FloatDistribution(0, 1, step=0.1)},
        {"x": FloatDistribution(1, 1)},
    ]:
        with pytest.raises(ValueError):
            Sampler(bad, config=config)
    sampler = Sampler(space(), config=config)
    study = optuna.create_study(sampler=sampler)
    trial = study.ask()
    with pytest.raises(ValueError, match="outside"):
        trial.suggest_float("other", 0, 1)
    study.tell(trial, state=TrialState.FAIL)
    with pytest.raises(ValueError, match="one objective"):
        multi = optuna.create_study(
            sampler=Sampler(space(), config=config), directions=["minimize", "maximize"]
        )
        suggest(multi.ask())
    with pytest.raises(RuntimeError, match="n_jobs"):
        sampler.reseed_rng()


def test_contract(config):
    sampler = Sampler(space(), config=config)
    study = optuna.create_study(sampler=sampler)
    trial = study.ask()
    trial.suggest_float("x", -2, 3)
    with pytest.raises(ValueError, match="every trial"):
        study.tell(trial, 0)
    with pytest.raises(ValueError, match="separate"):
        suggest(optuna.create_study(sampler=sampler).ask())
