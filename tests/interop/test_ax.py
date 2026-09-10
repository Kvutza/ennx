# ruff: noqa: E402
import pytest

pytest.importorskip("ax")
from ax.api.client import Client
from ax.api.configs import RangeParameterConfig
from ax.generation_strategy.generation_strategy import GenerationStrategy

from ennx.ax import Node


def client(node, objective="-loss"):
    result = Client()
    result.configure_experiment(
        parameters=[
            RangeParameterConfig(name="x", parameter_type="float", bounds=(-2, 3)),
            RangeParameterConfig(name="y", parameter_type="float", bounds=(0, 1)),
        ]
    )
    result.configure_optimization(objective=objective)
    result.set_generation_strategy(GenerationStrategy(nodes=[node]))
    return result


@pytest.mark.parametrize(
    "objective,weight", [("-loss", -1), ("loss", 1), ("2*loss", 2)]
)
def test_trials(config, objective, weight):
    node = Node(config=config, seed=9)
    study = client(node, objective)
    scores = {}
    for _ in range(5):
        for index, parameters in study.get_next_trials(max_trials=1).items():
            score = sum(value**2 for value in parameters.values())
            scores[index] = score
            study.complete_trial(index, raw_data={"loss": (score, 0.2)})
    node.update_generator_state(study._experiment, study._experiment.lookup_data())
    assert len(node._loop._seen) == 5
    for key, (_, score, variance) in node._loop._seen.items():
        assert score == scores[int(key.split(":")[0])] * weight
        assert variance == pytest.approx((0.2 * weight) ** 2)
    node.update_generator_state(study._experiment, study._experiment.lookup_data())
    assert len(node._loop._seen) == 5


def test_pending(config):
    node = Node(config=config)
    study = client(node)
    trials = study.get_next_trials(max_trials=3)
    assert len(trials) == 3
    assert len({tuple(params.values()) for params in trials.values()}) == 3
    indices = list(trials)
    study.mark_trial_failed(indices[0])
    study.mark_trial_abandoned(indices[1])
    study.complete_trial(indices[2], raw_data={"loss": 2.5})
    study.get_next_trials(max_trials=1)
    assert len(node._loop._seen) == 1
    assert next(iter(node._loop._seen.values()))[2] is None


def test_restart(config):
    node = Node(config=config)
    study = client(node)
    for index in study.get_next_trials(max_trials=2):
        study.complete_trial(index, raw_data={"loss": (1.0, 0.0)})
    restored = Node(config=config)
    restored.update_generator_state(study._experiment, study._experiment.lookup_data())
    assert len(restored._loop._seen) == 2
    proposal = restored.get_next_candidate([])
    assert -2 <= proposal["x"] <= 3 and 0 <= proposal["y"] <= 1


def test_contract(config):
    node = Node(config=config)
    with pytest.raises(RuntimeError, match="update"):
        node.get_next_candidate([])
    study = client(node)
    study.get_next_trials(max_trials=1)
    study.configure_optimization(objective="loss")
    with pytest.raises(ValueError, match="contract changes"):
        study.get_next_trials(max_trials=1)
    multi = client(Node(config=config), objective="loss, other")
    with pytest.raises(ValueError, match="one unscalarized"):
        multi.get_next_trials(max_trials=1)


def test_missing(config):
    node = Node(config=config)
    study = client(node)
    for index in study.get_next_trials(max_trials=1):
        study.complete_trial(index)
    with pytest.raises(ValueError, match="exactly one"):
        study.get_next_trials(max_trials=1)


def test_corrections(config):
    node = Node(config=config)
    study = client(node)
    for index in study.get_next_trials(max_trials=1):
        study.complete_trial(index, raw_data={"loss": (1.0, 0.1)})
    node.update_generator_state(study._experiment, study._experiment.lookup_data())
    study.attach_data(index, raw_data={"loss": (2.0, 0.1)})
    with pytest.raises(ValueError, match="observations changed"):
        node.update_generator_state(study._experiment, study._experiment.lookup_data())
