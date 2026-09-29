"""CPU coverage of the BO orchestration contract without the native extension."""

from itertools import product
import hashlib
import json
import sys
from dataclasses import FrozenInstanceError, asdict
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from click.testing import CliRunner

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("safetensors")

from ops.flame import bo, model
from ops.flame.config import ITERATION, MODEL_ID, REVISION, Config
from ops.flame.layout import Layout


from bo_fixtures import (
    cpu,
    bf16,
    FakeProposals,
    FakeSearchState,
    FakePairedSearch,
    FakeProblemEvaluator,
    minibatch_baseline,
    layout,
    flat,
    tiny_model,
)


def test_pairoverride(layout, flat):
    proposals = [
        FakeProposals(flat, [(1, 0.01), (1, 0.01)], radius=0.005),
        FakeProposals(flat, [(1, 0.01), (1, 0.01)], radius=0.0025),
    ]
    search = FakePairedSearch(flat, proposals, best=-0.1)
    evaluate = FakeProblemEvaluator([[4, 6], [3.5, 5.5], [1, 3], [2, 4]])
    settings = bo.Settings(evaluations=3, minibatch_size=2)
    events = []
    result = bo.optimize(
        search,
        layout,
        evaluate,
        settings,
        -0.1,
        events.append,
        baseline=minibatch_baseline(settings),
    )
    assert [event["accepted"] for event in events[1:]] == [True, False]
    assert evaluate.batches[0] == evaluate.batches[1]
    assert evaluate.batches[2] == evaluate.batches[3]
    assert evaluate.batches[0] != evaluate.batches[2]
    assert len(evaluate.batches) == 2 * (settings.evaluations - 1)
    for step in range(settings.evaluations - 1):
        assert (
            sum(len(batch) for batch in evaluate.batches[2 * step : 2 * step + 2]) == 4
        )
    assert [call[0] for call in search.calls] == ["enable_relative"] + [
        "incumbent",
        "ask",
        "tell_relative",
        "sync",
    ] * 2
    tells = [call for call in search.calls if call[0] == "tell_relative"]
    assert tells[0][1:] == (-4.5, 0.75, -5.0, 0.75, 0.5, 0.0, True, False)
    assert tells[1][1:] == (-3.0, 0.75, -2.0, 0.75, -1.0, 0.0, False, True)
    assert search.length == 0.005 and search.readbacks == 0
    assert result["base_version"] == 1 and result["last_accepted_step"] == 1
    assert result["evaluations"] == 3 and result["objective_evaluations"] == 5
    assert result["best_reward"] == -2.0
    assert result["final_minibatch"]["losses"] == [1, 3]
    assert result["best_reward_is_full_objective"] is False
    assert all(event["y_scale"] == 1.0 for event in events[1:])


def test_nullcandidate(layout, flat):
    search = FakePairedSearch(flat, [FakeProposals(flat, [(0, 0.0), (0, 0.0)])])
    evaluate = FakeProblemEvaluator([[1, 3]])
    settings = bo.Settings(evaluations=3, minibatch_size=2)
    events = []
    result = bo.optimize(
        search,
        layout,
        evaluate,
        settings,
        -0.1,
        events.append,
        baseline=minibatch_baseline(settings),
    )
    assert result["stop_reason"] == "selected_proposal_unchanged"
    assert result["evaluations"] == 1 and result["objective_evaluations"] == 2
    assert not events[1]["accepted"] and not events[1]["evaluated"]


def test_offset(layout, flat):
    runs = []
    for offsets in ([0, 0], [64, 128]):
        search = FakePairedSearch(
            flat,
            [FakeProposals(flat, [(1, 0.01), (1, 0.01)]) for _ in range(2)],
        )
        losses = []
        for offset in offsets:
            losses.extend(([offset + 4, offset + 8], [offset + 4.25, offset + 8.5]))
        settings = bo.Settings(
            evaluations=3, minibatch_size=2, paired_epistemic_scale=0.25
        )
        bo.optimize(
            search,
            layout,
            FakeProblemEvaluator(losses),
            settings,
            -0.1,
            lambda event: None,
            baseline=minibatch_baseline(settings),
        )
        asks = [call for call in search.calls if call[0] == "ask"]
        assert all(call[2]["epistemic_scale"] == 0.25 for call in asks)
        assert all(call[2]["aleatoric_scale"] == 0.0 for call in asks)
        tells = [call for call in search.calls if call[0] == "tell_relative"]
        runs.append((asks, [call[5:] for call in tells]))
    assert runs[0] == runs[1]


def test_budget(layout, flat):
    search = FakePairedSearch(
        flat,
        [FakeProposals(flat, [(1, 0.01), (1, 0.01)]) for _ in range(4)],
    )
    evaluate = FakeProblemEvaluator([[1, 2], [2, 3]] * 4)
    settings = bo.Settings(evaluations=5, minibatch_size=2)
    events = []
    result = bo.optimize(
        search,
        layout,
        evaluate,
        settings,
        -0.1,
        events.append,
        baseline=minibatch_baseline(settings),
    )
    assert [event["next_reference_radius"] for event in events[1:]] == [
        0.01,
        0.01,
        0.01,
        0.005,
    ]
    assert result["base_version"] == 0
    assert result["objective_evaluations"] == 9
    assert sum(map(len, evaluate.batches)) == 16
    assert search.restarts == search.readbacks == 0


def test_refreshreuse(layout, flat):
    search = FakePairedSearch(
        flat,
        [FakeProposals(flat, [(1, 0.01), (1, 0.01)]) for _ in range(3)],
    )
    evaluate = FakeProblemEvaluator(
        [[0.0, 0.0], [0.05, 0.05], [0.1, 0.1], [0.05, 0.05]]
    )
    settings = bo.Settings(evaluations=4, minibatch_size=2, minibatch_refresh=2)
    events = []
    result = bo.optimize(
        search,
        layout,
        evaluate,
        settings,
        -0.1,
        events.append,
        baseline=minibatch_baseline(settings),
    )

    assert len(evaluate.batches) == 4
    assert evaluate.batches[0] == evaluate.batches[1]
    assert evaluate.batches[2] == evaluate.batches[3]
    assert evaluate.batches[0] != evaluate.batches[2]
    assert [event["minibatch_reused"] for event in events[1:]] == [True, True, False]
    assert result["objective_evaluations"] == 5
    assert [call[0] for call in search.calls].count("incumbent") == 1


@pytest.mark.parametrize(
    "candidates,expected_failures,expected_outcomes",
    [
        (
            [
                [3, 3],
                [0, 3],
                [1, 4],
                [2, 2],
                [2 + 1e-8] * 2,
                [2 - 1e-8] * 2,
                [3, 3],
                [3, 3],
                [1, 4],
                [3, 3],
            ],
            [1, 1, 1, 1, 1, 1, 2, 3, 3, 0],
            [
                "deteriorated",
                *(["inconclusive"] * 5),
                "deteriorated",
                "deteriorated",
                "inconclusive",
                "deteriorated",
            ],
        ),
        (
            [[3, 3], [3, 3], [1, 4], [1, 1], [2, 2], [3, 3], [3, 3], [3, 3], [3, 3]],
            [1, 2, 2, 0, 0, 1, 2, 3, 0],
            [
                "deteriorated",
                "deteriorated",
                "inconclusive",
                "accepted",
                "inconclusive",
                *(["deteriorated"] * 4),
            ],
        ),
    ],
    ids=["inconclusive-holds-failures", "acceptance-resets-failures"],
)
def test_outcomes(layout, flat, candidates, expected_failures, expected_outcomes):
    search = FakePairedSearch(
        flat,
        [FakeProposals(flat, [(1, 0.01), (1, 0.01)]) for _ in candidates],
    )
    evaluate = FakeProblemEvaluator(
        [losses for candidate in candidates for losses in ([2, 2], candidate)]
    )
    settings = bo.Settings(evaluations=len(candidates) + 1, minibatch_size=2)
    events, failures = [], []

    def emit(event):
        if event["event"] == "proposal":
            events.append(event)
            failures.append(search.failures)

    result = bo.optimize(
        search,
        layout,
        evaluate,
        settings,
        -0.1,
        emit,
        baseline=minibatch_baseline(settings),
    )
    assert failures == expected_failures
    assert [event["outcome"] for event in events] == expected_outcomes
    assert [event["counted_failure"] for event in events] == [
        outcome == "deteriorated" for outcome in expected_outcomes
    ]
    assert [event["next_reference_radius"] for event in events] == (
        [0.01] * (len(candidates) - 1) + [0.005]
    )
    assert [event["history_len"] for event in events] == [
        1 if outcome == "accepted" else 2 for outcome in expected_outcomes
    ]
    tells = [call for call in search.calls if call[0] == "tell_relative"]
    assert len(tells) == len(candidates)
    assert [call[-1] for call in tells] == [
        event["counted_failure"] for event in events
    ]
    assert result["base_version"] == expected_outcomes.count("accepted")
    assert result["objective_evaluations"] == 1 + 2 * len(candidates)
    assert sum(map(len, evaluate.batches)) == 4 * len(candidates)
    assert all(
        evaluate.batches[index] == evaluate.batches[index + 1]
        for index in range(0, len(evaluate.batches), 2)
    )


@pytest.mark.parametrize(
    "losses,stage", [([[np.nan, 1]], "incumbent"), ([[1, 2], [np.inf, 1]], "FLAME")]
)
def test_nonfiniteloss(layout, flat, losses, stage):
    search = FakePairedSearch(flat, [FakeProposals(flat, [(1, 0.01), (1, 0.01)])])
    settings = bo.Settings(evaluations=2, minibatch_size=2)
    with pytest.raises(RuntimeError, match=stage):
        bo.optimize(
            search,
            layout,
            FakeProblemEvaluator(losses),
            settings,
            -0.1,
            lambda event: None,
            baseline=minibatch_baseline(settings),
        )
    assert not any(call[0] == "tell_relative" for call in search.calls)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"minibatch_size": 1},
        {"minibatch_size": True},
        {"minibatch_size": 2.0},
        {"minibatch_size": 2, "sampler": "gaussian"},
        {"minibatch_seed": -1},
        {"minibatch_seed": True},
        {"minibatch_refresh": 0},
        {"minibatch_refresh": True},
        {"minibatch_refresh": 1.0},
        {"minibatch_refresh": 2},
        {"acceptance_se": -1},
        {"acceptance_se": float("nan")},
        {"acceptance_se": float("inf")},
        {"acceptance_se": True},
    ],
)
def test_batch(kwargs):
    with pytest.raises(ValueError):
        bo.Settings(**kwargs)


def test_rng():
    first = bo.minibatch_rng(0).bit_generator.random_raw(8)
    np.testing.assert_array_equal(
        first, bo.minibatch_rng(0).bit_generator.random_raw(8)
    )
    assert not np.array_equal(
        first, np.random.default_rng(0).bit_generator.random_raw(8)
    )
    assert not np.array_equal(first, bo.minibatch_rng(1).bit_generator.random_raw(8))
