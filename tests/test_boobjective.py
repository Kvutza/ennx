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


@pytest.mark.parametrize("batched", [False, True])
def test_token(tiny_model, batched):
    config, params = tiny_model
    layout = Layout.from_params(params)
    tokens = np.array([[0, 1, 2], [2, 3, 4]], dtype=np.int64)
    evaluate = bo.loss_evaluator(layout, tokens, config)
    if batched:
        evaluate = evaluate.lower(
            jax.ShapeDtypeStruct((1, layout.size), jnp.bfloat16)
        ).compile()
    flat = layout.flatten(params)
    for candidate in (flat, flat * bf16(2)):
        argument = candidate.reshape((1, layout.size)) if batched else candidate
        reward = evaluate(argument).block_until_ready()
        assert reward.shape == () and reward.dtype == jnp.float32
        logits = np.asarray(model.forward(layout.unflatten(candidate), tokens, config))[
            :, :-1
        ]
        shifted = logits - logits.max(axis=-1, keepdims=True)
        log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
        expected = np.take_along_axis(log_probs, tokens[:, 1:, None], axis=-1).mean()
        np.testing.assert_allclose(reward, expected, rtol=1e-6)
        assert float(reward) < 0


@pytest.mark.parametrize(
    "tokens",
    [
        [],
        [0, 1],
        [[[0, 1]]],
        [[]],
        [[1]],
        [[0, 1, 2, 3, 4]],
        [[0.0, 1.0]],
        [[True, False]],
        [[-1, 0]],
        [[0, 8]],
    ],
)
def test_loss(tiny_model, tokens):
    config, params = tiny_model
    with pytest.raises(ValueError):
        bo.loss_evaluator(Layout.from_params(params), tokens, config)


@pytest.mark.parametrize("sampler", ["gaussian", "independent"])
def test_feedback(layout, flat, sampler):
    proposals = [
        FakeProposals(flat + bf16(0.25 * (i + 1)), [(2, 0.125), (1, 0.0625)], seed=i)
        for i in range(3)
    ]
    search = FakeSearchState(flat, proposals, sampler=sampler)
    values = iter([-9.0, -11.0, -8.0])
    seen = []

    def evaluate(candidate):
        assert candidate.shape == (1, layout.size) and candidate.dtype == jnp.bfloat16
        seen.append(np.array(candidate, copy=True))
        return jnp.asarray(next(values))

    events = []
    settings = bo.Settings(evaluations=4, candidates=3, seed=123, sampler=sampler)
    result = bo.optimize(search, layout, evaluate, settings, -10.0, events.append)
    assert result == {
        "evaluations": 4,
        "best_reward": -8.0,
        "base_version": 2,
        "stop_reason": "evaluation_budget",
    }
    assert events[0] == {
        "event": "baseline",
        "evaluations": 1,
        "reward": -10.0,
        "base_version": 0,
        "sampler": sampler,
        "controller": settings.controller,
        "reference_seed": 0,
        "reference_version": None,
        "reference_radius": None,
        "minibatch_refresh": None,
    }
    rows = events[1:]
    assert [row["accepted"] for row in rows] == [True, False, True]
    assert [row["base_version"] for row in rows] == [0, 1, 1]
    assert [row["evaluations"] for row in rows] == [2, 3, 4]
    assert [row["y_scale"] for row in rows] == [1e-6, 0.5, 1.0]
    assert [call[0] for call in search.calls] == ["ask", "tell", "sync"] * 3
    for proposal, candidate, row, ask, tell in zip(
        proposals, seen, rows, search.calls[::3], search.calls[1::3]
    ):
        assert proposal.exports == 1
        np.testing.assert_array_equal(candidate, proposal.array)
        assert ask[1][:3] == (1, 3, 2)
        assert ask[2] == {
            "epistemic_scale": 10000.0,
            "aleatoric_scale": 0.05,
            "y_scale": row["y_scale"],
            "acquisition": "thompson",
            "draw_seed": row["draw_seed"],
        }
        assert tell[1] is proposal and tell[2:] == ([row["reward"]], [0.0])
        assert row["changed"] == 3 and row["evaluated"] is True
        assert row["candidate_index"] == 2 and row["persistence"] == 0.0
        assert row["reference_version"] is row["reference_radius"] is None
        assert row["next_reference_version"] is row["next_reference_radius"] is None
        assert row["realized_distance"] == pytest.approx(
            np.sqrt(0.125 / 4 + 0.0625 / 8)
        )
        assert row["tensors"] == [
            {"name": "a", "changed": 2, "relative_rms": 0.25},
            {"name": "b", "changed": 1, "relative_rms": 0.125},
        ]
        assert all(
            row[key] >= 0 for key in ("ask_seconds", "evaluate_seconds", "tell_seconds")
        )
    json.dumps(events, allow_nan=False)
