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


def test_seeds(layout, flat):
    def run(seed, reference_seed=0):
        search = FakeSearchState(
            flat, [FakeProposals(flat, [(1, 1.0), (0, 0.0)]) for _ in range(3)]
        )
        events = []
        bo.optimize(
            search,
            layout,
            lambda _: jnp.asarray(-10.0),
            bo.Settings(evaluations=4, seed=seed, reference_seed=reference_seed),
            -10.0,
            events.append,
        )
        return [(row["seed_root"], row["draw_seed"]) for row in events[1:]]

    first = run(42)
    assert first == run(42)
    assert first == run(42, reference_seed=123)
    assert first != run(43)
    expected = np.random.PCG64(42).random_raw(6).reshape(3, 2).tolist()
    assert first == [tuple(pair) for pair in expected]


def test_updates(layout, flat):
    specs = [(0, 0.75, 0.005), (3, 0.0, 0.01), (1, 0.75, 0.01), (2, 0.0, 0.005)]
    proposals = [
        FakeProposals(
            flat + bf16(0.25),
            [(2, 0.125), (1, 0.0625)],
            candidate_index=index,
            persistence=persistence,
            radius=radius,
        )
        for index, persistence, radius in specs
    ]
    search = FakeSearchState(flat, proposals)
    rewards = iter([-9.0, -11.0, -8.0, -8.0])
    events = []
    settings = bo.Settings(evaluations=5, reference_seed=43)
    result = bo.optimize(
        search,
        layout,
        lambda _: jnp.asarray(next(rewards)),
        settings,
        -10.0,
        events.append,
    )
    rows = events[1:]
    assert result["base_version"] == 2 and result["evaluations"] == 5
    assert [row["accepted"] for row in rows] == [True, False, True, False]
    assert [row["candidate_index"] for row in rows] == [0, 3, 1, 2]
    assert [row["persistence"] for row in rows] == [0.75, 0.0, 0.75, 0.0]
    assert all("flip_probability" not in row for row in rows)
    assert [row["radius"] for row in rows] == [0.005, 0.01, 0.01, 0.005]
    assert [row["reference_version"] for row in rows] == [0, 1, 1, 2]
    assert [row["next_reference_version"] for row in rows] == [1, 1, 2, 2]
    assert [row["reference_radius"] for row in rows] == [0.01, 0.005, 0.005, 0.01]
    assert [row["next_reference_radius"] for row in rows] == [0.005, 0.005, 0.01, 0.01]
    assert [row["history_len"] for row in rows] == [2, 2, 2, 2]
    assert [row["y_scale"] for row in rows] == [1e-6, 0.5, 1.0, 1.5]
    assert all(row["restarts"] == 0 for row in rows)
    assert all(row["reference_seed"] == 43 for row in events)
    assert all(row["sampler"] == "correlated" for row in events)
    assert all(row["controller"] == "acquisition_selected_radius" for row in events)
    assert all(call[1][:3] == (1, 4, 2) for call in search.calls[::3])
    json.dumps(events, allow_nan=False)


@pytest.mark.parametrize(
    "sampler,geometry",
    [
        ("correlated", []),
        ("correlated", [(0, 0.75), (1, 0.75)]),
        ("correlated", [(-1, 0.75)]),
        ("correlated", [(4, 0.0)]),
        ("correlated", [(True, 0.75)]),
        ("correlated", [(0.0, 0.75)]),
        ("correlated", [(0, 0.0)]),
        ("correlated", [(0, 0.125)]),
        ("correlated", [(2, 0.75)]),
        ("correlated", [(0, float("nan"))]),
        ("independent", [(0, 0.5)]),
        ("independent", [(4, 0.0)]),
        ("gaussian", [(0, 0.75)]),
        ("gaussian", [(4, 0.0)]),
        ("gaussian", [(0, float("nan"))]),
    ],
)
def test_geometry(layout, flat, sampler, geometry):
    proposal = FakeProposals(flat, [(1, 1.0), (0, 0.0)])
    proposal.selected_geometry = geometry
    search = FakeSearchState(flat, [proposal], sampler=sampler)
    with pytest.raises(RuntimeError, match="geometry|diagnostics"):
        bo.optimize(
            search,
            layout,
            lambda _: pytest.fail("unexpected evaluation"),
            bo.Settings(sampler=sampler),
            -10.0,
            lambda _: None,
        )
    assert proposal.exports == 0
    assert [call[0] for call in search.calls] == ["ask"]


@pytest.mark.parametrize("restart_before_ask", [False, True])
def test_restart(layout, flat, restart_before_ask):
    class RestartingSearch(FakeSearchState):
        def sync(self):
            result = super().sync()
            self.restarts += 1
            return result

    search = RestartingSearch(flat, [FakeProposals(flat, [(1, 1.0), (0, 0.0)])])
    search.restarts = int(restart_before_ask)
    with pytest.raises(RuntimeError, match="must not use TuRBO restarts"):
        bo.optimize(
            search,
            layout,
            lambda _: jnp.asarray(-10.0),
            bo.Settings(),
            -10.0,
            lambda _: None,
        )
    if restart_before_ask:
        assert not search.calls


def test_manifest(layout, flat, tmp_path):
    torch = pytest.importorskip("torch")
    from safetensors.torch import load_file

    improved = flat + bf16(0.5)
    proposals = [
        FakeProposals(improved, [(2, 0.5), (1, 0.25)]),
        FakeProposals(flat, [(2, 0.5), (1, 0.25)]),
    ]
    search = FakeSearchState(flat, proposals)
    rewards = iter([-9.0, -11.0])
    result = bo.optimize(
        search,
        layout,
        lambda _: jnp.asarray(next(rewards)),
        bo.Settings(evaluations=3),
        -10.0,
        lambda _: None,
    )
    source = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "iteration": ITERATION,
        "config": asdict(Config()),
        "complete": True,
        "tensors": {"old": {"file": "old.safetensors"}},
    }
    original = json.dumps(source, sort_keys=True)
    bo.save_best(search, layout, source, tmp_path, result)
    assert search.readbacks == 1
    assert json.dumps(source, sort_keys=True) == original
    target = tmp_path / "best"
    manifest = json.loads((target / "manifest.json").read_text())
    assert manifest["complete"] is True
    assert manifest["optimization"] == {**result, "events": "../events.jsonl"}
    for key in ("model_id", "revision", "iteration", "config"):
        assert manifest[key] == source[key]
    assert list(manifest["tensors"]) == [block.name for block in layout.blocks]
    restored = restore_tensors(layout, target, manifest, torch, load_file)
    np.testing.assert_array_equal(layout.flatten(restored), improved)


def restore_tensors(layout, target, manifest, torch, load_file):
    restored = {}
    for index, block in enumerate(layout.blocks):
        record = manifest["tensors"][block.name]
        assert record["file"] == f"{index:03d}.safetensors"
        path = target / record["file"]
        assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        tensors = load_file(str(path))
        assert list(tensors) == [block.name]
        tensor = tensors[block.name]
        assert tensor.dtype == torch.bfloat16 and tuple(tensor.shape) == block.shape
        restored[block.name] = bf16(tensor.float().numpy())
    return restored
