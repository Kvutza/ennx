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


def install_extension(monkeypatch, tmp_path, experimental):
    extension_path = tmp_path / "ennx_rust.so"
    extension_path.write_bytes(b"test extension")
    extension = ModuleType("ennx.ennx_rust")
    extension.__file__ = str(extension_path)
    package = ModuleType("ennx")
    package.ennx_rust = extension
    monkeypatch.setitem(sys.modules, "ennx", package)
    monkeypatch.setitem(sys.modules, "ennx.ennx_rust", extension)
    monkeypatch.setitem(sys.modules, "ennx.experimental", experimental)


def test_initialsave(tiny_model, tmp_path, monkeypatch):
    config, params = tiny_model
    (tmp_path / "manifest.json").write_text(
        json.dumps({"revision": REVISION, "complete": True})
    )
    experimental = ModuleType("ennx.experimental")
    calls = []
    saved = []

    class Evaluator:
        def __call__(self, flat):
            pytest.fail("must not score the full training pool")

        def losses(self, flat, indices):
            assert flat.ndim == 1 and len(indices) == 2
            calls.append(("baseline", indices.tolist()))
            return np.asarray([1.0, 3.0])

        def compile(self, signature):
            assert signature.shape == (1, sum(p.size for p in params.values()))
            calls.append(("compile",))
            return self

        def memory_analysis(self):
            return SimpleNamespace(temp_size_in_bytes=0, output_size_in_bytes=4)

    class NativeSearch:
        def __init__(self, *args, **kwargs):
            assert args[1] == -2.0 and kwargs["base_variance"] == 0.5
            calls.append(("search",))

        def read_best(self):
            pytest.fail("saving is mocked")

        def incumbent(self):
            pytest.fail("optimization is mocked")

        def enable_relative(self):
            pytest.fail("optimization is mocked")

        def tell_relative(self, *, reject_is_failure=True):
            pytest.fail("optimization is mocked")

    experimental.SearchState = NativeSearch
    experimental.ParamBlock = lambda *args: args
    install_extension(monkeypatch, tmp_path, experimental)
    monkeypatch.setattr(
        jax,
        "devices",
        lambda: [SimpleNamespace(platform="gpu", device_kind="Tesla T4", id=0)],
    )
    monkeypatch.setenv("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    monkeypatch.setattr(bo, "Config", lambda: config)
    monkeypatch.setattr(model, "load_checkpoint", lambda _: (config, params))
    monkeypatch.setattr(bo, "SolutionEvaluator", Evaluator)
    monkeypatch.setattr(bo, "loss_evaluator", lambda *args: Evaluator())
    monkeypatch.setattr(
        bo.subprocess, "check_output", lambda *args, **kwargs: "15360\n"
    )
    monkeypatch.setattr(bo, "save_best", lambda *args: saved.append(args))
    settings = bo.Settings(evaluations=3, minibatch_size=2)
    document = {
        "format": "ennx.solution_tokens.v1",
        "provenance": {"dataset": "test"},
        "examples": [
            {"id": str(i), "tokens": [1, 2, 3], "loss_mask": [False, False, True]}
            for i in range(4)
        ],
    }

    def optimize(search, layout, evaluate, settings, base_reward, emit, *, baseline):
        assert calls[-1] == ("search",) and not saved
        assert baseline["losses"] == [1, 3] and baseline["variance"] == 0.5
        assert baseline["indices"] == calls[0][1]
        emit({"event": "baseline", "evaluations": 1, "reward": base_reward})
        return {"best_reward": -2.0, "evaluations": 3}

    monkeypatch.setattr(bo, "optimize", optimize)
    bo.run(tmp_path, document, tmp_path / "result", settings)
    assert [call[0] for call in calls] == ["baseline", "compile", "search"]
    assert len(saved) == 1
    report = json.loads((tmp_path / "result/run.json").read_text())
    assert (
        report["objective"]["normalization"] == "mean_per_problem_solution_token_loss"
    )
    assert (
        report["objective"]["checkpoint_policy"]
        == "incumbent_in_gpu_memory_single_final_write"
    )
    assert report["status"] == "complete"
    assert report["objective"]["rejection_policy"] == "deterioration"
    assert report["objective"]["screening"] == (
        "paired_standard_error_heuristic_not_a_statistical_guarantee"
    )


def test_nullproposal(layout, flat):
    proposal = FakeProposals(flat, [(0, 0.0), (0, 0.0)])
    search = FakeSearchState(flat, [proposal])
    events = []

    def evaluate(_):
        pytest.fail("null proposal must not evaluate the objective")

    result = bo.optimize(search, layout, evaluate, bo.Settings(), -10.0, events.append)
    assert result == {
        "evaluations": 1,
        "best_reward": -10.0,
        "base_version": 0,
        "stop_reason": "selected_proposal_unchanged",
    }
    assert proposal.exports == 0
    assert [call[0] for call in search.calls] == ["ask", "tell", "sync"]
    assert search.calls[1][2:] == ([-10.0], [0.0])
    assert events[1]["evaluated"] is False
    assert events[1]["evaluate_seconds"] == 0


@pytest.mark.parametrize(
    "changes",
    [
        [],
        [(0, 0.0)],
        [(0, 0.0)] * 3,
        [(-1, 0.0), (0, 0.0)],
        [(3, 1.0), (0, 0.0)],
        [(1, -1.0), (0, 0.0)],
        [(1, float("nan")), (0, 0.0)],
        [(1, float("inf")), (0, 0.0)],
    ],
)
def test_baddiags(layout, flat, changes):
    proposal = FakeProposals(flat, changes)
    search = FakeSearchState(flat, [proposal])

    def evaluate(_):
        pytest.fail("invalid diagnostics must not evaluate the objective")

    with pytest.raises(RuntimeError, match="diagnostics"):
        bo.optimize(search, layout, evaluate, bo.Settings(), -10.0, lambda _: None)
    assert proposal.exports == 0
    assert [call[0] for call in search.calls] == ["ask"]


@pytest.mark.parametrize(
    "changes,kwargs",
    [
        ([(0.5, 1.0), (0, 0.0)], {}),
        ([(1, 1.0), (0, 0.0)], {"score": float("nan")}),
        ([(1, 1.0), (0, 0.0)], {"score": float("inf")}),
        ([(1, 1.0), (0, 0.0)], {"radius": float("nan")}),
        ([(1, 1.0), (0, 0.0)], {"radius": float("inf")}),
        ([(1, 1.0), (0, 0.0)], {"radius": 0}),
        ([(1, 1.0), (0, 0.0)], {"radius": -1}),
    ],
)
def test_diagfields(layout, flat, changes, kwargs):
    search = FakeSearchState(flat, [FakeProposals(flat, changes, **kwargs)])
    with pytest.raises(RuntimeError, match="diagnostics"):
        bo.optimize(
            search,
            layout,
            lambda _: jnp.asarray(-9.0),
            bo.Settings(evaluations=2),
            -10.0,
            lambda _: None,
        )
    assert [call[0] for call in search.calls] == ["ask"]


@pytest.mark.parametrize("reward", [float("nan"), float("inf"), -float("inf")])
def test_baseline(layout, flat, reward):
    search = FakeSearchState(flat, [])
    events = []
    with pytest.raises(ValueError, match="Baseline reward must be finite"):
        bo.optimize(
            search,
            layout,
            lambda _: pytest.fail("unexpected evaluation"),
            bo.Settings(),
            reward,
            events.append,
        )
    assert search.calls == [] and events == []


def test_scale(layout, flat):
    class RestartingSearch(FakeSearchState):
        def sync(self):
            accepted = super().sync()
            self.restarts += 1
            self.history_len = 1
            return accepted

    search = RestartingSearch(
        flat, [FakeProposals(flat, [(1, 1.0), (0, 0.0)]) for _ in range(2)]
    )
    events = []
    bo.optimize(
        search,
        layout,
        lambda _: jnp.asarray(-20.0),
        bo.Settings(evaluations=3, sampler="independent"),
        -10.0,
        events.append,
    )
    assert [row["y_scale"] for row in events[1:]] == [1e-6, 1e-6]
    assert [row["restarts"] for row in events[1:]] == [1, 2]
    assert [row["history_len"] for row in events[1:]] == [1, 1]


@pytest.mark.parametrize("reward", [float("nan"), float("inf"), -float("inf")])
def test_reward(layout, flat, reward):
    proposal = FakeProposals(flat, [(1, 1.0), (0, 0.0)])
    search = FakeSearchState(flat, [proposal])
    events = []
    with pytest.raises(RuntimeError, match="Nonfinite FLAME loss"):
        bo.optimize(
            search,
            layout,
            lambda _: jnp.asarray(reward),
            bo.Settings(evaluations=2),
            -10.0,
            events.append,
        )
    assert proposal.exports == 1
    assert [call[0] for call in search.calls] == ["ask"]
    assert search.best == -10.0
    assert len(events) == 1


def test_exception(layout, flat):
    search = FakeSearchState(flat, [FakeProposals(flat, [(1, 1.0), (0, 0.0)])])

    def evaluate(_):
        raise ValueError("objective failed")

    with pytest.raises(ValueError, match="objective failed"):
        bo.optimize(search, layout, evaluate, bo.Settings(), -10.0, lambda _: None)
    assert [call[0] for call in search.calls] == ["ask"]


@pytest.mark.parametrize("fail", [False, True])
def test_batchcleanup(layout, flat, monkeypatch, fail):
    imported = []
    from_dlpack = jax.dlpack.from_dlpack

    def capture(proposals):
        batch = from_dlpack(proposals)
        imported.append(batch)
        return batch

    monkeypatch.setattr(jax.dlpack, "from_dlpack", capture)
    search = FakeSearchState(flat, [FakeProposals(flat, [(1, 1.0), (0, 0.0)])])

    def evaluate(candidate):
        assert candidate is imported[-1]
        assert candidate.shape == (1, layout.size)
        if fail:
            raise ValueError("objective failed")
        return jnp.asarray(-9.0)

    if fail:
        with pytest.raises(ValueError, match="objective failed"):
            bo.optimize(
                search,
                layout,
                evaluate,
                bo.Settings(evaluations=2),
                -10.0,
                lambda _: None,
            )
    else:
        bo.optimize(
            search, layout, evaluate, bo.Settings(evaluations=2), -10.0, lambda _: None
        )
    assert len(imported) == 1
    assert imported[0].is_deleted()


@pytest.mark.parametrize(
    "solution_objective,sampler,tolerance",
    [
        (case_0, *case_1)
        for case_0, case_1 in product(
            [False, True],
            [
                ("correlated", None),
                ("gaussian", None),
                ("gaussian", 7),
                ("independent", 4),
                ("independent", 7),
            ],
        )
    ],
)
def test_metadata(
    tiny_model, tmp_path, monkeypatch, sampler, tolerance, solution_objective
):
    config, params = tiny_model
    checkpoint = metadata_checkpoint(tmp_path)
    experimental = ModuleType("ennx.experimental")
    calls = []
    stages = []
    size = sum(value.size for value in params.values())

    def compiled_evaluate(batch):
        assert batch.shape == (1, size)
        return jnp.asarray(-10.0)

    def baseline_evaluate(flat):
        assert flat.shape == (size,)
        stages.append("baseline")
        return jnp.asarray(-10.0)

    def compile():
        stages.append("compile")
        return compiled_evaluate

    def lower(signature):
        assert stages == ["baseline"]
        assert isinstance(signature, jax.ShapeDtypeStruct)
        assert signature.shape == (1, size)
        assert signature.dtype == jnp.bfloat16
        stages.append("lower")

        return SimpleNamespace(compile=compile)

    baseline_evaluate.lower = lower

    class NativeSearch:
        def __init__(self, *args, **kwargs):
            assert stages == ["baseline", "lower", "compile"]
            stages.append("search")
            calls.append((args, kwargs))

        def read_best(self):
            pytest.fail("saving is mocked in this orchestration test")

    experimental.SearchState = NativeSearch
    experimental.ParamBlock = lambda *args: args
    install_extension(monkeypatch, tmp_path, experimental)
    device = SimpleNamespace(platform="gpu", device_kind="Tesla T4", id=0)
    monkeypatch.setattr(jax, "devices", lambda: [device])
    monkeypatch.setenv("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    monkeypatch.setattr(bo, "Config", lambda: config)
    monkeypatch.setattr(model, "load_checkpoint", lambda _: (config, params))
    monkeypatch.setattr(bo, "loss_evaluator", lambda *args: baseline_evaluate)
    monkeypatch.setattr(
        bo.subprocess, "check_output", lambda *args, **kwargs: "15360\n"
    )
    settings = bo.Settings(
        sampler=sampler, failure_tolerance=tolerance, reference_seed=37
    )
    result = {
        "evaluations": 1,
        "best_reward": -10.0,
        "base_version": 0,
        "stop_reason": "selected_proposal_unchanged",
    }

    def optimize(search, layout, evaluate, received, base_reward, emit):
        assert isinstance(search, NativeSearch)
        assert evaluate is compiled_evaluate
        assert stages == ["baseline", "lower", "compile", "search"]
        stages.append("optimize")
        assert calls[0][0][0].is_deleted()
        assert received == settings and base_reward == -10.0
        emit({"event": "baseline", "evaluations": 1, "reward": base_reward})
        return result

    saved = []
    monkeypatch.setattr(bo, "optimize", optimize)
    monkeypatch.setattr(bo, "save_best", lambda *args: saved.append(args))
    output = tmp_path / "run"
    tokens = (
        {
            "format": "ennx.solution_tokens.v1",
            "provenance": {"dataset": "test"},
            "examples": [
                {"id": "a", "tokens": [1, 2, 3], "loss_mask": [False, False, True]}
            ],
        }
        if solution_objective
        else [[1, 2, 3]]
    )
    assert bo.run(checkpoint, tokens, output, settings) == result
    assert len(calls) == len(saved) == 1
    assert stages == ["baseline", "lower", "compile", "search", "optimize"]
    args = assert_search(calls[0], params, settings, sampler)
    assert_metadata(
        output,
        tokens,
        settings,
        sampler=sampler,
        solution_objective=solution_objective,
        args=args,
        saved=saved,
        result=result,
    )


def assert_search(call, params, settings, sampler):
    args, kwargs = call
    assert args[0].size == sum(value.size for value in params.values())
    assert args[0].is_deleted()
    assert args[1] == -10.0 and args[3] == 2
    assert kwargs == {
        "max_pending": 1,
        "length_init": 0.01,
        "length_min": 0.0001,
        "length_max": 0.08,
        "sampler": sampler,
        "reference_seed": 37,
        "failure_tolerance": settings.failure_tolerance,
    }
    return args


def metadata_checkpoint(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    source = {"revision": REVISION, "complete": True}
    (checkpoint / "manifest.json").write_text(json.dumps(source))
    return checkpoint


def assert_metadata(
    output, tokens, settings, *, sampler, solution_objective, args, saved, result
):
    metadata = json.loads((output / "run.json").read_text())
    assert metadata["status"] == "complete"
    assert metadata["tokens"] == tokens
    if solution_objective:
        assert metadata["objective"]["scored_tokens"] == 1
        assert metadata["objective"]["input_tokens"] == 3
        assert metadata["objective"]["kind"] == "solution_token_cross_entropy"
    else:
        assert metadata["objective"] == {"kind": "all_token_cross_entropy"}
    assert metadata["sampler"] == sampler
    assert metadata["controller"] == settings.controller
    assert metadata["settings"] == asdict(settings)
    assert metadata["reference_seed"] == 37
    assert metadata["reference_storage"] == (
        "dense_bf16" if sampler == "correlated" else None
    )
    assert metadata["reference_bytes"] == (
        2 * args[0].size if sampler == "correlated" else 0
    )
    assert metadata["forward_headroom_bytes"] == bo.FORWARD_HEADROOM_BYTES[sampler]
    assert (
        metadata["noise_law"]
        == {
            "correlated": "correlated_gaussian",
            "gaussian": "independent_gaussian",
            "independent": "legacy_independent_signs",
        }[sampler]
    )
    assert metadata["scale_reference"] == "initial_checkpoint"
    assert metadata["estimated_memory_bytes"] == bo.memory_budget(
        args[0].size, 2, sampler
    ) + (14 if solution_objective else 0)
    assert metadata["extension_sha256"] == hashlib.sha256(b"test extension").hexdigest()
    assert saved[0][-1] == result
