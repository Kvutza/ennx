"""CPU coverage of the BO orchestration contract without the native extension."""

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


def test_clihelp():
    result = CliRunner().invoke(bo.main, ["--help"])
    assert result.exit_code == 0
    assert "--failure-tolerance" in result.output
    assert "--sampler" in result.output
    assert "--reference-seed" in result.output
    assert "--rejection-policy" in result.output
    assert "--minibatch-refresh" in result.output
    assert "--backend" in result.output and "[default: jax]" in result.output


@pytest.mark.parametrize("candidates,exit_code", [(4, 0), (3, 2), (9, 2)])
def test_order(tmp_path, monkeypatch, candidates, exit_code):
    tokens = tmp_path / "tokens.json"
    tokens.write_text("[[1, 2, 3]]")
    output = tmp_path / "result"
    calls = []
    monkeypatch.setattr(bo, "run", lambda *args, **kwargs: calls.append((args, kwargs)))
    result = CliRunner().invoke(
        bo.main,
        [
            str(tmp_path),
            "--tokens",
            str(tokens),
            "--output",
            str(output),
            "--candidates",
            str(candidates),
            "--evaluations",
            "4",
            "--seed",
            "17",
        ],
    )
    assert result.exit_code == exit_code, result.output
    if exit_code:
        assert calls == [] and "candidates" in result.output
    else:
        assert calls == [
            (
                (
                    tmp_path,
                    [[1, 2, 3]],
                    output,
                    bo.Settings(candidates=4, evaluations=4, seed=17),
                ),
                {"zero_scale": None},
            )
        ]


@pytest.mark.parametrize(
    "options,expected,error",
    [
        ([], bo.Settings(), None),
        (["--backend", "jax"], bo.Settings(), None),
        (
            ["--backend", "native"],
            bo.Settings(backend="native", minibatch_size=2),
            None,
        ),
        (
            ["--backend", "native", "--minibatch-size", "3"],
            bo.Settings(backend="native", minibatch_size=3),
            None,
        ),
        (
            ["--backend", "native", "--sampler", "gaussian"],
            None,
            "Native BO requires correlated sampling",
        ),
        (
            ["--backend", "native", "--minibatch-size", "1"],
            None,
            "minibatch_size",
        ),
        (["--minibatch-size", "2"], bo.Settings(minibatch_size=2), None),
        (
            ["--minibatch-size", "2", "--rejection-policy", "deterioration"],
            bo.Settings(minibatch_size=2, rejection_policy="deterioration"),
            None,
        ),
        (
            ["--minibatch-size", "2", "--rejection-policy", "all"],
            bo.Settings(minibatch_size=2, rejection_policy="all"),
            None,
        ),
        (["--rejection-policy", "unknown"], None, "Invalid value"),
        (
            ["--minibatch-size", "2", "--failure-tolerance", "3"],
            bo.Settings(minibatch_size=2, failure_tolerance=3),
            None,
        ),
        (
            ["--minibatch-size", "2", "--paired-epistemic-scale", "0.02"],
            bo.Settings(minibatch_size=2, paired_epistemic_scale=0.02),
            None,
        ),
        (
            ["--reference-seed", "18446744073709551615"],
            bo.Settings(reference_seed=2**64 - 1),
            None,
        ),
        (["--sampler", "independent"], bo.Settings(sampler="independent"), None),
        (["--sampler", "gaussian"], bo.Settings(sampler="gaussian"), None),
        (
            ["--sampler", "gaussian", "--candidates", "8", "--failure-tolerance", "7"],
            bo.Settings(sampler="gaussian", candidates=8, failure_tolerance=7),
            None,
        ),
        (
            [
                "--sampler",
                "independent",
                "--candidates",
                "3",
                "--failure-tolerance",
                "7",
            ],
            bo.Settings(sampler="independent", candidates=3, failure_tolerance=7),
            None,
        ),
        (["--failure-tolerance", "4"], None, "failure_tolerance"),
        (["--sampler", "unknown"], None, "Invalid value"),
        (["--reference-seed", "-1"], None, "reference_seed"),
    ],
)
def test_contract(tmp_path, monkeypatch, options, expected, error):
    tokens = tmp_path / "tokens.json"
    tokens.write_text("[[1, 2, 3]]")
    calls = []
    monkeypatch.setattr(bo, "run", lambda *args, **kwargs: calls.append((args, kwargs)))
    result = CliRunner().invoke(
        bo.main,
        [
            str(tmp_path),
            "--tokens",
            str(tokens),
            "--output",
            str(tmp_path / "result"),
            *options,
        ],
    )
    if error:
        assert result.exit_code == 2 and error in result.output
        assert not calls
    else:
        assert result.exit_code == 0, result.output
        assert len(calls) == 1 and calls[0][0][3] == expected


@pytest.fixture(autouse=True)
def cpu():
    with jax.default_device(jax.devices("cpu")[0]):
        yield


def bf16(value):
    return jnp.asarray(value, dtype=jnp.bfloat16)


class FakeProposals:
    def __init__(
        self,
        flat,
        changes,
        *,
        seed=17,
        score=0.5,
        radius=0.01,
        candidate_index=2,
        persistence=0.0,
    ):
        self.array = flat.reshape((1, flat.size))
        self.description = (seed, score, radius, changes)
        self.selected_geometry = [(candidate_index, persistence)]
        self.exports = 0

    def describe(self):
        return [self.description]

    def geometry(self):
        return self.selected_geometry

    def __dlpack_device__(self):
        return self.array.__dlpack_device__()

    def __dlpack__(self, *args, **kwargs):
        self.exports += 1
        return self.array.__dlpack__(*args, **kwargs)


class FakeSearchState:
    def __init__(self, flat, proposals, *, best=-10.0, history=2, sampler="correlated"):
        self.best_array = flat
        self.best = best
        self.length = 0.01
        self.restarts = 0
        self.history_len = 1
        self.history = history
        self.sampler = sampler
        self.proposals = iter(proposals)
        self.calls = []
        self.readbacks = 0

    def ask(self, *args, **kwargs):
        self.calls.append(("ask", args, kwargs))
        self.pending = next(self.proposals)
        return self.pending

    def tell(self, proposals, rewards, noise):
        assert proposals is self.pending
        self.calls.append(("tell", proposals, rewards, noise))
        self.reward = rewards[0]

    def sync(self):
        self.calls.append(("sync",))
        accepted = self.reward > self.best
        if accepted:
            self.best = self.reward
            self.best_array = self.pending.array.reshape(-1)
            if self.sampler == "correlated":
                self.length = self.pending.description[2]
        self.history_len = min(self.history, self.history_len + 1)
        return [accepted]

    def read_best(self):
        self.readbacks += 1
        return np.array(jax.lax.bitcast_convert_type(self.best_array, jnp.uint16))


class FakePairedSearch(FakeSearchState):
    def enable_relative(self, *, failure_tolerance):
        assert not self.calls
        self.calls.append(("enable_relative", failure_tolerance))
        self.failure_tolerance = failure_tolerance
        self.failures = 0

    def incumbent(self):
        self.calls.append(("incumbent",))
        return jnp.array(self.best_array, copy=True).reshape((1, -1))

    def tell(self, *args):
        pytest.fail("minibatch observations must not use legacy acceptance")

    def tell_relative(
        self,
        proposals,
        value,
        variance,
        incumbent_value,
        incumbent_variance,
        improvement,
        improvement_variance,
        accept,
        *,
        reject_is_failure=True,
    ):
        assert proposals is self.pending
        self.calls.append(
            (
                "tell_relative",
                value,
                variance,
                incumbent_value,
                incumbent_variance,
                improvement,
                improvement_variance,
                accept,
                reject_is_failure,
            )
        )
        self.decision = accept
        self.reject_is_failure = reject_is_failure
        self.best = value if accept else incumbent_value

    def sync(self):
        self.calls.append(("sync",))
        if self.decision:
            self.best_array = self.pending.array.reshape(-1)
            self.length = self.pending.description[2]
            self.history_len = 1
            self.failures = 0
        else:
            self.history_len = min(self.history, self.history_len + 1)
            if self.reject_is_failure:
                self.failures += 1
                if self.failures == self.failure_tolerance:
                    self.length = max(0.0001, self.length / 2)
                    self.failures = 0
        return [self.decision]


class FakeProblemEvaluator:
    def __init__(self, losses, population=8):
        self.tokens = [None] * population
        self.values = iter(losses)
        self.batches = []

    def losses(self, flat, indices):
        assert flat.ndim == 2 and flat.shape[0] == 1
        self.batches.append(indices.tolist())
        return np.asarray(next(self.values), dtype=np.float64)


def minibatch_baseline(settings, population=8):
    indices = bo.sample_indices(
        bo.minibatch_rng(settings.minibatch_seed), population, settings.minibatch_size
    )
    return {
        "indices": indices.tolist(),
        "losses": [0.1, 0.1],
        "variance": 0.0,
        "mean_loss": 0.1,
    }


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


def test_initialsave(tiny_model, tmp_path, monkeypatch):
    config, params = tiny_model
    (tmp_path / "manifest.json").write_text(
        json.dumps({"revision": REVISION, "complete": True})
    )
    extension_path = tmp_path / "ennx_rust.so"
    extension_path.write_bytes(b"test extension")
    extension = ModuleType("ennx.ennx_rust")
    extension.__file__ = str(extension_path)
    package = ModuleType("ennx")
    package.ennx_rust = extension
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
    monkeypatch.setitem(sys.modules, "ennx", package)
    monkeypatch.setitem(sys.modules, "ennx.ennx_rust", extension)
    monkeypatch.setitem(sys.modules, "ennx.experimental", experimental)
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


@pytest.fixture
def layout():
    return Layout.from_params({"a": bf16([1, 1]), "b": bf16(2)})


@pytest.fixture
def flat(layout):
    return layout.flatten({"a": bf16([1, 1]), "b": bf16(2)})


def test_settingbounds():
    settings = bo.Settings()
    assert settings.backend == "jax" and settings.minibatch_size is None
    assert (settings.evaluations, settings.candidates, settings.history) == (8, 4, 2)
    assert settings.seed == settings.reference_seed == 0
    assert settings.sampler == "correlated"
    assert settings.failure_tolerance is None
    for sampler in ("gaussian", "independent"):
        baseline = bo.Settings(sampler=sampler)
        assert baseline.candidates == 4 and baseline.failure_tolerance == 4
        assert (
            baseline.controller
            == "turbo_success_failure_with_explicit_failure_tolerance"
        )
    with pytest.raises(FrozenInstanceError):
        settings.seed = 1
    bo.Settings(
        evaluations=2,
        sampler="independent",
        candidates=1,
        history=128,
        failure_tolerance=2**32 - 1,
        seed=2**64 - 1,
        reference_seed=2**64 - 1,
    )
    bo.Settings(evaluations=1_000_000, radius=0.0001)
    bo.Settings(radius=0.08)


@pytest.mark.parametrize(
    "field,low,high",
    [
        ("evaluations", 2, 1_000_000),
        ("candidates", 1, 8),
        ("history", 2, 128),
        ("failure_tolerance", 1, 2**32 - 1),
        ("seed", 0, 2**64 - 1),
        ("reference_seed", 0, 2**64 - 1),
    ],
)
@pytest.mark.parametrize("sampler", ["gaussian", "independent"])
def test_integerguard(field, low, high, sampler):
    for value in (low - 1, high + 1, True, float(low), "2"):
        with pytest.raises(ValueError, match=field):
            bo.Settings(sampler=sampler, **{field: value})


@pytest.mark.parametrize("sampler", ["gaussian", "independent"])
@pytest.mark.parametrize("candidates", range(1, 9))
def test_bounds(sampler, candidates):
    settings = bo.Settings(sampler=sampler, candidates=candidates)
    assert settings.candidates == candidates and settings.failure_tolerance == 4


@pytest.mark.parametrize("candidates", [1, 2, 3, 5, 6, 7, 8])
def test_corrcount(candidates):
    with pytest.raises(ValueError, match="correlated.*candidates=4"):
        bo.Settings(candidates=candidates)


@pytest.mark.parametrize("tolerance", [0, 1, 4, True, "4"])
def test_corrtolerance(tolerance):
    with pytest.raises(ValueError, match="failure_tolerance.*paired minibatches"):
        bo.Settings(failure_tolerance=tolerance)


@pytest.mark.parametrize("tolerance", [0, -1, True, "4", 1.5, 2**32])
def test_tolerance(tolerance):
    with pytest.raises(ValueError, match="failure_tolerance"):
        bo.Settings(minibatch_size=2, failure_tolerance=tolerance)


@pytest.mark.parametrize(
    "scale", [0, -1, True, float("nan"), float("inf"), 1e-40, 1e40]
)
def test_epistemic(scale):
    with pytest.raises(ValueError, match="paired_epistemic_scale"):
        bo.Settings(minibatch_size=2, paired_epistemic_scale=scale)


def test_defaults():
    settings = bo.Settings(minibatch_size=2)
    assert settings.failure_tolerance == 4
    assert settings.paired_epistemic_scale == 1.0
    assert settings.rejection_policy == "deterioration"
    assert (
        settings.controller == "paired_accepted_radius_with_deterioration_contraction"
    )
    legacy = bo.Settings(minibatch_size=2, rejection_policy="all")
    assert legacy.controller == (
        "paired_accepted_radius_with_all_rejection_contraction_legacy"
    )


@pytest.mark.parametrize("policy", ["", "unknown", "Deterioration", None, True, 1])
def test_rejection(policy):
    with pytest.raises(ValueError, match="rejection_policy"):
        bo.Settings(minibatch_size=2, rejection_policy=policy)


@pytest.mark.parametrize("sampler", ["turbo", "Correlated", "", None])
def test_sampler(sampler):
    with pytest.raises(ValueError, match="sampler"):
        bo.Settings(sampler=sampler)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"radius": float("nan")},
        {"radius_min": float("inf")},
        {"radius_max": -float("inf")},
        {"radius_min": 0},
        {"radius": -0.01},
        {"radius": 0.1},
        {"radius": 0.00001},
        {"radius_min": 0.01, "radius_max": 0.01},
        {"radius_min": 1e-40},
        {"radius_max": 1e40},
        {"radius_min": 1e-30, "radius": 1e-25},
    ],
)
def test_invalidradii(kwargs):
    with pytest.raises(ValueError):
        bo.Settings(**kwargs)


def test_radius():
    with pytest.raises(ValueError):
        bo.Settings(radius_min=1e-201, radius=1e-200)


@pytest.mark.parametrize(
    "low,high",
    [
        (1.0 + 2**-52, 1.0 + 2**-51),
        (1.0 + 2**-25, 1.0 + 3 * 2**-25),
        (1.0 + 2**-24, 1.0 + 3 * 2**-24),
    ],
)
def test_collapsed(low, high):
    kwargs = {"radius": (low + high) / 2, "radius_min": low, "radius_max": high}
    with pytest.raises(ValueError, match="two distinct FP32 radii"):
        bo.Settings(**kwargs)
    for sampler in ("gaussian", "independent"):
        bo.Settings(sampler=sampler, **kwargs)


def test_adjacent():
    high = float(np.nextafter(np.float32(1), np.float32(np.inf)))
    settings = bo.Settings(radius=1.0, radius_min=1.0, radius_max=high)
    assert settings.radius_min == 1.0 and settings.radius_max == high
    defaults = bo.Settings()
    assert defaults.radius_min == 0.0001 and defaults.radius_max == 0.08


def test_cli(tmp_path, monkeypatch):
    tokens = tmp_path / "tokens.json"
    tokens.write_text("[[1, 2]]")
    monkeypatch.setattr(bo, "run", lambda *a, **k: pytest.fail("unexpected run"))
    result = CliRunner().invoke(
        bo.main,
        [
            str(tmp_path),
            "--tokens",
            str(tokens),
            "--output",
            str(tmp_path / "result"),
            "--radius",
            repr(1.0 + 2**-52),
            "--radius-min",
            repr(1.0 + 2**-52),
            "--radius-max",
            repr(1.0 + 2**-51),
        ],
    )
    assert result.exit_code == 2
    assert "two distinct FP32 radii" in result.output


@pytest.mark.parametrize(
    "size,padded",
    [
        (0, 0),
        (1, 128),
        (31, 128),
        (32, 128),
        (33, 128),
        (127, 128),
        (128, 128),
        (129, 256),
        (1_300_000_001, 1_300_000_128),
    ],
)
@pytest.mark.parametrize("history", [2, 128])
@pytest.mark.parametrize("sampler", ["correlated", "gaussian", "independent"])
def test_memorypadding(size, padded, history, sampler):
    reference_bytes = 2 * size if sampler == "correlated" else 0
    headroom = 9 * 1024**3 // 4 if sampler == "correlated" else 4 * 1024**3
    assert bo.memory_budget(size, history, sampler) == (
        (history + 2) * padded * 2 + reference_bytes + headroom
    )


def test_memoryguard():
    size = 1_300_024_320
    assert (
        bo.memory_budget(size, 2)
        == bo.memory_budget(size, 2, "gaussian") + 2_600_048_640 - 7 * 1024**3 // 4
    )
    assert bo.memory_budget(size, 2) / 1024**3 == pytest.approx(14.35742, abs=1e-5)
    with pytest.raises(ValueError, match="sampler"):
        bo.memory_budget(size, 2, "unknown")


@pytest.fixture
def tiny_model():
    config = Config(
        layers=1,
        width=4,
        heads=1,
        vocab=8,
        dense_width=4,
        expert_width=2,
        shared_width=2,
        experts=2,
        top_k=1,
        context=4,
    )
    rng = np.random.default_rng(17)
    params = {
        name: bf16(rng.normal(0, 0.2, shape)) for name, shape in config.shapes().items()
    }
    return config, params


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
    np.testing.assert_array_equal(layout.flatten(restored), improved)


@pytest.mark.parametrize(
    "sampler,tolerance",
    [
        ("correlated", None),
        ("gaussian", None),
        ("gaussian", 7),
        ("independent", 4),
        ("independent", 7),
    ],
)
@pytest.mark.parametrize("solution_objective", [False, True])
def test_metadata(
    tiny_model, tmp_path, monkeypatch, sampler, tolerance, solution_objective
):
    config, params = tiny_model
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    source = {"revision": REVISION, "complete": True}
    (checkpoint / "manifest.json").write_text(json.dumps(source))
    extension_path = tmp_path / "ennx_rust.so"
    extension_path.write_bytes(b"test extension")
    extension = ModuleType("ennx.ennx_rust")
    extension.__file__ = str(extension_path)
    package = ModuleType("ennx")
    package.ennx_rust = extension
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

    def lower(signature):
        assert stages == ["baseline"]
        assert isinstance(signature, jax.ShapeDtypeStruct)
        assert signature.shape == (1, size)
        assert signature.dtype == jnp.bfloat16
        stages.append("lower")

        def compile():
            stages.append("compile")
            return compiled_evaluate

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
    monkeypatch.setitem(sys.modules, "ennx", package)
    monkeypatch.setitem(sys.modules, "ennx.ennx_rust", extension)
    monkeypatch.setitem(sys.modules, "ennx.experimental", experimental)
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
    args, kwargs = calls[0]
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
