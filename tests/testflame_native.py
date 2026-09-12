"""Native orchestration coverage with CPU checkpoints and fake CUDA bindings."""

import hashlib
import json
import os
import subprocess
import sys
import weakref
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from click.testing import CliRunner

from ops.flame import bo, native
from ops.flame.config import ITERATION, MODEL_ID, REVISION, Config
from ops.flame.layout import Layout
from ops.flame.objective import SolutionObjective


@pytest.fixture
def config():
    return Config(
        layers=2,
        width=4,
        heads=2,
        vocab=8,
        dense_width=6,
        expert_width=2,
        shared_width=4,
        experts=3,
        top_k=2,
        context=8,
    )


@pytest.fixture
def document():
    return {
        "format": "ennx.solution_tokens.v1",
        "provenance": {"dataset": "test"},
        "examples": [
            {
                "id": "a",
                "tokens": [1, 2, 3, 4],
                "loss_mask": [False, False, True, True],
            },
            {"id": "b", "tokens": [2, 3, 4], "loss_mask": [False, False, True]},
            {"id": "c", "tokens": [1, 3], "loss_mask": [False, True]},
            {"id": "d", "tokens": [4, 2, 1], "loss_mask": [False, True, True]},
        ],
    }


@pytest.fixture
def checkpoint(tmp_path, config, monkeypatch):
    torch = pytest.importorskip("torch")
    save_file = pytest.importorskip("safetensors.torch").save_file
    directory = tmp_path / "checkpoint"
    directory.mkdir()
    manifest = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "iteration": ITERATION,
        "complete": True,
        "config": asdict(config),
        "tensors": {},
    }
    params = {}
    for i, (name, shape) in enumerate(config.shapes().items()):
        params[name] = torch.full(shape, (i + 1) / 16, dtype=torch.bfloat16)
        path = directory / f"{i:03d}.safetensors"
        save_file({name: params[name]}, str(path))
        manifest["tensors"][name] = {
            "file": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(native, "Config", lambda **kw: Config(**kw) if kw else config)
    monkeypatch.setattr(bo, "Config", lambda: config)
    return directory, params, manifest


def test_cpucheckpoint(checkpoint, config):
    torch = pytest.importorskip("torch")
    directory, expected, _ = checkpoint
    actual_config, params = native.load_checkpoint(directory)
    assert actual_config == config
    layout = Layout.from_torch(params)
    flat = layout.flatten_torch(params)
    assert flat.device.type == "cpu" and flat.dtype == torch.bfloat16
    assert flat.ndim == 1 and flat.numel() == layout.size
    assert [b.name for b in layout.blocks] == sorted(config.shapes())
    for block in layout.blocks:
        value = flat[block.offset : block.offset + block.length].reshape(block.shape)
        assert torch.equal(value, expected[block.name])
    assert layout == Layout.from_torch(dict(reversed(list(params.items()))))
    before = layout.describe()
    layout.flatten_torch({k: v * 2 for k, v in params.items()})
    assert layout.describe() == before


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("model_id", "wrong", "provenance"),
        ("revision", "wrong", "provenance"),
        ("iteration", "wrong", "provenance"),
        ("complete", False, "incomplete"),
        ("tensors", {}, "architecture"),
    ],
)
def test_manifest(checkpoint, field, value, match):
    directory, _, manifest = checkpoint
    manifest[field] = value
    (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=match):
        native.load_checkpoint(directory)


@pytest.mark.parametrize(
    "fault,match",
    [
        ("config", "architecture"),
        ("path", "filename"),
        ("checksum", "checksum"),
        ("name", "layout"),
        ("shape", "layout"),
        ("dtype", "layout"),
    ],
)
def test_tensor(checkpoint, fault, match):
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    directory, params, manifest = checkpoint
    name = next(iter(params))
    record = manifest["tensors"][name]
    path = directory / record["file"]
    if fault == "config":
        manifest["config"]["context"] += 1
    elif fault == "path":
        record["file"] = "../outside.safetensors"
    elif fault == "checksum":
        record["sha256"] = "bad"
    else:
        value = params[name]
        save_file(
            {
                "wrong" if fault == "name" else name: value.reshape(-1)
                if fault == "shape"
                else value.to(torch.float32)
                if fault == "dtype"
                else value
            },
            str(path),
        )
        record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=match):
        native.load_checkpoint(directory)


@pytest.mark.parametrize(
    "values,error",
    [
        ([], "nonempty"),
        ([float("nan")], "finite"),
        ([float("inf")], "finite"),
        ([0, 0], "zero_scale"),
    ],
)
def test_layout(values, error):
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match=error):
        Layout.from_torch({"x": torch.tensor(values, dtype=torch.bfloat16)})


def test_layoutflatten():
    torch = pytest.importorskip("torch")
    # Squaring these values directly overflows FP32; the metric still fits.
    params = {"x": torch.tensor([1e20, -1e20], dtype=torch.bfloat16)}
    layout = Layout.from_torch(params)
    assert np.isfinite(layout.blocks[0].scale) and layout.blocks[0].weight > 0
    zero = Layout.from_torch(
        {"x": torch.zeros(2, dtype=torch.bfloat16)}, zero_scale=0.25
    )
    assert zero.blocks[0].scale == 0.25
    for bad in ({"z": params["x"]}, {"x": params["x"].reshape(1, 2)}):
        with pytest.raises(ValueError):
            layout.flatten_torch(bad)
    with pytest.raises(TypeError, match="BF16"):
        Layout.from_torch({"x": torch.ones(2)})
    with pytest.raises(ValueError, match="finite"):
        layout.flatten_torch(
            {"x": torch.tensor([1, float("nan")], dtype=torch.bfloat16)}
        )


class Source:
    def __init__(self, role, *, blocks=(), unchanged=False, radius=0.005):
        self.role, self.blocks, self.unchanged = role, blocks, unchanged
        self.radius = radius
        self.leases = 0

    def __dlpack__(self, *args, **kwargs):
        pytest.fail("Python must pass the source directly to the native binding")

    @contextmanager
    def binding_lease(self):
        self.leases += 1
        try:
            yield
        finally:
            self.leases -= 1

    def describe(self):
        changes = [(0, 0.0) if self.unchanged else (1, 0.01) for _ in self.blocks]
        return [(17, 0.5, self.radius, changes)]

    def geometry(self):
        return [(0, 0.75)]


@pytest.fixture
def binding(monkeypatch, tmp_path, config):
    state = SimpleNamespace(
        calls=[],
        forwards=[],
        source_refs=[],
        search=None,
        workspace=1024,
        failure=None,
        unchanged=False,
        reads=0,
        input_ref=None,
        engine=None,
        leases=[],
        candidate_losses=[1, 3],
        incumbent_losses=[2, 4],
        ask_options=[],
        paired_updates=[],
    )
    extension = ModuleType("ennx.ennx_rust")
    extension.__file__ = str(tmp_path / "fake-extension.so")
    Path(extension.__file__).write_bytes(b"fake extension")
    package = ModuleType("ennx")
    package.ennx_rust = extension
    experimental = ModuleType("ennx.experimental")
    package.experimental = experimental

    class Engine:
        def __init__(self, received, max_tokens):
            assert received == asdict(config) and max_tokens == 4
            state.calls.append("engine")
            state.engine = self
            self.weights_len = sum(np.prod(s) for s in config.shapes().values())
            self.workspace_bytes = state.workspace

        def losses(self, weights, tokens, masks):
            assert isinstance(weights, Source)
            assert len(tokens) == len(masks) == 2
            assert all(
                len(t) == len(m) and not m[0] and m[-1] for t, m in zip(tokens, masks)
            )
            state.calls.append(weights.role)
            state.forwards.append((weights.role, tokens, masks))
            state.source_refs.append(weakref.ref(weights))
            with weights.binding_lease():
                if state.failure == weights.role:
                    state.leases.append(weights)
                    raise RuntimeError("native forward failed")
                return np.asarray(
                    state.candidate_losses
                    if weights.role == "proposal"
                    else state.incumbent_losses,
                    dtype=np.float32,
                )

    class Search:
        def __init__(self, flat, value, blocks, history, **kwargs):
            assert flat is state.input_ref() and flat.leases == 0
            assert value == -3 and kwargs["base_variance"] == 0.5
            assert history == 2 and kwargs["max_pending"] == 1
            assert (
                kwargs["sampler"] == "correlated"
                and kwargs["failure_tolerance"] is None
            )
            state.calls.append("search")
            state.search = self
            self.blocks = blocks
            self.best, self.length, self.restarts, self.history_len = value, 0.01, 0, 1
            self.best_variance = kwargs["base_variance"]
            self.minimum = kwargs["length_min"]
            self.configured = False
            self.failures = 0

        def enable_relative(self, failure_tolerance=4):
            assert not self.configured and "ask" not in state.calls
            assert type(failure_tolerance) is int and failure_tolerance > 0
            state.calls.append("enable_relative")
            self.configured = True
            self.failure_tolerance = failure_tolerance

        def incumbent(self):
            return Source("incumbent")

        def ask(self, arms, candidates, history, root, **kwargs):
            assert self.configured
            assert state.input_ref() is None, (
                "input must be freed before reference allocation"
            )
            assert all(
                ref() is None or (ref().role == "proposal" and ref().leases == 0)
                for ref in state.source_refs
            )
            assert (arms, candidates, history) == (1, 4, 2) and kwargs["y_scale"] == 1
            assert kwargs["aleatoric_scale"] == 0
            assert kwargs["acquisition"] == "thompson"
            state.ask_options.append(kwargs)
            state.calls.append("ask")
            self.pending = Source(
                "proposal",
                blocks=self.blocks,
                unchanged=state.unchanged,
                radius=max(self.minimum, self.length / 2),
            )
            return self.pending

        def tell_relative(
            self,
            proposals,
            value,
            variance,
            incumbent,
            incumbent_variance,
            improvement,
            improvement_variance,
            accept,
            *,
            reject_is_failure=True,
        ):
            assert self.configured
            assert proposals is self.pending and proposals.leases == 0
            state.calls.append("tell_relative")
            state.paired_updates.append(
                (
                    value,
                    variance,
                    incumbent,
                    incumbent_variance,
                    improvement,
                    improvement_variance,
                    accept,
                    reject_is_failure,
                )
            )
            self.accept = accept
            self.reject_is_failure = reject_is_failure
            self.best = value if accept else incumbent
            self.best_variance = variance if accept else incumbent_variance

        def sync(self):
            if self.accept:
                self.length = self.pending.radius
                self.history_len = 1
                self.failures = 0
            else:
                self.history_len = min(2, self.history_len + 1)
                if self.reject_is_failure:
                    self.failures += 1
                    if self.failures == self.failure_tolerance:
                        self.length = max(self.minimum, self.length / 2)
                        self.failures = 0
            return [self.accept]

        def read_best(self):
            state.reads += 1
            state.calls.append("save")
            return np.full(
                sum(block[2] for block in self.blocks), 0x3F80, dtype=np.uint16
            )

    experimental.FlameEvaluator = Engine
    experimental.SearchState = Search
    experimental.ParamBlock = lambda *args: args
    for name, module in (
        ("ennx", package),
        ("ennx.ennx_rust", extension),
        ("ennx.experimental", experimental),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(native, "cuda_device", lambda: "Tesla T4")
    monkeypatch.setattr(bo.subprocess, "check_output", lambda *a, **kw: "15360\n")

    def upload(flat):
        assert state.calls == ["engine"] and flat.device.type == "cpu"
        state.calls.append("upload")
        source = Source("input")
        state.input_ref = weakref.ref(source)
        return source

    def release():
        assert state.input_ref() is None
        state.calls.append("release")

    monkeypatch.setattr(native, "upload_weights", upload)
    monkeypatch.setattr(native, "release_inputcache", release)
    return state


@pytest.mark.parametrize("backend", ["native", "metal"])
@pytest.mark.parametrize("policy", ["deterioration", "all"])
def test_directsources(
    checkpoint, document, binding, tmp_path, monkeypatch, backend, policy
):
    if backend == "metal":
        from ops.flame import metal

        experimental = sys.modules["ennx.experimental"]
        experimental.MetalFlameEvaluator = experimental.FlameEvaluator
        experimental.MetalSearchState = experimental.SearchState
        experimental.MetalParamBlock = experimental.ParamBlock
        monkeypatch.setattr(metal, "metal_device", lambda: "Apple M4")
        monkeypatch.setattr(metal, "free_memory", lambda: 15 * 1024**3)
        monkeypatch.setattr(metal, "upload_weights", native.upload_weights)

        def no_cuda(*args, **kwargs):
            raise AssertionError("Metal must not query CUDA")

        monkeypatch.setattr(bo.subprocess, "check_output", no_cuda)
    directory, _, _ = checkpoint
    tokens = tmp_path / "tokens.json"
    tokens.write_text(json.dumps(document))
    output = tmp_path / "run"
    result = CliRunner().invoke(
        bo.main,
        [
            str(directory),
            "--backend",
            backend,
            "--tokens",
            str(tokens),
            "--output",
            str(output),
            "--evaluations",
            "2",
            "--rejection-policy",
            policy,
        ],
    )
    assert result.exit_code == 0, result.exception
    assert binding.calls == [
        "engine",
        "upload",
        "input",
        "search",
        *(["release"] if backend == "native" else []),
        "enable_relative",
        "incumbent",
        "ask",
        "proposal",
        "tell_relative",
        "save",
    ]
    assert binding.reads == 1
    assert binding.search.failure_tolerance == 4
    assert binding.search.history_len == 1
    assert binding.ask_options[0]["epistemic_scale"] == 1.0
    assert binding.paired_updates == [
        (-2.0, 0.5, -3.0, 0.5, 1.0, 0.0, True, policy == "all")
    ]
    assert sum(len(tokens) for _, tokens, _ in binding.forwards[1:]) == 4
    assert binding.forwards[1][1:] == binding.forwards[2][1:]
    report = json.loads((output / "run.json").read_text())
    assert report["backend"] == backend and "jax" not in report
    assert report["settings"]["minibatch_size"] == 2
    assert report["settings"]["rejection_policy"] == policy
    assert report["objective"]["rejection_policy"] == policy
    assert report["objective"]["screening"] == (
        "paired_standard_error_heuristic_not_a_statistical_guarantee"
    )
    assert report["controller"] == (
        "paired_accepted_radius_with_deterioration_contraction"
        if policy == "deterioration"
        else "paired_accepted_radius_with_all_rejection_contraction_legacy"
    )
    events = [
        json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()
    ]
    assert events[-1]["outcome"] == "accepted"
    assert events[-1]["counted_failure"] is False
    assert events[-1]["minibatch"]["deteriorated"] is False
    assert "outcome=accepted counted_failure=false" in result.output
    assert report["objective_evaluations"] == 3 and report["evaluations"] == 2
    assert report["base_version"] == 1 and report["best_reward"] == -2
    assert (
        report["objective"]["normalization"] == "mean_per_problem_solution_token_loss"
    )
    assert report["estimated_memory_bytes"] == native.memory_budget(
        binding.engine.weights_len, 2, 1024
    )
    restored_config, restored = native.load_checkpoint(output / "best")
    assert restored_config.shapes().keys() == restored.keys()
    assert all(bool((tensor == 1).all()) for tensor in restored.values())


def test_metalcompare(checkpoint, document, binding, tmp_path, monkeypatch, config):
    from ops.flame import metal_compare

    run = tmp_path / "run"
    bo.run(
        checkpoint[0],
        document,
        run,
        bo.Settings(backend="native", minibatch_size=2, evaluations=2),
    )
    report = json.loads((run / "run.json").read_text())
    report["backend"] = "metal"
    (run / "run.json").write_text(json.dumps(report))
    events = [
        json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()
    ]
    expected = [
        events[0]["minibatch"]["losses"],
        [3, 2, 1, 4],
        report["final_minibatch"]["losses"],
        [3, 1, 2, 2],
    ]
    calls = []

    class Evaluator:
        def __init__(self, objective, received):
            assert received == config
            self.tokens = objective.tokens

        def losses(self, weights, indices):
            assert weights.device.type == "cpu"
            values = expected[len(calls)]
            calls.append(list(indices))
            assert len(values) == len(indices)
            return np.asarray(values, dtype=np.float64)

    monkeypatch.setattr(metal_compare, "Config", lambda: config)
    monkeypatch.setattr(metal_compare, "NativeEvaluator", Evaluator)
    monkeypatch.setattr(metal_compare, "upload_weights", lambda flat: flat)
    result = metal_compare.compare(checkpoint[0], run)
    assert len(calls) == 4
    assert calls[1] == calls[3] == [0, 1, 2, 3]
    assert result["original_mean_loss"] == 2.5
    assert result["final_mean_loss"] == 2.0
    assert result["change_final_minus_original"] == -0.5
    assert result["improved_problems"] == 2
    assert result["worsened_problems"] == 1
    assert result["unchanged_problems"] == 1
    assert result["post_run_problem_forwards"] == 12
    assert result["checkpoint_reload_losses_exact"]
    report["extension_sha256"] = "wrong"
    (run / "run.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="exact extension"):
        metal_compare.compare(checkpoint[0], run)


@pytest.mark.parametrize("role", ["incumbent", "proposal"])
def test_runcleanup(checkpoint, document, binding, tmp_path, role):
    binding.failure = role
    with pytest.raises(RuntimeError, match="native forward failed"):
        bo.run(
            checkpoint[0],
            document,
            tmp_path / "run",
            bo.Settings(backend="native", minibatch_size=2, evaluations=2),
        )
    assert all(source.leases == 0 for source in binding.leases)
    assert binding.reads == 0 and "tell_relative" not in binding.calls
    assert json.loads((tmp_path / "run/run.json").read_text())["status"] == "failed"


def test_unchanged(checkpoint, document, binding, tmp_path):
    binding.unchanged = True
    result = bo.run(
        checkpoint[0],
        document,
        tmp_path / "run",
        bo.Settings(backend="native", minibatch_size=2, evaluations=2),
    )
    assert result["stop_reason"] == "selected_proposal_unchanged"
    assert result["objective_evaluations"] == 2 and result["base_version"] == 0
    assert "proposal" not in binding.calls and binding.reads == 1


def test_loopreuse(checkpoint, document, binding, tmp_path):
    result = bo.run(
        checkpoint[0],
        document,
        tmp_path / "run",
        bo.Settings(backend="native", minibatch_size=2, evaluations=3),
    )
    assert result["objective_evaluations"] == 5 and binding.reads == 1
    assert binding.calls.count("enable_relative") == 1
    assert binding.calls.count("ask") == binding.calls.count("tell_relative") == 2
    assert sum(len(tokens) for _, tokens, _ in binding.forwards[1:]) == 8
    for first, second in ((1, 2), (3, 4)):
        assert binding.forwards[first][1:] == binding.forwards[second][1:]


def test_var(checkpoint, document, binding, tmp_path):
    binding.candidate_losses = [1, 2]
    bo.run(
        checkpoint[0],
        document,
        tmp_path / "run",
        bo.Settings(
            backend="native",
            minibatch_size=2,
            evaluations=2,
            paired_epistemic_scale=0.25,
            failure_tolerance=3,
        ),
    )
    assert binding.calls.count("enable_relative") == 1
    assert binding.search.failure_tolerance == 3
    assert binding.ask_options[0]["epistemic_scale"] == 0.25
    # Paired variance is neither candidate variance nor their marginal sum.
    assert binding.paired_updates == [(-1.5, 0.125, -3.0, 0.5, 1.5, 0.125, True, False)]
    assert binding.reads == 1


@pytest.mark.parametrize(
    "losses,policy,outcome,counted_failure,measurement",
    [
        ([3, 5], "deterioration", "deteriorated", True, (-4.0, 0.5, -1.0, 0.0)),
        ([3, 4], "deterioration", "inconclusive", False, (-3.5, 0.125, -0.5, 0.125)),
        ([1, 4], "deterioration", "inconclusive", False, (-2.5, 1.125, 0.5, 0.125)),
        ([3, 4], "all", "inconclusive", True, (-3.5, 0.125, -0.5, 0.125)),
    ],
)
def test_budget(
    checkpoint,
    document,
    binding,
    tmp_path,
    losses,
    policy,
    outcome,
    counted_failure,
    measurement,
):
    binding.candidate_losses = losses
    output = tmp_path / "run"
    result = bo.run(
        checkpoint[0],
        document,
        output,
        bo.Settings(
            backend="native", minibatch_size=2, evaluations=5, rejection_policy=policy
        ),
    )
    value, variance, improvement, improvement_variance = measurement
    expected = (
        value,
        variance,
        -3.0,
        0.5,
        improvement,
        improvement_variance,
        False,
        counted_failure,
    )
    assert binding.paired_updates == [expected] * 4
    final_radius = 0.005 if counted_failure else 0.01
    assert binding.search.length == final_radius and binding.search.restarts == 0
    assert binding.search.history_len == 2
    assert binding.calls.count("enable_relative") == 1
    assert result["objective_evaluations"] == 9 and result["evaluations"] == 5
    assert result["best_reward"] == -3.0 and result["base_version"] == 0
    assert binding.reads == 1
    assert sum(len(tokens) for _, tokens, _ in binding.forwards[1:]) == 16
    for first in range(1, len(binding.forwards), 2):
        assert binding.forwards[first][1:] == binding.forwards[first + 1][1:]
    events = [
        json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()
    ]
    proposals = [event for event in events if event["event"] == "proposal"]
    assert [event["outcome"] for event in proposals] == [outcome] * 4
    assert all(event["counted_failure"] is counted_failure for event in proposals)
    controller = bo.Settings(minibatch_size=2, rejection_policy=policy).controller
    assert all(event["controller"] == controller for event in proposals)
    assert [event["next_reference_radius"] for event in proposals] == [
        0.01,
        0.01,
        0.01,
        final_radius,
    ]


@pytest.mark.parametrize("method", ["enable_relative", "tell_relative"])
def test_api(checkpoint, document, binding, tmp_path, monkeypatch, method):
    monkeypatch.delattr(sys.modules["ennx.experimental"].SearchState, method)
    with pytest.raises(RuntimeError, match="[Pp]aired"):
        bo.run(
            checkpoint[0],
            document,
            tmp_path / "run",
            bo.Settings(backend="native", minibatch_size=2, evaluations=2),
        )
    assert binding.calls == []
    assert binding.engine is None and binding.input_ref is None


@pytest.mark.parametrize("backend", ["native", "metal"])
@pytest.mark.parametrize("policy", ["deterioration", "all"])
@pytest.mark.parametrize(
    "signature", ["legacy", "positional-only", "uninspectable", "invalid-signature"]
)
def test_signature(
    checkpoint, document, binding, tmp_path, monkeypatch, backend, policy, signature
):
    experimental = sys.modules["ennx.experimental"]
    module = native
    if backend == "metal":
        from ops.flame import metal

        module = metal
        experimental.MetalSearchState = experimental.SearchState
        experimental.MetalParamBlock = experimental.ParamBlock
        monkeypatch.setattr(metal, "metal_device", lambda: "Apple M4")

    def legacy(self, *args):
        pytest.fail("stale binding must fail preflight")

    def positional_only(self, reject_is_failure=True, /):
        pytest.fail("positional-only flag must fail preflight")

    if signature == "invalid-signature":
        legacy.__signature__ = object()
    methods = {
        "legacy": legacy,
        "positional-only": positional_only,
        "uninspectable": dict.__getitem__,
        "invalid-signature": legacy,
    }
    monkeypatch.setattr(
        experimental.SearchState,
        "tell_relative",
        methods[signature],
    )

    def no_checkpoint(*args, **kwargs):
        pytest.fail("preflight must precede checkpoint loading")

    monkeypatch.setattr(module, "load_checkpoint", no_checkpoint)
    output = tmp_path / "run"
    with pytest.raises(
        RuntimeError, match="Rebuild.*reject_is_failure keyword support"
    ):
        bo.run(
            checkpoint[0],
            document,
            output,
            bo.Settings(
                backend=backend,
                minibatch_size=2,
                evaluations=2,
                rejection_policy=policy,
            ),
        )
    assert binding.calls == []
    assert binding.engine is None and binding.input_ref is None
    assert not output.exists()


def test_workspace(checkpoint, document, binding, tmp_path):
    binding.workspace = 16 * 1024**3
    with pytest.raises(RuntimeError, match="Native BO needs"):
        bo.run(
            checkpoint[0],
            document,
            tmp_path / "run",
            bo.Settings(backend="native", minibatch_size=2),
        )
    assert binding.calls == ["engine"] and binding.input_ref is None


@pytest.mark.parametrize("indices", [[], [0, 0], [-1], [4], [[0]], [True], [0.5]])
def test_indexprebind(config, document, binding, indices):
    evaluator = native.NativeEvaluator(
        SolutionObjective.parse(document, config), config
    )
    with pytest.raises(ValueError, match="indices"):
        evaluator.losses(Source("input"), indices)
    assert binding.forwards == []


@pytest.mark.parametrize(
    "backend,batch,explicit",
    [("native", 2, None), ("jax", None, None), ("native", 3, 3), ("jax", 2, 2)],
)
def test_backend(tmp_path, monkeypatch, backend, batch, explicit):
    tokens = tmp_path / "tokens.json"
    tokens.write_text("[[1, 2]]")
    calls = []
    monkeypatch.setattr(bo, "run", lambda *a, **kw: calls.append(a))
    options = [] if backend == "jax" else ["--backend", "native"]
    if explicit is not None:
        options += ["--minibatch-size", str(explicit)]
    result = CliRunner().invoke(
        bo.main,
        [
            str(tmp_path),
            "--tokens",
            str(tokens),
            "--output",
            str(tmp_path / "run"),
            *options,
        ],
    )
    assert result.exit_code == 0, result.exception
    assert calls[0][3].backend == backend and calls[0][3].minibatch_size == batch


def test_feature(config, document, binding, monkeypatch):
    monkeypatch.setattr(sys.modules["ennx.experimental"], "FlameEvaluator", None)
    with pytest.raises(RuntimeError, match="--features native-flame"):
        native.NativeEvaluator(SolutionObjective.parse(document, config), config)
    assert binding.calls == []


def test_legacyguard(tmp_path):
    for kwargs in ({}, {"sampler": "gaussian", "minibatch_size": 2}):
        with pytest.raises(ValueError, match="Native BO requires"):
            bo.Settings(backend="native", **kwargs)
    with pytest.raises(ValueError, match="solution-token corpora"):
        bo.run(
            tmp_path,
            [[1, 2]],
            tmp_path / "run",
            bo.Settings(backend="native", minibatch_size=2),
        )


@pytest.mark.parametrize(
    "name,index,visible,available",
    [
        ("A100", 0, None, True),
        ("Tesla T4", 1, None, True),
        ("Tesla T4", 0, "1", True),
        ("Tesla T4", 0, None, False),
    ],
)
def test_cudaguard(monkeypatch, name, index, visible, available):
    torch = pytest.importorskip("torch")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    if visible is not None:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: index)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: name)
    with pytest.raises(RuntimeError, match="device 0"):
        native.cuda_device()


IMPORT_GUARD = """
import importlib.abc, sys
class NoJax(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('jax', 'jaxlib') or fullname == 'ops.flame.model':
            raise AssertionError('forbidden native import: ' + fullname)
sys.meta_path.insert(0, NoJax())
"""


def test_jax():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            IMPORT_GUARD
            + """
import pytest
status = pytest.main(['tests/testflame_native.py', '-q', '-k',
    'directsources or cpucheckpoint or backend'])
assert not any(n.split('.')[0] in ('jax', 'jaxlib') for n in sys.modules)
assert 'ops.flame.model' not in sys.modules
raise SystemExit(status)
""",
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONPATH": str(root / "src") + os.pathsep + str(root),
            "XLA_PYTHON_CLIENT_ALLOCATOR": "bfc",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        },
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_lazynojax():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            IMPORT_GUARD
            + """
from types import ModuleType
import ennx
from ennx import experimental
assert 'ennx._rust' not in sys.modules
assert 'ennx.ennx_rust' not in sys.modules
sentinel = object()
class Namespace:
    def __getattr__(self, name):
        return sentinel
extension = ModuleType('ennx.ennx_rust')
for name in ('hypervolume', 'hash', 'util', 'model', 'fit', 'optimizer', 'experimental'):
    setattr(extension, name, Namespace())
sys.modules['ennx.ennx_rust'] = extension
from ennx.experimental import ParamBlock, SearchState, FlameEvaluator
assert ParamBlock is SearchState is FlameEvaluator is sentinel
assert not any(n.split('.')[0] in ('jax', 'jaxlib') for n in sys.modules)
""",
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
