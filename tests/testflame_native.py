"""Native orchestration coverage with CPU checkpoints and fake CUDA bindings."""

from itertools import product
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


from native_fixtures import Source, binding, checkpoint, config, document


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


@pytest.mark.parametrize(
    "policy,backend",
    [
        (case_0, case_1)
        for case_0, case_1 in product(["deterioration", "all"], ["native", "metal"])
    ],
)
def test_directsources(
    checkpoint, document, binding, tmp_path, monkeypatch, *, backend, policy
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
    assert_report(output, binding, backend, policy, result)


def assert_report(output, binding, backend, policy, result):
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
    *,
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
