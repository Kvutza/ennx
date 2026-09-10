from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pytest

from scripts.cmpxvers import (
    BASELINE_LABEL,
    CURRENT_LABEL,
    _buildparser2,
    _comparesummary,
    _envversion,
    _formatrow,
    _runsubprocess,
    _workermain,
    main,
)
from scripts.cmpxverscore import (
    OPTIMIZER_NAMES,
    PROBLEMS,
    BenchmarkResult,
    apply_overrides3,
    build_config,
    compute_hypervolume,
    experiment_combos,
    run_benchmark,
    separable_objective,
)


def test_001():
    combos = experiment_combos()
    assert len(combos) == 9
    assert len({c[0] for c in combos}) == 3


def test_002():
    assert OPTIMIZER_NAMES == ("turbo_enn", "turbo_one", "morbo")


def test_003():
    x = np.array([[120.0, 0.91]])
    y = separable_objective(x)
    assert y.shape == (1, 2)
    assert float(y[0, 0]) >= 499_000.0
    assert float(y[0, 1]) >= 11.0


def test_004():
    y = np.array([[1.0, 2.0], [3.0, 1.5], [0.5, 3.0]])
    hv = compute_hypervolume(y, np.array([0.0, 0.0]))
    assert hv > 0.0


def test_005():
    single = PROBLEMS["ackley_30d"]
    with pytest.raises(ValueError, match="num_metrics"):
        build_config("morbo", single)


def test_006():
    quick = apply_overrides3(PROBLEMS)
    assert quick["ackley_30d"].num_iterations < PROBLEMS["ackley_30d"].num_iterations


def test_007():
    problem = apply_overrides3(PROBLEMS)["ackley_30d"]
    result = run_benchmark(
        optimizer="turbo_enn",
        problem=problem,
        version_label="current",
    )
    assert result.quality_metric == "best_y"
    assert np.isfinite(result.quality)
    assert result.num_evals == problem.num_iterations * problem.num_arms


def test_008():
    problem = apply_overrides3(PROBLEMS)["separable_unimodal"]
    result = run_benchmark(
        optimizer="morbo",
        problem=problem,
        version_label="current",
    )
    assert result.quality_metric == "hypervolume"
    assert np.isfinite(result.quality)


def test_009():
    current = BenchmarkResult(
        optimizer="turbo_enn",
        problem="ackley_30d",
        version_label=CURRENT_LABEL,
        quality=2.0,
        quality_metric="best_y",
        wall_seconds=1.0,
        ask_seconds=0.5,
        num_evals=20,
        seed=18,
    )
    baseline = BenchmarkResult(
        **{**asdict(current), "version_label": BASELINE_LABEL, "quality": 1.0}
    )
    summary = _comparesummary(current, baseline)
    assert summary["quality_delta"] == pytest.approx(1.0)
    assert summary["wall_ratio"] == pytest.approx(1.0)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_010(seed: int):
    rng = np.random.default_rng(seed)
    cur_q = float(rng.normal())
    base_q = float(rng.normal())
    current = BenchmarkResult(
        optimizer="turbo_one",
        problem="ackley_30d",
        version_label=CURRENT_LABEL,
        quality=cur_q,
        quality_metric="best_y",
        wall_seconds=1.0,
        ask_seconds=0.5,
        num_evals=20,
        seed=seed,
    )
    baseline = BenchmarkResult(
        **{**asdict(current), "version_label": BASELINE_LABEL, "quality": base_q}
    )
    summary = _comparesummary(current, baseline)
    assert summary["quality_delta"] == pytest.approx(cur_q - base_q)
    print(f"compare_summary fuzz seed={seed}")


def test_011():
    env = _envversion(CURRENT_LABEL)
    assert str(env["PYTHONPATH"]).split(":")[0].endswith("/src")


def test_012():
    env = _envversion(BASELINE_LABEL)
    assert "/src" not in env.get("PYTHONPATH", "")


def test_013():
    row = _formatrow(
        BenchmarkResult(
            optimizer="morbo",
            problem="separable_unimodal",
            version_label=CURRENT_LABEL,
            quality=1.0,
            quality_metric="hypervolume",
            wall_seconds=2.0,
            ask_seconds=1.0,
            num_evals=40,
            seed=42,
        )
    )
    assert "morbo" in row
    assert "separable_unimodal" in row


def test_014(monkeypatch):
    from scripts import cmpxvers as module

    expected = BenchmarkResult(
        optimizer="turbo_enn",
        problem="ackley_30d",
        version_label=CURRENT_LABEL,
        quality=-1.0,
        quality_metric="best_y",
        wall_seconds=0.1,
        ask_seconds=0.05,
        num_evals=20,
        seed=18,
    )

    class Proc:
        returncode = 0
        stdout = json.dumps(expected.to_dict())
        stderr = ""

    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: Proc())
    result = _runsubprocess(
        optimizer="turbo_enn",
        problem="ackley_30d",
        version_label=CURRENT_LABEL,
        quick=True,
    )
    assert result.quality == pytest.approx(-1.0)


def test_mainquicksmoke(monkeypatch, tmp_path):
    from scripts import cmpxvers as module

    def fake_worker(*, optimizer, problem, version_label, quick):
        return BenchmarkResult(
            optimizer=optimizer,
            problem=problem,
            version_label=version_label,
            quality=0.5,
            quality_metric="best_y",
            wall_seconds=0.1,
            ask_seconds=0.05,
            num_evals=10,
            seed=18,
        )

    monkeypatch.setattr(module, "_runsubprocess", fake_worker)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    rc = main(["--quick"])
    assert rc == 0
    report = json.loads((tmp_path / "cmpxvers_report.json").read_text())
    assert report["baseline_version"] == "0.3.6"
    assert len(report["comparisons"]) == 9


def test_workermainandparser(monkeypatch, capsys):
    from scripts import cmpxvers as module

    monkeypatch.setattr(
        module,
        "run_benchmark",
        lambda **kwargs: BenchmarkResult(
            optimizer=kwargs["optimizer"],
            problem=kwargs["problem"],
            version_label=kwargs["version_label"],
            quality=0.0,
            quality_metric="best_y",
            wall_seconds=0.1,
            ask_seconds=0.05,
            num_evals=10,
            seed=18,
        ),
    )
    parser = _buildparser2()
    args = parser.parse_args(
        [
            "worker",
            "--optimizer",
            "turbo_enn",
            "--problem",
            "ackley_30d",
            "--version",
            "current",
            "--quick",
        ]
    )
    assert _workermain(args) == 0
    assert json.loads(capsys.readouterr().out)["optimizer"] == "turbo_enn"


def test_015():
    morbo_problems = [p for o, p in experiment_combos() if o == "morbo"]
    assert morbo_problems == [
        "double_ackley_30d",
        "separable_unimodal",
        "ackley_pair_30d",
    ]


def test_016():
    spec = PROBLEMS["ackley_30d"]
    bounds = spec.bounds()
    assert bounds.shape == (30, 2)


def test_017():
    spec = PROBLEMS["ackley_pair_30d"]
    objective = spec.make_objective()
    y = objective(np.zeros((2, 30)))
    assert y.shape == (2, 2)
