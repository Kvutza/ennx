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
    "sampler,field,low,high",
    [
        (case_0, *case_1)
        for case_0, case_1 in product(
            ["gaussian", "independent"],
            [
                ("evaluations", 2, 1_000_000),
                ("candidates", 1, 8),
                ("history", 2, 128),
                ("failure_tolerance", 1, 2**32 - 1),
                ("seed", 0, 2**64 - 1),
                ("reference_seed", 0, 2**64 - 1),
            ],
        )
    ],
)
def test_integerguard(field, low, high, sampler):
    for value in (low - 1, high + 1, True, float(low), "2"):
        with pytest.raises(ValueError, match=field):
            bo.Settings(sampler=sampler, **{field: value})


@pytest.mark.parametrize(
    "candidates,sampler",
    [
        (case_0, case_1)
        for case_0, case_1 in product(range(1, 9), ["gaussian", "independent"])
    ],
)
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
    "sampler,history,size,padded",
    [
        (case_0, case_1, *case_2)
        for case_0, case_1, case_2 in product(
            ["correlated", "gaussian", "independent"],
            [2, 128],
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
    ],
)
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
