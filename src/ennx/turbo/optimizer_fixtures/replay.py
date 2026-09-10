from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ennx.turbo.config.optimizer_config import OptimizerConfig

from .catalog import (
    FIXTURE_OBJECTIVES,
    PREFIX_CONFIG,
    entry_name,
    subdir_entry,
)

_PACKAGE_FIXTURES_ROOT = Path(__file__).resolve().parent / "data"
_SOURCE_FIXTURES_ROOT = Path(__file__).resolve().parents[4] / "tests" / "fixtures"
FIXTURES_ROOT = (
    _PACKAGE_FIXTURES_ROOT if _PACKAGE_FIXTURES_ROOT.is_dir() else _SOURCE_FIXTURES_ROOT
)

EXACT_RTOL = 1e-14
EXACT_ATOL = 1e-14
TR_RTOL = 1e-9
TR_ATOL = 1e-9


def _config(name: str) -> OptimizerConfig:
    return PREFIX_CONFIG[entry_name(name).config_key]


def _dirname(name: str) -> Path:
    entry = entry_name(name)
    return FIXTURES_ROOT / subdir_entry(entry)


def load_fixture(name: str) -> dict[str, Any]:
    path = _dirname(name) / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing optimizer replay fixture: {path}")
    with open(path) as f:
        return json.load(f)


def assert_invariants(data: dict[str, Any]) -> None:
    bounds = np.array(data["bounds"], dtype=float)
    objective_fn = FIXTURE_OBJECTIVES[str(data["objective"])]
    prev_tr_obs = 0
    for step in data["steps"]:
        x = np.array(step["ask"], dtype=float)
        y = np.array(step["tell_y"], dtype=float)
        assert x.shape[1] == bounds.shape[0]
        assert y.shape[0] == x.shape[0]
        assert np.all(np.isfinite(x))
        assert np.all(np.isfinite(y))
        assert np.all(x >= bounds[:, 0] - 1e-9)
        assert np.all(x <= bounds[:, 1] + 1e-9)
        np.testing.assert_allclose(objective_fn(x), y, rtol=EXACT_RTOL, atol=EXACT_ATOL)
        assert 0.0 < step["tr_length"] <= 2.5
        tr_obs = int(step["tr_count"])
        assert tr_obs >= prev_tr_obs
        prev_tr_obs = tr_obs


def replay_check(data: dict[str, Any], config: OptimizerConfig) -> None:
    from ennx import create_optimizer

    assert_invariants(data)
    bounds = np.array(data["bounds"], dtype=float)
    rng = np.random.default_rng(int(data["seed"]))
    opt = create_optimizer(bounds=bounds, config=config, rng=rng)
    num_arms = int(data["num_arms"])
    for step in data["steps"]:
        x_golden = np.array(step["ask"], dtype=float)
        y_golden = np.array(step["tell_y"], dtype=float)
        x = opt.ask(num_arms=num_arms)
        assert isinstance(x, np.ndarray)
        assert x.shape == (num_arms, bounds.shape[0])
        assert np.all(np.isfinite(x))
        assert np.all(x >= bounds[:, 0] - 1e-9)
        assert np.all(x <= bounds[:, 1] + 1e-9)
        np.testing.assert_allclose(x, x_golden, rtol=EXACT_RTOL, atol=EXACT_ATOL)
        opt.tell(x_golden, y_golden)
        assert int(opt.region_count) == int(step["tr_count"])
        np.testing.assert_allclose(
            opt.tr_length,
            float(step["tr_length"]),
            rtol=TR_RTOL,
            atol=TR_ATOL,
        )
