from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pytest


def _loadmodule():
    script_path = (
        pathlib.Path(__file__).resolve().parents[1] / "scripts" / "benchdtsel.py"
    )
    spec = importlib.util.spec_from_file_location("benchdtsel", script_path)
    assert spec is not None and spec.loader is not None
    sys.modules.pop("benchdtsel", None)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_001():
    module = _loadmodule()
    bounds = module.make_bounds(3)
    assert bounds.shape == (3, 2)
    assert np.all(bounds[:, 0] == 0.0)
    assert np.all(bounds[:, 1] == 1.0)


def test_002():
    module = _loadmodule()
    x = np.array([[0.2, 0.4], [0.8, 0.1]], dtype=float)
    y = module.synthetic_objective(x)
    assert y.shape == (2, 1)
    assert y.dtype == float


def test_003():
    module = _loadmodule()

    class Recorder:
        def __init__(self):
            self.tell_args = []

        def tell(self, x, y):
            self.tell_args.append((x.copy(), y.copy()))

    rng = np.random.default_rng(0)
    rec = Recorder()
    module.seed_observations(rec, rng, num_obs=5, num_dim=2, batch_size=2)
    assert len(rec.tell_args) == 3
    assert rec.tell_args[0][0].shape == (2, 2)
    assert rec.tell_args[-1][0].shape == (1, 2)


def test_004(capsys):
    module = _loadmodule()
    module.summarize([1.0, 2.0, 3.0], "dt_sel")
    out = capsys.readouterr().out
    assert "dt_sel_ms" in out
    assert "mean=" in out
    assert "p95=" in out


def test_005(monkeypatch, capsys):
    module = _loadmodule()

    class FakeTelemetry:
        dt_sel = 0.001
        dt_gen = 0.002
        dt_fit = 0.003

    class FakeOpt:
        def __init__(self):
            self.tell_calls = []
            self.ask_calls = 0

        def tell(self, x, y):
            self.tell_calls.append((x, y))

        def ask(self, num_arms):
            self.ask_calls += 1
            return np.zeros((num_arms, 2))

        def telemetry(self):
            return FakeTelemetry()

    created = {"n": 0}

    def enn_config4(*, enn, num_init, acq_type):
        created["n"] += 1
        assert created["n"] == 1
        return {"enn": enn, "num_init": num_init, "acq_type": acq_type}

    def fake_optimizer(*, bounds, config, rng):
        assert bounds.shape == (2, 2)
        assert config["num_init"] == 2
        assert rng is not None
        return FakeOpt()

    monkeypatch.setattr(module, "turbo_enn", enn_config4)
    monkeypatch.setattr(module, "create_optimizer", fake_optimizer)

    module.run_benchmark(
        [
            "--timed-asks",
            "1",
            "--warmup-asks",
            "1",
            "--num-obs",
            "3",
            "--num-dim",
            "2",
            "--seed-batch-size",
            "2",
            "--num-arms",
            "2",
        ]
    )
    out = capsys.readouterr().out
    assert "setup" in out
    assert "timed_asks=1" in out


def test_006():
    module = _loadmodule()
    with pytest.raises(
        ValueError, match="num_obs, num_dim, and k must all be positive"
    ):
        module.run_benchmark(["--num-obs", "0", "--num-dim", "2"])
