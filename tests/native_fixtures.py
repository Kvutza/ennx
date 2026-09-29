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
                return [self.accept]
            self.history_len = min(2, self.history_len + 1)
            if self.reject_is_failure:
                self.failures += 1
            if self.reject_is_failure and self.failures == self.failure_tolerance:
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
