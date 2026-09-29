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


@pytest.mark.parametrize(
    "signature,policy,backend",
    [
        (case_0, case_1, case_2)
        for case_0, case_1, case_2 in product(
            ["legacy", "positional-only", "uninspectable", "invalid-signature"],
            ["deterioration", "all"],
            ["native", "metal"],
        )
    ],
)
def test_signature(
    checkpoint, document, binding, tmp_path, monkeypatch, *, backend, policy, signature
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
