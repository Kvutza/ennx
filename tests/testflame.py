from __future__ import annotations

import io
import json
import math
import pickle
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest
from click.testing import CliRunner

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from ops.flame import checkpoint, model, reference
from ops.flame.config import Config


@pytest.mark.parametrize("command", ["checkpoint", "parity"])
@pytest.mark.parametrize("flag", ["-h", "--help"])
def test_clihelp(command, flag):
    from ops.flame import parity

    result = CliRunner().invoke(
        {"checkpoint": checkpoint.main, "parity": parity.main}[command], [flag]
    )
    assert result.exit_code == 0, result.output
    assert "--output" in result.output


@pytest.mark.parametrize("names", [[], ["first.weight", "second.weight"]])
def test_checkpointcli(monkeypatch, tmp_path, names):
    calls = []
    monkeypatch.setattr(
        checkpoint, "convert", lambda output, tensors: calls.append((output, tensors))
    )
    output = tmp_path / "weights"
    args = ["--output", str(output)]
    for name in names:
        args.extend(["--tensor", name])
    result = CliRunner().invoke(checkpoint.main, args)
    assert result.exit_code == 0, result.output
    assert calls == [(output, names or None)]


@pytest.mark.parametrize("allow_other", [False, True])
def test_paritycli(monkeypatch, tmp_path, allow_other):
    from ops.flame import parity

    calls = []

    def run(directory, tokens, *, require_t4):
        calls.append((directory, tokens, require_t4))
        return {"passed": True}

    monkeypatch.setattr(parity, "run", run)
    tokens = tmp_path / "tokens.json"
    tokens.write_text("[[1, 2]]")
    output = tmp_path / "report.json"
    args = [str(tmp_path), "--tokens", str(tokens), "--output", str(output)]
    if allow_other:
        args.append("--allow-other-device")
    result = CliRunner().invoke(parity.main, args)
    assert result.exit_code == 0, result.output
    assert calls == [(tmp_path, [[1, 2]], not allow_other)]
    assert json.loads(output.read_text()) == {"passed": True}


@pytest.mark.parametrize("command", ["checkpoint", "parity"])
def test_cliarguments(command):
    from ops.flame import parity

    result = CliRunner().invoke(
        {"checkpoint": checkpoint.main, "parity": parity.main}[command], []
    )
    assert result.exit_code == 2
    assert "Missing" in result.output


@pytest.fixture
def example():
    config = Config(
        layers=3,
        width=8,
        heads=2,
        vocab=32,
        dense_width=12,
        expert_width=4,
        shared_width=8,
        experts=4,
        top_k=2,
        context=16,
    )
    rng = np.random.default_rng(42)
    params = {
        name: rng.normal(0, 0.15, shape).astype(np.float32)
        for name, shape in config.shapes().items()
    }
    for name, value in params.items():
        if value.ndim == 1:
            params[name] = value + 1
    tokens = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int32)
    return config, params, tokens


@pytest.mark.parametrize("perturbed", [False, True])
@pytest.mark.parametrize("bf16", [False, True])
def test_forwardparity(example, perturbed, bf16):
    config, params, tokens = example
    if perturbed:
        rng = np.random.default_rng(97)
        params = {
            name: value
            + np.float32(0.01) * rng.choice(np.array([-1, 1], np.float32), value.shape)
            for name, value in params.items()
        }
    jp = jax.tree.map(jnp.asarray, params)
    tp = {name: torch.from_numpy(value) for name, value in params.items()}
    if bf16:
        jp = jax.tree.map(lambda value: value.astype(jnp.bfloat16), jp)
        tp = {name: value.bfloat16() for name, value in tp.items()}
    evaluate = jax.jit(lambda p, t: model.forward(p, t, config, capture=True))
    actual, traces = evaluate(jp, jnp.asarray(tokens))
    expected, taps = reference.forward(
        tp, torch.from_numpy(tokens).long(), config, capture=True
    )
    np.testing.assert_allclose(actual, expected.numpy(), atol=3e-6, rtol=3e-5)
    for layer in taps:
        for name, value in taps[layer].items():
            if name == "experts":
                np.testing.assert_array_equal(traces[layer][name], value.numpy())
            else:
                np.testing.assert_allclose(
                    traces[layer][name], value.numpy(), atol=3e-6, rtol=3e-5
                )
    losses = model.token_loss(actual, jnp.asarray(tokens))
    expected_loss = torch.nn.functional.cross_entropy(
        expected[:, :-1].reshape(-1, config.vocab),
        torch.from_numpy(tokens[:, 1:].copy()).long().reshape(-1),
        reduction="none",
    ).reshape(tokens.shape[0], -1)
    np.testing.assert_allclose(losses, expected_loss.numpy(), atol=1e-6, rtol=1e-6)


def test_causal(example):
    config, params, tokens = example
    altered = tokens.copy()
    altered[:, -1] += 1
    first = model.forward(params, jnp.asarray(tokens), config)
    second = model.forward(params, jnp.asarray(altered), config)
    np.testing.assert_array_equal(first[:, :-1], second[:, :-1])


def test_presoftmax():
    _, probs, ids = model.route(jnp.ones((1, 4)), jnp.zeros((8, 4)), 2)
    np.testing.assert_allclose(probs, [[0.125, 0.125]])
    np.testing.assert_array_equal(ids, [[0, 1]])
    assert float(probs.sum()) == 0.25


def test_routerties(example):
    config, params, tokens = example
    for name in params:
        if name.endswith("router.weight"):
            params[name] = np.zeros_like(params[name])
    actual, traces = model.forward(params, jnp.asarray(tokens), config, capture=True)
    expected, taps = reference.forward(
        {name: torch.from_numpy(value) for name, value in params.items()},
        torch.from_numpy(tokens).long(),
        config,
        capture=True,
    )
    np.testing.assert_allclose(actual, expected.numpy(), atol=3e-6, rtol=3e-5)
    for layer in ("1", "2"):
        np.testing.assert_array_equal(
            traces[layer]["experts"], taps[layer]["experts"].numpy()
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite(value):
    from ops.flame.parity import finite

    with pytest.raises(ValueError, match="Nonfinite"):
        finite("test", np.array([value]))


def test_badtokens(example):
    config, params, tokens = example
    tokens[0, 0] = -1
    assert np.isnan(model.forward(params, jnp.asarray(tokens), config)).all()
    with pytest.raises(ValueError, match="integer"):
        model.forward(params, jnp.asarray(tokens, dtype=jnp.float32), config)


def test_badlosstokens():
    logits = jnp.zeros((1, 2, 4))
    for tokens in (jnp.array([[0, -1]]), jnp.array([[0, 4]])):
        assert np.isnan(model.token_loss(logits, tokens)).all()
    for tokens in (jnp.array([0, 1]), jnp.array([[0.0, 1.0]])):
        with pytest.raises(ValueError, match="aligned"):
            model.token_loss(logits, tokens)


def test_treeisaview(example):
    _, params, _ = example
    tree = model.parameter_tree(params)
    assert (
        tree["embedding"]["word_embeddings"]["weight"]
        is params["embedding.word_embeddings.weight"]
    )


def test_params():
    assert (
        sum(math.prod(shape) for shape in Config().shapes().values()) == 1_300_024_320
    )
    assert Config().top_k == 6


@pytest.mark.parametrize(
    "kwargs", [{"heads": 3}, {"width": 0}, {"top_k": 65}, {"epsilon": float("nan")}]
)
def test_invalidconfig(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)


def chunk(offset, size):
    return checkpoint.metadata.ChunkStorageMetadata(
        torch.Size(offset), torch.Size(size)
    )


def test_chunkassembly():
    chunks = [chunk((0, 0), (1, 2)), chunk((1, 0), (1, 2))]
    name = "example"
    storage = {
        checkpoint.metadata.MetadataIndex(name, item.offsets): SimpleNamespace(
            relative_path=f"__{i}_0.distcp", offset=0, length=10
        )
        for i, item in enumerate(chunks)
    }
    item = SimpleNamespace(
        size=(2, 2), chunks=chunks, properties=SimpleNamespace(dtype=torch.bfloat16)
    )

    class Source:
        def read(self, path, offset, length):
            out = io.BytesIO()
            torch.save(
                torch.tensor(
                    [1, 2] if path == "__0_0.distcp" else [3, 4], dtype=torch.bfloat16
                ),
                out,
            )
            return out.getvalue()

    actual = checkpoint.read_tensor(
        SimpleNamespace(storage_data=storage), name, item, Source()
    )
    torch.testing.assert_close(
        actual, torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16)
    )


@pytest.mark.parametrize(
    "chunks",
    [
        [chunk((0, 0), (1, 2))],
        [chunk((0, 0), (2, 2)), chunk((0, 0), (2, 2))],
        [chunk((-1, 0), (2, 2))],
        [chunk((0,), (4,))],
    ],
)
def test_invalidchunks(chunks):
    with pytest.raises(ValueError):
        checkpoint.validate_chunks((2, 2), chunks)


def test_picklesafety():
    class Bad:
        def __reduce__(self):
            return eval, ("1 + 1",)

    with pytest.raises(pickle.UnpicklingError, match="Unsupported"):
        checkpoint.MetadataReader(io.BytesIO(pickle.dumps(Bad()))).load()


def test_ignoredrange(monkeypatch):
    class Response:
        status_code = 200
        headers: ClassVar[dict] = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def raise_forstatus(self):
            pass

    Response.raise_for_status = Response.raise_forstatus

    source = checkpoint.Source()
    monkeypatch.setattr(source.session, "get", lambda *args, **kwargs: Response())
    try:
        with pytest.raises(ValueError, match="honor"):
            source.read("__0_0.distcp", 100, 10)
        with pytest.raises(ValueError, match="shard name"):
            source.read("../other")
    finally:
        source.close()


@pytest.mark.parametrize(
    "body,content_range",
    [(b"abc", "bytes 100-109/1000"), (b"0123456789", "bytes 0-9/1000")],
)
def test_rangepayload(monkeypatch, body, content_range):
    class Response:
        status_code = 206
        headers: ClassVar[dict] = {"Content-Range": content_range}
        raw = io.BytesIO(body)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def raise_forstatus(self):
            pass

    Response.raise_for_status = Response.raise_forstatus

    source = checkpoint.Source()
    monkeypatch.setattr(source.session, "get", lambda *args, **kwargs: Response())
    try:
        with pytest.raises(ValueError):
            source.read("__0_0.distcp", 100, 10)
    finally:
        source.close()


def test_checkpoint(tmp_path):
    import json

    from ops.flame.config import ITERATION, MODEL_ID, REVISION

    manifest = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "iteration": ITERATION,
        "complete": False,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="incomplete"):
        model.load_checkpoint(tmp_path)
