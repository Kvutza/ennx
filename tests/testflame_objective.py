"""Solution masking, streaming reduction, and prepared-objective validation."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("safetensors")

from ops.flame import bo, model
from ops.flame.config import Config
from ops.flame.layout import Layout
from ops.flame.objective import FORMAT, SolutionObjective


@pytest.mark.parametrize("temporary,expected", [(10, 100), (150, 214)])
def test_workspace(temporary, expected):
    mib = 1024**2
    stats = SimpleNamespace(temp_size_in_bytes=temporary * mib, output_size_in_bytes=0)
    assert bo.memory_budget(100 * mib, 100 * mib, stats) == expected * mib


def test_mem():
    with pytest.raises(RuntimeError, match="statistics"):
        bo.memory_budget(100, 100, None)


@pytest.fixture
def config():
    return Config(
        layers=2,
        width=4,
        heads=1,
        vocab=8,
        dense_width=4,
        expert_width=2,
        shared_width=2,
        experts=2,
        top_k=1,
        context=8,
    )


@pytest.fixture
def document():
    return {
        "format": FORMAT,
        "provenance": {"dataset": "test-only"},
        "examples": [
            {
                "id": "a",
                "tokens": [1, 2, 3, 4],
                "loss_mask": [False, False, True, True],
            },
            {"id": "b", "tokens": [5, 6, 7], "loss_mask": [False, False, True]},
        ],
    }


def test_inventory(document, config):
    parsed = SolutionObjective.parse(document, config)
    assert parsed.metadata["input_tokens"] == 7
    assert parsed.metadata["scored_tokens"] == 3
    assert parsed.metadata["sequence_lengths"] == [4, 3]
    assert parsed.metadata["solution_tokens_per_example"] == [2, 1]
    assert parsed.metadata["microbatch_size"] == 1
    assert parsed.metadata["device_input_bytes"] == 38
    assert parsed.tokens.tolist() == [[1, 2, 3, 4], [5, 6, 7, 0]]
    assert parsed.mask.tolist() == [
        [False, False, True, True],
        [False, False, True, False],
    ]
    assert (
        parsed.metadata == SolutionObjective.parse(deepcopy(document), config).metadata
    )
    document["examples"][0]["tokens"][0] = 0
    assert (
        parsed.metadata["sha256"]
        != SolutionObjective.parse(document, config).metadata["sha256"]
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("id", ""),
        ("id", 1),
        ("tokens", [True, 2, 3, 4]),
        ("tokens", [1.0, 2, 3, 4]),
        ("tokens", [1, 2, 3, 8]),
        ("tokens", [-1, 2, 3, 4]),
        ("tokens", [1]),
        ("tokens", [1] * 9),
        ("loss_mask", [False] * 4),
        ("loss_mask", [True] * 4),
        ("loss_mask", [False, True, False, True]),
        ("loss_mask", [False, False, True, False]),
        ("loss_mask", [0, 0, 1, 1]),
        ("loss_mask", [False, True]),
    ],
)
def test_example(document, config, field, value):
    document["examples"][0][field] = value
    with pytest.raises(ValueError):
        SolutionObjective.parse(document, config)


@pytest.mark.parametrize(
    "field,value",
    [
        ("format", "unknown"),
        ("examples", []),
        ("examples", [None]),
        ("provenance", None),
    ],
)
def test_baddocument(document, config, field, value):
    document[field] = value
    with pytest.raises((TypeError, ValueError)):
        SolutionObjective.parse(document, config)


def test_duplicateids(document, config):
    document["examples"][1]["id"] = "a"
    with pytest.raises(ValueError, match="unique"):
        SolutionObjective.parse(document, config)


@pytest.mark.parametrize("batch_signature", [False, True])
def test_streamingloss(document, config, batch_signature):
    rng = np.random.default_rng(7)
    with jax.default_device(jax.devices("cpu")[0]):
        params = {
            key: jnp.asarray(rng.normal(0, 0.2, shape), dtype=jnp.bfloat16)
            for key, shape in config.shapes().items()
        }
        layout = Layout.from_params(params)
        flat = layout.flatten(params)
        evaluate = bo.loss_evaluator(layout, document, config)
        assert isinstance(evaluate.tokens, jax.Array)
        assert isinstance(evaluate.masks, jax.Array)
        assert evaluate.tokens.shape == (2, 4)
        assert evaluate.masks.shape == (2, 3)
        if batch_signature:
            evaluate = evaluate.compile(
                jax.ShapeDtypeStruct((1, layout.size), jnp.bfloat16)
            )
        for candidate in (flat, flat * jnp.bfloat16(2)):
            expected = []
            problem_means = []
            for example in document["examples"]:
                tokens = jnp.asarray([example["tokens"]])
                logits = np.asarray(
                    model.forward(layout.unflatten(candidate), tokens, config)
                )[0, :-1]
                logits = logits - logits.max(axis=-1, keepdims=True)
                log_probs = logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))
                losses = -log_probs[np.arange(len(logits)), example["tokens"][1:]]
                expected.extend(losses[np.array(example["loss_mask"][1:])])
                problem_means.append(
                    float(losses[np.array(example["loss_mask"][1:])].mean())
                )
            actual = evaluate(
                candidate.reshape(1, -1) if batch_signature else candidate
            ).block_until_ready()
            assert actual.shape == () and actual.dtype == jnp.float32
            np.testing.assert_allclose(actual, -np.mean(expected), rtol=1e-6)
            selected = evaluate.losses(
                candidate.reshape(1, -1) if batch_signature else candidate,
                np.array([1, 0]),
            )
            np.testing.assert_allclose(selected, problem_means[::-1], rtol=1e-6)


def test_masks(document, config, monkeypatch):
    class LayoutStub:
        size = 1

        def unflatten(self, flat):
            return flat

    def forward(params, tokens, config):
        # Every prediction position has a distinct loss, exposing shift mistakes.
        return jnp.arange(tokens.shape[1], dtype=jnp.float32)[None, :, None]

    monkeypatch.setattr(model, "forward", forward)
    monkeypatch.setattr(model, "token_loss", lambda logits, tokens: logits[:, :-1, 0])
    with jax.default_device(jax.devices("cpu")[0]):
        evaluate = bo.loss_evaluator(LayoutStub(), document, config)
        # a scores prediction positions 1 and 2; b scores position 1 only.
        assert float(evaluate(jnp.zeros(1))) == pytest.approx(-4 / 3)


@pytest.mark.parametrize("indices", [[], [0, 0], [-1], [2], [[0]], [True], [0.5]])
def test_index(indices):
    from ops.flame.objective import SolutionEvaluator

    def kernel(*args):
        pytest.fail("Invalid problem indices must fail before forward")

    evaluate = SolutionEvaluator(kernel, (None, None), (None, None), 3, (2, 1))
    with pytest.raises(ValueError, match="indices"):
        evaluate.losses(None, indices)
