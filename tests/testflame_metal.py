"""Metal driver boundaries; actual GPU parity lives in ops.flame.metal_parity."""

import os
import sys
from dataclasses import asdict
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from ops.flame import bo, metal
from ops.flame.config import Config


@pytest.mark.parametrize("sampler,batch", [("gaussian", 2), ("correlated", None)])
def test_pair(sampler, batch):
    with pytest.raises(ValueError, match="correlated sampling and paired minibatches"):
        bo.Settings(backend="metal", sampler=sampler, minibatch_size=batch)


def test_borrow():
    source = object()
    with bo.borrowed_weights(source, "metal") as actual:
        assert actual is source


def test_nonmacos(monkeypatch):
    monkeypatch.setattr(metal.sys, "platform", "linux")
    with pytest.raises(RuntimeError, match="macOS"):
        metal.device_info()


@pytest.mark.parametrize("allocated", [0, 2 * 1024**3, 20 * 1024**3])
def test_memory(monkeypatch, allocated):
    recommended = 18 * 1024**3
    monkeypatch.setattr(
        metal,
        "device_info",
        lambda: {
            "recommended_working_set_bytes": recommended,
            "allocated_bytes": allocated,
        },
    )
    assert metal.free_memory() == max(
        0, recommended - allocated - metal.SYSTEM_RESERVE_BYTES
    )


def test_uploadbits(monkeypatch):
    torch = pytest.importorskip("torch")
    captured = []
    package = ModuleType("ennx")
    experimental = ModuleType("ennx.experimental")
    experimental.MetalWeights = lambda bits: captured.append(bits.copy())
    package.experimental = experimental
    monkeypatch.setitem(sys.modules, "ennx", package)
    monkeypatch.setitem(sys.modules, "ennx.experimental", experimental)
    value = torch.tensor([1.0, -2.0, 0.0], dtype=torch.bfloat16)
    metal.upload_weights(value)
    np.testing.assert_array_equal(captured[0], [0x3F80, 0xC000, 0])
    for invalid in (value.float(), value.reshape(1, -1)):
        with pytest.raises(ValueError, match="flat CPU BF16"):
            metal.upload_weights(invalid)
    with pytest.raises(ValueError, match="flat CPU BF16"):
        metal.upload_weights(SimpleNamespace(device=SimpleNamespace(type="cuda")))


@pytest.fixture
def gpu():
    if os.environ.get("ENNX_TEST_METAL") != "1":
        pytest.skip("Set ENNX_TEST_METAL=1 to exercise actual Metal bindings")
    from ennx import experimental

    assert "Apple" in experimental.MetalWeights.device_info()["name"]
    return experimental


def test_weightupload(gpu):
    bits = np.array([0x3F80, 0xC000, 0], dtype=np.uint16)
    weights = gpu.MetalWeights(bits)
    bits[:] = 0
    np.testing.assert_array_equal(weights.read(), [0x3F80, 0xC000, 0])
    for invalid in (
        np.array([], dtype=np.uint16),
        np.array([0x7F80], dtype=np.uint16),
        np.array([0xFF80], dtype=np.uint16),
        np.array([0x7FC0], dtype=np.uint16),
        np.zeros((1, 2), dtype=np.uint16),
        np.zeros(2, dtype=np.float32),
        np.zeros(4, dtype=np.uint16)[::2],
    ):
        with pytest.raises((ValueError, TypeError)):
            gpu.MetalWeights(invalid)


def test_forward(gpu):
    config = Config(
        layers=1,
        width=4,
        heads=1,
        vocab=8,
        dense_width=6,
        expert_width=2,
        shared_width=4,
        experts=3,
        top_k=2,
        context=8,
    )
    for key, invalid in (
        ("width", True),
        ("heads", 0),
        ("epsilon", float("nan")),
        ("epsilon", True),
        ("top_k", 4),
    ):
        with pytest.raises((ValueError, TypeError)):
            gpu.MetalFlameEvaluator({**asdict(config), key: invalid}, 4)
    for invalid in (True, 0, 9):
        with pytest.raises((ValueError, TypeError)):
            gpu.MetalFlameEvaluator(asdict(config), invalid)
    evaluator = gpu.MetalFlameEvaluator(asdict(config), 4)
    weights = gpu.MetalWeights(np.full(evaluator.weights_len, 0x3D80, dtype=np.uint16))
    expected = evaluator.losses(weights, [[1, 2]], [[False, True]])
    for tokens, masks in (
        ([], []),
        ([[1]], [[False]]),
        ([[8, 1]], [[False, True]]),
        ([[1, 2]], [[True, True]]),
        ([[1, 2]], [[False, False]]),
        ([[1, 2]], [[False]]),
        ([[1] * 5], [[False] + [True] * 4]),
    ):
        with pytest.raises(ValueError):
            evaluator.losses(weights, tokens, masks)
    for invalid in (
        np.zeros(evaluator.weights_len, dtype=np.uint16),
        gpu.MetalWeights(np.zeros(1, dtype=np.uint16)),
    ):
        with pytest.raises((TypeError, ValueError)):
            evaluator.losses(invalid, [[1, 2]], [[False, True]])
    np.testing.assert_array_equal(
        evaluator.losses(weights, [[1, 2]], [[False, True]]), expected
    )


def test_nextlogits(gpu):
    config = Config(
        layers=2,
        width=40,
        heads=2,
        vocab=71,
        dense_width=15,
        expert_width=9,
        shared_width=11,
        experts=3,
        top_k=2,
        context=140,
    )
    evaluator = gpu.MetalFlameEvaluator(asdict(config), config.context)
    rng = np.random.default_rng(8123)
    values = rng.uniform(-0.1, 0.1, evaluator.weights_len).astype(np.float32)
    bits = (values.view(np.uint32) >> 16).astype(np.uint16)
    weights = gpu.MetalWeights(bits)
    workspace_bytes = evaluator.workspace_bytes
    for n in (1, 2, 7, 8, 9, 31, 32, 33, 127, 128, 129, 135, 136, 140, 1):
        tokens = [(i * 13) % config.vocab for i in range(n)]
        tokens[-1] = config.vocab - 1
        full = evaluator.logits(weights, tokens)
        next_logits = evaluator.next_logits(weights, tokens)
        assert_logits(full, next_logits, n, config.vocab)
        assert evaluator.workspace_bytes == workspace_bytes
    retained = next_logits.copy()
    expected = evaluator.next_logits(weights, [0])
    assert_invalid(evaluator, weights, config)
    for invalid in (bits, object(), gpu.MetalWeights(np.zeros(1, dtype=np.uint16))):
        with pytest.raises((ValueError, TypeError)):
            evaluator.next_logits(invalid, [0])
    # Finite BF16 weights can still overflow the LM head. The canonical layout
    # starts with final_layernorm, so scale that tensor to exercise output checks.
    assert_overflow(gpu, evaluator, config.width)
    np.testing.assert_array_equal(evaluator.next_logits(weights, [0]), expected)
    np.testing.assert_array_equal(next_logits, retained)


def assert_invalid(evaluator, weights, config):
    for tokens in ([], [-1], [config.vocab], [0] * 141, [[0]], [1.5], [2**40]):
        with pytest.raises((ValueError, TypeError, OverflowError)):
            evaluator.next_logits(weights, tokens)


def assert_overflow(gpu, evaluator, width):
    overflow = np.full(evaluator.weights_len, 0x3D80, dtype=np.uint16)
    overflow[:width] = 0x7F7F
    with pytest.raises(ValueError, match="nonfinite"):
        evaluator.next_logits(gpu.MetalWeights(overflow), [0])


def test_searchguards(gpu):
    def make():
        state = gpu.MetalSearchState(
            gpu.MetalWeights(np.full(257, 0x3F80, dtype=np.uint16)),
            0.0,
            [gpu.MetalParamBlock(1, 0, 257, 1.0)],
            2,
        )
        state.enable_relative()
        return state

    def ask(state):
        return state.ask(1, 4, 2, 123)

    state, other = make(), make()
    memory = state.memory_info()
    assert memory["row_bytes"] == 514
    assert memory["search_resident_bytes"] >= 5 * memory["row_bytes"]
    assert memory["device_allocated_bytes"] > 0
    assert memory["recommended_working_set_bytes"] > 0
    assert memory["max_buffer_bytes"] >= memory["row_bytes"]
    proposal, foreign = ask(state), ask(other)
    with pytest.raises(ValueError, match="foreign"):
        state.tell_relative(foreign, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, True)
    for index, invalid in (
        (0, float("nan")),
        (1, -1e-300),
        (3, -1.0),
        (4, 1e300),
        (5, -1e-300),
    ):
        values = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        values[index] = invalid
        with pytest.raises(ValueError):
            state.tell_relative(proposal, *values, True)
    view = state.incumbent()
    with pytest.raises(BufferError):
        state.tell_relative(proposal, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, True)
    del view
    state.tell_relative(proposal, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, True)
    assert state.sync() == [True]
    with pytest.raises(ValueError, match="Stale"):
        proposal.read()


def test_fifo(gpu):
    state = gpu.MetalSearchState(
        gpu.MetalWeights(np.full(257, 0x3F80, dtype=np.uint16)),
        0.0,
        [gpu.MetalParamBlock(1, 0, 257, 1.0)],
        5,
    )
    initial = state.length
    for step in range(8):
        proposal = state.ask(1, 4, 5, step)
        if step == 0:
            assert_variances(state, proposal)
            view = state.incumbent()
            with pytest.raises(BufferError):
                state.tell_paired(proposal, 1.0, 0.0, 0.0, 0.0, True)
            del view
        state.tell_paired(proposal, 1.0, 0.1, 0.0, 0.2, step == 0)
        assert state.sync() == [step == 0]
        assert state.history_len == min(step + 2, 5)
        assert state.length == initial
    with pytest.raises(ValueError):
        state.tell_paired(proposal, 1.0, 0.0, 0.0, 0.0, True)


def assert_variances(state, proposal):
    for variance in [-1e-300, float("nan"), 1e300]:
        with pytest.raises(ValueError):
            state.tell_paired(proposal, 1.0, variance, 0.0, 0.0, True)


def test_inconclusive(gpu):
    state = gpu.MetalSearchState(
        gpu.MetalWeights(np.full(257, 0x3F80, dtype=np.uint16)),
        0.0,
        [gpu.MetalParamBlock(1, 0, 257, 1.0)],
        2,
    )
    state.enable_relative()
    initial_radius = state.length
    for step in range(8):
        proposal = state.ask(1, 4, 2, step)
        state.tell_relative(
            proposal,
            0.01,
            0.1,
            0.0,
            0.1,
            0.01,
            0.1,
            False,
            reject_is_failure=False,
        )
        assert state.sync() == [False]
        assert state.history_len == 2
        assert state.length == initial_radius
    for step in range(4):
        proposal = state.ask(1, 4, 2, 100 + step)
        state.tell_relative(
            proposal,
            -1.0,
            0.1,
            0.0,
            0.1,
            -1.0,
            0.1,
            False,
            reject_is_failure=True,
        )
        assert state.sync() == [False]
        assert state.length == (initial_radius if step < 3 else initial_radius * 0.5)


def assert_logits(full, next_logits, n, vocab):
    assert full.shape == (n, vocab)
    assert next_logits.shape == (vocab,)
    assert next_logits.dtype == np.float32
    assert next_logits.flags.c_contiguous
    assert np.isfinite(next_logits).all()
    np.testing.assert_array_equal(next_logits.view(np.uint32), full[-1].view(np.uint32))
