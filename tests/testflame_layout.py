import json
from dataclasses import FrozenInstanceError, fields, replace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from ops.flame.layout import Block, Layout


def bf16(values):
    return jnp.asarray(values, dtype=jnp.bfloat16)


@pytest.fixture
def params():
    return {
        "z.bias": bf16([2, -2]),
        "a.weight": bf16([[1, -2], [3, -4]]),
        "scalar": bf16(0.5),
    }


def test_layout(params):
    layout = Layout.from_params(params)
    assert layout == Layout.from_params(dict(reversed(list(params.items()))))
    assert [b.name for b in layout.blocks] == ["a.weight", "scalar", "z.bias"]
    assert [b.offset for b in layout.blocks] == [0, 4, 5]
    assert [b.shape for b in layout.blocks] == [(2, 2), (), (2,)]
    assert [b.length for b in layout.blocks] == [4, 1, 2]
    assert layout.size == 7
    flat = layout.flatten(params)
    assert isinstance(flat, jax.Array)
    assert flat.dtype == jnp.bfloat16
    np.testing.assert_array_equal(flat, [1, -2, 3, -4, 0.5, 2, -2])
    restored = layout.unflatten(flat)
    for name, value in params.items():
        assert restored[name].dtype == jnp.bfloat16
        assert restored[name].shape == value.shape
        np.testing.assert_array_equal(restored[name], value)
    records = layout.describe()
    assert json.loads(json.dumps(records, allow_nan=False)) == records
    assert records[0] == {
        "name": "a.weight",
        "key": layout.blocks[0].key,
        "offset": 0,
        "shape": [2, 2],
        "length": 4,
        "scale": layout.blocks[0].scale,
        "weight": layout.blocks[0].weight,
    }
    records[0]["shape"][0] = 999
    assert layout.blocks[0].shape == (2, 2)


@pytest.mark.parametrize(
    "name,key",
    [
        ("", 1469598103934665603),
        ("blk.27.attn_q.weight", 3843877851495245630),
        ("a", 4953267810257967366),
        ("\u00e9", 11062259058118930795),
    ],
)
def test_keys(name, key):
    assert Layout.from_params({name: bf16(1)}).blocks[0].key == key


def test_scales(params):
    layout = Layout.from_params(params)
    for block in layout.blocks:
        host = np.asarray(params[block.name], dtype=np.float32)
        expected = np.sqrt(np.mean(host * host, dtype=np.float32))
        # Device reductions/sqrt may round differently from NumPy in FP32.
        np.testing.assert_array_max_ulp(np.float32(block.scale), expected, maxulp=4)
        expected_weight = np.float32(1 / (3 * host.size * block.scale**2))
        assert block.weight == float(expected_weight)
    before = layout.describe()
    candidates = {name: value * bf16(2) for name, value in params.items()}
    layout.flatten(candidates)
    layout.flatten({name: jnp.zeros_like(value) for name, value in params.items()})
    assert layout.describe() == before


def test_torchlayout(params):
    torch = pytest.importorskip("torch")
    reference = Layout.from_params(params)
    host = {
        name: torch.from_numpy(np.asarray(value, dtype=np.float32).copy()).bfloat16()
        for name, value in params.items()
    }
    native = Layout.from_torch(host)
    for actual, expected in zip(native.blocks, reference.blocks, strict=True):
        assert (actual.name, actual.key, actual.offset, actual.shape) == (
            expected.name,
            expected.key,
            expected.offset,
            expected.shape,
        )
        np.testing.assert_array_max_ulp(
            np.float32(actual.scale), np.float32(expected.scale), maxulp=4
        )
        assert actual.weight == float(
            np.float32(1 / (len(native.blocks) * actual.length * actual.scale**2))
        )
    np.testing.assert_array_equal(
        native.flatten_torch(host).view(torch.uint16).numpy(),
        jax.lax.bitcast_convert_type(reference.flatten(params), jnp.uint16),
    )


def test_unflatten(params):
    layout = Layout.from_params(params)

    @jax.jit
    def evaluate(flat, x):
        named = layout.unflatten(flat)
        return (named["a.weight"] @ x + named["z.bias"]) * named["scalar"]

    flat = layout.flatten(params)
    np.testing.assert_array_equal(evaluate(flat, bf16([2, 3])), [-1, -4])
    np.testing.assert_array_equal(evaluate(flat * bf16(2), bf16([2, 3])), [-4, -16])
    restored = jax.jit(layout.unflatten)(flat)
    for name in params:
        np.testing.assert_array_equal(restored[name], params[name])


def test_scalar(params, monkeypatch):
    params = {**params, "zero": bf16([0, 0])}
    array_type = type(params["a.weight"])
    original_array = array_type.__array__
    original_get = jax.device_get
    transfers = []

    def scalar_array(value, *args, **kwargs):
        assert value.shape == (), "full device array converted to NumPy"
        return original_array(value, *args, **kwargs)

    def scalar_get(value):
        leaves = jax.tree.leaves(value)
        assert all(leaf.shape == () for leaf in leaves), "full device array transferred"
        transfers.extend(leaves)
        return original_get(value)

    with monkeypatch.context() as patch:
        patch.setattr(array_type, "__array__", scalar_array)
        patch.setattr(jax, "device_get", scalar_get)
        layout = Layout.from_params(params, zero_scale=0.25)
        flat = layout.flatten(params)
        restored = jax.jit(layout.unflatten)(flat)

    assert transfers
    for name in params:
        np.testing.assert_array_equal(restored[name], params[name])


def test_immutable(params):
    layout = Layout.from_params(params)
    assert [field.name for field in fields(layout)] == ["blocks"]
    assert isinstance(layout.blocks, tuple)
    assert [field.name for field in fields(Block)] == [
        "name",
        "key",
        "offset",
        "shape",
        "scale",
        "weight",
    ]
    with pytest.raises(FrozenInstanceError):
        layout.blocks = ()
    with pytest.raises(FrozenInstanceError):
        layout.blocks[0].scale = 42


@pytest.mark.parametrize("bad", [{}, {"x": bf16([])}, {"x": bf16(np.empty((2, 0)))}])
def test_rejectempty(bad):
    with pytest.raises(ValueError, match="nonempty"):
        Layout.from_params(bad)


@pytest.mark.parametrize(
    "bad", [[], {1: bf16([1])}, {"x": [1]}, {"x": jnp.ones(2)}, {"x": jnp.array([1])}]
)
def test_types(bad):
    with pytest.raises(TypeError):
        Layout.from_params(bad)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_values(value):
    bad = {"x": bf16([1, value])}
    with pytest.raises(ValueError, match="finite"):
        Layout.from_params(bad)
    layout = Layout.from_params({"x": bf16([1, 2])})
    with pytest.raises(ValueError, match="finite"):
        layout.flatten(bad)


def test_scale():
    params = {"zero": bf16([0, -0.0]), "nonzero": bf16([2])}
    with pytest.raises(ValueError, match="zero_scale"):
        Layout.from_params(params)
    layout = Layout.from_params(params, zero_scale=0.25)
    assert [b.scale for b in layout.blocks] == [2, 0.25]
    assert layout.blocks[1].weight == 4


@pytest.mark.parametrize("scale", [0, -1, float("inf"), float("nan"), 1e-100, 1e100])
def test_zero(scale):
    with pytest.raises(ValueError, match="zero_scale"):
        Layout.from_params({"x": bf16([1])}, zero_scale=scale)


@pytest.mark.parametrize("scale", [True, "1", [1]])
def test_zeroscaletype(scale):
    with pytest.raises(TypeError, match="zero_scale"):
        Layout.from_params({"x": bf16([0])}, zero_scale=scale)


@pytest.mark.parametrize("value", [1e30, 1e-30])
def test_rms(value):
    with pytest.raises(ValueError, match="reference RMS scale"):
        Layout.from_params({"x": bf16([value])}, zero_scale=1)


@pytest.mark.parametrize("scale", [1e-20, 1e30])
def test_weight(scale):
    with pytest.raises(ValueError, match="metric weight"):
        Layout.from_params({"x": bf16([0])}, zero_scale=scale)


def test_wideweight():
    layout = Layout.from_params({"x": bf16([0, 0])}, zero_scale=1e20)
    assert 0 < layout.blocks[0].weight < np.finfo(np.float32).tiny


@pytest.mark.parametrize(
    "bad",
    [
        {"y": bf16([1, 2])},
        {"x": bf16([1, 2]), "y": bf16([1])},
        {"x": bf16([[1, 2]])},
        {"x": bf16([1])},
    ],
)
def test_flattenschema(bad):
    layout = Layout.from_params({"x": bf16([1, 2])})
    with pytest.raises(ValueError):
        layout.flatten(bad)


def test_flattendtype():
    layout = Layout.from_params({"x": bf16([1])})
    with pytest.raises(TypeError, match="BF16"):
        layout.flatten({"x": jnp.ones(1)})


@pytest.mark.parametrize("flat", [bf16([1]), bf16([1, 2, 3]), bf16([[1, 2]]), bf16(1)])
def test_shape(flat):
    layout = Layout.from_params({"x": bf16([1, 2])})
    for unpack in (layout.unflatten, jax.jit(layout.unflatten)):
        with pytest.raises(ValueError, match="shape"):
            unpack(flat)


def test_dtype():
    layout = Layout.from_params({"x": bf16([1, 2])})
    for unpack in (layout.unflatten, jax.jit(layout.unflatten)):
        with pytest.raises(TypeError, match="BF16"):
            unpack(jnp.ones(2))


def test_meta(params):
    layout = Layout.from_params(params)
    first, *rest = layout.blocks
    for blocks in (
        (),
        list(layout.blocks),
        tuple(reversed(layout.blocks)),
        (first, first),
        (replace(first, offset=1), *rest),
        (replace(first, weight=1), *rest),
    ):
        with pytest.raises(ValueError):
            Layout(blocks)
    for changes in (
        {"key": 0},
        {"offset": -1},
        {"shape": [2, 2]},
        {"shape": (0, 2)},
        {"scale": 0},
        {"weight": float("inf")},
    ):
        with pytest.raises(ValueError):
            replace(first, **changes)
