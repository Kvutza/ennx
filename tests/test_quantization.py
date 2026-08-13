from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from ennx.quantization import quantize_fp4_e2m1, quantize_int4


def test_pack_odd():
    x = np.array([0.0, 1.0, 2.0], dtype=np.float32)

    np.testing.assert_array_equal(quantize_int4(x), np.array([0x10, 0x02]))
    np.testing.assert_array_equal(quantize_fp4_e2m1(x), np.array([0x20, 0x04]))


def test_rust_parity():
    if importlib.util.find_spec("ennx.ennx_rust") is None:
        pytest.skip("ennx_rust extension unavailable")
    rust = pytest.importorskip("ennx._rust")

    x = np.array([-2.0, 0.5, 1.5, 2.5, 20.0], dtype=np.float32)

    np.testing.assert_array_equal(
        quantize_int4(x),
        np.asarray(rust.quantize_int4(x), dtype=np.uint8),
    )
    np.testing.assert_array_equal(
        quantize_fp4_e2m1(x),
        np.asarray(rust.quantize_fp4_e2m1(x), dtype=np.uint8),
    )
