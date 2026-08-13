"""Fast vectorized quantization helpers for ENNX bit-packed weights."""

import numpy as np

try:
    from ennx import _rust
except ImportError:  # pragma: no cover - source-only dev mode
    _rust = None

# FP4 E2M1 lookup table
FP4_E2M1_LUT = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)


def quantize_int4(arr: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Quantize a float array into 4-bit packed uint8 byte array.

    Pairs of 4-bit nibbles (low, high) are packed into single uint8 bytes.
    """
    if _rust is not None:
        return np.asarray(_rust.quantize_int4(arr, scale), dtype=np.uint8)
    flat = np.clip(np.round(arr.ravel() / scale), 0, 15).astype(np.uint8)
    if len(flat) % 2 != 0:
        flat = np.pad(flat, (0, 1), mode="constant")
    low = flat[0::2]
    high = flat[1::2]
    return (low | (high << 4)).astype(np.uint8)


def quantize_fp4_e2m1(arr: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Quantize float values to nearest FP4 (E2M1) representation and pack into uint8 bytes."""
    if _rust is not None:
        return np.asarray(_rust.quantize_fp4_e2m1(arr, scale), dtype=np.uint8)
    scaled = arr.ravel() / scale
    # Find nearest index in FP4 LUT
    diffs = np.abs(scaled[:, None] - FP4_E2M1_LUT[None, :])
    codes = np.argmin(diffs, axis=1).astype(np.uint8)
    if len(codes) % 2 != 0:
        codes = np.pad(codes, (0, 1), mode="constant")
    low = codes[0::2]
    high = codes[1::2]
    return (low | (high << 4)).astype(np.uint8)
