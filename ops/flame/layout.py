"""Immutable parameter metadata for the full-weight FLAME search space."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import jax


def _tensorkey(name: str) -> int:
    key = 1469598103934665603
    for byte in name.encode("utf-8"):
        key = ((key ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return key


def _positivefp32(value: float, label: str) -> float:
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        value = float(np.float32(value))
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be finite and positive in FP32")
    return value


def _weight(count: int, length: int, scale: float) -> float:
    # Use a wide intermediate; only the final metric weight must fit FP32.
    return _positivefp32(1.0 / (count * length * scale**2), "metric weight")


def _names(params: Mapping[str, jax.Array]) -> tuple[str, ...]:
    if not isinstance(params, Mapping):
        raise TypeError("params must be a mapping of names to BF16 arrays")
    if not params:
        raise ValueError("params must be nonempty")
    if any(not isinstance(name, str) for name in params):
        raise TypeError("parameter names must be strings")
    return tuple(sorted(params))


def _checkarray(value: jax.Array, label: str) -> None:
    import jax.numpy as jnp

    if not hasattr(value, "dtype") or value.dtype != jnp.bfloat16:
        raise TypeError(f"{label} must have BF16 dtype")
    if value.size == 0:
        raise ValueError(f"{label} must be nonempty")


@cache
def _jaxkernels():
    import jax
    import jax.numpy as jnp

    @jax.jit
    def all_finite(value):
        return jnp.all(jnp.isfinite(value))

    @jax.jit
    def reference_stats(value):
        fp32 = value.astype(jnp.float32)
        # Preserve FP32 square/reduction rounding for the reference backend.
        mean_square = jax.lax.optimization_barrier(
            jnp.mean(jnp.square(fp32), dtype=jnp.float32)
        )
        return all_finite(value), jnp.any(value != 0), jnp.sqrt(mean_square)

    return all_finite, reference_stats


def _checktorch(value, label):
    import torch

    if not isinstance(value, torch.Tensor) or value.dtype != torch.bfloat16:
        raise TypeError(f"{label} must have BF16 dtype")
    if value.device.type != "cpu":
        raise ValueError(f"{label} must be on CPU for checkpoint preparation")
    if value.numel() == 0:
        raise ValueError(f"{label} must be nonempty")


def _torchstats(value, label):
    _checktorch(value, label)
    fp32 = value.detach().float().numpy()
    if not np.all(np.isfinite(fp32)):
        return False, False, np.nan
    peak = np.max(np.abs(fp32))
    if peak == 0:
        return True, False, 0.0
    # Scale before squaring so finite BF16 inputs cannot overflow the FP32 RMS.
    scaled = fp32 / peak
    rms = peak * np.sqrt(np.mean(np.square(scaled), dtype=np.float32))
    return True, True, rms


@dataclass(frozen=True)
class Block:
    name: str
    key: int
    offset: int
    shape: tuple[int, ...]
    scale: float
    weight: float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or self.key != _tensorkey(self.name):
            raise ValueError("block key must match its UTF-8 name")
        if type(self.offset) is not int or self.offset < 0:
            raise ValueError("block offset must be a nonnegative integer")
        if not isinstance(self.shape, tuple) or any(
            type(dim) is not int or dim <= 0 for dim in self.shape
        ):
            raise ValueError("block shape must be a tuple of positive dimensions")
        object.__setattr__(self, "scale", _positivefp32(self.scale, "scale"))
        object.__setattr__(self, "weight", _positivefp32(self.weight, "metric weight"))

    @property
    def length(self) -> int:
        return math.prod(self.shape)


@dataclass(frozen=True)
class Layout:
    """Canonical blocks with scales fixed at reference construction time.

    Construction and flattening are eager validation boundaries. Unflattening
    checks only static dtype/shape metadata, so it can run inside JAX transforms;
    callers must ensure the flat candidate's values are finite.
    """

    blocks: tuple[Block, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.blocks, tuple) or not self.blocks:
            raise ValueError("blocks must be a nonempty tuple")
        if any(not isinstance(block, Block) for block in self.blocks):
            raise TypeError("blocks must contain Block records")
        names = tuple(block.name for block in self.blocks)
        if names != tuple(sorted(set(names))):
            raise ValueError("block names must be unique and canonically sorted")
        if len({block.key for block in self.blocks}) != len(self.blocks):
            raise ValueError("tensor key collision")
        offset = 0
        for block in self.blocks:
            if block.offset != offset:
                raise ValueError("block offsets must be contiguous starting at zero")
            if block.weight != _weight(len(self.blocks), block.length, block.scale):
                raise ValueError("block metric weight does not match layout")
            offset += block.length

    @classmethod
    def from_params(
        cls, params: Mapping[str, jax.Array], *, zero_scale: float | None = None
    ) -> Layout:
        import jax

        def stats(value, name):
            _checkarray(value, name)
            return jax.device_get(_jaxkernels()[1](value))

        return cls._fromreference(params, stats, zero_scale)

    @classmethod
    def from_torch(cls, params, *, zero_scale=None) -> Layout:
        """Build canonical metadata from CPU BF16 checkpoint tensors.

        Numerical RMS scales and metric weights may differ between backends
        because of FP32 reduction rounding.
        """
        return cls._fromreference(params, _torchstats, zero_scale)

    @classmethod
    def _fromreference(cls, params, stats, zero_scale):
        names = _names(params)
        if zero_scale is not None:
            if isinstance(zero_scale, (bool, str, bytes)) or np.ndim(zero_scale) != 0:
                raise TypeError("zero_scale must be a positive scalar")
            zero_scale = _positivefp32(zero_scale, "zero_scale")
        blocks = []
        offset = 0
        for name in names:
            value = params[name]
            finite, nonzero, rms = stats(value, name)
            if not finite:
                raise ValueError(f"{name} must contain only finite values")
            if not nonzero:
                if zero_scale is None:
                    raise ValueError(
                        f"{name}: zero tensor requires explicit zero_scale"
                    )
                scale = zero_scale
            else:
                scale = _positivefp32(rms, f"{name}: reference RMS scale")
            shape = tuple(value.shape)
            length = math.prod(shape)
            blocks.append(
                Block(
                    name,
                    _tensorkey(name),
                    offset,
                    shape,
                    scale,
                    _weight(len(names), length, scale),
                )
            )
            offset += length
        return cls(tuple(blocks))

    @property
    def size(self) -> int:
        return self.blocks[-1].offset + self.blocks[-1].length

    def flatten(self, params: Mapping[str, jax.Array]) -> jax.Array:
        import jax
        import jax.numpy as jnp

        if _names(params) != tuple(block.name for block in self.blocks):
            raise ValueError("parameter names must exactly match the layout")
        pieces = []
        for block in self.blocks:
            value = params[block.name]
            _checkarray(value, block.name)
            if tuple(value.shape) != block.shape:
                raise ValueError(f"{block.name}: shape must match {block.shape}")
            if not jax.device_get(_jaxkernels()[0](value)):
                raise ValueError(f"{block.name} must contain only finite values")
            pieces.append(jnp.reshape(value, (block.length,)))
        return jnp.concatenate(pieces)

    def flatten_torch(self, params):
        """Pack CPU BF16 weights; only the returned flat tensor is uploaded."""
        import torch

        if _names(params) != tuple(block.name for block in self.blocks):
            raise ValueError("parameter names must exactly match the layout")
        pieces = []
        for block in self.blocks:
            value = params[block.name]
            _checktorch(value, block.name)
            if tuple(value.shape) != block.shape:
                raise ValueError(f"{block.name}: shape must match {block.shape}")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{block.name} must contain only finite values")
            pieces.append(value.detach().reshape(block.length))
        return torch.cat(pieces)

    def unflatten(self, flat: jax.Array) -> dict[str, jax.Array]:
        import jax.numpy as jnp

        _checkarray(flat, "flat")
        if tuple(flat.shape) != (self.size,):
            raise ValueError(f"flat must have shape ({self.size},)")
        flat = jnp.asarray(flat)
        return {
            block.name: flat[block.offset : block.offset + block.length].reshape(
                block.shape
            )
            for block in self.blocks
        }

    def describe(self) -> list[dict[str, object]]:
        return [
            {
                "name": block.name,
                "key": block.key,
                "offset": block.offset,
                "shape": list(block.shape),
                "length": block.length,
                "scale": block.scale,
                "weight": block.weight,
            }
            for block in self.blocks
        ]
