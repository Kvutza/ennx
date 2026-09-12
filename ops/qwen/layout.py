"""Canonical BF16 packing for the native Qwen Metal evaluator."""

from __future__ import annotations

import math
from dataclasses import dataclass


def tensor_key(name: str) -> int:
    key = 1469598103934665603
    for byte in name.encode("utf-8"):
        key = ((key ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return key


def _stats(value, name: str) -> tuple[float, bool]:
    import torch

    if not isinstance(value, torch.Tensor) or value.dtype != torch.bfloat16:
        raise TypeError(f"{name} must be a CPU BF16 tensor")
    if value.device.type != "cpu" or value.numel() == 0:
        raise ValueError(f"{name} must be nonempty and on CPU")
    values = value.detach().float()
    if not bool(torch.isfinite(values).all()):
        raise ValueError(f"{name} must contain only finite values")
    peak = float(values.abs().max())
    if peak == 0.0:
        return 0.0, False
    scaled = values / peak
    rms = peak * float(torch.sqrt(torch.mean(scaled.square())).item())
    if not math.isfinite(rms) or rms <= 0.0:
        raise ValueError(f"{name} has an invalid FP32 RMS")
    return rms, True


@dataclass(frozen=True)
class Block:
    name: str
    key: int
    offset: int
    shape: tuple[int, ...]
    scale: float
    weight: float

    @property
    def length(self) -> int:
        return math.prod(self.shape)


@dataclass(frozen=True)
class Layout:
    blocks: tuple[Block, ...]

    @classmethod
    def from_torch(cls, params, *, zero_scale: float | None = None) -> "Layout":
        if not params or any(not isinstance(name, str) for name in params):
            raise ValueError("params must be a nonempty mapping of named tensors")
        names = tuple(sorted(params))
        if zero_scale is not None and (
            isinstance(zero_scale, bool)
            or not math.isfinite(float(zero_scale))
            or float(zero_scale) <= 0.0
        ):
            raise ValueError("zero_scale must be finite and positive")
        blocks = []
        offset = 0
        for name in names:
            value = params[name]
            rms, nonzero = _stats(value, name)
            if not nonzero:
                if zero_scale is None:
                    raise ValueError(f"{name}: zero tensor requires zero_scale")
                rms = float(zero_scale)
            shape = tuple(int(dim) for dim in value.shape)
            length = math.prod(shape)
            blocks.append(
                Block(
                    name=name,
                    key=tensor_key(name),
                    offset=offset,
                    shape=shape,
                    scale=rms,
                    weight=1.0 / (len(names) * length * rms * rms),
                )
            )
            offset += length
        return cls(tuple(blocks))

    @property
    def size(self) -> int:
        return sum(block.length for block in self.blocks)

    def flatten_torch(self, params):
        import torch

        if tuple(sorted(params)) != tuple(block.name for block in self.blocks):
            raise ValueError("parameter names must exactly match the layout")
        pieces = []
        for block in self.blocks:
            value = params[block.name]
            if tuple(value.shape) != block.shape:
                raise ValueError(f"{block.name}: shape differs from the layout")
            pieces.append(value.detach().reshape(block.length))
        return torch.cat(pieces)

    def metal_blocks(self):
        from ennx.experimental import MetalParamBlock

        return [
            MetalParamBlock(
                block.key,
                block.offset,
                block.length,
                block.scale,
                block.weight,
            )
            for block in self.blocks
        ]
