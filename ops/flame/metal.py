"""Resident Metal FLAME evaluation; Torch is used only for CPU checkpoint I/O."""

import sys

from .native import NativeEvaluator as _NativeEvaluator
from .native import load_checkpoint, memory_budget

# Unified memory also serves the OS and Python, not only the GPU.
CONTEXT_MARGIN_BYTES = 256 * 1024**2
SYSTEM_RESERVE_BYTES = 4 * 1024**3


def device_info():
    if sys.platform != "darwin":
        raise RuntimeError("Metal BO requires macOS and an Apple GPU")
    from ennx import experimental

    binding = getattr(experimental, "MetalWeights", None)
    if binding is None:
        raise RuntimeError("Rebuild the ENNX extension with --features metal")
    return binding.device_info()


def metal_device():
    return device_info()["name"]


def free_memory():
    info = device_info()
    # Treat Metal's recommended working set as an upper bound, not free RAM.
    # The reserve covers non-Metal checkpoint preparation and other applications.
    return max(
        0,
        info["recommended_working_set_bytes"]
        - info["allocated_bytes"]
        - SYSTEM_RESERVE_BYTES,
    )


def upload_weights(flat):
    import torch

    from ennx.experimental import MetalWeights

    if flat.device.type != "cpu" or flat.dtype != torch.bfloat16 or flat.ndim != 1:
        raise ValueError("Metal upload requires a flat CPU BF16 tensor")
    return MetalWeights(flat.contiguous().view(torch.uint16).numpy())


class NativeEvaluator(_NativeEvaluator):
    binding_name = "MetalFlameEvaluator"
    build_feature = "metal"


__all__ = [
    "NativeEvaluator",
    "load_checkpoint",
    "memory_budget",
]
