"""CPU checkpoint preparation and synchronous CUDA FLAME objective evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .config import ITERATION, MODEL_ID, REVISION, Config
from .objective import SolutionObjective, problem_indices

# CUDA/cuBLAS context, search metadata, and allocator rounding beyond engine scratch.
CONTEXT_MARGIN_BYTES = 256 * 1024**2


def cuda_device():
    import torch

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and visible.split(",")[0].strip() != "0":
        raise RuntimeError(
            "Native BO requires physical CUDA device 0; check CUDA_VISIBLE_DEVICES"
        )
    if not torch.cuda.is_available() or torch.cuda.current_device() != 0:
        raise RuntimeError("This experiment requires CUDA device 0 to be a T4")
    name = torch.cuda.get_device_name(0)
    if "T4" not in name:
        raise RuntimeError(
            f"This experiment requires CUDA device 0 to be a T4, got {name}"
        )
    return name


def memory_budget(size, history, workspace_bytes):
    if type(workspace_bytes) is not int or workspace_bytes < 0:
        raise ValueError("Native workspace_bytes must be a nonnegative integer")
    rows = (history + 2) * ((size + 127) & ~127) * 2
    # The BF16 input overlaps resident-row construction, then is freed before
    # ask allocates the equally sized correlated reference. Never count both.
    return rows + 2 * size + workspace_bytes + CONTEXT_MARGIN_BYTES


def load_checkpoint(directory: Path):
    """Verify the released manifest and load only CPU BF16 tensors."""
    import torch
    from safetensors.torch import load_file

    manifest = json.loads((directory / "manifest.json").read_text())
    if (
        manifest.get("model_id"),
        manifest.get("revision"),
        manifest.get("iteration"),
    ) != (MODEL_ID, REVISION, ITERATION):
        raise ValueError("Unexpected checkpoint provenance")
    if manifest.get("complete") is not True:
        raise ValueError("Cannot evaluate an incomplete checkpoint")
    config = Config(**manifest["config"])
    shapes = config.shapes()
    if config != Config() or manifest["tensors"].keys() != shapes.keys():
        raise ValueError("Checkpoint does not match the released FLAME architecture")
    params = {}
    for name, record in manifest["tensors"].items():
        filename = record["file"]
        if Path(filename).name != filename:
            raise ValueError("Invalid tensor filename")
        path = directory / filename
        with path.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        if checksum != record["sha256"]:
            raise ValueError(f"Tensor checksum mismatch: {name}")
        values = load_file(str(path), device="cpu")
        if (
            set(values) != {name}
            or tuple(values[name].shape) != shapes[name]
            or values[name].dtype != torch.bfloat16
        ):
            raise ValueError(f"Unexpected tensor layout: {name}")
        params[name] = values[name]
    return config, params


def upload_weights(flat):
    """Upload the one packed BF16 checkpoint buffer to CUDA device zero."""
    return flat.to(device="cuda:0")


def release_inputcache():
    import torch

    # The input's allocation must be reusable by ENNX's CUDA allocator at ask.
    torch.cuda.empty_cache()


class NativeEvaluator:
    """Borrow ENNX/Torch DLPack weights directly for per-problem mean losses.

    FlameEvaluator consumes and releases each DLPack borrow synchronously.
    Python never wraps resident weights in another tensor library.
    """

    binding_name = "FlameEvaluator"
    build_feature = "native-flame"

    def __init__(self, objective: SolutionObjective, config: Config):
        if not isinstance(objective, SolutionObjective):
            raise TypeError("Native BO supports prepared solution-token corpora only")
        from ennx import experimental

        engine_type = getattr(experimental, self.binding_name, None)
        if engine_type is None:
            raise RuntimeError(
                f"Rebuild the ENNX extension with --features {self.build_feature} for {self.binding_name} support"
            )
        lengths = objective.metadata["sequence_lengths"]
        self.tokens = [
            row[:n].tolist() for row, n in zip(objective.tokens, lengths, strict=True)
        ]
        self.masks = [
            row[:n].tolist() for row, n in zip(objective.mask, lengths, strict=True)
        ]
        self.engine = engine_type(asdict(config), max_tokens=max(lengths))
        self.weights_len = self.engine.weights_len
        self.workspace_bytes = self.engine.workspace_bytes
        if self.weights_len != sum(
            math.prod(shape) for shape in config.shapes().values()
        ):
            raise RuntimeError("Native weight layout does not match Config")
        if type(self.workspace_bytes) is not int or self.workspace_bytes < 0:
            raise RuntimeError("Invalid native workspace size")

    def losses(self, weights, indices):
        indices = problem_indices(indices, len(self.tokens))
        losses = np.asarray(
            self.engine.losses(
                weights,
                [self.tokens[i] for i in indices],
                [self.masks[i] for i in indices],
            )
        )
        if losses.shape != (len(indices),) or losses.dtype != np.float32:
            raise RuntimeError(
                "Native evaluator must return one FP32 mean loss per problem"
            )
        return losses.astype(np.float64)
