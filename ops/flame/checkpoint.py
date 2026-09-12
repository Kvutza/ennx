"""Extract only model tensors from a pinned public distributed checkpoint."""

from __future__ import annotations

import hashlib
import io
import json
import math
import pickle
import re
from dataclasses import asdict
from pathlib import Path, PosixPath

import click
import requests
import torch
from requests.adapters import HTTPAdapter
from safetensors.torch import save_file
from torch.distributed.checkpoint import metadata, planner
from torch.distributed.checkpoint.filesystem import _StorageInfo
from urllib3.util.retry import Retry

from .config import ITERATION, MODEL_ID, REFERENCE, REVISION, Config


class _SavePlan:
    """Inert placeholder for unused Megatron planning metadata."""


class MetadataReader(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {
            ("torch", "bfloat16"): torch.bfloat16,
            ("torch", "float32"): torch.float32,
            ("torch", "Size"): torch.Size,
            ("torch.serialization", "_get_layout"): torch.serialization._get_layout,
            ("pathlib", "PosixPath"): PosixPath,
            ("torch.distributed.checkpoint.filesystem", "_StorageInfo"): _StorageInfo,
            (
                "megatron.core.dist_checkpointing.strategies.torch",
                "MCoreSavePlan",
            ): _SavePlan,
        }
        for cls in (
            "Metadata",
            "TensorStorageMetadata",
            "TensorProperties",
            "ChunkStorageMetadata",
            "BytesStorageMetadata",
            "MetadataIndex",
            "_MEM_FORMAT_ENCODING",
            "StorageMeta",
        ):
            allowed[(metadata.__name__, cls)] = getattr(metadata, cls)
        for cls in ("SavePlan", "WriteItem", "WriteItemType", "TensorWriteData"):
            allowed[(planner.__name__, cls)] = getattr(planner, cls)
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(
                f"Unsupported checkpoint metadata class: {module}.{name}"
            )
        return allowed[(module, name)]


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class Source:
    def __init__(self):
        self.session = requests.Session()
        retry = Retry(
            total=3, backoff_factor=1, status_forcelist=(429, 500, 502, 503, 504)
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retry))

    def close(self):
        self.session.close()

    def read(
        self, name: str, offset: int | None = None, length: int | None = None
    ) -> bytes:
        if name != ".metadata" and not re.fullmatch(r"__\d+_\d+\.distcp", name):
            raise ValueError("Invalid checkpoint shard name")
        headers = {"Accept-Encoding": "identity"}
        if offset is not None:
            if offset < 0 or length is None or length <= 0:
                raise ValueError("Invalid byte range")
            headers["Range"] = f"bytes={offset}-{offset + length - 1}"
        limit = length if offset is not None else 8 * 1024 * 1024
        url = f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{ITERATION}/{name}"
        with self.session.get(
            url, headers=headers, stream=True, timeout=(30, 120)
        ) as response:
            response.raise_for_status()
            if offset is not None:
                expected = f"bytes {offset}-{offset + length - 1}/"
                if response.status_code != 206 or not response.headers.get(
                    "Content-Range", ""
                ).startswith(expected):
                    raise ValueError(
                        "Server did not honor the exact byte range; refusing a full shard download"
                    )
            body = response.raw.read(limit + 1)
            if len(body) > limit or (offset is not None and len(body) != length):
                raise ValueError("Checkpoint response has an invalid length")
            return body


def model_metadata(data: bytes, config: Config):
    checkpoint = MetadataReader(io.BytesIO(data)).load()
    selected = {
        name: item
        for name, item in checkpoint.state_dict_metadata.items()
        if name.startswith(("embedding.", "decoder.", "output_layer."))
        and isinstance(item, metadata.TensorStorageMetadata)
    }
    expected = config.shapes()
    if selected.keys() != expected.keys():
        raise ValueError(
            f"Checkpoint parameter names differ: {selected.keys() ^ expected.keys()}"
        )
    for name, item in selected.items():
        if (
            tuple(item.size) != expected[name]
            or item.properties.dtype != torch.bfloat16
        ):
            raise ValueError(f"Unexpected shape or dtype for {name}")
    return checkpoint, selected


def validate_chunks(shape, chunks):
    volume = 0
    boxes = []
    for chunk in chunks:
        start, size = tuple(chunk.offsets), tuple(chunk.sizes)
        if len(start) != len(shape) or len(size) != len(shape):
            raise ValueError("Chunk rank does not match tensor")
        if any(a < 0 or b <= 0 or a + b > n for a, b, n in zip(start, size, shape)):
            raise ValueError("Chunk exceeds tensor bounds")
        end = tuple(a + b for a, b in zip(start, size))
        if any(
            all(a < d and c < b for a, b, c, d in zip(start, end, lo, hi))
            for lo, hi in boxes
        ):
            raise ValueError("Checkpoint chunks overlap")
        boxes.append((start, end))
        volume += math.prod(size)
    if volume != math.prod(shape):
        raise ValueError("Checkpoint chunks do not cover the tensor")


def read_tensor(checkpoint, name, item, source):
    validate_chunks(tuple(item.size), item.chunks)
    tensor = torch.empty(tuple(item.size), dtype=item.properties.dtype)
    for chunk in item.chunks:
        index = metadata.MetadataIndex(name, chunk.offsets)
        storage = checkpoint.storage_data[index]
        if getattr(storage, "transform_descriptors", None):
            raise ValueError("Transformed checkpoint payloads are unsupported")
        body = source.read(storage.relative_path, storage.offset, storage.length)
        value = torch.load(io.BytesIO(body), weights_only=True, map_location="cpu")
        if not isinstance(value, torch.Tensor) or value.dtype != tensor.dtype:
            raise ValueError(f"Unexpected payload for {name}")
        if value.numel() != math.prod(chunk.sizes) or not torch.isfinite(value).all():
            raise ValueError(f"Invalid tensor payload for {name}")
        slices = tuple(slice(a, a + b) for a, b in zip(chunk.offsets, chunk.sizes))
        tensor[slices] = value.reshape(tuple(chunk.sizes))
    return tensor


def convert(output: Path, names: list[str] | None = None):
    config = Config()
    source = Source()
    try:
        data = source.read(".metadata")
        checkpoint, selected = model_metadata(data, config)
        wanted = sorted(selected if names is None else set(names))
        if not wanted or any(name not in selected for name in wanted):
            raise ValueError("Select at least one known model tensor")
        output.mkdir(parents=True, exist_ok=False)
        manifest = {
            "model_id": MODEL_ID,
            "revision": REVISION,
            "iteration": ITERATION,
            "megatron_revision": REFERENCE,
            "config": asdict(config),
            "metadata_sha256": hashlib.sha256(data).hexdigest(),
            "complete": False,
            "tensors": {},
        }
        for number, name in enumerate(wanted):
            value = read_tensor(checkpoint, name, selected[name], source)
            filename = f"{number:03d}.safetensors"
            path = output / filename
            save_file({name: value}, str(path))
            manifest["tensors"][name] = {"file": filename, "sha256": digest(path)}
            print(f"{number + 1}/{len(wanted)} {name} {tuple(value.shape)}", flush=True)
            del value
        manifest["complete"] = set(wanted) == set(selected)
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        return manifest
    finally:
        source.close()


@click.command(help=__doc__, context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--output", type=click.Path(file_okay=False, path_type=Path), required=True
)
@click.option(
    "--tensor", multiple=True, help="Extract named tensors only (partial checkpoint)"
)
def main(output: Path, tensor: tuple[str, ...]):
    convert(output, list(tensor) if tensor else None)


if __name__ == "__main__":
    main()
