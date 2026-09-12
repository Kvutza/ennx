"""Download and validate the dense Qwen2.5-Coder control checkpoint."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.request
from pathlib import Path

from .config import MODEL_ID, REVISION, Config

MAX_ARTIFACT_BYTES = 4 * 1024**3
SMALL_FILES = (
    "config.json",
    "merges.txt",
    "tokenizer_config.json",
    "vocab.json",
)


def _requiretorch():
    try:
        import torch
        from safetensors.torch import load_file
    except ImportError as error:
        raise RuntimeError(
            "Qwen checkpoint loading requires torch and safetensors; "
            "install ops/qwen/requirements.txt"
        ) from error
    return torch, load_file


def expected_shapes(config: Config) -> dict[str, tuple[int, ...]]:
    shapes = {
        "model.embed_tokens.weight": (config.vocab_size, config.hidden_size),
        "model.norm.weight": (config.hidden_size,),
    }
    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}."
        shapes.update(
            {
                prefix + "input_layernorm.weight": (config.hidden_size,),
                prefix + "post_attention_layernorm.weight": (config.hidden_size,),
                prefix + "self_attn.q_proj.weight": (
                    config.num_attention_heads * config.head_dim,
                    config.hidden_size,
                ),
                prefix + "self_attn.q_proj.bias": (
                    config.num_attention_heads * config.head_dim,
                ),
                prefix + "self_attn.k_proj.weight": (
                    config.num_key_value_heads * config.head_dim,
                    config.hidden_size,
                ),
                prefix + "self_attn.k_proj.bias": (
                    config.num_key_value_heads * config.head_dim,
                ),
                prefix + "self_attn.v_proj.weight": (
                    config.num_key_value_heads * config.head_dim,
                    config.hidden_size,
                ),
                prefix + "self_attn.v_proj.bias": (
                    config.num_key_value_heads * config.head_dim,
                ),
                prefix + "self_attn.o_proj.weight": (
                    config.hidden_size,
                    config.hidden_size,
                ),
                prefix + "mlp.gate_proj.weight": (
                    config.intermediate_size,
                    config.hidden_size,
                ),
                prefix + "mlp.up_proj.weight": (
                    config.intermediate_size,
                    config.hidden_size,
                ),
                prefix + "mlp.down_proj.weight": (
                    config.hidden_size,
                    config.intermediate_size,
                ),
            }
        )
    return shapes


def load_checkpoint(directory: Path):
    """Load a complete CPU BF16 state dict and reject architecture drift."""
    torch, load_file = _requiretorch()
    directory = Path(directory)
    config = Config.from_file(directory / "config.json")
    files = sorted(directory.glob("*.safetensors"))
    if not files:
        raise ValueError(f"No safetensors weights found in {directory}")

    params = {}
    for path in files:
        loaded = load_file(str(path), device="cpu")
        for name, value in loaded.items():
            if name in params:
                raise ValueError(f"Duplicate tensor in checkpoint: {name}")
            params[name] = value

    expected = expected_shapes(config)
    allowed = set(expected) | {"lm_head.weight"}
    unexpected = set(params) - allowed
    missing = set(expected) - set(params)
    if unexpected or missing:
        raise ValueError(
            f"Checkpoint tensor names differ; missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    for name, shape in expected.items():
        value = params[name]
        if value.dtype != torch.bfloat16 or tuple(value.shape) != shape:
            raise ValueError(f"Unexpected dtype or shape for {name}")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"Non-finite values in {name}")
    if "lm_head.weight" in params:
        if params["lm_head.weight"].dtype != torch.bfloat16:
            raise ValueError("Unexpected dtype for lm_head.weight")
        if not torch.equal(
            params["lm_head.weight"], params["model.embed_tokens.weight"]
        ):
            raise ValueError("Tied embedding and lm_head weights differ")
        # The layout and the native evaluator use the tied embedding once.
        del params["lm_head.weight"]
    return config, params


def _url(name: str) -> str:
    if name not in SMALL_FILES and name != "model.safetensors":
        raise ValueError(f"Unsupported Qwen artifact: {name}")
    return f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{name}"


def _download(name: str, output: Path, *, force: bool) -> str:
    target = output / name
    if target.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite {target}; pass --force")
    output.mkdir(parents=True, exist_ok=True)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(
        _url(name), headers={"Accept-Encoding": "identity"}
    )
    with opener.open(request, timeout=120) as response:
        length = response.headers.get("Content-Length")
        if length is not None and int(length) > MAX_ARTIFACT_BYTES:
            raise ValueError(f"Refusing oversized Qwen artifact: {name}")
        fd, temporary = tempfile.mkstemp(prefix=f".{name}.", dir=output)
        size = 0
        try:
            with os.fdopen(fd, "wb") as stream:
                while chunk := response.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_ARTIFACT_BYTES:
                        raise ValueError(f"Refusing oversized Qwen artifact: {name}")
                    stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
    return hashlib.sha256(target.read_bytes()).hexdigest()


def fetch_model(output: Path, *, weights: bool, force: bool = False) -> dict:
    """Download pinned metadata and optionally the explicit 3.1 GB weight file."""
    output = Path(output)
    manifest_path = output / "manifest.json"
    existing = None
    if manifest_path.exists() and not force:
        try:
            existing = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid Qwen manifest: {manifest_path}") from error
        if (existing.get("model_id"), existing.get("revision")) != (
            MODEL_ID,
            REVISION,
        ):
            raise ValueError("Existing Qwen manifest has different model provenance")
        if existing.get("weights_downloaded") or not weights:
            return existing
        names = ["model.safetensors"]
        records = dict(existing.get("files", {}))
    else:
        names = list(SMALL_FILES)
        if weights:
            names.append("model.safetensors")
        records = {}
    for name in names:
        digest = _download(name, output, force=force)
        records[name] = {
            "sha256": digest,
            "bytes": (output / name).stat().st_size,
        }
    config = Config.from_file(output / "config.json")
    manifest = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "config": config.__dict__,
        "weights_downloaded": weights
        or bool(existing and existing.get("weights_downloaded")),
        "files": records,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
