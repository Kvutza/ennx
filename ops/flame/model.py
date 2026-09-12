"""Readable FP32 JAX evaluation of the released FLAME checkpoint layout.

This is a correctness-first forward path, not an optimized MoE training engine.
Inputs are unpadded token sequences; the first block is dense, the rest are MoE.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
from safetensors.flax import load_file

from .config import ITERATION, MODEL_ID, REVISION, Config


def linear(x, weight):
    return jnp.matmul(
        x.astype(jnp.float32),
        weight.astype(jnp.float32).T,
        precision=jax.lax.Precision.HIGHEST,
    )


def rms_norm(x, weight, epsilon):
    x = x.astype(jnp.float32)
    return x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + epsilon) * weight


def rotary(x, base):
    size = x.shape[-1]
    inverse = base ** (-jnp.arange(0, size, 2, dtype=jnp.float32) / size)
    angle = jnp.arange(x.shape[1], dtype=jnp.float32)[:, None] * inverse[None, :]
    angle = jnp.concatenate((angle, angle), axis=-1)[None, :, None, :]
    left, right = jnp.split(x, 2, axis=-1)
    rotated = jnp.concatenate((-right, left), axis=-1)
    return x * jnp.cos(angle) + rotated * jnp.sin(angle)


def attention(x, qkv_weight, output_weight, config):
    batch, sequence, _ = x.shape
    width = config.width // config.heads
    # Megatron packs Q/K/V within each head, not three contiguous head groups.
    qkv = linear(x, qkv_weight).reshape(batch, sequence, config.heads, 3, width)
    q, k, v = (qkv[:, :, :, index, :] for index in range(3))
    q, k = rotary(q, config.rope_base), rotary(k, config.rope_base)
    scores = (
        jnp.einsum("bthd,bshd->bhts", q, k, precision=jax.lax.Precision.HIGHEST)
        / width**0.5
    )
    causal = jnp.arange(sequence)[:, None] >= jnp.arange(sequence)[None, :]
    probabilities = jax.nn.softmax(jnp.where(causal, scores, -jnp.inf), axis=-1)
    output = jnp.einsum(
        "bhts,bshd->bthd", probabilities, v, precision=jax.lax.Precision.HIGHEST
    )
    return linear(output.reshape(batch, sequence, config.width), output_weight)


def mlp(x, first, second):
    gate, up = jnp.split(linear(x, first), 2, axis=-1)
    return linear(jax.nn.silu(gate) * up, second)


def route(x, router, top_k):
    logits = linear(x, router)
    # Pre-softmax routing retains mass from the full expert softmax. Do not
    # renormalize the selected probabilities to sum to one. JAX breaks exact
    # ties by lower expert index; the independent reference uses the same rule.
    probabilities, indices = jax.lax.top_k(jax.nn.softmax(logits, axis=-1), top_k)
    return logits, probabilities, indices


def mixture(x, first, second, probabilities, indices):
    first, second = jnp.asarray(first), jnp.asarray(second)
    shape = x.shape
    flat = x.reshape(-1, shape[-1])
    weights = probabilities.reshape(flat.shape[0], -1)
    selected = indices.reshape(flat.shape[0], -1)

    def token(args):
        hidden, probs, experts = args
        fc1 = first[experts].astype(jnp.float32)
        fc2 = second[experts].astype(jnp.float32)
        projected = jnp.einsum(
            "eoh,h->eo", fc1, hidden, precision=jax.lax.Precision.HIGHEST
        )
        gate, up = jnp.split(projected, 2, axis=-1)
        output = jnp.einsum(
            "ehi,ei->eh",
            fc2,
            jax.nn.silu(gate) * up,
            precision=jax.lax.Precision.HIGHEST,
        )
        return jnp.sum(output * probs[:, None], axis=0)

    # Bound gathered expert-weight storage independently of sequence length.
    return jax.lax.map(token, (flat, weights, selected)).reshape(shape)


def forward(params, tokens, config=None, *, capture=False):
    config = Config() if config is None else config
    if tokens.ndim != 2 or not jnp.issubdtype(tokens.dtype, jnp.integer):
        raise ValueError("Tokens must be a rank-two integer array")
    if tokens.shape[0] < 1 or not 1 <= tokens.shape[1] <= config.context:
        raise ValueError("Empty input or sequence exceeds model context")
    x = jnp.asarray(params["embedding.word_embeddings.weight"])[tokens].astype(
        jnp.float32
    )
    traces = {}
    for layer in range(config.layers):
        prefix = f"decoder.layers.{layer}."
        norm = rms_norm(
            x,
            params[prefix + "self_attention.linear_qkv.layer_norm_weight"],
            config.epsilon,
        )
        attended = attention(
            norm,
            params[prefix + "self_attention.linear_qkv.weight"],
            params[prefix + "self_attention.linear_proj.weight"],
            config,
        )
        x = x + attended
        trace = {"attention": attended}
        if layer == 0:
            norm = rms_norm(
                x, params[prefix + "mlp.linear_fc1.layer_norm_weight"], config.epsilon
            )
            update = mlp(
                norm,
                params[prefix + "mlp.linear_fc1.weight"],
                params[prefix + "mlp.linear_fc2.weight"],
            )
        else:
            norm = rms_norm(
                x, params[prefix + "pre_mlp_layernorm.weight"], config.epsilon
            )
            logits, probs, indices = route(
                norm, params[prefix + "mlp.router.weight"], config.top_k
            )
            shared = mlp(
                norm,
                params[prefix + "mlp.shared_experts.linear_fc1.weight"],
                params[prefix + "mlp.shared_experts.linear_fc2.weight"],
            )
            routed = mixture(
                norm,
                params[prefix + "mlp.experts.experts.linear_fc1.weight"],
                params[prefix + "mlp.experts.experts.linear_fc2.weight"],
                probs,
                indices,
            )
            update = shared + routed
            trace.update(router_logits=logits, routing_weights=probs, experts=indices)
        x = x + update
        if capture:
            traces[str(layer)] = dict(trace, output=x)
    hidden = rms_norm(x, params["decoder.final_layernorm.weight"], config.epsilon)
    logits = linear(hidden, params["output_layer.weight"])
    valid = jnp.all((tokens >= 0) & (tokens < config.vocab))
    logits = jnp.where(valid, logits, jnp.nan)
    return (logits, traces) if capture else logits


def token_loss(logits, tokens):
    if (
        tokens.ndim != 2
        or not jnp.issubdtype(tokens.dtype, jnp.integer)
        or logits.ndim != 3
        or tokens.shape[0] < 1
        or tokens.shape[1] < 2
        or logits.shape[:2] != tokens.shape
    ):
        raise ValueError("Next-token loss requires at least two aligned tokens")
    predictions = logits[:, :-1, :].astype(jnp.float32)
    targets = tokens[:, 1:]
    selected = jnp.take_along_axis(predictions, targets[..., None], axis=-1)[..., 0]
    valid = jnp.all((tokens >= 0) & (tokens < logits.shape[-1]))
    return jnp.where(valid, jax.nn.logsumexp(predictions, axis=-1) - selected, jnp.nan)


def load_checkpoint(directory: Path):
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
    if config != Config() or manifest["tensors"].keys() != config.shapes().keys():
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
        values = load_file(str(path))
        if (
            set(values) != {name}
            or values[name].shape != config.shapes()[name]
            or values[name].dtype != jnp.bfloat16
        ):
            raise ValueError(f"Unexpected tensor layout: {name}")
        params[name] = values[name]
    return config, params


def parameter_tree(params):
    tree = {}
    for name, value in params.items():
        node = tree
        parts = name.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return tree
