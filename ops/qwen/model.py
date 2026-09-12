"""Independent dense Qwen2 forward pass used as the control implementation."""

from __future__ import annotations

from collections.abc import Mapping

from .config import Config


def _requiretorch():
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as error:
        raise RuntimeError(
            "Qwen evaluation requires torch; install ops/qwen/requirements.txt"
        ) from error
    return torch, F


def _linear(x, weight, bias=None):
    result = x @ weight.float().transpose(-1, -2)
    return result + bias.float() if bias is not None else result


def _rmsnorm(x, weight, epsilon):
    torch, _ = _requiretorch()
    dtype = x.dtype
    x = x.float()
    return (
        (x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + epsilon))
        .mul(weight.float())
        .to(dtype)
    )


def _rotatehalf(x):
    torch, _ = _requiretorch()
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _rotary(x, positions, theta):
    torch, _ = _requiretorch()
    half = x.shape[-1] // 2
    inverse = 1.0 / theta ** (torch.arange(0, half, device=x.device).float() / half)
    angles = positions.float()[:, None] * inverse[None, :]
    cos = torch.cat((angles.cos(), angles.cos()), dim=-1)[None, :, None, :]
    sin = torch.cat((angles.sin(), angles.sin()), dim=-1)[None, :, None, :]
    return x * cos + _rotatehalf(x) * sin


def _attention(x, params: Mapping, prefix: str, config: Config):
    torch, F = _requiretorch()
    batch, sequence, _ = x.shape
    q = _linear(
        x,
        params[prefix + "self_attn.q_proj.weight"],
        params[prefix + "self_attn.q_proj.bias"],
    )
    k = _linear(
        x,
        params[prefix + "self_attn.k_proj.weight"],
        params[prefix + "self_attn.k_proj.bias"],
    )
    v = _linear(
        x,
        params[prefix + "self_attn.v_proj.weight"],
        params[prefix + "self_attn.v_proj.bias"],
    )
    q = q.reshape(batch, sequence, config.num_attention_heads, config.head_dim)
    k = k.reshape(batch, sequence, config.num_key_value_heads, config.head_dim)
    v = v.reshape(batch, sequence, config.num_key_value_heads, config.head_dim)
    positions = torch.arange(sequence, device=x.device)
    q = _rotary(q, positions, config.rope_theta).transpose(1, 2)
    k = _rotary(k, positions, config.rope_theta).transpose(1, 2)
    k = k.repeat_interleave(config.kv_repeat, dim=1)
    v = v.transpose(1, 2).repeat_interleave(config.kv_repeat, dim=1)
    scores = torch.matmul(q, k.transpose(-1, -2)) / config.head_dim**0.5
    causal = torch.triu(
        torch.ones(sequence, sequence, dtype=torch.bool, device=x.device), diagonal=1
    )
    scores = scores.float().masked_fill(causal, torch.finfo(torch.float32).min)
    probabilities = F.softmax(scores, dim=-1)
    output = (
        torch.matmul(probabilities, v.float())
        .transpose(1, 2)
        .reshape(batch, sequence, config.hidden_size)
    )
    return _linear(output, params[prefix + "self_attn.o_proj.weight"])


def forward(params: Mapping, tokens, config: Config | None = None):
    """Return full-sequence logits for a CPU or accelerator token batch."""
    torch, F = _requiretorch()
    config = Config() if config is None else config
    if tokens.ndim != 2 or tokens.dtype not in (torch.int32, torch.int64):
        raise ValueError("tokens must be a rank-two int32 or int64 tensor")
    if (
        tokens.shape[0] < 1
        or not 1 <= tokens.shape[1] <= config.max_position_embeddings
    ):
        raise ValueError("empty input or sequence exceeds Qwen context")
    if bool(((tokens < 0) | (tokens >= config.vocab_size)).any()):
        raise ValueError("token ID is outside the Qwen vocabulary")
    x = params["model.embed_tokens.weight"][tokens].float()
    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}."
        norm = _rmsnorm(
            x, params[prefix + "input_layernorm.weight"], config.rms_norm_eps
        )
        x = x + _attention(norm, params, prefix, config)
        norm = _rmsnorm(
            x, params[prefix + "post_attention_layernorm.weight"], config.rms_norm_eps
        )
        gate = _linear(norm, params[prefix + "mlp.gate_proj.weight"])
        up = _linear(norm, params[prefix + "mlp.up_proj.weight"])
        x = x + _linear(
            F.silu(gate) * up,
            params[prefix + "mlp.down_proj.weight"],
        )
    hidden = _rmsnorm(x, params["model.norm.weight"], config.rms_norm_eps)
    embedding = params["model.embed_tokens.weight"]
    output_weight = params.get("lm_head.weight", embedding)
    return _linear(hidden, output_weight)


def token_losses(logits, tokens):
    """Return unreduced next-token losses, shaped like the target sequence."""
    _, F = _requiretorch()
    if logits.ndim != 3 or tokens.ndim != 2 or logits.shape[:2] != tokens.shape:
        raise ValueError("logits and tokens must have aligned batch and sequence axes")
    return F.cross_entropy(
        logits[:, :-1].float().transpose(1, 2), tokens[:, 1:], reduction="none"
    )


def loss(logits, tokens, mask=None):
    """Reduce next-token loss, optionally over a bool mask on token positions."""
    torch, _ = _requiretorch()
    losses = token_losses(logits, tokens)
    if mask is None:
        return losses.mean()
    if mask.shape != tokens.shape or mask.dtype is not torch.bool:
        raise ValueError("mask must be a bool tensor aligned with tokens")
    selected = losses[mask[:, 1:]]
    if selected.numel() == 0:
        raise ValueError("mask selects no target tokens")
    return selected.mean()


def greedy_generate(params: Mapping, prompt, max_new_tokens: int, config=None):
    """Append greedy tokens without sampling or hidden post-processing."""
    torch, _ = _requiretorch()
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens:
        raise ValueError("max_new_tokens must be a positive integer")
    config = Config() if config is None else config
    tokens = prompt.clone()
    if (
        tokens.ndim != 2
        or tokens.shape[1] < 1
        or tokens.shape[1] > config.max_position_embeddings
    ):
        raise ValueError("prompt must fit in a nonempty rank-two token tensor")
    for _ in range(max_new_tokens):
        if tokens.shape[1] >= config.max_position_embeddings:
            break
        logits = forward(params, tokens, config)
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        tokens = torch.cat((tokens, next_token), dim=1)
        if bool((next_token == config.eos_token_id).all()):
            break
    return tokens
