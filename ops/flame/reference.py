"""Independent, unfused PyTorch comparator for the pinned Megatron equations.

This does not run Megatron/TransformerEngine, and is not a substitute for testing
against that runtime. It provides a second implementation and per-block taps.
"""

import math

import torch
import torch.nn.functional as F


@torch.inference_mode()
def forward(params, tokens, config, *, capture=False):
    def norm(x, w):
        return F.rms_norm(x, (config.width,), w.float(), config.epsilon)

    def dense(x, w1, w2):
        gate, up = F.linear(x, w1.float()).chunk(2, dim=-1)
        return F.linear(F.silu(gate) * up, w2.float())

    x = F.embedding(tokens, params["embedding.word_embeddings.weight"].float())
    batch, sequence, _ = x.shape
    dim = config.width // config.heads
    positions = torch.arange(sequence, device=x.device).float()
    frequencies = config.rope_base ** (
        -torch.arange(0, dim, 2, device=x.device).float() / dim
    )
    angles = torch.outer(positions, frequencies)
    cosine = angles.cos()[None, :, None, :]
    sine = angles.sin()[None, :, None, :]
    traces = {}
    for layer in range(config.layers):
        prefix = f"decoder.layers.{layer}."
        normalized = norm(
            x, params[prefix + "self_attention.linear_qkv.layer_norm_weight"]
        )
        packed = F.linear(
            normalized, params[prefix + "self_attention.linear_qkv.weight"].float()
        )
        packed = packed.reshape(batch, sequence, config.heads, 3 * dim)
        q, k, v = packed.split(dim, dim=-1)
        rotated = []
        for item in (q, k):
            left, right = item.chunk(2, dim=-1)
            rotated.append(
                torch.cat(
                    (left * cosine - right * sine, right * cosine + left * sine), dim=-1
                )
            )
        q, k = (item.transpose(1, 2) for item in rotated)
        scores = q @ k.transpose(-1, -2) / math.sqrt(dim)
        mask = torch.ones(sequence, sequence, dtype=torch.bool, device=x.device).triu(1)
        scores.masked_fill_(mask, -torch.inf)
        attended = (
            (scores.softmax(-1) @ v.transpose(1, 2)).transpose(1, 2).reshape_as(x)
        )
        attended = F.linear(
            attended, params[prefix + "self_attention.linear_proj.weight"].float()
        )
        x = x + attended
        trace = {"attention": attended}
        if layer == 0:
            normalized = norm(x, params[prefix + "mlp.linear_fc1.layer_norm_weight"])
            update = dense(
                normalized,
                params[prefix + "mlp.linear_fc1.weight"],
                params[prefix + "mlp.linear_fc2.weight"],
            )
        else:
            normalized = norm(x, params[prefix + "pre_mlp_layernorm.weight"])
            logits = F.linear(normalized, params[prefix + "mlp.router.weight"].float())
            scores = logits.softmax(-1)
            ids = scores.argsort(dim=-1, descending=True, stable=True)[
                ..., : config.top_k
            ]
            probs = scores.gather(-1, ids)
            shared = dense(
                normalized,
                params[prefix + "mlp.shared_experts.linear_fc1.weight"],
                params[prefix + "mlp.shared_experts.linear_fc2.weight"],
            )
            # Dispatch by expert, independently of the JAX token-by-token path.
            flat = normalized.reshape(-1, config.width)
            routed = torch.zeros_like(flat)
            for expert in range(config.experts):
                rows, slots = torch.where(ids.reshape(-1, config.top_k) == expert)
                if rows.numel():
                    output = dense(
                        flat[rows],
                        params[prefix + "mlp.experts.experts.linear_fc1.weight"][
                            expert
                        ],
                        params[prefix + "mlp.experts.experts.linear_fc2.weight"][
                            expert
                        ],
                    )
                    routed.index_add_(
                        0,
                        rows,
                        output * probs.reshape(-1, config.top_k)[rows, slots, None],
                    )
            update = shared + routed.reshape_as(x)
            trace.update(router_logits=logits, routing_weights=probs, experts=ids)
        x = x + update
        if capture:
            traces[str(layer)] = dict(trace, output=x)
    logits = F.linear(
        norm(x, params["decoder.final_layernorm.weight"]),
        params["output_layer.weight"].float(),
    )
    return (logits, traces) if capture else logits
