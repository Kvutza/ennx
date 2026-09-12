"""Architecture recorded in FLAME-MoE-290M-1.3B / iter_0005473/common.pt."""

from dataclasses import dataclass

MODEL_ID = "CMU-FLAME/FLAME-MoE-290M-1.3B"
REVISION = "04bded20e1eafa97c5c84cee9522798a2e83606f"
ITERATION = "iter_0005473"
REFERENCE = "cbaf684c5d03997e0fdd5347c5e2d371c381a3d8"


@dataclass(frozen=True)
class Config:
    layers: int = 9
    width: int = 1024
    heads: int = 16
    vocab: int = 50304
    dense_width: int = 5472
    expert_width: int = 704
    shared_width: int = 1408
    experts: int = 64
    top_k: int = 6
    context: int = 2048
    epsilon: float = 1e-6
    rope_base: float = 10000.0

    def __post_init__(self):
        sizes = (
            self.layers,
            self.width,
            self.heads,
            self.vocab,
            self.dense_width,
            self.expert_width,
            self.shared_width,
            self.experts,
            self.top_k,
            self.context,
        )
        if any(type(x) is not int or x <= 0 for x in sizes):
            raise ValueError("Model dimensions must be positive integers")
        if self.width % self.heads or (self.width // self.heads) % 2:
            raise ValueError(
                "Attention heads must divide width and have even dimension"
            )
        if (
            self.top_k > self.experts
            or not 0 < self.epsilon < 1
            or not 1 < self.rope_base < float("inf")
        ):
            raise ValueError("Invalid routing, normalization, or RoPE configuration")

    def shapes(self) -> dict[str, tuple[int, ...]]:
        h = self.width
        shapes = {
            "embedding.word_embeddings.weight": (self.vocab, h),
            "output_layer.weight": (self.vocab, h),
            "decoder.final_layernorm.weight": (h,),
        }
        for layer in range(self.layers):
            prefix = f"decoder.layers.{layer}."
            shapes[prefix + "self_attention.linear_qkv.weight"] = (3 * h, h)
            shapes[prefix + "self_attention.linear_qkv.layer_norm_weight"] = (h,)
            shapes[prefix + "self_attention.linear_proj.weight"] = (h, h)
            if layer == 0:
                shapes[prefix + "mlp.linear_fc1.layer_norm_weight"] = (h,)
                shapes[prefix + "mlp.linear_fc1.weight"] = (2 * self.dense_width, h)
                shapes[prefix + "mlp.linear_fc2.weight"] = (h, self.dense_width)
            else:
                shapes[prefix + "pre_mlp_layernorm.weight"] = (h,)
                shapes[prefix + "mlp.router.weight"] = (self.experts, h)
                shapes[prefix + "mlp.shared_experts.linear_fc1.weight"] = (
                    2 * self.shared_width,
                    h,
                )
                shapes[prefix + "mlp.shared_experts.linear_fc2.weight"] = (
                    h,
                    self.shared_width,
                )
                shapes[prefix + "mlp.experts.experts.linear_fc1.weight"] = (
                    self.experts,
                    2 * self.expert_width,
                    h,
                )
                shapes[prefix + "mlp.experts.experts.linear_fc2.weight"] = (
                    self.experts,
                    h,
                    self.expert_width,
                )
        return shapes
