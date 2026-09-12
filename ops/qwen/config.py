"""Validated architecture metadata for the pinned Qwen2.5-Coder checkpoint."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B"
# Pin the model snapshot used by the dense control. Do not silently follow main.
REVISION = "dba20987fcbfb46dcca5a257e10b5ab39c9ec7ce"


@dataclass(frozen=True)
class Config:
    vocab_size: int = 151936
    hidden_size: int = 1536
    intermediate_size: int = 8960
    num_hidden_layers: int = 28
    num_attention_heads: int = 12
    num_key_value_heads: int = 2
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    bos_token_id: int = 151643
    eos_token_id: int = 151643
    tie_word_embeddings: bool = True

    def __post_init__(self) -> None:
        integer_fields = (
            self.vocab_size,
            self.hidden_size,
            self.intermediate_size,
            self.num_hidden_layers,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.max_position_embeddings,
            self.bos_token_id,
            self.eos_token_id,
        )
        if any(type(value) is not int or value <= 0 for value in integer_fields):
            raise ValueError("Qwen dimensions and token IDs must be positive integers")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("Attention heads must be divisible by KV heads")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("Hidden size must be divisible by attention heads")
        head_dim = self.hidden_size // self.num_attention_heads
        if head_dim % 2 or not 0 < self.rms_norm_eps < 1 or self.rope_theta <= 1:
            raise ValueError("Invalid Qwen normalization or RoPE configuration")
        if self.bos_token_id >= self.vocab_size or self.eos_token_id >= self.vocab_size:
            raise ValueError("Special token ID exceeds the vocabulary")

    @classmethod
    def from_json(cls, data: dict) -> "Config":
        if not isinstance(data, dict) or data.get("model_type") != "qwen2":
            raise ValueError("Checkpoint is not a Qwen2 model")
        required = {
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "max_position_embeddings",
            "rms_norm_eps",
            "rope_theta",
            "bos_token_id",
            "eos_token_id",
            "tie_word_embeddings",
        }
        if not required.issubset(data):
            raise ValueError("Qwen config is missing required architecture fields")
        config = cls(**{name: data[name] for name in required})
        if config != cls():
            raise ValueError("Checkpoint architecture is not Qwen2.5-Coder-1.5B")
        return config

    @classmethod
    def from_file(cls, path: Path) -> "Config":
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Could not read Qwen config: {path}") from error
        return cls.from_json(data)

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def kv_repeat(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def context(self) -> int:
        """Compatibility name used by the shared token-objective validator."""
        return self.max_position_embeddings

    @property
    def vocab(self) -> int:
        """Compatibility name used by the shared token-objective validator."""
        return self.vocab_size
