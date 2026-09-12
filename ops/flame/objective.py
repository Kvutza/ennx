"""Fixed, solution-only teacher-forcing objectives for full-weight BO."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import jax

FORMAT = "ennx.solution_tokens.v1"


def problem_indices(indices, population):
    indices = np.asarray(indices)
    if (
        indices.ndim != 1
        or not np.issubdtype(indices.dtype, np.integer)
        or not indices.size
        or indices.min() < 0
        or indices.max() >= population
        or len(np.unique(indices)) != len(indices)
    ):
        raise ValueError("Expected distinct, in-range problem indices")
    return indices


@dataclass(frozen=True)
class SolutionObjective:
    tokens: np.ndarray
    mask: np.ndarray
    metadata: dict

    @classmethod
    def parse(cls, document, config):
        if not isinstance(document, dict) or document.get("format") != FORMAT:
            raise ValueError(f"Expected objective format {FORMAT}")
        examples = document.get("examples")
        if not isinstance(examples, list) or not examples:
            raise ValueError("Objective requires a nonempty examples list")
        if not isinstance(document.get("provenance"), dict):
            raise TypeError("Objective requires provenance metadata")
        seen = set()
        lengths, counts = [], []
        for example in examples:
            if not isinstance(example, dict):
                raise TypeError("Each objective example must be an object")
            key = example.get("id")
            if not isinstance(key, str) or not key or key in seen:
                raise ValueError("Objective example IDs must be nonempty and unique")
            seen.add(key)
            tokens, mask = example.get("tokens"), example.get("loss_mask")
            if (
                not isinstance(tokens, list)
                or not 2 <= len(tokens) <= config.context
                or any(type(t) is not int or not 0 <= t < config.vocab for t in tokens)
            ):
                raise ValueError(f"Invalid token sequence for {key}")
            if (
                not isinstance(mask, list)
                or len(mask) != len(tokens)
                or any(type(value) is not bool for value in mask)
                or mask[0]
                or not mask[-1]
                or any(mask[i] and not mask[i + 1] for i in range(len(mask) - 1))
            ):
                raise ValueError(
                    f"Expected an unscored prompt then a scored solution for {key}"
                )
            lengths.append(len(tokens))
            counts.append(sum(mask))
        padded = np.zeros((len(examples), max(lengths)), dtype=np.int32)
        mask = np.zeros(padded.shape, dtype=np.bool_)
        for row, (example, length) in enumerate(zip(examples, lengths, strict=True)):
            padded[row, :length] = example["tokens"]
            mask[row, :length] = example["loss_mask"]
        serialized = json.dumps(document, sort_keys=True, allow_nan=False).encode()
        metadata = {
            "kind": "solution_token_cross_entropy",
            "normalization": "total_scored_tokens",
            "sha256": hashlib.sha256(serialized).hexdigest(),
            "examples": len(examples),
            "example_ids": [example["id"] for example in examples],
            "input_tokens": sum(lengths),
            "scored_tokens": sum(counts),
            "sequence_lengths": lengths,
            "solution_tokens_per_example": counts,
            "padded_sequence_length": max(lengths),
            "microbatch_size": 1,
            "device_input_bytes": padded.nbytes + mask[:, 1:].nbytes,
            "provenance": document["provenance"],
        }
        return cls(padded, mask, metadata)

    def evaluator(self, layout, config):
        import jax
        import jax.numpy as jnp

        from . import model

        @jax.jit
        def evaluate(flat, tokens, mask, index):
            tokens = jax.lax.dynamic_slice_in_dim(tokens, index, 1, axis=0)
            mask = jax.lax.dynamic_slice_in_dim(mask, index, 1, axis=0)
            params = layout.unflatten(flat.reshape((layout.size,)))
            losses = model.token_loss(model.forward(params, tokens, config), tokens)
            return jnp.sum(jnp.where(mask, losses, 0.0))

        return SolutionEvaluator(
            evaluate,
            jnp.asarray(self.tokens),
            jnp.asarray(self.mask[:, 1:]),
            self.metadata["scored_tokens"],
            tuple(self.metadata["solution_tokens_per_example"]),
        )


@dataclass(frozen=True)
class SolutionEvaluator:
    kernel: object
    tokens: jax.Array
    masks: jax.Array
    scored_tokens: int
    solution_counts: tuple[int, ...]

    def __call__(self, flat):
        import jax.numpy as jnp

        # An outer XLA loop materializes weight slices on T4. Keep only the
        # forward compiled, synchronizing each scalar before the next example.
        total = sum(
            float(
                self.kernel(
                    flat, self.tokens, self.masks, np.int32(i)
                ).block_until_ready()
            )
            for i in range(len(self.tokens))
        )
        return jnp.asarray(-total / self.scored_tokens, dtype=jnp.float32)

    def losses(self, flat, indices):
        indices = problem_indices(indices, len(self.tokens))
        return np.asarray(
            [
                float(
                    self.kernel(
                        flat, self.tokens, self.masks, np.int32(i)
                    ).block_until_ready()
                )
                / self.solution_counts[i]
                for i in indices
            ],
            dtype=np.float64,
        )

    def compile(self, signature):
        kernel = self.kernel.lower(
            signature, self.tokens, self.masks, np.int32(0)
        ).compile()
        return replace(self, kernel=kernel)

    def memory_analysis(self):
        return self.kernel.memory_analysis()
