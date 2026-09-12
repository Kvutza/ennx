"""Wheel-backed Qwen Metal evaluation and correlated BO setup."""

from __future__ import annotations

import math
import sys
from pathlib import Path

from .config import Config


def proposal_stats(proposals, total_elements: int) -> dict[str, object]:
    """Summarize the selected proposal's realized BF16 perturbation."""
    if total_elements < 1:
        raise ValueError("Metal proposal population must be nonempty")
    described = proposals.describe()
    if len(described) != 1:
        raise RuntimeError("Metal proposal description must contain one proposal")
    seed, acquisition_score, radius, changes = described[0]
    geometry = proposals.geometry()
    if len(geometry) != 1:
        raise RuntimeError("Metal proposal geometry must contain one proposal")
    candidate_index, reference_correlation = geometry[0]
    if type(candidate_index) is not int or not 0 <= candidate_index < 4:
        raise RuntimeError("Metal proposal geometry is invalid")
    expected_correlation = 0.75 if candidate_index < 2 else 0.0
    if reference_correlation != expected_correlation:
        raise RuntimeError("Metal proposal geometry is invalid")
    base_id = proposals.base_id()
    if type(base_id) is not int or base_id < 1:
        raise RuntimeError("Metal proposal base observation ID is invalid")
    changed_elements = sum(int(changed) for changed, _ in changes)
    pool = proposals.pool()
    if len(pool) != 4 or [candidate[0] for candidate in pool] != list(range(4)):
        raise RuntimeError("Metal proposal pool is invalid")
    radii, cosines, ref_cosines = proposals.pool_geometry()
    expected_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    if (
        len(radii) != 4
        or any(not math.isfinite(value) or value < 0.0 for value in radii)
        or len(cosines) != len(expected_pairs)
        or [(left, right) for left, right, _ in cosines] != expected_pairs
        or any(
            value is not None and (not math.isfinite(value) or abs(value) > 1.0)
            for _, _, value in cosines
        )
        or len(ref_cosines) != 4
        or any(
            value is not None and (not math.isfinite(value) or abs(value) > 1.0)
            for value in ref_cosines
        )
    ):
        raise RuntimeError("Metal proposal pool geometry is invalid")
    candidate_pool = []
    for index, pool_seed, pool_radius, correlation, distances in pool:
        expected = 0.75 if index < 2 else 0.0
        if correlation != expected:
            raise RuntimeError("Metal proposal pool geometry is invalid")
        candidate_pool.append(
            {
                "candidate_index": int(index),
                "seed": int(pool_seed),
                "radius": float(pool_radius),
                "realized_radius": float(radii[index]),
                "reference_correlation": float(correlation),
                "realized_reference_cosine": (
                    None if ref_cosines[index] is None else float(ref_cosines[index])
                ),
                "history_distances": [
                    {
                        "observation_id": int(observation_id),
                        "squared_distance": float(distance),
                    }
                    for observation_id, distance in distances
                ],
            }
        )
    selected_pool = candidate_pool[candidate_index]
    selected_distances = [
        {
            "observation_id": int(observation_id),
            "squared_distance": float(distance),
        }
        for observation_id, distance in proposals.history_dists()
    ]
    if (
        selected_pool["seed"] != int(seed)
        or selected_pool["radius"] != float(radius)
        or selected_pool["history_distances"] != selected_distances
    ):
        raise RuntimeError("Metal selected proposal disagrees with its pool")
    return {
        "base_observation_id": base_id,
        "seed": int(seed),
        "candidate_index": candidate_index,
        "reference_correlation": float(reference_correlation),
        "realized_reference_cosine": selected_pool["realized_reference_cosine"],
        "reference_normalization": "per_tensor_bf16_rms",
        "acquisition_score": float(acquisition_score),
        "radius": float(radius),
        "realized_radius": float(radii[candidate_index]),
        "changed_bf16_elements": changed_elements,
        "total_bf16_elements": total_elements,
        "changed_bf16_fraction": changed_elements / total_elements,
        "changed_blocks": sum(int(changed) > 0 for changed, _ in changes),
        "total_blocks": len(changes),
        "squared_delta": sum(float(squared) for _, squared in changes),
        "history_distances": selected_distances,
        "pairwise_cosines": [
            {
                "left": int(left),
                "right": int(right),
                "cosine": None if cosine is None else float(cosine),
            }
            for left, right, cosine in cosines
        ],
        "candidate_pool": candidate_pool,
    }


class Evaluator:
    """Own one resident checkpoint and expose proposal-compatible loss calls.

    ``losses`` accepts either ``MetalWeights`` or a live ``MetalProposals``
    object. This lets the BO loop reuse the checkpoint allocation for every
    candidate instead of creating one Python tensor per proposal.
    """

    def __init__(self, checkpoint: Path, *, max_tokens: int):
        if sys.platform != "darwin":
            raise RuntimeError("Qwen Metal evaluation requires macOS")
        from ennx.experimental import MetalQwenEvaluator

        if MetalQwenEvaluator is None:
            raise RuntimeError(
                "Rebuild the ENNX wheel with the Metal Qwen evaluator enabled"
            )
        checkpoint = Path(checkpoint)
        config = Config.from_file(checkpoint / "config.json")
        self.engine = MetalQwenEvaluator(max_tokens)
        self.weights = self.engine.load_weights(str(checkpoint / "model.safetensors"))
        self.config = config

    def losses(self, weights, tokens, masks):
        values = self.engine.losses(weights, tokens, masks)
        if not isinstance(values, list) or any(
            not math.isfinite(value) for value in values
        ):
            raise RuntimeError("Metal Qwen returned a nonfinite loss")
        return values

    @property
    def loss_profile(self):
        return self.engine.last_loss_profile

    @property
    def weights_len(self) -> int:
        return self.engine.weights_len

    @property
    def blocks(self):
        return self.engine.blocks()

    def logits(self, weights, tokens):
        values = self.engine.logits(weights, tokens)
        expected = (len(tokens), self.engine.vocab)
        if len(values) != expected[0] or any(len(row) != expected[1] for row in values):
            raise RuntimeError(f"Metal Qwen logits must have shape {expected}")
        return values

    def next_logits(self, weights, tokens):
        values = self.engine.next_logits(weights, tokens)
        if len(values) != self.engine.vocab or any(
            not math.isfinite(value) for value in values
        ):
            raise RuntimeError("Metal Qwen next_logits returned an invalid shape")
        return values

    def generate(self, weights, tokens, max_new_tokens):
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        values = self.engine.generate(weights, tokens, max_new_tokens)
        if len(values) < len(tokens) + 1:
            raise RuntimeError("Metal Qwen generation returned no token")
        return values

    def bench_generate(self, weights, tokens, max_new_tokens, *, mode: str = "greedy"):
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        profile = self.engine.bench_generate(weights, tokens, max_new_tokens, mode)
        if not isinstance(profile, dict) or profile.get("generated_tokens", 0) < 1:
            raise RuntimeError("Metal Qwen benchmark returned an invalid profile")
        return profile

    def generate_batch(self, weights, prompts, max_new_tokens):
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        if not prompts:
            raise ValueError("prompts must be nonempty")
        values = self.engine.generate_batch(weights, prompts, max_new_tokens)
        if len(values) != len(prompts) or any(
            len(value) < len(prompt) + 1 for value, prompt in zip(values, prompts)
        ):
            raise RuntimeError(
                "Metal Qwen batched generation returned an invalid shape"
            )
        return values

    def ask_generate(self, search, prompts, max_new_tokens, *, seed, draw_seed):
        """Queue one correlated candidate and its first greedy batch together."""
        if not prompts:
            raise ValueError("prompts must be nonempty")
        return search.ask_generate(
            self.engine,
            prompts,
            max_new_tokens,
            seed=seed,
            draw_seed=draw_seed,
        )

    def ask_losses(
        self,
        search,
        tokens,
        masks,
        *,
        seed,
        draw_seed,
        arms=1,
        candidates=4,
        neighbors=2,
    ):
        """Queue one correlated candidate and its teacher-forced losses together."""
        if not tokens:
            raise ValueError("tokens must be nonempty")
        values = search.ask_losses(
            self.engine,
            tokens,
            masks,
            arms=arms,
            candidates=candidates,
            neighbors=neighbors,
            seed=seed,
            draw_seed=draw_seed,
        )
        losses = values[1]
        if (
            not isinstance(losses, list)
            or len(losses) != len(tokens)
            or any(not math.isfinite(value) for value in losses)
        ):
            raise RuntimeError("Metal Qwen returned an invalid loss vector")
        return values

    def sample(
        self,
        weights,
        tokens,
        max_new_tokens,
        *,
        temperature: float,
        top_p: float,
        top_k: int = 0,
        seed: int,
    ):
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        values = self.engine.generate_sampled(
            weights,
            tokens,
            max_new_tokens,
            float(temperature),
            float(top_p),
            int(top_k),
            seed,
        )
        if len(values) < len(tokens) + 1:
            raise RuntimeError("Metal Qwen sampled generation returned no token")
        return values

    def sample_batch(
        self,
        weights,
        prompts,
        max_new_tokens,
        *,
        temperature: float,
        top_p: float,
        top_k: int = 0,
        seeds,
    ):
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        if not prompts:
            raise ValueError("prompts must be nonempty")
        if len(seeds) != len(prompts) or any(
            type(seed) is not int or seed < 0 for seed in seeds
        ):
            raise ValueError("seeds must contain one nonnegative integer per prompt")
        values = self.engine.generate_sampled_batch(
            weights,
            prompts,
            max_new_tokens,
            float(temperature),
            float(top_p),
            int(top_k),
            list(seeds),
        )
        if len(values) != len(prompts) or any(
            len(value) < len(prompt) + 1 for value, prompt in zip(values, prompts)
        ):
            raise RuntimeError("Metal Qwen sampled batch returned an invalid shape")
        return values

    generate_sampled_batch = sample_batch

    def search(
        self,
        base_value: float,
        *,
        capacity: int = 2,
        radius: float = 0.01,
        radius_min: float = 0.0001,
        radius_max: float = 0.08,
        reference_seed: int = 0,
    ):
        from ennx.experimental import MetalSearchState
        from ennx.experimental import MetalParamBlock

        values = (base_value, radius, radius_min, radius_max)
        if any(not math.isfinite(float(value)) for value in values):
            raise ValueError("base_value and radius settings must be finite")
        return MetalSearchState(
            self.weights,
            float(base_value),
            [MetalParamBlock(*block) for block in self.engine.blocks()],
            capacity,
            max_pending=1,
            length_init=radius,
            length_min=radius_min,
            length_max=radius_max,
            sampler="correlated",
            reference_seed=reference_seed,
        )


__all__ = ["Evaluator"]
