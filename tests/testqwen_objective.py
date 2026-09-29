"""Architecture and numerical smoke tests for the dense Qwen control path."""

import json

import pytest

from ops.qwen import checkpoint
from ops.qwen.config import Config


def test_coverage():
    from ops.qwen.joint import ScoreTiming
    from ops.qwen.metal import proposal_stats

    class FakeProposals:
        def describe(self):
            return [(7, 0.25, 0.01, [(3, 1.5), (0, 0.0), (2, 0.5)])]

        def geometry(self):
            return [(0, 0.75)]

        def base_id(self):
            return 5

        def history_dists(self):
            return [(11, 0.75), (19, 1.25)]

        def pool(self):
            return [
                (0, 7, 0.01, 0.75, [(11, 0.75), (19, 1.25)]),
                (1, 7, 0.04, 0.75, [(11, 0.8), (19, 1.1)]),
                (2, 9, 0.01, 0.0, [(11, 0.9), (19, 1.0)]),
                (3, 9, 0.04, 0.0, [(11, 1.0), (19, 0.9)]),
            ]

        def pool_geometry(self):
            return (
                [0.011, 0.039, 0.0105, 0.041],
                [
                    (0, 1, 0.99),
                    (0, 2, 0.02),
                    (0, 3, 0.01),
                    (1, 2, 0.03),
                    (1, 3, 0.02),
                    (2, 3, 0.98),
                ],
                [0.76, 0.74, 0.01, -0.02],
            )

    metrics = proposal_stats(FakeProposals(), 10)
    assert metrics["changed_bf16_elements"] == 5
    assert metrics["changed_bf16_fraction"] == 0.5
    assert metrics["changed_blocks"] == 2
    assert metrics["total_blocks"] == 3
    assert metrics["base_observation_id"] == 5
    assert metrics["candidate_index"] == 0
    assert metrics["reference_correlation"] == 0.75
    assert metrics["realized_reference_cosine"] == 0.76
    assert metrics["reference_normalization"] == "per_tensor_bf16_rms"
    assert metrics["realized_radius"] == 0.011
    assert metrics["history_distances"] == [
        {"observation_id": 11, "squared_distance": 0.75},
        {"observation_id": 19, "squared_distance": 1.25},
    ]
    assert metrics["candidate_pool"][2]["reference_correlation"] == 0.0
    assert metrics["candidate_pool"][2]["realized_reference_cosine"] == 0.01
    assert metrics["candidate_pool"][3]["realized_radius"] == 0.041
    assert metrics["pairwise_cosines"][0] == {
        "left": 0,
        "right": 1,
        "cosine": 0.99,
    }

    timing = ScoreTiming(
        elapsed_seconds=2.0,
        generation_seconds=1.5,
        decode_seconds=0.1,
        checker_seconds=0.2,
        batches=2,
        prompt_tokens=8,
        generated_tokens=16,
    )
    assert timing.as_dict()["python_overhead_seconds"] == pytest.approx(0.2)

    class BadGeometry(FakeProposals):
        def geometry(self):
            return [("persistent", 0.75)]

    with pytest.raises(RuntimeError, match="geometry is invalid"):
        proposal_stats(BadGeometry(), 10)

    class ZeroGeometry(FakeProposals):
        def pool_geometry(self):
            return (
                [0.0] * 4,
                [
                    (0, 1, None),
                    (0, 2, None),
                    (0, 3, None),
                    (1, 2, None),
                    (1, 3, None),
                    (2, 3, None),
                ],
                [None] * 4,
            )

    zero = proposal_stats(ZeroGeometry(), 10)
    assert zero["realized_radius"] == 0.0
    assert all(row["cosine"] is None for row in zero["pairwise_cosines"])
    assert zero["realized_reference_cosine"] is None


def test_losspipe():
    from ops.qwen.metal import Evaluator

    calls = []

    class FakeSearch:
        def ask_losses(self, engine, tokens, masks, **kwargs):
            calls.append((engine, tokens, masks, kwargs))
            return "proposal", [0.25, 0.5]

    evaluator = object.__new__(Evaluator)
    evaluator.engine = object()
    tokens = [[1, 2], [3, 4]]
    masks = [[False, True], [False, True]]

    assert evaluator.ask_losses(
        FakeSearch(),
        tokens,
        masks,
        seed=11,
        draw_seed=13,
        arms=1,
        candidates=4,
        neighbors=7,
    ) == ("proposal", [0.25, 0.5])
    assert calls == [
        (
            evaluator.engine,
            tokens,
            masks,
            {"arms": 1, "candidates": 4, "neighbors": 7, "seed": 11, "draw_seed": 13},
        )
    ]
