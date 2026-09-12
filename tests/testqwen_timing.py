"""Whole-round accounting without requiring a checkpoint or Metal device."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from ops.qwen import bo


@pytest.mark.parametrize("refresh,expected_evaluations", [(1, [2, 2]), (4, [1, 1])])
def test_roundtime(monkeypatch, tmp_path, refresh, expected_evaluations):
    now = 0.0
    evaluators = []

    def advance(seconds):
        nonlocal now
        now += seconds

    class Search:
        best = -1.0

        def memory_info(self):
            return {
                "row_bytes": 20,
                "search_resident_bytes": 100,
                "device_allocated_bytes": 120,
                "recommended_working_set_bytes": 1_000,
                "max_buffer_bytes": 2_000,
            }

        def controller_info(self):
            return {
                "dimensions": 10,
                "evaluated_arms": 1,
                "length": 0.01,
                "length_min": 0.0001,
                "length_max": 0.08,
                "success_tolerance": 3,
                "failure_tolerance": 10,
                "success_counter": 0,
                "failure_counter": 1,
                "restarts": 0,
            }

        def incumbent(self):
            return None

        def tell_paired(self, *args, **kwargs):
            advance(0.2)

        def sync(self):
            advance(0.3)
            return [False]

    class Evaluator:
        weights_len = 10
        blocks = [(3, 0, 4, 0.5, 0.1), (7, 4, 6, 0.25, 0.2)]
        loss_profile = None

        def __init__(self, *args, **kwargs):
            self.weights = None
            evaluators.append(self)
            advance(2.0)

        def search(self, *args, **kwargs):
            return Search()

        def losses(self, *args):
            advance(1.0)
            return [1.0, 1.0]

        def ask_losses(self, *args, **kwargs):
            advance(1.5)
            proposals = SimpleNamespace(
                describe=lambda: [(7, 0.25, 0.01, [(3, 1.5), (0, 0.0), (2, 0.5)])],
                geometry=lambda: [(2, 0.0)],
                base_id=lambda: 1,
                history_dists=lambda: [(1, 0.75)],
                pool_geometry=lambda: (
                    [0.01, 0.04, 0.01, 0.04],
                    [
                        (0, 1, 0.99),
                        (0, 2, 0.0),
                        (0, 3, 0.0),
                        (1, 2, 0.0),
                        (1, 3, 0.0),
                        (2, 3, 0.99),
                    ],
                    [0.75, 0.75, 0.0, 0.0],
                ),
                pool=lambda: [
                    (0, 5, 0.01, 0.75, [(1, 0.5)]),
                    (1, 5, 0.04, 0.75, [(1, 0.6)]),
                    (2, 7, 0.01, 0.0, [(1, 0.75)]),
                    (3, 7, 0.04, 0.0, [(1, 0.9)]),
                ],
            )
            return proposals, [1.0, 1.0]

    objective = SimpleNamespace(
        tokens=np.ones((2, 4), dtype=np.int32),
        mask=np.ones((2, 4), dtype=bool),
        metadata={},
    )
    monkeypatch.setattr(bo, "Evaluator", Evaluator)
    monkeypatch.setattr(bo.SolutionObjective, "parse", lambda *args: objective)
    monkeypatch.setattr(bo.time, "perf_counter", lambda: now)
    monkeypatch.setattr(bo.click, "echo", lambda *args: advance(0.4))
    source = tmp_path / "objective.json"
    source.write_text("{}")
    (tmp_path / "manifest.json").write_text("{}")
    output = tmp_path / "run"
    report = bo.run(
        tmp_path,
        source,
        output,
        evaluations=3,
        candidates=4,
        history=2,
        minibatch_size=2,
        minibatch_refresh=refresh,
        seed=42,
        radius=0.01,
        export_checkpoint=False,
    )
    assert "weights" not in vars(evaluators[0])
    timing = report["timing"]
    settings = report["settings"]
    assert settings["history_policy"] == "fifo_absolute"
    assert settings["surrogate_fit_objective"] == "row_id_loocv_likelihood_fixed"
    assert settings["perturbation_semantics"] == "dense_full_tensor_correlated_bf16"
    assert settings["reference_seed"] == 42
    assert settings["objective_context_tokens"] == 4
    assert settings["context_targets"] == [4096, 16384, 32768]
    assert settings["context_target_reached"] is False
    assert settings["long_context_loss_path"] == (
        "cached_causal_attention_bf16_kv_above_256"
    )
    assert settings["comparison_baseline"] == {
        "name": "EGGROLL",
        "method": "hyperscale_evolution_strategies",
        "evaluated_in_run": False,
    }
    assert settings["controller"] == {
        "dimensions": 10,
        "evaluated_arms": 1,
        "length": 0.01,
        "length_min": 0.0001,
        "length_max": 0.08,
        "success_tolerance": 3,
        "failure_tolerance": 10,
        "success_counter": 0,
        "failure_counter": 1,
        "restarts": 0,
    }
    assert report["resources"] == {
        "model_bf16_elements": 10,
        "search": {
            "row_bytes": 20,
            "search_resident_bytes": 100,
            "device_allocated_bytes": 120,
            "recommended_working_set_bytes": 1_000,
            "max_buffer_bytes": 2_000,
        },
        "perturbation_layout": {
            "scale_scheme": "checkpoint_tensor_bf16_rms_fp32",
            "distance_scheme": "equal_tensor_weighted_relative_squared_l2",
            "rounding": "bf16_nearest_even",
            "blocks": [
                {
                    "key": 3,
                    "offset": 0,
                    "elements": 4,
                    "rms_scale": 0.5,
                    "distance_weight": 0.1,
                },
                {
                    "key": 7,
                    "offset": 4,
                    "elements": 6,
                    "rms_scale": 0.25,
                    "distance_weight": 0.2,
                },
            ],
        },
    }
    assert timing["setup_seconds"] == pytest.approx(3.0)
    assert timing["baseline_seconds"] == pytest.approx(1.0)
    assert timing["export_seconds"] == 0.0
    assert timing["bo_loop_seconds"] == pytest.approx(
        sum(row["round_seconds"] for row in timing["rounds"])
    )
    for row, evaluations in zip(timing["rounds"], expected_evaluations):
        assert row["objective_evaluations"] == evaluations
        assert row["acquisition_candidates"] == 4
        assert row["selected_candidates_evaluated"] == 1
        assert row["decision_sync_seconds"] == pytest.approx(0.3)
        assert row["logging_seconds"] == pytest.approx(0.4)
        assert row["round_seconds"] == pytest.approx(2.4 + (evaluations - 1))
        stages = [
            "incumbent_seconds",
            "proposal_and_score_seconds",
            "decision_submit_seconds",
            "decision_sync_seconds",
            "logging_seconds",
            "other_host_seconds",
        ]
        assert sum(row[key] for key in stages) == pytest.approx(row["round_seconds"])
    for event in report["events"]:
        assert event["controller"] == settings["controller"]
        assert event["perturbation"] == {
            "base_observation_id": 1,
            "seed": 7,
            "candidate_index": 2,
            "reference_correlation": 0.0,
            "realized_reference_cosine": 0.0,
            "reference_normalization": "per_tensor_bf16_rms",
            "acquisition_score": 0.25,
            "radius": 0.01,
            "realized_radius": 0.01,
            "changed_bf16_elements": 5,
            "total_bf16_elements": 10,
            "changed_bf16_fraction": 0.5,
            "changed_blocks": 2,
            "total_blocks": 3,
            "squared_delta": 2.0,
            "history_distances": [{"observation_id": 1, "squared_distance": 0.75}],
            "pairwise_cosines": [
                {"left": 0, "right": 1, "cosine": 0.99},
                {"left": 0, "right": 2, "cosine": 0.0},
                {"left": 0, "right": 3, "cosine": 0.0},
                {"left": 1, "right": 2, "cosine": 0.0},
                {"left": 1, "right": 3, "cosine": 0.0},
                {"left": 2, "right": 3, "cosine": 0.99},
            ],
            "candidate_pool": [
                {
                    "candidate_index": 0,
                    "seed": 5,
                    "radius": 0.01,
                    "realized_radius": 0.01,
                    "reference_correlation": 0.75,
                    "realized_reference_cosine": 0.75,
                    "history_distances": [
                        {"observation_id": 1, "squared_distance": 0.5}
                    ],
                },
                {
                    "candidate_index": 1,
                    "seed": 5,
                    "radius": 0.04,
                    "realized_radius": 0.04,
                    "reference_correlation": 0.75,
                    "realized_reference_cosine": 0.75,
                    "history_distances": [
                        {"observation_id": 1, "squared_distance": 0.6}
                    ],
                },
                {
                    "candidate_index": 2,
                    "seed": 7,
                    "radius": 0.01,
                    "realized_radius": 0.01,
                    "reference_correlation": 0.0,
                    "realized_reference_cosine": 0.0,
                    "history_distances": [
                        {"observation_id": 1, "squared_distance": 0.75}
                    ],
                },
                {
                    "candidate_index": 3,
                    "seed": 7,
                    "radius": 0.04,
                    "realized_radius": 0.04,
                    "reference_correlation": 0.0,
                    "realized_reference_cosine": 0.0,
                    "history_distances": [
                        {"observation_id": 1, "squared_distance": 0.9}
                    ],
                },
            ],
        }
    assert json.loads((output / "run.json").read_text())["timing"] == timing
