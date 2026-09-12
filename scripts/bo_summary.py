import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


DESCRIPTION = "Summarize the bounded GPU selection and geometry audits."


def records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def geometry(path):
    groups = defaultdict(list)
    for row in records(path):
        groups[(row["dimensions"], row["perturbation"])].append(row)
    expected = {
        (dimension, noise)
        for dimension in [4096, 65536, 1_048_576]
        for noise in ["independent_gaussian", "independent_rademacher"]
    }
    assert set(groups) == expected, "incomplete geometry experiment"
    result = []
    for (dimensions, noise), rows in sorted(groups.items()):
        assert len(rows) == 45
        assert {(r["seed"], r["round"]) for r in rows} == {
            (seed, step) for seed in [123, 1009, 2027] for step in range(10, 25)
        }
        result.append(
            {
                "dimensions": dimensions,
                "noise": noise,
                "pools": len(rows),
                "winner_changes": sum(r["winner_changed"] for r in rows),
                "radius_changes": sum(r["radius_changed"] for r in rows),
                "cpu_gpu_winner_disagreements": sum(
                    r["cpu_approx_selected"] != r["gpu_selected"] for r in rows
                ),
                "max_relative_distance_error": max(
                    r["max_relative_distance_error"] for r in rows
                ),
                "max_resident_relative_distance_error": max(
                    r["max_resident_relative_distance_error"] for r in rows
                ),
                "max_exact_score_regret": max(r["exact_score_regret"] for r in rows),
                "winner_changes_with_positive_regret": sum(
                    r["winner_changed"] and r["exact_score_regret"] > 0 for r in rows
                ),
            }
        )
    return result


def selection(path):
    data = records(path)
    assert data[-1]["kind"] == "completed", "incomplete selection experiment"
    protocol = data[0]
    results = []
    for seed in protocol["proposal_seeds"]:
        heldout = {
            (r["policy"], r["phase"]): r
            for r in data
            if r["kind"] == "heldout" and r["seed"] == seed
        }
        rounds = {
            policy: [
                r
                for r in data
                if r["kind"] == "round" and r["seed"] == seed and r["policy"] == policy
            ]
            for policy in ["enn", "random"]
        }
        assert all(len(rows) == protocol["rounds"] for rows in rounds.values())
        assert (
            heldout["enn", "initial"]["sequence_nlls"]
            == heldout["random", "initial"]["sequence_nlls"]
        )
        validate_initialization(seed, rounds)
        initial = heldout["enn", "initial"]["mean_nll"]
        enn = heldout["enn", "final"]["mean_nll"]
        random = heldout["random", "final"]["mean_nll"]
        results.append(
            {
                "seed": seed,
                "initial_nll": initial,
                "enn_final_nll": enn,
                "random_final_nll": random,
                "enn_minus_random_nll": enn - random,
                "enn_nll_improvement": initial - enn,
                "random_nll_improvement": initial - random,
                "accepted": {
                    policy: sum(r["accepted"] for r in rows)
                    for policy, rows in rounds.items()
                },
                "accepted_after_initialization": {
                    policy: sum(r["accepted"] for r in rows if not r["initializing"])
                    for policy, rows in rounds.items()
                },
            }
        )
    return {
        "pairs": results,
        "mean_enn_minus_random_nll": statistics.mean(
            r["enn_minus_random_nll"] for r in results
        ),
        "scope": (
            "three proposal-seed replicates; fixed model initialization "
            "and validation set; pilot only"
        ),
    }


def validate_initialization(seed, rounds):
    keys = [
        "initializing",
        "candidate_seed",
        "radius",
        "reward",
        "variance",
        "accepted",
    ]
    for left, right in zip(rounds["enn"], rounds["random"]):
        if not (left["initializing"] or right["initializing"]):
            continue
        for key in keys:
            assert left[key] == right[key], (seed, key, left, right)


def distance_scaling(path):
    rows = records(path)
    assert len(rows) == 3, "incomplete distance-scaling experiment"
    assert {row["seed"] for row in rows} == {123, 1009, 2027}
    assert all(row["kind"] == "distance_scaling_same_pool" for row in rows)
    deltas = [row["self_tuning_minus_global_nll"] for row in rows]
    return {
        "pairs": [
            {
                "seed": row["seed"],
                "global_selected": row["global_selected"],
                "self_tuning_selected": row["self_tuning_selected"],
                "selection_changed": row["selection_changed"],
                "self_tuning_minus_global_nll": row["self_tuning_minus_global_nll"],
            }
            for row in rows
        ],
        "selection_changes": sum(row["selection_changed"] for row in rows),
        "self_tuning_wins": sum(delta < 0 for delta in deltas),
        "ties": sum(delta == 0 for delta in deltas),
        "self_tuning_losses": sum(delta > 0 for delta in deltas),
        "mean_self_tuning_minus_global_nll": statistics.mean(deltas),
        "median_self_tuning_minus_global_nll": statistics.median(deltas),
        "scope": "three paired first-guided pools; fixed validation; pilot only",
    }


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--geometry")
    parser.add_argument("--selection")
    parser.add_argument("--same-pool")
    parser.add_argument("--distance-scaling")
    args = parser.parse_args()
    summary = {}
    if args.geometry:
        summary["geometry"] = geometry(args.geometry)
    if args.selection:
        summary["selection"] = selection(args.selection)
    if args.same_pool:
        rows = records(args.same_pool)
        assert len(rows) == 3
        assert {r["seed"] for r in rows} == {123, 1009, 2027}
        summary["same_pool"] = [
            {
                "seed": r["seed"],
                "selected": r["selected"],
                "candidate_mean_nlls": r["candidate_mean_nlls"],
                "enn_minus_uniform_expected_nll": r["enn_minus_uniform_expected_nll"],
                "enn_minus_same_radius_expected_nll": r[
                    "enn_minus_same_radius_expected_nll"
                ],
                "incumbent_mean_nll": r["incumbent_mean_nll"],
            }
            for r in rows
        ]
    if args.distance_scaling:
        summary["distance_scaling"] = distance_scaling(args.distance_scaling)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
