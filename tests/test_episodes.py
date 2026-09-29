from pathlib import Path
import tomllib

import pytest

from ops import pretrain
from ops.pretrain_episodes import build_episodes, documents, write_episodes


def pool(path: Path, tokens):
    with path.open("xb") as stream:
        pretrain.write_values(stream, tokens)
    return path


def test_boundaries(tmp_path):
    paths = {
        "train": {
            "implementation": pool(
                tmp_path / "implementation.u16", [10] * 7 + [1, 2] + [11] * 7 + [1]
            ),
            "tests": pool(tmp_path / "tests.u16", [12] * 7 + [1]),
        }
    }
    quotas = {"implementation": 3, "tests": 1}
    first = build_episodes(paths, {"train": 3}, quotas, 2, 5, 7)
    second = build_episodes(paths, {"train": 3}, quotas, 2, 5, 7)
    assert first == second
    rows = first["train"]["episodes"]
    assert len(rows) == 3
    assert {row["prompt"][0] for row in rows} == {10, 11, 12}
    for row in rows:
        assert len(row["prompt"]) == 2
        assert len(row["expected"]) == 5
        assert len(set(row["prompt"] + row["expected"])) == 1


def test_short(tmp_path):
    path = pool(tmp_path / "short.u16", [10] * 4 + [1] + [11] * 4 + [1])
    assert len(list(documents(path))) == 2
    with pytest.raises(ValueError, match="only 0 document-aligned episodes"):
        build_episodes({"train": {"code": path}}, {"train": 1}, {"code": 1}, 2, 5, 9)


def test_provenance(tmp_path):
    path = pool(tmp_path / "code.u16", [10] * 7 + [1])
    with pytest.raises(ValueError, match="episode provenance is missing"):
        write_episodes(
            tmp_path / "episodes.json",
            {"train": {"code": path}},
            {"train": 1},
            {"code": 1},
            2,
            5,
            9,
        )


def test_holdout(tmp_path):
    paths = {
        "validation": {
            "implementation": pool(tmp_path / "validation-code.u16", [10] * 7 + [1]),
            "documentation": pool(tmp_path / "validation-docs.u16", [20] * 7 + [1]),
        },
        "test": {
            "implementation": pool(tmp_path / "test-code.u16", [11] * 7 + [1]),
            "documentation": pool(tmp_path / "test-docs.u16", [21] * 7 + [1]),
        },
    }
    result = build_episodes(
        paths,
        {"validation": 1, "test": 1},
        {"implementation": 3, "documentation": 1},
        2,
        5,
        7,
        sources={"validation": ["validation"], "test": ["test"]},
        buckets=["implementation"],
    )
    assert result["validation"]["episodes"][0]["prompt"][0] == 10
    assert result["test"]["episodes"][0]["prompt"][0] == 11
    assert result["validation"]["mixture_episodes"] == {"implementation": 1}
    assert result["test"]["mixture_episodes"] == {"implementation": 1}


def test_recipe(tmp_path):
    base = {
        "version": 1,
        "study": "pretrain",
        "model": pretrain.MODEL_PRESET,
        "corpus": pretrain.CORPUS_PRESET,
    }
    old = pretrain.plan_study(base, tmp_path, "source")
    new = pretrain.plan_study(
        {
            **base,
            "generation": {
                "max_tokens": 4096,
                "temperature": 0.0,
                "corpus_prompt_tokens": 128,
                "reward": {"kind": "free_running_cross_entropy"},
            },
        },
        tmp_path,
        "source",
    )
    assert old["corpus"] != new["corpus"]
    assert old["stage"] == new["stage"]
    assert new["recipe"]["episodes"]["prompt_tokens"] == 128
    assert new["recipe"]["episodes"]["schema"] == "ennx.document_episodes.v2"
    assert new["recipe"]["episodes"]["counts"] == {
        "train": 16,
        "validation": 3,
        "test": 3,
    }
    assert new["recipe"]["episodes"]["sources"]["validation"] == ["validation"]
    assert new["recipe"]["episodes"]["sources"]["test"] == ["test"]
    assert new["recipe"]["episodes"]["buckets"] == ["implementation", "tests"]


def test_arms(tmp_path):
    arms = []
    base = {
        "version": 1,
        "study": "pretrain",
        "model": pretrain.MODEL_PRESET,
        "corpus": pretrain.CORPUS_PRESET,
        "rounds": 512,
        "reps": 3,
        "generation": {
            "purpose": "systems_probe",
            "max_tokens": 4096,
            "corpus_prompt_tokens": 128,
            "temperature": 0.0,
            "reward": {"kind": "code_contrastive", "max_ngram": 4, "decoys": 3},
            "signal_gate": {
                "rounds": 12,
                "min_distinct_rewards": 3,
                "min_reward_span": 0.000001,
            },
        },
    }
    for selection in ("enn", "random"):
        raw = {**base, "selection": selection}
        assert "seed" not in raw["generation"]
        document = pretrain.lower_study(raw)
        plan = pretrain.plan_study(document, tmp_path, "source-a")
        resolved = tomllib.loads(pretrain.resolved_toml(document, plan))
        assert resolved["rounds"] == 512
        assert resolved["reps"] == 3
        assert resolved["generation"]["max_tokens"] == 4096
        assert resolved["generation"]["reward"] == {
            "kind": "code_contrastive",
            "max_ngram": 4,
            "decoys": 3,
        }
        assert resolved["generation"]["signal_gate"]["rounds"] == 12
        arms.append(resolved)
    assert arms[0]["selection"] == "enn"
    assert arms[1]["selection"] == "random"
    for key in set(arms[0]) - {"selection", "output"}:
        assert arms[0][key] == arms[1][key]
