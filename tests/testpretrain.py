import tomllib
import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace

import pytest

from ops import pretrain


def test_tuneschema():
    # Legacy syntax remains readable; public v2 syntax is validated in Rust.
    text = """version=1
[pretrain]
model='fbt_pisa1_moe_v1'
corpus='stack_v3_python_pilot_v1'
[surrogate]
method = "enn"
fit_candidates = 30
fit_samples = 10
"""
    legacy = text.replace("[surrogate]", "[surrogate.resident-enn]")
    legacy = legacy.replace('method = "enn"\n', "")
    legacy = legacy.replace("fit_candidates = 30", "candidates = 30")
    legacy = legacy.replace("fit_samples = 10", "samples = 10")
    current = pretrain.lower_study(tomllib.loads(text))
    legacy_lowered = pretrain.lower_study(tomllib.loads(legacy))
    assert current == legacy_lowered
    assert current["num_candidates"] == 30
    assert current["num_samples"] == 10
    assert current == pretrain.lower_study(current)

    reliability = pretrain.lower_study(
        tomllib.loads(
            text
            + """
[trust-region]
method='reliability'
[trust-region.reliability]
local_scale_neighbors=8
"""
        )
    )
    assert reliability["trust_region_kind"] == "reliability"
    assert reliability["reliability_controller"]["local_scale_neighbors"] == 8
    assert reliability == pretrain.lower_study(reliability)

    kernel = pretrain.lower_study(
        tomllib.loads(text + "\n[objective]\nreference='frozen_initial'")
    )
    assert kernel["objective_reference"] == "frozen_initial"
    assert kernel == pretrain.lower_study(kernel)


def test_alias():
    document = {
        "version": 1,
        "study": "pretrain",
        "model": "fbt_pisa1_moe_v1",
        "corpus": pretrain.CORPUS_PRESET,
    }
    alias = pretrain.plan_study(document, Path("."), "study.toml")
    document["model"] = "fbt_pisa1_legacy_v1"
    canonical = pretrain.plan_study(document, Path("."), "study.toml")
    assert alias["experiment_id"] == canonical["experiment_id"]


def test_reposplit():
    values = {
        pretrain.repository_split(f"owner/repository-{index}", f"commit-{index}")
        for index in range(100)
    }
    assert values == {"train", "validation", "test"}
    for index in range(100):
        repository = f"owner/repository-{index}"
        assert pretrain.repository_split(
            repository, "old-commit"
        ) == pretrain.repository_split(repository, "new-commit")


def test_split(monkeypatch):
    recipe = pretrain.corpus_recipe()
    stage = pretrain.token_stage({"train": {"implementation": 100}})
    assert recipe["split_policy"] == stage["split_policy"] == pretrain.SPLIT_POLICY
    monkeypatch.setattr(pretrain, "SPLIT_POLICY", "legacy_repository_and_commit")
    assert pretrain.corpus_recipe() != recipe
    assert pretrain.token_stage(stage["required_characters"]) != stage


def test_classification():
    assert pretrain.classify("src/model.py", "Python") == "implementation"
    assert pretrain.classify("tests/test_model.py", "Python") == "tests"
    assert pretrain.classify("README.md", "Markdown") == "documentation"
    assert pretrain.classify("pyproject.toml", "TOML") == "configuration"
    assert pretrain.classify("main.go", "Go") is None


def test_packmix():
    sequences = 4
    pools = {
        bucket: [index] * (sequences * quota)
        for index, (bucket, quota) in enumerate(pretrain.QUOTAS.items())
    }
    tokens, counts = pretrain.pack(pools, sequences, 7)
    assert len(tokens) == sequences * pretrain.CONTEXT
    assert counts == {
        bucket: sequences * quota for bucket, quota in pretrain.QUOTAS.items()
    }
    for sequence in range(sequences):
        row = tokens[sequence * pretrain.CONTEXT : (sequence + 1) * pretrain.CONTEXT]
        assert {
            index: row.count(index) for index in range(len(pretrain.QUOTAS))
        } == dict(enumerate(pretrain.QUOTAS.values()))


def test_streamhead(tmp_path):
    path = tmp_path / "stream.ennxptn"
    sequences = 2
    digest = pretrain.write_stream(
        path, list(range(pretrain.VOCAB)) * (sequences // 2), sequences
    )
    data = path.read_bytes()
    assert data[:8] == pretrain.MAGIC
    assert int.from_bytes(data[8:12], "little") == pretrain.VOCAB
    assert int.from_bytes(data[12:16], "little") == pretrain.CONTEXT
    assert int.from_bytes(data[16:20], "little") == sequences
    assert len(digest) == 64


def test_diskpack(tmp_path):
    sequences = 2
    pools = {
        bucket: list(range(index, sequences * quota + index))
        for index, (bucket, quota) in enumerate(pretrain.QUOTAS.items())
    }
    tokens, mixture = pretrain.pack(pools, sequences, 19)
    expected = tmp_path / "expected.ennxptn"
    expected_hash = pretrain.write_stream(expected, tokens, sequences)
    paths = {}
    counts = {}
    for bucket, values in pools.items():
        path = tmp_path / f"{bucket}.u16"
        with path.open("xb") as stream:
            counts[bucket] = pretrain.write_values(stream, values)
        paths[bucket] = path

    actual = tmp_path / "actual.ennxptn"
    actual_hash, actual_mixture = pretrain.pack_stream(
        actual, paths, counts, sequences, 19
    )

    assert actual.read_bytes() == expected.read_bytes()
    assert actual_hash == expected_hash
    assert actual_mixture == mixture


def test_spool(tmp_path):
    tokenizer = SimpleNamespace(
        token_to_id=lambda token: pretrain.SPECIALS.index(token),
        encode=lambda text, add_special_tokens: SimpleNamespace(
            ids=[10 + ord(character) - ord("a") for character in text]
        ),
    )
    documents = {"train": {bucket: [] for bucket in pretrain.QUOTAS}}
    documents["train"]["implementation"] = [
        pretrain.Document("first", "a", "a", "a.py", "implementation", "ab"),
        pretrain.Document("second", "b", "b", "b.py", "implementation", "c"),
    ]

    paths, counts = pretrain.spool_tokens(tokenizer, documents, tmp_path / "pools")
    data = paths["train"]["implementation"].read_bytes()
    values = [
        int.from_bytes(data[index : index + 2], "little")
        for index in range(0, len(data), 2)
    ]

    assert values == [10, 11, 1, 2, 12, 1]
    assert counts["train"]["implementation"] == len(values)


def test_cachepaths(tmp_path):
    document = {
        "version": 1,
        "study": "pretrain",
        "model": pretrain.MODEL_PRESET,
        "corpus": pretrain.CORPUS_PRESET,
        "rounds": 10,
        "target_round_ms": 1000,
    }
    first = pretrain.plan_study(document, tmp_path, "source-a")
    second = pretrain.plan_study(dict(document), tmp_path, "source-a")
    changed = pretrain.plan_study({**document, "rounds": 9}, tmp_path, "source-a")
    assert first["corpus"] == second["corpus"] == changed["corpus"]
    assert first["output"] == second["output"]
    assert first["output"] != changed["output"]
    assert first["corpus"].parent == tmp_path / ".cache" / "ennx" / "corpora"
    assert first["output"].parent.parent == tmp_path / ".cache" / "ennx" / "runs"
    extended = pretrain.plan_study({**document, "rounds": 100}, tmp_path, "source-a")
    repeated = pretrain.plan_study({**document, "rounds": 100}, tmp_path, "source-b")
    assert first["corpus_id"] == "5fc12ffb3c1b75386012"
    assert extended["corpus"] == repeated["corpus"] == first["corpus"]
    assert extended["corpus_id"] == repeated["corpus_id"] == first["corpus_id"]
    assert extended["stage"] == first["stage"] == repeated["stage"]
    assert extended["recipe"]["sequences"] == pretrain.PRESET_SEQUENCES
    assert pretrain.PRESET_SEQUENCES["train"] == 20


def test_models(tmp_path):
    base = {
        "version": 1,
        "study": "pretrain",
        "corpus": pretrain.CORPUS_PRESET,
        "rounds": 1,
    }
    plans = [
        pretrain.plan_study({**base, "model": model}, tmp_path, "source")
        for model in pretrain.MODEL_PRESETS
    ]
    assert len({plan["output"] for plan in plans}) == len(pretrain.MODEL_PRESETS)
    assert len({plan["corpus"] for plan in plans}) == 1


def test_resolved(tmp_path):
    document = {
        "version": 1,
        "study": "pretrain",
        "model": pretrain.MODEL_PRESET,
        "corpus": pretrain.CORPUS_PRESET,
        "rounds": 10,
        "target_round_ms": 1000,
    }
    plan = pretrain.plan_study(document, tmp_path, "source-a")
    resolved = tomllib.loads(pretrain.resolved_toml(document, plan))
    assert resolved["output"] == str(plan["output"].resolve())
    assert resolved["dataset"] == str((plan["corpus"] / "train.ennxptn").resolve())


def test_generated(tmp_path):
    raw = {
        "version": 1,
        "pretrain": {
            "model": pretrain.MODEL_PRESET,
            "corpus": pretrain.CORPUS_PRESET,
        },
        "rounds": {"count": 1, "reps": 1, "target_ms": 200},
        "generation": {
            "max_tokens": 4096,
            "temperature": 0.0,
            "corpus_prompt": [1],
            "verify": {
                "mode": "accepted_prefix",
                "window": 128,
                "max_window": 1024,
                "passes": 2,
            },
            "reward": {"kind": "free_running_cross_entropy"},
        },
    }
    document = pretrain.lower_study(raw)
    plan = pretrain.plan_study(document, tmp_path, "source-a")
    resolved = tomllib.loads(pretrain.resolved_toml(document, plan))

    assert resolved["study"] == "pretrain"
    assert resolved["rounds"] == 1
    assert resolved["target_round_ms"] == 200
    assert resolved["generation"] == raw["generation"]
    assert resolved["dataset"] == str((plan["corpus"] / "train.ennxptn").resolve())


@pytest.mark.parametrize(
    ("rounds", "cached"),
    [(10, False), (10, True), (100, False), (100, True)],
)
def test_resolve(tmp_path, monkeypatch, caplog, rounds, cached):
    caplog.set_level("INFO", logger="ennx.pretrain")
    document = {
        "version": 1,
        "study": "pretrain",
        "model": pretrain.MODEL_PRESET,
        "corpus": pretrain.CORPUS_PRESET,
        "rounds": rounds,
        "target_round_ms": 1000,
    }
    config = tmp_path / "study.toml"
    config.write_text(
        "\n".join(
            f"{key} = {pretrain.toml_value(value)}" for key, value in document.items()
        )
        + "\n"
    )
    monkeypatch.setattr(pretrain, "source_identity", lambda _root: "source-a")
    plan = pretrain.plan_study(document, tmp_path, "source-a")
    calls = []

    stages = []

    def prepare(output, train, validation, test, recipe_id, stage, *, episodes=None):
        calls.append((output, train, validation, test, recipe_id))
        stages.append(stage)

    monkeypatch.setattr(pretrain, "prepare", prepare)
    validated = []
    monkeypatch.setattr(
        pretrain,
        "validate_cache",
        lambda corpus, manifest, recipe: validated.append((corpus, manifest, recipe)),
    )
    if cached:
        plan["corpus"].mkdir(parents=True)
        (plan["corpus"] / "manifest.json").write_text(
            '{"recipe_id": "' + plan["corpus_id"] + '"}\n'
        )

    resolved_path = pretrain.resolve_study(config, tmp_path)
    resolved = tomllib.loads(resolved_path.read_text())

    assert resolved_path == plan["resolved"]
    assert resolved["dataset"] == str((plan["corpus"] / "train.ennxptn").resolve())
    assert resolved["output"] == str(plan["output"].resolve())
    assert resolved["rounds"] == rounds
    log = plan["resolved"].with_suffix(".prepare.log").read_text()
    assert (
        f"training batches {pretrain.PRESET_SEQUENCES['train'] // 2}" in log
        if not cached
        else "cache | hit" in log
    )
    assert f"ready | rounds {rounds}" in log
    assert calls == (
        []
        if cached
        else [
            (
                plan["corpus"],
                pretrain.PRESET_SEQUENCES["train"],
                pretrain.PRESET_SEQUENCES["validation"],
                pretrain.PRESET_SEQUENCES["test"],
                plan["corpus_id"],
            )
        ]
    )
    assert stages == ([] if cached else [plan["stage"]])
    assert len(validated) == int(cached)


def test_phases(caplog):
    caplog.set_level("INFO", logger="ennx.pretrain")
    with pretrain.phase("test success"):
        pass
    with pytest.raises(ValueError, match="failure"):
        with pretrain.phase("test failure"):
            raise ValueError("failure")
    assert "test success | start" in caplog.text
    assert "test success | complete | elapsed" in caplog.text
    assert "test failure | failed | elapsed" in caplog.text


@pytest.mark.parametrize("field", ["output", "dataset"])
def test_pathreject(tmp_path, field):
    document = {
        "version": 1,
        "study": "pretrain",
        "model": pretrain.MODEL_PRESET,
        "corpus": pretrain.CORPUS_PRESET,
        "rounds": 10,
        field: "user/path",
    }
    with pytest.raises(ValueError, match="generated automatically"):
        pretrain.plan_study(document, tmp_path, "source-a")
