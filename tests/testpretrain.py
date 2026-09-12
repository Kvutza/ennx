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
    path = Path(__file__).resolve().parents[1] / "examples/tuning/code-pretrain.toml"
    text = path.read_text()
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

    reliability_path = (
        Path(__file__).resolve().parents[1]
        / "examples/tuning/code-pretrain-ablation-local-reliability.toml"
    )
    reliability = pretrain.lower_study(tomllib.loads(reliability_path.read_text()))
    assert reliability["trust_region_kind"] == "reliability"
    assert reliability["reliability_controller"]["local_scale_neighbors"] == 8
    assert reliability == pretrain.lower_study(reliability)


def test_reposplit():
    values = {
        pretrain.repository_split(f"owner/repository-{index}", f"commit-{index}")
        for index in range(100)
    }
    assert values == {"train", "validation", "test"}
    assert pretrain.repository_split(
        "owner/repository", "commit"
    ) == pretrain.repository_split("owner/repository", "commit")


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
    assert first["corpus_id"] == "1282dec2b55702e33f58"
    assert extended["corpus"] == repeated["corpus"] == first["corpus"]
    assert extended["corpus_id"] == repeated["corpus_id"] == first["corpus_id"]
    assert extended["stage"] == first["stage"] == repeated["stage"]
    assert extended["recipe"]["sequences"] == pretrain.PRESET_SEQUENCES
    assert pretrain.PRESET_SEQUENCES["train"] == 20


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


@pytest.mark.parametrize("rounds", [10, 100])
@pytest.mark.parametrize("cached", [False, True])
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

    def prepare(output, train, validation, test, recipe_id, stage):
        calls.append((output, train, validation, test, recipe_id))
        stages.append(stage)

    monkeypatch.setattr(pretrain, "prepare", prepare)
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


def mock_stage(monkeypatch):
    calls = []

    def collect_native(targets):
        calls.append(targets)
        documents = {
            split: {bucket: [] for bucket in pretrain.QUOTAS} for split in targets
        }
        return documents, targets

    corpus = ModuleType("ops.corpus")
    corpus.collect_native = collect_native
    monkeypatch.setitem(sys.modules, "ops.corpus", corpus)
    monkeypatch.setattr(pretrain, "version", lambda _package: "test-version")

    class Tokenizer:
        def save(self, path):
            Path(path).write_text("tokenizer")

    monkeypatch.setattr(pretrain, "train_tokenizer", lambda _documents: Tokenizer())

    def spool(_tokenizer, documents, directory):
        directory.mkdir()
        paths = {split: {} for split in documents}
        counts = {split: {} for split in documents}
        for split in documents:
            for bucket, quota in pretrain.QUOTAS.items():
                count = 200 * quota
                path = directory / f"{split}-{bucket}.u16"
                path.write_bytes(bytes(count * 2))
                paths[split][bucket] = path
                counts[split][bucket] = count
        return paths, counts

    monkeypatch.setattr(pretrain, "spool_tokens", spool)

    def pack(path, _paths, _counts, sequences, _seed):
        path.write_bytes(b"packed")
        mixture = {
            bucket: sequences * quota for bucket, quota in pretrain.QUOTAS.items()
        }
        return hashlib.sha256(b"packed").hexdigest(), mixture

    monkeypatch.setattr(pretrain, "pack_stream", pack)
    return calls


def test_poolreuse(tmp_path, monkeypatch):
    calls = mock_stage(monkeypatch)
    stage = tmp_path / "token-pools" / "shared"
    pretrain.prepare(tmp_path / "round-10", 20, 16, 16, stage=stage)
    pretrain.prepare(tmp_path / "round-100", 200, 16, 16, stage=stage)

    assert len(calls) == 1
    assert (stage / "tokenizer.json").is_file()
    assert (stage / "pools" / "train-implementation.u16").is_file()
    assert (tmp_path / "round-10" / "train.ennxptn").is_file()
    assert (tmp_path / "round-100" / "train.ennxptn").is_file()
    for output in (tmp_path / "round-10", tmp_path / "round-100"):
        tokenizer = output / "tokenizer.json"
        manifest = json.loads((output / "manifest.json").read_text())
        assert tokenizer.is_file()
        assert pretrain.file_digest(tokenizer) == manifest["tokenizer"]["sha256"]


def test_poolmissing(tmp_path, monkeypatch):
    calls = mock_stage(monkeypatch)
    stage = tmp_path / "token-pools" / "shared"
    pretrain.prepare(tmp_path / "first", 20, 16, 16, stage=stage)
    (stage / "pools" / "train-tests.u16").unlink()

    with pytest.raises(ValueError, match="token pool is missing or corrupt"):
        pretrain.prepare(tmp_path / "second", 200, 16, 16, stage=stage)
    assert len(calls) == 1
    assert not (tmp_path / "second").exists()


def test_poolatomic(tmp_path, monkeypatch):
    mock_stage(monkeypatch)
    stage = tmp_path / "token-pools" / "shared"
    replace = pretrain.os.replace

    def fail_publish(source, destination):
        if Path(destination) == stage:
            raise OSError("simulated stage publish failure")
        return replace(source, destination)

    monkeypatch.setattr(pretrain.os, "replace", fail_publish)
    with pytest.raises(OSError, match="simulated stage publish failure"):
        pretrain.prepare(tmp_path / "output", 20, 16, 16, stage=stage)

    assert not stage.exists()
    assert not (tmp_path / "output").exists()
    assert not list(stage.parent.glob(f".{stage.name}.*"))


def test_poolrace(tmp_path, monkeypatch):
    calls = mock_stage(monkeypatch)
    stage = tmp_path / "token-pools" / "shared"
    replace = pretrain.os.replace

    def publish_winner(source, destination):
        if Path(destination) == stage:
            replace(source, destination)
            raise FileExistsError("simulated concurrent publisher")
        return replace(source, destination)

    monkeypatch.setattr(pretrain.os, "replace", publish_winner)
    pretrain.prepare(tmp_path / "output", 20, 16, 16, stage=stage)

    assert len(calls) == 1
    assert (tmp_path / "output" / "train.ennxptn").is_file()
    assert pretrain.file_digest(stage / "tokenizer.json")
    assert not list(stage.parent.glob(f".{stage.name}.*"))
