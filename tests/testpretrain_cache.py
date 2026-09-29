import tomllib
import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace

import pytest

from ops import pretrain


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
        (directory / "document-index.json").write_text("{}\n")
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
