"""Integration parity for the native corpus reader and the Python reference."""

from itertools import product
from dataclasses import asdict

import pytest

from ops import corpus, pretrain

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


def repository(split):
    return next(
        f"owner/repo-{i}"
        for i in range(1000)
        if pretrain.repository_split(f"owner/repo-{i}", "commit") == split
    )


def file(content_id, path="src/model.py", language="Python", **changes):
    return {
        "content_id": content_id,
        "content": "x" * 80,
        "file_path": path,
        "language": language,
        "is_vendor": False,
        "license_type": "permissive",
        "unused": "not projected",
        **changes,
    }


def records():
    rows = []
    for split in ("train", "validation", "test"):
        base = {"repo_path": repository(split), "commit_id": "commit"}
        rows.append(
            {
                **base,
                "github_metadata": {"is_fork": True},
                "files": [file("fork")],
            }
        )
        rows.append(
            {
                **base,
                "github_metadata": {"is_fork": False},
                "files": [
                    file("vendor", is_vendor=True),
                    file("license", license_type="copyleft"),
                    file("short", content="x" * 63),
                    file("long", content="x" * 262145),
                    file("nul", content="x" * 80 + "\0"),
                    file("null", content=None),
                    file("foreign", language="Go"),
                    file("shared"),
                    file(split + "-code", content="\u05d0" * 80),
                    file(split + "-test", "TESTS/TEST_MODEL.PY"),
                    file(split + "-doc", "Readme.custom", "Other"),
                    file(split + "-config", "pyproject.toml", "TOML"),
                ],
            }
        )
    return rows


def job(tmp_path, paths, targets):
    return {
        "root": str(tmp_path),
        "paths": paths,
        "targets": targets,
        "max_file": pretrain.MAX_FILE_CHARS,
        "max_repository": pretrain.MAX_REPOSITORY_CHARS,
    }


@pytest.mark.parametrize(
    "repository_limit,row_group",
    [
        (case_0, case_1)
        for case_0, case_1 in product([256, pretrain.MAX_REPOSITORY_CHARS], [1, 64])
    ],
)
def test_native(tmp_path, monkeypatch, row_group, repository_limit):
    monkeypatch.setattr(pretrain, "MAX_REPOSITORY_CHARS", repository_limit)
    rows = records()
    rows += [
        {
            "repo_path": repository(split),
            "commit_id": "different-commit",
            "github_metadata": {"is_fork": False},
            "files": [file(split + "-extra", "config.toml", "TOML")],
        }
        for split in pretrain.PRESET_SEQUENCES
    ]
    paths = []
    for index, shard in enumerate((rows[:3], rows[3:])):
        name = f"{index}.parquet"
        pq.write_table(
            pa.Table.from_pylist(shard),
            tmp_path / name,
            row_group_size=row_group,
            compression="snappy",
        )
        paths.append(name)
    targets = {
        split: {bucket: 80 for bucket in pretrain.QUOTAS}
        for split in pretrain.PRESET_SEQUENCES
    }
    documents, characters = pretrain.collect(rows, targets)
    selected = corpus.read_native(job(tmp_path, paths, targets))
    assert selected["characters"] == characters
    assert selected["documents"] == {
        split: {
            bucket: [asdict(document) for document in documents[split][bucket]]
            for bucket in pretrain.QUOTAS
        }
        for split in targets
    }


def test_exhausted(tmp_path):
    pq.write_table(pa.Table.from_pylist(records()), tmp_path / "data.parquet")
    targets = {"train": {bucket: 10000 for bucket in pretrain.QUOTAS}}
    with pytest.raises(ValueError, match="source exhausted before corpus quotas"):
        corpus.read_native(job(tmp_path, ["data.parquet"], targets))
