import hashlib
import json
import sys
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from ops.flame import coding


class CharacterTokenizer:
    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return SimpleNamespace(
            ids=[ord(char) for char in text],
            offsets=[(i, i + 1) for i in range(len(text))],
        )


def row(task_id=601, code="x = 1\n", text="Set x to one."):
    return {
        "task_id": task_id,
        "text": text,
        "code": code,
        "test_setup_code": "",
        "test_list": ["assert x == 1"],
        "challenge_test_list": [],
    }


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*args, **kwargs):
        pytest.fail("Offline tests must not access the network")

    monkeypatch.setattr(coding.requests.Session, "get", denied)


def fake_sources(monkeypatch, *, corrupt=False):
    body = b"\n".join(
        json.dumps(row(i)).encode() for i in [1, 11, 600, *coding.TRAIN_IDS]
    )
    digest = hashlib.sha256(body).hexdigest()
    monkeypatch.setattr(coding, "DATASET_SHA256", digest)
    metadata = {
        "full": {
            "download_checksums": {"original": {"checksum": digest}},
            "splits": {"train": {"num_examples": 374}},
        }
    }
    sources = {
        coding.DATASET_INFO_URL: json.dumps(metadata).encode(),
        coding.DATASET_URL: body + (b" " if corrupt else b""),
    }
    monkeypatch.setattr(coding, "read_public", sources.__getitem__)
