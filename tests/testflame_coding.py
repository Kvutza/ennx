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


def test_provenance():
    sample = row(code="x = '\u00e9'\r\n", text="Use a Unicode string.")
    calls = []

    class RecordingTokenizer(CharacterTokenizer):
        def encode(self, text, **kwargs):
            calls.append(text)
            return super().encode(text, **kwargs)

    result = coding.prepare([sample], RecordingTokenizer(), count=1)
    example = result["examples"][0]
    prompt = coding.prompt_for(sample)
    assert calls == [prompt + sample["code"], prompt]
    assert result["format"] == "ennx.solution_tokens.v1"
    assert example["id"] == "mbpp/train/601"
    assert example["tokens"] == [ord(c) for c in prompt + sample["code"]] + [0]
    assert example["loss_mask"] == [False] * len(prompt) + [True] * (
        len(sample["code"]) + 1
    )
    assert all(type(value) is bool for value in example["loss_mask"])
    assert all(type(value) is int for value in example["tokens"])
    assert example["solution"] == sample["code"]
    for key, text in (
        ("prompt_sha256", prompt),
        ("reference_sha256", sample["code"]),
        ("prompt_solution_sha256", prompt + sample["code"]),
    ):
        assert example[key] == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert example["execution_verified"] is False
    provenance = result["provenance"]
    assert "no programs executed" in provenance["reference_verification"]
    assert "BigCodeBench-Hard" in provenance["evaluation_separation"]
    stats = provenance["statistics"]
    assert stats["total_tokens"] == len(example["tokens"])
    assert stats["total_input_tokens"] == len(example["tokens"]) - 1
    assert stats["total_scored_tokens"] == sum(example["loss_mask"])
    assert json.loads(json.dumps(result)) == result


def test_idoversize():
    rows = [row(604, "x" * 400), row(602), row(601, "x" * 400), row(603)]
    result = coding.prepare(rows, CharacterTokenizer(), count=1)
    assert result == coding.prepare(list(reversed(rows)), CharacterTokenizer(), count=1)
    selection = result["provenance"]["selection"]
    assert selection["candidate_task_ids"] == [601, 602, 603, 604]
    assert selection["selected_task_ids"] == [602]
    assert [ex["task_id"] for ex in selection["excluded_oversize"]] == [601, 604]
    assert result["provenance"]["statistics"]["excluded_oversize_count"] == 2


def test_eosunderfill():
    sample = row()
    size = len(coding.prompt_for(sample) + sample["code"]) + 1
    assert (
        len(
            coding.prepare([sample], CharacterTokenizer(), count=1, max_tokens=size)[
                "examples"
            ][0]["tokens"]
        )
        == size
    )
    with pytest.raises(ValueError, match="only 0 eligible; excluded oversize=1"):
        coding.prepare([sample], CharacterTokenizer(), count=1, max_tokens=size - 1)
    with pytest.raises(ValueError, match="Requested 2 examples, only 1 eligible"):
        coding.prepare([sample], CharacterTokenizer(), count=2)


@pytest.mark.parametrize(
    "task_id", [1, 11, 510, 600, 975, "601", True, "BigCodeBench/601"]
)
def test_tasks(task_id):
    with pytest.raises(ValueError, match="Only MBPP TRAIN"):
        coding.prepare([row(task_id)], CharacterTokenizer(), count=1)


def test_dups():
    with pytest.raises(ValueError, match="Duplicate"):
        coding.prepare([row(), row()], CharacterTokenizer(), count=1)


@pytest.mark.parametrize(
    "field,value",
    [
        ("text", ""),
        ("code", "  "),
        ("code", None),
        ("test_setup_code", []),
        ("test_list", "assert x"),
        ("test_list", [3]),
    ],
)
def test_malformedrows(field, value):
    sample = row()
    sample[field] = value
    with pytest.raises(ValueError):
        coding.prepare([sample], CharacterTokenizer(), count=1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"count": 0},
        {"count": True},
        {"count": 375},
        {"max_tokens": 1},
        {"max_tokens": 2049},
        {"max_tokens": False},
    ],
)
def test_optionbounds(kwargs):
    with pytest.raises(ValueError):
        coding.prepare([row()], CharacterTokenizer(), **kwargs)


def test_ref(tmp_path):
    marker = tmp_path / "executed"
    code = f"open({str(marker)!r}, 'w').write('bad')"
    coding.prepare([row(code=code)], CharacterTokenizer(), count=1, max_tokens=2048)
    assert not marker.exists()


class StraddlingTokenizer(CharacterTokenizer):
    def encode(self, text, **kwargs):
        encoded = super().encode(text, **kwargs)
        if "\nX" in text:
            start = text.index("\nX")
            encoded.ids[start : start + 2] = [1234]
            encoded.offsets[start : start + 2] = [(start, start + 2)]
        return encoded


def test_boundary():
    with pytest.raises(coding.BoundaryError, match="straddles"):
        coding.solution_tokens(StraddlingTokenizer(), "p\n", "X")
    result = coding.prepare([row(601, "X"), row(602)], StraddlingTokenizer(), count=1)
    assert result["provenance"]["selection"]["selected_task_ids"] == [602]
    assert result["provenance"]["statistics"]["excluded_boundary_count"] == 1
    with pytest.raises(ValueError, match="boundary=1"):
        coding.prepare([row(601, "X")], StraddlingTokenizer(), count=1)


@pytest.mark.parametrize(
    "failure", ["prefix", "truncation", "empty_offsets", "trimmed_boundary", "vocab"]
)
def test_tokensafety(failure):
    class BrokenTokenizer(CharacterTokenizer):
        def encode(self, text, **kwargs):
            encoded = super().encode(text, **kwargs)
            if text == "p X":
                if failure == "prefix":
                    encoded.ids[0] += 1
                elif failure == "truncation":
                    encoded.ids.pop()
                    encoded.offsets.pop()
                elif failure == "empty_offsets":
                    encoded.offsets[0] = (0, 0)
                elif failure == "trimmed_boundary":
                    encoded.offsets[1] = (2, 3)
                elif failure == "vocab":
                    encoded.ids[0] = 50304
            return encoded

    with pytest.raises(ValueError):
        coding.solution_tokens(BrokenTokenizer(), "p ", "X")


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


def test_pinned(monkeypatch):
    fake_sources(monkeypatch)
    assert [sample["task_id"] for sample in coding.load_train()] == list(
        coding.TRAIN_IDS
    )
    assert coding.DATASET_REVISION in coding.DATASET_INFO_URL
    assert coding.DATASET_SOURCE_REVISION in coding.DATASET_URL
    assert coding.TOKENIZER_REVISION in coding.TOKENIZER_URL


def test_checksum(monkeypatch):
    fake_sources(monkeypatch, corrupt=True)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        coding.load_train()


@pytest.mark.parametrize("failure", [None, "checksum", "eos", "vocab"])
def test_tokenizer(monkeypatch, failure):
    calls = []

    class Tokenizer:
        @classmethod
        def from_str(cls, value):
            assert value == "{}"
            return cls()

        def no_truncation(self):
            calls.append("no_truncation")

        def no_padding(self):
            calls.append("no_padding")

        def token_toid(self, token):
            assert token == "<|endoftext|>"
            return 1 if failure == "eos" else 0

        def get_vocabsize(self):
            return 50257 if failure == "vocab" else 50277

    Tokenizer.token_to_id = Tokenizer.token_toid
    Tokenizer.get_vocab_size = Tokenizer.get_vocabsize

    def byte_level(*, trim_offsets):
        assert trim_offsets is False
        return "untrimmed"

    module = SimpleNamespace(
        Tokenizer=Tokenizer, processors=SimpleNamespace(ByteLevel=byte_level)
    )
    monkeypatch.setitem(sys.modules, "tokenizers", module)
    monkeypatch.setattr(coding, "TOKENIZER_SHA256", hashlib.sha256(b"{}").hexdigest())

    def read(url):
        assert url == coding.TOKENIZER_URL
        return b"bad" if failure == "checksum" else b"{}"

    monkeypatch.setattr(coding, "read_public", read)
    if failure:
        with pytest.raises(
            ValueError,
            match={"checksum": "SHA256", "eos": "EOS ID", "vocab": "vocabulary"}[
                failure
            ],
        ):
            coding.load_tokenizer()
    else:
        tokenizer = coding.load_tokenizer()
        assert calls == ["no_truncation", "no_padding"]
        assert tokenizer.post_processor == "untrimmed"


def test_clidefaults(monkeypatch, tmp_path):
    monkeypatch.setattr(coding, "load_train", lambda: [row(i) for i in range(601, 610)])
    monkeypatch.setattr(coding, "load_tokenizer", CharacterTokenizer)
    monkeypatch.setattr(coding, "version", lambda name: "fake-test-tokenizer")
    output = tmp_path / "objective.json"
    runner = CliRunner()
    result = runner.invoke(coding.main, ["--output", str(output)])
    assert result.exit_code == 0, result.output
    data = json.loads(output.read_text())
    assert len(data["examples"]) == 8
    assert data["provenance"]["selection"]["max_tokens_including_eos"] == 256
    assert "total_input_tokens" in result.output
    assert "total_scored_tokens" in result.output
    assert "excluded_oversize_count" in result.output
    before = output.read_bytes()
    assert runner.invoke(coding.main, ["--output", str(output)]).exit_code != 0
    assert output.read_bytes() == before


def test_cliunderfill(monkeypatch, tmp_path):
    monkeypatch.setattr(coding, "load_train", lambda: [row()])
    monkeypatch.setattr(coding, "load_tokenizer", CharacterTokenizer)
    output = tmp_path / "objective.json"
    result = CliRunner().invoke(coding.main, ["--output", str(output)])
    assert result.exit_code != 0
    assert "Requested 8 examples, only 1 eligible" in result.output
    assert not output.exists()


def test_context(tmp_path):
    result = CliRunner().invoke(
        coding.main, ["--output", str(tmp_path / "bad.json"), "--max-tokens", "2049"]
    )
    assert result.exit_code == 2


def test_policy(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def raise_forstatus(self):
            pass

        def iter_content(self, chunk_size):
            yield b"abc"
            yield b"def"

    Response.raise_for_status = Response.raise_forstatus

    class Session(Response):
        trust_env = True

        def get(self, url, **kwargs):
            assert self.trust_env is False
            assert kwargs == {"stream": True, "timeout": (30, 120)}
            return Response()

    monkeypatch.setattr(coding.requests, "Session", Session)
    assert coding.read_public(coding.DATASET_URL) == b"abcdef"
    monkeypatch.setattr(coding, "MAX_DOWNLOAD_BYTES", 5)
    with pytest.raises(ValueError, match="size limit"):
        coding.read_public(coding.DATASET_URL)
