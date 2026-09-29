import hashlib
import json
import sys
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from ops.flame import coding


from code_fixtures import CharacterTokenizer, row, no_network, fake_sources


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
