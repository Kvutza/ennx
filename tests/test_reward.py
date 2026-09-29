from types import SimpleNamespace

import pytest

from ops.reward_tokenizer import encode


def test_completion():
    tokenizer = SimpleNamespace(
        encode=lambda text, add_special_tokens: SimpleNamespace(
            ids=[ord(character) for character in text]
        )
    )
    row = encode(tokenizer, "def f():\n", "    return 1\n", 100)
    assert row["tokens"][1:] == [ord(c) for c in "def f():\n    return 1\n"]
    assert sum(row["mask"]) == len("    return 1\n")
    assert row["mask"][: 1 + len("def f():\n")] == [False] * 10
    assert row["completion_bytes"] == len("    return 1\n")


def test_overflow():
    tokenizer = SimpleNamespace(
        encode=lambda text, add_special_tokens: SimpleNamespace(ids=list(text.encode()))
    )
    with pytest.raises(ValueError, match="exceeds"):
        encode(tokenizer, "abc", "def", 6)
