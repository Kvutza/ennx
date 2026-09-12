"""Generation boundaries without a model, network, or code execution."""

from types import SimpleNamespace

import numpy as np
import pytest
from click.testing import CliRunner

from ops.flame import coding, generate
from ops.flame.config import Config


class Engine:
    def __init__(self, rows):
        self.rows = iter(rows)
        self.prefixes = []

    def next_logits(self, weights, tokens):
        assert weights == "weights"
        self.prefixes.append(tokens.copy())
        return next(self.rows)


def decode(engine, prompt=None, **overrides):
    options = {"vocab": 4, "eos_id": 0, "max_new_tokens": 3, "context": 10}
    options.update(overrides)
    return generate.greedy(
        engine, "weights", [1, 2] if prompt is None else prompt, **options
    )


def test_greedy():
    engine = Engine(
        [
            np.array(row, dtype=np.float32)
            for row in ([0, 1, 1, 0, 100], [4, 1, 2, 3, 100])
        ]
    )
    prompt = [1, 2]
    result = decode(engine, prompt)
    assert result["token_ids"] == [1, 0]
    assert result["problem_forwards"] == 2 and result["stop_reason"] == "eos"
    assert engine.prefixes == [[1, 2], [1, 2, 1]]
    assert prompt == [1, 2]


def test_fixedbudget():
    engine = Engine([np.array([0, 0, 0, 1], dtype=np.float32)] * 3)
    result = decode(engine)
    assert result["token_ids"] == [3, 3, 3]
    assert result["stop_reason"] == "token_limit" and result["problem_forwards"] == 3


def test_first():
    first = np.array([0, 4, 3, 2], dtype=np.float32)
    branch_one = np.array([10, 0, 1, 2], dtype=np.float32)
    branch_two = np.array([0, 2, 10, 1], dtype=np.float32)
    engine = Engine([first, branch_one, first, branch_two])
    result = generate.first_candidates(
        engine,
        "weights",
        [1, 2],
        vocab=4,
        eos_id=0,
        max_new_tokens=2,
        context=10,
        candidates=2,
    )
    assert [candidate["token_ids"] for candidate in result] == [[1, 0], [2, 2]]
    assert [candidate["problem_forwards"] for candidate in result] == [2, 2]
    assert len(engine.prefixes) == 4
    assert engine.prefixes == [[1, 2], [1, 2, 1], [1, 2], [1, 2, 2]]


@pytest.mark.parametrize("prompt", [[], [True], [-1], [4], (1, 2), [1.0]])
def test_invalidprompt(prompt):
    engine = Engine([])
    with pytest.raises(ValueError, match="Prompt"):
        decode(engine, prompt)
    assert engine.prefixes == []


@pytest.mark.parametrize(
    "options",
    [
        {"vocab": 0},
        {"vocab": True},
        {"context": 0},
        {"context": 4},
        {"max_new_tokens": 0},
        {"max_new_tokens": True},
        {"eos_id": 4},
        {"eos_id": False},
    ],
)
def test_decodeguard(options):
    engine = Engine([])
    with pytest.raises(ValueError):
        decode(engine, **options)
    assert engine.prefixes == []


@pytest.mark.parametrize(
    "logits",
    [
        np.zeros((1, 4), dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        np.zeros(4, dtype=np.float64),
        np.array([0, 1, 2, np.nan], dtype=np.float32),
        np.array([0, 1, 2, np.inf], dtype=np.float32),
    ],
)
def test_logits(logits):
    with pytest.raises(ValueError):
        decode(Engine([logits]))


def test_artifactcache(tmp_path, monkeypatch):
    data = b"pinned bytes"
    monkeypatch.setattr(coding, "read_public", lambda _: data)
    expected = coding.sha256(data)
    assert generate.artifact(tmp_path, "data", "url", expected) == data
    monkeypatch.setattr(coding, "read_public", lambda _: pytest.fail("cache miss"))
    assert generate.artifact(tmp_path, "data", "url", expected) == data
    with pytest.raises(ValueError, match="checksum"):
        generate.artifact(tmp_path, "data", "url", "0" * 64)
    assert (tmp_path / "data").read_bytes() == data


def case_fixture():
    row = {
        "task_id": 602,
        "text": "Task",
        "code": "def f(): return 1",
        "test_setup_code": "",
        "test_list": ["assert f() == 1"],
    }
    prompt = coding.prompt_for(row)
    example = {
        "id": "mbpp/train/602",
        "task_id": 602,
        "prompt": prompt,
        "solution": row["code"],
        "prompt_sha256": coding.text_hash(prompt),
        "tokens": [1, 2, 3, 0],
        "loss_mask": [False, False, True, True],
    }
    document = {
        "format": "ennx.solution_tokens.v1",
        "examples": [example],
        "provenance": {},
    }
    tokenizer = SimpleNamespace(
        encode=lambda text, **kwargs: SimpleNamespace(ids=[1, 2])
    )
    return document, tokenizer, [row]


def test_prompt():
    args = case_fixture()
    cases = generate.cases_for(*args, 1, 128, Config())
    assert cases[0]["prompt_token_ids"] == [1, 2]
    assert cases[0]["reference"] == args[2][0]["code"]


@pytest.mark.parametrize(
    "change", ["prompt", "reference", "hash", "tokenization", "duplicate", "split"]
)
def test_prov(change):
    document, tokenizer, rows = case_fixture()
    if change == "prompt":
        document["examples"][0]["prompt"] += "solution leak"
    elif change == "reference":
        document["examples"][0]["solution"] = "different"
    elif change == "hash":
        document["examples"][0]["prompt_sha256"] = "wrong"
    elif change == "tokenization":
        tokenizer.encode = lambda *args, **kwargs: SimpleNamespace(ids=[1, 2, 3])
    elif change == "duplicate":
        rows += rows
    else:
        document["examples"][0]["task_id"] = rows[0]["task_id"] = 500
    with pytest.raises(ValueError):
        generate.cases_for(document, tokenizer, rows, 1, 128, Config())


def test_escape():
    result = {
        "token_ids": [1],
        "text": "<script>alert(1)</script>",
        "stop_reason": "token_limit",
        "check": {"status": "failed"},
    }
    document = {
        "cases": [{"id": "mbpp/train/602", "prompt": "<prompt>"}],
        "results": {
            label: {"mbpp/train/602": result} for label in ("original", "optimized")
        },
    }
    report = generate.render_report(document)
    assert "<script>" not in report and "&lt;script&gt;" in report
    assert "1-problem" in report and "Content-Security-Policy" in report


def test_budget():
    result = CliRunner().invoke(generate.main, ["run", "--help"])
    assert result.exit_code == 0
    assert "--max-new-tokens" in result.output and "--count" in result.output
