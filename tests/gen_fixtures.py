"""Exercise generation orchestration without native bindings or code execution."""

import json
import sys
from dataclasses import asdict
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from click.testing import CliRunner

from ops.flame import codecheck, coding, generate
from ops.flame.config import Config


def generation_state(tmp_path):
    original = tmp_path / "original"
    run_directory = tmp_path / "run"
    best = run_directory / "best"
    original.mkdir()
    best.mkdir(parents=True)
    (original / "manifest.json").write_text('{"checkpoint": "original"}')
    prompts = [[10, 11], [20, 21, 22], [30, 31, 32, 33]]
    rows, examples = [], []
    for index, task_id in enumerate((602, 604, 605)):
        row = {
            "task_id": task_id,
            "text": f"Task {task_id}",
            "code": f"def reference_{task_id}(): return {task_id}",
            "test_setup_code": f"setup_{task_id} = True",
            "test_list": [f"assert reference_{task_id}() == {task_id}"],
        }
        prompt = coding.prompt_for(row)
        solution_ids = [900 + index, 910 + index, 920 + index, coding.EOS_ID]
        rows.append(row)
        examples.append(
            {
                "id": f"mbpp/train/{task_id}",
                "task_id": task_id,
                "prompt": prompt,
                "solution": row["code"],
                "prompt_sha256": coding.text_hash(prompt),
                "tokens": prompts[index] + solution_ids,
                "loss_mask": [False] * len(prompts[index]) + [True] * 4,
            }
        )
    optimization = {
        "base_version": 6,
        "evaluations": 51,
        "final_minibatch": {"indices": [2, 0], "losses": [1.25, 2.5]},
    }
    training = {
        "status": "complete",
        "backend": "metal",
        "source_manifest_sha256": coding.sha256(
            (original / "manifest.json").read_bytes()
        ),
        "extension_sha256": "training-extension",
        "tokens": {
            "format": "ennx.solution_tokens.v1",
            "examples": examples,
            "provenance": {},
        },
        "blocks": [
            {
                "name": "fake",
                "key": 1,
                "offset": 0,
                "shape": [4],
                "scale": 1,
                "weight": 1,
            }
        ],
        **optimization,
    }
    (run_directory / "run.json").write_text(json.dumps(training))
    (best / "manifest.json").write_text(
        json.dumps({"optimization": {**optimization, "events": "../events.jsonl"}})
    )
    state = SimpleNamespace(
        original=original,
        run_directory=run_directory,
        best=best,
        output=tmp_path / "generation",
        training=training,
        rows=rows,
        prompts=prompts,
        calls=[],
        loads=[],
        engines=[],
        checks=[],
        decodes=[],
        downloads=[],
        reference_status="passed",
        completion_status=None,
        parity_mismatch=None,
        bad_logits=None,
        reload_losses=np.array([1.25, 2.5], dtype=np.float32),
    )
    sequences = {
        "original": {10: [101, 0], 20: [102, 102, 102]},
        "optimized": {10: [201, 201, 0], 20: [202, 0]},
    }
    pieces = {
        101: "```python\n  original_a()\n``` \n",
        102: " original_b() \n",
        201: " optimized_a()\n",
        202: "<raw> optimized_b() \n",
    }
    state.sequences = sequences
    state.pieces = pieces

    return state


@pytest.fixture
def harness(tmp_path, monkeypatch):
    state = generation_state(tmp_path)
    examples = state.training["tokens"]["examples"]
    prompts, rows = state.prompts, state.rows
    sequences, pieces = state.sequences, state.pieces

    class Tokenizer:
        @classmethod
        def from_str(cls, text):
            assert text == "{}"
            return cls()

        def no_padding(self):
            state.calls.append(("no_padding",))

        def no_truncation(self):
            state.calls.append(("no_truncation",))

        def get_vocabsize(self):
            return 50277

        def token_toid(self, token):
            assert token == "<|endoftext|>"
            return coding.EOS_ID

        def encode(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            index = next(
                i for i, example in enumerate(examples) if example["prompt"] == text
            )
            return SimpleNamespace(ids=prompts[index].copy())

        def decode(self, ids, *, skip_special_tokens):
            assert skip_special_tokens is False
            state.decodes.append(ids.copy())
            return "".join(pieces[token] for token in ids)

    Tokenizer.get_vocab_size = Tokenizer.get_vocabsize
    Tokenizer.token_to_id = Tokenizer.token_toid

    def logits_for(token):
        values = np.zeros(Config().vocab, dtype=np.float32)
        values[token] = 1
        values[50277:] = 100  # Padded model IDs must never win decoding.
        return values

    class Engine:
        def __init__(self, config, *, max_tokens):
            assert config == asdict(Config())
            self.capacity = max_tokens
            self.probed = False
            self.generated_calls = 0
            state.engines.append(self)

        def next_logits(self, weights, tokens):
            assert len(tokens) <= self.capacity
            label = weights.label
            if not self.probed:
                assert tokens == prompts[0]
                self.probed = True
                state.calls.append(("parity_next", label, tokens.copy()))
                return logits_for(101)
            state.calls.append(("generate", label, tokens.copy()))
            self.generated_calls += 1
            if state.bad_logits == label and self.generated_calls == 2:
                return np.zeros((1, Config().vocab), dtype=np.float32)
            index = next(
                i for i, prompt in enumerate(prompts[:2]) if prompt[0] == tokens[0]
            )
            prompt = prompts[index]
            sequence = sequences[label][tokens[0]]
            offset = len(tokens) - len(prompt)
            assert tokens == prompt + sequence[:offset]
            return logits_for(sequence[offset])

        def logits(self, weights, tokens):
            assert tokens == prompts[0]
            state.calls.append(("parity_full", weights.label, tokens.copy()))
            values = logits_for(101)
            if state.parity_mismatch == weights.label:
                values[101] = np.nextafter(values[101], np.float32(2))
            return np.tile(values, (len(tokens), 1))

        def losses(self, weights, tokens, masks):
            assert weights.label == "optimized"
            assert max(map(len, tokens)) == self.capacity
            state.calls.append(("reload", weights.label, tokens, masks))
            return state.reload_losses.copy()

    class Layout:
        def __init__(self, blocks):
            assert len(blocks) == 1 and blocks[0].name == "fake"
            self.size = 4

        def flatten_torch(self, params):
            state.calls.append(("flatten", params.label))
            return params

    def load(directory):
        assert directory in (state.original, state.best)
        state.loads.append(directory)
        label = "original" if directory == state.original else "optimized"
        state.calls.append(("load", label))
        return Config(), SimpleNamespace(label=label)

    def upload(flat):
        state.calls.append(("upload", flat.label))
        return flat

    def check(code, setup, tests):
        state.checks.append((code, setup, tests.copy()))
        reference = any(code == row["code"] for row in rows)
        state.calls.append(("reference" if reference else "check", code))
        status = (
            state.reference_status
            if reference
            else state.completion_status
            or ("passed" if "optimized" in code else "failed")
        )
        return {"status": status, "isolation_verified": status in ("passed", "failed")}

    tokenizer_bytes = b"{}"
    dataset_bytes = "\n".join(json.dumps(row) for row in rows).encode()

    def read_public(url):
        state.downloads.append(url)
        return {
            coding.TOKENIZER_URL: tokenizer_bytes,
            coding.DATASET_URL: dataset_bytes,
        }[url]

    package = ModuleType("ennx")
    extension = ModuleType("ennx.ennx_rust")
    extension.__file__ = __file__
    experimental = ModuleType("ennx.experimental")
    experimental.MetalFlameEvaluator = Engine
    package.ennx_rust = extension
    package.experimental = experimental
    tokenizers = ModuleType("tokenizers")
    tokenizers.Tokenizer = Tokenizer
    for module in (package, extension, experimental, tokenizers):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(generate, "Block", SimpleNamespace)
    monkeypatch.setattr(generate, "Layout", Layout)
    monkeypatch.setattr(generate, "load_checkpoint", load)
    monkeypatch.setattr(generate, "version", lambda _: "test-version")
    monkeypatch.setattr(generate.metal, "metal_device", lambda: "fake Metal")
    monkeypatch.setattr(generate.metal, "upload_weights", upload)
    monkeypatch.setattr(codecheck, "check_solution", check)
    monkeypatch.setattr(coding, "read_public", read_public)
    monkeypatch.setattr(coding, "TOKENIZER_SHA256", coding.sha256(tokenizer_bytes))
    monkeypatch.setattr(coding, "DATASET_SHA256", coding.sha256(dataset_bytes))
    return state
