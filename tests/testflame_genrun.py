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


@pytest.fixture
def harness(tmp_path, monkeypatch):
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
        assert directory in (original, best)
        state.loads.append(directory)
        label = "original" if directory == original else "optimized"
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


def run(harness):
    return generate.run(
        harness.original,
        harness.run_directory,
        harness.output,
        count=2,
        max_new_tokens=3,
        candidates=1,
    )


def failed_report(harness, *, generation, parity, reload):
    report = json.loads((harness.output / "generation.json").read_text())
    assert report["status"] == "failed"
    assert report["forward_counts_complete"] is False
    assert report["problem_forwards"] == generation
    assert report["parity_problem_forwards"] == parity
    assert report["reload_problem_forwards"] == reload
    assert report["error"]
    assert not (harness.output / "report.html").exists()
    return report


def test_outputs(harness):
    result = run(harness)
    assert result == json.loads((harness.output / "generation.json").read_text())
    assert result["status"] == "complete"
    assert result["forward_counts_complete"] is True
    assert result["reference_checks_valid"] is True
    assert result["problem_forwards"] == 10
    assert result["parity_problem_forwards"] == 4
    assert result["reload_problem_forwards"] == 2
    assert result["test_passes"] is None
    assert result["max_new_tokens"] == 3
    assert harness.loads == [harness.original, harness.best]
    assert [engine.capacity for engine in harness.engines] == [6, 6, 8]
    assert result["run_sha256"] == coding.sha256(
        (harness.run_directory / "run.json").read_bytes()
    )
    for label, directory in (
        ("original", harness.original),
        ("optimized", harness.best),
    ):
        assert result["checkpoint_manifest_sha256"][label] == coding.sha256(
            (directory / "manifest.json").read_bytes()
        )
        assert result["next_logits_parity"][label] == "exact_full_logits_last_row"
        for index, row in enumerate(harness.rows[:2]):
            completion = result["results"][label][f"mbpp/train/{row['task_id']}"]
            ids = harness.sequences[label][harness.prompts[index][0]]
            text_ids = ids[:-1] if ids[-1] == coding.EOS_ID else ids
            text = "".join(harness.pieces[token] for token in text_ids)
            assert completion["token_ids"] == ids
            assert completion["text"] == text
            assert completion["problem_forwards"] == len(ids)
            assert completion["stop_reason"] == (
                "eos" if ids[-1] == 0 else "token_limit"
            )
            assert completion["check"] == {"status": "not_checked"}
    assert harness.decodes == [[101], [102, 102, 102], [201, 201], [202]]
    assert [case["prompt_token_ids"] for case in result["cases"]] == harness.prompts[:2]
    assert harness.checks == [
        (row["code"], row["test_setup_code"], row["test_list"])
        for row in harness.rows[:2]
    ]
    calls = harness.calls
    first_load = next(i for i, call in enumerate(calls) if call[0] == "load")
    assert sum(call[0] == "reference" for call in calls[:first_load]) == 2
    reload_index = next(i for i, call in enumerate(calls) if call[0] == "reload")
    selected = [harness.training["tokens"]["examples"][i] for i in (2, 0)]
    assert calls[reload_index] == (
        "reload",
        "optimized",
        [e["tokens"] for e in selected],
        [e["loss_mask"] for e in selected],
    )
    assert not any(
        call[:2] == ("generate", "optimized") for call in calls[:reload_index]
    )
    assert sum(call[0] == "generate" for call in calls) == 10
    html = (harness.output / "report.html").read_text()
    assert "2-problem" in html and "&lt;raw&gt;" in html


@pytest.mark.parametrize(
    "change",
    [
        "absent",
        "empty",
        "type",
        "missing_final_minibatch",
        "missing_base_version",
        "missing_evaluations",
        "wrong_base_version",
        "wrong_evaluations",
        "wrong_losses",
    ],
)
def test_badmanifest(harness, change):
    path = harness.best / "manifest.json"
    manifest = json.loads(path.read_text())
    if change == "absent":
        del manifest["optimization"]
    elif change == "empty":
        manifest["optimization"] = {}
    elif change == "type":
        manifest["optimization"] = []
    elif change.startswith("missing_"):
        del manifest["optimization"][change.removeprefix("missing_")]
    elif change == "wrong_losses":
        manifest["optimization"]["final_minibatch"]["losses"][0] = 99
    else:
        manifest["optimization"][change.removeprefix("wrong_")] += 1
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="optimization metadata"):
        run(harness)
    assert harness.loads == harness.engines == harness.checks == harness.downloads == []
    assert not harness.output.exists()


@pytest.mark.parametrize("status", ["sandbox_unavailable", "failed"])
def test_ref(harness, status):
    harness.reference_status = status
    with pytest.raises(RuntimeError, match="Reference checks failed"):
        run(harness)
    report = failed_report(harness, generation=0, parity=0, reload=0)
    assert harness.loads == harness.engines == []
    assert len(harness.checks) == 2
    assert {check["status"] for check in report["reference_checks"].values()} == {
        status
    }
    assert report["results"] == {"original": {}, "optimized": {}}


def test_reloadparity(harness):
    harness.reload_losses[0] = np.nextafter(harness.reload_losses[0], np.float32(2))
    with pytest.raises(AssertionError):
        run(harness)
    report = failed_report(harness, generation=5, parity=2, reload=2)
    assert len(report["results"]["original"]) == 2
    assert report["results"]["optimized"] == {}
    assert harness.loads == [harness.original, harness.best]
    assert not any(
        call[0] in ("generate", "parity_next", "parity_full") and call[1] == "optimized"
        for call in harness.calls
    )


@pytest.mark.parametrize("label", ["original", "optimized"])
def test_parityreport(harness, label):
    harness.parity_mismatch = label
    with pytest.raises(AssertionError):
        run(harness)
    optimized = label == "optimized"
    report = failed_report(
        harness,
        generation=5 if optimized else 0,
        parity=4 if optimized else 2,
        reload=2 if optimized else 0,
    )
    assert report["results"][label] == {}
    assert label not in report["next_logits_parity"]
    assert not any(call[:2] == ("generate", label) for call in harness.calls)
    assert [
        call[0]
        for call in harness.calls
        if call[:2] in (("parity_next", label), ("parity_full", label))
    ] == ["parity_next", "parity_full"]


@pytest.mark.parametrize("label", ["original", "optimized"])
def test_logits(harness, label):
    harness.bad_logits = label
    with pytest.raises(ValueError, match="one FP32 vocabulary vector"):
        run(harness)
    optimized = label == "optimized"
    report = failed_report(
        harness,
        generation=7 if optimized else 2,
        parity=4 if optimized else 2,
        reload=2 if optimized else 0,
    )
    calls = [call for call in harness.calls if call[:2] == ("generate", label)]
    assert len(calls) == 2
    assert calls[0][2] == harness.prompts[0]
    assert calls[1][2] == harness.prompts[0] + harness.sequences[label][10][:1]
    assert report["results"][label] == {}
    completed = sum(
        value["problem_forwards"]
        for values in report["results"].values()
        for value in values.values()
    )
    assert report["problem_forwards"] == completed + 2


@pytest.mark.parametrize("entry", ["function", "cli"])
def test_raw(harness, entry):
    before = run(harness)
    harness.checks.clear()
    loads = harness.loads.copy()
    engines = harness.engines.copy()
    call_count = len(harness.calls)
    if entry == "function":
        checked = generate.check_outputs(harness.output)
    else:
        invocation = CliRunner().invoke(generate.main, ["check", str(harness.output)])
        assert invocation.exit_code == 0, invocation.output
        checked = json.loads((harness.output / "generation.json").read_text())
    assert checked == json.loads((harness.output / "generation.json").read_text())
    assert checked["test_passes"] == {"original": 0, "optimized": 2}
    assert checked["all_completions_checked"] is True
    assert harness.loads == loads and harness.engines == engines
    assert all(call[0] == "check" for call in harness.calls[call_count:])
    expected_checks = []
    for label, values in before["results"].items():
        for case in before["cases"]:
            completion = values[case["id"]]
            expected_checks.extend(
                (
                    candidate["text"],
                    case["setup"],
                    case["tests"],
                )
                for candidate in completion["candidates"]
            )
            actual = checked["results"][label][case["id"]]
            assert actual["selected_candidate"] == 0
            assert actual["selected_text"] == completion["text"]
            assert [
                candidate["check"]["status"] for candidate in actual["candidates"]
            ] == [actual["check"]["status"]]
            for key, value in completion.items():
                if key != "check":
                    assert actual[key] == value or key in (
                        "candidates",
                        "selected_candidate",
                        "selected_text",
                    )
    assert harness.checks == expected_checks
    for key, value in before.items():
        if key not in ("results", "test_passes"):
            assert checked[key] == value
    html = (harness.output / "report.html").read_text()
    assert "Original: failed" in html and "Optimized: passed" in html
    with pytest.raises(ValueError, match="already been scored"):
        generate.check_outputs(harness.output)
    assert harness.checks == expected_checks


@pytest.mark.parametrize(
    "change", ["incomplete", "invalid_references", "already_scored"]
)
def test_invalidreport(harness, change):
    report = run(harness)
    if change == "incomplete":
        report["status"] = "failed"
    elif change == "invalid_references":
        report["reference_checks_valid"] = False
    else:
        report["test_passes"] = {"original": 0, "optimized": 0}
    path = harness.output / "generation.json"
    path.write_text(json.dumps(report))
    before = path.read_bytes()
    calls = harness.calls.copy()
    with pytest.raises(ValueError):
        generate.check_outputs(harness.output)
    assert harness.calls == calls
    assert path.read_bytes() == before


def test_isolation(harness):
    run(harness)
    harness.completion_status = "sandbox_unavailable"
    report = generate.check_outputs(harness.output)
    assert report["reference_checks_valid"] is True
    assert report["all_completions_checked"] is False
    assert report["test_passes"] == {"original": 0, "optimized": 0}
    assert all(
        completion["check"]["status"] == "sandbox_unavailable"
        for values in report["results"].values()
        for completion in values.values()
    )
