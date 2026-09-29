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


from gen_fixtures import harness


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
    assert_completions(harness, result)
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
    expected_checks = assert_results(before, checked)
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


def assert_completions(harness, result):
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


def assert_results(before, checked):
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
            assert_completion(checked["results"][label][case["id"]], completion)
    return expected_checks


def assert_completion(actual, completion):
    assert actual["selected_candidate"] == 0
    assert actual["selected_text"] == completion["text"]
    assert [candidate["check"]["status"] for candidate in actual["candidates"]] == [
        actual["check"]["status"]
    ]
    for key, value in completion.items():
        if key != "check":
            assert actual[key] == value or key in (
                "candidates",
                "selected_candidate",
                "selected_text",
            )
