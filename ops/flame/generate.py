"""Compare greedy Metal completions from the original and a saved BO checkpoint."""

from __future__ import annotations

import gc
import html
import json
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import click
import numpy as np

from . import coding, metal
from .config import Config
from .layout import Block, Layout
from .native import load_checkpoint
from .objective import SolutionObjective


def greedy(
    engine, weights, prompt, *, vocab, eos_id, max_new_tokens, context, on_forward=None
):
    """Feed only prompt and already generated tokens; ties choose the lowest ID."""
    for name, value in (
        ("vocab", vocab),
        ("context", context),
        ("max_new_tokens", max_new_tokens),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if type(eos_id) is not int or not 0 <= eos_id < vocab:
        raise ValueError("EOS must be in the tokenizer vocabulary")
    if (
        not isinstance(prompt, list)
        or not prompt
        or any(type(token) is not int or not 0 <= token < vocab for token in prompt)
    ):
        raise ValueError("Prompt must be a nonempty list of valid token IDs")
    if len(prompt) + max_new_tokens > context:
        raise ValueError("Prompt plus generation budget exceeds context; no truncation")
    prefix = prompt.copy()
    generated = []
    stop = "token_limit"
    started = time.perf_counter()
    for _ in range(max_new_tokens):
        logits = engine.next_logits(weights, prefix)
        if on_forward is not None:
            on_forward()
        logits = np.asarray(logits)
        if logits.ndim != 1 or logits.size < vocab or logits.dtype != np.float32:
            raise ValueError("Expected one FP32 vocabulary vector from Metal")
        if not np.isfinite(logits).all():
            raise ValueError("Cannot generate from nonfinite logits")
        # The model's embedding vocabulary is padded beyond the tokenizer's IDs.
        token = int(np.argmax(logits[:vocab]))
        generated.append(token)
        if token == eos_id:
            stop = "eos"
            break
        prefix.append(token)
    return {
        "token_ids": generated,
        "stop_reason": stop,
        "problem_forwards": len(generated),
        "seconds": time.perf_counter() - started,
    }


def _checkedlogits(engine, weights, prefix, vocab, on_forward):
    logits = np.asarray(engine.next_logits(weights, prefix))
    if on_forward is not None:
        on_forward()
    if logits.ndim != 1 or logits.size < vocab or logits.dtype != np.float32:
        raise ValueError("Expected one FP32 vocabulary vector from Metal")
    if not np.isfinite(logits).all():
        raise ValueError("Cannot generate from nonfinite logits")
    return logits[:vocab]


def first_candidates(
    engine,
    weights,
    prompt,
    *,
    vocab,
    eos_id,
    max_new_tokens,
    context,
    candidates,
    on_forward=None,
):
    """Complete distinct top-first-token paths without executing them.

    Candidate zero is exactly the existing greedy decode. Branching only at the
    first token keeps the extra work bounded while recovering nearby code starts
    that greedy decoding can discard immediately.
    """
    for name, value in (
        ("vocab", vocab),
        ("context", context),
        ("max_new_tokens", max_new_tokens),
        ("candidates", candidates),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if type(eos_id) is not int or not 0 <= eos_id < vocab:
        raise ValueError("EOS must be in the tokenizer vocabulary")
    if (
        not isinstance(prompt, list)
        or not prompt
        or any(type(token) is not int or not 0 <= token < vocab for token in prompt)
    ):
        raise ValueError("Prompt must be a nonempty list of valid token IDs")
    if len(prompt) + max_new_tokens > context:
        raise ValueError("Prompt plus generation budget exceeds context; no truncation")
    first_started = time.perf_counter()
    first = _checkedlogits(engine, weights, prompt.copy(), vocab, on_forward)
    first_order = np.argsort(-first, kind="stable")[:candidates]
    results = []
    for index, first_token in enumerate(first_order.tolist()):
        branch_started = first_started if index == 0 else time.perf_counter()
        logits = (
            first
            if index == 0
            else _checkedlogits(engine, weights, prompt.copy(), vocab, on_forward)
        )
        token = int(first_token)
        prefix = prompt.copy()
        prefix.append(token)
        generated = [token]
        score = float(logits[token] - np.logaddexp.reduce(logits))
        stop = "eos" if token == eos_id else "token_limit"
        while stop != "eos" and len(generated) < max_new_tokens:
            logits = _checkedlogits(engine, weights, prefix, vocab, on_forward)
            normalizer = float(np.logaddexp.reduce(logits))
            token = int(np.argmax(logits))
            score += float(logits[token] - normalizer)
            generated.append(token)
            if token == eos_id:
                stop = "eos"
            else:
                prefix.append(token)
        results.append(
            {
                "token_ids": generated,
                "stop_reason": stop,
                "problem_forwards": len(generated),
                "seconds": time.perf_counter() - branch_started,
                "log_probability": score,
            }
        )
    return results


def artifact(directory, name, url, expected):
    path = directory / name
    data = path.read_bytes() if path.exists() else coding.read_public(url)
    if coding.sha256(data) != expected:
        raise ValueError(f"Pinned artifact checksum mismatch: {name}")
    if not path.exists():
        with path.open("xb") as stream:
            stream.write(data)
    return data


def cases_for(document, tokenizer, dataset, count, max_new_tokens, config):
    objective = SolutionObjective.parse(document, config)
    if type(count) is not int or not 1 <= count <= len(document["examples"]):
        raise ValueError("Count must select a nonempty prefix of the corpus")
    by_id = {}
    for row in dataset:
        if row["task_id"] in by_id:
            raise ValueError("Duplicate dataset task ID")
        by_id[row["task_id"]] = row
    cases = []
    for index, example in enumerate(document["examples"][:count]):
        row = by_id.get(example["task_id"])
        if row is None or row["task_id"] not in coding.TRAIN_IDS:
            raise ValueError("Generation sample must match the pinned TRAIN data")
        prompt = coding.prompt_for(row)
        if prompt != example["prompt"] or row["code"] != example["solution"]:
            raise ValueError("Corpus prompt/reference differs from the pinned dataset")
        if coding.text_hash(prompt) != example["prompt_sha256"]:
            raise ValueError("Corpus prompt checksum mismatch")
        boundary = int(np.flatnonzero(objective.mask[index])[0])
        prompt_ids = objective.tokens[index, :boundary].tolist()
        if tokenizer.encode(prompt, add_special_tokens=False).ids != prompt_ids:
            raise ValueError("Standalone prompt tokenization differs from training")
        if len(prompt_ids) + max_new_tokens > config.context:
            raise ValueError("Generation exceeds model context; no truncation")
        cases.append(
            {
                "id": example["id"],
                "task_id": row["task_id"],
                "prompt": prompt,
                "prompt_token_ids": prompt_ids,
                "reference": row["code"],
                "setup": row["test_setup_code"],
                "tests": row["test_list"],
            }
        )
    return cases


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def render_report(record):
    esc = html.escape
    sections = []
    for case in record["cases"]:
        columns = []
        for label in ("original", "optimized"):
            result = record["results"][label][case["id"]]
            status = result.get("check", {}).get("status", "not_checked")
            text = result.get("selected_text", result["text"])
            candidate_count = len(result.get("candidates", [result]))
            selected = result.get("selected_candidate")
            selection = (
                f"; selected candidate {selected}" if selected is not None else ""
            )
            columns.append(
                f"<div><h3>{esc(label.title())}: {esc(status)}</h3>"
                f"<p>{candidate_count} candidates{selection}</p>"
                f"<pre>{esc(text)}</pre></div>"
            )
        sections.append(
            f"<section><h2>{esc(case['id'])}</h2><pre>{esc(case['prompt'])}</pre>"
            f'<div class="pair">{"".join(columns)}</div></section>'
        )
    return (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'none'; style-src 'unsafe-inline'\">"
        "<title>FLAME generation comparison</title><style>"
        "body{font:15px system-ui;margin:24px auto;padding:0 20px;max-width:1400px;color:#172121;background:#fff}"
        "h1{font-size:26px}h2{font-size:20px}h3{font-size:16px}section{border-top:1px solid #ccd4d4;padding:20px 0}"
        ".pair{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px;line-height:1.5}"
        "@media(max-width:700px){.pair{grid-template-columns:1fr}}"
        "</style><body><h1>FLAME generation comparison</h1>"
        f"<p>{len(record['cases'])}-problem TRAIN sanity check. Public tests are in the prompts. "
        "Candidate zero is greedy; nearby first-token candidates are completed with "
        "the same model and budget. No reference tokens are fed to the model. "
        "The checker selects a passing raw candidate without repair. This is not "
        "hidden-test accuracy.</p>" + "".join(sections) + "</body></html>"
    )


def run(original, run_directory, output, *, count=8, max_new_tokens=128, candidates=4):
    import ennx.ennx_rust as extension
    from tokenizers import Tokenizer

    from ennx.experimental import MetalFlameEvaluator

    from .codecheck import check_solution

    if not hasattr(MetalFlameEvaluator, "next_logits"):
        raise ValueError("Rebuild the Metal extension with next_logits support")
    record_path = run_directory / "run.json"
    training = json.loads(record_path.read_text())
    if training.get("status") != "complete" or training.get("backend") != "metal":
        raise ValueError("Generation requires a completed Metal BO run")
    original_hash = coding.sha256((original / "manifest.json").read_bytes())
    if original_hash != training["source_manifest_sha256"]:
        raise ValueError("Original checkpoint does not match the BO run")
    saved = json.loads((run_directory / "best/manifest.json").read_text())
    optimization = saved.get("optimization")
    if (
        not isinstance(optimization, dict)
        or not optimization
        or any(
            key != "events" and (key not in training or training[key] != value)
            for key, value in optimization.items()
        )
        or not {"final_minibatch", "base_version", "evaluations"} <= optimization.keys()
    ):
        raise ValueError(
            "Saved checkpoint optimization metadata differs from the BO run"
        )
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 512:
        raise ValueError("max_new_tokens must be an integer in [1, 512]")
    if type(candidates) is not int or not 1 <= candidates <= 16:
        raise ValueError("candidates must be an integer in [1, 16]")
    if type(count) is not int or not 1 <= count <= len(training["tokens"]["examples"]):
        raise ValueError("Count must select a nonempty prefix of the corpus")
    if output.exists():
        raise FileExistsError("Refusing to overwrite generation output")
    inputs = output.parent / "generation-inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    tokenizer_bytes = artifact(
        inputs, "tokenizer.json", coding.TOKENIZER_URL, coding.TOKENIZER_SHA256
    )
    dataset_bytes = artifact(
        inputs, "mbpp.jsonl", coding.DATASET_URL, coding.DATASET_SHA256
    )
    tokenizer = Tokenizer.from_str(tokenizer_bytes.decode("utf-8"))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    vocab = tokenizer.get_vocab_size()
    if vocab != 50277 or tokenizer.token_to_id("<|endoftext|>") != coding.EOS_ID:
        raise ValueError("Tokenizer vocabulary/EOS mismatch")
    config = Config()
    cases = cases_for(
        training["tokens"],
        tokenizer,
        [json.loads(line) for line in dataset_bytes.splitlines()],
        count,
        max_new_tokens,
        config,
    )
    layout = Layout(
        tuple(
            Block(
                name=b["name"],
                key=b["key"],
                offset=b["offset"],
                shape=tuple(b["shape"]),
                scale=b["scale"],
                weight=b["weight"],
            )
            for b in training["blocks"]
        )
    )
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "status": "running",
        "scope": "Fixed TRAIN prefix sanity check; public tests in prompts, not hidden-test accuracy",
        "backend": "metal",
        "device": metal.metal_device(),
        "parameters": layout.size,
        "tokenizers_version": version("tokenizers"),
        "numpy_version": np.__version__,
        "selection": "first_n_corpus_problems_before_generation",
        "decoding": "top_first_token_candidates_then_greedy_full_prefix_recompute",
        "first_candidates": candidates,
        "max_new_tokens": max_new_tokens,
        "tokenizer_vocab": vocab,
        "padded_model_vocab": config.vocab,
        "padded_token_ids_excluded": True,
        "run_sha256": coding.sha256(record_path.read_bytes()),
        "bo_extension_sha256": training["extension_sha256"],
        "generation_extension_sha256": coding.sha256(
            Path(extension.__file__).read_bytes()
        ),
        "generation_source_sha256": coding.sha256(Path(__file__).read_bytes()),
        "tokenizer_sha256": coding.TOKENIZER_SHA256,
        "dataset_sha256": coding.DATASET_SHA256,
        "cases": cases,
        "results": {"original": {}, "optimized": {}},
        "checkpoint_manifest_sha256": {},
        "reference_checks": {},
        "next_logits_parity": {},
        "parity_problem_forwards": 0,
        "reload_problem_forwards": 0,
        "problem_forwards": 0,
        "forward_counts_complete": False,
    }
    report = output / "generation.json"
    write_json(report, result)
    try:
        for case in cases:
            result["reference_checks"][case["id"]] = check_solution(
                case["reference"], case["setup"], case["tests"]
            )
        write_json(report, result)
        if any(
            check["status"] != "passed" for check in result["reference_checks"].values()
        ):
            raise RuntimeError(
                "Reference checks failed; no model generation was started"
            )

        def on_forward():
            result["problem_forwards"] += 1

        max_tokens = (
            max(len(case["prompt_token_ids"]) for case in cases) + max_new_tokens
        )
        for label, directory in (
            ("original", original),
            ("optimized", run_directory / "best"),
        ):
            loaded_config, params = load_checkpoint(directory)
            if loaded_config != config:
                raise ValueError("Checkpoint architecture mismatch")
            flat = layout.flatten_torch(params)
            del params
            gc.collect()
            weights = metal.upload_weights(flat)
            del flat
            gc.collect()
            engine = MetalFlameEvaluator(asdict(config), max_tokens=max_tokens)
            if label == "optimized":
                check = training["final_minibatch"]
                examples = [training["tokens"]["examples"][i] for i in check["indices"]]
                # Reload comparison is independent of the generation sample.
                reload_engine = MetalFlameEvaluator(
                    asdict(config),
                    max_tokens=max(len(example["tokens"]) for example in examples),
                )
                losses = reload_engine.losses(
                    weights,
                    [e["tokens"] for e in examples],
                    [e["loss_mask"] for e in examples],
                )
                result["reload_problem_forwards"] += len(examples)
                np.testing.assert_array_equal(losses, check["losses"])
                del reload_engine
            probe = cases[0]["prompt_token_ids"]
            next_row = engine.next_logits(weights, probe)
            result["parity_problem_forwards"] += 1
            full_row = engine.logits(weights, probe)[-1]
            result["parity_problem_forwards"] += 1
            np.testing.assert_array_equal(next_row, full_row)
            result["next_logits_parity"][label] = "exact_full_logits_last_row"
            result["checkpoint_manifest_sha256"][label] = coding.sha256(
                (directory / "manifest.json").read_bytes()
            )
            for case in cases:
                candidate_results = first_candidates(
                    engine,
                    weights,
                    case["prompt_token_ids"],
                    vocab=vocab,
                    eos_id=coding.EOS_ID,
                    max_new_tokens=max_new_tokens,
                    context=config.context,
                    candidates=candidates,
                    on_forward=on_forward,
                )
                for candidate in candidate_results:
                    text_ids = candidate["token_ids"]
                    if candidate["stop_reason"] == "eos":
                        text_ids = text_ids[:-1]
                    candidate["text"] = tokenizer.decode(
                        text_ids, skip_special_tokens=False
                    )
                    candidate["check"] = {"status": "not_checked"}
                completion = candidate_results[0].copy()
                completion["candidates"] = candidate_results
                completion["problem_forwards"] = sum(
                    candidate["problem_forwards"] for candidate in candidate_results
                )
                completion["seconds"] = sum(
                    candidate["seconds"] for candidate in candidate_results
                )
                result["results"][label][case["id"]] = completion
                write_json(report, result)
                click.echo(
                    f"{label} {case['id']}: {len(candidate_results)} candidates, "
                    f"{completion['problem_forwards']} forwards, "
                    f"{completion['stop_reason']}, {completion['check']['status']}"
                )
            del engine, weights
            gc.collect()
        result["status"] = "complete"
        assert result["problem_forwards"] == sum(
            value["problem_forwards"]
            for values in result["results"].values()
            for value in values.values()
        )
        result["generation_seconds"] = sum(
            value["seconds"]
            for values in result["results"].values()
            for value in values.values()
        )
        result["reference_checks_valid"] = all(
            check["status"] == "passed" for check in result["reference_checks"].values()
        )
        result["test_passes"] = None
        result["forward_counts_complete"] = True
        write_json(report, result)
        (output / "report.html").write_text(render_report(result))
    except Exception as error:
        result.update(status="failed", error=str(error))
        write_json(report, result)
        raise
    return result


def check_outputs(output):
    """Explicit post-generation scoring after inspection; never runs a model."""
    from .codecheck import check_solution

    report = output / "generation.json"
    result = json.loads(report.read_text())
    if result.get("status") != "complete" or not result.get("reference_checks_valid"):
        raise ValueError(
            "Scoring requires complete generation and passing reference checks"
        )
    if result.get("test_passes") is not None:
        raise ValueError("These completions have already been scored")
    for label, values in result["results"].items():
        for case in result["cases"]:
            completion = values[case["id"]]
            candidates = completion.get("candidates", [completion])
            for index, candidate in enumerate(candidates):
                candidate["check"] = check_solution(
                    candidate["text"], case["setup"], case["tests"]
                )
                click.echo(
                    f"{label} {case['id']} candidate {index}: "
                    f"{candidate['check']['status']}"
                )
                write_json(report, result)
            selected = next(
                (
                    (index, candidate)
                    for index, candidate in enumerate(candidates)
                    if candidate["check"]["status"] == "passed"
                ),
                (0, candidates[0]),
            )
            completion["selected_candidate"] = selected[0]
            completion["selected_text"] = selected[1]["text"]
            completion["check"] = selected[1]["check"]
            write_json(report, result)
    result["test_passes"] = {
        label: sum(value["check"]["status"] == "passed" for value in values.values())
        for label, values in result["results"].items()
    }
    result["all_completions_checked"] = all(
        value["check"].get("isolation_verified") is True
        for values in result["results"].values()
        for value in values.values()
    )
    write_json(report, result)
    (output / "report.html").write_text(render_report(result))
    return result


@click.group(help=__doc__)
def main():
    pass


@main.command("run")
@click.argument(
    "original", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.argument(
    "run_directory", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option(
    "--output", required=True, type=click.Path(file_okay=False, path_type=Path)
)
@click.option("--count", default=8, type=click.IntRange(1, 312), show_default=True)
@click.option(
    "--max-new-tokens", default=128, type=click.IntRange(1, 512), show_default=True
)
@click.option(
    "--candidates",
    default=4,
    type=click.IntRange(1, 16),
    show_default=True,
    help="Distinct first-token paths to complete; candidate 0 remains greedy.",
)
def run_command(original, run_directory, output, count, max_new_tokens, candidates):
    run(
        original,
        run_directory,
        output,
        count=count,
        max_new_tokens=max_new_tokens,
        candidates=candidates,
    )


@main.command(
    "check", help="Run inspected completions in OS confinement; no model forwards."
)
@click.argument("output", type=click.Path(exists=True, file_okay=False, path_type=Path))
def check_command(output):
    check_outputs(output)


if __name__ == "__main__":
    main()
