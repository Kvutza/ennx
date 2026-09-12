"""Evaluate greedy Qwen code completions on a fixed MBPP objective."""

from __future__ import annotations

import json
import time
from pathlib import Path

import click

from ops.flame.codecheck import check_solution
from ops.flame.objective import SolutionObjective

from .config import Config
from .metal import Evaluator
from .tokenizer import load

FIM_BOUNDARIES = (
    "<|fim_prefix|>",
    "<|fim_suffix|>",
    "<|fim_middle|>",
    "<|fim_pad|>",
    "<|repo_name|>",
)


def validate_obj(document: dict) -> None:
    """Reject objective artifacts that cannot produce checker feedback."""
    provenance = document.get("provenance")
    if (
        not isinstance(provenance, dict)
        or provenance.get("prompt_template") != "ennx.qwen_fim_code.v1"
    ):
        raise ValueError(
            "Qwen generation requires an FIM objective prepared by ops.qwen"
        )
    examples = document.get("examples")
    if not isinstance(examples, list) or not examples:
        raise ValueError("Qwen generation objective must contain examples")
    for example in examples:
        if not isinstance(example, dict):
            raise ValueError("Qwen generation objective contains an invalid example")
        setup = example.get("setup")
        tests = example.get("tests")
        if (
            not isinstance(setup, str)
            or not isinstance(tests, list)
            or not 1 <= len(tests) <= 256
            or any(not isinstance(test, str) or not test.strip() for test in tests)
        ):
            raise ValueError(
                "Qwen generation objective is missing checker setup/tests; rerun ops.qwen prepare"
            )


def run(
    checkpoint: Path,
    objective_path: Path,
    output: Path,
    *,
    count: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
    samples: int,
) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if not 1 <= max_new_tokens <= Config().context:
        raise ValueError("max_new_tokens must be within the Qwen context")
    if temperature < 0.0 or not 0.0 < top_p <= 1.0 or top_k < 0:
        raise ValueError("invalid sampling settings")
    if samples < 1:
        raise ValueError("samples must be positive")
    document = json.loads(objective_path.read_text())
    validate_obj(document)
    objective = SolutionObjective.parse(document, Config())
    if not 1 <= count <= len(document["examples"]):
        raise ValueError("count must select a nonempty objective prefix")
    tokenizer = load(checkpoint)
    prompts = []
    for index in range(count):
        example = document["examples"][index]
        boundary = next(
            (
                position
                for position, scored in enumerate(example["loss_mask"])
                if scored
            ),
            None,
        )
        if boundary is None or boundary == 0:
            raise ValueError(f"Example {example['id']} has no prompt boundary")
        prompt_ids = objective.tokens[index, :boundary].tolist()
        prompts.append(prompt_ids)
    max_tokens = max(len(prompt) + max_new_tokens for prompt in prompts)
    if max_tokens > Config().context:
        raise ValueError("prompt plus max_new_tokens exceeds the Qwen context")
    evaluator = Evaluator(checkpoint, max_tokens=max_tokens)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "running",
        "model": json.loads((checkpoint / "manifest.json").read_text()),
        "objective": objective.metadata,
        "count": count,
        "max_new_tokens": max_new_tokens,
        "decoding": (
            "native Metal greedy decode"
            if temperature == 0.0
            else "native Metal seeded temperature/top-p decode"
        ),
        "sampling": {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "seed": seed,
            "samples": samples,
        },
        "cases": [],
    }
    started = time.perf_counter()
    for index, (example, prompt_ids) in enumerate(
        zip(document["examples"][:count], prompts, strict=True)
    ):
        samples_result = []
        for sample in range(samples):
            if temperature == 0.0:
                generated = evaluator.generate(
                    evaluator.weights, prompt_ids, max_new_tokens
                )
            else:
                generated = evaluator.sample(
                    evaluator.weights,
                    prompt_ids,
                    max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    seed=(seed + index + sample * 0x9E3779B97F4A7C15) & ((1 << 64) - 1),
                )
            completion_ids = generated[len(prompt_ids) :]
            text = tokenizer.decode(completion_ids, skip_special_tokens=True)
            boundary = min(
                (
                    position
                    for marker in FIM_BOUNDARIES
                    if (position := text.find(marker)) >= 0
                ),
                default=len(text),
            )
            text = text[:boundary]
            check = check_solution(
                text, example.get("setup", ""), example.get("tests", [])
            )
            samples_result.append(
                {
                    "sample": sample,
                    "generated_tokens": len(completion_ids),
                    "text": text,
                    "check": check,
                }
            )
        check = next(
            (
                item["check"]
                for item in samples_result
                if item["check"]["status"] == "passed"
            ),
            samples_result[0]["check"],
        )
        case = {
            "id": example["id"],
            "task_id": example.get("task_id"),
            "prompt_tokens": len(prompt_ids),
            "generated_tokens": samples_result[0]["generated_tokens"],
            "text": samples_result[0]["text"],
            "check": check,
            "samples": samples_result,
        }
        result["cases"].append(case)
        click.echo(
            f"{case['id']}: {check['status']} "
            f"({check['tests_passed']}/{check['tests_total']} tests)"
        )
    result.update(
        {
            "status": "complete",
            "elapsed_seconds": time.perf_counter() - started,
            "passed": sum(
                any(item["check"]["status"] == "passed" for item in case["samples"])
                for case in result["cases"]
            ),
            "total": len(result["cases"]),
        }
    )
    output.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n")
    return result


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("checkpoint", type=click.Path(file_okay=False, path_type=Path))
@click.argument("objective", type=click.Path(dir_okay=False, path_type=Path))
@click.option(
    "--output", required=True, type=click.Path(file_okay=False, path_type=Path)
)
@click.option("--count", default=8, show_default=True, type=click.IntRange(1, 374))
@click.option(
    "--max-new-tokens",
    default=512,
    show_default=True,
    type=click.IntRange(1, Config().context),
)
@click.option("--temperature", default=0.0, show_default=True, type=float)
@click.option("--top-p", default=1.0, show_default=True, type=float)
@click.option(
    "--top-k", default=0, show_default=True, type=click.IntRange(0, Config().vocab)
)
@click.option("--seed", default=0, show_default=True, type=click.IntRange(0, 2**64 - 1))
@click.option("--samples", default=1, show_default=True, type=click.IntRange(1, 16))
def main(
    checkpoint,
    objective,
    output,
    count,
    max_new_tokens,
    temperature,
    top_p,
    top_k,
    seed,
    samples,
):
    """Evaluate untouched or optimized Qwen weights on MBPP."""
    try:
        run(
            checkpoint,
            objective,
            output,
            count=count,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            samples=samples,
        )
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        raise click.ClickException(str(error)) from error


if __name__ == "__main__":
    main()
