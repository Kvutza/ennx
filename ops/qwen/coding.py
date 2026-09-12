"""Prepare a fixed MBPP solution-token objective for Qwen Coder."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import click

from .config import MODEL_ID, REVISION, Config
from .tokenizer import load as load_tokenizer

FORMAT = "ennx.solution_tokens.v1"


class BoundaryError(ValueError):
    """A tokenizer token crossed the prompt/solution boundary."""


def _texthash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prompt_for(row: dict) -> str:
    """Use Qwen Coder's FIM protocol with the task as the code prefix."""
    prefix = "# " + row["text"].strip() + "\n"
    return "<|fim_prefix|>" + prefix + "<|fim_suffix|><|fim_middle|>"


def solution_tokens(
    tokenizer, prompt: str, solution: str
) -> tuple[list[int], list[bool]]:
    if not prompt or not solution.strip():
        raise ValueError("Prompt and solution must be nonempty")
    text = prompt + solution
    boundary = len(prompt)
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if not encoded.ids or len(encoded.ids) != len(encoded.offsets):
        raise ValueError("Tokenizer returned missing IDs or offsets")
    mask = []
    previous_start = previous_end = 0
    for start, end in encoded.offsets:
        if not (0 <= start < end <= len(text)):
            raise BoundaryError("Tokenizer returned an invalid offset")
        if start < previous_start or end < previous_end:
            raise BoundaryError("Tokenizer offsets are not monotonic")
        if start < boundary < end:
            raise BoundaryError("Token straddles the prompt/solution boundary")
        mask.append(start >= boundary)
        previous_start, previous_end = start, end
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    prompt_count = sum(not value for value in mask)
    if (
        not prompt_count
        or prompt_count == len(encoded.ids)
        or encoded.ids[:prompt_count] != list(prompt_ids)
        or encoded.offsets[0][0] != 0
        or encoded.offsets[prompt_count - 1][1] != boundary
        or encoded.offsets[prompt_count][0] != boundary
        or encoded.offsets[-1][1] != len(text)
    ):
        raise BoundaryError("Tokenization does not preserve the exact text boundary")
    tokens = list(encoded.ids) + [Config().eos_token_id]
    mask.append(True)
    mask[0] = False
    return tokens, mask


def prepare(rows: list[dict], tokenizer, *, count: int, max_tokens: int) -> dict:
    from ops.flame import coding as mbpp

    if type(count) is not int or not 1 <= count <= len(mbpp.TRAIN_IDS):
        raise ValueError("count must be between 1 and 374")
    if type(max_tokens) is not int or not 2 <= max_tokens <= Config().context:
        raise ValueError("max_tokens must be between 2 and 32768")
    by_id = {}
    for row in rows:
        task_id = row.get("task_id")
        if type(task_id) is not int or task_id not in mbpp.TRAIN_IDS:
            raise ValueError("Only MBPP TRAIN task IDs 601..974 are permitted")
        if task_id in by_id:
            raise ValueError(f"Duplicate MBPP task ID {task_id}")
        by_id[task_id] = row

    examples, excluded = [], []
    for task_id in sorted(by_id):
        row = by_id[task_id]
        prompt = prompt_for(row)
        try:
            tokens, mask = solution_tokens(tokenizer, prompt, row["code"])
        except BoundaryError as error:
            excluded.append({"task_id": task_id, "reason": str(error)})
            continue
        if len(tokens) > max_tokens:
            excluded.append(
                {"task_id": task_id, "reason": "oversize", "tokens": len(tokens)}
            )
            continue
        examples.append(
            {
                "id": f"mbpp/train/{task_id}",
                "task_id": task_id,
                "tokens": tokens,
                "loss_mask": mask,
                "prompt": prompt,
                "solution": row["code"],
                "setup": row["test_setup_code"],
                "tests": row["test_list"],
                "prompt_sha256": _texthash(prompt),
                "reference_sha256": _texthash(row["code"]),
                "prompt_solution_sha256": _texthash(prompt + row["code"]),
                "reference_source": "MBPP full code field (dataset supplied)",
                "execution_verified": False,
            }
        )
        if len(examples) == count:
            break
    if len(examples) < count:
        raise ValueError(
            f"Requested {count} examples, only {len(examples)} are eligible; "
            "no truncation or partial objective was written"
        )
    stats = {
        "candidate_count": len(by_id),
        "example_count": len(examples),
        "excluded_count": len(excluded),
        "total_tokens": sum(len(example["tokens"]) for example in examples),
        "total_scored_tokens": sum(sum(example["loss_mask"]) for example in examples),
    }
    return {
        "format": FORMAT,
        "provenance": {
            "model": {"id": MODEL_ID, "revision": REVISION},
            "dataset": {
                "id": mbpp.DATASET_ID,
                "revision": mbpp.DATASET_REVISION,
                "source_revision": mbpp.DATASET_SOURCE_REVISION,
                "split": "train",
                "train_task_id_range_inclusive": [601, 974],
            },
            "tokenizer": {
                "id": "local Qwen tokenizer",
                "eos_id": Config().eos_token_id,
                "add_special_tokens": False,
            },
            "selection": {
                "rule": "ascending numeric TRAIN task_id, first count eligible; no model scores",
                "selected_task_ids": [example["task_id"] for example in examples],
                "excluded": excluded,
            },
            "prompt_template": "ennx.qwen_fim_code.v1",
            "loss_mask_semantics": "mask[i] scores tokens[i] from tokens[:i]; mask[0]=false",
            "eos_policy": "append exactly one Qwen EOS and score it",
            "statistics": stats,
        },
        "examples": examples,
    }


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--checkpoint", required=True, type=click.Path(file_okay=False, path_type=Path)
)
@click.option(
    "--output", required=True, type=click.Path(dir_okay=False, path_type=Path)
)
@click.option("--count", default=8, show_default=True, type=click.IntRange(1, 374))
@click.option(
    "--max-tokens", default=512, show_default=True, type=click.IntRange(2, 32768)
)
def main(checkpoint: Path, output: Path, count: int, max_tokens: int):
    """Prepare a Qwen-tokenized MBPP TRAIN objective."""
    try:
        from ops.flame import coding as mbpp

        if output.exists():
            raise FileExistsError(f"Refusing to overwrite {output}")
        tokenizer = load_tokenizer(checkpoint)
        result = prepare(
            mbpp.load_train(), tokenizer, count=count, max_tokens=max_tokens
        )
        output.write_text(
            json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
        )
    except (OSError, ValueError, KeyError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(json.dumps(result["provenance"]["statistics"], sort_keys=True), err=True)


if __name__ == "__main__":
    main()
