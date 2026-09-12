"""Benchmark native Metal Qwen generation and print JSON."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import click

from .metal import Evaluator
from .tokenizer import load


REFERENCE_MAX_TOKENS = 256
TARGET_CONTEXT_LENGTHS = (4096, 16384, 32768)


def _fitpromptctx(
    tokens: list[int], target_context: int, max_new_tokens: int
) -> list[int]:
    prefill_tokens = target_context - max_new_tokens
    if prefill_tokens < 1:
        raise ValueError("--context-length must exceed --max-new-tokens")
    if len(tokens) >= prefill_tokens:
        return tokens[:prefill_tokens]
    repeats = prefill_tokens // len(tokens)
    remainder = prefill_tokens % len(tokens)
    return tokens * repeats + tokens[:remainder]


def _refgreedy(evaluator: Evaluator, tokens: list[int], count: int) -> list[int]:
    result = list(tokens)
    for _ in range(count):
        logits = evaluator.next_logits(evaluator.weights, result)
        token = max(range(len(logits)), key=logits.__getitem__)
        result.append(token)
    return result


def _refmeasure(
    evaluator: Evaluator,
    tokens: list[int],
    max_new_tokens: int,
    repeats: int,
) -> tuple[float, list[int]]:
    result = _refgreedy(evaluator, tokens, max_new_tokens)
    start = time.perf_counter()
    for _ in range(repeats):
        result = _refgreedy(evaluator, tokens, max_new_tokens)
    return (time.perf_counter() - start) * 1000.0 / repeats, result


def _mean(rows: list[dict], key: str) -> float:
    return float(statistics.fmean(float(row[key]) for row in rows))


def _sumkerneltimes(rows: list[dict]) -> list[tuple[str, float]]:
    totals: dict[str, float] = {}
    for row in rows:
        for name, milliseconds in row.get("decode_kernel_times_ms", []):
            totals[str(name)] = totals.get(str(name), 0.0) + float(milliseconds)
    return sorted(totals.items(), key=lambda item: item[1], reverse=True)


def _nativebatch(
    evaluator: Evaluator,
    tokens: list[int],
    max_new_tokens: int,
    batch_size: int,
) -> dict:
    prompts = [tokens for _ in range(batch_size)]
    start = time.perf_counter()
    generated = evaluator.generate_batch(evaluator.weights, prompts, max_new_tokens)
    total_ms = (time.perf_counter() - start) * 1000.0
    generated_tokens = sum(len(row) - len(tokens) for row in generated)
    if generated_tokens < batch_size:
        raise RuntimeError("Metal Qwen batch benchmark generated no token")
    return {
        "device_name": "native Metal batch",
        "prompt_tokens": len(tokens),
        "generated_tokens": generated_tokens,
        "requested_generated_tokens": batch_size * max_new_tokens,
        "kv_cache_bytes": 0,
        "logits_read_to_cpu": False,
        "token_selection_on_gpu": True,
        "command_buffer_count": 0,
        "prefill_ms": 0.0,
        "first_token_ms": 0.0,
        "decode_ms_after_first": total_ms,
        "steady_decode_tokens_per_second": generated_tokens / (total_ms / 1000.0)
        if total_ms > 0.0
        else 0.0,
        "total_ms": total_ms,
        "end_to_end_generated_tokens_per_second": generated_tokens / (total_ms / 1000.0)
        if total_ms > 0.0
        else 0.0,
        "host_overhead_ms": 0.0,
        "decode_kernel_trace": ["qwen_batched_prefill", "qwen_batched_decode"],
        "decode_kernel_times_ms": [("qwen_batched_generation_total", total_ms)],
        "decode_bottleneck_candidates": ["qwen_batched_generation_total"],
        "lm_head_argmax_decision": "batched greedy decode keeps token selection on GPU",
        "native_batch_profile": True,
    }


def _benchctx(
    directory: Path,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    target_context: int,
    mode: str,
    batch_size: int,
    repeats: int,
    warmups: int,
    reference: bool,
) -> dict:
    encoded = tokenizer.encode(prompt, add_special_tokens=False)
    tokens = list(encoded.ids)
    if not tokens:
        raise ValueError("Prompt tokenized to an empty sequence")
    tokens = _fitpromptctx(tokens, target_context, max_new_tokens)

    evaluator = Evaluator(directory, max_tokens=target_context)
    for _ in range(warmups):
        if batch_size == 1:
            evaluator.bench_generate(
                evaluator.weights,
                tokens,
                max_new_tokens,
                mode=mode,
            )
        else:
            _nativebatch(evaluator, tokens, max_new_tokens, batch_size)

    profiles = []
    for _ in range(repeats):
        if batch_size == 1:
            streams = [
                evaluator.bench_generate(
                    evaluator.weights,
                    tokens,
                    max_new_tokens,
                    mode=mode,
                )
            ]
        else:
            streams = [
                _nativebatch(
                    evaluator,
                    tokens,
                    max_new_tokens,
                    batch_size,
                )
            ]
        profiles.append(
            {
                **streams[0],
                "streams": streams,
                "generated_tokens": sum(row["generated_tokens"] for row in streams),
                "requested_generated_tokens": sum(
                    row["requested_generated_tokens"] for row in streams
                ),
                "prefill_ms": sum(row["prefill_ms"] for row in streams),
                "first_token_ms": sum(row["first_token_ms"] for row in streams),
                "decode_ms_after_first": sum(
                    row["decode_ms_after_first"] for row in streams
                ),
                "total_ms": sum(row["total_ms"] for row in streams),
                "host_overhead_ms": sum(row["host_overhead_ms"] for row in streams),
                "command_buffer_count": sum(
                    row["command_buffer_count"] for row in streams
                ),
                "kv_cache_bytes": sum(row["kv_cache_bytes"] for row in streams),
                "decode_kernel_times_ms": _sumkerneltimes(streams),
            }
        )
    first = profiles[0]
    kernel_times = _sumkerneltimes(profiles)
    report = {
        "model_name": "Qwen2.5-Coder-1.5B",
        "checkpoint": str(directory),
        "device_name": first["device_name"],
        "target_context_length": target_context,
        "prefill_tokens": first["prompt_tokens"],
        "generated_tokens": first["generated_tokens"],
        "requested_generated_tokens": first["requested_generated_tokens"],
        "batch_size": batch_size,
        "mode": mode,
        "kv_cache_bytes": first["kv_cache_bytes"],
        "logits_read_to_cpu": first["logits_read_to_cpu"],
        "token_selection_on_gpu": first["token_selection_on_gpu"],
        "command_buffer_count": first["command_buffer_count"],
        "prefill_ms": _mean(profiles, "prefill_ms"),
        "first_token_ms": _mean(profiles, "first_token_ms"),
        "steady_decode_ms": _mean(profiles, "decode_ms_after_first"),
        "steady_decode_ms_per_token": (
            _mean(profiles, "decode_ms_after_first")
            / max(1, first["generated_tokens"] - batch_size)
        ),
        "steady_decode_tps": (
            (first["generated_tokens"] - batch_size)
            / (_mean(profiles, "decode_ms_after_first") / 1000.0)
            if _mean(profiles, "decode_ms_after_first") > 0.0
            else 0.0
        ),
        "total_ms": _mean(profiles, "total_ms"),
        "total_tps": (
            first["generated_tokens"] / (_mean(profiles, "total_ms") / 1000.0)
            if _mean(profiles, "total_ms") > 0.0
            else 0.0
        ),
        "host_overhead_ms": _mean(profiles, "host_overhead_ms"),
        "decode_kernel_trace": first["decode_kernel_trace"],
        "decode_kernel_times_ms": kernel_times,
        "decode_bottleneck_candidates": [
            name for name, _milliseconds in kernel_times[:3]
        ],
        "lm_head_argmax_decision": first["lm_head_argmax_decision"],
        "native_tps_path": "rust_metal",
        "python_token_loop": False,
        "native_streams_per_repeat": 1,
        "native_batch_profile": batch_size > 1,
        "repeats": repeats,
        "warmups": warmups,
        "runs": profiles,
        "passed": first["prompt_tokens"] == len(tokens),
    }
    if reference:
        if batch_size != 1:
            raise ValueError("--reference currently supports --batch-size 1")
        if len(tokens) > REFERENCE_MAX_TOKENS:
            raise ValueError(
                "--reference uses next_logits, which is limited to "
                f"{REFERENCE_MAX_TOKENS} tokens; rerun long-context benchmarks "
                "with --no-reference"
            )
        reference_ms, reference_tokens = _refmeasure(
            evaluator,
            tokens,
            max_new_tokens,
            repeats,
        )
        cached_tokens = evaluator.generate(evaluator.weights, tokens, max_new_tokens)
        report.update(
            {
                "reference_ms": reference_ms,
                "reference_tokens_per_second": (len(reference_tokens) - len(tokens))
                / (reference_ms / 1000.0),
                "token_parity": cached_tokens == reference_tokens,
            }
        )
    return report


@click.command()
@click.argument("directory", type=click.Path(file_okay=False, path_type=Path))
@click.option(
    "--mode", default="greedy", show_default=True, type=click.Choice(["greedy"])
)
@click.option(
    "--batch-size", default=1, show_default=True, type=click.IntRange(1, 1024)
)
@click.option("--prompt", required=True)
@click.option(
    "--max-new-tokens", default=8, show_default=True, type=click.IntRange(1, 4096)
)
@click.option(
    "--context-length",
    default=4096,
    show_default=True,
    type=click.Choice(["4096", "16384", "32768", "all"]),
)
@click.option("--repeats", default=3, show_default=True, type=click.IntRange(1, 100))
@click.option("--warmups", default=1, show_default=True, type=click.IntRange(0, 100))
@click.option("--reference/--no-reference", default=False, show_default=True)
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Write the benchmark JSON report to this path as well as stdout.",
)
def main(
    directory: Path,
    mode: str,
    batch_size: int,
    prompt: str,
    max_new_tokens: int,
    context_length: str,
    repeats: int,
    warmups: int,
    reference: bool,
    output: Path | None,
):
    """Measure native Rust/Metal generation and optionally the reference path."""
    try:
        tokenizer = load(directory)
        target_contexts = (
            TARGET_CONTEXT_LENGTHS
            if context_length == "all"
            else (int(context_length),)
        )
        reports = [
            _benchctx(
                directory,
                tokenizer,
                prompt,
                max_new_tokens,
                target_context,
                mode,
                batch_size,
                repeats,
                warmups,
                reference,
            )
            for target_context in target_contexts
        ]
        context_status = {
            str(length): {
                "requested": length in target_contexts,
                "passed": any(
                    report["target_context_length"] == length and report["passed"]
                    for report in reports
                ),
            }
            for length in TARGET_CONTEXT_LENGTHS
        }
        for report in reports:
            report["target_context_status"] = context_status
        report = (
            reports[0]
            if len(reports) == 1
            else {
                "model_name": "Qwen2.5-Coder-1.5B",
                "checkpoint": str(directory),
                "context_sweep": [str(length) for length in TARGET_CONTEXT_LENGTHS],
                "target_context_status": context_status,
                "native_tps_path": "rust_metal",
                "python_token_loop": False,
                "repeats": repeats,
                "warmups": warmups,
                "results": reports,
            }
        )
        payload = json.dumps(report, indent=2, sort_keys=True)
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(payload + "\n")
        click.echo(payload)
    except (OSError, RuntimeError, ValueError) as error:
        raise click.ClickException(str(error)) from error


if __name__ == "__main__":
    main()
