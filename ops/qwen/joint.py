"""Joint pilot for Qwen dense weight proposals and generated-code scoring."""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import click

from ops.flame.codecheck import check_solution
from ops.flame.objective import SolutionObjective

from .bo import _exportbest, _paired
from .config import Config
from .eval import validate_obj
from .metal import Evaluator, proposal_stats
from .tokenizer import load

FIM_BOUNDARIES = (
    "<|fim_prefix|>",
    "<|fim_suffix|>",
    "<|fim_middle|>",
    "<|fim_pad|>",
    "<|repo_name|>",
)


@dataclass(frozen=True)
class DecoderPolicy:
    temperature: float
    top_p: float
    top_k: int = 0


@dataclass
class ScoreTiming:
    elapsed_seconds: float = 0.0
    generation_seconds: float = 0.0
    decode_seconds: float = 0.0
    checker_seconds: float = 0.0
    batches: int = 0
    prompt_tokens: int = 0
    generated_tokens: int = 0

    def as_dict(self) -> dict[str, float | int]:
        accounted = self.generation_seconds + self.decode_seconds + self.checker_seconds
        return {
            "elapsed_seconds": self.elapsed_seconds,
            "generation_seconds": self.generation_seconds,
            "decode_seconds": self.decode_seconds,
            "checker_seconds": self.checker_seconds,
            "python_overhead_seconds": max(0.0, self.elapsed_seconds - accounted),
            "batches": self.batches,
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
        }


def _decoderpolicy(point) -> DecoderPolicy:
    if len(point) != 2:
        raise ValueError("decoder policy requires temperature and top_p")
    temperature, top_p = (float(value) for value in point)
    if not 0.0 < temperature <= 1.0 or not 0.0 < top_p <= 1.0:
        raise ValueError("decoder policy is outside the valid sampling domain")
    return DecoderPolicy(temperature, top_p)


def _validatepolicy(policy: DecoderPolicy) -> None:
    if not 0.0 <= policy.temperature <= 1.0:
        raise ValueError("decoder temperature must be in [0, 1]")
    if not 0.0 < policy.top_p <= 1.0:
        raise ValueError("decoder top_p must be in (0, 1]")
    if policy.top_k < 0 or policy.top_k > Config().vocab:
        raise ValueError("decoder top_k exceeds the vocabulary")


def _completiontext(tokenizer, token_ids: list[int]) -> str:
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    boundary = min(
        (position for marker in FIM_BOUNDARIES if (position := text.find(marker)) >= 0),
        default=len(text),
    )
    return text[:boundary]


def _writejson(path: Path, value: dict, *, indent: int | None = None) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=indent, sort_keys=indent is None) + "\n"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _score(
    evaluator: Evaluator,
    tokenizer,
    examples: list[dict],
    prompts: list[list[int]],
    weights,
    indices: list[int],
    policy: DecoderPolicy,
    max_new_tokens: int,
    seed: int,
    samples: int,
    batch_size: int,
    timing: ScoreTiming | None = None,
    initial_generated: list[list[int]] | None = None,
) -> tuple[list[float], list[dict]]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    started = time.perf_counter()
    score_samples = [[] for _ in indices]
    details = [{"id": examples[index]["id"], "samples": []} for index in indices]
    for sample in range(samples):
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            batch_prompts = [prompts[index] for index in batch_indices]
            generation_started = time.perf_counter()
            if initial_generated is not None and sample == 0 and start == 0:
                generated_batch = initial_generated
            elif batch_size == 1:
                generated_batch = [
                    (
                        evaluator.generate(weights, batch_prompts[0], max_new_tokens)
                        if policy.temperature == 0.0
                        else evaluator.sample(
                            weights,
                            batch_prompts[0],
                            max_new_tokens,
                            temperature=policy.temperature,
                            top_p=policy.top_p,
                            top_k=policy.top_k,
                            seed=(seed + batch_indices[0] + sample * 0x9E3779B97F4A7C15)
                            & ((1 << 64) - 1),
                        )
                    )
                ]
            elif policy.temperature == 0.0:
                generated_batch = evaluator.generate_batch(
                    weights, batch_prompts, max_new_tokens
                )
            else:
                generated_batch = evaluator.sample_batch(
                    weights,
                    batch_prompts,
                    max_new_tokens,
                    temperature=policy.temperature,
                    top_p=policy.top_p,
                    top_k=policy.top_k,
                    seeds=[
                        (seed + index + sample * 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
                        for index in batch_indices
                    ],
                )
            if timing is not None:
                timing.generation_seconds += time.perf_counter() - generation_started
                timing.batches += 1
                timing.prompt_tokens += sum(len(prompt) for prompt in batch_prompts)
                timing.generated_tokens += sum(
                    max(0, len(generated) - len(prompt))
                    for generated, prompt in zip(
                        generated_batch, batch_prompts, strict=True
                    )
                )
            for local, (index, generated, prompt) in enumerate(
                zip(batch_indices, generated_batch, batch_prompts)
            ):
                example = examples[index]
                decode_started = time.perf_counter()
                text = _completiontext(tokenizer, generated[len(prompt) :])
                if timing is not None:
                    timing.decode_seconds += time.perf_counter() - decode_started
                checker_started = time.perf_counter()
                check = check_solution(text, example["setup"], example["tests"])
                if timing is not None:
                    timing.checker_seconds += time.perf_counter() - checker_started
                if check["status"] == "sandbox_unavailable":
                    raise RuntimeError("MBPP checker sandbox is unavailable")
                score = check["tests_passed"] / check["tests_total"]
                score_samples[start + local].append(score)
                details[start + local]["samples"].append(
                    {"sample": sample, "score": score, "check": check, "text": text}
                )
    if timing is not None:
        timing.elapsed_seconds = time.perf_counter() - started
    per_problem = [sum(values) / samples for values in score_samples]
    return per_problem, details


def run(
    checkpoint: Path,
    objective_path: Path,
    output: Path,
    *,
    rounds: int,
    tasks: int,
    samples: int,
    max_new_tokens: int,
    seed: int,
    temperature: float,
    top_p: float,
    top_k: int,
    batch_size: int,
) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if not 1 <= rounds or not 2 <= tasks or not 1 <= samples or not 1 <= batch_size:
        raise ValueError(
            "rounds, samples, and batch_size must be positive; tasks must be at least 2"
        )
    run_started = time.perf_counter()
    document = json.loads(objective_path.read_text())
    validate_obj(document)
    objective = SolutionObjective.parse(document, Config())
    examples = document["examples"]
    if tasks > len(examples):
        raise ValueError("tasks cannot exceed the objective population")
    tokenizer = load(checkpoint)
    prompts = []
    for index, example in enumerate(examples):
        boundary = next(
            position for position, scored in enumerate(example["loss_mask"]) if scored
        )
        prompts.append(objective.tokens[index, :boundary].tolist())
    max_tokens = max(len(prompt) + max_new_tokens for prompt in prompts[:tasks])
    if max_tokens > Config().context:
        raise ValueError("prompt plus generation exceeds the Qwen context")
    evaluator = Evaluator(checkpoint, max_tokens=max_tokens)
    base_policy = DecoderPolicy(temperature, top_p, top_k)
    _validatepolicy(base_policy)
    task_indices = list(range(tasks))
    baseline_weights = evaluator.weights
    baseline_started = time.perf_counter()
    baseline_timing = ScoreTiming()
    baseline_scores, baseline_details = _score(
        evaluator,
        tokenizer,
        examples,
        prompts,
        baseline_weights,
        task_indices,
        base_policy,
        max_new_tokens,
        seed,
        samples,
        batch_size,
        timing=baseline_timing,
    )
    baseline_losses = [1.0 - score for score in baseline_scores]
    baseline = _paired(baseline_losses, baseline_losses, len(examples))
    search = evaluator.search(
        -baseline["candidate_loss"], capacity=2, reference_seed=seed
    )
    if hasattr(search, "set_profiling"):
        search.set_profiling(True)
    # The search owns a copy of the base row. Release the baseline handle before
    # correlated reference/proposal rows and the Qwen generation cache are added.
    del baseline_weights
    del evaluator.weights
    search_memory = search.memory_info()
    controller = search.controller_info()
    rng = random.Random(seed)
    output.mkdir(parents=True)
    report = {
        "status": "running",
        "model": json.loads((checkpoint / "manifest.json").read_text()),
        "objective": objective.metadata,
        "resources": {
            "model_bf16_elements": evaluator.weights_len,
            "search": search_memory,
        },
        "settings": {
            "rounds": rounds,
            "tasks": tasks,
            "samples": samples,
            "batch_size": batch_size,
            "max_new_tokens": max_new_tokens,
            "seed": seed,
            "history": 2,
            "history_policy": "fifo_absolute",
            "weight_controller": "turbo",
            "controller": controller,
            "sampler": "correlated_weight_proposals",
            "decoder_policy": base_policy.__dict__,
        },
        "baseline": {
            "policy": base_policy.__dict__,
            "scores": baseline_scores,
            "details": baseline_details,
            "timing": baseline_timing.as_dict(),
        },
        "objective_evaluations": 1,
        "accepted": 0,
        "events_file": "events.jsonl",
    }
    cached_incumbent_scores = baseline_scores
    cached_incumbent_details = baseline_details
    round_timing_totals = {
        "round_seconds": 0.0,
        "proposal_seconds": 0.0,
        "candidate_pipeline_seconds": 0.0,
        "incumbent_score_seconds": 0.0,
        "candidate_score_seconds": 0.0,
        "decision_seconds": 0.0,
        "tell_seconds": 0.0,
        "sync_seconds": 0.0,
        "logging_seconds": 0.0,
    }
    events_io_seconds = 0.0
    bo_loop_started = time.perf_counter()
    with (output / "events.jsonl").open("w") as events:
        for step in range(rounds):
            round_started = time.perf_counter()
            incumbent_timing = None
            decoder_point = None
            policy = base_policy
            # Fixed policies plus fixed seeds make the incumbent score
            # deterministic. Keep it across rounds; only candidates need
            # a new generation pass.
            incumbent_scores = cached_incumbent_scores
            incumbent_details = cached_incumbent_details
            root, draw = rng.getrandbits(64), rng.getrandbits(64)
            proposal_started = time.perf_counter()
            initial_generated = None
            candidate_pipeline_seconds = 0.0
            if policy.temperature == 0.0:
                initial_indices = task_indices[:batch_size]
                initial_prompts = [prompts[index] for index in initial_indices]
                proposals, initial_generated = evaluator.ask_generate(
                    search,
                    initial_prompts,
                    max_new_tokens,
                    seed=root,
                    draw_seed=draw,
                )
                candidate_pipeline_seconds = time.perf_counter() - proposal_started
                proposal_seconds = 0.0
            else:
                proposals = search.ask(
                    1, 4, 2, root, acquisition="thompson", draw_seed=draw
                )
                proposal_seconds = time.perf_counter() - proposal_started
            perturbation = proposal_stats(proposals, evaluator.weights_len)
            native_profile = getattr(search, "last_profile", None)
            candidate_timing = ScoreTiming()
            candidate_started = time.perf_counter()
            candidate_scores, candidate_details = _score(
                evaluator,
                tokenizer,
                examples,
                prompts,
                proposals,
                task_indices,
                policy,
                max_new_tokens,
                seed,
                samples,
                batch_size,
                timing=candidate_timing,
                initial_generated=initial_generated,
            )
            candidate_score_seconds = time.perf_counter() - candidate_started
            candidate_losses = [1.0 - score for score in candidate_scores]
            incumbent_losses = [1.0 - score for score in incumbent_scores]
            decision_started = time.perf_counter()
            paired = _paired(candidate_losses, incumbent_losses, len(examples))
            decision_seconds = time.perf_counter() - decision_started
            tell_started = time.perf_counter()
            search.tell_paired(
                proposals,
                -paired["candidate_loss"],
                paired["candidate_variance"],
                -paired["incumbent_loss"],
                paired["incumbent_variance"],
                paired["accepted"],
            )
            tell_seconds = time.perf_counter() - tell_started
            sync_started = time.perf_counter()
            accepted = search.sync()[0]
            sync_seconds = time.perf_counter() - sync_started
            if accepted != paired["accepted"]:
                raise RuntimeError(
                    "Native Qwen acceptance disagrees with paired decision"
                )
            if accepted:
                cached_incumbent_scores = candidate_scores
                cached_incumbent_details = candidate_details
            candidate_reward = sum(candidate_scores) / len(candidate_scores)
            # The fixed-policy path reuses the incumbent evaluation above.
            report["objective_evaluations"] += 1
            if accepted:
                report["accepted"] += 1
            event = {
                "step": step,
                "decoder_point": (
                    None
                    if decoder_point is None
                    else [float(value) for value in decoder_point]
                ),
                "policy": policy.__dict__,
                "incumbent_scores": incumbent_scores,
                "candidate_scores": candidate_scores,
                "incumbent_details": incumbent_details,
                "candidate_details": candidate_details,
                "accepted": accepted,
                "decoder_reward": candidate_reward,
                "objective_evaluations": report["objective_evaluations"],
                "perturbation": perturbation,
                "controller": search.controller_info(),
                "native_proposal_profile_ms": (
                    None
                    if native_profile is None
                    else {
                        "gpu_pipeline_ms": float(native_profile[0]),
                        "host_selection_ms": float(native_profile[1]),
                        "host_materialize_ms": float(native_profile[2]),
                        "total_ms": float(native_profile[3]),
                    }
                ),
                "timing": {
                    "round_seconds": 0.0,
                    "proposal_seconds": proposal_seconds,
                    "candidate_pipeline_seconds": candidate_pipeline_seconds,
                    "incumbent_score_seconds": (
                        0.0
                        if incumbent_timing is None
                        else incumbent_timing.elapsed_seconds
                    ),
                    "candidate_score_seconds": candidate_score_seconds,
                    "decision_seconds": decision_seconds,
                    "tell_seconds": tell_seconds,
                    "sync_seconds": sync_seconds,
                    "logging_seconds": 0.0,
                    "incumbent_score": (
                        None if incumbent_timing is None else incumbent_timing.as_dict()
                    ),
                    "candidate_score": candidate_timing.as_dict(),
                },
                **paired,
            }
            logging_started = time.perf_counter()
            event["timing"]["round_seconds"] = time.perf_counter() - round_started
            json.dumps(event, sort_keys=True)
            event["timing"]["logging_seconds"] = time.perf_counter() - logging_started
            event["timing"]["round_seconds"] = time.perf_counter() - round_started
            events_io_started = time.perf_counter()
            events.write(json.dumps(event, sort_keys=True) + "\n")
            events.flush()
            events_io_seconds += time.perf_counter() - events_io_started
            for key in round_timing_totals:
                round_timing_totals[key] += event["timing"][key]
            click.echo(
                f"round={step + 1} temperature={policy.temperature:.3f} "
                f"accepted={str(accepted).lower()} "
                f"score={candidate_reward:.4f}"
            )

    bo_loop_seconds = time.perf_counter() - bo_loop_started
    best_score = float(search.best)
    export_started = time.perf_counter()
    _exportbest(checkpoint, output, search.read_best())
    export_seconds = time.perf_counter() - export_started
    report["timing"] = {
        "setup_seconds": baseline_started - run_started,
        "baseline_seconds": baseline_timing.elapsed_seconds,
        "bo_loop_seconds": bo_loop_seconds,
        "export_seconds": export_seconds,
        "total_seconds": time.perf_counter() - run_started,
        "events_io_seconds": events_io_seconds,
        "round_totals": round_timing_totals,
    }
    report.update(
        {
            "status": "complete",
            "elapsed_seconds": report["timing"]["total_seconds"],
            "best_score": best_score,
            "checkpoint": str(output / "model.safetensors"),
        }
    )
    _writejson(output / "run.json", report, indent=2)
    return report


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("checkpoint", type=click.Path(file_okay=False, path_type=Path))
@click.argument("objective", type=click.Path(dir_okay=False, path_type=Path))
@click.option(
    "--output", required=True, type=click.Path(file_okay=False, path_type=Path)
)
@click.option("--rounds", default=5, show_default=True, type=click.IntRange(1, 1000))
@click.option("--tasks", default=2, show_default=True, type=click.IntRange(2, 374))
@click.option("--samples", default=1, show_default=True, type=click.IntRange(1, 16))
@click.option("--batch-size", default=2, show_default=True, type=click.IntRange(1, 32))
@click.option(
    "--max-new-tokens",
    default=128,
    show_default=True,
    type=click.IntRange(1, Config().context),
)
@click.option("--seed", default=0, show_default=True, type=click.IntRange(0, 2**64 - 1))
@click.option("--temperature", default=0.0, show_default=True, type=float)
@click.option("--top-p", default=1.0, show_default=True, type=float)
@click.option(
    "--top-k", default=0, show_default=True, type=click.IntRange(0, Config().vocab)
)
def main(
    checkpoint,
    objective,
    output,
    rounds,
    tasks,
    samples,
    batch_size,
    max_new_tokens,
    seed,
    temperature,
    top_p,
    top_k,
):
    """Run full-weight BO with fixed decoder settings."""
    try:
        run(
            checkpoint,
            objective,
            output,
            rounds=rounds,
            tasks=tasks,
            samples=samples,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            seed=seed,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        raise click.ClickException(str(error)) from error


if __name__ == "__main__":
    main()
