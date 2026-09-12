"""Correlated full-weight BO for the dense Qwen Coder Metal evaluator."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import struct
import time
from pathlib import Path

import click

from ops.flame.objective import SolutionObjective

from .config import Config
from .metal import Evaluator, proposal_stats

_FILE_CHUNK_BYTES = 16 * 1024 * 1024
_CONTEXT_TARGETS = (4096, 16384, 32768)


def _sample(rng: random.Random, population: int, size: int) -> list[int]:
    return rng.sample(range(population), size)


def _paired(candidate: list[float], incumbent: list[float], population: int) -> dict:
    if len(candidate) != len(incumbent) or len(candidate) < 2:
        raise ValueError("paired losses must have equal size of at least two")
    count = len(candidate)
    candidate_mean = sum(candidate) / count
    incumbent_mean = sum(incumbent) / count
    differences = [old - new for new, old in zip(candidate, incumbent, strict=True)]
    improvement = sum(differences) / count
    correction = (1.0 - count / population) / count

    def variance(values: list[float]) -> float:
        mean = sum(values) / len(values)
        return (
            correction
            * sum((value - mean) ** 2 for value in values)
            / (len(values) - 1)
        )

    improvement_variance = variance(differences)
    improvement_se = math.sqrt(improvement_variance)
    threshold = 2.0 * improvement_se
    return {
        "candidate_loss": candidate_mean,
        "incumbent_loss": incumbent_mean,
        "candidate_variance": variance(candidate),
        "incumbent_variance": variance(incumbent),
        "improvement": improvement,
        "improvement_variance": improvement_variance,
        "improvement_se": improvement_se,
        "threshold": threshold,
        "accepted": improvement > threshold and candidate_mean < incumbent_mean,
        "deteriorated": improvement < -threshold,
    }


def _ckptheader(path: Path) -> dict:
    with path.open("rb") as stream:
        raw_size = stream.read(8)
        if len(raw_size) != 8:
            raise ValueError("Qwen safetensors header is truncated")
        size = struct.unpack("<Q", raw_size)[0]
        if size > 64 * 1024 * 1024:
            raise ValueError("Qwen safetensors header is unreasonably large")
        try:
            return json.loads(stream.read(size))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Qwen safetensors header is invalid") from error


def _sha256file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_FILE_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exportbest(source: Path, output: Path, best) -> None:
    """Write the native BF16 vector back as a loadable safetensors checkpoint."""
    import numpy as np

    if not isinstance(best, np.ndarray) or best.dtype != np.uint16 or best.ndim != 1:
        raise ValueError("Metal search returned an invalid BF16 incumbent")
    header = _ckptheader(source / "model.safetensors")
    records = sorted(
        (name, value)
        for name, value in header.items()
        if name not in {"__metadata__", "lm_head.weight"}
    )
    expected = sum(math.prod(record["shape"]) for _, record in records)
    if expected != best.size:
        raise ValueError(f"Best vector has {best.size} values; expected {expected}")

    new_header = {}
    offset = 0
    for name, record in records:
        length = math.prod(record["shape"])
        new_header[name] = {
            "dtype": "BF16",
            "shape": record["shape"],
            "data_offsets": [offset * 2, (offset + length) * 2],
        }
        offset += length
    encoded = json.dumps(new_header, separators=(",", ":"), sort_keys=True).encode()
    padding = b" " * ((8 - (len(encoded) % 8)) % 8)
    best_bytes = memoryview(best).cast("B")
    weights_path = output / "model.safetensors"
    temporary_weights_path = output / ".model.safetensors.tmp"
    try:
        with temporary_weights_path.open("wb") as stream:
            stream.write(struct.pack("<Q", len(encoded) + len(padding)))
            stream.write(encoded)
            stream.write(padding)
            offset = 0
            for _, record in records:
                length = math.prod(record["shape"])
                start = offset * 2
                end = (offset + length) * 2
                for chunk_start in range(start, end, _FILE_CHUNK_BYTES):
                    stream.write(
                        best_bytes[
                            chunk_start : min(chunk_start + _FILE_CHUNK_BYTES, end)
                        ]
                    )
                offset += length
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_weights_path, weights_path)
    finally:
        temporary_weights_path.unlink(missing_ok=True)

    for name in ("config.json", "merges.txt", "tokenizer_config.json", "vocab.json"):
        shutil.copyfile(source / name, output / name)
    source_manifest = json.loads((source / "manifest.json").read_text())
    manifest = {
        "model_id": source_manifest["model_id"],
        "revision": source_manifest["revision"],
        "config": source_manifest["config"],
        "weights_downloaded": True,
        "optimized_from": str(source),
        "files": {
            "model.safetensors": {
                "bytes": (output / "model.safetensors").stat().st_size,
                "sha256": _sha256file(output / "model.safetensors"),
            }
        },
    }
    manifest_path = output / "manifest.json"
    temporary_manifest_path = output / ".manifest.json.tmp"
    try:
        temporary_manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        os.replace(temporary_manifest_path, manifest_path)
    finally:
        temporary_manifest_path.unlink(missing_ok=True)


def run(
    checkpoint: Path,
    objective_path: Path,
    output: Path,
    *,
    evaluations: int,
    candidates: int,
    history: int,
    minibatch_size: int,
    minibatch_refresh: int,
    seed: int,
    radius: float,
    export_checkpoint: bool = True,
) -> dict:
    run_started = time.perf_counter()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if evaluations < 1 or candidates != 4 or history < 1:
        raise ValueError(
            "evaluations must be positive, candidates must be 4, history positive"
        )
    if minibatch_size < 2:
        raise ValueError("minibatch_size must be at least 2")
    if minibatch_refresh < 1:
        raise ValueError("minibatch_refresh must be positive")

    document = json.loads(objective_path.read_text())
    objective = SolutionObjective.parse(document, Config())
    population = len(objective.tokens)
    if minibatch_size > population:
        raise ValueError("minibatch_size cannot exceed the objective population")
    max_tokens = int(objective.tokens.shape[1])
    evaluator = Evaluator(checkpoint, max_tokens=max_tokens)
    rng = random.Random(seed)
    data_rng = random.Random(seed ^ 0x4D424154)
    indices = _sample(data_rng, population, minibatch_size)

    def rows(selected: list[int]) -> tuple[list[list[int]], list[list[bool]]]:
        return (
            [objective.tokens[index, :].tolist() for index in selected],
            [objective.mask[index, :].tolist() for index in selected],
        )

    token_rows, mask_rows = rows(indices)
    baseline_started = time.perf_counter()
    baseline_losses = evaluator.losses(evaluator.weights, token_rows, mask_rows)
    baseline_seconds = time.perf_counter() - baseline_started
    baseline = _paired(baseline_losses, baseline_losses, population)
    search = evaluator.search(
        -baseline["candidate_loss"],
        capacity=history,
        radius=radius,
        reference_seed=seed,
    )
    perturbation_blocks = [
        {
            "key": int(key),
            "offset": int(offset),
            "elements": int(length),
            "rms_scale": float(scale),
            "distance_weight": float(weight),
        }
        for key, offset, length, scale, weight in evaluator.blocks
    ]
    # Search owns the incumbent row after construction. Keeping the upload
    # alive would retain one extra model-sized Metal buffer for the whole run.
    del evaluator.weights
    search_memory = search.memory_info()
    controller = search.controller_info()
    incumbent_losses = list(baseline_losses)
    output.mkdir(parents=True)
    report = {
        "status": "running",
        "model": json.loads((checkpoint / "manifest.json").read_text()),
        "settings": {
            "evaluations": evaluations,
            "candidates": candidates,
            "history": history,
            "history_policy": "fifo_absolute",
            "weight_controller": "turbo",
            "controller": controller,
            "surrogate_fit_objective": "row_id_loocv_likelihood_fixed",
            "minibatch_size": minibatch_size,
            "minibatch_refresh": minibatch_refresh,
            "seed": seed,
            "reference_seed": seed,
            "radius": radius,
            "sampler": "correlated",
            "perturbation_semantics": "dense_full_tensor_correlated_bf16",
            "backend": "metal",
            "objective_evaluator": "teacher_forced_solution_loss",
            "objective_context_tokens": max_tokens,
            "context_targets": list(_CONTEXT_TARGETS),
            "context_target_reached": max_tokens in _CONTEXT_TARGETS,
            "long_context_loss_path": "cached_causal_attention_bf16_kv_above_256",
            "comparison_baseline": {
                "name": "EGGROLL",
                "method": "hyperscale_evolution_strategies",
                "evaluated_in_run": False,
            },
            "proposal_pipeline": "shared_metal_command_queue",
        },
        "objective": objective.metadata,
        "resources": {
            "model_bf16_elements": evaluator.weights_len,
            "search": search_memory,
            "perturbation_layout": {
                "scale_scheme": "checkpoint_tensor_bf16_rms_fp32",
                "distance_scheme": "equal_tensor_weighted_relative_squared_l2",
                "rounding": "bf16_nearest_even",
                "blocks": perturbation_blocks,
            },
        },
        "baseline": {
            "indices": indices,
            "losses": baseline_losses,
            "mean_loss": baseline["candidate_loss"],
        },
        "objective_evaluations": 1,
        "accepted": 0,
        "events": [],
    }
    round_timings = []
    started = time.perf_counter()
    setup_seconds = started - run_started
    with (output / "events.jsonl").open("w") as events:
        for step in range(evaluations - 1):
            round_started = time.perf_counter()
            evaluations_before = report["objective_evaluations"]
            refresh = (
                minibatch_refresh == 1 or step > 0 and step % minibatch_refresh == 0
            )
            incumbent_seconds = 0.0
            if refresh:
                indices = _sample(data_rng, population, minibatch_size)
                token_rows, mask_rows = rows(indices)
                incumbent = search.incumbent()
                incumbent_start = time.perf_counter()
                incumbent_losses = evaluator.losses(incumbent, token_rows, mask_rows)
                incumbent_seconds = time.perf_counter() - incumbent_start
                incumbent_loss_profile = evaluator.loss_profile
                del incumbent
                report["objective_evaluations"] += 1
            else:
                incumbent_loss_profile = None
            root, draw = rng.getrandbits(64), rng.getrandbits(64)
            candidate_start = time.perf_counter()
            proposals, candidate_losses = evaluator.ask_losses(
                search,
                token_rows,
                mask_rows,
                candidates=candidates,
                neighbors=history,
                seed=root,
                draw_seed=draw,
            )
            candidate_seconds = time.perf_counter() - candidate_start
            candidate_loss_profile = evaluator.loss_profile
            perturbation = proposal_stats(proposals, evaluator.weights_len)
            report["objective_evaluations"] += 1
            decision_started = time.perf_counter()
            paired = _paired(candidate_losses, incumbent_losses, population)
            search.tell_paired(
                proposals,
                -paired["candidate_loss"],
                paired["candidate_variance"],
                -paired["incumbent_loss"],
                paired["incumbent_variance"],
                paired["accepted"],
            )
            decision_seconds = time.perf_counter() - decision_started
            sync_started = time.perf_counter()
            accepted = search.sync()[0]
            sync_seconds = time.perf_counter() - sync_started
            if accepted != paired["accepted"]:
                raise RuntimeError(
                    "Native Qwen acceptance disagrees with paired decision"
                )
            if accepted:
                report["accepted"] += 1
                incumbent_losses = list(candidate_losses)
            event = {
                "step": step,
                "indices": indices,
                "accepted": accepted,
                "candidate_losses": candidate_losses,
                "incumbent_losses": incumbent_losses,
                "candidate_seconds": candidate_seconds,
                "incumbent_seconds": incumbent_seconds,
                "candidate_loss_profile": candidate_loss_profile,
                "incumbent_loss_profile": incumbent_loss_profile,
                "perturbation": perturbation,
                "controller": search.controller_info(),
                "objective_evaluations": report["objective_evaluations"],
                **paired,
            }
            logging_started = time.perf_counter()
            events.write(json.dumps(event, sort_keys=True) + "\n")
            events.flush()
            report["events"].append(event)
            click.echo(
                f"round={step + 1} accepted={str(accepted).lower()} "
                f"loss={paired['candidate_loss']:.7f} "
                f"objective_evaluations={report['objective_evaluations']}"
            )
            round_finished = time.perf_counter()
            logging_seconds = round_finished - logging_started
            round_seconds = round_finished - round_started
            # Nested native profiles may overlap. Only these serial wall-time
            # phases partition the round; all unlabelled host work stays visible.
            round_timings.append(
                {
                    "step": step,
                    "round_seconds": round_seconds,
                    "incumbent_seconds": incumbent_seconds,
                    "proposal_and_score_seconds": candidate_seconds,
                    "decision_submit_seconds": decision_seconds,
                    "decision_sync_seconds": sync_seconds,
                    "logging_seconds": logging_seconds,
                    "other_host_seconds": round_seconds
                    - (
                        incumbent_seconds
                        + candidate_seconds
                        + decision_seconds
                        + sync_seconds
                        + logging_seconds
                    ),
                    "acquisition_candidates": candidates,
                    "selected_candidates_evaluated": 1,
                    "objective_evaluations": report["objective_evaluations"]
                    - evaluations_before,
                    "minibatch_size": len(indices),
                }
            )

    bo_loop_seconds = time.perf_counter() - started
    export_started = time.perf_counter()
    if export_checkpoint:
        _exportbest(checkpoint, output, search.read_best())
    export_seconds = time.perf_counter() - export_started
    report.update(
        {
            "status": "complete",
            "elapsed_seconds": time.perf_counter() - started,
            "best_loss": float(-search.best),
            "best_reward_is_full_objective": False,
            "checkpoint": str(output / "model.safetensors")
            if export_checkpoint
            else None,
            "timing": {
                "setup_seconds": setup_seconds,
                "baseline_seconds": baseline_seconds,
                "baseline_included_in_setup": True,
                "bo_loop_seconds": bo_loop_seconds,
                "export_seconds": export_seconds,
                "rounds": round_timings,
            },
        }
    )
    (output / "run.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("checkpoint", type=click.Path(file_okay=False, path_type=Path))
@click.argument("objective", type=click.Path(dir_okay=False, path_type=Path))
@click.option(
    "--output", required=True, type=click.Path(file_okay=False, path_type=Path)
)
@click.option(
    "--evaluations", default=8, show_default=True, type=click.IntRange(1, 10000)
)
@click.option("--candidates", default=4, show_default=True, type=click.IntRange(4, 4))
@click.option("--history", default=2, show_default=True, type=click.IntRange(1, 128))
@click.option(
    "--minibatch-size", default=2, show_default=True, type=click.IntRange(2, 374)
)
@click.option(
    "--minibatch-refresh", default=1, show_default=True, type=click.IntRange(1, 10000)
)
@click.option("--seed", default=0, show_default=True, type=int)
@click.option("--radius", default=0.01, show_default=True, type=float)
@click.option(
    "--export/--no-export",
    "export_checkpoint",
    default=True,
    show_default=True,
    help="Export best weights as safetensors",
)
def main(
    checkpoint,
    objective,
    output,
    evaluations,
    candidates,
    history,
    minibatch_size,
    minibatch_refresh,
    seed,
    radius,
    export_checkpoint,
):
    """Optimize Qwen weights against a Qwen-tokenized solution objective."""
    try:
        run(
            checkpoint,
            objective,
            output,
            evaluations=evaluations,
            candidates=candidates,
            history=history,
            minibatch_size=minibatch_size,
            minibatch_refresh=minibatch_refresh,
            seed=seed,
            radius=radius,
            export_checkpoint=export_checkpoint,
        )
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        raise click.ClickException(str(error)) from error


if __name__ == "__main__":
    main()
