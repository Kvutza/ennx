"""Run full-weight FLAME BO with resident ENNX search engines."""

from __future__ import annotations

import gc
import hashlib
import inspect
import json
import math
import os
import subprocess
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import click
import numpy as np

from .config import Config
from .layout import Layout
from .minibatch import paired_losses, sample_indices
from .objective import SolutionEvaluator, SolutionObjective

SAMPLERS = ("correlated", "gaussian", "independent")
REJECTION_POLICIES = ("deterioration", "all")
# Incremental workspace estimates, excluding resident rows and the reference.
# Correlated covers the fixed 1x8 fixture's measured ~2234 MiB forward workspace
# plus ~24 MiB native context; this is not a guarantee for arbitrary batches.
FORWARD_HEADROOM_BYTES = {
    "correlated": 9 * 1024**3 // 4,
    "gaussian": 4 * 1024**3,
    "independent": 4 * 1024**3,
}
NOISE_LAWS = {
    "correlated": "correlated_gaussian",
    "gaussian": "independent_gaussian",
    "independent": "legacy_independent_signs",
}


@dataclass(frozen=True)
class Settings:
    evaluations: int = 8
    candidates: int = 4
    history: int = 2
    radius: float = 0.01
    radius_min: float = 0.0001
    radius_max: float = 0.08
    sampler: str = "correlated"
    failure_tolerance: int | None = None
    seed: int = 0
    reference_seed: int = 0
    minibatch_size: int | None = None
    minibatch_seed: int = 0
    minibatch_refresh: int = 1
    acceptance_se: float = 2.0
    rejection_policy: str = "deterioration"
    paired_epistemic_scale: float = 1.0
    backend: str = "jax"

    def __post_init__(self):
        if self.rejection_policy not in REJECTION_POLICIES:
            raise ValueError("rejection_policy must be deterioration or all")
        if self.backend not in ("native", "jax", "metal"):
            raise ValueError("backend must be native, jax, or metal")
        if self.backend in ("native", "metal") and (
            self.sampler != "correlated" or self.minibatch_size is None
        ):
            raise ValueError(
                "Native BO requires correlated sampling and paired minibatches; use --backend jax for legacy objectives"
            )
        if self.sampler not in SAMPLERS:
            raise ValueError("sampler must be correlated, gaussian, or independent")
        for name, low, high in (
            ("evaluations", 2, 1_000_000),
            ("candidates", 1, 8),
            ("history", 2, 128),
            ("seed", 0, 2**64 - 1),
            ("reference_seed", 0, 2**64 - 1),
            ("minibatch_seed", 0, 2**64 - 1),
            ("minibatch_refresh", 1, 1_000_000),
        ):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} must be an integer in [{low}, {high}]")
        if self.sampler == "correlated":
            if self.candidates != 4:
                raise ValueError("correlated sampler requires candidates=4")
            if self.failure_tolerance is not None and self.minibatch_size is None:
                raise ValueError(
                    "failure_tolerance requires paired minibatches or a baseline sampler"
                )
        if self.sampler != "correlated" or self.minibatch_size is not None:
            if self.failure_tolerance is None:
                object.__setattr__(self, "failure_tolerance", 4)
            if (
                type(self.failure_tolerance) is not int
                or not 1 <= self.failure_tolerance <= 2**32 - 1
            ):
                raise ValueError(
                    "failure_tolerance must be an integer in [1, 4294967295]"
                )
        if self.minibatch_size is not None and (
            type(self.minibatch_size) is not int
            or self.minibatch_size < 2
            or self.sampler != "correlated"
        ):
            raise ValueError(
                "minibatch_size requires correlated sampling and at least two problems"
            )
        if self.minibatch_refresh != 1 and self.minibatch_size is None:
            raise ValueError("minibatch_refresh requires minibatch_size")
        if self.backend == "metal" and (
            self.history != 2 or self.failure_tolerance != 4
        ):
            raise ValueError(
                "Metal BO currently requires history=2 and failure_tolerance=4"
            )
        if (
            isinstance(self.acceptance_se, bool)
            or not math.isfinite(self.acceptance_se)
            or self.acceptance_se < 0
        ):
            raise ValueError("acceptance_se must be finite and nonnegative")
        if (
            isinstance(self.paired_epistemic_scale, bool)
            or not math.isfinite(self.paired_epistemic_scale)
            or not float(np.finfo(np.float32).tiny)
            <= self.paired_epistemic_scale
            <= float(np.finfo(np.float32).max)
        ):
            raise ValueError(
                "paired_epistemic_scale must be positive and finite in FP32"
            )
        if not all(
            math.isfinite(value)
            for value in (self.radius, self.radius_min, self.radius_max)
        ):
            raise ValueError("Radii must be finite")
        if (
            not 0 < self.radius_min <= self.radius <= self.radius_max
            or self.radius_min == self.radius_max
        ):
            raise ValueError(
                "Require 0 < radius_min <= radius <= radius_max and distinct bounds"
            )
        smallest, largest = (
            float(np.finfo(np.float32).tiny),
            float(np.finfo(np.float32).max),
        )
        for value in (self.radius_min, self.radius_max):
            if not smallest <= value <= largest:
                raise ValueError(
                    "Radius and metric scaling must be representable in FP32"
                )
        if not smallest <= 1.0 / self.radius**2 <= largest:
            raise ValueError("Radius and metric scaling must be representable in FP32")
        if self.sampler == "correlated":
            low, high = np.float32(self.radius_min), np.float32(self.radius_max)
            # Match the native inward rounding without altering requested settings.
            if float(low) < self.radius_min:
                low = np.nextafter(low, np.float32(np.inf))
            if float(high) > self.radius_max:
                high = np.nextafter(high, np.float32(-np.inf))
            if low >= high:
                raise ValueError(
                    "Correlated radius bounds must contain two distinct FP32 radii"
                )

    @property
    def controller(self):
        if self.sampler == "correlated":
            if self.minibatch_size is not None:
                return (
                    "paired_accepted_radius_with_deterioration_contraction"
                    if self.rejection_policy == "deterioration"
                    else "paired_accepted_radius_with_all_rejection_contraction_legacy"
                )
            return "acquisition_selected_radius"
        return "turbo_success_failure_with_explicit_failure_tolerance"


def memory_budget(size: int, history: int, sampler: str = "correlated") -> int:
    """Resident rows, a dense BF16 reference, and estimated forward workspace."""
    if sampler not in SAMPLERS:
        raise ValueError("sampler must be correlated, gaussian, or independent")
    reference_bytes = 2 * size if sampler == "correlated" else 0
    return (
        (history + 2) * ((size + 127) & ~127) * 2
        + reference_bytes
        + FORWARD_HEADROOM_BYTES[sampler]
    )


def memory_budget(required, allowance, stats):
    if stats is None:
        raise RuntimeError("Compiler memory statistics are required for coding BO")
    workspace = stats.temp_size_in_bytes + stats.output_size_in_bytes
    if workspace < 0:
        raise RuntimeError("Invalid compiler memory statistics")
    return required + max(0, workspace + 64 * 1024**2 - allowance)


def minibatch_rng(seed):
    # Separate problem sampling from the proposal/acquisition random stream.
    return np.random.default_rng(np.random.SeedSequence([seed, 0x4D424154]))


def loss_evaluator(layout, tokens, config):
    import jax
    import jax.numpy as jnp

    from . import model

    if isinstance(tokens, dict):
        return SolutionObjective.parse(tokens, config).evaluator(layout, config)
    tokens = np.asarray(tokens)
    if (
        tokens.ndim != 2
        or not np.issubdtype(tokens.dtype, np.integer)
        or tokens.shape[0] < 1
        or not 2 <= tokens.shape[1] <= config.context
    ):
        raise ValueError(
            "Expected a nonempty batch of unpadded integer token sequences"
        )
    if tokens.min() < 0 or tokens.max() >= config.vocab:
        raise ValueError("Token IDs are outside the model vocabulary")
    tokens = jnp.asarray(tokens, dtype=jnp.int32)

    @jax.jit
    def evaluate(flat):
        # Reshape inside the compiled forward to avoid an eager DLPack input copy.
        flat = flat.reshape((layout.size,))
        logits = model.forward(layout.unflatten(flat), tokens, config)
        return -jnp.mean(model.token_loss(logits, tokens))

    return evaluate


@contextmanager
def borrowed_weights(weights, backend):
    if backend in ("native", "metal"):
        # The native call releases its DLPack lease before returning or raising.
        yield weights
    else:
        import jax

        batch = jax.dlpack.from_dlpack(weights)
        try:
            yield batch
        finally:
            batch.delete()


def optimize(search, layout, evaluate, settings, base_reward, emit, *, baseline=None):
    """Sequential objective evaluations; candidates are ranked by native ENNX."""
    rewards = deque([base_reward], maxlen=settings.history)
    rng = np.random.Generator(np.random.PCG64(settings.seed))
    version = 0
    evaluations = 1
    stop = "evaluation_budget"
    restarts = search.restarts
    correlated = settings.sampler == "correlated"
    minibatch = settings.minibatch_size is not None
    objective_evaluations = 1
    last_accepted_step = None
    if minibatch:
        if baseline is None:
            raise ValueError("Minibatch optimization requires its baseline measurement")
        population = len(evaluate.tokens)
        data_rng = minibatch_rng(settings.minibatch_seed)
        initial_indices = sample_indices(data_rng, population, settings.minibatch_size)
        if initial_indices.tolist() != baseline["indices"]:
            raise ValueError("Baseline minibatch does not match its sampling seed")
    reuse_minibatch = minibatch and settings.minibatch_refresh > 1
    indices = None
    incumbent_losses = None
    if reuse_minibatch:
        # The baseline already measured this exact batch. Reuse it until the
        # configured refresh boundary instead of paying for the same incumbent.
        indices = initial_indices
        incumbent_losses = np.asarray(baseline["losses"], dtype=np.float64).copy()
        if not np.all(np.isfinite(incumbent_losses)):
            raise RuntimeError("Nonfinite baseline loss; refusing to propose")
    if correlated and restarts != 0:
        raise RuntimeError("Correlated search must not use TuRBO restarts")
    if not math.isfinite(base_reward):
        raise ValueError("Baseline reward must be finite")
    if minibatch:
        search.enable_relative(failure_tolerance=settings.failure_tolerance)
    emit(
        {
            "event": "baseline",
            "evaluations": 1,
            "reward": base_reward,
            "base_version": 0,
            "sampler": settings.sampler,
            "controller": settings.controller,
            "reference_seed": settings.reference_seed,
            "reference_version": 0 if correlated else None,
            "reference_radius": float(search.length) if correlated else None,
            "minibatch_refresh": settings.minibatch_refresh if minibatch else None,
            **(
                {"minibatch": baseline, "objective_evaluations": 1} if minibatch else {}
            ),
        }
    )
    for step in range(settings.evaluations - 1):
        root, draw = (int(value) for value in rng.bit_generator.random_raw(2))
        # Minibatch observation variances and outcomes share raw loss units.
        y_scale = 1.0 if minibatch else max(float(np.std(rewards)), 1e-6)
        reference_radius = float(search.length) if correlated else None
        incumbent_seconds = 0.0
        if minibatch:
            refresh = not reuse_minibatch or (
                step > 0 and step % settings.minibatch_refresh == 0
            )
            if refresh:
                indices = sample_indices(data_rng, population, settings.minibatch_size)
                incumbent = search.incumbent()
                try:
                    with borrowed_weights(incumbent, settings.backend) as batch:
                        started = time.perf_counter()
                        incumbent_losses = evaluate.losses(batch, indices)
                        incumbent_seconds = time.perf_counter() - started
                finally:
                    del incumbent
                    gc.collect()
                del batch
                if not np.all(np.isfinite(incumbent_losses)):
                    raise RuntimeError("Nonfinite incumbent loss; refusing to propose")
                objective_evaluations += 1
        started = time.perf_counter()
        proposals = search.ask(
            1,
            settings.candidates,
            settings.history,
            root,
            epistemic_scale=settings.paired_epistemic_scale
            if minibatch
            else 1.0 / settings.radius**2,
            aleatoric_scale=0.0 if minibatch else 0.05,
            y_scale=y_scale,
            acquisition="thompson",
            draw_seed=draw,
        )
        descriptions = proposals.describe()
        geometry = proposals.geometry()
        if len(descriptions) != 1 or len(geometry) != 1:
            raise RuntimeError("Expected diagnostics for exactly one selected proposal")
        seed, score, radius, changes = descriptions[0]
        if not math.isfinite(score) or not math.isfinite(radius) or radius <= 0:
            raise RuntimeError("Invalid native acquisition diagnostics")
        candidate_index, persistence = geometry[0]
        if (
            type(candidate_index) is not int
            or not 0 <= candidate_index < settings.candidates
            or persistence != (0.75 if correlated and candidate_index < 2 else 0.0)
        ):
            raise RuntimeError("Invalid native proposal geometry")
        if len(changes) != len(layout.blocks):
            raise RuntimeError(
                "Native perturbation diagnostics do not match the tensor layout"
            )
        tensors = []
        distance = 0.0
        changed_total = 0
        for block, (changed, squared) in zip(layout.blocks, changes):
            if (
                type(changed) is not int
                or not 0 <= changed <= block.length
                or not math.isfinite(squared)
                or squared < 0
            ):
                raise RuntimeError("Invalid native perturbation diagnostics")
            tensors.append(
                {
                    "name": block.name,
                    "changed": changed,
                    "relative_rms": math.sqrt(squared / block.length) / block.scale,
                }
            )
            distance += squared * block.weight
            changed_total += changed
        ask_seconds = time.perf_counter() - started
        evaluated = distance > 0 and changed_total > 0
        evaluate_seconds = incumbent_seconds
        candidate_seconds = 0.0
        if evaluated:
            with borrowed_weights(proposals, settings.backend) as batch:
                started = time.perf_counter()
                if minibatch:
                    candidate_losses = evaluate.losses(batch, indices)
                    value = -float(np.mean(candidate_losses))
                else:
                    value = float(evaluate(batch).block_until_ready())
                candidate_seconds = time.perf_counter() - started
                evaluate_seconds += candidate_seconds
            del batch
            gc.collect()
            if not math.isfinite(value):
                raise RuntimeError("Nonfinite FLAME loss; refusing to update ENNX")
            evaluations += 1
            objective_evaluations += 1
        else:
            value = float(search.best)
            stop = "selected_proposal_unchanged"
            if minibatch:
                candidate_losses = incumbent_losses.copy()
        started = time.perf_counter()
        if minibatch:
            paired = paired_losses(
                candidate_losses,
                incumbent_losses,
                population,
                acceptance_se=settings.acceptance_se,
            )
            value = -paired.candidate_loss
            reject_is_failure = (
                paired.deteriorated
                if settings.rejection_policy == "deterioration"
                else True
            )
            search.tell_relative(
                proposals,
                value,
                paired.candidate_variance,
                -paired.incumbent_loss,
                paired.incumbent_variance,
                paired.improvement,
                paired.improvement_variance,
                paired.accepted,
                reject_is_failure=reject_is_failure,
            )
        else:
            search.tell(proposals, [value], [0.0])
        accepted = search.sync()[0]
        if minibatch and accepted != paired.accepted:
            raise RuntimeError(
                "Native acceptance disagrees with paired minibatch decision"
            )
        if accepted:
            last_accepted_step = step + 1
            if reuse_minibatch:
                incumbent_losses = candidate_losses.copy()
        tell_seconds = time.perf_counter() - started
        if correlated and search.restarts != 0:
            raise RuntimeError("Correlated search must not use TuRBO restarts")
        emit(
            {
                "event": "proposal",
                "step": step,
                "evaluations": evaluations,
                "evaluated": evaluated,
                "seed_root": root,
                "draw_seed": draw,
                "seed": seed,
                "score": score,
                "radius": radius,
                "sampler": settings.sampler,
                "controller": settings.controller,
                "candidate_index": candidate_index,
                "persistence": persistence,
                "reference_seed": settings.reference_seed,
                "reference_version": version if correlated else None,
                "reference_radius": reference_radius,
                "next_reference_version": version + int(accepted)
                if correlated
                else None,
                "next_reference_radius": float(search.length) if correlated else None,
                "realized_distance": math.sqrt(distance),
                "changed": changed_total,
                "tensors": tensors,
                "base_version": version,
                "accepted": accepted,
                "reward": value,
                "best_reward": float(search.best),
                "next_radius": float(search.length),
                "restarts": search.restarts,
                "history_len": search.history_len,
                "y_scale": y_scale,
                "ask_seconds": ask_seconds,
                "evaluate_seconds": evaluate_seconds,
                "tell_seconds": tell_seconds,
                **(
                    {
                        "outcome": "accepted"
                        if paired.accepted
                        else "deteriorated"
                        if paired.deteriorated
                        else "inconclusive",
                        "counted_failure": not paired.accepted and reject_is_failure,
                        "minibatch": {
                            "indices": indices.tolist(),
                            "candidate_losses": candidate_losses.tolist(),
                            "incumbent_losses": incumbent_losses.tolist(),
                            **asdict(paired),
                        },
                        "objective_evaluations": objective_evaluations,
                        "incumbent_evaluate_seconds": incumbent_seconds,
                        "candidate_evaluate_seconds": candidate_seconds,
                        "minibatch_reused": (
                            reuse_minibatch and incumbent_seconds == 0.0
                        ),
                    }
                    if minibatch
                    else {}
                ),
            }
        )
        version += int(accepted)
        rewards.append(value)
        if not correlated and search.restarts != restarts:
            rewards.clear()
            rewards.append(float(search.best))
            restarts = search.restarts
        if not evaluated:
            break
    return {
        "evaluations": evaluations,
        "best_reward": float(search.best),
        "base_version": version,
        "stop_reason": stop,
        **(
            {
                "objective_evaluations": objective_evaluations,
                "last_accepted_step": last_accepted_step,
                "checkpoint_selection": "paired_minibatch_training",
                "minibatch_refresh": settings.minibatch_refresh,
                "best_reward_is_full_objective": False,
                "final_minibatch": {
                    "indices": indices.tolist(),
                    "losses": (
                        candidate_losses if accepted else incumbent_losses
                    ).tolist(),
                    "variance": paired.candidate_variance
                    if accepted
                    else paired.incumbent_variance,
                },
            }
            if minibatch
            else {}
        ),
    }


def save_best(search, layout, source, output, result):
    import torch
    from safetensors.torch import save_file

    from .checkpoint import digest

    bits = search.read_best()
    weights = torch.from_numpy(bits).view(torch.bfloat16)
    target = output / "best"
    target.mkdir()
    manifest = {
        **source,
        "tensors": {},
        "complete": False,
        "optimization": {**result, "events": "../events.jsonl"},
    }
    for index, block in enumerate(layout.blocks):
        filename = f"{index:03d}.safetensors"
        tensor = weights[block.offset : block.offset + block.length].reshape(
            block.shape
        )
        save_file({block.name: tensor}, str(target / filename))
        manifest["tensors"][block.name] = {
            "file": filename,
            "sha256": digest(target / filename),
        }
    manifest["complete"] = True
    (target / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )


def run(checkpoint: Path, tokens, output: Path, settings: Settings, *, zero_scale=None):
    is_native = settings.backend != "jax"
    is_metal = settings.backend == "metal"
    if is_native and not isinstance(tokens, dict):
        raise ValueError(
            "Native BO supports prepared solution-token corpora only; use --backend jax for legacy token batches"
        )
    solution = (
        SolutionObjective.parse(tokens, Config()) if isinstance(tokens, dict) else None
    )
    objective = (
        solution.metadata
        if solution is not None
        else {"kind": "all_token_cross_entropy"}
    )
    minibatch = settings.minibatch_size is not None
    baseline = None
    if minibatch:
        if (
            not isinstance(tokens, dict)
            or settings.minibatch_size > objective["examples"]
        ):
            raise ValueError(
                "Minibatch BO requires a solution corpus at least as large as its batch"
            )
        objective.update(
            normalization="mean_per_problem_solution_token_loss",
            sampling="uniform_without_replacement_within_batch",
            minibatch_size=settings.minibatch_size,
            acceptance="paired_mean_improvement_exceeds_standard_error_threshold",
            acceptance_se=settings.acceptance_se,
            rejection_policy=settings.rejection_policy,
            minibatch_refresh=settings.minibatch_refresh,
            screening="paired_standard_error_heuristic_not_a_statistical_guarantee",
            uncertainty="finite_population_variance_of_mean_across_problems",
            checkpoint_policy="incumbent_in_gpu_memory_single_final_write",
        )
    if is_metal:
        from . import metal as native

        device_name = native.metal_device()
    elif is_native:
        from . import native

        device_name = native.cuda_device()
    else:
        import jax
        import jax.numpy as jnp

        from . import model

        if os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR") != "platform":
            raise RuntimeError(
                "Run BO in a fresh process with XLA_PYTHON_CLIENT_ALLOCATOR=platform"
            )
        device = jax.devices()[0]
        if device.platform != "gpu" or "T4" not in device.device_kind or device.id != 0:
            raise RuntimeError(
                f"This experiment requires CUDA device 0 to be a T4, got {device}"
            )
        device_name = device.device_kind
    import ennx.ennx_rust as extension

    if is_metal:
        from ennx.experimental import MetalParamBlock as ParamBlock
        from ennx.experimental import MetalSearchState as SearchState
    else:
        from ennx.experimental import ParamBlock, SearchState

    if SearchState is None or not hasattr(SearchState, "read_best"):
        raise RuntimeError("Rebuild the ENNX extension for the selected backend")
    if minibatch and not all(
        hasattr(SearchState, name)
        for name in ("incumbent", "enable_relative", "tell_relative")
    ):
        raise RuntimeError(
            "Rebuild the ENNX extension with paired-relative minibatch support"
        )
    if minibatch:
        try:
            parameters = inspect.signature(SearchState.tell_relative).parameters
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                "Rebuild the ENNX extension with inspectable paired-relative "
                "reject_is_failure keyword support"
            ) from error
        rejection_flag = parameters.get("reject_is_failure")
        if not (
            rejection_flag is not None
            and rejection_flag.kind
            in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        ) and not any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        ):
            raise RuntimeError(
                "Rebuild the ENNX extension with paired-relative "
                "reject_is_failure keyword support"
            )
    size = sum(math.prod(shape) for shape in Config().shapes().values())
    required = (
        native.memory_budget(size, settings.history, 0)
        if is_native
        else memory_budget(size, settings.history, settings.sampler)
        + objective.get("device_input_bytes", 0)
    )
    free = (
        native.free_memory()
        if is_metal
        else (
            int(
                subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--id=0",
                        "--query-gpu=memory.free",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                ).strip()
            )
            * 1024**2
        )
    )
    if required > free:
        raise RuntimeError(
            f"BO needs an estimated {required / 1024**3:.2f} GiB, but only {free / 1024**3:.2f} GiB is free; reduce history"
        )
    if is_native:
        # Allocate bounded engine scratch first, before any checkpoint upload.
        evaluate = native.NativeEvaluator(solution, Config())
        required = native.memory_budget(
            size, settings.history, evaluate.workspace_bytes
        )
        if required > free:
            raise RuntimeError(
                f"Native BO needs an estimated {required / 1024**3:.2f} GiB, "
                f"but only {free / 1024**3:.2f} GiB was free; reduce sequence length or history"
            )
        objective["native_workspace_bytes"] = evaluate.workspace_bytes
        objective["native_context_margin_bytes"] = native.CONTEXT_MARGIN_BYTES
        objective["device_input_bytes"] = (
            0  # Per-sequence inputs are in engine scratch.
        )
    output.mkdir(parents=True, exist_ok=False)
    source = json.loads((checkpoint / "manifest.json").read_text())
    if is_native:
        config, params = native.load_checkpoint(checkpoint)
        layout = Layout.from_torch(params, zero_scale=zero_scale)
        if layout.size != evaluate.weights_len:
            raise RuntimeError("Native weight layout does not match checkpoint")
        flat = layout.flatten_torch(params)
    else:
        config, params = model.load_checkpoint(checkpoint)
        layout = Layout.from_params(params, zero_scale=zero_scale)
        evaluate = loss_evaluator(layout, tokens, config)
        flat = layout.flatten(params).block_until_ready()
    del params
    gc.collect()
    if is_native:
        flat = native.upload_weights(flat)
    base_variance = 0.0
    if minibatch:
        indices = sample_indices(
            minibatch_rng(settings.minibatch_seed),
            objective["examples"],
            settings.minibatch_size,
        )
        losses = evaluate.losses(flat, indices)
        initial = paired_losses(
            losses, losses, objective["examples"], acceptance_se=settings.acceptance_se
        )
        base_reward, base_variance = -initial.candidate_loss, initial.candidate_variance
        baseline = {
            "indices": indices.tolist(),
            "losses": losses.tolist(),
            "variance": base_variance,
            "mean_loss": initial.candidate_loss,
        }
    else:
        base_reward = float(evaluate(flat).block_until_ready())
    if not math.isfinite(base_reward):
        raise RuntimeError("Nonfinite baseline FLAME loss")
    # Compile/autotune the borrowed-batch signature before allocating resident rows.
    if not is_native:
        signature = jax.ShapeDtypeStruct((1, layout.size), jnp.bfloat16)
    if not is_native and isinstance(evaluate, SolutionEvaluator):
        evaluate = evaluate.compile(signature)
        memory = evaluate.memory_analysis()
        required = memory_budget(
            required, FORWARD_HEADROOM_BYTES[settings.sampler], memory
        )
        if required > free:
            raise RuntimeError(
                f"Compiled coding BO needs an estimated {required / 1024**3:.2f} GiB, "
                f"but only {free / 1024**3:.2f} GiB was free; reduce sequence length or history"
            )
        objective["compiled_temporary_bytes"] = memory.temp_size_in_bytes
        objective["compiler_workspace_margin_bytes"] = 64 * 1024**2
    elif not is_native:
        evaluate = evaluate.lower(signature).compile()
    blocks = [
        ParamBlock(b.key, b.offset, b.length, b.scale, b.weight) for b in layout.blocks
    ]
    search = SearchState(
        flat,
        base_reward,
        blocks,
        settings.history,
        max_pending=1,
        length_init=settings.radius,
        length_min=settings.radius_min,
        length_max=settings.radius_max,
        sampler=settings.sampler,
        reference_seed=settings.reference_seed,
        failure_tolerance=None
        if settings.sampler == "correlated"
        else settings.failure_tolerance,
        **({"base_variance": base_variance} if minibatch else {}),
    )
    # Native reference initialization is deferred until ask; free its input first.
    if not is_native:
        flat.delete()
    del flat
    gc.collect()
    if is_native and not is_metal:
        native.release_inputcache()
    metadata = {
        "status": "running",
        "settings": asdict(settings),
        "device": device_name,
        "backend": settings.backend,
        **({} if is_native else {"jax": jax.__version__}),
        "numpy": np.__version__,
        "blocks": layout.describe(),
        "tokens": tokens if isinstance(tokens, dict) else np.asarray(tokens).tolist(),
        "objective": objective,
        "source_revision": source["revision"],
        "source_manifest_sha256": hashlib.sha256(
            (checkpoint / "manifest.json").read_bytes()
        ).hexdigest(),
        "extension_sha256": hashlib.sha256(
            Path(extension.__file__).read_bytes()
        ).hexdigest(),
        "implementation_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(__file__).parent.glob("*.py"))
        },
        "sampler": settings.sampler,
        "noise_law": NOISE_LAWS[settings.sampler],
        "rounding": "bf16_nearest_even",
        "scale_reference": "initial_checkpoint",
        "controller": settings.controller,
        "reference_seed": settings.reference_seed,
        "reference_storage": "dense_bf16" if settings.sampler == "correlated" else None,
        "reference_bytes": 2 * layout.size if settings.sampler == "correlated" else 0,
        "forward_headroom_bytes": (
            evaluate.workspace_bytes + native.CONTEXT_MARGIN_BYTES
            if is_native
            else FORWARD_HEADROOM_BYTES[settings.sampler]
        ),
        "history_policy": "incumbent_anchor_and_rejected_fifo_reset_on_accept"
        if minibatch
        else "resident_fifo",
        "surrogate_observation": "incumbent_relative_paired_improvement"
        if minibatch
        else "absolute_reward",
        "estimated_memory_bytes": required,
    }
    report = output / "run.json"
    report.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    with (output / "events.jsonl").open("w") as stream:

        def emit(event):
            stream.write(json.dumps(event, allow_nan=False) + "\n")
            stream.flush()
            click.echo(
                f"evaluations={event['evaluations']} reward={event['reward']:.7f}"
                + (
                    f" outcome={event['outcome']}"
                    f" counted_failure={str(event['counted_failure']).lower()}"
                    if "outcome" in event
                    else ""
                )
            )

        try:
            result = optimize(
                search,
                layout,
                evaluate,
                settings,
                base_reward,
                emit,
                **({"baseline": baseline} if minibatch else {}),
            )
            save_best(search, layout, source, output, result)
        except Exception as error:
            report.write_text(
                json.dumps(
                    {**metadata, "status": "failed", "error": str(error)}, indent=2
                )
                + "\n"
            )
            raise
    report.write_text(
        json.dumps(
            {**metadata, **result, "status": "complete"}, indent=2, allow_nan=False
        )
        + "\n"
    )
    return result


@click.command(help=__doc__)
@click.argument(
    "checkpoint", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option(
    "--tokens",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Unpadded token batch or a prepared solution-token objective JSON",
)
@click.option(
    "--output", type=click.Path(file_okay=False, path_type=Path), required=True
)
@click.option("--evaluations", type=int, default=8, show_default=True)
@click.option(
    "--backend",
    type=click.Choice(("native", "jax", "metal")),
    default="jax",
    show_default=True,
)
@click.option("--candidates", type=int, default=4, show_default=True)
@click.option(
    "--sampler",
    type=click.Choice(SAMPLERS),
    default="correlated",
    show_default=True,
    help="Correlated Gaussian, independent Gaussian, or legacy independent signs",
)
@click.option("--history", type=int, default=2, show_default=True)
@click.option("--radius", type=float, default=0.01, show_default=True)
@click.option("--radius-min", type=float, default=0.0001, show_default=True)
@click.option("--radius-max", type=float, default=0.08, show_default=True)
@click.option(
    "--failure-tolerance",
    type=int,
    default=None,
    help="Counted failures before radius contraction for paired minibatches; baseline TuRBO failure tolerance otherwise. Defaults to 4",
)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--reference-seed", type=int, default=0, show_default=True)
@click.option(
    "--minibatch-size",
    type=int,
    default=None,
    help=(
        "Problems per paired minibatch; 2 gives four problem forwards per BO step. "
        "Correlated solution objectives only; defaults to 2 for native and metal"
    ),
)
@click.option("--minibatch-seed", type=int, default=0, show_default=True)
@click.option(
    "--minibatch-refresh",
    type=click.IntRange(1, 1_000_000),
    default=1,
    show_default=True,
    help=(
        "Reuse each paired minibatch for this many BO rounds. Values above 1 "
        "reduce incumbent forwards but increase minibatch reuse."
    ),
)
@click.option(
    "--rejection-policy",
    type=click.Choice(REJECTION_POLICIES),
    default="deterioration",
    show_default=True,
    help="Count only paired deterioration as failure, or all rejections for a legacy ablation",
)
@click.option(
    "--acceptance-se",
    type=float,
    default=2.0,
    show_default=True,
    help="Paired improvement screening threshold in estimated standard errors",
)
@click.option(
    "--paired-epistemic-scale",
    type=float,
    default=1.0,
    show_default=True,
    help="Paired surrogate loss variance per unit squared tensor-normalized distance",
)
@click.option(
    "--zero-scale",
    type=float,
    default=None,
    help="Explicit absolute scale for all-zero tensors",
)
def main(checkpoint, tokens, output, zero_scale, **kwargs):
    if kwargs["backend"] in ("native", "metal") and kwargs["minibatch_size"] is None:
        kwargs["minibatch_size"] = 2
    try:
        settings = Settings(**kwargs)
    except ValueError as error:
        raise click.BadParameter(str(error)) from error
    run(
        checkpoint,
        json.loads(tokens.read_text()),
        output,
        settings,
        zero_scale=zero_scale,
    )


if __name__ == "__main__":
    main()
