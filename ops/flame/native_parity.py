"""Compare native CUDA FLAME evaluation with the retained JAX reference."""

import gc
import json
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path

import click
import numpy as np

from .config import Config


def expected_logits(params, tokens, config):
    import jax
    import jax.numpy as jnp

    from . import model

    with jax.default_device(jax.devices("cpu")[0]):
        reference = {
            name: jnp.asarray(value.float().numpy(), dtype=jnp.bfloat16)
            for name, value in params.items()
        }
        return np.asarray(model.forward(reference, jnp.asarray([tokens]), config))[0]


def reference_loss(logits, tokens, mask):
    values = logits[:-1].astype(np.float64)
    maximum = values.max(axis=1)
    normalizer = maximum + np.log(np.exp(values - maximum[:, None]).sum(axis=1))
    losses = normalizer - values[np.arange(len(tokens) - 1), tokens[1:]]
    return float(losses[np.asarray(mask[1:])].mean())


def expect_error(action):
    try:
        action()
    except (ValueError, TypeError, OverflowError, BufferError):
        return
    raise AssertionError("Invalid native FLAME input unexpectedly succeeded")


def toy_parity():
    import torch

    from ennx.experimental import FlameEvaluator, ParamBlock, SearchState

    rng = np.random.default_rng(73)
    maximum_error = 0.0
    cases = 0
    for layers, width, heads, experts, top_k in (
        (1, 8, 2, 4, 2),
        (3, 8, 2, 4, 2),
        (2, 16, 4, 3, 3),
        (12, 8, 2, 3, 3),
    ):
        length = 130 if layers == 12 else 9
        config = Config(
            layers=layers,
            width=width,
            heads=heads,
            vocab=19,
            dense_width=13,
            expert_width=5,
            shared_width=7,
            experts=experts,
            top_k=top_k,
            context=160,
            epsilon=1e-5,
            rope_base=257.0,
        )
        params = {
            name: torch.from_numpy(rng.normal(0, 0.15, shape).astype(np.float32)).to(
                torch.bfloat16
            )
            for name, shape in config.shapes().items()
        }
        for name, value in params.items():
            if value.ndim == 1:
                params[name] = value + 1
        engine = FlameEvaluator(asdict(config), max_tokens=length)
        for changed in (False, True):
            if changed:
                for name, value in params.items():
                    if "router.weight" in name:
                        value.zero_()
                    else:
                        value.add_(0.015625)
            flat = torch.cat(
                [params[name].reshape(-1) for name in sorted(params)]
            ).cuda()
            assert engine.weights_len == flat.numel()
            rows = [[1, 2, 3], rng.integers(0, config.vocab, length).tolist()]
            masks = [[False, True, True], [False] * (length - 3) + [True] * 3]
            expected = []
            for row, mask in zip(rows, masks, strict=True):
                reference = expected_logits(params, row, config)
                actual = engine.logits(flat, row)
                maximum_error = max(
                    maximum_error, float(np.max(np.abs(actual - reference)))
                )
                np.testing.assert_allclose(actual, reference, atol=1e-4, rtol=5e-4)
                expected.append(reference_loss(reference, row, mask))
                cases += 1
            losses = engine.losses(flat, rows, masks)
            np.testing.assert_allclose(losses, expected, atol=2e-5, rtol=2e-5)
            if layers == 3 and not changed:
                original = flat.clone()
                offset = 0
                for name in sorted(params):
                    count = params[name].numel()
                    if "router.weight" in name:
                        flat[offset : offset + count].fill_(float("nan"))
                        break
                    offset += count
                expect_error(partial(engine.losses, flat, rows, masks))
                flat.copy_(original)
                np.testing.assert_array_equal(engine.losses(flat, rows, masks), losses)
                del original
            search = SearchState(
                flat,
                -float(losses.mean()),
                [ParamBlock(17, 0, flat.numel(), 1.0)],
                2,
                sampler="correlated",
                reference_seed=3,
                length_init=0.01,
                length_min=0.001,
                length_max=0.1,
            )
            np.testing.assert_array_equal(
                engine.losses(search.incumbent(), rows, masks), losses
            )
            proposals = search.ask(1, 4, 1, 5)
            candidate_losses = engine.losses(proposals, rows, masks)
            search.tell_paired(
                proposals,
                -float(candidate_losses.mean()),
                0.0,
                -float(losses.mean()),
                0.0,
                True,
            )
            assert search.sync() == [True]
            np.testing.assert_array_equal(
                engine.losses(search.incumbent(), rows, masks), candidate_losses
            )
            del search, proposals, flat
            gc.collect()
            torch.cuda.empty_cache()
        del engine
    config = Config(
        layers=1,
        width=8,
        heads=2,
        vocab=19,
        dense_width=13,
        expert_width=5,
        shared_width=7,
        experts=4,
        top_k=2,
        context=16,
    )
    for name, value in (("heads", 0), ("top_k", 5), ("epsilon", 0.0), ("width", True)):
        expect_error(
            lambda name=name, value=value: FlameEvaluator(
                {**asdict(config), name: value}, 9
            )
        )
    engine = FlameEvaluator(asdict(config), 9)
    expect_error(lambda: FlameEvaluator(asdict(config), True))

    class NeverExport:
        def __dlpack_device__(self):
            raise AssertionError(
                "invalid host inputs must fail before borrowing GPU weights"
            )

        def __dlpack__(self, **kwargs):
            raise AssertionError(
                "invalid host inputs must fail before borrowing GPU weights"
            )

    for tokens, masks in (
        ([], []),
        ([[1, 19]], [[False, True]]),
        ([[1, 2]], [[True, True]]),
        ([[1, 2]], [[False, False]]),
    ):
        expect_error(
            lambda tokens=tokens, masks=masks: engine.losses(
                NeverExport(), tokens, masks
            )
        )
    # Rejected exports and CUDA failures must not poison the evaluator or retain
    # a producer lease. The next valid call must still see the current weights.
    original = ((torch.arange(engine.weights_len, device="cuda") % 17).float() / 16).to(
        torch.bfloat16
    )
    flat = original.clone()
    rows, masks = [[1, 2, 3]], [[False, True, True]]
    baseline = engine.losses(flat, rows, masks)

    class WrongDevice(NeverExport):
        def __init__(self, device):
            self.device = device

        def __dlpack_device__(self):
            return self.device

    for device in ((1, 0), (2, 1)):
        bad = WrongDevice(device)
        expect_error(lambda bad=bad: engine.losses(bad, rows, masks))
        expect_error(lambda bad=bad: engine.logits(bad, rows[0]))
        expect_error(
            lambda bad=bad: SearchState(
                bad, 0.0, [ParamBlock(0, 0, engine.weights_len, 1.0)], 2
            )
        )
        np.testing.assert_array_equal(engine.losses(flat, rows, masks), baseline)
    for bad in (flat.cpu(), flat.float(), flat[:-1], flat[::2]):
        expect_error(lambda bad=bad: engine.losses(bad, rows, masks))
        np.testing.assert_array_equal(engine.losses(flat, rows, masks), baseline)
    flat.fill_(float("nan"))
    expect_error(lambda: engine.losses(flat, rows, masks))
    flat.copy_(original)
    np.testing.assert_array_equal(engine.losses(flat, rows, masks), baseline)
    flat.fill_(1)
    # output_layer.weight sorts last; identical large logits still cost log(vocab).
    flat[-config.vocab * config.width :].fill_(1e20)
    np.testing.assert_allclose(
        engine.losses(flat, rows, masks), np.log(config.vocab), atol=2e-6, rtol=2e-6
    )
    flat.copy_(original)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        flat.fill_(0.5)
        asynchronous = engine.losses(flat, rows, masks)
    torch.cuda.synchronize()
    np.testing.assert_array_equal(engine.losses(flat, rows, masks), asynchronous)
    return {
        "toy_cases": cases,
        "max_logit_error": maximum_error,
        "changed_weights": True,
        "routing_ties": True,
        "bo_leases": True,
        "chunk_boundary": True,
        "decimal_layer_order": True,
        "invalid_exports_and_recovery": True,
        "producer_stream": True,
        "large_tied_logits": True,
        "nonfinite_router_recovery": True,
    }


def checkpoint_parity(checkpoint, document):
    import jax
    import torch
    from safetensors.torch import load_file

    from ennx.experimental import FlameEvaluator

    from . import model
    from .checkpoint import digest
    from .layout import Layout
    from .objective import SolutionObjective

    config = Config()
    objective = SolutionObjective.parse(document, config)
    if objective.metadata["examples"] < 2:
        raise ValueError("Checkpoint parity requires at least two prepared problems")
    examples = document["examples"][:2]
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    assert manifest["complete"]
    params = {}
    for name, item in manifest["tensors"].items():
        path = checkpoint / item["file"]
        assert digest(path) == item["sha256"], name
        params.update(load_file(path))
    assert {
        name: tuple(value.shape) for name, value in params.items()
    } == config.shapes()
    flat = torch.cat([params[name].reshape(-1) for name in sorted(params)]).cuda()
    del params
    gc.collect()
    engine = FlameEvaluator(
        asdict(config), max_tokens=objective.metadata["padded_sequence_length"]
    )
    rows = [example["tokens"] for example in examples]
    masks = [example["loss_mask"] for example in examples]
    engine.losses(flat, rows, masks)
    started = time.perf_counter()
    actual = engine.losses(flat, rows, masks)
    seconds = time.perf_counter() - started
    workspace = engine.workspace_bytes
    del engine, flat
    gc.collect()
    torch.cuda.empty_cache()
    _, params = model.load_checkpoint(checkpoint)
    layout = Layout.from_params(params)
    evaluate = objective.evaluator(layout, config)
    reference_flat = layout.flatten(params)
    expected = evaluate.losses(reference_flat, np.array([0, 1]))
    np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=2e-4)
    assert "T4" in jax.devices()[0].device_kind
    return {
        "checkpoint_losses": actual.tolist(),
        "jax_losses": expected.tolist(),
        "native_two_problem_seconds": seconds,
        "workspace_bytes": workspace,
    }


@click.command()
@click.option(
    "--checkpoint", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option("--tokens", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--output", type=click.Path(dir_okay=False, path_type=Path), required=True
)
def main(checkpoint, tokens, output):
    if (checkpoint is None) != (tokens is None):
        raise click.UsageError("--checkpoint and --tokens must be supplied together")
    report = toy_parity()
    if checkpoint is not None:
        report.update(checkpoint_parity(checkpoint, json.loads(tokens.read_text())))
    report["passed"] = True
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
