"""Check Metal FLAME against an independent FP32 CPU forward and native T4 losses."""

import gc
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import click
import numpy as np

from .config import Config
from .native_parity import reference_loss


def toy_forward():
    import torch

    from ennx.experimental import MetalFlameEvaluator, MetalWeights

    from .reference import forward

    rng = np.random.default_rng(73)
    maximum_error = 0.0
    cases = 0
    for layers, width, heads, experts, top_k in (
        (1, 8, 2, 4, 2),
        (3, 8, 2, 4, 2),
        (2, 16, 4, 3, 3),
    ):
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
        for value in params.values():
            if value.ndim == 1:
                value.add_(1)
        engine = MetalFlameEvaluator(asdict(config), max_tokens=130)
        for changed in (False, True):
            if changed:
                for name, value in params.items():
                    if "router.weight" in name:
                        value.zero_()
                    else:
                        value.add_(0.015625)
            flat = torch.cat([params[name].reshape(-1) for name in sorted(params)])
            weights = MetalWeights(flat.view(torch.uint16).numpy())
            rows = [
                [1, 2, 3],
                rng.integers(0, config.vocab, 9).tolist(),
                rng.integers(0, config.vocab, 130).tolist(),
            ]
            masks = [[False] * (len(row) - 2) + [True, True] for row in rows]
            expected_losses = []
            for row, mask in zip(rows, masks, strict=True):
                expected = forward(params, torch.tensor([row]), config)[0].numpy()
                actual = engine.logits(weights, row)
                maximum_error = max(
                    maximum_error, float(np.max(abs(actual - expected)))
                )
                np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=5e-4)
                expected_losses.append(reference_loss(expected, row, mask))
                cases += 1
            np.testing.assert_allclose(
                engine.losses(weights, rows, masks),
                expected_losses,
                atol=2e-5,
                rtol=2e-5,
            )
    return {"cases": cases, "maximum_logit_error": maximum_error}


def search_parity():
    from ennx.experimental import MetalParamBlock, MetalSearchState, MetalWeights
    from ops.correlated_parity import candidate, check_rounding, decode, draw_normal

    initial = np.resize(np.asarray([0x3FC0, 0xBFC0], dtype=np.uint16), 513)
    leaves = [(17, 0, 257, 1.0), (29, 257, 256, 0.5)]
    metric = np.concatenate([np.full(257, 1 / 257), np.full(256, 4 / 256)])

    def make(value):
        search = MetalSearchState(
            MetalWeights(initial),
            value,
            [MetalParamBlock(*leaf, float(metric[leaf[1]])) for leaf in leaves],
            2,
            reference_seed=123,
            length_init=0.125,
            length_min=0.03125,
            length_max=0.5,
        )
        search.enable_relative()
        return search

    def ask(search, seed, acquisition):
        return search.ask(
            1, 4, 2, seed, acquisition=acquisition, beta=0.5, draw_seed=seed + 1000
        )

    rounding_cells = 0
    rounds = 0
    for acquisition in ("ucb", "thompson"):
        left, right = make(0.0), make(100.0)
        base = initial.copy()
        history = [(base, 0.0, 0.0)]
        failures = 0
        for step, delta in enumerate([-0.5, -0.25, -0.75, -0.125, 0.5, -0.25]):
            root = 300 + step
            reference = left.read_reference()
            previous_radius = left.length
            a, b = ask(left, root, acquisition), ask(right, root, acquisition)
            assert a.describe() == b.describe() and a.geometry() == b.geometry()
            actual = a.read()
            np.testing.assert_array_equal(actual, b.read())
            seed, score, radius, changes = a.describe()[0]
            index, _ = a.geometry()[0]
            expected_seed, _, _, raw, allowance = candidate(
                base, leaves, reference, root, index, radius
            )
            assert seed == expected_seed
            rounding_cells += check_rounding(actual, raw, allowance)
            decoded = decode(actual).astype(np.float64)
            distances = np.asarray(
                [
                    np.sum(np.square(decoded - decode(row).astype(np.float64)) * metric)
                    for row, _, _ in history
                ]
            )
            weights = 1 / (1e-9 + distances + [variance for _, _, variance in history])
            mean = np.dot(weights, [reward for _, reward, _ in history]) / weights.sum()
            noise = (
                0.5
                if acquisition == "ucb"
                else np.dot(
                    weights,
                    [draw_normal(root + 1000, i + 1) for i in range(len(history))],
                )
                / np.linalg.norm(weights)
            )
            np.testing.assert_allclose(
                score, mean + noise / np.sqrt(weights.sum()), atol=4e-5, rtol=4e-5
            )
            for leaf, (changed, squared) in zip(leaves, changes, strict=True):
                _, offset, length, _ = leaf
                sl = slice(offset, offset + length)
                assert changed == int(np.count_nonzero(actual[sl] != base[sl]))
                np.testing.assert_allclose(
                    squared,
                    np.square(decoded[sl] - decode(base[sl])).sum(),
                    atol=4e-5,
                    rtol=4e-5,
                )
            accept = delta > 0
            for search, proposal, offset in ((left, a, 0), (right, b, 100)):
                search.tell_relative(
                    proposal, offset + delta, 0.5, offset, 0.75, delta, 0.0625, accept
                )
                assert search.sync() == [accept]
            if accept:
                base = actual.copy()
                history = [(base, 0.0, 0.0)]
                failures = 0
                assert left.length == radius
            else:
                history = [history[0], (actual.copy(), delta, 0.0625)]
                failures += 1
                expected_radius = previous_radius
                if failures == 4:
                    expected_radius = max(0.03125, previous_radius / 2)
                    failures = 0
                assert left.length == expected_radius
            assert left.history_len == len(history) and left.restarts == 0
            np.testing.assert_array_equal(left.read_best(), base)
            try:
                a.read()
            except ValueError:
                pass
            else:
                raise AssertionError("A resolved Metal proposal remained readable")
            rounds += 1
        incumbent = left.incumbent()
        try:
            ask(left, 900, acquisition)
        except BufferError:
            pass
        else:
            raise AssertionError("An outstanding incumbent view allowed mutation")
        del incumbent
    return {"rounds": rounds, "adjacent_bf16_rounding_cells": rounding_cells}


def full_forward(checkpoint, corpus):
    from .layout import Layout
    from .metal import NativeEvaluator, upload_weights
    from .native import load_checkpoint
    from .objective import SolutionObjective

    # These exact inputs and losses were archived by the native CUDA T4 run.
    expected_corpus = "78ea8e3b8a7e443fe7918a34c99fde648efcf14ba128c322c6e555700e95c69f"
    if hashlib.sha256(corpus.read_bytes()).hexdigest() != expected_corpus:
        raise ValueError("Full parity requires the pinned 312-problem TRAIN corpus")
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    tensor_hashes = json.dumps(
        {name: record["sha256"] for name, record in manifest["tensors"].items()},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    if (
        hashlib.sha256(tensor_hashes).hexdigest()
        != "3d55c614eeb5c1a06e886380fed2cedc86cd5399360d048f39d7e683c2d060dc"
    ):
        raise ValueError("Full parity requires the original checkpoint tensor hashes")
    config, params = load_checkpoint(checkpoint)
    layout = Layout.from_torch(params)
    flat = layout.flatten_torch(params)
    del params
    gc.collect()
    weights = upload_weights(flat)
    del flat
    gc.collect()
    objective = SolutionObjective.parse(json.loads(corpus.read_text()), config)
    engine = NativeEvaluator(objective, config)
    started = time.perf_counter()
    actual = engine.losses(weights, [96, 21])
    seconds = time.perf_counter() - started
    expected = [1.9969390630722046, 1.8652825355529785]
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)
    return {
        "parameters": layout.size,
        "indices": [96, 21],
        "losses": actual.tolist(),
        "cuda_losses": expected,
        "seconds": seconds,
        "workspace_bytes": engine.workspace_bytes,
        "source_manifest_sha256": hashlib.sha256(
            (checkpoint / "manifest.json").read_bytes()
        ).hexdigest(),
    }


@click.command(help=__doc__)
@click.option(
    "--checkpoint", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option("--corpus", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--output", required=True, type=click.Path(dir_okay=False, path_type=Path)
)
def main(checkpoint, corpus, output):
    if (checkpoint is None) != (corpus is None):
        raise click.UsageError("--checkpoint and --corpus must be supplied together")
    if output.exists():
        raise click.ClickException("Refusing to overwrite an existing parity report")
    from .metal import device_info

    report = {
        "device": device_info(),
        "search": search_parity(),
        "toy_forward": toy_forward(),
    }
    gc.collect()
    if checkpoint is not None:
        report["full_forward"] = full_forward(checkpoint, corpus)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    click.echo(json.dumps(report, allow_nan=False))


if __name__ == "__main__":
    main()
