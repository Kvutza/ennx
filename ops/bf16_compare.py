"""Compare Gaussian CUDA search overhead against legacy independent signs."""

from __future__ import annotations

import gc
import json
import os
import statistics
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import click
import jax
import jax.numpy as jnp

from ennx.experimental import ParamBlock, turbo_enn


def measure(dimensions, rounds, sampler):
    base = jnp.zeros(dimensions, dtype=jnp.bfloat16).block_until_ready()
    search = turbo_enn(
        base,
        0.0,
        [ParamBlock(71, 0, dimensions, 1.0, 1.0 / dimensions)],
        2,
        sampler=sampler,
        length_init=0.01,
        length_min=0.0001,
        length_max=0.08,
        failure_tolerance=4 if sampler != "correlated" else None,
    )
    base.delete()
    del base
    search.profile(True)
    samples = []
    for step in range(rounds + 2):
        start = time.perf_counter()
        proposals = search.ask(1, 4, 2, step, acquisition="thompson", draw_seed=step)
        ask_ms = 1000 * (time.perf_counter() - start)
        profile = search.last_profile
        start = time.perf_counter()
        # Equal accepted/rejected schedule isolates engine cost from objective quality.
        reward = float(step + 1) if step % 2 == 0 else -1.0
        search.tell(proposals, [reward])
        assert search.sync() == [step % 2 == 0]
        tell_ms = 1000 * (time.perf_counter() - start)
        if step >= 2:
            samples.append(
                {
                    "ask_ms": ask_ms,
                    "tell_ms": tell_ms,
                    "score_ms": profile[0],
                    "materialize_ms": profile[2],
                    "accepted": step % 2 == 0,
                }
            )
    return {
        "sampler": sampler,
        "dimensions": dimensions,
        "candidates": 4,
        "history": 2,
        "rounds": rounds,
        "samples": samples,
        "median_ms": {
            key: statistics.median(row[key] for row in samples)
            for key in ("ask_ms", "tell_ms", "score_ms", "materialize_ms")
        },
    }


@click.command(help=__doc__)
@click.option(
    "--dimensions",
    type=click.IntRange(16, 1_300_024_320),
    default=1_000_000,
    show_default=True,
)
@click.option("--rounds", type=click.IntRange(2, 1000), default=8, show_default=True)
def main(dimensions, rounds):
    device = jax.devices()[0]
    if device.platform != "gpu" or "T4" not in device.device_kind:
        raise click.ClickException(f"Expected a CUDA T4, got {device}")
    if os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR") != "platform":
        raise click.ClickException("Use a fresh process with the platform allocator")
    for sampler in ("gaussian", "correlated", "independent"):
        result = measure(dimensions, rounds, sampler)
        gc.collect()
        click.echo(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
