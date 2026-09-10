"""Benchmark million-weight CUDA-resident BF16 TuRBO proposals."""

from __future__ import annotations

import statistics
import time

import jax
import jax.numpy as jnp

from ennx.experimental import ParamBlock, turbo_enn


def main() -> None:
    dimensions = 1_000_000
    base = jax.device_put(jnp.zeros(dimensions, dtype=jnp.bfloat16))
    base.block_until_ready()
    search = turbo_enn(
        base,
        0.0,
        [ParamBlock(71, 0, dimensions, 0.01, 1.0 / dimensions)],
        8,
    )
    search.profile(True)
    elapsed = []
    profiles = []
    for round_index in range(18):
        started = time.perf_counter()
        proposals = search.ask(
            1,
            8,
            8,
            round_index,
            acquisition="thompson",
            draw_seed=round_index,
        )
        elapsed.append((time.perf_counter() - started) * 1_000)
        profiles.append(search.last_profile)
        search.tell(proposals, [float(round_index + 1)])

    accepted = search.sync()
    if accepted != [True] or search.best != 18.0 or search.history_len != 8:
        raise RuntimeError("resident rounds did not preserve CUDA search state")

    measured = elapsed[8:]
    kernel = profiles[8:]
    median = statistics.median(measured)
    score = statistics.median(profile[0] for profile in kernel)
    pick = statistics.median(profile[1] for profile in kernel)
    write = statistics.median(profile[2] for profile in kernel)
    total = statistics.median(profile[3] for profile in kernel)
    p95 = max(measured)
    print(
        "CUDA_BF16 resident=true "
        f"dimensions={dimensions} candidates=8 history=8 "
        f"median_ms={median:.3f} p95_ms={p95:.3f} "
        f"score_ms={score:.3f} pick_ms={pick:.3f} "
        f"write_ms={write:.3f} kernel_ms={total:.3f}"
    )
    if median >= 250.0:
        raise RuntimeError(f"BF16 proposal median is too slow: {median:.3f} ms")


if __name__ == "__main__":
    main()
