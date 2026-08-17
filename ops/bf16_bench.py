"""Benchmark million-weight CUDA-resident BF16 TuRBO proposals."""

from __future__ import annotations

import statistics
import time

import jax
import jax.numpy as jnp

from ennx.experimental import Bf16Search


MASK64 = (1 << 64) - 1


def mix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return (value ^ (value >> 31)) & MASK64


def seed_rows(round_index: int) -> list[list[int]]:
    return [[mix64((round_index << 8) | candidate) for candidate in range(8)]]


def main() -> None:
    dimensions = 1_000_000
    base = jax.device_put(jnp.zeros(dimensions, dtype=jnp.bfloat16))
    base.block_until_ready()
    search = Bf16Search(
        base,
        0.0,
        [(71, 0, dimensions, 0.01, 1.0 / dimensions)],
        8,
    )
    search.profile(True)
    elapsed = []
    profiles = []
    for round_index in range(18):
        started = time.perf_counter()
        trials = search.ask_batch(
            seed_rows(round_index),
            min(8, search.history_len),
            acquisition="thompson",
            seed=round_index,
        )
        elapsed.append((time.perf_counter() - started) * 1_000)
        profiles.append(search.last_profile)
        search.tell_batch(trials, [float(round_index + 1)])

    measured = elapsed[8:]
    kernel = profiles[8:]
    median = statistics.median(measured)
    score = statistics.median(profile[0] for profile in kernel)
    pick = statistics.median(profile[1] for profile in kernel)
    write = statistics.median(profile[2] for profile in kernel)
    total = statistics.median(profile[3] for profile in kernel)
    p95 = sorted(measured)[-1]
    print(
        "CUDA_BF16 "
        f"dimensions={dimensions} candidates=8 history=8 "
        f"median_ms={median:.3f} p95_ms={p95:.3f} "
        f"score_ms={score:.3f} pick_ms={pick:.3f} "
        f"write_ms={write:.3f} kernel_ms={total:.3f}"
    )
    if median >= 250.0:
        raise RuntimeError(f"BF16 proposal median is too slow: {median:.3f} ms")


if __name__ == "__main__":
    main()
