"""Benchmark the million-parameter CUDA-resident sparse TuRBO proposal path."""

from __future__ import annotations

import statistics
import time

import numpy as np

from ennx.experimental import PackedTurbo


def main() -> None:
    dimensions = 1_000_000
    base = np.full(dimensions // 2, 0x88, dtype=np.uint8)
    leaves = [(0, dimensions, 4, 0.25, 1.0 / dimensions, 0.25)]
    search = PackedTurbo(
        base,
        0.0,
        leaves,
        8,
        device="cuda",
        num_pert=20,
    )
    rng = np.random.default_rng(71)
    elapsed = []
    for round_index in range(18):
        seeds = rng.integers(0, np.iinfo(np.uint64).max, 8, dtype=np.uint64)
        started = time.perf_counter()
        search.ask(seeds, min(8, search.history_len), acquisition="thompson")
        elapsed.append((time.perf_counter() - started) * 1_000)
        search.tell(float(round_index + 1))
    measured = elapsed[8:]
    median = statistics.median(measured)
    p95 = sorted(measured)[-1]
    print(
        "CUDA_SPARSE "
        f"dimensions={dimensions} candidates=8 history=8 num_pert=20 "
        f"median_ms={median:.3f} p95_ms={p95:.3f}"
    )
    if median >= 250.0:
        raise RuntimeError(f"sparse proposal median is too slow: {median:.3f} ms")


if __name__ == "__main__":
    main()
