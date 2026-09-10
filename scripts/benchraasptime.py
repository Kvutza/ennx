from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass

import numpy as np

from ennx import CandidateRV, create_optimizer, turbo_zero


@dataclass(frozen=True)
class BenchResult:
    num_candidates: int
    times_s: list[float]
    error: str | None = None


def _parselist(csv: str) -> list[int]:
    parts = [p.strip() for p in csv.split(",") if p.strip()]
    values = [int(p) for p in parts]
    if not values:
        raise ValueError("no candidates provided")
    if any(v <= 0 for v in values):
        raise ValueError(f"all candidates must be > 0, got {values}")
    return values


def _bench(
    *,
    num_dim: int,
    num_candidates: int,
    rep_seed: int,
) -> float:
    rng = np.random.default_rng(rep_seed)
    bounds = np.tile(np.array([[0.0, 1.0]]), (num_dim, 1))
    opt = create_optimizer(
        bounds=bounds,
        config=turbo_zero(
            num_init=1,
            num_candidates=num_candidates,
            candidate_rv=CandidateRV.RAASP,
        ),
        rng=rng,
    )
    x_init = opt.ask(num_arms=1)
    opt.tell(x_init, np.zeros((1, 1)))
    t0 = time.perf_counter()
    x = opt.ask(num_arms=1)
    _ = float(np.sum(x))
    return time.perf_counter() - t0


def _benchrepeats(
    *,
    num_dim: int,
    num_candidates: int,
    repeats: int,
    seed: int,
) -> tuple[list[float], str | None]:
    times_s: list[float] = []
    for rep in range(repeats):
        rep_seed = seed + rep
        try:
            times_s.append(
                _bench(
                    num_dim=num_dim,
                    num_candidates=num_candidates,
                    rep_seed=rep_seed,
                )
            )
        except MemoryError:
            return times_s, "MemoryError"
    return times_s, None


def bench_raasp(
    *,
    num_dim: int,
    num_candidates_list: list[int],
    repeats: int,
    seed: int,
) -> list[BenchResult]:
    if num_dim <= 0:
        raise ValueError(num_dim)
    if repeats <= 0:
        raise ValueError(repeats)

    results: list[BenchResult] = []
    for num_candidates in num_candidates_list:
        times_s, error = _benchrepeats(
            num_dim=num_dim,
            num_candidates=num_candidates,
            repeats=repeats,
            seed=seed,
        )
        results.append(
            BenchResult(num_candidates=num_candidates, times_s=times_s, error=error)
        )
    return results


def _fmtstats(times_s: list[float]) -> str:
    if not times_s:
        return "n/a"
    if len(times_s) == 1:
        return f"{times_s[0]:.4f}s"
    return (
        f"min={min(times_s):.4f}s "
        f"median={statistics.median(times_s):.4f}s "
        f"mean={statistics.mean(times_s):.4f}s "
        f"(n={len(times_s)})"
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Benchmark native RAASP optimizer ask time (generation plus selection)."
    )
    p.add_argument("--num-dim", type=int, default=10_000)
    p.add_argument(
        "--candidates",
        type=str,
        default="50,100,200,500,1000,2000",
        help="Comma-separated list of num_candidates values.",
    )
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    num_candidates_list = _parselist(args.candidates)
    results = bench_raasp(
        num_dim=int(args.num_dim),
        num_candidates_list=num_candidates_list,
        repeats=int(args.repeats),
        seed=int(args.seed),
    )

    print(f"num_dim={args.num_dim} repeats={args.repeats} seed={args.seed}")
    print("num_candidates\tRAASP optimizer ask (s)")
    for r in results:
        if r.error is not None:
            print(f"{r.num_candidates}\t\tERROR: {r.error}")
        else:
            print(f"{r.num_candidates}\t\t{_fmtstats(r.times_s)}")


if __name__ == "__main__":
    main()
