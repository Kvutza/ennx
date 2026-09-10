import time

import numpy as np
import pandas as pd

from ennx import (
    CandidateRV,
    ENN,
    create_optimizer,
    turbo_zero,
)


def _asktime(d, num_candidates, candidate_rv, seed=0):
    rng = np.random.default_rng(seed)
    bounds = np.tile(np.array([[0.0, 1.0]]), (d, 1))
    opt = create_optimizer(
        bounds=bounds,
        config=turbo_zero(
            num_init=1,
            num_candidates=num_candidates,
            candidate_rv=candidate_rv,
        ),
        rng=rng,
    )
    x_init = opt.ask(num_arms=1)
    opt.tell(x_init, np.zeros((1, 1)))
    t0 = time.perf_counter()
    x = opt.ask(num_arms=1)
    _ = float(np.sum(x))
    return time.perf_counter() - t0


def d_scaling(ds=None, n=1000, num_candidates=5000):
    if ds is None:
        ds = [100, 1000, 5000, 10000]
    print(f"Benchmarking scaling with D (N={n}, num_candidates={num_candidates})\n")

    results = []

    for d in ds:
        print(f"Running D={d}...")
        row = {"D": d}

        rng = np.random.default_rng(0)
        train_x = rng.random((n, d))
        train_y = rng.random((n, 1))
        cand_x = rng.random((num_candidates, d))
        t0 = time.perf_counter()
        model = ENN(train_x, train_y, scale_x=True)
        row["ENN_Init (s)"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        model.rust_backend.nearest_neighbors(
            cand_x,
            10,
            False,
        )
        row["ENN native search (s)"] = time.perf_counter() - t0

        row["Sobol optimizer ask (s)"] = _asktime(
            d,
            num_candidates,
            CandidateRV.SOBOL,
        )

        row["Uniform optimizer ask (s)"] = _asktime(
            d,
            num_candidates,
            CandidateRV.UNIFORM,
        )

        results.append(row)

    df = pd.DataFrame(results)
    print("\n" + df.to_string(index=False))

    print("\nEmpirical scaling (log-log slope vs D):")
    for col in df.columns[1:]:
        y = np.log(df[col].values[-2:])
        x = np.log(df["D"].values[-2:])
        slope = (y[1] - y[0]) / (x[1] - x[0])
        print(f"  {col:20}: {slope:.2f}")


if __name__ == "__main__":
    d_scaling(ds=[100, 1000, 5000, 10000])
