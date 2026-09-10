from __future__ import annotations

import gc
import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from ennx import ENNStatefulFitter, ENN
from ennx.ennx.eclssup import (
    posterior_neighbors,
    nearest_neighbors,
)
from ennx.ennx.enn_params import ENNParams, PosteriorFlags

pytestmark = pytest.mark.slow

_POSTERIOR_SPEED_BASELINE = (
    Path(__file__).resolve().parent / "fixtures" / "postspdbase.json"
)


def enn_data(num_samples: int, k: int, noise: float, m: int = 1):
    x = np.sort(np.random.rand(num_samples + 4))
    x[-3] = x[-4]
    x[-2] = x[-4]
    x[-1] = x[-4]
    x[1] = x[0] + 0.03
    eps = np.random.randn(num_samples + 4)
    y = np.sin(2 * m * np.pi * x) + noise * eps
    yvar = (noise**2) * np.ones_like(y)
    model = ENN(x[:, None], y[:, None], yvar[:, None])
    rng = np.random.default_rng(0)
    fitter = ENNStatefulFitter(k=k, rng=rng)
    x_all, y_all, yvar_all = model.train_rows(list(range(len(model))))
    fitter.tell(x_all, y_all, yvar_all)
    params = fitter.ask(
        model,
        num_candidates=100,
        num_samples=min(10, num_samples),
    )
    return x, y, model, params


def enn_logic(model, params):
    x_hat = np.linspace(0.0, 1.0, 30)
    x_hat_2d = x_hat[:, None]
    posterior = model.posterior(
        x_hat_2d, params=params, flags=PosteriorFlags(exclude_nearest=False)
    )
    return posterior.mu[:, 0], posterior.se[:, 0]


@pytest.mark.skip(reason="Timing unreliable across machines; run manually to verify")
def test_enndemoperformance():
    np.random.seed(1)
    t0 = time.time()
    _x, _y, model, params = enn_data(num_samples=1_000_000, k=5, noise=0.3, m=3)
    _mu, _se = enn_logic(model, params)
    elapsed = time.time() - t0
    print(f"\nTime taken: {elapsed:.3f} seconds")
    assert elapsed < 0.3, f"Expected < 0.3s, got {elapsed:.3f}s"


_PERF_WARMUP = 1
_PERF_REPS = 1


def _medianseconds(fn, *, reps: int = _PERF_REPS) -> float:
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return float(np.median(samples))


def _medianratio(
    fn_num: Callable[[], None],
    fn_den: Callable[[], None],
    *,
    warmup: int = _PERF_WARMUP,
    reps: int = _PERF_REPS,
) -> float:
    """Median of per-rep num/den ratios with interleaved warmup and alternating order."""
    for i in range(warmup):
        if i % 2 == 0:
            fn_den()
            fn_num()
        else:
            fn_num()
            fn_den()
    ratios: list[float] = []
    for i in range(reps):
        if i % 2 == 0:
            t0 = time.perf_counter()
            fn_num()
            t_num = time.perf_counter() - t0
            t0 = time.perf_counter()
            fn_den()
            t_den = time.perf_counter() - t0
        else:
            t0 = time.perf_counter()
            fn_den()
            t_den = time.perf_counter() - t0
            t0 = time.perf_counter()
            fn_num()
            t_num = time.perf_counter() - t0
        ratios.append(t_num / max(t_den, 1e-12))
    return float(np.median(ratios))


def _neighborratio(n: int, *, d: int = 1, k: int = 10, seed: int = 42) -> float:
    """Wall-clock ratio: index_search (f64 exact) vs FAISS-only neighbor lookup."""
    rng = np.random.default_rng(seed)
    train_x = rng.standard_normal((n, d))
    train_y = rng.normal(0.0, 100.0, size=(n, d))
    model = ENN(train_x, train_y, train_yvar=None)
    rust = model.rust_backend

    def index_neighbors() -> None:
        posterior_neighbors(
            rust,
            train_x,
            search_k=k,
        )

    def faiss_neighbors() -> None:
        nearest_neighbors(
            rust,
            train_x,
            search_k=k,
            exclude_nearest=False,
        )

    for _ in range(3):
        index_neighbors()
        faiss_neighbors()

    t_index = _medianseconds(index_neighbors)
    t_faiss = _medianseconds(faiss_neighbors)
    return t_index / max(t_faiss, 1e-12)


@pytest.mark.parametrize(
    ("n", "max_ratio"),
    [
        (20, 2.5),  # user-reported scenario (n=20, k=10)
        (500, 3.0),  # n_query=n_train; batched f64 matrix should track FAISS
    ],
)
def test_001(n: int, max_ratio: float):
    """index_search (f64 exact) must not regress far beyond FAISS-only lookup."""
    ratio = _neighborratio(n)
    assert ratio <= max_ratio, (
        f"index_search must be <= {max_ratio}x FAISS-only at n={n}, "
        f"got {ratio:.2f}x (slowdown bad)"
    )


def test_002():
    """posterior(train_x) tie-break-on must track tie-break-off at n=1024."""
    rng = np.random.default_rng(0)
    n, d, m = 1024, 10, 5
    x = rng.standard_normal((n, d))
    y = rng.normal(0.0, 100.0, size=(n, m))
    model = ENN(x, y)
    params = ENNParams(k_neighbors=10, epistemic_scale=1.0, aleatoric_scale=0.1)

    def run_off() -> None:
        model.posterior(
            x,
            params=params,
            flags=PosteriorFlags(tie_neighbors=False),
        )

    def run_on() -> None:
        model.posterior(
            x,
            params=params,
            flags=PosteriorFlags(tie_neighbors=True),
        )

    ratio = _medianratio(run_on, run_off, warmup=3, reps=5)
    assert ratio <= 1.22, (
        f"tie-break posterior must be <= 1.22x no-tie-break at n=1024, got {ratio:.2f}x"
    )


def test_003():
    """lattice posterior(x_train) tie-break-on must track tie-break-off at n=1024."""
    n, d = 1024, 10
    x = np.linspace(0.0, 1.0, n)[:, None]
    x = np.tile(x, (1, d))
    y = np.sin(8.0 * np.pi * x[:, 0:1])
    model = ENN(x, y)
    params = ENNParams(k_neighbors=10, epistemic_scale=1.0, aleatoric_scale=0.1)

    def run_off() -> None:
        model.posterior(
            x,
            params=params,
            flags=PosteriorFlags(tie_neighbors=False),
        )

    def run_on() -> None:
        model.posterior(
            x,
            params=params,
            flags=PosteriorFlags(tie_neighbors=True),
        )

    ratio = _medianratio(run_on, run_off)
    assert ratio <= 1.15, (
        f"lattice tie-break posterior must be <= 1.15x no-tie-break at n=1024, "
        f"got {ratio:.2f}x"
    )


def test_004():
    """Ratio should stay bounded as n_query=n_train grows."""
    n_small, n_large = 256, 1024
    ratio_small = _neighborratio(n_small, seed=7)
    ratio_large = _neighborratio(n_large, seed=7)
    assert ratio_large <= max(ratio_small * 1.5, 3.0), (
        f"expected bounded slowdown ratio: n={n_small} -> {ratio_small:.2f}x, "
        f"n={n_large} -> {ratio_large:.2f}x"
    )


def _selfratios(
    scenario: dict,
    *,
    tie_neighbors: bool,
) -> tuple[float, float, float, float, float]:
    """Return (t_post, t_index, t_faiss, index/faiss, posterior/index) medians."""
    rng = np.random.default_rng(int(scenario["seed"]))
    n, d, m, k = scenario["n"], scenario["d"], scenario["m"], scenario["k"]
    x = rng.standard_normal((n, d))
    y = rng.normal(0.0, 100.0, size=(n, m))
    model = ENN(x, y)
    params = ENNParams(k_neighbors=k, epistemic_scale=1.0, aleatoric_scale=0.1)
    flags = PosteriorFlags(tie_neighbors=tie_neighbors)
    rust = model.rust_backend
    warmup = 3
    reps = 5

    def posterior() -> None:
        model.posterior(x, params=params, flags=flags)

    def index_neighbors() -> None:
        posterior_neighbors(
            rust,
            x,
            search_k=k,
            flags=PosteriorFlags(tie_neighbors=tie_neighbors),
        )

    def faiss_neighbors() -> None:
        nearest_neighbors(rust, x, search_k=k, exclude_nearest=False)

    for _ in range(warmup):
        posterior()
        index_neighbors()
        faiss_neighbors()

    t_post = _medianseconds(posterior, reps=reps)
    t_index = _medianseconds(index_neighbors, reps=reps)
    t_faiss = _medianseconds(faiss_neighbors, reps=reps)
    index_vs_faiss = _medianratio(
        index_neighbors, faiss_neighbors, warmup=warmup, reps=reps
    )
    posterior_vs_index = _medianratio(
        posterior, index_neighbors, warmup=warmup, reps=reps
    )
    return t_post, t_index, t_faiss, index_vs_faiss, posterior_vs_index


@pytest.mark.parametrize("tie_neighbors", [False, True])
def test_005(
    tie_neighbors: bool,
):
    """Self-search: index_search vs FAISS and posterior stats overhead stay bounded."""
    gc.collect()
    with open(_POSTERIOR_SPEED_BASELINE) as f:
        baseline = json.load(f)
    for scenario in baseline["scenarios"]:
        t_post, t_index, t_faiss, index_vs_faiss, posterior_vs_index = _selfratios(
            scenario, tie_neighbors=tie_neighbors
        )
        if tie_neighbors and "max_index_vs_faiss_tie_break" in scenario:
            max_index_vs_faiss = float(scenario["max_index_vs_faiss_tie_break"])
        else:
            max_index_vs_faiss = float(scenario["max_index_vs_faiss"])
        if tie_neighbors and "max_posterior_vs_index_tie_break" in scenario:
            max_posterior_vs_index = float(scenario["max_posterior_vs_index_tie_break"])
        else:
            max_posterior_vs_index = float(scenario["max_posterior_vs_index"])
        assert index_vs_faiss <= max_index_vs_faiss, (
            f"{scenario['name']} tie_break={tie_neighbors}: "
            f"index/faiss={index_vs_faiss:.3f}x exceeds cap {max_index_vs_faiss:.2f}x "
            f"(t_index={t_index:.4f}s t_faiss={t_faiss:.4f}s; "
            f"baseline_rev={baseline['baseline_revision'][:7]})"
        )
        assert posterior_vs_index <= max_posterior_vs_index, (
            f"{scenario['name']} tie_break={tie_neighbors}: "
            f"posterior/index={posterior_vs_index:.3f}x exceeds cap "
            f"{max_posterior_vs_index:.2f}x "
            f"(t_post={t_post:.4f}s t_index={t_index:.4f}s; "
            f"baseline_rev={baseline['baseline_revision'][:7]})"
        )
