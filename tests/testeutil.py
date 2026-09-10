from __future__ import annotations

import numpy as np
import pytest

from ennx.ennx.enn_util import (
    pareto_arms,
    sobol_indices,
    pareto2d_max,
    standardize_y,
)


def _soboldata(*, rng, n: int, d: int, y_2d: bool) -> tuple:
    x = rng.standard_normal((n, d))
    y = x[:, 0] + 0.1 * rng.standard_normal(n)
    if y_2d:
        y = y.reshape(-1, 1)
    return x, y


@pytest.mark.parametrize(
    "n,d,y_2d,expected_check",
    [
        (50, 3, False, lambda S: S[0] > S[1] and S[0] > S[2]),
        (50, 3, True, lambda S: np.all(S >= 0) and np.all(S <= 1)),
    ],
)
def test_001(n, d, y_2d, expected_check):
    rng = np.random.default_rng(42)
    x, y = _soboldata(rng=rng, n=n, d=d, y_2d=y_2d)
    S = sobol_indices(x, y)
    assert S.shape == (d,) and np.all(S >= 0) and np.all(S <= 1) and expected_check(S)


@pytest.mark.parametrize(
    "make_data,expected_check",
    [
        (
            lambda rng: (rng.standard_normal((5, 2)), rng.standard_normal(5)),
            lambda S: np.all(S == 1.0),
        ),
        (
            lambda rng: (rng.standard_normal((50, 3)), np.ones(50)),
            lambda S: np.all(S == 1.0),
        ),
    ],
)
def test_002(make_data, expected_check):
    rng = np.random.default_rng(42)
    x, y = make_data(rng)
    S = sobol_indices(x, y)
    d = x.shape[1]
    assert S.shape == (d,) and expected_check(S)


def test_003():
    rng = np.random.default_rng(42)
    n, d = 50, 3
    x = np.zeros((n, d))
    x[:, 0], x[:, 1], x[:, 2] = (
        rng.standard_normal(n),
        1e-15 * rng.standard_normal(n),
        rng.standard_normal(n),
    )
    y = x[:, 0] + x[:, 2] + 0.1 * rng.standard_normal(n)
    S = sobol_indices(x, y)
    assert S.shape == (d,) and S[1] == 0.0 and S[0] > 0 and S[2] > 0


def test_004():
    rng = np.random.default_rng(42)
    n, d = 50, 3
    x, y = (
        rng.standard_normal((n, d)).astype(np.float32),
        rng.standard_normal(n).astype(np.float32),
    )
    S = sobol_indices(x, y)
    assert S.shape == (d,) and S.dtype == np.float32


def test_005():
    x_cand = np.arange(12, dtype=float).reshape(6, 2)
    mu, se = (
        np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.0]),
        np.array([0.10, 0.20, 0.15, 0.40, 0.05, 0.50]),
    )
    rng = np.random.default_rng(0)
    out4 = pareto_arms(x_cand, mu, se, num_arms=4, rng=rng)
    assert out4.shape == (4, 2) and np.allclose(out4, x_cand[[0, 1, 3, 5]])
    rng = np.random.default_rng(0)
    out5 = pareto_arms(x_cand, mu, se, num_arms=5, rng=rng)
    assert out5.shape == (5, 2) and np.allclose(out5, x_cand[[0, 1, 2, 3, 5]])


def test_006():
    a, b = np.array([1.0, 0.5, 0.2]), np.array([0.5, 1.0, 0.2])
    idx = pareto2d_max(a, b)
    assert set(idx.tolist()) == {0, 1}


def test_007():
    a = np.array([1.0, 0.5, 0.2, 0.9])
    b = np.array([0.5, 1.0, 0.2, 0.4])
    idx = pareto2d_max(a, b, idx=np.array([0, 1, 3], dtype=int))
    assert set(idx.tolist()) == {0, 1}


def test_008():
    a, b = np.array([1.0, 0.5, 0.2]), np.array([0.5, 1.0, 0.2])
    with pytest.raises(ValueError, match="negative"):
        pareto2d_max(a, b, idx=np.array([-1], dtype=int))


def test_009():
    a, b = np.array([1.0, 0.5, 0.2]), np.array([0.5, 1.0, 0.2])
    with pytest.raises(ValueError, match="out of bounds"):
        pareto2d_max(a, b, idx=np.array([99], dtype=int))


def test_010():
    a = np.array([1.0, np.nan, 0.5])
    b = np.array([0.5, 1.0, 1.0])
    with pytest.raises(ValueError, match="finite"):
        pareto2d_max(a, b)


def test_011():
    a = np.array([1.0, np.nan, 0.5, 0.9])
    b = np.array([0.5, 1.0, 0.2, 0.4])
    with pytest.raises(ValueError, match="finite"):
        pareto2d_max(a, b, idx=np.array([0, 1, 3], dtype=int))


def test_012():
    a = np.array([1.0, np.nan, 0.5])
    b = np.array([0.5, 1.0, 1.0])
    idx = pareto2d_max(a, b, idx=np.array([0, 2], dtype=int))
    assert set(idx.tolist()) == {0, 2}


@pytest.mark.parametrize(
    "y_input,expected_center,check_scale",
    [
        (
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            3.0,
            lambda s, y: np.isclose(s, np.std(y)),
        ),
        ([10.0, 20.0, 30.0], 20.0, lambda s, y: s > 0),
        (np.array([5.0, 5.0, 5.0, 5.0]), 5.0, lambda s, y: s == 1.0),
        ([42.0], 42.0, lambda s, y: s == 1.0),
    ],
)
def test_standardizey(y_input, expected_center, check_scale):
    center, scale = standardize_y(y_input)
    assert center == expected_center and check_scale(scale, y_input)
