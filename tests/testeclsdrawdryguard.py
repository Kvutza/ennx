"""Regression guards for ENN function-draw paths before/after DRY refactors."""

from __future__ import annotations

import numpy as np

from ennx.ennx.enn_class import ENN
from ennx.ennx.enn_params import ENNParams, PosteriorFlags


def _tinymodel():
    train_x = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
    train_y = np.array([[0.1], [0.2], [0.3]], dtype=float)
    return ENN(train_x, train_y)


def test_001():
    model = _tinymodel()
    x = np.array([[0.25, 0.25]], dtype=float)
    params = ENNParams(2, 1.0, 0.1)
    seeds = np.array([7, 8, 9], dtype=np.int64)
    d1, i1 = model.posterior_draw(x, params, function_seeds=seeds, flags=None)
    d2, i2 = model.posterior_draw(
        x, params, function_seeds=seeds, flags=PosteriorFlags()
    )
    assert np.allclose(d1, d2)
    assert np.array_equal(np.asarray(i1, dtype=int), np.asarray(i2, dtype=int))


def test_002():
    model = _tinymodel()
    x = np.array([[0.25, 0.25], [0.5, 0.1]], dtype=float)
    params = ENNParams(2, 1.0, 0.1)
    flags = PosteriorFlags(exclude_nearest=True, observation_noise=False)
    seeds = [1001, 1002]
    a1, ix1 = model.posterior_draw(x, params, function_seeds=seeds, flags=flags)
    a2, ix2 = model.posterior_draw(x, params, function_seeds=seeds, flags=flags)
    assert np.allclose(a1, a2)
    assert np.array_equal(np.asarray(ix1, dtype=int), np.asarray(ix2, dtype=int))


def test_003():
    model = _tinymodel()
    x = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=float)
    params = ENNParams(2, 1.0, 0.1)
    seeds = [1]
    draws, idx = model.posterior_draw(
        x, params, function_seeds=seeds, flags=PosteriorFlags()
    )
    assert draws.shape == (x.shape[0], model.num_outputs, 1)
    idx_arr = np.asarray(idx, dtype=int)
    assert idx_arr.shape == (x.shape[0], params.k_neighbors)


def test_004():
    """Locks parity between delegation branch and posterior_draw."""
    model = _tinymodel()
    x = np.array([[0.11, 0.22]], dtype=float)
    params = ENNParams(2, 1.0, 0.05)
    flags = PosteriorFlags(exclude_nearest=False, observation_noise=True)
    seeds = np.array([3, 4], dtype=np.int64)
    d_post, i_post = model.posterior_draw(x, params, function_seeds=seeds, flags=flags)
    d_cond, i_cond = model.conditional_draw(
        np.zeros((0, 2), dtype=float),
        np.zeros((0, 1), dtype=float),
        x,
        params=params,
        function_seeds=seeds,
        flags=flags,
    )
    assert np.allclose(d_post, d_cond, rtol=0.0, atol=0.0)
    assert np.array_equal(np.asarray(i_post, dtype=int), np.asarray(i_cond, dtype=int))


def test_005():
    """Byte-level guard on one deterministic draw (update only if ENN math intentionally changes)."""
    train_x = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float)
    train_y = np.array([[0.0], [1.0], [1.0], [2.0]], dtype=float)
    model = ENN(train_x, train_y)
    x = np.array([[0.25, 0.25]], dtype=float)
    params = ENNParams(2, 1.0, 0.1)
    seeds = np.array([42, 43], dtype=np.int64)
    draws, _idx = model.posterior_draw(
        x, params, function_seeds=seeds, flags=PosteriorFlags()
    )
    # Expected from ennx ENN + Rust draw path (fixed data/seeds).
    # Shape is (batch, metrics, num_samples) to match posterior().sample().
    expected = np.array(
        [[[-0.71338448, -0.40793862]]],
        dtype=np.float64,
    )
    np.testing.assert_allclose(draws, expected, rtol=1e-7, atol=1e-7)
