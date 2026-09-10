from __future__ import annotations

import numpy as np
import pytest
from enn_helpers import enn_model

from ennx.ennx.enn_class import ENN
from ennx.turbo.config.eidxdrv import ENNIndexDriver


def test_001():
    model, train_x, train_y, _, _ = enn_model()
    indices = model.neighbors(np.zeros(3, dtype=float), k=5, exclude_nearest=False)
    assert indices.shape == (5,)
    assert np.all((0 <= indices) & (indices < len(train_x)))
    distances = [np.linalg.norm(train_x[i]) for i in indices]
    assert all(
        distances[i] <= distances[i + 1] + 1e-6 for i in range(len(distances) - 1)
    )
    assert train_y[indices].shape == (5, 1)


def test_002():
    rng = np.random.default_rng(3)
    n, d = 35, 4
    train_x = rng.standard_normal((n, d))
    train_y = train_x.sum(axis=1, keepdims=True).astype(float)
    model = ENN(train_x, train_y, scale_x=False, index_driver=ENNIndexDriver.FLAT)
    x_query = rng.standard_normal(d)
    neighbors = model.neighbors(x_query, k=6, exclude_nearest=False)
    assert neighbors.shape == (6,)
    assert np.all((0 <= neighbors) & (neighbors < n))


def test_003():
    model, train_x, _ytrain, _trainyvar, _rng = enn_model()
    x_query = train_x[5].copy()
    neighbors_exclude = model.neighbors(x_query, k=5, exclude_nearest=True)
    neighbors_include = model.neighbors(x_query, k=5, exclude_nearest=False)
    assert neighbors_exclude.shape == (5,) and neighbors_include.shape == (5,)
    assert neighbors_include[0] == 5
    assert neighbors_exclude[0] != 5
    assert 5 not in set(neighbors_exclude.tolist())


def test_004():
    d = 3
    train_x = np.zeros((0, d), dtype=float)
    train_y = np.zeros((0, 1), dtype=float)
    train_yvar = np.ones((0, 1), dtype=float)
    model = ENN(train_x, train_y, train_yvar)
    x_query = np.zeros(d, dtype=float)
    neighbors = model.neighbors(x_query, k=5, exclude_nearest=False)
    assert neighbors.shape == (0,)


def test_005():
    rng = np.random.default_rng(0)
    n, d = 5, 3
    train_x = rng.standard_normal((n, d))
    train_y = train_x.sum(axis=1, keepdims=True).astype(float)
    train_yvar = 0.1 * np.ones_like(train_y)
    model = ENN(train_x, train_y, train_yvar)
    x_query = np.zeros(d, dtype=float)
    neighbors = model.neighbors(x_query, k=20, exclude_nearest=False)
    assert neighbors.shape == (n,)


def test_neighborskzero():
    model, _train_x, _ytrain, _trainyvar, _rng = enn_model(n=10)
    x_query = np.zeros(3, dtype=float)
    neighbors = model.neighbors(x_query, k=0, exclude_nearest=False)
    assert neighbors.shape == (0,)


def test_006():
    rng = np.random.default_rng(0)
    n, d = 15, 3
    train_x = rng.standard_normal((n, d))
    train_y = rng.standard_normal((n, 2))
    train_yvar = 0.1 * np.ones_like(train_y)
    model = ENN(train_x, train_y, train_yvar)
    x_query = np.zeros(d, dtype=float)
    neighbors = model.neighbors(x_query, k=5, exclude_nearest=False)
    assert neighbors.shape == (5,)
    assert np.all((0 <= neighbors) & (neighbors < n))


def test_007():
    model, _train_x, _ytrain, _trainyvar, _rng = enn_model(n=10)
    d = 3
    x_query_1d = np.zeros(d, dtype=float)
    neighbors_1d = model.neighbors(x_query_1d, k=3, exclude_nearest=False)
    x_query_2d = np.zeros((1, d), dtype=float)
    neighbors_2d = model.neighbors(x_query_2d, k=3, exclude_nearest=False)
    assert neighbors_1d.shape == neighbors_2d.shape == (3,)
    assert np.all(neighbors_1d == neighbors_2d)


def test_008():
    rng = np.random.default_rng(0)
    d = 3
    train_x = rng.standard_normal((1, d))
    train_y, train_yvar = np.array([[1.0]]), np.array([[0.1]])
    model = ENN(train_x, train_y, train_yvar)
    x_query = np.zeros(d, dtype=float)
    with pytest.raises(
        ValueError, match="exclude_nearest=True requires at least 2 observations"
    ):
        model.neighbors(x_query, k=1, exclude_nearest=True)
