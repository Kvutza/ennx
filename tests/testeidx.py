from __future__ import annotations

import numpy as np
import pytest

from ennx.ennx.enn_class import ENN
from ennx.ennx.eclssup import (
    posterior_neighbors,
    nearest_neighbors,
)
from ennx.ennx.enn_hash import (
    normal_philox,
    normal_hash,
)
from ennx.ennx.enn_params import ENNParams, PosteriorFlags
from ennx.turbo.config.eidxdrv import ENNIndexDriver


def _enn(train_x, *, scale_x=False, index_driver=ENNIndexDriver.FLAT, train_y=None):
    if train_y is None:
        train_y = np.zeros((train_x.shape[0], 1), dtype=float)
    return ENN(train_x, train_y, scale_x=scale_x, index_driver=index_driver)


def test_001():
    rng = np.random.default_rng(0)
    n_train = 3
    train_x = rng.standard_normal((n_train, 2))
    enn = _enn(train_x, scale_x=False)
    query = rng.standard_normal((1, 2))
    search_k = 8
    _dist2s, idx = nearest_neighbors(
        enn.rust_backend, query, search_k=search_k, exclude_nearest=False
    )
    assert idx.shape == (1, search_k)
    assert np.all(idx >= 0)
    assert np.all(idx < n_train)


def test_002():
    train_x = np.zeros((0, 2), dtype=float)
    enn = _enn(train_x, scale_x=False)
    query = np.array([[1.0, 2.0], [-0.5, 0.25]], dtype=float)
    search_k = 4
    dist2s, idx = nearest_neighbors(
        enn.rust_backend, query, search_k=search_k, exclude_nearest=False
    )
    assert dist2s.shape == (2, search_k) and idx.shape == (2, search_k)
    assert np.all(np.isposinf(dist2s))
    assert np.all(idx == 0)
    assert np.all(idx >= 0)
    d2_ex, idx_ex = nearest_neighbors(
        enn.rust_backend, query, search_k=search_k, exclude_nearest=True
    )
    assert d2_ex.shape == (2, search_k - 1) and idx_ex.shape == (2, search_k - 1)
    assert np.all(np.isposinf(d2_ex))
    assert np.all(idx_ex == 0)


def test_003():
    rng = np.random.default_rng(7)
    train_x = rng.standard_normal((10, 2))
    enn = _enn(train_x, scale_x=False)
    q = rng.standard_normal((3, 2))
    dist2s, idx = nearest_neighbors(
        enn.rust_backend, q, search_k=1, exclude_nearest=True
    )
    assert dist2s.shape == (3, 0) and idx.shape == (3, 0)


def test_004():
    rng = np.random.default_rng(11)
    n_train, dim = 40, 3
    train_x = rng.standard_normal((n_train, dim))
    enn = _enn(train_x, scale_x=False, index_driver=ENNIndexDriver.FLAT)
    query = rng.standard_normal((4, dim))
    search_k = 6
    dist2s, idx = nearest_neighbors(
        enn.rust_backend, query, search_k=search_k, exclude_nearest=False
    )
    assert dist2s.shape == (4, search_k) and idx.shape == (4, search_k)
    assert np.all(idx >= 0) and np.all(idx < n_train)
    finite = np.isfinite(dist2s)
    assert np.any(finite)
    assert np.all(dist2s[finite] >= 0.0)


def test_005():
    rng = np.random.default_rng(42)
    train_x = rng.standard_normal((20, 3))
    enn = _enn(train_x, scale_x=False)
    query = rng.standard_normal((5, 3))
    dist2s, idx = nearest_neighbors(
        enn.rust_backend, query, search_k=3, exclude_nearest=False
    )
    assert dist2s.shape == (5, 3) and idx.shape == (5, 3)
    assert np.all(idx >= 0) and np.all(idx < 20)


def test_006():
    train_x = np.array([[0.0], [0.0], [1.0], [2.0]])
    np.array([[0.0], [1.0], [2.0], [3.0]])
    enn = _enn(train_x, scale_x=False)
    query = np.array([[0.0]])
    _, idx_on = posterior_neighbors(
        enn.rust_backend,
        query,
        search_k=2,
        flags=PosteriorFlags(tie_neighbors=True),
    )
    _, idx_off = posterior_neighbors(
        enn.rust_backend,
        query,
        search_k=2,
        flags=PosteriorFlags(tie_neighbors=False),
    )
    assert idx_on[0].tolist() == [0, 1]
    assert idx_off[0].tolist() in ([0, 1], [1, 0])


def test_007():
    train_x = np.array([[(i - 9.5) / 3.0 + 0.01 * i] for i in range(20)])
    train_y = np.array([[(i + 1) * 0.37 - 2.1] for i in range(20)])
    enn = _enn(train_x, scale_x=False, train_y=train_y)
    k = 10
    _, idx_batch = posterior_neighbors(
        enn.rust_backend,
        train_x,
        search_k=k,
        flags=PosteriorFlags(tie_neighbors=True),
    )
    for i in range(train_x.shape[0]):
        _, idx_one = posterior_neighbors(
            enn.rust_backend,
            train_x[i : i + 1],
            search_k=k,
            flags=PosteriorFlags(tie_neighbors=True),
        )
        assert idx_batch[i].tolist() == idx_one[0].tolist()


def test_008():
    train_x = np.array([[0.0], [0.0], [0.0], [1.0]])
    np.array([[0.0], [1.0], [2.0], [3.0]])
    enn = _enn(train_x, scale_x=False)
    query = np.array([[0.0]])
    _, idx_on = posterior_neighbors(
        enn.rust_backend,
        query,
        search_k=2,
        flags=PosteriorFlags(tie_neighbors=True),
    )
    assert idx_on[0].tolist() == [0, 1]


def test_009():
    train_x = np.array([[(i - 9.5) / 3.0 + 0.01 * i] for i in range(20)])
    train_y = np.array([[(i + 1) * 0.37 - 2.1] for i in range(20)])
    enn = _enn(train_x, scale_x=False, train_y=train_y)
    params = ENNParams(k_neighbors=10, epistemic_scale=1.0, aleatoric_scale=0.1)
    flags = PosteriorFlags(tie_neighbors=True)
    result = enn.posterior(train_x, params=params, flags=flags)
    assert result.idx is not None
    assert len(result.idx) == train_x.shape[0]


@pytest.mark.parametrize("scale_x", [False, True])
def test_010(scale_x):
    rng = np.random.default_rng(42)
    train_x = rng.standard_normal((20, 3))
    enn = _enn(train_x, scale_x=scale_x)
    query = rng.standard_normal((5, 3))
    search_k = 3
    faiss_d2, faiss_idx = nearest_neighbors(
        enn.rust_backend, query, search_k=search_k, exclude_nearest=False
    )
    exact_d2, exact_idx = posterior_neighbors(
        enn.rust_backend, query, search_k=search_k
    )
    assert exact_d2.shape == faiss_d2.shape == (5, search_k)
    assert exact_idx.shape == faiss_idx.shape
    assert np.all(exact_idx >= 0) and np.all(exact_idx < 20)


def test_011():
    rng = np.random.default_rng(42)
    train_x = rng.standard_normal((20, 3))
    query = train_x[:3]
    enn = _enn(train_x, scale_x=False, index_driver=ENNIndexDriver.FLAT)
    dist2s, idx = posterior_neighbors(
        enn.rust_backend,
        query,
        search_k=3,
        flags=PosteriorFlags(exclude_nearest=True),
    )
    assert dist2s.shape == (3, 2) and idx.shape == (3, 2)
    assert np.all(idx >= 0) and np.all(idx < 20)


def test_012():
    rng = np.random.default_rng(7)
    train_x = rng.standard_normal((10, 2))
    enn = _enn(train_x, scale_x=False)
    q = rng.standard_normal((3, 2))
    dist2s, idx = posterior_neighbors(
        enn.rust_backend,
        q,
        search_k=1,
        flags=PosteriorFlags(exclude_nearest=True),
    )
    assert dist2s.shape == (3, 0) and idx.shape == (3, 0)


def test_013():
    train_x = np.array([[0.0], [1.0], [2.0]], dtype=float)
    enn = _enn(train_x, scale_x=False)
    query = np.array([[0.5], [1.5]], dtype=float)
    d0, i0 = posterior_neighbors(enn.rust_backend, query, search_k=0)
    assert d0.shape == (2, 0) and i0.shape == (2, 0)
    d_all, i_all = posterior_neighbors(enn.rust_backend, query, search_k=10)
    assert d_all.shape == (2, 10) and i_all.shape == (2, 10)
    assert np.all(i_all[:, :3] >= 0) and np.all(i_all[:, :3] < 3)
    assert np.all(i_all[:, 3:] == -1)


def test_014():
    rng = np.random.default_rng(42)
    train_x = rng.standard_normal((20, 3))
    enn = _enn(train_x, scale_x=False)
    query = train_x[:3]
    dist2s_include, _idx_include = nearest_neighbors(
        enn.rust_backend, query, search_k=3, exclude_nearest=False
    )
    dist2s_exclude, _idx_exclude = nearest_neighbors(
        enn.rust_backend, query, search_k=3, exclude_nearest=True
    )
    assert dist2s_include.shape == (3, 3) and dist2s_exclude.shape == (3, 2)
    assert np.allclose(dist2s_include[:, 0], 0.0, atol=1e-6)


def test_015():
    rng = np.random.default_rng(42)
    train_x = rng.standard_normal((20, 3))
    enn = _enn(train_x, scale_x=True)
    query = rng.standard_normal((5, 3))
    dist2s, idx = nearest_neighbors(
        enn.rust_backend, query, search_k=3, exclude_nearest=False
    )
    assert dist2s.shape == (5, 3) and idx.shape == (5, 3)


@pytest.mark.parametrize("query_shape,search_k", [((5, 3), 0), ((5, 4), 3)])
def test_016(query_shape, search_k):
    rng = np.random.default_rng(42)
    train_x = rng.standard_normal((20, 3))
    enn = _enn(train_x, scale_x=False)
    with pytest.raises(ValueError):
        nearest_neighbors(
            enn.rust_backend,
            rng.standard_normal(query_shape),
            search_k=search_k,
            exclude_nearest=False,
        )


def test_017():
    function_seeds = np.array([1, 2, 3], dtype=np.int64)
    data_indices = np.array([[0, 1, 2], [3, 4, 5]], dtype=int)
    result = normal_philox(function_seeds, data_indices, num_metrics=2)
    assert result.shape == (3, 2, 3, 2)


def test_018():
    function_seeds = np.array([42], dtype=np.int64)
    data_indices = np.array([[0, 1]], dtype=int)
    result1 = normal_philox(function_seeds, data_indices, num_metrics=1)
    result2 = normal_philox(function_seeds, data_indices, num_metrics=1)
    assert np.allclose(result1, result2)


def test_019():
    data_indices = np.array([[0, 1]], dtype=int)
    result1 = normal_philox(np.array([1], dtype=np.int64), data_indices, num_metrics=1)
    result2 = normal_philox(np.array([2], dtype=np.int64), data_indices, num_metrics=1)
    assert not np.allclose(result1, result2)


def test_020():
    function_seeds = np.array([1, 2, 3], dtype=np.int64)
    data_indices = np.array([[0, 1, 2], [3, 4, 5]], dtype=int)
    out1 = normal_hash(function_seeds, data_indices, num_metrics=2)
    out2 = normal_hash(function_seeds, data_indices, num_metrics=2)
    assert out1.shape == (3, 2, 3, 2)
    assert np.allclose(out1, out2)


def test_021():
    data_indices = np.array([[0, 1]], dtype=int)
    out1 = normal_hash(np.array([1], dtype=np.int64), data_indices, num_metrics=3)
    out2 = normal_hash(np.array([2], dtype=np.int64), data_indices, num_metrics=3)
    assert not np.allclose(out1, out2)
