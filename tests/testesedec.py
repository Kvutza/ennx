"""Tests for se_epi / se_ale decomposition invariants."""

from __future__ import annotations

import numpy as np
import pytest

from ennx import ENN
from ennx.ennx.enn_params import ENNParams, PosteriorFlags

try:
    from ennx._rust import ENN as RustENN

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False


def _params(k: int = 2, *, aleatoric: float = 0.1) -> ENNParams:
    return ENNParams(
        k_neighbors=k,
        epistemic_scale=1.0,
        aleatoric_scale=aleatoric,
    )


def test_001():
    train_x = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float)
    train_y = np.array([[0.0], [1.0], [1.0], [2.0]], dtype=float)
    query = np.array([[0.5, 0.5]], dtype=float)
    flags = PosteriorFlags(observation_noise=False)
    model = ENN(train_x, train_y, scale_x=False)
    post = model.posterior(query, params=_params(), flags=flags)

    np.testing.assert_allclose(post.se_epi, post.se, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(post.se_ale, 0.0, rtol=0, atol=1e-15)


def test_002():
    train_x = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float)
    train_y = np.array([[0.0], [1.0], [1.0], [2.0]], dtype=float)
    query = np.array([[0.5, 0.5]], dtype=float)
    flags = PosteriorFlags(observation_noise=True)
    model = ENN(train_x, train_y, scale_x=False)
    post = model.posterior(query, params=_params(aleatoric=0.1), flags=flags)

    recomposed = np.hypot(post.se_epi, post.se_ale)
    np.testing.assert_allclose(recomposed, post.se, rtol=1e-10, atol=1e-10)
    assert np.all(post.se_ale > 0)


def test_003():
    train_x = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float)
    train_y = np.array([[0.0], [1.0], [1.0], [2.0]], dtype=float)
    train_yvar = np.full((4, 1), 0.05, dtype=float)
    query = np.array([[0.5, 0.5]], dtype=float)
    flags = PosteriorFlags(observation_noise=False)
    params = _params()

    model_yvar = ENN(train_x, train_y, train_yvar=train_yvar, scale_x=False)
    model_plain = ENN(train_x, train_y, scale_x=False)

    post_yvar = model_yvar.posterior(query, params=params, flags=flags)
    post_plain = model_plain.posterior(query, params=params, flags=flags)

    np.testing.assert_allclose(post_yvar.se_ale, 0.0, rtol=0, atol=1e-15)
    np.testing.assert_allclose(post_yvar.se_epi, post_yvar.se, rtol=1e-12, atol=1e-12)
    assert not np.allclose(post_yvar.se, post_plain.se)


def test_004():
    train_x = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float)
    train_y = np.array([[0.0], [1.0], [1.0], [2.0]], dtype=float)
    query = np.array([[0.5, 0.5], [0.2, 0.8]], dtype=float)
    flags = PosteriorFlags(observation_noise=True)
    params = _params(aleatoric=0.1)
    model = ENN(train_x, train_y, scale_x=False)

    batch = model.batch_posterior(query, paramss=[params], flags=flags)
    single = model.posterior(query, params=params, flags=flags)

    np.testing.assert_allclose(batch.mu[0], single.mu, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(batch.se[0], single.se, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(batch.se_epi[0], single.se_epi, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(batch.se_ale[0], single.se_ale, rtol=1e-12, atol=1e-12)


def test_005():
    train_x = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float)
    train_y = np.array([[0.0], [1.0], [1.0], [2.0]], dtype=float)
    query = np.array([[0.5, 0.5]], dtype=float)
    flags = PosteriorFlags(observation_noise=True)
    params = _params(aleatoric=0.1)
    model = ENN(train_x, train_y, scale_x=False)

    post = model.posterior(query, params=params, flags=flags)
    cond = model.conditional_posterior(
        np.zeros((0, 2), dtype=float),
        np.zeros((0, 1), dtype=float),
        query,
        params=params,
        flags=flags,
    )

    np.testing.assert_allclose(cond.se_epi, post.se_epi, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(cond.se_ale, post.se_ale, rtol=1e-12, atol=1e-12)


def test_006():
    model = ENN(
        np.zeros((0, 2), dtype=float),
        np.zeros((0, 1), dtype=float),
        scale_x=False,
    )
    query = np.array([[0.5, 0.5]], dtype=float)
    post = model.posterior(query, params=_params())

    np.testing.assert_allclose(post.se_epi, post.se)
    np.testing.assert_allclose(post.se_ale, 0.0)


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust not available")
def test_007():
    train_x = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    train_y = np.array([[0.0], [1.0]], dtype=float)
    query = np.array([[0.5, 0.5]], dtype=float)
    rs_model = RustENN(train_x, train_y, scale_x=False, index_driver="Exact")
    out = rs_model.posterior(
        query,
        k_neighbors=2,
        epistemic_scale=1.0,
        aleatoric_scale=0.1,
        exclude_nearest=False,
        observation_noise=False,
    )
    assert len(out) == 5
    rs_mu, rs_se, rs_se_epi, rs_se_ale, rs_idx = out
    assert rs_mu.shape == rs_se.shape == rs_se_epi.shape == rs_se_ale.shape
    assert rs_idx is not None
