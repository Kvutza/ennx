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

pytestmark = pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust not available")


def test_001():
    """Public API posterior matches Rust implementation (via backend delegation)."""
    train_x = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float)
    train_y = np.array([[0.0], [1.0], [1.0], [2.0]], dtype=float)
    query = np.array([[0.5, 0.5]], dtype=float)
    params = ENNParams(k_neighbors=2, epistemic_scale=1.0, aleatoric_scale=0.1)
    flags = PosteriorFlags(exclude_nearest=False, observation_noise=False)

    model = ENN(train_x, train_y, scale_x=False)
    out = model.posterior(query, params=params, flags=flags)

    rs_model = RustENN(train_x, train_y, scale_x=False, index_driver="Exact")
    rs_mu, rs_se, rs_se_epi, rs_se_ale, _rs_idx = rs_model.posterior(
        query,
        k_neighbors=2,
        epistemic_scale=1.0,
        aleatoric_scale=0.1,
        exclude_nearest=False,
        observation_noise=False,
    )

    np.testing.assert_allclose(out.mu, rs_mu, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(out.se, rs_se, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(out.se_epi, rs_se_epi, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(out.se_ale, rs_se_ale, rtol=1e-12, atol=1e-12)
    assert out.idx is not None and out.idx.shape[0] == 1


def test_addandlencontract():
    train_x = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    train_y = np.array([[0.0], [1.0]], dtype=float)
    rs_model = RustENN(train_x, train_y, scale_x=False, index_driver="Exact")
    assert len(rs_model) == 2
    rs_model.add(
        np.array([[0.0, 1.0]], dtype=float), np.array([[1.0]], dtype=float), None
    )
    assert len(rs_model) == 3
    assert rs_model.num_outputs == 1


def test_002():
    """Python wrapper delegates to Rust correctly when train_yvar is provided."""
    train_x = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float)
    train_y = np.array([[0.0], [1.0], [1.0], [2.0]], dtype=float)
    train_yvar = np.array([[0.01], [0.01], [0.01], [0.01]], dtype=float)
    query = np.array([[0.5, 0.5]], dtype=float)
    params = ENNParams(
        k_neighbors=2,
        epistemic_scale=1.0,
        aleatoric_scale=0.1,
    )
    flags = PosteriorFlags(exclude_nearest=False, observation_noise=True)

    model = ENN(train_x, train_y, train_yvar=train_yvar, scale_x=False)
    out = model.posterior(query, params=params, flags=flags)

    rs_model = RustENN(
        train_x, train_y, train_yvar=train_yvar, scale_x=False, index_driver="Exact"
    )
    rs_mu, rs_se, rs_se_epi, rs_se_ale, rs_idx = rs_model.posterior(
        query,
        k_neighbors=2,
        epistemic_scale=1.0,
        aleatoric_scale=0.1,
        exclude_nearest=False,
        observation_noise=True,
    )

    np.testing.assert_allclose(out.mu, rs_mu, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(out.se, rs_se, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(out.se_epi, rs_se_epi, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(out.se_ale, rs_se_ale, rtol=1e-10, atol=1e-10)
    assert rs_idx is not None and len(rs_idx) == 1


def test_003():
    """exclude_nearest=True with 1 observation raises ValueError."""
    train_x = np.array([[0.0, 0.0]], dtype=float)
    train_y = np.array([[0.0]], dtype=float)
    query = np.array([[0.5, 0.5]], dtype=float)
    params = ENNParams(
        k_neighbors=2,
        epistemic_scale=1.0,
        aleatoric_scale=0.1,
    )
    flags = PosteriorFlags(exclude_nearest=True, observation_noise=False)

    model = ENN(train_x, train_y, scale_x=False)
    with pytest.raises(ValueError):
        model.posterior(query, params=params, flags=flags)

    rs_model = RustENN(train_x, train_y, scale_x=False, index_driver="Exact")
    with pytest.raises(ValueError):
        rs_model.posterior(
            query,
            k_neighbors=2,
            epistemic_scale=1.0,
            aleatoric_scale=0.1,
            exclude_nearest=True,
            observation_noise=False,
        )
