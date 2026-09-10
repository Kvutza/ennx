from __future__ import annotations

import numpy as np
import pytest
from enn_helpers import enn_model, enn_rows

from ennx.ennx.enn_class import ENN
from ennx.ennx.enn_params import ENNParams, PosteriorFlags


def _params(
    k: int,
    *,
    epistemic_scale: float = 1.0,
    aleatoric_scale: float = 0.0,
):
    return ENNParams(
        k_neighbors=int(k),
        epistemic_scale=float(epistemic_scale),
        aleatoric_scale=float(aleatoric_scale),
    )


def test_001():
    from ennx.ennx.enn_normal import ENNNormal

    rng = np.random.default_rng(0)
    mu = np.array([[0.0, 1.0]], dtype=float)
    se = np.array([[1.0, 2.0]], dtype=float)
    normal = ENNNormal(mu=mu, se=se, se_epi=se.copy(), se_ale=np.zeros_like(se))
    samples = normal.sample(5, clip=1.0, rng=rng)
    assert samples.shape == (1, 2, 5)
    assert np.all(samples >= mu.min() - 2.0)
    assert np.all(samples <= mu.max() + 2.0)


def test_002():
    model, _train_x, _ytrain, _trainyvar, rng = enn_model()
    x_test = rng.standard_normal((4, 3))
    params = ENNParams(k_neighbors=3, epistemic_scale=1.0, aleatoric_scale=0.0)
    post = model.posterior(x_test, params=params)
    assert post.mu.shape == (4, 1)
    assert post.se.shape == (4, 1)
    post_changed = model.posterior(
        x_test,
        params=ENNParams(
            k_neighbors=5,
            epistemic_scale=0.5,
            aleatoric_scale=0.0,
        ),
        flags=PosteriorFlags(exclude_nearest=True),
    )
    assert post_changed.mu.shape == (4, 1)
    assert post_changed.se.shape == (4, 1)


def test_003():
    rng = np.random.default_rng(0)
    d = 3
    x = np.zeros((0, d), dtype=float)
    y = np.zeros((0, 1), dtype=float)
    yvar = np.ones_like(y, dtype=float)
    model = ENN(x, y, yvar)
    x_test = rng.standard_normal((5, d))
    post = model.posterior(
        x_test,
        params=ENNParams(
            k_neighbors=3,
            epistemic_scale=1.0,
            aleatoric_scale=0.0,
        ),
    )
    assert post.mu.shape == (5, 1)
    assert post.se.shape == (5, 1)
    assert np.allclose(post.mu, 0.0)
    assert np.allclose(post.se, 1.0)


@pytest.mark.parametrize("num_obs", [1, 2, 3])
def test_004(
    num_obs: int,
):
    rng = np.random.default_rng(0)
    d = 3
    x = rng.standard_normal((num_obs, d))
    y = (x.sum(axis=1, keepdims=True)).astype(float)
    yvar = 0.1 * np.ones_like(y)
    model = ENN(x, y, yvar)
    x_test = rng.standard_normal((5, d))
    post = model.posterior(x_test, params=_params(3))
    assert post.mu.shape == (5, 1)
    assert post.se.shape == (5, 1)
    assert np.all(np.isfinite(post.mu))
    assert np.all(np.isfinite(post.se))


@pytest.mark.parametrize(
    "flags,k_vals",
    [
        (PosteriorFlags(), [3, 5, 7]),
        (PosteriorFlags(exclude_nearest=True), [3, 5]),
    ],
)
def test_005(flags, k_vals):
    model, _train_x, _ytrain, _trainyvar, rng = enn_model()
    x_test = rng.standard_normal((4, 3))
    paramss = [
        ENNParams(
            k_neighbors=k,
            epistemic_scale=1.0 / (i + 1),
            aleatoric_scale=0.0,
        )
        for i, k in enumerate(k_vals)
    ]
    post_batch = model.batch_posterior(x_test, paramss, flags=flags)
    assert post_batch.mu.shape == (len(paramss), x_test.shape[0], model.num_outputs)
    assert post_batch.se.shape == (len(paramss), x_test.shape[0], model.num_outputs)
    for i, params in enumerate(paramss):
        post = model.posterior(x_test, params=params, flags=flags)
        assert np.allclose(post_batch.mu[i], post.mu) and np.allclose(
            post_batch.se[i], post.se
        )


def test_006():
    rng = np.random.default_rng(0)
    n = 50
    d = 3
    x = rng.standard_normal((n, d))
    y = (x[:, 0] + 0.1 * x[:, 1] + 0.01 * rng.standard_normal(n)).reshape(-1, 1)
    yvar = 0.1 * np.ones_like(y)
    model = ENN(x, y, yvar)
    x_test = rng.standard_normal((4, d))
    params = ENNParams(k_neighbors=3, epistemic_scale=1.0, aleatoric_scale=0.0)
    post = model.posterior(x_test, params=params)
    assert post.mu.shape == (4, 1) and post.se.shape == (4, 1)
    assert np.all(np.isfinite(post.mu)) and np.all(np.isfinite(post.se))


def test_007():
    rng = np.random.default_rng(0)
    n = 20
    d = 3
    x = rng.standard_normal((n, d))
    y = rng.standard_normal((n, 2))
    yvar = 0.1 * np.ones_like(y)
    model = ENN(x, y, yvar)
    x_test = rng.standard_normal((4, d))
    params = ENNParams(k_neighbors=3, epistemic_scale=1.0, aleatoric_scale=0.0)
    post = model.posterior(x_test, params=params)
    assert post.mu.shape == (4, 2) and post.se.shape == (4, 2)


def test_008():
    rng = np.random.default_rng(0)
    n = 5
    d = 3
    train_x = rng.standard_normal((n, d))
    train_y = (train_x.sum(axis=1, keepdims=True)).astype(float)
    train_yvar = 0.1 * np.ones_like(train_y)
    model = ENN(train_x, train_y, train_yvar)
    x_test = rng.standard_normal((4, d))
    params = ENNParams(k_neighbors=10, epistemic_scale=1.0, aleatoric_scale=0.0)
    post = model.batch_posterior(
        x_test, [params], flags=PosteriorFlags(exclude_nearest=True)
    )
    assert post.mu.shape == (1, 4, 1) and post.se.shape == (1, 4, 1)
    assert np.all(np.isfinite(post.mu))
    assert np.all(np.isfinite(post.se))


def test_009():
    rng = np.random.default_rng(42)
    n = 20
    d = 3
    train_x = rng.standard_normal((n, d))
    train_y = train_x.sum(axis=1, keepdims=True) + rng.standard_normal((n, 1)) * 0.1
    model = ENN(train_x, train_y, train_yvar=None)
    assert len(model) == n
    _, _, yvar = enn_rows(model)
    assert yvar is None
    x_test = rng.standard_normal((10, d))
    params = ENNParams(k_neighbors=5, epistemic_scale=1.0, aleatoric_scale=0.0)
    post = model.posterior(x_test, params=params)
    assert post.mu.shape == (10, 1)
    assert post.se.shape == (10, 1)
    assert np.all(np.isfinite(post.mu))
    assert np.all(np.isfinite(post.se))


def test_010():
    rng = np.random.default_rng(0)
    n = 20
    d = 3
    train_x = rng.standard_normal((n, d))
    train_y = np.zeros((n, 1), dtype=float)
    train_yvar = 0.1 * np.ones_like(train_y)
    model = ENN(train_x, train_y, train_yvar)
    x_test = rng.standard_normal((5, d))
    params = ENNParams(k_neighbors=5, epistemic_scale=1.0, aleatoric_scale=0.0)
    post = model.posterior(x_test, params=params)
    assert np.all(np.isfinite(post.mu))
    assert np.all(np.isfinite(post.se))


def test_011():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        ENN(rng.random(10), np.zeros((10, 1)))
    with pytest.raises(ValueError):
        ENN(rng.random((10, 3)), rng.random(10))
    with pytest.raises(ValueError):
        ENN(rng.random((10, 3)), rng.random((5, 1)))
    with pytest.raises(ValueError):
        ENN(rng.random((10, 3)), rng.random((10, 1)), rng.random(10))
    with pytest.raises(ValueError):
        ENN(rng.random((10, 3)), rng.random((10, 1)), rng.random((10, 2)))


def test_012():
    rng = np.random.default_rng(42)
    n, d = 20, 3
    train_x = rng.standard_normal((n, d))
    train_y = train_x.sum(axis=1, keepdims=True)
    train_yvar = 0.1 * np.ones_like(train_y)
    model = ENN(train_x, train_y, train_yvar)
    assert len(model) == n
    assert model.num_outputs == 1
    x_at, y_at, yvar_at = enn_rows(model)
    assert x_at is not None
    assert y_at is not None
    assert yvar_at is not None


def test_013():
    """Test that add() updates _yscale so posterior se values remain correct.

    Bug: When add() is called with data that has different variance than the
    initial data, the _yscale is not recalculated. This causes posterior
    standard error estimates to be dramatically wrong - potentially orders
    of magnitude off.
    """
    rng = np.random.default_rng(42)
    d = 3

    # Create initial model with small y values
    x_init = rng.random((5, d))
    y_init = rng.random((5, 1)) * 0.1  # y values in [0, 0.1]

    model_incremental = ENN(x_init, y_init)

    # Add data with much larger y values
    x_new = rng.random((3, d))
    y_new = rng.random((3, 1)) * 100  # y values in [0, 100]
    model_incremental.add(x_new, y_new)

    # Create fresh model with all data at once
    all_x = np.vstack([x_init, x_new])
    all_y = np.vstack([y_init, y_new])
    model_fresh = ENN(all_x, all_y)

    # Compare posteriors - they should be equivalent
    params = ENNParams(k_neighbors=3, epistemic_scale=1.0, aleatoric_scale=0.1)
    x_test = rng.random((2, d))

    post_incremental = model_incremental.posterior(x_test, params=params)
    post_fresh = model_fresh.posterior(x_test, params=params)

    # mu values should match (they're based on weighted neighbor values)
    np.testing.assert_allclose(post_incremental.mu, post_fresh.mu, rtol=1e-6)

    # se values should also match - this is where the bug manifests
    # Without the fix, se values differ by ~1000x
    np.testing.assert_allclose(post_incremental.se, post_fresh.se, rtol=0.01)


def test_014():
    """Regression: incremental add() with scale_x=True must match fresh construction."""
    rng = np.random.default_rng(42)
    d = 3
    x_init = rng.standard_normal((10, d))
    y_init = rng.standard_normal((10, 1))
    x_new = rng.standard_normal((5, d))
    y_new = rng.standard_normal((5, 1))

    model_incremental = ENN(x_init, y_init, scale_x=True)
    model_incremental.add(x_new, y_new)

    all_x = np.vstack([x_init, x_new])
    all_y = np.vstack([y_init, y_new])
    model_fresh = ENN(all_x, all_y, scale_x=True)

    params = ENNParams(k_neighbors=3, epistemic_scale=1.0, aleatoric_scale=0.1)
    x_test = rng.standard_normal((4, d))
    post_incremental = model_incremental.posterior(x_test, params=params)
    post_fresh = model_fresh.posterior(x_test, params=params)
    np.testing.assert_allclose(post_incremental.mu, post_fresh.mu, rtol=1e-6)
    np.testing.assert_allclose(post_incremental.se, post_fresh.se, rtol=0.01)
