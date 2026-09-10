from __future__ import annotations

import numpy as np

from ennx.ennx.enn_class import ENN
from ennx.ennx.enn_params import ENNParams, PosteriorFlags


def test_001():
    rng = np.random.default_rng(42)
    train_x = rng.standard_normal((20, 3))
    train_y = train_x.sum(axis=1, keepdims=True)
    model = ENN(train_x, train_y, 0.1 * np.ones_like(train_y))
    x_test = rng.standard_normal((5, 3))
    params = ENNParams(k_neighbors=5, epistemic_scale=1.0, aleatoric_scale=0.0)
    draws, idx = model.posterior_draw(x_test, params, function_seeds=[123])
    sample = draws[:, :, 0]
    assert sample.shape == (5, 1) and np.all(np.isfinite(sample))
    assert idx.shape == (5, 5)


def test_002():
    rng = np.random.default_rng(42)
    train_x = rng.standard_normal((20, 3))
    model = ENN(train_x, train_x.sum(axis=1, keepdims=True))
    x_test = rng.standard_normal((5, 3))
    params = ENNParams(k_neighbors=5, epistemic_scale=1.0, aleatoric_scale=0.0)
    sample1 = model.posterior_draw(x_test, params, function_seeds=[42])[0][:, :, 0]
    assert np.allclose(
        sample1,
        model.posterior_draw(x_test, params, function_seeds=[42])[0][:, :, 0],
    )
    assert not np.allclose(
        sample1,
        model.posterior_draw(x_test, params, function_seeds=[43])[0][:, :, 0],
    )


def test_003():
    rng = np.random.default_rng(42)
    train_x = rng.standard_normal((20, 3))
    train_y = train_x.sum(axis=1, keepdims=True)
    model = ENN(train_x, train_y, 0.1 * np.ones_like(train_y))
    x_test = rng.standard_normal((5, 3))
    params = ENNParams(k_neighbors=5, epistemic_scale=1.0, aleatoric_scale=0.0)
    samples, idx = model.posterior_draw(x_test, params, function_seeds=[10, 20, 30])
    assert samples.shape == (5, 1, 3) and np.all(np.isfinite(samples))
    assert idx.shape == (5, 5)


def test_004():
    rng = np.random.default_rng(42)
    train_x = rng.standard_normal((20, 3))
    model = ENN(train_x, train_x.sum(axis=1, keepdims=True))
    x_test = rng.standard_normal((5, 3))
    params = ENNParams(k_neighbors=5, epistemic_scale=1.0, aleatoric_scale=0.0)
    batch, _ = model.posterior_draw(x_test, params, function_seeds=[100, 200, 300])
    for i, seed in enumerate([100, 200, 300]):
        assert np.allclose(
            batch[:, :, i],
            model.posterior_draw(x_test, params, function_seeds=[seed])[0][:, :, 0],
        )


def test_005():
    rng = np.random.default_rng(42)
    train_x = rng.standard_normal((20, 3))
    model = ENN(train_x, rng.standard_normal((20, 2)))
    x_test = rng.standard_normal((5, 3))
    params = ENNParams(k_neighbors=5, epistemic_scale=1.0, aleatoric_scale=0.0)
    samples, _ = model.posterior_draw(x_test, params, function_seeds=[1, 2, 3, 4])
    assert samples.shape == (5, 2, 4) and np.all(np.isfinite(samples))


def test_006():
    rng = np.random.default_rng(42)
    train_x = rng.standard_normal((2, 3))
    train_y = train_x.sum(axis=1, keepdims=True)
    model = ENN(train_x, train_y)
    x_test = rng.standard_normal((5, 3))
    params = ENNParams(k_neighbors=5, epistemic_scale=1.0, aleatoric_scale=0.0)
    samples, idx = model.posterior_draw(
        x_test,
        params,
        function_seeds=[1, 2],
        flags=PosteriorFlags(exclude_nearest=True),
    )
    assert samples.shape == (5, 1, 2)
    assert idx.shape == (5, 1)


def test_007():
    rng = np.random.default_rng(42)
    train_x = rng.standard_normal((20, 3))
    train_y = train_x.sum(axis=1, keepdims=True)
    model = ENN(train_x, train_y, 0.5 * np.ones_like(train_y))
    x_test = rng.standard_normal((5, 3))
    params = ENNParams(k_neighbors=5, epistemic_scale=1.0, aleatoric_scale=0.0)
    sample_no_noise = model.posterior_draw(x_test, params, function_seeds=[42])[0][
        :, :, 0
    ]
    sample_with_noise = model.posterior_draw(
        x_test,
        params,
        function_seeds=[42],
        flags=PosteriorFlags(observation_noise=True),
    )[0][:, :, 0]
    assert sample_no_noise.shape == sample_with_noise.shape == (5, 1)
    assert not np.allclose(sample_no_noise, sample_with_noise)
