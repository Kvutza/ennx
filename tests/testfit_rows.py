import numpy as np
import pytest

from ennx.ennx.enn_class import ENN
from ennx.ennx.enn_fit import row_loglik, subsample_loglik
from ennx.ennx.enn_params import ENNParams


def test_duplicates():
    x = np.zeros((3, 1))
    y = np.array([[0.0], [5.0], [10.0]])
    model = ENN(x, y)
    params = [ENNParams(k_neighbors=2, epistemic_scale=1.0, aleatoric_scale=1.0)]
    options = {"paramss": params, "P": 3, "y_std": np.ones(1)}
    actual = row_loglik(model, [2, 0, 1], rng=np.random.default_rng(42), **options)
    variance = (50.0 / 3.0) * (1.5 + 0.5e-9)
    expected = -1.5 * np.log(2 * np.pi * variance) - 0.5 * 112.5 / variance
    assert actual[0] == pytest.approx(expected)
    ordered = subsample_loglik(model, x, y, rng=np.random.default_rng(42), **options)
    assert actual == pytest.approx(ordered)
    with pytest.raises(ValueError, match="row_loglik"):
        subsample_loglik(model, x[:1], y[:1], rng=np.random.default_rng(42), **options)
    with pytest.raises(ValueError, match="row IDs"):
        row_loglik(model, [3], rng=np.random.default_rng(42), **options)


def test_invalid():
    x = np.array([[-1.0], [1.0]])
    model = ENN(x, x)
    params = [
        ENNParams(k_neighbors=1, epistemic_scale=scale, aleatoric_scale=0.0)
        for scale in (1.0, np.finfo(float).max)
    ]
    scores = subsample_loglik(
        model, x, x, paramss=params, P=2, rng=np.random.default_rng(42)
    )
    assert np.isfinite(scores[0])
    assert scores[1] == -np.inf


def test_subset():
    x = np.arange(6.0).reshape(3, 2)
    y = np.arange(3.0).reshape(3, 1)
    model = ENN(x, y, scale_x=True)
    params = [ENNParams(k_neighbors=2, epistemic_scale=1.0, aleatoric_scale=0.1)]
    scores = row_loglik(
        model, [2, 0], paramss=params, P=1, rng=np.random.default_rng(42)
    )
    assert np.isfinite(scores).all()
    with pytest.raises(ValueError, match="row order"):
        subsample_loglik(
            model, x[::-1], y[::-1], paramss=params, P=3, rng=np.random.default_rng(42)
        )
