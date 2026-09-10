from __future__ import annotations

import numpy as np
import pytest

from ennx import ENN
from ennx.ennx.enn_params import ENNParams
from ennx.turbo.config import ENNSurrogateConfig


def _model() -> ENN:
    return ENN(
        np.array([[0.0], [1.0]]),
        np.array([[0.2], [0.8]]),
        y_bounds=np.array([[0.0, 1.0]]),
    )


def test_boundedmodelapi() -> None:
    model = _model()
    np.testing.assert_allclose(model.y_bounds, [[0.0, 1.0]])
    np.testing.assert_allclose(model._ytrain, [[0.2], [0.8]])
    with pytest.raises(ValueError):
        model.add(np.array([[2.0]]), np.array([[1.0]]))


def test_001() -> None:
    model = _model()
    posterior = model.posterior(
        np.array([[0.5]]),
        params=ENNParams(
            k_neighbors=1,
            epistemic_scale=1.0,
            aleatoric_scale=0.0,
        ),
    )
    assert np.all((posterior.mu > 0.0) & (posterior.mu < 1.0))
    draws = posterior.sample(32, np.random.default_rng(0))
    assert np.all((draws > 0.0) & (draws < 1.0))


def test_002() -> None:
    bounds = np.array([[0.0, 1.0]])
    config = ENNSurrogateConfig(y_bounds=bounds)
    np.testing.assert_array_equal(config.y_bounds, bounds)
    with pytest.raises(ValueError, match="shape"):
        ENNSurrogateConfig(y_bounds=np.array([0.0, 1.0]))
