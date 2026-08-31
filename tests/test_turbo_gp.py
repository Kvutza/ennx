from __future__ import annotations

import numpy as np
import pytest

from ennx.turbo.gp import Surrogate
from ennx.turbo.gp.model import Noisy
from ennx.turbo.gp.train import fit

pytestmark = [pytest.mark.slow, pytest.mark.gp]


def _data(n=12, d=3, m=1):
    rng = np.random.default_rng(0)
    x = rng.random((n, d))
    w = rng.normal(size=(d, m))
    y = x @ w
    return x, y[:, 0] if m == 1 else y


def test_scalar_fit():
    x, y = _data()
    out = fit(x, y, x.shape[1], steps=2)
    assert out.model is not None
    assert out.likelihood is not None
    assert out.std > 0.0


def test_empty_fit():
    out = fit([], [], 2, steps=0)
    assert out.model is None
    assert out.mean == 0.0
    assert out.std == 1.0


def test_noisy_fit():
    x, y = _data()
    out = fit(x, y, x.shape[1], var=np.full(len(x), 0.05), steps=2)
    assert isinstance(out.model, Noisy)


def test_bad_noise_shape():
    x, y = _data()
    with pytest.raises(ValueError, match="shape"):
        fit(x, y, x.shape[1], var=np.ones(len(x) - 1), steps=0)


def test_multi_posterior():
    x, y = _data(m=2)
    gp = Surrogate()
    gp.fit(x, y, steps=2)
    post = gp.predict(x)
    assert post.mu.shape == y.shape
    assert post.sigma.shape == y.shape


def test_seeded_draw():
    x, y = _data()
    gp = Surrogate()
    gp.fit(x, y, steps=2)
    a = gp.draw(x[:4], 3, 42)
    b = gp.draw(x[:4], 3, 42)
    assert a.shape == (3, 4, 1)
    np.testing.assert_array_equal(a, b)
