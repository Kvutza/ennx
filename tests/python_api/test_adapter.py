from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from ennx import _rust


class StubGP:
    def __init__(self) -> None:
        self.fits: list[tuple[np.ndarray, np.ndarray, np.ndarray | None]] = []
        self.preds: list[np.ndarray] = []

    def fit(self, x, y, yvar, *, steps):
        assert steps == 7
        self.fits.append(
            (
                np.asarray(x).copy(),
                np.asarray(y).copy(),
                None if yvar is None else np.asarray(yvar).copy(),
            )
        )
        return SimpleNamespace(lengthscales=np.ones(x.shape[1]))

    def predict(self, x):
        x = np.asarray(x)
        self.preds.append(x.copy())
        return SimpleNamespace(
            mu=x.sum(1, keepdims=True),
            sigma=np.full((len(x), 1), 0.1),
        )

    def draw(self, x, n, seed):
        return np.random.default_rng(seed).normal(size=(n, len(x), 1))


def _make(gp: StubGP):
    bounds = np.array([[-2.0, 2.0], [10.0, 20.0]])
    opt = _rust.create_optimizer(
        bounds,
        "gp",
        10,
        1,
        4,
        42,
        cfg={
            "acquisition": "ucb",
            "min_candidates": 8,
            "max_candidates": 8,
            "num_candidates_factor": 1.0,
        },
        gp=gp,
        fit_steps=7,
    )
    return bounds, opt


def test_gpcallsarebatched():
    gp = StubGP()
    bounds, opt = _make(gp)
    x = opt.ask(1, 101)
    y = np.array([[1.25]])
    yvar = np.array([[0.05]])
    opt.tell(x, y, 102, yvar)

    assert np.all((bounds[:, 0] <= x) & (x <= bounds[:, 1]))
    assert len(gp.fits) == 1
    fit_x, fit_y, fit_var = gp.fits[0]
    assert np.all((0.0 <= fit_x) & (fit_x <= 1.0))
    np.testing.assert_array_equal(fit_y, y)
    np.testing.assert_array_equal(fit_var, yvar)

    assert opt.ask(2, 103).shape == (2, 2)
    assert len(gp.preds) == 1
    assert gp.preds[0].shape == (8, 2)


def test_yvarerrorisatomic():
    gp = StubGP()
    _, opt = _make(gp)
    x = opt.ask(1, 201)
    opt.tell(x, np.array([[1.0]]), 202, np.array([[0.1]]))

    with pytest.raises(ValueError, match="y_var must be provided"):
        opt.tell(x, np.array([[2.0]]), 203)

    assert opt.region_count() == 1
    assert len(gp.fits) == 1


def test_fiterrorisatomic():
    class FailOnce(StubGP):
        fail = True

        def fit(self, x, y, yvar, *, steps):
            if self.fail:
                self.fail = False
                raise RuntimeError("fit failed")
            return super().fit(x, y, yvar, steps=steps)

    gp = FailOnce()
    _, opt = _make(gp)
    x = opt.ask(1, 301)

    with pytest.raises(ValueError, match="fit failed"):
        opt.tell(x, np.array([[1.0]]), 302)

    assert opt.region_count() == 0
    assert opt.init_progress() == (0, 1)
    opt.tell(x, np.array([[1.0]]), 303)
    assert opt.region_count() == 1
    assert opt.init_progress() is None


def test_001():
    class BadPredict(StubGP):
        def predict(self, x):
            return SimpleNamespace(mu=np.zeros((1, 1)), sigma=np.ones((1, 1)))

    gp = BadPredict()
    _, opt = _make(gp)
    x = opt.ask(1, 401)
    opt.tell(x, np.array([[1.0]]), 402)

    with pytest.raises(ValueError, match="predict must return"):
        opt.ask(2, 403)
