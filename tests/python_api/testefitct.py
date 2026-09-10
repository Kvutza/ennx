"""Contract tests for ENNStatefulFitter."""

from __future__ import annotations

import inspect

import numpy as np

from ennx import ENNStatefulFitter, ENN
from ennx.ennx.enn_params import ENNParams


class Test_001:
    def test_initsignature(self):
        sig = inspect.signature(ENNStatefulFitter.__init__)
        params = list(sig.parameters.keys())
        assert "k" in params
        assert "rng" in params
        assert "infer_aleatoric_variance_scale" in params

    def test_001(self):
        assert "tell" in dir(ENNStatefulFitter)
        sig = inspect.signature(ENNStatefulFitter.ask)
        params = list(sig.parameters.keys())
        assert "model" in params
        assert "num_candidates" in params
        assert "num_samples" in params
        assert "params_warm_start" in params

    def test_askreturnsenparams(self):
        train_x = np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            dtype=float,
        )
        train_y = np.array([[0.0], [1.0], [1.0], [2.0]], dtype=float)
        model = ENN(train_x, train_y, scale_x=False)
        rng = np.random.default_rng(42)

        fitter = ENNStatefulFitter(k=2, rng=rng)
        fitter.tell(train_x, train_y)
        result = fitter.ask(
            model,
            num_candidates=3,
            num_samples=2,
        )

        assert isinstance(result, ENNParams)
        assert result.k_neighbors == 2
