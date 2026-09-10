"""Contract tests for ENNNormal (posterior result type)."""

from __future__ import annotations

import inspect

import numpy as np

from ennx import ENN
from ennx.ennx.enn_normal import ENNNormal
from ennx.ennx.enn_params import ENNParams, PosteriorFlags


class TestENNNormalContract:
    """API contract tests for ENNNormal return type."""

    def test_existsandisclass3(self):
        from ennx.ennx.enn_normal import ENNNormal

        assert inspect.isclass(ENNNormal)

    def test_hasmuseidxattrs(self):
        mu = np.array([[1.0]], dtype=float)
        se = np.array([[0.2]], dtype=float)
        se_epi = se.copy()
        se_ale = np.zeros_like(se)
        obj = ENNNormal(mu=mu, se=se, se_epi=se_epi, se_ale=se_ale)
        assert hasattr(obj, "mu")
        assert hasattr(obj, "se")
        assert hasattr(obj, "se_epi")
        assert hasattr(obj, "se_ale")
        assert hasattr(obj, "idx")
        assert obj.mu is mu
        assert obj.se is se
        assert obj.se_epi is se_epi
        assert obj.se_ale is se_ale
        assert obj.idx is None

    def test_idxoptional(self):
        mu = np.array([[1.0]], dtype=float)
        se = np.array([[0.2]], dtype=float)
        se_epi = se.copy()
        se_ale = np.zeros_like(se)
        idx = np.array([[0, 1]], dtype=int)
        obj = ENNNormal(mu=mu, se=se, se_epi=se_epi, se_ale=se_ale, idx=idx)
        assert obj.idx is idx

    def test_001(self):
        train_x = np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float
        )
        train_y = np.array([[0.0], [1.0], [1.0], [2.0]], dtype=float)
        model = ENN(train_x, train_y, scale_x=False)
        params = ENNParams(
            k_neighbors=2,
            epistemic_scale=1.0,
            aleatoric_scale=0.1,
        )
        flags = PosteriorFlags()
        query = np.array([[0.5, 0.5]], dtype=float)

        out = model.posterior(query, params=params, flags=flags)
        assert isinstance(out, ENNNormal)
        assert out.mu.shape == (1, 1)
        assert out.se.shape == (1, 1)
        assert out.se_epi.shape == (1, 1)
        assert out.se_ale.shape == (1, 1)
        assert np.all(np.isfinite(out.mu))
        assert np.all(np.isfinite(out.se))
        assert np.all(np.isfinite(out.se_epi))
        assert np.all(np.isfinite(out.se_ale))

    def test_samplemethodexists(self):
        mu = np.array([[1.0]], dtype=float)
        se = np.array([[0.2]], dtype=float)
        se_epi = se.copy()
        se_ale = np.zeros_like(se)
        obj = ENNNormal(mu=mu, se=se, se_epi=se_epi, se_ale=se_ale)
        assert hasattr(obj, "sample")
        assert callable(obj.sample)

    def test_samplesignature(self):
        sig = inspect.signature(ENNNormal.sample)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "num_samples" in params
        assert "rng" in params
        assert "clip" in params

    def test_002(self):
        rng = np.random.default_rng(42)
        mu = np.array([[1.0, 2.0]], dtype=float)
        se = np.array([[0.1, 0.2]], dtype=float)
        se_epi = se.copy()
        se_ale = np.zeros_like(se)
        obj = ENNNormal(mu=mu, se=se, se_epi=se_epi, se_ale=se_ale)
        samples = obj.sample(num_samples=10, rng=rng)
        # shape is (*se.shape, num_samples)
        assert samples.shape == (1, 2, 10)
        assert np.all(np.isfinite(samples))
