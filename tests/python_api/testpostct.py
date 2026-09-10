"""Contract tests for ENN posterior methods."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from ennx import ENN
from ennx.ennx.enn_params import ENNParams, PosteriorFlags


@pytest.fixture
def simple_model():
    train_x = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float)
    train_y = np.array([[0.0], [1.0], [1.0], [2.0]], dtype=float)
    return ENN(train_x, train_y, scale_x=False)


@pytest.fixture
def query():
    return np.array([[0.5, 0.5]], dtype=float)


class TestPosteriorContract:
    """API contract tests for posterior."""

    def test_001(self):
        assert hasattr(ENN, "posterior")
        assert callable(ENN.posterior)

    def test_002(self, simple_model, query):
        params = ENNParams(
            k_neighbors=2,
            epistemic_scale=1.0,
            aleatoric_scale=0.1,
        )
        flags = PosteriorFlags()
        out = simple_model.posterior(query, params=params, flags=flags)
        assert hasattr(out, "mu")
        assert hasattr(out, "se")
        assert hasattr(out, "se_epi")
        assert hasattr(out, "se_ale")
        assert out.mu.shape == (1, 1)
        assert out.se.shape == (1, 1)
        assert out.se_epi.shape == (1, 1)
        assert out.se_ale.shape == (1, 1)
        assert np.all(np.isfinite(out.mu))
        assert np.all(np.isfinite(out.se))


class Test_001:
    """API contract tests for batch_posterior."""

    def test_003(self):
        assert hasattr(ENN, "batch_posterior")
        assert callable(ENN.batch_posterior)

    def test_004(self):
        sig = inspect.signature(ENN.batch_posterior)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "x" in params
        assert "paramss" in params
        assert "flags" in params

    def test_005(self, simple_model, query):
        params1 = ENNParams(
            k_neighbors=2,
            epistemic_scale=1.0,
            aleatoric_scale=0.1,
        )
        params2 = ENNParams(
            k_neighbors=3,
            epistemic_scale=2.0,
            aleatoric_scale=0.2,
        )
        out = simple_model.batch_posterior(query, paramss=[params1, params2])
        assert hasattr(out, "mu")
        assert hasattr(out, "se")
        assert hasattr(out, "se_epi")
        assert hasattr(out, "se_ale")
        assert out.mu.shape == (2, 1, 1)
        assert out.se.shape == (2, 1, 1)
        assert out.se_epi.shape == (2, 1, 1)
        assert out.se_ale.shape == (2, 1, 1)

    def test_006(self, simple_model, query):
        with pytest.raises(ValueError, match="paramss must be non-empty"):
            simple_model.batch_posterior(query, paramss=[])


class Test_002:
    """API contract tests for conditional_posterior."""

    def test_007(self):
        assert hasattr(ENN, "conditional_posterior")
        assert callable(ENN.conditional_posterior)

    def test_008(self):
        sig = inspect.signature(ENN.conditional_posterior)
        params = list(sig.parameters.keys())
        assert "x_whatif" in params
        assert "y_whatif" in params
        assert "x" in params
        assert "params" in params

    def test_009(self, simple_model, query):
        params = ENNParams(
            k_neighbors=2,
            epistemic_scale=1.0,
            aleatoric_scale=0.1,
        )
        flags = PosteriorFlags()
        x_whatif = np.zeros((0, 2), dtype=float)
        y_whatif = np.zeros((0, 1), dtype=float)

        post = simple_model.posterior(query, params=params, flags=flags)
        cond_post = simple_model.conditional_posterior(
            x_whatif, y_whatif, query, params=params, flags=flags
        )

        np.testing.assert_allclose(post.mu, cond_post.mu, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(post.se, cond_post.se, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(
            post.se_epi, cond_post.se_epi, rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            post.se_ale, cond_post.se_ale, rtol=1e-12, atol=1e-12
        )

    def test_010(self, simple_model, query):
        x_whatif = np.array([[0.5, 0.5]], dtype=float)
        y_whatif = np.array([[1.5]], dtype=float)
        params = ENNParams(
            k_neighbors=2,
            epistemic_scale=1.0,
            aleatoric_scale=0.1,
        )
        flags = PosteriorFlags()

        out = simple_model.conditional_posterior(
            x_whatif, y_whatif, query, params=params, flags=flags
        )
        assert hasattr(out, "mu")
        assert hasattr(out, "se")
        assert hasattr(out, "se_epi")
        assert hasattr(out, "se_ale")
        assert out.mu.shape == (1, 1)
        assert out.se.shape == (1, 1)
        assert out.se_epi.shape == (1, 1)
        assert out.se_ale.shape == (1, 1)


class Test_003:
    """API contract tests for posterior_draw."""

    def test_011(self):
        assert hasattr(ENN, "posterior_draw")
        assert callable(ENN.posterior_draw)

    def test_012(self):
        sig = inspect.signature(ENN.posterior_draw)
        params = list(sig.parameters.keys())
        assert "x" in params
        assert "params" in params
        assert "function_seeds" in params

    def test_013(self, simple_model, query):
        params = ENNParams(
            k_neighbors=2,
            epistemic_scale=1.0,
            aleatoric_scale=0.1,
        )
        flags = PosteriorFlags()
        seeds = [42, 43]

        draws, idx = simple_model.posterior_draw(
            query, params=params, function_seeds=seeds, flags=flags
        )

        assert isinstance(draws, np.ndarray)
        assert draws.shape == (1, 1, 2)
        assert isinstance(idx, (list, np.ndarray))
        assert len(idx) == 1

    def test_014(self):
        assert hasattr(ENN, "conditional_draw")
        assert callable(ENN.conditional_draw)

    def test_015(self, simple_model, query):
        x_whatif = np.array([[0.5, 0.5]], dtype=float)
        y_whatif = np.array([[1.5]], dtype=float)
        params = ENNParams(
            k_neighbors=2,
            epistemic_scale=1.0,
            aleatoric_scale=0.1,
        )
        flags = PosteriorFlags()
        seeds = [42]

        draws, idx = simple_model.conditional_draw(
            x_whatif, y_whatif, query, params=params, function_seeds=seeds, flags=flags
        )

        assert isinstance(draws, np.ndarray)
        assert draws.shape == (1, 1, 1)
        assert isinstance(idx, (list, np.ndarray))
