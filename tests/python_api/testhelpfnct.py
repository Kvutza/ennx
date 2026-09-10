from __future__ import annotations

import inspect

import numpy as np
import pytest


class TestHypervolumeContract:
    """API contract tests for hypervolume2d_max function."""

    def test_001(self):
        """hypervolume2d_max must exist and be callable."""
        from ennx.turbo.hypervolume import hypervolume2d_max

        assert callable(hypervolume2d_max)

    def test_signaturecontract3(self):
        """Function signature must match contract: (y, ref_point) -> float."""
        from ennx.turbo.hypervolume import hypervolume2d_max

        sig = inspect.signature(hypervolume2d_max)
        params = list(sig.parameters.keys())
        assert params == ["y", "ref_point"]
        # Note: return_annotation may be string 'float' due to future annotations
        assert str(sig.return_annotation) in ("float", "<class 'float'>")

    def test_002(self):
        """Valid 2D input returns non-negative float."""
        from ennx.turbo.hypervolume import hypervolume2d_max

        y = np.array([[1.0, 0.5], [0.5, 1.0]])
        ref = np.array([0.0, 0.0])
        result = hypervolume2d_max(y, ref)
        assert isinstance(result, float)
        assert result >= 0.0

    def test_003(self):
        """Empty input array returns 0.0."""
        from ennx.turbo.hypervolume import hypervolume2d_max

        y = np.array([]).reshape(0, 2)
        ref = np.array([0.0, 0.0])
        result = hypervolume2d_max(y, ref)
        assert result == 0.0

    def test_004(self):
        """When no points dominate ref_point, returns 0.0."""
        from ennx.turbo.hypervolume import hypervolume2d_max

        y = np.array([[-1.0, -1.0], [-0.5, -0.5]])  # All below ref
        ref = np.array([0.0, 0.0])
        result = hypervolume2d_max(y, ref)
        assert result == 0.0

    def test_invalidyndimraises(self):
        """1D y array raises ValueError."""
        from ennx.turbo.hypervolume import hypervolume2d_max

        y = np.array([1.0, 0.5])
        ref = np.array([0.0, 0.0])
        with pytest.raises(ValueError):
            hypervolume2d_max(y, ref)

    def test_invalidyshaperaises(self):
        """y with wrong second dimension raises ValueError."""
        from ennx.turbo.hypervolume import hypervolume2d_max

        y = np.array([[1.0, 0.5, 0.3]])  # 3D instead of 2D
        ref = np.array([0.0, 0.0])
        with pytest.raises(ValueError):
            hypervolume2d_max(y, ref)

    def test_005(self):
        """ref_point with wrong shape raises ValueError."""
        from ennx.turbo.hypervolume import hypervolume2d_max

        y = np.array([[1.0, 0.5]])
        ref = np.array([0.0])  # Wrong shape
        with pytest.raises(ValueError):
            hypervolume2d_max(y, ref)


class TestEnnHashContract:
    """API contract tests for enn_hash RNG functions."""

    def test_006(self):
        """normal_philox function exists."""
        from ennx.ennx.enn_hash import normal_philox

        assert callable(normal_philox)

    def test_007(self):
        """normal_hash function exists."""
        from ennx.ennx.enn_hash import normal_hash

        assert callable(normal_hash)

    def test_008(self):
        """Hash functions have signature (seeds, indices, num_metrics) -> array."""
        from ennx.ennx.enn_hash import (
            normal_philox,
            normal_hash,
        )

        for fn in [normal_philox, normal_hash]:
            sig = inspect.signature(fn)
            params = list(sig.parameters.keys())
            assert params == ["function_seeds", "data_indices", "num_metrics"]

    def test_009(self):
        """Same inputs produce same outputs (determinism)."""
        from ennx.ennx.enn_hash import normal_hash

        seeds = np.array([42], dtype=np.int64)
        indices = np.array([[0, 1]], dtype=int)

        result1 = normal_hash(seeds, indices, num_metrics=2)
        result2 = normal_hash(seeds, indices, num_metrics=2)

        assert np.allclose(result1, result2)

    def test_010(self):
        """Output shape is (num_seeds, *data_indices.shape, num_metrics)."""
        from ennx.ennx.enn_hash import normal_hash

        seeds = np.array([1, 2], dtype=np.int64)  # 2 seeds
        indices = np.array([[0, 1, 2], [3, 4, 5]])  # shape (2, 3)
        num_metrics = 4

        result = normal_hash(seeds, indices, num_metrics)

        assert result.shape == (2, 2, 3, 4)

    def test_011(self):
        """Different seeds produce different outputs."""
        from ennx.ennx.enn_hash import normal_hash

        seeds1 = np.array([42], dtype=np.int64)
        seeds2 = np.array([99], dtype=np.int64)
        indices = np.array([[0, 1]], dtype=int)

        result1 = normal_hash(seeds1, indices, num_metrics=2)
        result2 = normal_hash(seeds2, indices, num_metrics=2)

        assert not np.allclose(result1, result2)

    def test_012(self):
        """num_metrics <= 0 raises ValueError."""
        from ennx.ennx.enn_hash import normal_hash

        seeds = np.array([42], dtype=np.int64)
        indices = np.array([[0, 1]], dtype=int)

        with pytest.raises(ValueError):
            normal_hash(seeds, indices, num_metrics=0)

        with pytest.raises(ValueError):
            normal_hash(seeds, indices, num_metrics=-1)


class Test_001:
    """API contract tests for WeightedStats dataclass."""

    def test_weightedstatsexists(self):
        """WeightedStats dataclass exists."""
        from ennx.ennx.weighted_stats import WeightedStats

        assert inspect.isclass(WeightedStats)

    def test_weightedstatsfields(self):
        """WeightedStats has expected fields."""
        from ennx.ennx.weighted_stats import WeightedStats

        # Create instance with dummy data
        ws = WeightedStats(
            w_normalized=np.array([0.5, 0.5]),
            l2=np.array([1.0, 2.0]),
            mu=np.array([0.0, 1.0]),
            se=np.array([0.1, 0.2]),
            se_epi=np.array([0.1, 0.2]),
            se_ale=np.array([0.0, 0.0]),
        )

        assert hasattr(ws, "w_normalized")
        assert hasattr(ws, "l2")
        assert hasattr(ws, "mu")
        assert hasattr(ws, "se")
        assert hasattr(ws, "se_epi")
        assert hasattr(ws, "se_ale")

    def test_013(self):
        """WeightedStats is frozen (immutable)."""
        from ennx.ennx.weighted_stats import WeightedStats

        ws = WeightedStats(
            w_normalized=np.array([0.5]),
            l2=np.array([1.0]),
            mu=np.array([0.0]),
            se=np.array([0.1]),
            se_epi=np.array([0.1]),
            se_ale=np.array([0.0]),
        )

        # Attempting to modify should raise
        with pytest.raises((AttributeError, TypeError)):
            ws.mu = np.array([1.0])


class TestEnnUtilContract:
    """API contract tests for enn_util functions."""

    def test_standardizeyexists(self):
        """standardize_y function exists."""
        from ennx.ennx.enn_util import standardize_y

        assert callable(standardize_y)

    def test_014(self):
        """standardize_y returns (center, scale) tuple."""
        from ennx.ennx.enn_util import standardize_y

        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        center, scale = standardize_y(y)

        assert isinstance(center, float)
        assert isinstance(scale, float)

    def test_015(self):
        """sobol_indices function exists."""
        from ennx.ennx.enn_util import sobol_indices

        assert callable(sobol_indices)

    def test_016(self):
        """sobol_indices has signature (x, y) -> array."""
        from ennx.ennx.enn_util import sobol_indices

        sig = inspect.signature(sobol_indices)
        params = list(sig.parameters.keys())
        assert params == ["x", "y"]

    def test_017(self):
        """pareto2d_max function exists."""
        from ennx.ennx.enn_util import pareto2d_max

        assert callable(pareto2d_max)

    def test_018(self):
        """pareto_arms function exists."""
        from ennx.ennx.enn_util import pareto_arms

        assert callable(pareto_arms)
