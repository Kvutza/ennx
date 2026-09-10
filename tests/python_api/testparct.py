"""Contract tests for ENNParams and PosteriorFlags."""

from __future__ import annotations

import inspect

import pytest

from ennx.ennx.enn_params import ENNParams, PosteriorFlags


class TestENNParamsContract:
    """API contract tests for ENNParams."""

    def test_existsandisclass3(self):
        from ennx.ennx.enn_params import ENNParams

        assert inspect.isclass(ENNParams)

    def test_signaturecontract3(self):
        sig = inspect.signature(ENNParams)
        params = list(sig.parameters.keys())
        assert "k_neighbors" in params
        assert "epistemic_scale" in params
        assert "aleatoric_scale" in params

    def test_001(self):
        p = ENNParams(
            k_neighbors=2,
            epistemic_scale=1.0,
            aleatoric_scale=0.1,
        )
        assert p.k_neighbors == 2
        assert p.epistemic_scale == 1.0
        assert p.aleatoric_scale == 0.1

    def test_invalidkraises(self):
        with pytest.raises(ValueError, match="k_neighbors"):
            ENNParams(
                k_neighbors=0,
                epistemic_scale=1.0,
                aleatoric_scale=0.0,
            )

    def test_002(self):
        with pytest.raises(ValueError, match="epistemic_scale"):
            ENNParams(
                k_neighbors=2,
                epistemic_scale=-1.0,
                aleatoric_scale=0.0,
            )

    def test_003(self):
        with pytest.raises(ValueError, match="aleatoric_scale"):
            ENNParams(
                k_neighbors=2,
                epistemic_scale=1.0,
                aleatoric_scale=-0.1,
            )

    def test_isfrozen(self):
        p = ENNParams(
            k_neighbors=2,
            epistemic_scale=1.0,
            aleatoric_scale=0.0,
        )
        with pytest.raises((AttributeError, TypeError)):
            p.k_neighbors = 3


class Test_001:
    """API contract tests for PosteriorFlags."""

    def test_existsandisclass3(self):
        from ennx.ennx.enn_params import PosteriorFlags

        assert inspect.isclass(PosteriorFlags)

    def test_defaultvalues(self):
        flags = PosteriorFlags()
        assert flags.exclude_nearest is False
        assert flags.observation_noise is False
        assert flags.tie_neighbors is True

    def test_explicitvalues(self):
        flags = PosteriorFlags(
            exclude_nearest=True,
            observation_noise=True,
            tie_neighbors=False,
        )
        assert flags.exclude_nearest is True
        assert flags.observation_noise is True
        assert flags.tie_neighbors is False

    def test_signaturecontract3(self):
        sig = inspect.signature(PosteriorFlags)
        params = list(sig.parameters.keys())
        assert "exclude_nearest" in params
        assert "observation_noise" in params
        assert "tie_neighbors" in params
