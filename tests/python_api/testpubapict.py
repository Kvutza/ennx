from __future__ import annotations

import importlib.util
import inspect

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("ennx.ennx_rust") is None,
    reason="ennx_rust extension unavailable",
)


class TestPublicAPIExports:
    """Verify public API surface is exported correctly from ennx package."""

    def test_001(self):
        """All _LAZY_ATTRS entries must be importable."""
        import ennx

        for name in ennx._LAZY_ATTRS:
            attr = getattr(ennx, name)
            assert attr is not None, f"Attribute {name} not found"

    def test_002(self):
        """ENN is a class with expected signature."""
        from ennx import ENN

        assert inspect.isclass(ENN)
        # Check constructor signature has required parameters
        sig = inspect.signature(ENN.__init__)
        params = list(sig.parameters.keys())
        assert "train_x" in params
        assert "train_y" in params
        assert "train_yvar" in params

    def test_003(self):
        from ennx import ENNStatefulFitter

        assert ENNStatefulFitter is not None

    def test_004(self):
        """create_optimizer is a callable function."""
        from ennx import create_optimizer

        assert callable(create_optimizer)

    def test_configclasses(self):
        """All config classes are available and constructible."""
        from ennx import (
            MorboTRConfig,
            NoTRConfig,
            OptimizerConfig,
            TurboTRConfig,
        )

        # These are dataclasses or similar - should be constructible
        assert inspect.isclass(OptimizerConfig)
        assert inspect.isclass(TurboTRConfig)
        assert inspect.isclass(MorboTRConfig)
        assert inspect.isclass(NoTRConfig)

    def test_005(self):
        """All config factory functions are callable."""
        from ennx import (
            lhd_only,
            turbo_enn,
            turbo_one,
            turbo_zero,
        )

        assert callable(turbo_one)
        assert callable(turbo_zero)
        assert callable(turbo_enn)
        assert callable(lhd_only)

    def test_telemetryclass(self):
        """Telemetry class is available."""
        from ennx import Telemetry

        assert inspect.isclass(Telemetry)

    def test_006(self):
        """Experimental namespace is importable and exposes low-level symbols."""
        import pytest

        pytest.importorskip("ennx.ennx_rust")
        import ennx.experimental as experimental_mod
        from ennx import experimental

        assert inspect.ismodule(experimental)
        assert experimental is experimental_mod
        assert experimental.experimental is experimental
        assert inspect.isclass(experimental.SharingPolicy)
        assert inspect.isclass(experimental.MultiTrustRegionLoop)
        assert callable(experimental.make_region)
        assert inspect.isclass(experimental.Optimizer)
        assert inspect.isclass(experimental.MultiTrustRegion)
        assert inspect.isclass(experimental.ResidentBoSession)
        assert callable(experimental.enn_optimizer)
        assert callable(experimental.quantize_int4)
        assert callable(experimental.quantize_e2m1)
        for name in [
            "ParamBlock",
            "ParamBuffer",
            "Proposals",
            "SearchState",
            "turbo_enn",
        ]:
            assert name in experimental.__all__
            assert hasattr(experimental, name)
        if experimental.SearchState is not None:
            assert inspect.isclass(experimental.ParamBlock)
            assert inspect.isclass(experimental.ParamBuffer)
            assert inspect.isclass(experimental.Proposals)
            assert inspect.isclass(experimental.SearchState)
            assert callable(experimental.turbo_enn)

    def test_enumtypes(self):
        """Enum types are available."""
        from ennx import AcqType, CandidateRV, InitStrategy

        assert inspect.isclass(CandidateRV)
        assert inspect.isclass(InitStrategy)
        assert inspect.isclass(AcqType)


class Test_001:
    """Verify public API does not change unexpectedly."""

    def test_007(self):
        """__all__ must match _LAZY_ATTRS keys."""
        import ennx

        assert set(ennx.__all__) == set(ennx._LAZY_ATTRS.keys())

    def test_008(self):
        import ennx

        assert "quantize_int4" not in ennx.__all__
        assert "quantize_e2m1" not in ennx.__all__

    def test_009(self):
        import ennx

        for name in ("quantize_int4", "quantize_e2m1"):
            with pytest.raises(AttributeError):
                getattr(ennx, name)
