from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture
def config():
    from ennx import turbo_enn
    from ennx.turbo.config import CandidateGenConfig, ENNFitConfig, ENNSurrogateConfig

    return turbo_enn(
        num_init=2,
        candidates=CandidateGenConfig(num_candidates=16),
        enn=ENNSurrogateConfig(k=4, fit=ENNFitConfig(num_samples=8, num_candidates=8)),
    )


@pytest.fixture(autouse=True)
def wheel_origin():
    import ennx

    root = os.environ.get("ENNX_TEST_WHEEL_ROOT")
    if root:
        assert Path(ennx.__file__).is_relative_to(Path(root))
