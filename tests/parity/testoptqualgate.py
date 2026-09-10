from __future__ import annotations

import pytest

pytest.importorskip("ennx._rust")
pytestmark = pytest.mark.slow


def test_001():
    from .parqualgate import assert_ci

    assert_ci()
