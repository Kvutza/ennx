from __future__ import annotations

import pytest

pytest.importorskip("ennx._rust")
pytestmark = pytest.mark.slow


def test_001():
    from .parspdgate import assert_ci2

    assert_ci2()
