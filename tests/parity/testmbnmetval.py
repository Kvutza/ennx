"""Morbo num_metrics validation: Python requires >= 2; Rust must match."""

from __future__ import annotations

import pytest

from ennx.turbo.config import MultiObjectiveConfig


def test_001():
    with pytest.raises(ValueError, match="num_metrics must be >= 2"):
        MultiObjectiveConfig(num_metrics=1)
