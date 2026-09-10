from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ennx._rust import ENN as RustENN
from ennx._rust import ensure_config_file, set_path


def _assertwritten(cfg: Path) -> None:
    path = ensure_config_file()
    assert Path(path) == cfg
    assert cfg.is_file()
    text = cfg.read_text(encoding="utf-8")
    assert "[bpann]" in text
    assert "index_fragment" in text
    assert "soft_threshold" in text
    assert "hard_threshold" in text
    assert "structured_limit" in text
    assert "search_width" in text
    assert "exhaustive_limit" in text
    assert "skip_limit" in text
    assert "search_limit" in text
    assert "index_fragment = 10000" in text
    assert "soft_threshold = 250" in text
    assert "hard_threshold = 3000" in text
    assert "structured_limit = 1024" in text
    assert "search_width = 1" in text
    assert "exhaustive_limit = 2500" in text
    assert "skip_limit = 150000" in text
    assert "search_limit = 1" in text


def test_001(tmp_path: Path) -> None:
    cfg = tmp_path / "config.toml"
    assert not cfg.exists()
    set_path(str(cfg))
    try:
        _assertwritten(cfg)
    finally:
        set_path(None)


def test_002(tmp_path: Path) -> None:
    """Absent hard must not wipe soft>1000 via full-default-fallback (Q5)."""
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        "[bpann]\nsoft_threshold = 2000\nsearch_width = 1\n",
        encoding="utf-8",
    )
    set_path(str(cfg))
    try:
        path = ensure_config_file()
    finally:
        set_path(None)
    assert Path(path) == cfg
    text = cfg.read_text(encoding="utf-8")
    assert "soft_threshold = 2000" in text
    assert "hard_threshold" not in text


@pytest.fixture
def active_configuration(tmp_path):
    path = tmp_path / "active.toml"
    set_path(str(path))
    try:
        yield path
    finally:
        set_path(None)


@pytest.mark.parametrize(
    "text",
    [
        "[bpann",
        "[bpann]\nsearch_width = 0\n",
        "[bpann]\nsearch_beem_width = 4\n",
        "[bpann]\nsoft_threshold = 5000\nhard_threshold = 1\n",
    ],
)
def test_003(tmp_path, active_configuration, text):
    invalid = tmp_path / "invalid.toml"
    invalid.write_text(text)
    with pytest.raises(ValueError, match="invalid.toml"):
        set_path(str(invalid))
    assert Path(ensure_config_file()) == active_configuration
    assert invalid.read_text() == text


def test_004(tmp_path):
    with pytest.raises(ValueError, match="read config"):
        set_path(str(tmp_path))


def test_005(tmp_path, active_configuration):
    active_configuration.write_text("[bpann")
    with pytest.raises(ValueError, match="parse config"):
        ensure_config_file()
    with pytest.raises(ValueError, match="parse config"):
        RustENN(
            np.zeros((1, 1)),
            np.zeros((1, 1)),
            index_driver="bpann_disk",
            work_dir=str(tmp_path / "observations"),
        )
    assert not (tmp_path / "observations").exists()
