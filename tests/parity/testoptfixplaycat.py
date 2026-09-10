from __future__ import annotations

from pathlib import Path

import pytest

from ennx.turbo.optimizer_fixtures import (
    EXPECTED_OPTIMIZER_FIXTURE_NAMES,
    FIXTURE_GENERATOR_ENTRIES,
    PREFIX_CONFIG,
    catalog_names,
    entry_name,
    name_prefix,
    output_path,
    subdir_entry,
    load_fixture,
)
from ennx.turbo.optimizer_fixtures.replay import FIXTURES_ROOT, _config


def test_catalogfixturenames():
    assert EXPECTED_OPTIMIZER_FIXTURE_NAMES == catalog_names()
    assert len(EXPECTED_OPTIMIZER_FIXTURE_NAMES) == 21


def test_catalog_files():
    actual = {
        path.stem
        for subdir in ("optimizer", "morbo")
        for path in (FIXTURES_ROOT / subdir).glob("*.json")
    }
    assert set(catalog_names()) == actual


def test_001():
    root = Path(__file__).resolve().parents[2]
    for entry in FIXTURE_GENERATOR_ENTRIES:
        assert entry.config_key in PREFIX_CONFIG
        subdir = subdir_entry(entry)
        assert (subdir == "morbo") == entry.morbo
        for seed in (0, 1, 2):
            name = f"{entry.prefix}{seed}"
            assert entry_name(name) == entry
            assert name_prefix(name) == entry.prefix
            assert _config(name) is PREFIX_CONFIG[entry.config_key]
            gen_path = output_path(entry, seed, root)
            assert gen_path == root / "tests" / "fixtures" / subdir / f"{name}.json"
            assert load_fixture(name) is not None


@pytest.mark.parametrize(
    "name", ["missing_seed0", "teucboneseed0.json", "../teucboneseed0"]
)
def test_unknown_fixture(name):
    with pytest.raises(ValueError, match="unknown fixture name"):
        entry_name(name)
