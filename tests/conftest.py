from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
_SRC = _ROOT / "src"
_PACKAGE = Path(os.environ.get("ENNX_TEST_WHEEL_ROOT", str(_SRC)))
for path in (str(_ROOT), str(_PACKAGE)):
    if path not in sys.path:
        sys.path.insert(0, path)

_NATIVE_FP_CACHE = _ROOT / ".pytest_cache" / "enn_native_extension_fingerprint"
_TESTMON_DB = (
    _ROOT / ".testmondata",
    _ROOT / ".testmondata-shm",
    _ROOT / ".testmondata-wal",
)


def _nativepath() -> Path | None:
    for entry in (_ROOT / "src" / "ennx").glob("ennx_rust*.so"):
        return entry
    return None


def _testmonkey() -> str:
    native_path = _nativepath()
    if native_path is None:
        return "missing"
    native_stat = native_path.stat()
    return f"{native_path}:{native_stat.st_mtime_ns}:{native_stat.st_size}"


def _wipedata() -> None:
    for db_path in _TESTMON_DB:
        if db_path.exists():
            db_path.unlink()


def pytest_configure(config) -> None:
    if not config.pluginmanager.hasplugin("testmon") or config.getoption("no-testmon"):
        return
    fingerprint = _testmonkey()
    previous = _NATIVE_FP_CACHE.read_text() if _NATIVE_FP_CACHE.exists() else None
    if previous == fingerprint:
        return
    _wipedata()
    _NATIVE_FP_CACHE.parent.mkdir(parents=True, exist_ok=True)
    _NATIVE_FP_CACHE.write_text(fingerprint)
