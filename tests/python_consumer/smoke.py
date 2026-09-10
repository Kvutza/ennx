from __future__ import annotations

import ennx
import ennx.experimental
import ennx.experimental.mtrregn

assert ennx.__file__
assert ennx.experimental.__file__
assert ennx.experimental.mtrregn.__file__

try:
    import ennx.ennx_rust
except ImportError:
    print("ennx wheel consumer: skipped (ennx_rust unavailable)")
else:
    assert ennx.ennx_rust.__file__
    print("ennx wheel consumer: ok")
