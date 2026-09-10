# Tests

```sh
./ennx dev
./ennx test
ENNX_WHEEL_PATH=dist/ennx-...whl ./ennx test --python
```

Dev formats, builds all supported Python ABI wheels, verifies each against the
Python suite, and runs native tests. Standalone Python tests require an existing
wheel; set `ENNX_PYTHON_VERSION` for Python 3.12 or 3.14.

Native tests cover Rust unit and integration tests, the CLI, and platform
kernels. The Python suite runs against the built wheel, including BO adapters,
with selection `not slow or gp`. Stress, performance, and additional hardware
checks require explicit runs.

When local `formal/` exists, test also builds its Lean contracts and rejects
proof holes. These contracts do not prove GPU kernels, compilers, or drivers.

Use fixed seeds and independent reference results. Test edge cases, errors,
state transitions, and backend parity. State whether comparisons require
bitwise equality or a numerical tolerance. Measure performance separately.

Bazel consumer checks are separate. GitHub automation runs tagged releases,
not push or pull-request CI. Passing dev does not establish complete backend
parity or performance.
