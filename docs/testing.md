# Tests

```sh
./ennx dev
./ennx test
ENNX_WHEEL_PATH=dist/ennx-...whl ./ennx test --python
```

`./ennx dev` formats, builds, and runs Rust and Python tests. Python tests run
against each of the three built wheels, including the BoTorch, Optuna, and Ax
integrations.

`./ennx test` runs Rust unit tests, integration tests, CLI tests, and kernel
tests. GPU checks require the corresponding hardware and driver.

For `--python`, replace `ennx-...whl` with the wheel filename. Python 3.13 is the
default; set `ENNX_PYTHON_VERSION=3.12` or `3.14` to test another wheel.

The standard Python suite selects `not slow or gp`. Other stress and performance
tests run separately. [Bazel checks](bazel.md) are also separate.

If a local `formal/` directory exists, `./ennx test` also checks its Lean proofs.
These cover the stated contracts, not GPU kernels or drivers.
