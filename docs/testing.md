# Tests

```sh
./ennx dev
./ennx test
ENNX_WHEEL_PATH=dist/ennx-...whl ./ennx test --python
kiss test                    # inside: pixi shell -e ennx
./ennx test --affected       # Test only targets affected by working changes
```

`./ennx dev` formats, builds, and runs Rust and Python tests. Python tests run
against each of the three built wheels, including the BoTorch, Optuna, and Ax
integrations.

`./ennx test` runs Rust unit tests, integration tests, CLI tests, and kernel
tests. GPU checks require the corresponding hardware and driver.

`./ennx test --affected` queries codebase AST reachability through Radar to
discover and run only the test targets impacted by working directory changes.

Inside the `ennx` Pixi shell, `kiss test` is provided by the repository adapter:
Rust tests run through Buck2, while `kiss test --lang python` delegates to
upstream KISS. Pixi and uv continue to own the Python verification environments.

`./ennx fmt` is the formatting and source-policy gate. In addition to Rust and
Python formatting, it runs KISS for the repository and for every Rust crate. It
also rejects sentence-like code names: local Python/Rust definitions and source
file stems may contain at most two snake-case words (one underscore) and 24
characters. Put the explanation in a comment or docstring, not in the name.

For `--python`, replace `ennx-...whl` with the wheel filename. Python 3.13 is the
default; set `ENNX_PYTHON_VERSION=3.12` or `3.14` to test another wheel.

The standard Python suite selects `not slow or gp`. Other stress and performance
tests run separately. Bazel checks are also separate.

If a local `formal/` directory exists, `./ennx test` also checks its Lean proofs.
These cover the stated contracts, not GPU kernels or drivers.
