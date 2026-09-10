# ENNX

Bayesian optimization with epistemic nearest-neighbor models, in Rust and Python.
Works with BoTorch, Optuna, and Ax.

The Rust library and Python bindings share a build managed through `./ennx`.

```sh
./ennx build   # Build the library, CLI, tests, and Python wheels
./ennx test    # Run Rust and kernel tests
./ennx dev     # Format, build, and run Rust and Python tests
./ennx --help
```

Buck2 runs the build. Cargo manifests declare Rust dependencies; Reindeer
generates their Buck2 targets. Pixi manages the Python test environments.
Unchanged build results are reused across commands.

`build` checks each Python 3.12–3.14 wheel and writes it to `dist/`.
`dev` runs the full Python suite against every wheel.

[API](docs/api.md) · [Integrations](docs/interop.md) ·
[Changelog](CHANGELOG.md) · [Build](docs/buck2.md) · [Tests](docs/testing.md) ·
[Notice](NOTICE)
