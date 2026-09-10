# ENNX

Bayesian optimization with epistemic nearest-neighbor models, in Rust and Python.
Works with BoTorch, Optuna, and Ax.

```sh
./ennx build   # Build the library, tests, CLI, and Python wheels
./ennx test    # Run Rust and kernel tests
./ennx dev     # Format, build, and run all standard tests
./ennx --help
```

Python 3.12–3.14 wheels are written to `dist/`.

[API](docs/api.md) · [Integrations](docs/interop.md) ·
[Changelog](CHANGELOG.md) · [Build](docs/buck2.md) · [Tests](docs/testing.md) ·
[Notice](NOTICE)
