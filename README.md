# ENNX

Rust and Python APIs for nearest-neighbor models and Bayesian optimization.
Includes BoTorch, Optuna, and Ax adapters.

```sh
./ennx build   # Build native code, tests, and Python 3.12–3.14 wheels
./ennx test    # Run native tests and local formal contracts
./ennx dev     # Format, build, and run Python and native tests
./ennx --help
```

Wheels go to `dist/`. Install the wheel matching your Python version and platform.

[API](docs/api.md) · [Integrations](docs/interop.md) ·
[Changelog](CHANGELOG.md) · [Build](docs/buck2.md) · [Tests](docs/testing.md) ·
[Notice](NOTICE)
