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

`./ennx build` prepares dependencies and builds in one command. Buck2 downloads
and caches pinned toolchains, including Rust, Clang/LLD, and the Linux sysroot.
Pixi manages only Python wheel verification environments. No system Rust, Python,
or Clang is required; macOS still needs Apple's SDK. Unchanged build results and
existing verification environments are reused.

Cargo manifests declare Rust dependencies; Reindeer generates their Buck2 targets.

`build` checks each Python 3.12–3.14 wheel and writes it to `dist/`.
`dev` runs the full Python suite against every wheel.

[API](docs/api.md) · [Integrations](docs/interop.md) ·
[Changelog](CHANGELOG.md) · [Build](docs/buck2.md) · [Tests](docs/testing.md) ·
[Notice](NOTICE)
