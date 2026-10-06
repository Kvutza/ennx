# Documentation

ENNX systems, API, and developer documentation.

## Core Reference

- [API Reference](api.md): Python and Rust Prelude APIs.
- [Python Integrations](interop.md): BoTorch, Optuna, and Ax adapters and zero-copy DLPack support.
- [Build System](buck2.md): Hermetic Buck2 builds, pinned toolchains, and wheel generation.
- [Testing & Quality Gates](testing.md): Rust tests, accelerator kernels, KISS quality gate, and `./ennx test --affected`.

## Repository Entrypoints

- `./ennx build`: Compile Rust crates, CLI, and Python 3.12–3.14 wheels.
- `./ennx test`: Run unit tests and GPU kernels (`--affected` for changes only, `--python` for wheel matrix).
- `./ennx tune <config>`: Launch full-space Bayesian optimization with the live terminal UI.
- `./ennx radar <command>`: Query codebase AST, blast radius, callers/callees, and quality gates.
