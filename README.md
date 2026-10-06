# ENNX

Bayesian optimization for neural architectures at scale, in Rust and Python.
Works with BoTorch, Optuna, and Ax.

## Quickstart

Hermetic builds via Buck2. Downloads and caches pinned Rust, Clang/LLD, and toolchains automatically:

```sh
./ennx build             # Build library, CLI, and Python 3.12–3.14 wheels
./ennx test              # Run unit tests and accelerator kernels
./ennx test --affected   # Run only tests affected by working changes
./ennx test --python     # Run Python test suite
./ennx dev               # Format, build, and test verification cycle
```

## Bayesian Optimization & Tuning

Run full-space zeroth-order optimization with the live terminal dashboard:

```sh
./ennx tune examples/tuning/code-pretrain.toml
```

## Codebase Radar

Tree-sitter AST queries, reachability analysis, and quality gates:

```sh
./ennx radar find <pattern>   # AST symbol search
./ennx radar impact <target>  # Blast-radius analysis
./ennx radar gate             # Structural quality checks
```

## Hardware Backends

- **Apple Silicon (Metal)**: Zero-copy unified memory buffers and SIMDgroup matrix primitives.
- **NVIDIA CUDA**: PTX synthesis for Turing (SM_75) and Hopper (SM_90a) architectures.

## Documentation

- [API Reference](docs/api.md)
- [Python Integrations](docs/interop.md)
- [Build System](docs/buck2.md)
- [Testing & Quality Gates](docs/testing.md)
- [Architecture & Research](docs/README.md)
