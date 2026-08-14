# Testing `ennx-py`

`ennx-py` is a Python extension crate (PyO3 `cdylib`).
The reliable test path is Python-side after installing the extension.

## Why not `cargo test -p ennx-py` for wrapper behavior?

On some systems, Rust test binaries for PyO3 crates fail to link Python C symbols.
That is an embedding/link mode issue, not a missing algorithm implementation.

## Recommended workflow

From repo root:

1. Run source-only config tests:

```bash
cd /path/to/repo
./ennx python fast
```

2. Build the wheel and run its isolated wheel smoke and API tests:

```bash
cd /path/to/repo
./ennx verify
```

Run the same gate against the CPython 3.12 wheel with:

```bash
pixi run -e ennx-py312 buck2-verify-py312
```

Do not combine `PYTHONPATH=src` with an extension installed only in
`site-packages`; the source package would shadow the installed wheel.

## Rust-side checks that should still pass

```bash
cd /path/to/repo/rust
cargo test -p ennx-bpann
cargo test -p ennx --lib --tests
cargo clippy --all-targets --all-features -- -D warnings
```

Prefer `./ennx rust fast` or `./ennx rust full` for normal repo testing. Use
raw Cargo only when narrowing a Rust-only failure or checking Clippy directly.

Do not use `cargo test --workspace` as the `ennx-py` gate. `ennx-py` is a
PyO3 `cdylib`, and generic Cargo test binaries can fail to link Python C
symbols on macOS even when the wheel builds and Python API tests pass.
