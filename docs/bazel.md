# Build and distribution

Bazel resolves dependencies, builds native code, runs tests, and creates the
wheel. Python uses the wheel. Bazel projects use the public targets.

## Public targets

| Target | Contents |
| --- | --- |
| `//:cpu` | Pure CPU target (FAISS CPU + BPANN) |
| `//:gpu` | Hardware-accelerated GPU target (Metal on macOS, OpenCL on Linux) |
| `//:wheel` | Prebuilt Python `.whl` release package |
| `//:audit` | Release wheel artifact verification test |
| `//:check` | Canonical test suite |

`//:python_wheel` and `//:rust_tests` are compatibility aliases for `//:wheel`
and `//:check`.

## Native dependency contract

FAISS is present in every native target. Accelerator drivers augment the
index layer; they do not replace the CPU FAISS capability.

Bazel fetches checksum-pinned FAISS 1.15.0 source and compiles it directly.
The Rust graph does not run the legacy `faiss-sys` CMake build and does not
search Homebrew or another host package manager for FAISS.

On macOS, Bazel uses:

- the pinned LLVM OpenMP module;
- the Apple SDK Accelerate framework for BLAS/LAPACK;
- the system Metal framework for the host-selected wheel.

The explicit `//:rust_opencl` target remains available for OpenCL development
on macOS, while the default macOS wheel contains Metal rather than both GPU
stacks.

Linux selects OpenCL rather than Metal. Bazel also fetches checksum-pinned
OpenBLAS 0.3.32 source and builds a static library for Linux. The build does
not fall back to an ambient `-lopenblas`.

OpenBLAS uses its upstream CMake build through Bazel-managed
`rules_foreign_cc` toolchains. The source, CMake, and Ninja inputs are resolved
by Bazel rather than discovered through a user package manager.

## Dependency locking

`Cargo.Bazel.lock` is the Crate Universe generator lockfile for the complete
Rust dependency graph, including Metal and OpenCL. It is checked in
deliberately: Bzlmod dependency modules cannot repin repositories inside a
consumer's read-only module cache.

The lockfiles cover:

- `aarch64-apple-darwin`;
- `x86_64-unknown-linux-gnu`.

When Cargo manifests, supported triples, or crate annotations change, repin
from the ENNX repository root:

```sh
CARGO_BAZEL_REPIN=1 bazel build //:cpu
```

Consumers must not patch or regenerate these lockfiles.

## Build and test

Build and run tests directly with Bazel:

```sh
bazel test //:check //:audit --config=release --config=constrained
bazel build //:cpu //:gpu //:wheel --config=release --config=constrained
```

Format Bazel files locally:

```sh
bazel run @buildifier_prebuilt//:buildifier -- -r .
```


The wheel is a release artifact produced directly by `//:python_wheel`; local
development and tests do not install it into a separate Python environment.
Consumer workspaces reference the matching release wheel under a
platform-specific `pypi-dependencies` table, as shown in
[`examples/consumer/pixi.toml`](../examples/consumer/pixi.toml).

The local wheel target is tagged for CPython 3.13. Other Python minor versions
require separately compiled wheels; the native extension is not falsely marked
as a stable-ABI wheel.

No user-specific path is part of the build or install contract.

## Consumer checks

The repository contains two owner-boundary smoke fixtures:

- `tests/bazel_consumer` consumes ENNX as a non-root Bzlmod module;
- `tests/python_consumer/smoke.py` imports an installed wheel.

They exist to catch root-only Crate Universe behavior and malformed wheel
layouts before a release is published.
