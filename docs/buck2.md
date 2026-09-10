# Build

```sh
./ennx build                  # Native code, CLI, tests, and all Python ABI wheels
./ennx build --tests          # Full Python verification instead of smoke tests
./ennx build --out artifacts
./ennx test
./ennx dev                    # Format, build, full Python verification, native tests
./ennx fmt --check
```

Build prepares Python environments and writes CPython 3.12–3.14 wheels to
`dist/`. Dev verifies each wheel once. Tests reuse compiled outputs when inputs
and configuration are unchanged.

Buck2 owns the build graph. Cargo manifests declare Rust dependencies; Reindeer
generates `rust/BUCK`. Update handwritten first-party BUCK targets when changing
crate dependencies or features. Normal commands validate manifests without
regenerating the dependency graph. Resolver locks are ignored local artifacts.

Rust 1.96.0 is checksum-pinned in `toolchains/rust.bzl`. Development uses
optimization level 1; wheels use level 3 and ThinLTO. Build tools use level 0.
Commands share the `dev` isolation directory; override `BUCK_ISOLATION_DIR` for
a separate build cache.

macOS requires Xcode command-line tools and uses Metal. Linux uses OpenCL;
distributed wheels require auditwheel verification against manylinux 2.28.
Python environments are managed internally by Pixi.

CUDA uses a separate Cargo-Oxide action and toolchain; see
[CUDA setup](../cuda/README.md). Its BF16 search API is not implemented by Metal
or OpenCL.

[Bazel](bazel.md) is a separate correctness build. GitHub runs release workflows
only, with no push or pull-request CI gate.
