# Build

```sh
./ennx build                  # Library, CLI, tests, and Python 3.12–3.14 wheels
./ennx build --tests          # Also run the full Python suite for each wheel
./ennx build --out artifacts
./ennx test
./ennx dev                    # Format, build, and test
./ennx fmt --check
```

Wheels go to `dist/`. The CLI reuses existing Python verification interpreters,
creating missing Pixi environments only when the default `.pixi/envs/...`
interpreter for a wheel is absent. Build results are reused when the inputs and
configuration have not changed.
Set `ENNX_PYTHON_312`, `ENNX_PYTHON_313`, or `ENNX_PYTHON_314` to use an
existing CPython interpreter for a specific wheel verifier instead of Pixi.

Buck2 builds the project. Cargo manifests declare Rust dependencies; Reindeer
generates their targets in `rust/BUCK`. When changing a crate's dependencies or
features, also update its handwritten BUCK target. Normal builds check the
dependency declarations without regenerating the graph. Lockfiles stay local.

Rust 1.96.0 is pinned in `toolchains/rust.bzl`. Development uses optimization
level 1; wheels use level 3 and ThinLTO. Build tools use level 0.
Set `BUCK_ISOLATION_DIR` to use a separate cache; the default is `dev`.

macOS builds require Xcode command-line tools. Metal runs on macOS; OpenCL
requires an installed driver. Linux wheels are audited for manylinux 2.28.

[CUDA](../cuda/README.md) uses a separate toolchain and build.
[Bazel](bazel.md) provides an independent correctness check.

GitHub Actions builds and tests tagged releases.
