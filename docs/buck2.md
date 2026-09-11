# Build

```sh
./ennx build                  # Library, CLI, tests, and Python 3.12–3.14 wheels
./ennx build --tests          # Also run the full Python suite for each wheel
./ennx build --out artifacts
./ennx test
./ennx dev                    # Format, build, and test
./ennx fmt --check
```

Wheels go to `dist/`. `./ennx build` prepares missing prerequisites automatically.
DotSlash downloads verified, pinned Buck2 and Pixi binaries; an installed Pixi is
reused. Python verification environments live in `.pixi/envs/`, with missing wheel
utilities in `.pixi/envs/build-tools/`. Existing usable environments are reused
without running Pixi. Build results are reused when inputs and configuration
have not changed. No system Rust or Python installation is needed.
Set `ENNX_PYTHON_312`, `ENNX_PYTHON_313`, or `ENNX_PYTHON_314` to use an
existing CPython interpreter for a specific wheel verifier instead of Pixi.
Overrides must include the verification dependencies declared in `pixi.toml`;
invalid overrides fail before compilation and are never modified.

Buck2 builds the project. Cargo manifests declare Rust dependencies; Reindeer
generates their targets in `rust/BUCK`. When changing a crate's dependencies or
features, also update its handwritten BUCK target and run `tools/buck2-deps`
(requires Cargo). Normal builds use the checked-in graph. Lockfiles stay local.

Rust 1.96.0 is pinned in `toolchains/rust.bzl`. Development uses optimization
level 1; wheels use level 3 and ThinLTO. Build tools use level 0.
Set `BUCK_ISOLATION_DIR` to use a separate cache; the default is `dev`.

Supported hosts are Apple Silicon macOS and x86_64/aarch64 Linux. Bootstrap needs
standard shell utilities, `curl`, `tar`, and `shasum` or `sha256sum`.
macOS requires Xcode command-line tools (`xcode-select --install`). Linux requires
Clang, LLD and system C/C++ development headers. These native prerequisites are
checked before setup; install them through the platform's system tools.
Metal runs on macOS; OpenCL requires an installed driver. Linux wheels are audited
for manylinux 2.28, so release builds need a compatible build host.

[CUDA](../cuda/README.md) uses a separate toolchain and build.
[Bazel](bazel.md) provides an independent correctness check.

GitHub Actions builds and tests tagged releases.
