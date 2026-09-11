# Build

```sh
./ennx build                  # Library, CLI, tests, and Python 3.12–3.14 wheels
./ennx build --tests          # Also run the full Python suite for each wheel
./ennx build --out artifacts
./ennx test
./ennx dev                    # Format, build, and test
./ennx fmt --check
```

Wheels go to `dist/`. `./ennx build` prepares dependencies and builds in one command.
DotSlash bootstraps Buck2. Buck downloads and caches pinned Rust, Python,
Clang/LLD, and the Linux sysroot and support libraries. Builds always use these
managed toolchains. No system Rust, Python, or Clang installation is needed.
Build results are reused when inputs and configuration have not changed.

Pixi manages only Python wheel verification environments in `.pixi/envs/`.
Missing environments are prepared automatically; existing environments are reused.
On Linux, these environments include `auditwheel`, `patchelf`, and `binutils`
(for `readelf`).
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
macOS requires Apple's SDK (`xcode-select --install`); the managed compiler uses
its C++ headers and system runtime. On Linux, Buck supplies the native sysroot,
headers, and support libraries.
Metal runs on macOS; OpenCL requires an installed driver. Linux wheels are audited
for manylinux 2.28, so release builds need a compatible build host.

[CUDA](../cuda/README.md) uses a separate toolchain and build.
[Bazel](bazel.md) provides an independent correctness check.

GitHub Actions builds and tests tagged releases.
