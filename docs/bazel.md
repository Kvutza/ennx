# Bazel

Bazel checks the build independently of [Buck2](buck2.md).

```sh
bazel test //:check //:audit --config=release --config=constrained
bazel build //:cpu //:gpu //:wheel --config=release --config=constrained
```

| Target | Purpose |
| --- | --- |
| `//:cpu` | CPU library |
| `//:gpu` | Metal on macOS, OpenCL on Linux |
| `//:wheel` | CPython 3.13 wheel |
| `//:audit` | Wheel artifact checks |
| `//:check` | Test suite |

FAISS, OpenMP, and OpenBLAS sources are pinned. macOS uses Accelerate;
Linux uses OpenBLAS. `//:rust_opencl` builds OpenCL support on macOS.

After changing Cargo manifests, platform triples, or crate annotations:

```sh
CARGO_BAZEL_REPIN=1 bazel build //:cpu
```

Lockfiles stay local. Consumer tests are in `tests/bazel_consumer` and
`tests/python_consumer/smoke.py`. `./ennx dev` does not run Bazel.
