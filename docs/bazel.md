# Bazel

Bazel provides a secondary correctness build. Normal development uses
[`./ennx` and Buck2](buck2.md).

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

Bazel builds checksum-pinned FAISS sources. macOS uses Accelerate and pinned
OpenMP; Linux builds pinned OpenBLAS. `//:rust_opencl` supports explicit OpenCL
builds on macOS.

After changing Cargo manifests, platform triples, or crate annotations:

```sh
CARGO_BAZEL_REPIN=1 bazel build //:cpu
```

Resolver state is generated locally; lockfiles are not tracked.
Consumer checks live in `tests/bazel_consumer` and
`tests/python_consumer/smoke.py`. The normal Buck2 workflow does not run Bazel.
