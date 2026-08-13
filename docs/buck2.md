# Buck2 pilot

```sh
./ennx build
./ennx test
./ennx wheel
./ennx verify
./ennx ci
```

`tools/buck2-deps` regenerates both `BUCK` dependency graphs from the checked-in
Cargo locks. Bazel remains intact; removing the pilot means removing the Buck2
files and the wrapper tasks behind `./ennx`.

Development builds use `opt-level=1` in the `dev` isolation directory. Wheels
use `opt-level=3` in `release`; both graphs stay warm independently.

See [Tracy](tracy.md) for the default-on Rust profiler integration and the
Buck2 profiling workload.
