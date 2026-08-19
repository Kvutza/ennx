# Buck2 path

```sh
./ennx build
./ennx test
./ennx wheel
./ennx verify
./ennx ci
./ennx cuda wheel
./ennx cuda parity
```

`tools/buck2-deps` regenerates both `BUCK` dependency graphs from the checked-in
Cargo locks. Buck2 is the primary repo path behind `./ennx`; Bazel remains as a
secondary compatibility and audit path.

Development builds use `opt-level=1` in the `dev` isolation directory. Wheels
use `opt-level=3` in `release`; both graphs stay warm independently.

See [Tracy](tracy.md) for the default-on Rust profiler integration and the
Buck2 profiling workload.
