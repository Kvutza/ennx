# Buck2 path

```sh
./ennx dev
./ennx dev --full
./ennx ci
./ennx wheel
./ennx tune CONFIG.toml
```

`tools/buck2-deps` regenerates both `BUCK` dependency graphs from the checked-in
Cargo locks. Buck2 is the primary repo path behind `./ennx`; Bazel remains as a
secondary compatibility and audit path.

Development builds use `opt-level=1` in the `dev` isolation directory. Wheels
use `opt-level=3` in `release`; both graphs stay warm independently.

Use Buck2 benchmark targets and platform profilers for performance work.
