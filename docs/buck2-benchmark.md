# Buck2 pilot evidence

## Optimized graph

The final graph keeps generated third-party crates in `rust/BUCK` and defines
BPANN, ENNX, and PyO3 beside their sources. Stable `dev` and `release`
isolation directories keep both profiles hot.

```text
warm full gate:       2.71 s
warm release smoke:   0.91 s
ENNX edit + Metal:    7.77 s
```

The full gate regenerates and verifies dependency locks, runs FAISS, BPANN,
Metal, and OpenCL tests, builds the release wheel, audits its loader paths,
imports it in a clean directory, and calls the compiled ENNX extension.

Measured 2026-07-28 on an Apple M4 MacBook Air with 10 cores, 24 GB memory,
macOS 26.5.1, Buck2 `2026-07-14-1560aca`, Bazel 9.2.0, and Rust 1.88.0.
All builds used four local threads.

## Results

| Case | Buck2 | Bazel |
| --- | ---: | ---: |
| Warm Metal test, median of 5 | 0.10 s | 0.22 s |
| Rebuild after the same Rust content edit | 7.57 s | 7.86 s |
| Cold Metal test | 153.14 s | 229.11 s |

The edit added `const _BUILD_REBUILD_PROBE: () = ();` to
`rust/crates/ennx/src/weights.rs`. Its SHA-256 changed from
`d92059858888240131d5a1a1c4838ca206db57e4cd8e3bda3fcc9aba2758f8d0`
to `cac0c9332526aff1ea988133527e337c3da15ee141a274ccb5d6a27add37a172`
and was restored byte-for-byte.

## Warm runs

Buck2 command, repeated five times:

```sh
/usr/bin/time -lp ./buck2w test //buck2/tests:metal \
  --local-only --num-threads 4 --console none
```

```text
wall_s  max_rss_bytes
0.10    38141952
0.10    38158336
0.10    38158336
0.10    38158336
0.10    38158336
```

Bazel command, repeated five times:

```sh
/usr/bin/time -lp bazel test //rust/crates/ennx:trial_search_metal_test \
  --config=release --config=constrained --noshow_progress \
  --ui_event_filters=-info,-stdout,-stderr
```

```text
wall_s  max_rss_bytes
0.19    16941056
0.19    16973824
0.23    17006592
0.22    17039360
0.22    17039360
```

The Bazel sample started only after one release/configuration warm-up run had
completed; that warm-up was excluded.

The RSS values above cover each client process, not its persistent daemon.
Post-run daemon snapshots were 380,992 KiB for Buck2 and 696,016 KiB for
Bazel; these are not peak measurements.

## Content-changing rebuild

Both commands used the same edited source:

```sh
/usr/bin/time -lp ./buck2w test //buck2/tests:metal \
  --local-only --num-threads 4 --console simple
# 7.57 real; 38,338,560 bytes client max RSS; 2 local actions; PASS

/usr/bin/time -lp bazel test //rust/crates/ennx:trial_search_metal_test \
  --config=release --config=constrained --noshow_progress \
  --ui_event_filters=-info,-stdout,-stderr
# 7.86 real; 16,990,208 bytes client max RSS; PASS
```

## Cold runs

Buck2 used a never-created isolation directory and disabled remote cache. No
cache was deleted.

```sh
/usr/bin/time -lp ./buck2w --isolation-dir bench-cold-20260728a \
  test //buck2/tests:metal --local-only --no-remote-cache \
  --num-threads 4 --console simple
```

```text
153.14 real
42,352,640 bytes client max RSS
459 local commands; 0 cache hits; PASS
isolated daemon after run: 374,720 KiB
isolated output: 680 MiB
```

Bazel used new output and repository-cache directories. The Bazel/Bazelisk
executable was already installed, but no build or repository cache was reused.

```sh
/usr/bin/time -lp bazel \
  --output_base=/tmp/ennx-bazel-cold-20260728a \
  test //rust/crates/ennx:trial_search_metal_test \
  --config=release --config=constrained \
  --repository_cache=/tmp/ennx-bazel-repo-cache-20260728a \
  --noshow_progress --ui_event_filters=-info,-stdout,-stderr
```

```text
229.11 real
17,465,344 bytes client max RSS
target verification: cached PASS in 0.701 Bazel seconds
server after run: 710,160 KiB
output base: 239 MiB
repository cache: 2.8 GiB
```

The timing harness's final status was nonzero because its trailing process
probe missed Bazel's `/private/tmp` path. The timed Bazel test itself completed,
and the immediate verification above confirmed the target and cached test pass.

## Runtime parity

Bazel's release Rust actions use `opt-level=3`, no debug info, stripped debug
info, edition 2021, and `aarch64-apple-darwin`. Buck2 uses edition 2021,
`aarch64-apple-darwin`, `opt-level=3`, disabled debug assertions, and disabled
overflow checks. The source FAISS action uses `-O2`, OpenMP, and macOS 11 as
its deployment target.

## Wheel

```sh
./ennx verify
```

```text
@rpath/libomp.dylib
path @loader_path/.dylibs
wheel smoke: center=2.0, scale=0.8164965809277261
ennx-0.0.0-cp313-cp313-macosx_11_0_arm64.whl
```

The smoke extracts into a new empty temporary directory, rejects host-specific
library paths, imports with CPython 3.13 without loader environment variables,
and calls the compiled ENNX extension.

## Dependency regeneration

```sh
tools/buck2-deps
```

SHA-256 before and after regeneration:

```text
rust/BUCK                 dac58f1c8ab3b1d1afe3663ee0c6c31c290c0054fe5f7aa923d9dce561768cca
buck2/accelerators/BUCK   85c3c29e469c5ea4f9709d5461f67e549055bd51477e087b5a7e2cb6d8359e61
pixi.lock                 059d2090ef1ffa0f9436aa58728860d7f596a2bc1b9ecbf6117ab3858bceb44a
```

Both generated files were identical, `pixi.lock` was unchanged, and
`buck2/accelerators/Cargo.lock` matched `Cargo.Accelerators.lock`.

## Linux arm64

A clean Debian 12 container used Rust 1.88, Clang, LLD, system
FAISS/OpenBLAS/OpenCL, and PoCL. With four threads:

```text
BPANN test:        PASS
OpenCL test:       PASS
ENNX library:      BUILD SUCCEEDED
```

The source ran on the container's native filesystem. Colima's macOS bind mount
cannot preserve links in Buck's downloaded CPython toolchain.

## Limits

- macOS arm64 is fully exercised. Linux arm64 covers BPANN, OpenCL, ENNX, and
  clean wheel import.
- Linux imports pinned FAISS/OpenBLAS from Pixi; the OpenCL loader remains
  system-provided.
- Linux requires Clang and LLD.
- Rust and Buck2 are pinned. The native FAISS action still uses Apple's
  `/usr/bin/clang++`, `bsdtar`, and `libtool`; that part is not a pinned compiler
  toolchain.
- Client RSS and post-run daemon RSS are reported separately; peak daemon CPU
  and RSS were not captured.
