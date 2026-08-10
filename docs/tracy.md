# Tracy

Tracy is part of every ENNX build. There is no feature flag or ENNX profiling
command. The first instrumented ENNX operation starts the process-wide client,
and a Tracy 0.13.1 profiler can connect to the running process.

ENNX pins `tracy-client` 0.18.4. Its native client is generated into the Buck2
graph by Reindeer and linked into `//rust/crates/ennx:ennx`.

## Live profile

Start Tracy 0.13.1, then run any application linked with ENNX. For the checked-in
posterior workload:

```sh
./buck2w --isolation-dir release run \
  //rust/crates/ennx:posterior_frontier \
  --config ennx.profile=release --local-only --num-threads 4 -- \
  8192 1024 32 32 16 31
```

Connect the profiler to the process on the default Tracy port. ENNX reports CPU
zones for optimizer, strategy, candidate generation, fitting, posterior work,
and neighbor search. Optimizer ask/tell operations are non-continuous frames;
observation, candidate, and arm counts are plots.

On Metal, ENNX also records hardware timestamp-counter zones. KNN traces split
the command buffer into `knn.init`, `knn.distance`, `knn.topk`, and `knn.merge`.
Resident trial search reports `trials.distance`, `trials.score`, `trials.pick`,
and `trials.write`. Timestamp storage is pooled in the shared Metal runtime, so
repeated searches reuse the same 4096-sample ring instead of allocating a
counter buffer per operation.

KNN also reports `ennx.knn.gpu_ns`, `scan_ns`, `select_ns`, and `reduce_ns`
plots alongside rows, queries, dimensions, and `k`. The experimental
`KnnIndex::profile` surface returns the same stage totals for BXL artifacts.

For a headless recording, start the matching upstream capture tool before the
workload:

```sh
tracy-capture -o ennx.tracy -f
tracy-csvexport ennx.tracy
tracy-csvexport -g ennx.tracy
```

## Optimization loop

Use a release Buck2 build, capture the same shape before and after a change,
then compare both CPU self time and GPU execution time. The checked-in KNN
workload is shown above. The resident trial workload is:

```sh
./buck2w --isolation-dir release run \
  //rust/crates/ennx:trial_bench \
  --config ennx.profile=release --local-only --num-threads 4 -- \
  16777216 10 8 10 metal 1
```

For the 8192-row, 1024-query, 32-dimensional KNN workload, Tracy identified
top-k and distance as the dominant Metal stages. The resulting SIMD top-k path
and distance-width autotuning reduced the measured medians from 21.944 ms to
9.892 ms on Metal and from 20.064 ms to 7.817 ms on AGX. These figures are
machine-specific; preserve the workload and compare traces on the target host.

The checked-in BXL frontier includes single-query BPANN shortlist shapes at
1K/128D, 16K/512D, and 65K/1024D. It also races the SIMD scan against an 8x64
SIMD-group Gram tile. Tracy selects the Gram diagram only when query batching
provides enough reuse; single-query reranking retains the SIMD diagram.

The same loop exposed the `k=2048` wide-plan fold as 74.924 ms of a 94.779 ms
GPU search. Both fold inputs were already ordered, so replacing the full
4096-item bitonic sort with an exact parallel merge-path fold reduced that
stage to 5.243 ms, GPU time to 21.618 ms, and wall time from 132.370 ms to
45.470 ms while retaining recall 1.0. These are target-host measurements; use
the checked-in `k` sweep and Tracy stages when evaluating another GPU.

## Application zones

Applications can use the same process-wide client without starting a second
connection:

```rust
let _zone = ennx::tracy::zone(ennx::tracy::span_location!("app.evaluate"));
ennx::tracy::client().message("evaluation started", 0);
```

Keep zone names stable and hierarchical. ENNX uses `optimizer.*`, `strategy.*`,
`candidates.*`, `surrogate.*`, `posterior.*`, `model.*`, and `knn.*`.
