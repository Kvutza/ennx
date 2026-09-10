# Changelog

## 0.2.0 — Unreleased

`0.2.0` is a breaking API release. It removes the compatibility aliases from
`0.1.x`; callers must use the names below.

### Python API

BO integrations are available by default as `ennx.botorch.Model`,
`ennx.botorch.Posterior`, `ennx.botorch.Sampler`, `ennx.optuna.Sampler`,
and `ennx.ax.Node`. BoTorch, Optuna, and Ax are now required Python dependencies;
no integration extras are needed. BoTorch
uses ENNX's native joint draws without an autograd bridge; Optuna and Ax use
the existing optimizer for fixed continuous, single-objective experiments.
See [integration contracts and examples](docs/interop.md) for supported paths
and restart limitations.

The primary model is now `ENN`:

| `0.1.x` | `0.2.0` |
| --- | --- |
| `EpistemicNearestNeighbors` | `ENN` |
| `create_optimizer_enn` | `enn_optimizer` |
| `create_optimizer_zero` | `create_zero` |
| `create_optimizer_lhd` | `create_lhd` |
| `turbo_one_config` | `turbo_one` |
| `turbo_zero_config` | `turbo_zero` |
| `turbo_enn_config` | `turbo_enn` |
| `lhd_only_config` | `lhd_only` |

The general Python optimizer entry point remains
`create_optimizer(bounds, config, rng)`. Its `ask`, `tell`, and `telemetry`
workflow is unchanged.

`ENNParams` uses shorter field names:

| `0.1.x` | `0.2.0` |
| --- | --- |
| `k_num_neighbors` | `k_neighbors` |
| `epistemic_variance_scale` | `epistemic_scale` |
| `aleatoric_variance_scale` | `aleatoric_scale` |

`ENNFitConfig.num_fit_samples` and `num_fit_candidates` are now `num_samples`
and `num_candidates`. `PosteriorFlags.tie_break_neighbors` is now
`tie_neighbors`.

The high-level `ENN.posterior`, `ENN.batch_posterior`,
`ENN.conditional_posterior`, `ENN.neighbors`, and `ENN.add` workflows remain.
The renamed model operations are:

| `0.1.x` | `0.2.0` |
| --- | --- |
| `posterior_function_draw` | `posterior_draw` |
| `conditional_posterior_function_draw` | `conditional_draw` |
| `ensure_index_sync` | `ensure_sync` |
| `schedule_background_flush` | `schedule_flush` |
| `persist_index_to_disk` | `persist_index` |
| `train_rows_at` | `train_rows` |
| `index_memory_bytes` | `index_bytes` |

The `AGX` index choice was removed. Use `METAL`; `AUTO` may select Metal and
falls back to the CPU implementation when the accelerated path is unavailable
or unsupported for a query shape. The remaining choices are `FLAT`, `AUTO`,
`USEARCH`, `BPANN_DISK`, `METAL`, `OPENCL`, and `CUDA`.

### Candidate search API

`ennx.search` is the new Python API for encoded parameter search. It exports:

- `Parameter`: an immutable encoded parameter range. Supported encodings are
  `int4`, `int8`, `fp4_e2m1`, `fp8_e4m3`, and `fp8_e5m2`.
- `Search`: candidate generation, scoring, history, and explicit
  `ask`/`tell` coordination.
- `Optimizer`: `Search` plus trust-region adaptation and batched pending trials.
- `Trial`: the explicit handle returned by every ask operation, containing the
  selected index, regeneration seed, and score.

The old experimental search types map as follows:

| `0.1.x` | `0.2.0` |
| --- | --- |
| `ennx.experimental.PackedSearch` | `ennx.search.Search` |
| `ennx.experimental.PackedTurbo` | `ennx.search.Optimizer` |
| `ennx.experimental.TurboTrial` | `ennx.search.Trial` |
| tuple-based packed leaves | `ennx.search.Parameter` |

Search state no longer hides one implicit pending trial. `row`, `tell`, and
device-row access take a `Trial`; the optimizer also supports `ask_batch`,
`tell_batch`, `ask_stream`, and `batch_stream`. Stream methods generate
candidates from a scalar seed so the candidate seed array does not need to be
uploaded. CPU, Metal, and OpenCL use the same public lifecycle. Backend support
for each operation is reported by the Rust capability API described below.

### Experimental Python API

Low-level quantization stays under `ennx.experimental`; the deprecated
top-level quantization exports were removed.

| `0.1.x` | `0.2.0` |
| --- | --- |
| `quantize_fp4_e2m1` | `quantize_e2m1` |
| `create_optimizer_enn_multi_tr` | `enn_tr` |
| `make_multi_trust_region` | `make_region` |
| `allocate_region_batches` | `allocate_batches` |
| `select_region_candidates` | `select_candidates` |
| `multi_trust_region` | `mtrregn` |

### Rust API

The curated Rust import surface remains `ennx::prelude`. Its principal
migrations are:

| `0.1.x` | `0.2.0` |
| --- | --- |
| `EpistemicNearestNeighbors` | `ENN` |
| `ConditionalPosteriorDrawInternals` | `ConditionalDraw` |
| `IncrementalIncumbentTracker` | `IncumbentTracker` |
| `create_optimizer_enn` | `enn_optimizer` |
| `create_optimizer_zero` | `create_optimizer` |
| `create_optimizer_lhd` | `create_lhd` |
| `subsample_loglik_model` | `subsample_model` |
| `compute_posterior_internals` | `compute_internals` |
| `compute_conditional_posterior_internals` | `conditional_internals` |
| `hypervolume_2d_max` | `hypervolume2d_max` |
| `argmax_random_tie` | `argmax_tie` |
| `pareto_front_2d_maximize` | `pareto2d_max` |
| `calculate_sobol_indices` | `sobol_indices` |

The main Rust method migrations are:

| Type | `0.1.x` | `0.2.0` |
| --- | --- | --- |
| `ENN` | `new_with_storage` | `new_storage` |
| `ENN` | `new_with_options` | `new_options` |
| `ENN` | `schedule_background_flush` | `schedule_flush` |
| `ENN` | `persist_index_to_disk` | `persist_index` |
| `ENN` | `has_bounded_outputs` | `bounded_outputs` |
| `Optimizer` | `new_with_strategy` | `new_strategy` |
| `Optimizer` | `new_with_surrogate` | `new_surrogate` |
| `Optimizer` | `tell_with_yvar` | `tell_variance` |
| `Optimizer` | `trust_region_mut` | `trust_mut` |

The new `ennx::search` module provides `Parameter`, `Search`, `Trial`,
`Optimizer`, `TrustRegion`, and borrowed `DeviceView` values. An evaluator can
consume a `DeviceView` without materializing the selected encoded row on the
CPU, then commit the result through `observe` or `tell`.

The new `ennx::capability` module exposes `Backend`, `Operation`, `Support`,
and `Capability`, plus `backends()`, `operations()`, `support()`, and
`matrix()`. `Support` distinguishes `Direct`, `Fallback`, and `Missing`; callers
can therefore inspect the actual execution boundary instead of inferring it
from a backend name.

`IndexDriver::Agx` was removed and its supported behavior was consolidated
under `IndexDriver::Metal`. Direct `apple_gpu` and `forward_metal` modules are
no longer public; supported low-level exports remain available through
`ennx::experimental`.

### Compatibility

No aliases are provided for the removed `0.1.x` names. This is intentional:
stale imports and calls fail immediately instead of silently selecting an old
path.
