# Changelog

## 0.2.0 — 2026-09-10

Breaking release. Renamed APIs have no compatibility aliases.

### Python API

Added BoTorch, Optuna, and Ax adapters. All three libraries are now required
Python dependencies. See [Python integrations](docs/interop.md) for supported
features and limitations.

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

`create_optimizer(bounds, config, rng)`, `ask`, `tell`, and `telemetry` are unchanged.

`ENNParams` uses shorter field names:

| `0.1.x` | `0.2.0` |
| --- | --- |
| `k_num_neighbors` | `k_neighbors` |
| `epistemic_variance_scale` | `epistemic_scale` |
| `aleatoric_variance_scale` | `aleatoric_scale` |

`ENNFitConfig.num_fit_samples` and `num_fit_candidates` are now `num_samples`
and `num_candidates`. `PosteriorFlags.tie_break_neighbors` is now
`tie_neighbors`.

`ENN.posterior`, `ENN.batch_posterior`, `ENN.conditional_posterior`,
`ENN.neighbors`, and `ENN.add` are unchanged. Renamed methods:

| `0.1.x` | `0.2.0` |
| --- | --- |
| `posterior_function_draw` | `posterior_draw` |
| `conditional_posterior_function_draw` | `conditional_draw` |
| `ensure_index_sync` | `ensure_sync` |
| `schedule_background_flush` | `schedule_flush` |
| `persist_index_to_disk` | `persist_index` |
| `train_rows_at` | `train_rows` |
| `index_memory_bytes` | `index_bytes` |

Removed `AGX`; use `METAL`. `AUTO` selects Metal when supported and falls back
to CPU otherwise. Index choices are `FLAT`, `AUTO`,
`USEARCH`, `BPANN_DISK`, `METAL`, `OPENCL`, and `CUDA`.

### Candidate search API

Added `ennx.search` for low-bit parameter optimization:

- `Parameter`: a parameter range stored as
  `int4`, `int8`, `fp4_e2m1`, `fp8_e4m3`, and `fp8_e5m2`.
- `Search`: generates and scores candidates, and stores observations.
- `Optimizer`: adds trust-region adaptation and pending trials.
- `Trial`: identifies a proposed candidate by index, seed, and score.

The old experimental search types map as follows:

| `0.1.x` | `0.2.0` |
| --- | --- |
| `ennx.experimental.PackedSearch` | `ennx.search.Search` |
| `ennx.experimental.PackedTurbo` | `ennx.search.Optimizer` |
| `ennx.experimental.TurboTrial` | `ennx.search.Trial` |
| tuple-based packed leaves | `ennx.search.Parameter` |

`row`, `tell`, and GPU buffer access now take a `Trial`. Added `ask_batch`,
`tell_batch`, `ask_stream`, and `batch_stream`. Stream methods generate
candidates from one seed, avoiding an upload of candidate seeds.

### Experimental Python API

Quantization is now imported from `ennx.experimental`.

| `0.1.x` | `0.2.0` |
| --- | --- |
| `quantize_fp4_e2m1` | `quantize_e2m1` |
| `create_optimizer_enn_multi_tr` | `enn_tr` |
| `make_multi_trust_region` | `make_region` |
| `allocate_region_batches` | `allocate_batches` |
| `select_region_candidates` | `select_candidates` |
| `multi_trust_region` | `mtrregn` |

### Rust API

Use `ennx::prelude` for the main Rust types and functions. Renamed exports:

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

Renamed methods:

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

Added `ennx::search` with `Parameter`, `Search`, `Trial`, `Optimizer`,
`TrustRegion`, and `DeviceView`. Evaluators can read GPU buffers through
`DeviceView` and record results with `observe` or `tell`.

Added `ennx::capability` to report whether each backend implements an operation,
uses a fallback, or lacks support. Query it with `support()` or `matrix()`.

Removed `IndexDriver::Agx`; use `IndexDriver::Metal`. The `apple_gpu` and
`forward_metal` modules are private. Public GPU helpers are in
`ennx::experimental`.
