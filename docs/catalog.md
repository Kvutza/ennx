# Experiment catalog

`./ennx catalog` lists library algorithms and generates categorical variants of
a version 2 tuning baseline. Rust validates each configuration before export.
The command generates files without executing experiments.

The inventory is [examples/catalog/library.toml](../examples/catalog/library.toml):
six execution families, 44 component groups, and 162 option labels reviewed on
2026-10-03. Each entry records sources, access, and constraints.

## Commands

```sh
./ennx catalog
./ennx catalog --json
./ennx catalog examples/tuning/code-pretrain.toml
mkdir -p .cache/ennx/catalogs
./ennx catalog examples/tuning/code-pretrain.toml \
  --out .cache/ennx/catalogs/pretrain-campaign
```

| Invocation | Output |
| --- | --- |
| `catalog` | Components and choices |
| `catalog --json` | Full inventory, including sources and adapter requirements |
| `catalog BASE.toml` | Configuration counts and categorical axes |
| `catalog BASE.toml --json` | Enumeration manifest |
| `catalog BASE.toml --out DIR` | TOMLs and manifest in a new directory |

The export directory requires an existing parent and must not already exist.
It contains `baseline.toml`, numbered experiment TOMLs, and `manifest.json`.
Each experiment receives a separate output directory under `runs/` and a unique
checkpoint-save path when configured. Relative input paths become absolute paths
based on the original baseline's directory. Inputs are checked when the experiment
runs. The manifest is written last to mark a completed export.

## Categorical axes

| Axis | Choices |
| --- | --- |
| Scalar acquisition | UCB / Thompson |
| Vector acquisition | Pareto; MORBO with restart or proposal rescalarization when MORBO settings are supplied |
| Proposal law | Independent / polynomial threshold |
| Perturbation | Gaussian / Rademacher |
| Four-slot pool layout | One, two, or four proposal arms |
| History geometry | Realized / latent |
| Distance scaling | Global / self tuning |
| Neighbor fitting | Fixed / fitted |
| Region shape | Scalar / static tensor families / learned tensor families |
| Controller | TuRBO / reliability / MORBO |
| Pretraining selection | ENN / random pool slot |
| Corpus-prefix reference | Moving incumbent / frozen initial |
| Generated model initialization | Patterned / GPT-2 / Megatron / Xavier uniform |
| Generated feedback | Identity / projected sigmoid |
| Verification implementation | Serial / accepted prefix |
| Frozen-Qwen scorer | Three backends, two attention implementations, two readout implementations |

Search axes apply to full-weight pretrain and generation studies. Generation adds
feedback and verification, plus initialization when starting without a checkpoint.
Frozen-Qwen scoring axes apply to that reward. Corpus-prefix reference applies
without generation. Other studies receive acquisition substitutions only.

The pool uses four slots divided among one, two, or four arms. Auto verification
resolves to serial or accepted prefix; the catalog enumerates those implementations.

The baseline fixes experiment, model, corpus, reward, purpose, tasks, resources,
budgets, seeds, numeric settings, diagnostics, artifact policies, MORBO clipping,
and temperature-search policy. Author a separate baseline for each reward with
its required tasks and resources. A Pareto baseline supplies Pareto settings;
MORBO enumeration requires a MORBO baseline.

Existing numeric settings are retained. Activating UCB, reliability control, or
self-tuning distance scaling uses Rust defaults where settings are absent.
The manifest records this convention, and exported TOMLs contain the defaults.

## Validation and coverage

Candidates pass through `TuneSpec::overrides()` and `TuneSpec::from_overrides()`.
Rejections are grouped by the first Rust validation error.

Resolved TOML determines uniqueness. MORBO is recorded as the trust-region
method that owns its adaptation. IDs are deterministic
within a fixed baseline, axis set, and library version.

The manifest contains the baseline, inventory snapshot, axes, experiment choices,
counts, rejection reasons, and coverage. Coverage reports valid cases per choice,
including zeros, and pairs differing in one categorical field.

Enumeration supports up to one million attempted combinations. Runtime
qualification follows each workload's resource, hardware, and parity checks.
Compare variants against a common evaluation objective.

## Configuration counts

Checked-in baselines, enumerated on 2026-10-03:

| Baseline | Axes | Attempted | Distinct valid | Rejected | Duplicate paths |
| --- | ---: | ---: | ---: | ---: | ---: |
| `code-pretrain.toml` | 11 | 6,912 | 2,304 | 4,608 | 0 |
| `code-pretrain-generated.toml` | 13 | 82,944 | 9,216 | 69,120 | 4,608 |
| `code-environment.toml` | 12 | 41,472 | 6,912 | 31,104 | 3,456 |

## Execution families

| Family | Interface | Adapter requirements |
| --- | --- | --- |
| Ordinary vector optimization | Rust/Python ask/tell; eval subset | Expose surrogate, candidate, initialization, region, and index policies |
| Disk ENN | Model and disk BPANN APIs | Configure compatible storage and metric fitting |
| Encoded weight search | Search/trials APIs and proposal benchmark | Supply an objective and matching encoding/device evaluator |
| Full FP16 model weights | Version 2 tune | Categorical generation implemented |
| BF16 Qwen/FLAME checkpoints | Separate workflows | Preserve each model's layout, precision, objective, and history contracts |
| Kernel programs | Kernel-search manifests | Enumerate declared source candidates and use existing ABI/parity/trajectory gates |

Each family needs a renderer for its experiment contract. The current generator
targets version 2 tune. Python cosine-distance, RAASP variants, and region-sharing
options require an execution-path audit before inclusion in generated campaigns.

Update the inventory and generator axes together when adding policies. Trace each
option to its execution path and record its constraints.

## References

| Project | Convention |
| --- | --- |
| [ModularCMAES](https://github.com/IOHprofiler/ModularCMAES) | Compose algorithm variants from modules |
| [ConfigSpace](https://github.com/automl/ConfigSpace) | Declare configuration spaces |
| [Nevergrad](https://github.com/facebookresearch/nevergrad) | Separate optimizers from benchmark suites |
| [Benchopt](https://github.com/benchopt/benchopt) | Structure benchmarks around objectives, datasets, and solvers |
