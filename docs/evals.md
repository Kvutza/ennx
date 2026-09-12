# Optimizer evals

`./ennx eval` is ENNX's model-eval-style harness for black-box optimization.
The optimizer is the submitted system, a registered objective is the task, and
objective calls are the inference budget. The harness emits immutable JSONL
trial records and a Markdown scorecard.

Run the fast end-to-end check with a fresh artifact path:

```sh
./ennx eval examples/evals/optimizer-smoke.toml \
  --output artifacts/evals/smoke-001.jsonl
```

Run the broader CPU suite with:

```sh
./ennx eval examples/evals/optimizer-core.toml \
  --output artifacts/evals/core-001.jsonl
```

The harness creates one deterministic LHD initialization from `(task, seed)`
and feeds the identical batch to every policy. Guided evaluations then use
common per-evaluation random streams, preventing a surrogate fit from shifting
another policy's proposal randomness. Policy execution order alternates by
seed. Noisy tasks use common noise by evaluation number, and the noise-free
objective determines regret. Candidate hashes make the pairing auditable. One
failed trial is recorded as a failure instead of erasing the remaining suite.
Optimizer coordinates remain in the unit hypercube; each task applies its
declared physical bounds only at the objective boundary.

## Metrics

- **Final simple regret:** best noise-free objective seen by the budget.
- **Normalized regret AUC:** mean best-so-far regret divided by the first
  evaluation's regret, clipped at one. Lower is better and rewards early gains.
- **Success rate:** fraction of seeds reaching the task's declared target.
- **Evaluations to target:** sample efficiency among successful trials.
- **Failures and wall time:** reliability and operational cost, reported beside
  quality rather than hidden.

The JSONL begins with the complete TOML configuration, contains every
evaluation, and ends with aggregates plus a completion marker. A missing
completion marker means the run is not a valid result. Existing artifacts are
never overwritten.

## Scope

The core suite exercises the general in-memory TuRBO-ENN implementation over
smooth, ill-conditioned, multimodal, noisy, and sparse high-dimensional tasks.
It does **not** stand in for the billion-weight pretraining workload. That
workload has a different resident accelerator ENN and must be judged with fixed
training/validation data, matched candidate pools, and held-out NLL through the
production scorer. The bounded production protocol remains in
[`bo-audit.md`](bo-audit.md); its results and the core-suite results must not be
merged into one number.

The production controller/geometry experiment is a matched two-by-two
factorial, driven entirely through `./ennx tune`:

- global ENN distances with TuRBO control;
- self-tuned ENN distances with TuRBO control;
- global ENN distances with reliability-aware control;
- self-tuned ENN distances with reliability-aware control.

The four checked-in `code-pretrain-ablation-*.toml` files have the same budget,
model, corpus, acquisition, fit policy, and trust bounds. They intentionally
omit literal seeds so the shared configuration derivation supplies common
random streams.

## Adding a task

Add the objective to the `Function` registry in
`rust/crates/dev-cli/src/eval.rs`, define its optimum as zero, give it explicit
bounds and a target in TOML, and add a zero-at-the-optimum test. A task should
be deterministic from `(task, seed, evaluation)` so policies can be paired.
External simulators and production workloads should get adapters that preserve
the same JSONL record contract; they should not shell out from the objective
function or report optimizer-selected validation data.
