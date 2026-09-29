# Optimizer evals

`./ennx eval` is ENNX's model-eval-style harness for black-box optimization.
The optimizer is the submitted system, a registered objective is the task, and
objective calls are the inference budget. The harness emits immutable JSONL
trial records and a Markdown scorecard.

Storage cleanup on 2026-10-01 removed existing raw run outputs, generated
checkpoints, kernel-search archives, eval artifacts, and GPU traces at the
user's request. Historical measurements predating that cleanup remain dated records; their
listed raw artifact paths are no longer available. Source code, experiment
inputs, corpus/tokenizer caches, and source model checkpoints were retained.

## FineWeb corpus-prefix pilot: 2026-10-01

```sh
./ennx tune examples/tuning/fineweb-enn.toml
./ennx tune examples/tuning/fineweb-random-selection.toml
```

Both completed native Rust/Metal runs use 1,047,732,224 independently
perturbable coordinates, identical derived model/proposal/acquisition streams,
the same pinned FineWeb corpus, and the same train-only 8,192-token byte BPE.
This experiment uses **teacher-forced corpus-prefix next-token cross-entropy**;
it generates no text. It neither replaces the free-generation reward experiment
nor reproduces the official nanoGPT tokenizer, architecture, or target loss.

| Measurement | ENN selection | Random pool selection |
| --- | ---: | ---: |
| Initial held-out NLL | 9.013232470 | 9.013232470 |
| Final held-out NLL | 9.012842178 | 9.013103008 |
| Candidate rounds / guided rounds | 12 / 3 | 12 / 3 |
| Accepted updates | 2 | 2 |
| Median candidate-round wall | 496.070 ms | 495.023 ms |
| Maximum candidate-round wall | 584.710 ms | 539.072 ms |
| Total learning time | 15.369 s | 14.922 s |
| Training objective calls / causal targets | 25 / 204,750 | 25 / 204,750 |

The random arm is a ranking ablation: it selects a precommitted candidate-pool
slot while preserving ENN fitting, acceptance, and controller logic. It is not
standalone random search or ES. ES and gradient baselines are explicitly marked
unimplemented in `comparison.json`. One pair with only three guided evaluations
cannot establish an ENN advantage or a scaling law.

Held-out scoring never feeds acceptance, fitting, or the controller. All 16
validation sequences are scored initially and after rounds 4, 8, and 12:
65,520 causal targets per measurement. Each observation takes 1.54-1.60 seconds.
Individual candidate-round timers exclude this periodic validation; learning
time includes initial training/validation and subsequent validation, but excludes
corpus preparation, model setup, and artifact writing. The 200 ms target is not
met; the complete validation-inclusive cycle is not subsecond.

The bounded cache `.cache/ennx/corpora/47d7c71aed89017466a3` took about 7.3 seconds
to prepare and occupies about 7.1 MB. It pins FineWeb revision
`9bb295ddab0e05d785b879661af7260fed5140fc`, reads `sample/10BT`, partitions by
lowercase URL host (not registrable domain), and rejects exact duplicate content
across admitted splits. Training has 20 packed sequences; validation and test
have 16 each, all of length 4,096. The test split remains untouched. Cache reuse
checks the recipe, split sizes, and SHA256 hashes of tokenizer and packed files.

Retained run roots:

- ENN: `.cache/ennx/runs/pretrain/f0a4af981cdff4eedf0d/run-1790832432668-81756-0`
- Random selection: `.cache/ennx/runs/pretrain/482eb27b4512a433f95b/run-1790832495019-82245-0`

Each includes `validation.json`, `comparison.json`, `result.toml`, and per-round
controller records. These post-cleanup artifacts were retained. To extend the
pilot, change round/repetition budgets in the TOMLs; model and proposal seeds
remain derived rather than checked-in literals. The pilot corpus is bounded,
so longer runs revisit its training blocks rather than stream additional data.

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

Generated coding experiments are additionally governed by the
[coding-agent research contract](coding-agent-contract.md). In particular,
text diversity and overlap are diagnostics rather than evidence of coding
ability, and an untrained checkpoint cannot be promoted into a learning experiment
by changing its reward. Executable held-out outcomes are required for a coding
claim.

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

## Generated-text reward audit: 2026-09-30

The generated-text experiment is separate from the corpus-likelihood ablations
above. `./ennx tune examples/tuning/code-pretrain-evaluated.toml` uses
document-aligned 128-token prompts, 4,096 generated student tokens, and a fixed
Qwen2.5-Coder-1.5B evaluator. Student initialization remains untrained. This
configuration is an objective experiment, not a validated pretraining recipe.

A one-round smoke run completed at
`.cache/ennx/runs/pretrain/448104ffebc79b11636a/run-1790822824590-69700-0`.
It perturbed the full 1,047,732,224-coordinate generated model. Its measured
round was **132,093.721 ms**, including 24,028.906 ms generating and
106,542.150 ms evaluating reward. The 200 ms target was not met.

| Completion | Teacher mean token NLL | Unique student token IDs | Repeated four-gram fraction |
| --- | ---: | ---: | ---: |
| Initial generated | 0.428053 | 11 | 94.19% |
| Accepted candidate | 0.288029 | 8 | 96.43% |
| Real corpus continuation | 1.613926 | — | — |

The reward is negative teacher NLL, so the repetitive candidate scored better
than both the initial generation and real code. Acceptance improved the
configured reward while worsening these diversity diagnostics. This falsifies
the use of this observed reward improvement as evidence of useful pretraining;
it does not establish an ENN-versus-random result. There were only one training
episode, two held-out episodes, and one candidate round. The held-out reward
improvements are subject to the same objective failure.

Subsequent instrumentation writes `reward-audit.json` with the initial/corpus
comparison and explicitly leaves learning quality unestablished. It is not an
acceptance gate or a complete control suite: shuffled, unrelated, and controlled
repetition comparisons remain unmeasured. `reward.json` records evaluator
forward/readout timings and whether the text needed UTF-8 replacement.
`completion.bin` preserves original decoded bytes; `completion.txt` is the
lossy UTF-8 display. These additions were made after the dated smoke run and
must not be assumed present in its artifacts.

### Full-power attention comparison

Two fresh one-round `./ennx tune` runs used the same model initialization,
proposal/acquisition/sampling streams, training and validation tasks, fixed
teacher checkpoint, and full 4,096-token student generation. Both recorded
active Low Power Mode disabled on battery. The only changed scorer setting
was `attention = "reference"` versus `attention = "tiled16"`;
`readout = "reference"` was retained.

| Candidate round | Reference | Tiled16 |
| --- | ---: | ---: |
| Whole round | 46,322.310 ms | 25,498.401 ms |
| Student generation wall | 12,270.862 ms | 12,230.897 ms |
| Reward wall | 33,665.900 ms | 12,998.389 ms |
| Scorer forward | 31,763.736 ms | 10,969.137 ms |
| Scorer readout | 1,897.613 ms | 2,024.692 ms |
| Proposal | 229.664 ms | 121.945 ms |

The whole-round ratio was **1.82x**, with **2.59x** faster reward scoring.
These are one sequential reference/tiled pair, not a statistically qualified
latency distribution. Proposal timing also varied despite unchanged proposal
code; do not attribute that variation to attention. The historical 132-second
run lacked power metadata and is not the reference for this comparison.

Initial and candidate student token sequences matched exactly across the pair;
both candidates were accepted. Teacher mean NLL differed by 0.000243 on the
initial completion and 0.000064 on the candidate. The largest absolute
held-out mean-NLL difference was 0.000083. Thus the tiled path preserves the
observed decision, not bitwise evaluator parity or a guarantee for near-ties.
The evaluated example now selects it explicitly; the general frozen-scorer
default remains the reference oracle.

The repetitive-text reward failure reproduced in both runs. Neither met the
200 ms target or establishes learning quality. Generation alone remains about
12.2 seconds on this candidate.

Artifacts:

- Reference: `.cache/ennx/runs/pretrain/3e21205b772bccc67561/run-1790826177677-72653-0`.
- Tiled16: `.cache/ennx/runs/pretrain/106ef368f7159e75e195/run-1790826393363-73170-0`.

## Native generated-code reconstruction probe: 2026-10-01

`./ennx tune examples/tuning/code-pretrain-reconstruction.toml` adds a native
`code_reconstruction` reward. The student freely generates all 4,096 tokens;
the scorer compares exact decoded bytes with one document-aligned code
continuation using normalized unit-cost Levenshtein distance. No teacher
forcing or additional model forward is used. BPE segmentation does not affect
the score, and invalid UTF-8 bytes are not replaced before scoring.

Resident reference masks process 64 reference positions per operation. Scoring
time grows with completion bytes times reference words; workspace grows with
reference words, not a full dynamic-programming matrix. Focused tests agree
with scalar dynamic programming at word boundaries, including insertions,
deletions, arbitrary bytes, and alternative BPE segmentations.

The retained run is
`.cache/ennx/runs/pretrain/3a3d53f1838290cbff96/run-1790830389506-76877-0`.
Active Low Power Mode was disabled on battery. It perturbed the full
1,047,732,224-coordinate model and used untrained initialization.

| Measurement | Result |
| --- | ---: |
| Complete candidate BO round | 584.255 ms |
| Proposal | 100.910 ms |
| Reported rollout GPU time | 377.509 ms |
| Reward wall, including its artifact write | 6.305 ms |
| Exact-byte scoring component | 5.839 ms |
| Initial / candidate reward | 0.119119585 / 0.119240403 |
| Generated candidate tokens / bytes | 4,096 / 17,536 |

This is one subsecond round, not sustained throughput or the 200 ms target.
Initial scoring, four held-out rollouts, final checkpoint export, and terminal
printing are outside the candidate round. Production reward byte decoding is
inside it. Candidate and initial outputs differ at four token positions; both
have only three distinct IDs and about 99.78% repeated four-grams. One held-out
reward fell from 0.111853063 to 0.099680960; the other stayed at 0.082345128.
The controller was still in initialization, not ENN-guided selection.

The prior whole-token reconstruction probe used matching model/proposal/
acquisition/sampling streams and the same task. It took 584.364 ms, but both
rewards were zero. Exact-byte scoring resolves that flat feedback in this pair;
it does not establish useful coding learning.

Offline controls independently reproduced candidate edit distance 15,445.
The corpus continuation scores 1.0, shuffled reference bytes score 0.184743,
and 14,328 space bytes score 0.291457. A budget-compatible completion of
4,096 repeated four-space BPE tokens also scores 0.254883, above the generated
candidate. Byte reconstruction therefore has an explicit whitespace-collapse
gaming risk. It remains a latency/dispersion bootstrap probe, not the final
coding objective; these controls must not be interpreted as execution or
functional-correctness tests. No longer learning campaign was launched.

This measurement uses the retained legacy corpus cache, which lacks repository
provenance for these episodes. Subsequently, both native and Python corpus
readers were changed to split by repository alone, keeping all commits in the
same partition. Both corpus and token-pool identities now include the versioned
split policy, so future preparation rebuilds rather than silently reusing the
legacy split. That preparation is outside the BO timing; the fresh data has not
yet been collected or benchmarked.

## Adding a task

Add the objective to the `Function` registry in
`rust/crates/dev-cli/src/eval.rs`, define its optimum as zero, give it explicit
bounds and a target in TOML, and add a zero-at-the-optimum test. A task should
be deterministic from `(task, seed, evaluation)` so policies can be paired.
External simulators and production workloads should get adapters that preserve
the same JSONL record contract; they should not shell out from the objective
function or report optimizer-selected validation data.
