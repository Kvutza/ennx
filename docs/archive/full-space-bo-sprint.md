> Archived on 2026-09-29. Historical evidence and superseded plans, not current
> instructions. Read [current state](../handoff.md) and [runbook](../turbo-enn.md).
> Original dated results are retained; relative documentation links were relocated.

# Full-Space BO Sprint Constitution

> Scope update, 2026-09-23: the target includes pretraining from random
> initialization and post-training, with a subsecond complete full-parameter
> BO round on the local GPU. The active harness is synthetic local search,
> not evidence of global optimization or training quality. This document
> records research requirements, not a verified implementation checklist.
> Use [handoff](handoff.md) for current code, commands and validation status.

This sprint asks whether ENNX can make full-parameter derivative-free
optimization useful for model pretraining and post-training on the local GPU.
Every scalar weight is an independent search coordinate. Each proposal gives
every coordinate its own Gaussian innovation, with fixed per-tensor RMS scales.
Acquisition distances use the complete realized weight vectors. Low-dimensional
generating factors, Kronecker updates, sketches and sampled-coordinate distances
do not satisfy this target. Local proposals do not establish global optimality.

This document is the contract for the next implementation work. It is also a
filter: if a change does not make one of these requirements more true, it does
not belong in the sprint.

## Fixed Decisions

1. **LOOCV stays fixed.** ENNX fitting continues to use the current row-ID
   leave-one-out likelihood path. We are not changing the surrogate fitting
   objective during this sprint. The current fitting knobs remain `k`,
   `ENNFitConfig.num_samples`, and `ENNFitConfig.num_candidates`.
2. **The comparison baseline is hyperscale evolution strategies.** EGGROLL and
   related ES work are the standard to beat or learn from. We do not adopt
   rank-one perturbations as the default design.
3. **Every scalar weight is independently perturbed.** Each candidate is
   represented and evaluated as a dense full-model perturbation over all
   participating tensors, subject only to precision rounding. Touching every
   weight through shared factors does not meet this requirement.
4. **Context lengths are first-class.** Measurements must name 4096, 16384, and
   32768-token targets when they concern inference or the BO loop. Short-token
   checks are correctness probes, not sprint performance evidence.
5. **Speed and math are co-equal.** A perturbation law that is elegant but slow
   is not acceptable. A fast law with no defensible geometry is not acceptable.

## Perturbation Law Requirements

The candidate record is an audit record, not a reduced search object:

```text
candidate_record = {
  base_id,
  seed,
  radius,
  tensor_scales,
  correlation_settings,
  normalization_scheme
}
```

The record is not the scientific object by itself. The scientific object is the
dense perturbation it deterministically induces over the model weights.
History remains full-realization geometry: ENNX needs candidate geometry and
stable row identity without changing the search space.

Run-constant and candidate-varying fields are stored separately. The run record
contains the complete tensor key, offset, length, checkpoint RMS scale, distance
weight, rounding rule, and initial persistent-reference seed. Each candidate
records its base observation ID, selected stream index, seed, radius, and
reference correlation. The initial reference plus the accepted candidate chain
must be sufficient to replay the evolving persistent direction; a seed without
that lineage is not a reproducible correlated perturbation record.

### Current Geometry Baseline

The Metal implementation currently stores every retained history point as a
complete BF16 weight row. Acquisition uses the exact block-weighted squared
distance between each realized candidate and each realized history row:

```text
distance(candidate, history) =
  sum over blocks b [block_weight[b] * sum over i in b (candidate[i] - history[i])^2]
```

This is the correctness baseline. It includes BF16 rounding and therefore does
not have a descriptor-versus-realization mismatch. It is not the scalable end
state: for `D` parameters and history capacity `H`, projected search residency
is approximately `(H + 3) * 2D` bytes for history, incumbent, proposal, and
correlated reference rows, before evaluator weights, inference workspace, and
KV cache. At one billion parameters with `H = 2`, that is about 10 GB for the
search state alone.

The pinned Qwen2.5-Coder-1.5B checkpoint currently contains `1,543,714,304`
modeled BF16 elements. Its row is `3,087,428,608` bytes, so `H = 2` projects
to `15,437,143,040` bytes (about 14.38 GiB) of search residency. Run reports
must record the measured row and projected residency values rather than relying
on the model's marketing-size label.

Reduced representations are not an implementation target for this sprint.
Memory pressure is handled by bounded history, measurement, and systems work
that preserves dense full-realization geometry. Every run records the exact
four-candidate distance matrix to retained observations, keyed by stable
observation ID, as well as the selected candidate. The matrix is produced by the
existing selection reduction; gathering it must not add another model-sized
pass.

The perturbation law must satisfy these gates before it is treated as a serious
candidate:

- all modeled tensors participate unless explicitly excluded by an experiment
  manifest;
- per-tensor energy is controlled by fixed checkpoint-derived scales or a
  documented replacement;
- candidate generation is reproducible from the recorded full-space fields;
- realized BF16/FP16 changes are measured, including the fraction of weights
  changed after rounding;
- the ENNX kernel or distance used for history agrees with the induced dense
  geometry closely enough to justify acquisition ranking;
- GPU generation/application does not require materializing one full dense noise
  tensor per candidate on the host.

Rank-one outer-product perturbations are not the default route for this sprint.
If literature uses low rank, we extract its estimator, variance, memory, and
systems lessons without accepting its geometry as our answer.

## Trust Region Requirements

The current shared TuRBO controller is a baseline controller, not a theorem about
this model class. It remains in place while perturbation and performance are
made measurable. Any replacement must be tested against the same perturbation
law and objective so we can attribute changes.

The baseline's effective settings must be visible. With one evaluated arm, its
current failure tolerance is the ambient parameter count; for the pinned Qwen
checkpoint that is `1,543,714,304`. Contraction is therefore operationally
unreachable in the intended run budget, while three consecutive improvements
can still double the length. Run metadata records dimensions, evaluated arms,
length bounds, effective tolerances, counters, and restarts. This does not change
the baseline; it prevents the word "TuRBO" from hiding what the controller
actually does at this scale. This describes the Qwen baseline. The FBT round
study now explicitly uses a four-failure budget and a measured initial reward;
see [the active runbook](turbo-enn.md). It retains the shared controller's
three-success expansion rule and reports counters and length after every round.

A future controller should reason about actual perturbation magnitude, not just
nominal radius. It may eventually separate global radius from tensor-group
scale, but the first implementation rule is simpler: no hidden controller
changes inside perturbation or inference work.

## Objective and Evaluation Requirements

Teacher-forced loss is allowed as a fast objective, but it is not assumed to be
the final quality signal. Any run must record enough metadata to tell whether it
is optimizing teacher-forced solution loss, generated-code pass/fail, a rollout
reward, or another scalar objective.

For BO-versus-ES comparisons, the following must match:

- checkpoint and model revision;
- task/objective corpus;
- perturbation dtype and radius policy where applicable;
- candidate/evaluation budget;
- context length and token budget;
- minibatch refresh policy;
- random seeds or common-random-number scheme.

### Current Long-Context Loss Gate

The native Qwen teacher-forced path keeps the materialized attention reference
for sequences of at most 256 tokens. Longer sequences use 256-token cached
prefill chunks, a reusable BF16 K/V cache, per-chunk output projection, and one
full-sequence loss reduction. The loss profile records effective tokens rather
than padded array width, tokens processed through cached attention, and resident
K/V cache bytes.

This path is correctness-qualified by two distinct checks: a tiny-model test
compares cached and materialized losses and crosses the 256-token boundary, and
`cache_parity` compares both paths using the pinned Qwen checkpoint.
Both passed on 2026-09-18 with the checkpoint environment forwarded through the
Buck2 test runner rather than inherited from the invoking shell.

The first production-checkpoint 4096-token component run also completed on
2026-09-18. It scored the final 64 targets of a 4096-token synthetic sequence
and reported:

```text
cached tokens:   4096
KV cache bytes:  117,440,516
write:           0.009 ms
forward:         36,047.867 ms
output and loss: 61.263 ms
evaluator total: 36,109.310 ms
```

Forward execution is therefore 99.83% of the measured evaluator total. The 4K
execution gate is met, but the performance gate is not: this is a component
diagnostic with a synthetic token stream, not a complete BO round or model
quality result. The next optimization must reduce cached forward execution; a
faster output projection or host write cannot materially move the end-to-end
number.

A 32-wide K staging experiment in the aligned SIMD-group GEMM preserved exact
test loss but increased forward time from 36,047.867 ms to 37,892.445 ms, a
5.1% regression. It was reverted. Fewer barriers did not compensate for the
larger threadgroup footprint, so this design is not a candidate for the sprint.

A second loader experiment kept BF16 weights in native transposed layout,
vector-loaded a 32-wide tile, and widened only fragment elements. It reduced one
forward measurement to 35,690.800 ms, a 0.99% change. That is too close to
single-run variance to justify the added kernel complexity, so it was also
reverted rather than promoted as an optimization.

The same local checkpoint was then measured with `mlx_lm.benchmark` on the same
base M4, using one 4096-token prompt, batch size one, one generated token, a
2048-token prefill step, and three timed trials. Prompt throughput was 503.551,
502.160, and 502.784 tokens/s, for a mean of 502.831 tokens/s and about 8.15 s
per prompt. Peak reported memory was 3.708--3.709 GB. This is not objective
parity: MLX measures prompt prefill plus one decode, whereas the ENNX diagnostic
computes cached hidden states and scores 64 teacher-forced targets. It is still
a valid same-device implementation ceiling for the shared dominant operation,
the dense transformer prefill.

The comparison changes the implementation decision. ENNX is 4.4x slower than a
production Metal stack before BO overhead is considered. Reaching one second at
4K would require 4096 prompt tokens/s, another 8.1x over the observed MLX rate.
We therefore do not treat small edits to the current SIMD-group loader as a
credible route to the sprint target. The next qualified experiment is an
opt-in production GEMM path that keeps the canonical BF16 candidate weights
resident, widens one matrix at a time on the GPU, and submits FP32 matrix
multiplication through the already-tested MPS bridge. It must preserve the
objective within the existing parity tolerance and improve the complete 4K
forward materially; otherwise it is removed. Larger prefill chunks are tested
only after this backend isolates matrix throughput from chunk scheduling.

That experiment passed matrix-level parity and production-checkpoint parity.
With the original 256-token chunk, MPS reduced 4K forward time only to
34,964.510 ms, a 3.0% improvement that is not sufficient by itself. Coupling
MPS to a 2048-token chunk reduced forward time to 31,200.190 ms while retaining
the reported `0.004400` loss. The original kernel at the same 2048-token chunk
regressed to 38,680.105 ms. After wiring backend-specific workspace and chunk
selection, a confirming MPS run measured 30,638.680 ms forward and 30,729.975 ms
evaluator total with the same loss. The schedule is therefore backend-specific:
the default kernel retains 256-token chunks, while the opt-in MPS experiment
uses 2048-token chunks and accounts for its larger workspace. The combined path
is about 15% faster than the original ENNX baseline, but remains roughly 3.8x
slower than the MLX comparison. It stays experimental rather than becoming the
default.

An FP16 MPS extension was then measured against a fresh FP32 MPS control using
the same checkpoint, synthetic 4096-token input, 64 scored targets, and test
binary. The results were:

| MPS dtype | loss | forward | evaluator total |
| --- | ---: | ---: | ---: |
| FP32 | 0.004400 | 30,605.258 ms | 30,688.105 ms |
| FP16 | 0.004323 | 29,084.748 ms | 29,159.910 ms |

FP16 reduced both forward and evaluator time by about 5.0%, but shifted this
objective by `0.000077`. Production-checkpoint cached/materialized parity passed,
and a dedicated 128-token FP32-versus-FP16 checkpoint test measured losses of
`5.446201` and `5.446386`, respectively. Its absolute delta of `0.000185` passes
the explicit `0.0002` guard.

The full-space ranking gate then evaluated three independent proposal roots,
twelve dense correlated BF16 candidates in total, at the production checkpoint.
It scored each resident candidate buffer under both MPS dtypes before reusing
the allocation. The ascending candidate orders matched exactly in all pools:

| proposal root | FP32 and FP16 order | maximum loss drift |
| --- | --- | ---: |
| `0x7b6d0a13` | `[3, 2, 0, 1]` | 0.000080 |
| `0xac917e2d` | `[1, 0, 2, 3]` | 0.000297 |
| `0xe45b38c7` | `[1, 0, 3, 2]` | 0.000245 |

The same winner was therefore selected in every pool, and all candidate losses
passed the explicit `0.0005` cross-backend drift gate. The test uses the real
perturbation kernel and never retains four model-sized candidate rows.

The same production-checkpoint gate now gathers realized radial and angular
geometry inside the existing four-candidate proposal pass. For root
`0x7b6d0a13`, nominal radii of `0.005` and `0.020` became
`[0.005161, 0.020060, 0.005156, 0.020039]` after BF16 rounding. Candidates
`0/1` and `2/3` share their respective random stream at different radii; their
realized cosines were `0.951054` and `0.950813`. Cross-stream cosine magnitude
was at most `0.000299`, consistent with near orthogonality at this dimension.
The same pass measures each realized candidate against the normalized persistent
reference. Those cosines were `[0.717974, 0.747968, -0.000427, -0.000474]`.
The larger correlated proposal therefore preserved the configured `0.75`
coefficient closely, while BF16 rounding reduced the smaller proposal to about
`0.718`; both independent-stream proposals remained nearly orthogonal to the
reference. This distinction matters: configured correlation is a property of
the continuous law, while these values describe the perturbations the model
actually receives.

The steady proposal pass took `924.907 ms` after the persistent reference had
already been initialized. This includes generation of four dense candidates,
exact distances to the retained row, acquisition, selected-row materialization,
and the in-pass radial and pairwise reductions. Adding the five reference
geometry reductions produced `951.039 ms` in the immediately following run, an
observed increase of `26.132 ms` or 2.8%. One before/after pair is not a stable
latency estimate, and no pre-geometry baseline exists, so neither number
isolates total diagnostic cost. They do establish that proposal work nearly
exhausts the ideal one-second whole-round budget before objective scoring. The
proposal path therefore remains a performance target rather than a negligible
optimizer overhead.

A test-only split of the same kernels then measured `650.531 ms` for the
four-candidate pool, `35.740 ms` for selection including a separate command
submission and wait, and `261.880 ms` for selected-row materialization. The
ordinary single-command path measured `872.859 ms` in that run, so the split
times must not be summed as a prediction of production latency. They do localize
the work: Gaussian generation plus distance/geometry accumulation dominates the
pool pass, while recomputing one selected Gaussian stream and writing the 3.09 GB
row remains a substantial second cost. Selection arithmetic itself is not a
credible optimization target; its isolated wall time is mostly the deliberately
introduced command boundary.

The Box--Muller implementation originally called `sin` and `cos` separately
even though each thread consumes both members of the pair. Replacing those calls
with Metal's paired `sincos` intrinsic preserved every BF16 candidate bit in the
scalar parity fixture and preserved all twelve production-checkpoint losses,
ranks, radii, and cosines. Two production runs measured pool times of `614.774`
and `615.264 ms`; fused proposal times were `833.530` and `848.355 ms`, averaging
`840.943 ms`. Against the immediately preceding `650.531 ms` pool and
`872.859 ms` fused measurements, the observed changes are 5.5% and 3.7%.
Because the baseline has only one sample, these are evidence for keeping a
semantics-preserving primitive substitution, not a precise speedup claim.

The path remains opt-in. Three pools are useful evidence that FP16 can preserve
the local decision, not a statistical claim across objectives, reference seeds,
or radii; moreover, a 5% gain does not change the sprint feasibility verdict.
FP16 is retained for further controlled experiments but is not promoted to the
default scorer.

The FP16 path was then profiled at the production checkpoint with explicit
command boundaries around the transformer stages. The same synthetic
4096-token sequence and 64 scored targets produced loss `0.004323` and the
following GPU times:

| stage | GPU time | share |
| --- | ---: | ---: |
| embedding | 0.924 ms | <0.01% |
| QKV projections | 758.147 ms | 2.71% |
| cached attention | 20,673.014 ms | 73.86% |
| attention output | 369.203 ms | 1.32% |
| MLP expansion | 3,920.429 ms | 14.01% |
| MLP reduction | 2,267.312 ms | 8.10% |
| final norm | 0.745 ms | <0.01% |
| **total** | **27,989.777 ms** | **100%** |

The measured forward wall time was `28,233.625 ms`, and total evaluator time
was `28,319.955 ms`. All 284 profiling command buffers reported valid GPU
timestamps. These boundaries deliberately perturb normal scheduling, so their
sum is a localization measurement, not a prediction of production latency.
They do preserve the objective and settle the next implementation decision:
the serial cached-attention kernel is the dominant target. Further GEMM or
weight-conversion work cannot plausibly deliver a large whole-forward gain
while attention retains nearly three quarters of GPU time.

The current cached-attention kernel assigns one SIMD group to one query row and
query head, then walks every causal key serially. Every query rereads its K/V
history and performs online-softmax rescaling inside that serial loop. The next
qualified experiment is exact tiled fused attention: blocks of query rows share
K/V tiles, maintain row-wise online-softmax state, honor the 12-query-head to
2-KV-head GQA mapping, and write the same BF16-cache result. This changes data
reuse and parallelism, not the mathematical attention rule.

For 28 layers, 12 query heads, head width 128, and full causal attention, QK
plus probability-times-V costs `2 * 12 * 128 * N * (N + 1)` FLOPs per layer.
The serial kernel's logical BF16 K/V traffic is numerically the same count in
bytes because it rereads both 128-wide cache rows for every query head:

| context | attention FLOPs, whole model | serial logical K/V reads |
| ---: | ---: | ---: |
| 4,096 | 1.443 TFLOP | 1.443 TB |
| 16,384 | 23.091 TFLOP | 23.091 TB |
| 32,768 | 92.362 TFLOP | 92.362 TB |

These are algorithmic counts, not DRAM-counter claims; device caches may serve
some repeated reads. They still identify the reusable dimension. The first
implementation gate uses the repository's existing four-SIMD-group matrix
pattern with 8 queries and 16 keys per tile, specialized to head width 128. It
reduces logical K/V loads by about 8x before tails while replacing scalar dot
products with `simdgroup_multiply_accumulate`. This deliberately bounded step
reuses a parity-tested Metal mechanism. If it passes end-to-end, the next tile
study moves toward MLX's 64-query by 32-key structure; if it fails to move the
complete 4K evaluator materially, it is removed rather than elaborated.

The 8-by-16 implementation passed its first gate. A 19-row, 12-query-head,
2-KV-head fixture with a nonzero cached prefix compared every output element
against the serial kernel, including partial query and key tiles. The full test
suite then passed with `518` tests and `7` ignored. A paired production 4K run
on the same code and checkpoint measured:

| attention path | loss | forward | evaluator total |
| --- | ---: | ---: | ---: |
| serial SIMD | 0.004323 | 28,379.693 ms | 28,458.756 ms |
| tiled MMA | 0.004368 | 15,160.393 ms | 15,243.587 ms |

The loss delta is `0.000045`, inside the `0.0002` FP16 gate. Tiling reduced
forward time by 46.6% and total evaluator time by 46.4%. Under stage
instrumentation, attention fell from `20,673.014 ms` to `8,928.458 ms`, a
56.8% reduction; the profiled tiled run reported `15,184.363 ms` total GPU time
with no missing timestamps. The kernel therefore stays as a qualified opt-in
path. It does not satisfy the sprint target: the evaluator remains 15.24x over
the one-second whole-round budget before proposal work and about 1.87x slower
than the non-objective-parity MLX prefill comparison. Attention is still the
largest tiled stage, so the next kernel experiment is a larger query tile with
the same oracle, loss, and complete-evaluator gates.

The next bounded refinement uses 16 queries by 16 keys and 256 threads. A
straightforward 16-row extension would exceed the M4 threadgroup-memory budget
if it retained separate V and product tiles. Instead, all eight SIMD groups
finish the probability-times-V matrix products in registers, synchronize, and
then reuse the 8 KiB V tile as the 16-by-128 product tile. The explicit static
arrays total 25,792 bytes before compiler alignment: 8 KiB each for Q, V/product,
and the accumulated result, 1 KiB for scores, and 192 bytes for online-softmax
state. This is why the implementation advances to 16-by-16 rather than copying
MLX's 64-by-32 shape without a viable memory schedule.

The 19-row parity fixture now compares the serial, 8-by-16, and 16-by-16 kernels
element by element, including a partial final query tile and nonzero cached
prefix. It passed, as did the complete `518`-test suite. The production 4K gate
then measured:

| tiled attention | loss | forward | evaluator total |
| --- | ---: | ---: | ---: |
| 8 queries by 16 keys | 0.004368 | 15,160.393 ms | 15,243.587 ms |
| 16 queries by 16 keys | 0.004368 | 11,908.621 ms | 11,991.141 ms |

The larger tile preserves the recorded loss exactly and reduces complete
evaluator time by 21.3% relative to the qualified 8-by-16 path. In the profiled
run, attention fell again from `8,928.458 ms` to `4,127.137 ms`, a 53.8%
reduction. That run reported `11,711.963 ms` of GPU time across all 284 commands
with no missing timestamps. Its MLP expansion and reduction stages took
`4,117.839 ms` and `2,303.831 ms`, respectively, so the combined MLP is now the
largest stage at 54.8% of measured GPU time; attention is 35.2%. The next
qualified inference experiment must therefore attack the MLP path or eliminate
work across stages. Another attention-only tile change is not justified by the
current whole-evaluator profile.

A follow-up chunk-schedule experiment also exposed and removed a control-flow
footgun. Long-context loss had been choosing the cached or materialized path by
comparing sequence length with the backend's prefill chunk. Raising the MPS
chunk from 2048 to 4096 therefore sent a 4096-token objective through the
materialized reference path, reported zero cached tokens, and took 67.5 seconds
before failing the existing cache-use assertion. The reference limit is now an
independent 256-row constant with a boundary test at 256/257; backend chunk
tuning can no longer change the attention algorithm.

After that fix, the intended cached 4096-row schedule preserved loss but
measured `12,211.152 ms` total, 1.8% slower than the `11,991.141 ms` 2048-row
control. One pair is not a stable latency estimate, but it supplies no evidence
for increasing memory by roughly 73 MB or changing the production schedule.
The MPS chunk therefore remains 2048.

The first MLP refinement removes avoidable precision round trips without
changing either GEMM or model arithmetic. On the FP16 MPS path, normalized
activations are narrowed once and shared by the gate and up projections. Their
half outputs remain in existing workspace buffers; one Metal kernel evaluates
SiLU in FP32 registers and writes the rounded half activation consumed directly
by the down projection. Only the down result is widened for the FP32 residual.
The FP32 MPS and native paths are unchanged.

The production 4K gate preserved loss exactly at `0.004368` and reduced forward
time from `11,908.621 ms` to `11,560.349 ms`; evaluator total fell from
`11,991.141 ms` to `11,635.314 ms`, a 3.0% improvement. The profiled run
attributed the gain to the intended stages: MLP expansion fell from
`4,117.839 ms` to `3,876.906 ms`, and reduction fell from `2,303.831 ms` to
`2,229.241 ms`. Combined MLP time remains `6,106.147 ms`, or 54.2% of the
reported `11,271.450 ms` GPU total, while tiled attention is `4,035.157 ms`.
The mechanism stays; it is not enough to change the feasibility verdict.

## Performance Contract

The ideal target is a complete BO round under one second. The current code is
far from that. We still measure whole rounds, not isolated kernel wins, because
the optimizer fails if any part of the loop dominates.

A complete round includes minibatch selection, incumbent rescoring when required,
proposal/acquisition, perturbation application, selected-candidate objective
scoring, controller update, synchronization, and logging. Measurements that
omit any of these are allowed only as component diagnostics.

For the 4096/16384/32768 context targets, reports must distinguish:

- acquisition candidates generated;
- selected candidates actually evaluated by the objective;
- objective evaluations in the round;
- prompt/context tokens;
- generated tokens when generation is involved;
- host time, GPU wait time, and logging time where available.

## Research Intake Rules

Read papers for mechanisms, not names. For each candidate idea, extract:

- what geometry it imposes on perturbations;
- what estimator it uses and what variance it reduces;
- what memory movement it avoids;
- what operation count it changes at 4K/16K/32K context;
- what assumptions make its result true;
- what would falsify it in our setting.

Do not implement a paper because it is new. Implement only when there is at
least a concrete path from its mechanism to one of our sprint gates.

### Current Mechanism Ledger

- [Evolution Strategies at the Hyperscale](https://arxiv.org/abs/2511.16652)
  replaces a dense matrix perturbation with a low-rank product and amortizes the
  update over a large population. Its memory and fused-forward analysis is a
  required comparison, but its low-rank geometry is not this sprint's default.
  We use it as the EGGROLL baseline and do not import the factorization unless a
  dense-law ablation first shows that rank, rather than objective evaluations,
  is the limiting variable.
- [Evolution Strategies at Scale](https://arxiv.org/abs/2509.24372) is direct
  evidence that full-parameter ES can move billion-parameter LLMs. It does not
  establish that BO is competitive. A valid comparison must match checkpoint,
  task distribution, forward-pass budget, context length, and seeds; until that
  run exists, `run.json` must continue to mark EGGROLL/ES as unevaluated.
- [FZOO](https://arxiv.org/abs/2506.09034) combines batched one-sided
  Rademacher estimates with loss-dispersion step scaling. This changes both the
  estimator and update rule, so it belongs as a forward-budget-matched optimizer
  ablation, not as an unexplained mutation of ENNX acquisition or trust-region
  behavior. Its systems lesson is to share baseline work across a candidate
  batch; its falsification test is end-to-end forwards-to-quality, not kernel
  latency alone.
- [Cylindrical Thompson Sampling](https://proceedings.mlr.press/v238/rashidi24a.html)
  separates radial and angular behavior to avoid pathological geometry in high
  dimensions. ENNX already uses a scalar radius with tensor-normalized dense
  directions rather than a TuRBO hyperrectangle. The transferable experiment is
  therefore to measure realized per-tensor radii and inter-candidate angles after
  BF16 rounding. Copying its GP kernel before those measurements would solve a
  different problem.
- [AdaScale-TuRBO](https://arxiv.org/abs/2604.22967) scales GP lengthscales with
  ambient dimension and trust-region size to preserve prior complexity. ENNX's
  current surrogate is not that GP. The paper is not evidence for changing the
  current dense full-model failure-tolerance baseline in this sprint.
- [Zeroth-Order Optimization for Self-Evolving LLM Agents](https://arxiv.org/abs/2608.09292)
  uses answer perplexity as a smoother signal and parallel perturbation inference,
  but perturbs LoRA parameters. Its objective lesson is relevant to the concern
  about sparse generated-code rewards; its parameter geometry is not evidence
  for a full-space law. A teacher-forced/perplexity objective may be compared
  only with held-out generated-code quality and identical evaluation budgets.
- [The Ziggurat Method for Generating Random Variables](https://www.jstatsoft.org/article/view/v005i08)
  replaces Box--Muller transcendental work with table lookup and a high-probability
  integer fast path while retaining an exact normal target through rejection.
  [GPU comparisons](https://doi.org/10.1016/j.future.2017.01.011) report that
  Ziggurat and related nonstandard normal generators can outperform common GPU
  generators, but that result does not establish a win on Apple GPUs or in this
  fused distance kernel. Rejection divergence, table access, tail handling, and
  deterministic counter advancement are all relevant here. Ziggurat is therefore
  the next perturbation-generation experiment only as an opt-in implementation
  shared by scalar, Metal, and CUDA paths, with distribution tests, descriptor
  replay parity, realized-geometry checks, and end-to-end proposal timing. It is
  not yet permission to change the default seed-to-direction mapping.
- [FlashAttention](https://arxiv.org/abs/2205.14135) supplies the exact,
  IO-aware tiled-attention mechanism: stage Q/K/V blocks in fast memory and
  carry row-wise online-softmax statistics so the full score matrix is never
  materialized. The mechanism preserves exact attention up to floating-point
  order; it is relevant because the current kernel rereads K/V for every query.
  The first ENNX implementation must retain causal masking, GQA cache indexing,
  and the existing BF16-cache parity gate.
- [FlashAttention-2](https://arxiv.org/abs/2307.08691) reduces non-matrix FLOPs
  and parallelizes long sequences more effectively. Current MLX Metal kernels
  use a `64`-query by `32`-key dispatch baseline for ordinary head widths; that
  is a design reference, not a copied constant. ENNX must derive threadgroup
  memory and occupancy for head width `128` before selecting its tile. The
  experiment is rejected if tiny cached/materialized parity fails, if the 4K
  loss moves outside the existing `0.0002` FP16 gate, or if complete 4K
  evaluator time does not improve materially.

This ledger is intentionally selective. Methods that restrict the active
parameter subspace are out of scope while the sprint requires every tensor to
participate densely. Their structural assumptions must not be smuggled into the
main experiment under the generic label "high-dimensional BO."

## Implementation Standard

No speculative large rewrites. Each code change must leave behind one of:

- stronger metadata that prevents experiment ambiguity;
- a parity or correctness test;
- a whole-loop or component timing measurement;
- a smaller hot path with the same objective;
- a perturbation law whose dense semantics are clearer than before.

Quality means we can explain what changed, why it aligns with this document, and
which evidence would show it failed.
