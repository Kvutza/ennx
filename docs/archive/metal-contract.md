> Archived on 2026-09-29. Historical evidence and superseded plans, not current
> instructions. Read [current state](../handoff.md) and [runbook](../turbo-enn.md).
> Original dated results are retained; relative documentation links were relocated.

# Full-bandwidth transformer: native Metal architecture contract

Status: complete LocalV1 token-ID scoring graph implemented; paper ambiguities remain, 2026-09-16.
Confirmed behavior is specified below; unresolved backbone details are explicitly
listed, not silently filled with Qwen or Nanochat conventions. This is not yet a
claim of an exact executable reproduction.

The current GPU scorer and arithmetic accounting are described in
[the GPU workload](m4-scoring-design.md). Optimized and reference paths use
different intermediate precision; small-fixture tolerance checks do not prove
bitwise equivalence. Current commands and validation status are in
[handoff](handoff.md).

## Scope

Build a standalone Rust/Metal implementation of the paper's architecture,
initialize it from scratch, and optimize all its independent learned parameters
through ENNX full-space BO. No pretrained Qwen weights, adapters, backpropagation,
gradient optimizer, Python model runtime, MLX, or PyTorch in the execution path.
The paper's gradient-training recipe is evidence about its experiments, not a
prerequisite or an implicit replacement for our BO experiment.

This chunk specifies architecture and execution semantics only. It does not pick
the BO objective, dataset, proposal budget, or replacement controller. In
particular, it does not silently choose teacher-forced loss instead of generation
scoring. BO from random initialization is a new experiment, not a reproduction of
the paper's reported performance.

## Source hierarchy

1. [Paper v1](https://arxiv.org/html/2608.08888v1), especially Eq. 4, Eqs. 8-11,
   Figure 2, Section 3.3, and Appendices A/C/D. Architectural authority. The paper
   is CC BY 4.0; this document paraphrases and attributes its specifications to
   Xi Wang et al., *Full-bandwidth transformer* (2026).
2. [Author's reproduction README](https://github.com/xidulu/Full-bandwidth-transformer/blob/7037c60924870aca6e30fac95212b0c7caee052d/README.md).
   Explicitly an independent Nanochat reproduction, not original Microsoft assets.
3. [Pinned model code](https://github.com/xidulu/Full-bandwidth-transformer/blob/7037c60924870aca6e30fac95212b0c7caee052d/nanochat/gpt.py)
   and [generation code](https://github.com/xidulu/Full-bandwidth-transformer/blob/7037c60924870aca6e30fac95212b0c7caee052d/nanochat/engine.py).
   Corroboration of feedback mechanics only, not authority for missing paper details.

Repository revision checked: `7037c60924870aca6e30fac95212b0c7caee052d`.
Labels used below: **paper** = stated; **derived** = follows from stated values
under explicit assumptions; **local** = our engineering contract; **open** = needs
resolution before the affected model behavior can be finalized.

## Paper model, not the Nanochat backbone

| Property | Contract | Evidence/status |
|---|---|---|
| Transformer layers | 24 | Appendix A, paper |
| Residual width | 1,536 | Appendix A, paper |
| FFN intermediate width | 6,656 | Appendix A, paper |
| FFN | SiLU GLU | Appendix A, paper |
| Vocabulary size | 100,352 | Appendix A, paper; tokenizer identity open |
| Embedding/output head | Tied | Appendix A, paper |
| Query / KV heads | 16 / 8 | Appendix A, paper |
| Head width | 96 if query projection width equals residual width | Derived; assumption must be recorded |
| Attention | Causal GQA, headwise gates, QK RMSNorm, RoPE | Appendix A, paper |
| Local attention | 2,048-token window on most layers | Appendix A, paper; endpoint convention open |
| Full attention | Every sixth layer | Appendix A, paper; use one-based 6/12/18/24 locally |
| Base context | 8,192 | Appendix A, paper |
| Norms | Around residual blocks and at final output | Appendix A; exact ordering open |
| Stabilization | Depth scaling, fused-input RMSNorm, tied head | Section 3.3; exact depth coefficient open |
| Feedback | Two width-by-width learned matrices | Eq. 4, paper |

The reproduction uses a different FFN (squared ReLU), additional value embeddings,
residual/input mixing, smear and backout mechanisms. Its parameter-free norms,
initialization and attention scaling must not become our defaults by accident.
It also allocates inactive alternative fusion matrices. Our model includes only
the active paper fusion, not unused tensors from a comparison harness.

## Feedback operation and state

Use column vectors in this specification and row-major `[out, in]` matrix storage
in Rust/Metal. Converting the paper's row-vector pseudocode requires a transpose
interpretation, not transposing the algorithm.

For current token embedding $e$ and previous completed top-layer state $z$, the
core operation is:

$$
g(e,z) = (W_U z) \odot \operatorname{sigmoid}(W_G e).
$$

Both matrices have shape `[1536, 1536]`. There is no additive token bypass,
concatenation projection, latent-token insertion, or extra transformer pass per
generated token. The feedback core adds 4,718,592 parameters (derived).

This equation alone is not the complete numerical forward specification:
Appendix C normalizes the token input to the gate and the fused input to the
stack. Section 3.3 also requires fused-input normalization. Define these as
explicit operations `N_token` and `N_fused`, rather than hiding them in a GEMM.
Whether they share learned scale parameters, their epsilon, and ordinary-prefill
input normalization remain open. The appendix repeats `input_rmsnorm_1`; that
does not establish unambiguously whether sharing is intentional.

`z` is the state associated with the logits that select the next token. Appendix
A mentions a final norm, while Appendix D says trunk hidden state: the exact
pre-/post-final-norm capture point must be settled explicitly. No arbitrary
extra normalization of `z` is allowed as an unlabelled stability fix.

## Prefill and generation semantics

These are different execution modes, not interchangeable implementations of one
forward pass. Every output/checkpoint/run configuration must name the mode.

### Standard (diagnostic control)

Run plain-token causal prefill. Each subsequent token enters through its plain
embedding. Feedback weights are unused in this mode. It is a control, not the
main full-bandwidth BO evaluator.

### Soft (paper Figure 2)

1. Prefill the entire prompt with ordinary token inputs and causal attention.
2. Retain that pass's KV cache and final prompt state.
3. Sample the first completion token from the final prompt logits. Do not run an
   additional stack pass merely to sample this token.
4. If continuing, fuse that sampled token's embedding with the retained state,
   run one stack step at the next absolute position, append its K/V, and retain
   the new state. Its logits select the next completion token.
5. Repeat until EOS or the explicit total-context/output budget.

The first completion distribution is identical to Standard for identical weights,
prompt and numerical settings. Feedback can change later distributions.
Prompt chunking in Soft must preserve ordinary-prefill semantics; chunk boundaries
must not turn into feedback boundaries.

### Fused (paper Figure 2, Eqs. 9-11)

1. Compute a plain causal prompt pass, retaining its state at every position.
2. Construct a second prompt input: first position plain; every later position
   uses its own token embedding and the previous position's state from pass one.
3. Recompute the prompt with a fresh logical KV cache at the same positions.
   Never append this pass to pass one's cache or attend to stale first-pass K/V.
4. Retain pass two's cache and final state; sample and continue as in Soft.

Use separate previous-pass and next-pass state storage if passes are tiled.
Overwriting a state before its consumer reads it changes the algorithm.
Additional passes must be explicitly requested, never inferred from context size.

### Exact sequential recurrence (reference mode)

The first position is plain; every subsequent supplied token is fused with the
state just computed at the preceding position. This implements Eq. 8 and can use
teacher-supplied tokens without sampling. It is not equivalent to a fixed two- or
three-pass parallel prefill. Comparing those modes must not be a parity test.

### Boundaries and ownership (local)

- Reject empty prompts unless an explicit tokenizer/BOS policy supplies a token.
- BOS/new-document starts use plain input and independent sequence state. Packed
  documents cannot share feedback or attention history across their boundaries.
- Identify state by candidate weight version, sequence ID, mode, pass and position.
- Changing any candidate weight invalidates its KV and carried feedback state,
  even for the same prompt. Reusing token IDs is safe; reusing old activations is not.
- Finished sequences stop updating state. Reusing their slots resets logical
  lengths and validity. EOS cannot accidentally begin another feedback step.
- Count context as prompt plus generated tokens. Check capacity before writes;
  sliding-window eviction does not reset the absolute RoPE position.

## Tensor inventory and physical parameter space

Provisional matrix inventory, assuming 96-wide heads and the conventional
three-projection SiLU GLU. Gate/norm/bias details below still need resolution.

| Tensor | Shape `[out, in]` unless lookup | Multiplicity |
|---|---|---|
| Token embedding / output head | `[100352, 1536]` | One shared allocation |
| Attention Q | `[1536, 1536]` | 24 |
| Attention K, V | `[768, 1536]` each | 24 each |
| Attention output | `[1536, 1536]` | 24 |
| FFN gate, up | `[6656, 1536]` each | 24 each |
| FFN down | `[1536, 6656]` | 24 |
| Feedback state, token gate | `[1536, 1536]` each | One each |
| Attention head gates, norms, any biases | Open | Must be explicitly inventoried |

The listed matrices total **1,064,828,928 unique parameters**, before unresolved
gates/norms/biases, approximately 2.13 GB in BF16. This is a derived subtotal,
not an exact total claimed by the paper. Count the tied head once, not twice.

Local storage contract:

- A versioned tensor manifest specifies name, shape, offset, dtype, layout and
  aliasing. Padding is not part of the BO parameter vector.
- Every learned parameter belongs to the BO vector, including learned norm/gate
  parameters once specified. Fixed RoPE tables, caches and activation buffers do not.
- Perturb the shared embedding/head allocation exactly once. Readout uses that
  same updated allocation. Alternative packed kernel layouts cannot become stale.
- Full-space means all independent learned coordinates are eligible. It does not
  mean every BF16 value changes bits on every proposal.
- Zero-initialized tensors require an explicit positive proposal scale; their
  zero RMS cannot silently freeze them.

Random initialization must specify distribution, scale per tensor class, seed,
RNG algorithm, rounding, tied aliases, and depth-scaling placement. Original-paper
initialization is not established by the available architecture description.
Do not claim the Nanochat initializer is the original initializer.

## Native Metal execution contract (local)

Rust owns manifests, buffers, command encoding, checkpoint I/O and ENNX orchestration.
Metal owns tensor computation. A small scalar Rust implementation is the numerical
test oracle only; it is not a hidden runtime inference fallback.

Required primitives: lookup, RMS reductions, Q/K/V projections, QK norm, RoPE,
causal local/full GQA, head gates, attention output projection, residual/depth
scaling, SiLU GLU, feedback fusion, final norm, tied readout and sampling. The
unresolved ordering of these primitives is not permission to omit any of them.

BF16 weight storage is the proposed initial BO format. Explicit FP32 accumulation
for reductions/softmax is required; GEMM accumulation and activation/cache dtype
must be declared and tested, not inferred from storage dtype. A precise reference
path precedes relaxed-math kernels. No quantization or weight-space restriction
is introduced by this document.

Use persistent buffers and reuse existing Apple GPU runtime infrastructure where
appropriate. Encoding separate dispatches into a command buffer does not imply
global synchronization inside a kernel. Producer/consumer ordering must be valid
for the selected storage and hazard-tracking modes. No per-layer host readback or
completion wait in the intended fast path. CPU output/token inspection happens
only at declared boundaries, with GPU completion established first.

Do not materialize quadratic attention matrices. Prefill uses tiled attention;
decode reads cached K/V. Local-layer ring buffers retain absolute position tags.
At 32K, full layers retain full history rather than silently becoming local.
GEMM tiles must support head width 96 and all remainder shapes without unsafe
loads/stores or hidden padding entering normalization.

Requested context cases are **4,096 / 16,384 / 32,768 total tokens**. The paper's
base context is 8,192: larger buffer capacity alone does not establish a matching
long-context positional configuration or useful model behavior. RoPE settings for
16K/32K remain an explicit configuration decision.

Derived BF16 KV allocation per sequence, assuming 96-wide heads, four full layers,
20 local layers and a 2,048-entry local ring:

| Total tokens | Dense cache at every layer | Local rings plus full-layer caches |
|---|---|---|
| 4,096 | 288 MiB | 168 MiB |
| 16,384 | 1,152 MiB | 312 MiB |
| 32,768 | 2,304 MiB | 504 MiB |

These exclude allocator overhead, logits, attention scratch, feedback-pass states,
weights and BO history. One BF16 prompt-state array costs 12/48/96 MiB respectively;
Fused may need two. Multiply sequence-dependent storage by concurrent sequences
and candidate-dependent storage by resident candidates. Estimate actual peak
liveness, not just model-file size. The 24 GB Mac has shared OS/application usage.

### Current size feedback

The matrix subtotal above occupies **1.9834 GiB** in BF16, excluding unresolved
learned norms/gates. A possible bias-free `[16, 1536]` head-gate projection per
layer adds 589,824 parameters (1.125 MiB). That is a sizing scenario, not a settled
paper interpretation. Feedback's two matrices alone cost 9 MiB in BF16.

The implemented primitives use FP32 activations and full-capacity KV allocations:

| Total tokens | Current KV, all 24 layers | One FP32 feedback-state array | Full FP32 vocabulary logits |
|---|---|---|---|
| 4,096 | 288 MiB | 24 MiB | 1.53125 GiB |
| 16,384 | 1,152 MiB | 96 MiB | 6.125 GiB |
| 32,768 | 2,304 MiB | 192 MiB | 12.25 GiB |

These are derived allocation sizes, not measured peak process memory. The local
rings in the earlier table are a future optimization, not current behavior.
Full-prompt vocabulary logits must not be a default allocation: use last-position
readout for decode and chunked/selected-position readout for scoring. Four fully
resident BF16 weight copies plus four 32K caches alone cost roughly 16.94 GiB,
before scratch, state, BO history and OS usage; candidate concurrency therefore
needs an explicit memory budget. Sequential candidates can reuse scratch/cache
allocations after successful completion and invalidation.

Full-context FP32 gate/up FFN intermediates also cost 1.625 GiB at 32K if both
are retained. Chunked prefill bounds this independently of the KV capacity; do
not equate support for a 32K sequence with a requirement for a 32K work chunk.

## Ambiguity register

| ID | Unresolved detail | Required disposition before affected code |
|---|---|---|
| A1 | Residual pre/post-norm graph, learned scales and epsilon | Obtain original config/code or publish an explicitly local graph |
| A2 | Head-gate input, nonlinearity, projection and placement | Define exact formula and parameter shapes; do not substitute Nanochat value gating |
| A3 | QK norm axes/scale, order relative to RoPE, score scaling | Freeze numerical order and matching test vectors |
| A4 | Depth-scaling factor and placement | Resolve runtime multiplier versus initialization scaling |
| A5 | RoPE base, pairing, rotary fraction and long-context scaling | Freeze all settings and absolute-position tests |
| A6 | Feedback norm sharing/epsilon and state capture point | Resolve Appendix A/C/D ambiguity; write full operation graph |
| A7 | Biases, tokenizer assets/IDs, exact initialization | Specify manifest and provenance; vocabulary size alone is not a tokenizer |
| A8 | Local-window inclusion and full-layer indexing | Record explicit allowed key positions at boundary cases |
| A9 | Training jitter placement | Prose/Appendix C perturb carried state; Eq. 13 depicts noise after fusion. No jitter in default evaluation |
| A10 | Multi-pass loss weighting | Eq. 12 averages feedback losses; listings simply sum. Relevant only if chosen as BO objective |

Do not wait on A9/A10 to build deterministic inference: they are training/objective
questions, not mandatory inference features. A1-A8 prevent an exact-paper parity
claim. Missing source details can be resolved as named local choices, but cannot
be retrospectively described as paper facts.

## Repository integration boundaries

Existing `rust/crates/ennx/src/apple_gpu.rs` supplies shared device/queue/pipeline
infrastructure. `bf16_metal.rs` exposes `ParamBlock`, `SearchState`, and device
buffers; these are integration candidates, not proof of a zero-copy evaluator.
[Existing proposal specification](../perturbations.md) distinguishes full continuous
support before rounding from actual BF16 changes. Preserve that distinction.

Proposed later files: `fbt.rs` for config/manifest, `fbt_metal.rs` for native
execution, `fbt.metal` for kernels, and focused Rust/Metal tests. Names are a plan,
not files created by this chunk. Do not route the new model through Qwen wrappers.
Reuse generic kernels only after checking their shapes and numerical semantics.
Avoid unrelated edits to the already-modified inference code.

## Verification and handoff

Before calling chunk 2 a correct native forward pass:

1. Close A1-A8 with sources or explicit local decisions. Pin the resulting config.
2. Build seeded tiny Rust fixtures, including asymmetric matrices that expose
   transpose errors. Use the same configurable graph as the full-size model.
3. Compare operation-level outputs against the scalar oracle; declare dtype-aware
   absolute/relative tolerances before evaluating results, and record max/RMS error.
4. Check plain full-prefill versus chunked/cached plain execution; separately check
   sequential feedback versus its reference. Never demand Soft/Fused equivalence.
5. Check Fused pass-two cache rebuilding, first-position handling and shifted-state
   indexing. Verify future-token edits cannot alter earlier states/logits.
6. Check sequence/candidate isolation, slot reuse, EOS, capacity boundaries and
   window wraparound. Changed weights must force new state construction.
7. Verify every manifest parameter is covered exactly once by BO, aliases remain
   tied, and checkpoint round trips reproduce stored BF16 weights and config.
8. Exercise 4K/16K/32K allocation/indexing and tiled remainders. Tiny fixtures prove
   semantics, not full-model throughput or learned long-context quality.

Later performance records must separate proposal, materialization, prefill,
decode, score and BO decision, reporting both synchronized wall time and GPU time.
Tokens/second must identify prompt length, output length, batch, mode, dtype and
whether prefill is included. This specification supplies no fabricated throughput
or sub-second end-to-end guarantee.

## Implementation status

The first chunk-2 slice is implemented in `fbt.rs`, `fbt_metal.rs`, and `fbt.metal`:
explicit feedback configuration, checked two-matrix BF16 layout, batched native
feedback, optional unit-scale token/fused RMS normalization, and plain-row bypass.
The evaluator uses FP32 arithmetic and persistent scratch. It encodes into a
caller-owned command buffer without submission, host synchronization or readback.
The current projection kernel is a SIMD GEMV correctness path, not a tuned batched
prefill GEMM. Normalization choices have no implicit default. Standalone learned
RMSNorm is now available, but is not wired into feedback; A6 remains open.

The next slice implements a native bias-free SiLU-GLU core and residual update
through `FeedForward`. Gate/up matrices are `[intermediate, width]`; down is
`[width, intermediate]`. BF16 weights are read directly on each call and all
activation arithmetic is FP32. Gate/up plus SiLU multiplication share one
dispatch; down projection plus scaled residual addition share a second. The
caller supplies the input, separate residual and explicit residual multiplier.
Surrounding norms and depth scaling are not silently selected by this primitive.
This is still a correctness-first SIMD path, not a tuned prefill GEMM.

`fbt_attention.rs` and `fbt_attention.metal` implement learned RMSNorm and causal
GQA with persistent BF16 K/V storage. Attention uses optional unit QK RMSNorm
before full-dimension, split-half RoPE; theta and score scale are explicit.
Q stays FP32; K/V round to BF16. These are explicit local numerical conventions,
not a claim that the paper resolves every choice. Head output multipliers are
supplied by the caller, not computed from gate logits inside attention.

`Linear` now provides bias-free BF16 projections with explicit identity/sigmoid
activation and FP32 accumulation/output. It can supply Q/K/V, output and head-gate
projections without caching converted weights. The original SIMD path remains
the reference; `encode_tiled` adds explicit prefill GEMM. Graph wiring and the head-gate formula remain
open; providing a sigmoid option does not resolve the paper ambiguity.

Each cache belongs to one layer and one candidate/sequence/pass key. Encoding
rejects mismatched keys, capacity overflow, aliased buffers, a different command
queue and uncommitted predecessor commands. Reset requires all retained commands
to have finished; `check_completed()` checks every retained command for successful
completion before results may be consumed. The position is a scheduled cursor,
not evidence of completion. Encoding performs no GPU allocation, submission,
wait or host readback; host command tracking retains outstanding commands.

Attention uses online softmax without a quadratic score buffer. Local windows
include the current token. Storage remains full-capacity rather than a ring so
large prefill chunks cannot overwrite keys needed by earlier queries. This is a
correctness-first SIMD implementation, not a tiled high-throughput prefill kernel.

`fbt_metaltests.rs` contains an independent f64 scalar oracle and GPU tests for
asymmetric matrices, width 1/31/33/96/1536, all four normalization combinations,
masked state isolation, output bounds, live weight changes, saturated gates, and
chained dispatches over 4,096 rows. `tests/fbt_metaltest.rs` is a standalone
harness following the existing native-backend test pattern.

FFN tests cover asymmetric matrices, non-aligned dimensions, width 1,536 and
intermediate width 6,656 independently, signed/zero residual scales, live weight
updates, output bounds and invalid buffers. A chained feedback-to-FFN test uses
one command buffer with no intermediate host readback.

Attention tests compare 16-query/8-KV-head, head-dimension-96 outputs against an
independent dense f64 reference, with full/local attention and QK normalization
on/off. GPU outputs are bit-identical across chunk sizes 1, 3 and 7. Tests cover
causality, learned RMSNorm, live gamma changes, output bounds, cache ownership,
reset and rejected invalid configurations. The 4K/16K/32K capacity tests use tiny
heads and a one-token window: they verify indexing, not model throughput or
long-context language quality.

Projection tests cover asymmetric/non-aligned shapes, width 1,536, identity and
sigmoid, live BF16 weight updates, output bounds, aliasing, short buffers and
submitted-command rejection against an independent scalar reference.

Verification on the local M4: twelve integrated FBT tests passed with Metal API
and GPU validation enabled; no validation errors were reported. Focused
`rustfmt --check` passed. The Qwen `timed_command!` macro-hygiene compilation error
was fixed by passing its command-buffer binding as a macro argument. A small-model
regression test compares profiled and unprofiled decode logits/tokens across three
steps and passes with Metal validation. This Qwen maintenance does not introduce
Qwen dependencies or architecture choices into FBT.

The full Buck `ennx-unit` run before adding Linear, including the Qwen regression, new attention tests
and its native-index feature set, passed 490 tests with one ignored and zero
failures. After adding Linear, the full Cargo Metal library suite passed 487
tests with one ignored and zero failures (a different feature set from Buck).
Reproducible commands:

```sh
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 \
  cargo test --offline -p ennx --no-default-features --features metal \
  --lib fbt -- --test-threads=1
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 \
  cargo test --offline -p ennx --no-default-features --features metal \
  --lib profiled_decode_matches_unprofiled_decode -- --test-threads=1
cargo test --offline -p ennx --no-default-features --features metal \
  --lib -- --test-threads=1
BUCK_ISOLATION_DIR=dev ./buck2w test //rust/crates/ennx:ennx-unit --local-only
```

Cargo still reports an existing unused runtime `schedule` helper and a dependency
future-compatibility warning for `block` 0.1.6. These are warnings, not test or
compilation failures. No claim of CUDA/Python-wheel or whole-workspace validation
is made by this Metal library test record.

A complete local forward/scoring graph now exists (LocalV1 below), with explicit
initialization and feedback state management. Tokenization, free-running sampled
generation, checkpoint serialization and BO orchestration are not implemented
by this graph. A1-A8 remain unresolved as paper-reproduction questions; local
choices are not evidence of the original architecture.

No long BO run, commit or push is authorized by this file.

## Whole-model LocalV1 execution

`fbt_model.rs` exposes `Model`, `ModelConfig`, `ScoreMode` and `Score` through
`ennx::fbt`. It is a native Rust/Metal graph, not a Python framework wrapper.
The named **LocalV1** choices are:

- Learned pre-attention and pre-FFN RMS scales; final learned RMS scale.
- QK unit RMS before split-half RoPE, followed by causal GQA.
- Sigmoid per-head gates projected from the normalized attention input, applied
  to head outputs before attention output projection.
- Caller-specified residual scale for both branches; no biases.
- Feedback captures post-final-norm states. Token/fused unit normalization is
  explicitly configured. The first sequence position remains plain.
- BF16 tied embedding/readout, one independent parameter allocation, no copy.
- SplitMix64 initialization: uniform [-sqrt(3/fan_in), sqrt(3/fan_in)], BF16 RNE,
  with norm vectors exactly one. Tensor order and seed determine the stream.

This is an explicit executable variant, not a claimed exact paper reproduction.
The target timing configuration uses RoPE base 10000, epsilon 1e-5 and residual
scale 1/sqrt(48); these are local experiment choices, not recovered paper values.
There is no pretrained language capability claim for scratch initialization.

The graph includes lookup, norms, Q/K/V and head-gate projections, attention,
output/residual updates, SiLU GLU and down projection, feedback, final norm,
tied vocabulary projection and stable cross entropy. Projections use the tiled
path for at least 64 rows and the SIMD path for smaller tails; 64 is a conservative
initial dispatch choice, not a measured optimal crossover. All layer activations
stay on GPU; output logits exist only for the current chunk. Losses are read once
after successful completion, not after each layer or token.

`score(tokens, targets, mode)` predicts `targets[i]` after consuming `tokens[i]`.
Labels are explicit: it does not silently shift tokens, tokenize text or sample
generations. Standard is the ordinary causal control; Fused runs a plain pass
then a fresh-cache shifted-feedback pass; Sequential runs exact teacher-supplied
recurrence one token at a time. Only the final pass is scored. Sequential is not
claimed equivalent to Fused. Each invocation begins a new sequence and fresh KV
validity. Parameter replacement rejects wrong shapes/non-finite values and bumps
the candidate revision; tied readout automatically sees the changed embedding.

Commands are queued per chunk, with no per-layer host waits. A pass boundary is
currently a host completion boundary. All submitted commands are checked; error
paths drain submitted/partially encoded work before returning, so scratch and
state cannot be reused while prior GPU work remains active. `Score` reports total
call wall time and individual pass times. It also separates host encoding/submission
(including queue backpressure) from the final completion waits. GPU work overlaps
encoding, so neither number is labelled GPU-only execution time. Initialization
is outside scoring time.

The scorer also records completed command-buffer GPU intervals per chunk, using
Apple's public [GPU start](https://developer.apple.com/documentation/metal/mtlcommandbuffer/gpustarttime)
and [GPU end](https://developer.apple.com/documentation/metal/mtlcommandbuffer/gpuendtime)
properties through Objective-C because the current metal-rs binding lacks wrappers.
Unavailable timestamps are `None`, not fabricated zeros. These are whole-chunk
intervals, not per-kernel attribution. Public construction/scoring own their
autorelease pools; callers do not need an Objective-C runtime wrapper. Total
score wall time includes temporary-object cleanup.

`ModelConfig::memory()` reports checked weight, KV, workspace, total and largest
buffer sizes before model allocation. The requested target allocation is 4.7731
GiB: 1.9846 GiB weights, 2.25 GiB KV, 0.5385 GiB workspace. This excludes driver
rounding/pipelines. Construction checks the aggregate budget before initialization
and retains per-allocation checks for failures and concurrent memory pressure.

End-to-end tests use an independent dense f64 graph, compare all final states and
per-token losses, and exercise Standard/Fused/Sequential, chunk sizes 1/3/64,
local/full layers, reset/repeated calls, invalid IDs/lengths and changed weights.
State tolerance is 0.006 absolute and loss tolerance 0.004, accounting for BF16 KV
rounding at numerically close boundaries; mean-loss tolerance is 0.003. These are
small-graph correctness checks, not a language-quality benchmark.

The full-size timing test uses 24 layers, width 1536, FFN 6656, 16/8 heads,
vocabulary 100352, context capacity 32768, chunks of 256 and seed 42. It warms a
256-token Fused evaluation, then actually scores 4K/16K/32K synthetic sequences,
one candidate each. Report these as single measured runs, not median throughput.
The manifest contains **1,065,494,016** parameters (1.9846 GiB BF16). Local Metal
allocation after construction was **4.7735 GiB**, excluding unrelated process/OS
memory. Construction took 2.310 seconds in the first target run.

The initial 4K complete Fused run took **56.200058 seconds**, with pass times
27.073864 and 29.126170 seconds. This includes all 24 layers, two passes and final
readout/scoring, not just isolated projections. Synthetic mean NLL was 12.016938;
it is a finite-output check, not a coding-quality result. Larger-context results
are exploratory single runs, not controlled benchmark medians; toolchain setup,
host compilation and later small correctness tests overlapped parts of the run.
The 16K Fused evaluation completed in **371.145810 seconds**, with pass times
180.923676 and 190.222057 seconds, mean NLL 12.004553. A host process sample during
this run found the scorer blocked obtaining a Metal command buffer; this shows
queue backpressure, not a specific GPU kernel bottleneck. It is the reason for
adding separate submission/wait timings and completed GPU intervals. The initial
long run predates these diagnostic fields, so they cannot be reconstructed from
its existing wall-time results.

The original attention baseline also completed at **32K: 1081.813792 seconds**,
with pass times 530.567943 and 551.245707 seconds, mean NLL 12.011120. All three
full-size cases returned finite losses and successful command completion. The
whole initial harness took 1513.63 seconds. No per-kernel attribution follows
from those baseline wall times, especially with the documented overlapping work.

New model Rust files are formatted with the latest stable formatter installed on
2026-09-16: rustfmt 1.9.0-stable, build 48a229ceae (2026-09-01), from Rust 1.98.1,
preserving the repository's existing formatting conventions. An initial 2024-style
formatting pass was aligned back to the repository policy without downgrading the
formatter. The workspace's Rust language edition 2021 and pinned build compiler
1.96.0 are intentionally unchanged; language edition is not the formatter's
release date. Use `rustfmt +stable` for this work.

```sh
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 \
  cargo test --offline -p ennx --no-default-features --features metal \
  --lib fbt_whole_model -- --test-threads=1
cargo test --offline -p ennx --no-default-features --features metal \
  --lib fbt_whole_model_target_timing -- --ignored --nocapture --test-threads=1
```

Initial whole-model tiled-attention measurements (single runs after the original
baseline finished; no overlapping build/test workload during timed scoring):

| Context | Original attention, seconds | Tiled attention, seconds | Tiled mean NLL |
|---|---|---|---|
| 4,096 | 56.200058 | 34.239418 | 12.016938 |
| 16,384 | 371.145810 | 164.486244 | 12.004552 |
| 32,768 | 1081.813792 | 396.723249 | 12.011120 |

The 4K tiled pass times were 16.242421/17.996916 seconds; 16K pass times were
77.456996/87.028932 seconds. Host encoding/submission took 0.011427 seconds at
4K and 0.040960 seconds at 16K; completion waits were 34.227883/164.444904 seconds.
Completed GPU command intervals closely accounted for the pass times in these
runs. These intervals are not a per-kernel breakdown. The original baseline had
the overlapping work noted above, so these are exploratory comparisons, not
controlled repeated benchmark speedup estimates.

At 32K the tiled pass times were 188.774507/207.948165 seconds. Encoding/submission
wall time was 164.569678 seconds and final waits 232.152889 seconds. Submission
includes waiting for command-buffer slots while GPU work progresses, not that
much CPU computation. The sums of completed GPU intervals were
188.771182/207.944245 seconds, closely matching pass wall times. The last chunk's
interval was 2.375299/2.079331 seconds versus 0.815652/1.274971 for the first chunk.
The entire three-context tiled harness took 599.54 seconds. All commands completed
successfully and final scores were finite. This establishes a whole-model
execution baseline; it does not establish coding quality or a complete BO round.

Remaining work: full-graph profiling/optimization (feedback remains the earlier
SIMD kernel; tiled attention is now an explicit option), sampling/tokenization, durable checkpoints,
and BO integration. The native scoring endpoint is ready for integration, but
this is not yet a working post-training experiment on coding data.

## Optimization experiment H1: grouped projections

Hypothesis: grouping independent projections sharing an input reduces CPU/encoder
and dispatch overhead. It does not reduce mathematical FLOPs or matrix bytes.
`Linear::encode_grouped` handles two or three projections in one dispatch while
preserving separate live BF16 weight allocations and separate FP32 outputs.
There is no repacking copy, and the default individual-projection path is unchanged.
The FFN cases below measure only two plain projections, not the existing fused
SiLU-GLU kernel; they do not justify replacing that kernel.

Budget for width 1536: Q/K/V have 4,718,592 weights (9 MiB), requiring 9,437,184
FLOPs per row. FFN gate/up have 20,447,232 weights (39 MiB), requiring 40,894,464
FLOPs per row. Grouping reduces dispatches from three/two to one. Current SIMD
GEMV-style kernels do not explicitly reuse weights across prompt rows; tiled GEMM
is a distinct, higher-leverage hypothesis for prefill.

Local M4 measurements, 2026-09-16, validation disabled: five warmups and twenty
samples per mode, alternating order, median wall time including encoding,
submission and completion wait. Compilation and allocation are excluded. These
are warm repeated isolated projection tests, not full-model latency estimates.

| Projection shapes (input width 1536) | Rows | Separate | Grouped |
|---|---|---|---|
| Q/K/V outputs 1536/768/768 | 1 | 1.0982 ms | 1.0300 ms |
| Gate/up outputs 6656/6656 | 1 | 2.4330 ms | 2.4567 ms |
| Q/K/V outputs 1536/768/768 | 16 | 1.8444 ms | 1.8356 ms |
| Gate/up outputs 6656/6656 | 16 | 8.1279 ms | 8.0274 ms |

Conclusion: modest single-row Q/K/V improvement in this run; no compelling
large-shape improvement. No automatic dispatch policy is selected from this one
run. First optimize row/weight reuse and measure GPU execution separately from
submission overhead before extrapolating. Small differences need repeated trials.

Correctness: grouped/separate outputs are bit-identical for two/three projections,
mixed activations, tail dimensions, multiple rows and model-sized Q/K/V. Metal
API/GPU validation passed all 13 active FBT tests; the timing test is ignored by
default. Reproduce the timing experiment without validation:

```sh
cargo test --offline -p ennx --no-default-features --features metal \
  --lib fbt_grouped_projection_timing -- --ignored --nocapture --test-threads=1
```

### Full-context acceptance workload

`fbt_full_context_projection_timing` executes every row at 4,096, 16,384 and
32,768 tokens. Each token has distinct synthetic FP32 inputs, prepared outside
timing. Each projection uses the actual model width and output dimensions, live
BF16 weights and FP32 accumulation. Persistent output scratch holds 256 rows;
chunk commands are submitted to the same ordered queue with one explicit host
wait at the end of each context. Queue backpressure is included in wall time.

One warmup per mode precedes three timed trials, alternating separate/grouped
order. Allocation, compilation and output inspection are excluded; encoding,
submission and final completion are included. Each command's success is checked.
Final-chunk outputs must be finite and bit-identical across modes/trials;
whole-context outputs are not retained. Existing independent scalar and grouped
parity tests cover numerical correctness; this harness measures projection work,
not causal attention or a complete scoring graph. GPU-only timestamps are not
reported because the current Metal binding does not expose them.

The experiment is **one layer's projections, one candidate**, not a four-candidate
BO round, autoregressive generation or a full 24-layer teacher-forced evaluation.
It does not include FFN activation/down projection, feedback, readout or attention.
Long contexts are executed, not extrapolated from a one-token or 256-row sample.
256 is an execution chunk size, not the evaluation's context length. This first
baseline fixes chunk size; a chunk-size sweep belongs with the tiled GEMM path.

Measured local M4 medians (seconds), 2026-09-16, validation disabled:

| Context | Q/K/V separate | Q/K/V grouped | Gate/up separate | Gate/up grouped |
|---|---|---|---|---|
| 4,096 | 0.655516 | 0.673724 | 2.878061 | 2.955270 |
| 16,384 | 2.626542 | 2.687655 | 11.526407 | 11.944480 |
| 32,768 | 5.265597 | 5.460155 | 23.072549 | 24.346923 |

All commands completed successfully and final-chunk parity checks passed. The
entire repeated experiment took 380.36 seconds; this is harness runtime, not one
evaluation. Grouping regressed every tested full-context case in this run. Do not
enable it for prefill based on the earlier single-token result. Next hypothesis:
reuse weights across rows with tiled SIMD-group matmul, using the existing native
matmul implementation as a candidate rather than adding another GEMV variant.
Validate its layout and numerical error before benchmarking the same contexts.

```sh
cargo test --offline -p ennx --no-default-features --features metal \
  --lib fbt_full_context_projection_timing -- --ignored --nocapture --test-threads=1
```

## Optimization experiment H2: tiled prefill matmul

Hypothesis: reuse BF16 weights across prompt rows rather than recomputing each
row with an independent GEMV. `Linear::encode_tiled` adapts the repo's existing
SIMD-group matmul strategy, not its Qwen model architecture. Each 128-thread
group computes a 64-row by 32-output tile with an 8-element reduction tile.
Weight elements are widened into shared FP32 staging and reused across rows;
the mathematical FLOP count is unchanged. Matrix accumulators are FP32, BF16
weights are read live, and there is no packed or converted persistent copy.

Unlike the source kernel's aligned output stores, the FBT kernel stages results
in threadgroup memory and bounds-checks writes. Input/reduction tails are zero
padded locally; padding never enters the parameter vector. Threadgroup storage
is 11 KiB (2 KiB input, 1 KiB weights, 8 KiB output). Identity/sigmoid activation
is applied during the final store. No GPU scratch allocation, submission or host
readback occurs in the encode API. The 128-thread capability is checked at setup.
The existing SIMD and grouped APIs remain available; no decode policy is changed.

Correctness tests compare to an independent f64 scalar oracle, including rows
1/3/63/64/65, odd input/output dimensions, model-width K/V and FFN shapes, both
activations, sentinel output bounds, invalid buffers/commands and live weight
changes. The acceptance tolerance is absolute error <= 0.0002 * (1 + abs(reference));
different reduction order means bitwise equality is not required. All 14 active
FBT tests passed under Metal API/GPU validation (three timing tests ignored).

The full-context comparison reuses H1's workload and scheduling, replacing only
the grouped mode with separate tiled projections. Both implementations are timed
in the same run; allocation/compilation remain excluded. Final-chunk outputs use
the same tolerance as the independent correctness tests. This still measures
only one layer's projections, not complete model scoring or a BO round.

Measured local M4 medians (seconds), 2026-09-16, validation disabled:

| Context | Q/K/V SIMD | Q/K/V tiled | Gate/up SIMD | Gate/up tiled |
|---|---|---|---|---|
| 4,096 | 0.660279 | 0.057094 | 2.880002 | 0.242223 |
| 16,384 | 2.622703 | 0.242339 | 11.536355 | 1.011904 |
| 32,768 | 5.260981 | 0.483815 | 23.106561 | 2.010553 |

The measured projection speedups are 10.8-11.9x. All commands and final-output
checks passed; the repeated harness took 203.01 seconds including both modes.
This supports using tiled projection for the tested 256-row prefill chunks, not
a claim of equivalent model/BO speedup. The API remains explicit pending graph
integration and small-row crossover measurements. Next work is tiled FFN
activation/down-projection integration and attention/feedback/readout profiling,
with full-context scoring as the eventual end-to-end acceptance test.

```sh
cargo test --offline -p ennx --no-default-features --features metal \
  --lib fbt_full_context_tiled_timing -- --ignored --nocapture --test-threads=1
```

## Whole-graph attention optimization

`Attention::encode_tiled` and `ModelConfig::tiled_attention` add a 96-wide-head
prefill path. Eight queries reuse a 16-key BF16 tile; QK and probability-times-V
use FP32 SIMD-group matrix operations, with online softmax and exact causal/local
masking. No quadratic score allocation is introduced. Other head dimensions
reject explicit tiled requests; the model uses the reference path for chunks
shorter than eight rows. This is an execution choice, not an architecture change.

The initial 32-key prototype failed Metal validation: instrumentation raised its
threadgroup memory requirement to 45,248 bytes, above the 32,768-byte limit.
The corrected 16-key tile uses 15,968 bytes before validation instrumentation
(31,936 with the observed instrumentation), and construction checks pipeline
threadgroup memory/thread capability. The failed prototype was not used by the
model. The scalar-SIMD attention path remains available for comparison.

Tests compare full/local windows 1/3/33, 67 positions, chunk sizes 1/8/17/67,
QK normalization on/off, signed/zero head multipliers and tail bounds against
the dense f64 attention reference (0.002 absolute tolerance). Complete 96-wide-head
model tests compare Standard/Fused states and losses to both the original GPU
path and the independent scalar graph. These passed Metal API/GPU validation.

Final verification after integration: **19 active FBT tests passed** with Metal
API/GPU validation (four manual timing tests ignored by the ordinary run).
The full Buck `ennx-unit` target passed **498 tests**, with five ignored and zero
failures. Both full-size timing variants were executed explicitly, including all
4K/16K/32K contexts. Latest-stable scoped rustfmt checks passed. No commit or push
was performed.

The target timing harness defaults to tiled attention; set
`ENNX_FBT_TILED_ATTENTION=0` to reproduce the original path. An optional
`ENNX_FBT_CONTEXTS` comma-separated list selects only 4096, 16384 and/or 32768;
without it all three execute. This avoids rerunning every context when isolating
a change without substituting a short-token microbenchmark.

```sh
ENNX_FBT_CONTEXTS=4096 ENNX_FBT_TILED_ATTENTION=1 \
  cargo test --offline -p ennx --no-default-features --features metal \
  --lib fbt_whole_model_target_timing -- --ignored --nocapture --test-threads=1
```

## Complete BO round measurement

The optimization target is elapsed wall time for a complete BO round at 4096,
16384 and 32768 positions. Whole-model scoring is one component, not the final
metric. The FBT measurements above are synthetic, randomly initialized model
scoring; they do not yet measure a BO loop or autoregressive generation.

The round boundary starts before minibatch selection and finishes after the
completed decision, incumbent update, event flush and progress output. Include:

1. Minibatch preparation and incumbent rescoring when required.
2. Acquisition search, dense correlated proposal materialization and application.
3. Selected-candidate objective scoring, including both passes in Fused mode.
4. Paired decision, search/controller update and completed GPU synchronization.
5. Incumbent bookkeeping and logging.

Do not multiply scoring time by the acquisition pool size. The existing Metal
search selects one of four acquisition candidates for objective evaluation.
Refreshing a minibatch additionally evaluates the incumbent: two objective calls
in that round, not four. Each objective call can itself score multiple examples.
Report all three counts separately.

The existing Qwen loss driver (`ops/qwen/bo.py`) now writes complete-round wall
timings to `run.json` under `timing.rounds`. Proposal and candidate scoring remain
one combined phase because their native pipeline shares a command queue; nested
native profiles must not be added to that phase. Decision submission, decision
synchronization, logging and remaining host work are separately accounted for.
The outer `bo_loop_seconds` also includes loop/file lifecycle and timing-record
bookkeeping. Setup includes the separately reported baseline; checkpoint export
is reported outside the loop. Final report serialization is not timed. Completed
round timings are in the final report; live events retain their existing format.

The historical FBT manual harness connected the native search's
flat BF16 proposal allocation to model parameter buffers using synchronous GPU
blits. It preserves tied weights, covers every parameter, advances the model
revision and restores the accepted incumbent after each decision. A tiny-model
test compares scoring against explicit parameter replacement and verifies exact
restoration for rejection and acceptance. This bridge is currently test-harness
code, not a public model/search API. Do not substitute Qwen timings for FBT.
Measure cold setup/first round separately from repeated steady-state rounds;
record memory usage, objective mode, batch size, context, candidate count and
history configuration with each result. Kernel optimizations are accepted on
complete-round improvement with unchanged objective and numerical checks, not
isolated kernel speedups alone.

### Native 4K BO timing workload

The historical timing harness ran three complete rounds on the 1,065,494,016-parameter
LocalV1 model, capacity 4096, chunk 256, Fused scoring and tiled attention. It
uses deterministic synthetic token/target IDs and random model initialization,
not a pretrained checkpoint or a coding-quality objective. No checkpoint is
exported. The existing BF16 search uses Thompson acquisition, a four-candidate
pool, one selected proposal, two history slots, initial radius 0.01, bounded
absolute FIFO history and the shared TuRBO controller. Each tensor's initial RMS
sets its fixed perturbation scale; norms and feedback matrices participate too.

Every round refreshes two examples and scores incumbent and selected proposal
on the same examples. This means two batch-level objective evaluations, four
sequence scores and eight complete transformer passes per round. Acceptance
requires positive paired mean improvement greater than twice its estimated
standard error. These synthetic batches are treated as samples, with no finite
population correction. Tiny bridge tests force both controller branches; the
timing run makes actual measured acceptance decisions.

Setup includes initial model/search allocation and CPU flattening/RMS work.
There are no per-round CPU weight copies. The first ask includes lazy reference
initialization; there is no unreported warmup. Timings separately expose
incumbent scoring, ask, proposal application, candidate scoring and completed
decision/restoration. The outer loop timer includes result logging. GPU blits
and synchronization are deliberately retained and measured, not presented as
an optimized zero-copy pipeline.

The bridge test passes with Metal API validation. GPU shader validation currently
aborts in the existing `bf16_propose_pool`: instrumentation doubles its 20,480
bytes of threadgroup storage to 40,960, above the device's 32,768-byte limit.
This is an unresolved validation limitation; timing runs disable instrumentation.

The duplicate ignored timing-test entry point has been removed. Run the current
study through the [TuRBO-ENN runbook](turbo-enn.md):

```sh
./ennx tune examples/tuning/turbo-enn.toml
```

The implementation details and measurements in this section describe the
historical harness, before the current parameter-buffer binding implementation.

Measured on the local M4 on 2026-09-16, validation disabled, with no overlapping
GPU tests/builds during the timed loop:

| Round | Incumbent score | Ask | Apply weights | Candidate score | Decision + restore | Whole round | Accepted |
|---|---|---|---|---|---|---|---|
| 1 (cold) | 67.603774 s | 3.722139 s | 0.619578 s | 67.374770 s | 0.237469 s | 139.557772 s | No |
| 2 | 70.039128 s | 0.712350 s | 0.053969 s | 67.843460 s | 0.224014 s | 138.872950 s | No |
| 3 | 68.201037 s | 1.035466 s | 0.044844 s | 69.219317 s | 0.929017 s | 139.429763 s | Yes |

Setup took 7.320522 seconds; the complete loop took 417.860530 seconds
(139.286828 seconds per round on average). Metal allocations were 12.4014 GiB
after each round. Scoring accounted for 98.19% of loop wall time. Whole-round
times include writing each result line; the outer loop also includes the
subsequent timing-line writes. These three observations are not a statistically
stable performance distribution. The accepted third proposal took longer in
the decision stage; this branch updates the reference as well as the incumbent,
but these timings do not isolate the cost of each update. Selected proposal
radii were 0.005, 0.02 and 0.005 respectively;
these are not changes to the initial controller radius.

Paired incumbent/candidate mean NLLs were 12.007409502/12.008251559,
12.036529561/12.038116154 and 12.023275490/12.021930606. Different rounds use
different examples, so cross-round NLL comparisons are not improvement claims.
Raw output: `.cache/fbt-bo-4k-20260916.log`. The native harness passed and
completed all three rounds. No generated-code quality or generation TPS was
measured.

Post-run verification: the standalone FBT/search integration target passed
27 tests with Metal API validation; five manual timing tests were ignored.
The 4K BO timing test above was executed separately, not skipped. Scoped checks
used the latest installed stable rustfmt; no commit or push was performed.

## Fused scoring backend (2026-09-17)

The replacement path spans matrix projection, FFN/residual output operations
and vocabulary loss; it is not a change to the objective or weight precision.
`Model::set_optimized(false)` retains the original graph for parity checks.
The BO harness accepts `ENNX_FBT_OPTIMIZED=0|1` and `ENNX_FBT_ROUNDS=1|2|3`;
the default round count remains three. A one-round run still performs the same
complete 4K workload, including incumbent rescoring and decision restoration.

The new GEMM reads full activation tiles directly from device memory and widens
live BF16 weights over 32-element reduction slabs. Consecutive threads load
consecutive weight elements; padded shared storage transposes their placement
for matrix operations. FP32 accumulation and original parameter allocations
are preserved. This still uses SIMD-group matrix instructions, not MPP, an M5
accelerator path or a new precision format. Tail tiles use bounded staging.

The up-projection epilogue applies SiLU(gate) times up without writing a separate
up tensor. Attention-output and FFN-down epilogues update the residual directly.
Each vocabulary tile emits its maximum, exponential sum and target contribution;
a second reduction computes exact full-vocabulary cross entropy. It avoids
materializing logits but does not prune vocabulary arithmetic. At the target
vocabulary size, output scratch is one eighth of the original logits allocation.
The reference graph's FFN/branch scratch allocations are retained for comparison.

Transformer rows below eight retain the original GEMV/elementwise path. Routing
single-token recurrent feedback through the new matrix reductions caused
per-token differences outside the existing tolerance on a 67-token test; the
tiny-row fallback preserves the original recurrent execution. No tolerance was
loosened. Full/tail tiles, Standard/Fused/Sequential scoring, chunk sizes 3/64/67
and targets at both vocabulary edges pass comparisons against the original
graph with Metal API and shader validation.

The first wider-reduction implementation completed one full round in
151.838252 seconds (incumbent 70.306390, ask 2.954259, apply 0.688132,
candidate 76.682490, decision/restore 1.206908). This was not a speedup over
the earlier run. The remaining rounds were deliberately interrupted, and the
strided weight loads were replaced before the next measurement. The log
`.cache/fbt-bo-4k-fused-20260917.log` records that interrupted experiment, not
a successful three-round run.

Cross-example batched GEMMs, revised attention scheduling, dispatch replay and
an optimized weight-update layout are not implemented by this change. The two
examples remain sequential. This is an integrated scoring-backend step, not a
completed SOTA engine or a claim that the sub-second BO goal has been reached.
