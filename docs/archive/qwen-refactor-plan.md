> Archived on 2026-09-29. Historical evidence and superseded plans, not current
> instructions. Read [current state](../handoff.md) and [runbook](../turbo-enn.md).
> Original dated results are retained; relative documentation links were relocated.

# Qwen Metal Inference Refactor Plan

## Target

We need a reliable answer to one question: how fast can this Qwen2.5-Coder-1.5B
Metal path use the model's real context budget when the hot path is written like
an inference engine, not a correctness reference.

The benchmark target is the long-context path at exactly three context lengths:

- 4K tokens;
- 16K tokens;
- 32K tokens.

Short toy runs are not the target. A 128-token decode run can catch basic
breakage, but it cannot validate the performance shape we care about. The
critical question is what happens as the KV cache grows toward the full 32,768
token context.

The native report must split:

- prefill latency;
- first-token latency;
- steady decode latency per token;
- end-to-end generated tokens/sec;
- host overhead outside Metal.

Python may tokenize, call the native function, and print JSON. Python must not
own a per-token loop or any repeated logits readback for the TPS benchmark.

## Current State

The path is better than the original full-prefix generation, but it is still
too reference-shaped.

Current implementation files:

- `rust/crates/ennx/src/qwen_metal.rs`
- `rust/crates/ennx/src/qwen.metal`
- `rust/crates/ennx/src/qwen_linear.metal`
- `rust/crates/ennx/src/qwen_attention.metal`
- `rust/crates/ennx-py/src/py_qwen.rs`
- `ops/qwen/bench.py`

Known current behavior:

- `generate` is Rust-owned and uses a BF16 KV cache.
- prefill is chunked through `forward`.
- greedy decode keeps argmax on GPU.
- sampled decode still reads logits to CPU.
- batch generation exists, but initial prefill is still per prompt.
- `ops/qwen/bench.py` reports only total cached generation time.
- the current benchmark does not separate prefill, first token, and steady
  decode.
- the loss/BO path is teacher-forced forward, not autoregressive generation.

The current benchmark is therefore not good enough to compare against reported
high-TPS systems. It mixes different phases and does not expose the real decode
cost.

## What Is Probably Slow

### 1. Decode Uses Too Many Kernels Per Layer

For each decoded token, every transformer layer still encodes a chain of small
kernels:

- RMSNorm;
- QKV projection;
- RoPE/cache write;
- decode attention;
- output projection;
- residual;
- RMSNorm;
- MLP projections;
- SiLU;
- down projection;
- residual.

This is structurally expensive on Apple GPUs because one-token decode has tiny
row count. The arithmetic is not the only cost. Kernel launch, encoder count,
memory traffic, and poor small-GEMV occupancy matter.

High-throughput inference stacks win by making decode a small number of
well-shaped kernels and by keeping the GPU fed.

### 2. One-Token GEMV Is The Center Of The Problem

Decode for a dense transformer is dominated by many matrix-vector products:

- Q projection;
- K projection;
- V projection;
- O projection;
- gate projection;
- up projection;
- down projection;
- final LM head projection.

For Qwen2.5-Coder-1.5B, hidden size is 1536 and intermediate size is 8960.
These are not huge matrices by GPU standards, but one-token GEMV has low reuse.
If every projection is a separate generic kernel, throughput will be poor.

The LM head is especially large: hidden to 151,936 vocab. Greedy generation only
needs the max token, not a host-visible logits vector.

### 3. Prefill And Decode Need Separate Benchmarks

Prefill is closer to GEMM because it has many rows. Decode is mostly GEMV.
Comparing total generation TPS across systems without separating these phases
is misleading.

For the BO loop, the important paths are:

- teacher-forced minibatch loss for weight candidates;
- optional generation objective for code completions;
- any decoder-parameter search around temperature/top-p.

Those must not be collapsed into one vague "generation speed" number.

### 4. Batch Decode Is Underspecified

The code has `generate_batch`, but the path is not yet designed like an
inference engine:

- prompt prefill is serial per prompt;
- cache layout is batch-aware but not yet tuned;
- greedy batch avoids CPU logits readback;
- sampled batch reads full logits to CPU.

If the benchmark tries to match high TPS results, batch size must be explicit.
Single-stream decode TPS and batch decode TPS are different claims.

### 5. The BO Loss Path Has A Different Bottleneck

The measured BO loss path spends almost all time in model forward and LM head
projection. Proposal materialization is not the bottleneck right now.

Previous measured two-row minibatch loss calls were roughly:

- candidate loss: about 3.5 to 4.0 seconds after warmup;
- incumbent loss: about 2.8 to 3.5 seconds after warmup;
- output projection alone: about 0.33 to 0.43 seconds per two-row call;
- proposal/search overhead: under a few milliseconds in the measured path.

That means optimizing proposal code before fixing inference/forward is the wrong
order for the sub-second BO-loop target.

## Measurement Contract

Add one native profiling surface first.

### Rust API

Add a method on `MetalQwenEvaluator`:

```text
bench_generate(weights, prompt, max_new_tokens, mode) -> QwenGenerationProfile
```

The profile must include:

- prompt token count;
- requested generated token count;
- actual generated token count;
- prefill milliseconds;
- first token milliseconds;
- decode milliseconds after first token;
- total milliseconds;
- steady decode tokens/sec;
- end-to-end generated tokens/sec;
- command buffer count;
- whether logits were read to CPU;
- whether token selection stayed on GPU.

For greedy mode, the native benchmark should not read logits to CPU.

### Python CLI

`ops/qwen/bench.py` should call this native method and print JSON. It should not
time a Python loop as the main result.

The CLI should support:

- `--mode greedy`;
- `--batch-size`;
- `--prompt`;
- `--max-new-tokens`;
- `--context-length`, restricted to `4096`, `16384`, or `32768`;
- `--repeats`;
- `--warmups`;
- `--reference/--no-reference`.

The reference path can remain slow. It is for parity, not TPS.

### Required Benchmark Rows

Every run should record at least:

- model name and checkpoint;
- device name;
- target context length;
- prefill tokens;
- generated tokens;
- batch size;
- greedy or sampled;
- KV cache bytes;
- prefill ms;
- first-token ms;
- steady decode ms/token;
- steady decode TPS;
- total TPS.

Without these fields, the number is not useful.

## Current Research Ideas To Use

The research direction as of September 2026 is not "just make matmul faster."
For long-context inference, the useful ideas cluster around memory movement,
KV-cache layout, attention sparsity, batching, and phase separation. These are
the ideas worth translating into this Rust/Metal path.

## How To Use Research Without Fooling Ourselves

Every paper or system idea must pass through the same adoption process before it
becomes ENNX code. The aim is to extract the useful primitive, not copy an
architecture that was tuned for a different device, batch shape, quantization, or
serving workload.

### 1. Read For The Primitive, Not The Branding

For each paper/system, write down the actual primitive it contributes:

- a cache layout;
- a scheduler;
- a kernel decomposition;
- a sparse pattern;
- a quantization rule;
- a verification loop;
- a benchmark method.

Do not implement a named method until we can state the primitive in our own
terms and identify which bottleneck it attacks in our 4K/16K/32K profile.

### 2. Match Assumptions To ENNX

Before adopting an idea, record whether the original work assumes:

- NVIDIA CUDA, AMD ROCm, Apple Metal, or CPU;
- batch serving or single-stream decode;
- prefill, decode, or mixed scheduling;
- quantized weights, quantized KV, or BF16;
- dense attention or sparse attention;
- exact outputs or approximate outputs;
- one model, draft/target speculative models, or multi-model serving.

If the assumptions do not match ENNX, the idea may still be useful, but only as
a translated primitive.

### 3. Preserve A Reference Path

Every aggressive path needs a slower exact path beside it:

- BF16 dense KV before quantized KV;
- dense attention before sparse attention;
- ordinary greedy decode before speculative decode;
- simple contiguous cache before paged/ragged cache;
- current token parity tests before benchmark claims.

The optimized path can be experimental. The reference path must stay boring and
clear.

### 4. Add One Idea At A Time

Do not combine sparse attention, KV quantization, paged cache, and speculative
decode in one patch. That makes every performance result uninterpretable.

Patch rule:

- one idea;
- one benchmark before/after;
- one correctness comparison;
- one revert path.

### 5. Prefer Device-Resident Control

For this project, an inference idea is much more valuable if it keeps control on
the GPU:

- no per-token Python loop;
- no logits readback for greedy decode;
- no host-side cache page chasing;
- no CPU-side sampling unless we are explicitly benchmarking sampled mode;
- no per-token allocation.

If an idea needs a host decision each token, it is probably not the first version
we want.

### 6. Benchmark At The Target Contexts

Every research-derived optimization must report the same context sweep:

- 4K;
- 16K;
- 32K.

It must also state whether the reported number is:

- prefill-only;
- first-token after prefill;
- steady decode;
- total generation;
- single-stream;
- batch aggregate.

No result counts if it only improves a short prompt.

### 7. Reject Ideas Quickly When They Do Not Map

Good research can still be wrong for this repo. Reject or defer it if:

- it needs a second full weight copy;
- it saves compute but adds more memory traffic at 32K;
- it depends on CUDA-only primitives we cannot map cleanly to Metal;
- it improves batch serving but slows the single-stream baseline;
- it breaks deterministic greedy parity without a clear experimental reason;
- it hides time in Python or host synchronization.

### 8. Document The Decision

Each adopted research idea should leave a short implementation note:

- source idea;
- ENNX bottleneck it targets;
- exact files changed;
- correctness comparison;
- 4K/16K/32K benchmark before/after;
- whether it remains experimental or becomes the default.

This keeps the plan from becoming a pile of interesting names.

### 1. Treat Decode As Memory-Bandwidth Bound

Modern inference systems increasingly frame decode as a memory-bandwidth
problem: every new token streams model weights and scans a growing KV cache.
At 32K context, the KV read dominates more than it does at short context.

For ENNX this means:

- prioritize bytes moved per token;
- report KV bytes read per decoded token;
- optimize the decode attention kernel before chasing unrelated host overhead;
- avoid full-logit readback for greedy decode;
- keep all hot state resident on the GPU.

### 2. Page Or Tile The KV Cache Deliberately

Serving systems such as FlashInfer and vLLM-style engines use paged KV cache
metadata because long contexts and batches need predictable memory management.
For a single stream, contiguous cache is simpler and may be faster. For batch,
variable prompt lengths, or multiple BO candidates, a paged/ragged layout
becomes more important.

For ENNX:

- keep the first benchmark on the current contiguous cache;
- design the profile so it can compare contiguous versus paged later;
- use page sizes that match attention-kernel tile boundaries, likely 64 or 128;
- store cache metadata as compact device-side `u32`/`i32` arrays;
- avoid host-side page table work in the decode hot path.

### 3. Implement Long-Context Decode Attention Like FlashAttention Inference

The important inference attention pattern is not prompt-sized dense attention.
It is query length 1, very long K/V length, online softmax, and split/reduce
work across threadgroups when the cache is long.

For ENNX at 4K/16K/32K:

- split the K/V scan across threadgroups for long positions;
- reduce partial max/sum/value results deterministically;
- keep K and V layout friendly to sequential cache reads;
- benchmark `NHD`-like token-major versus `HND`-like head-major layout before
  committing to a paged format;
- avoid materializing attention scores.

### 4. Sparse Long-Context Attention Is A Real Candidate, But Not First

MInference-style work and newer native sparse-attention serving systems show
that long-context attention often has exploitable sparse structure. ECHO-style
systems also show that sparse attention changes the cache-management problem:
less attention compute can expose KV residency and prefetching as bottlenecks.

For ENNX:

- do dense exact attention first at 4K/16K/32K;
- add sparse attention only behind a correctness flag;
- start with simple local/global or heavy-hitter patterns before dynamic sparse
  pattern search;
- never use sparse attention for benchmark claims unless the generated tokens
  and objective behavior are separately reported.

### 5. KV Quantization Is Probably Necessary For 32K Headroom

KV-cache quantization papers now focus on dynamic and mixed precision: recent
tokens stay high precision, important long-range tokens stay higher fidelity,
and less important older cache entries drop to lower precision.

For Qwen2.5-Coder-1.5B, BF16 KV at 32K is large but feasible on high-memory
Apple Silicon. The optimization question is whether lower-precision KV improves
TPS enough without corrupting code generation.

For ENNX:

- benchmark BF16 KV first;
- add an optional int8 or fp8-like KV cache path only after BF16 is correct;
- preserve a recent-token BF16 window;
- measure token parity and top-k drift against BF16;
- never mix KV quantization into the first native TPS measurement.

### 6. Speculative Decoding Helps Generation, Not Teacher-Forced BO Loss

Speculative decoding can raise generated-token throughput by reducing target
model decode iterations. It does not help the teacher-forced loss path directly.
It is relevant if ENNX scores candidates by generated code quality rather than
only solution-token loss.

For ENNX:

- keep speculative decoding out of the first benchmark;
- consider a tiny draft model or n-gram/draft head later;
- report acceptance length and verification cost;
- do not let speculative speedups hide weak base decode performance.

### 7. Prefix Caching Matters For Repeated Evaluation

Apple's published inference architecture discusses static prefix caches for
shared prompt material. BO evaluation can also have repeated prompt prefixes:
same benchmark prompt, different weights or decoder settings.

For ENNX:

- prefix caching across different weights is not directly valid because weight
  perturbations change all hidden states and KV values;
- prefix caching is valid across decoder-parameter-only search with fixed
  weights;
- for full-weight BO, reuse should target tokenization, prompt packing, and
  resident buffers rather than hidden/KV cache reuse.

### 8. Continuous Batching Is Useful Only If We Batch Real Work

Continuous batching and mixed prefill/decode scheduling matter when serving many
requests. Our first target is one controlled long-context stream at 4K/16K/32K.
After that, batching matters for parallel BO candidates and multiple benchmark
tasks.

For ENNX:

- first make one stream honest and fast;
- then add packed/ragged batch prefill;
- then add batched decode for candidate groups;
- report single-stream TPS separately from batch aggregate TPS.

## Correctness Gates

No performance refactor is accepted unless these pass:

- greedy tokens match the current cached path for fixed prompts;
- cached generation matches `next_logits` reference for short prompts where the
  reference API is valid;
- absolute RoPE position does not reset across prefill chunks;
- EOS handling is unchanged;
- invalid/nonfinite logits still fail deterministically;
- no host logits readback in greedy native benchmark;
- no Python loop in the native TPS path;
- explicit pass/fail status at 4K, 16K, and 32K.

## Refactor Work

### Step 1: Native Generation Profile

Implement timing inside Rust around:

- prefill;
- first token selection after prefill;
- subsequent decode loop;
- total generation.

This is the smallest useful patch. It gives us a non-muddy TPS number.

Expected edits:

- add `QwenGenerationProfile` in `qwen_metal.rs`;
- store no global mutable profile unless needed;
- expose `bench_generate` through `py_qwen.rs`;
- add `Evaluator.bench_generate` in `ops/qwen/metal.py`;
- rewrite `ops/qwen/bench.py` to report native profile JSON.

### Step 2: Count Command Buffers

Add explicit command-buffer accounting in the generation path.

The desired steady-state greedy decode is:

- one command buffer per token, or fewer if later batching/fusion allows it;
- no extra output command buffer after decode;
- no CPU readback except the final token vector.

This should be reported in the benchmark profile.

### Step 3: Fix Greedy Decode Around The LM Head

For greedy decode, the LM head should produce only the selected token.

Current good direction:

- compute final norm;
- project to logits;
- reduce logits to argmax on GPU;
- read one `u32` token.

Next improvement:

- fuse LM projection and argmax so the full vocab logits buffer is optional for
  greedy decode;
- keep the full logits path only for diagnostics and sampling.

This avoids writing and scanning a large logits buffer as a separate phase.

### Step 4: Make Decode Projections Purpose-Built

The decode path needs specialized one-row kernels. Generic linear kernels are
unlikely to reach high TPS for one-token GEMV.

Work items:

- audit `qwen_gemv`;
- measure each projection's time with Metal counters;
- specialize common dimensions:
  - 1536 -> 1536;
  - 1536 -> 256;
  - 1536 -> 8960;
  - 8960 -> 1536;
  - 1536 -> 151936;
- consider fused QKV for decode, already present as `qwen_qkv_rows`;
- consider fused gate/up/SwiGLU for decode, already present as
  `qwen_mlp_rows`;
- ensure those fused kernels are actually used in decode and benchmarked.

### Step 5: Make Long-Context Attention A Real Kernel

At 16K and 32K, decode attention should not be a single naive scan if that
underutilizes the GPU. It needs a long-KV kernel shape.

Work items:

- split K/V reads across threadgroups for long sequences;
- reduce partial softmax statistics and value accumulators;
- benchmark head-major versus token-major KV layout;
- record bandwidth estimate per decoded token;
- keep an exact dense path as the correctness reference.

### Step 6: Reduce Decode Kernel Count

The first fusion targets:

- RMSNorm + QKV input staging when practical;
- QKV projection + bias;
- RoPE + cache write;
- gate/up + SiLU;
- final projection + argmax.

Do not fuse blindly. Each fusion must remove either:

- a command encoder;
- a large buffer write/read;
- repeated BF16 decode;
- or a CPU/GPU synchronization point.

### Step 7: Prefill Bulk Path

Prefill needs a separate optimization track:

- use row-batched kernels;
- avoid per-prompt serial prefill for batch generation;
- fix irregular row tails safely;
- keep causal attention online or block-tiled;
- avoid materialized `sequence * sequence` scores in the optimized path.

The BO teacher-forced loss path depends more on this than on one-token decode.

### Step 8: Loss Path For BO

The current BO loss path has three obvious wastes:

- rows are padded to the longest sequence;
- each example is forwarded serially;
- LM head projects many positions that are not scored.

Required changes:

- pass exact row lengths from the objective instead of padded rows;
- batch examples into one packed forward where possible;
- gather only scored hidden states before LM projection;
- compute cross entropy only for scored positions;
- avoid projecting prompt-only positions.

This is separate from generation TPS, but it is central for the sub-second BO
loop.

## Kernel-Level Checklist

For every Metal kernel in the inference path, record:

- input/output shapes;
- bytes read;
- bytes written;
- BF16 decode count;
- FP32 accumulation count;
- threadgroup size;
- grid size;
- whether it is memory-bound or compute-bound;
- whether it runs in decode, prefill, loss, or all three.

Priority kernels:

- `qwen_gemv`;
- `qwen_qkv_rows`;
- `qwen_mlp_rows`;
- `qwen_attn_dec`;
- `qwen_attn_row`;
- `qwen_simd_gemm`;
- final LM projection;
- argmax.

## What Not To Do

Do not optimize based on generated text alone.

Do not compare against "2000 tokens/sec" without matching:

- model size;
- quantization;
- batch size;
- prompt length;
- generated token count;
- greedy vs sampled;
- whether prefill is included;
- whether the number is decode-only or total TPS.

Do not move the token loop to Python.

Do not make a second FP16 weight copy unless a measurement proves it wins after
memory pressure is accounted for.

Do not add MPS, MLX, PyTorch, or another framework into the hot path until the
native Rust/Metal path has a clean profile.

## Milestones

### Milestone A: Honest TPS Number

Deliverables:

- native `bench_generate`;
- JSON benchmark report;
- prefill/first-token/decode split;
- greedy no-logit-readback assertion.

Acceptance:

- benchmark runs from `./ennx` or `python -m ops.qwen bench`;
- profile clearly says decode-only TPS and total TPS.

### Milestone B: Decode Hot Path Audit

Deliverables:

- per-kernel Metal trace;
- command-buffer count;
- top three slowest decode kernels;
- decision on LM-head+argmax fusion.

Acceptance:

- no more guessing where decode time goes.

### Milestone C: First Real Speed Patch

Deliverables:

- one focused decode optimization;
- before/after benchmark on the same prompt;
- parity preserved.

Best first candidates:

- final projection + argmax fusion;
- stronger decode GEMV;
- reduce QKV/MLP kernel count.

### Milestone D: BO Loss Path Patch

Deliverables:

- exact-length objective rows;
- scored-position-only LM projection;
- updated one-round BO timing breakdown.

Acceptance:

- lower forward/output time on the current MBPP minibatch;
- no change to objective semantics.

## Near-Term Patch Order

1. Add native generation profile.
2. Rewrite benchmark output around that profile.
3. Run greedy long-context benchmarks at 4K, 16K, and 32K.
4. Trace one decode run at 4K and one at 32K.
5. Patch long-context decode attention if it is the largest 32K bottleneck.
6. Patch final projection/argmax or decode GEMV if weight streaming dominates.
7. Separately patch BO loss exact-length/scored-token waste.

That order keeps the work measurable and avoids changing ten kernels before we
know which one is actually limiting TPS.

## Research References

- FlashInfer: paged KV layouts, ragged tensors, batch prefill/decode, and
  cascade inference.
- FlashAttention inference path: query-length-1 decode with KV cache update,
  rotary embedding, paged KV support, and split work for fast cache reads.
- MInference: dynamic sparse attention patterns for long-context prefill.
- ECHO: KV offload/prefetch design for native sparse-attention long-context
  serving.
- KV Pareto and KVC-Q: system-level KV/cache precision tradeoffs for
  long-context inference.
- Apple PCC/MetalLM public architecture notes: Metal-based inference,
  distributed inference, and static prefix caching.
