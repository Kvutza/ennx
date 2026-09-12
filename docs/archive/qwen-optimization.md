> Archived on 2026-09-29. Historical evidence and superseded plans, not current
> instructions. Read [current state](../handoff.md) and [runbook](../turbo-enn.md).
> Original dated results are retained; relative documentation links were relocated.

# Qwen Inference Optimization

## Decision

ENNX should have two explicit execution paths:

1. **Prefill** processes a prompt and populates the per-layer KV cache.
2. **Decode** consumes one token, appends one K/V pair per layer, and returns the next-token decision.

The Python API should remain thin. Rust owns the generation state, cache, token loop, EOS handling, and greedy selection. Metal owns tensor work and reductions. This keeps Python out of the per-token critical path without making the public API GPU-specific.

The original implementation was a correctness baseline, not the final inference engine: `generate` called `next_logits` for every token, and `next_logits` reran the full prefix. The short-input reference path still materializes repeated GQA keys/values and a full attention score matrix; the cached generation path now avoids both.

The cache increment, bounded prefill, and native GPU argmax are now implemented: generation prefills a Rust-owned BF16 cache in 256-token chunks, uses a one-token decode path with online attention, and writes the selected token on Metal. The full-prefix path remains available through `next_logits` as the short-input reference oracle. GPU parity still needs to be run on a Metal-equipped Mac before treating the optimized path as experiment-grade.

## What the research supports

### 1. KV cache before micro-optimizations

Autoregressive attention reuses all previous keys and values. Recomputing them for every generated token is the dominant avoidable cost. FlashAttention-2 shows that attention performance depends heavily on work partitioning, memory traffic, and avoiding unnecessary intermediate movement, not just arithmetic throughput ([FlashAttention-2](https://arxiv.org/abs/2307.08691)).

For ENNX's current single-sequence workload, use a preallocated contiguous cache first:

- layout: `[layer, kv_head, position, head_dim]` for K and V;
- storage: BF16, with FP32 accumulation in attention;
- capacity: rounded up in fixed chunks, initially 256 tokens;
- ownership: one Rust `GenerationState` owns the cache and current position;
- no per-token allocation or buffer replacement.

PagedAttention is valuable when many requests or candidates share a device and sequences grow independently. Its central result is avoiding waste from large contiguous reservations ([PagedAttention](https://arxiv.org/abs/2309.06180)). It is not the first choice for one generation stream: a contiguous cache has simpler addressing and fewer indirections. We can add pages when ENNX batches parallel BO candidates or serves multiple sequences.

For Qwen2.5-Coder-1.5B, the configured 2 KV heads and 128-wide heads make the BF16 K/V cache about 1 KiB per token per layer, or about 896 MiB at 32K context across 28 layers. This is large enough that cache dtype and capacity policy are correctness constraints, not cosmetic details.

### 2. Decode attention must be a different kernel

The decode kernel should:

1. compute the one-token Q/K/V projections;
2. apply RoPE using the absolute position;
3. write the new K/V pair into the cache;
4. map each query head directly to its KV head, without `qwen_repeat_kv`;
5. run online softmax over cached K/V and write one attended vector.

It must never allocate a `heads * sequence * sequence` score matrix for decode. The existing `qwen_rope` also derives position from the local row index (`rust/crates/ennx/src/qwen.metal:63-83`), so the cache path must pass an absolute `position`; otherwise a continuation silently applies the wrong rotary phase.

Prefill uses the same resident cache and online attention kernel in bounded 256-token chunks. Each chunk attends to all earlier cached positions plus its causal prefix, so the implementation preserves causal semantics without materializing a prompt-sized score matrix. The non-cached reference path remains intentionally bounded at 256 tokens.

### 3. Reduce command-buffer and readback overhead

Apple recommends submitting the fewest command buffers that still keep the GPU utilized; excessive submissions can stall the CPU or starve the GPU ([Metal command buffers best practices](https://developer.apple.com/library/archive/documentation/3DDrawing/Conceptual/MTLBestPracticesGuide/CommandBuffers.html)). The current generated-token path commits one full forward buffer and one output buffer per token (`qwen_metal.rs:973-1233`, `qwen_metal.rs:1268-1318`).

The decode path should encode the complete one-token pipeline into one command buffer, then synchronize once. Resource dependencies must remain explicit because a cache write is consumed by a later attention read; Apple documents these read/write hazards and synchronization requirements ([Metal resource synchronization](https://developer.apple.com/documentation/metal/resource-synchronization)).

Use `dispatchThreads` where the minimum supported Apple GPU family permits it, or retain explicit aligned threadgroups with bounds checks. Apple documents that aligned grids can exceed the data domain and that nonuniform threadgroups can simplify edge handling on supported devices ([threadgroup and grid sizing](https://developer.apple.com/documentation/metal/calculating-threadgroup-and-grid-sizes)).

### 3.1 Couple candidate materialization to evaluation

The BO path now exposes an explicit `ask_generate` operation for greedy
candidate scoring. Search selection and BF16 candidate materialization are
committed first on the shared Metal queue; Qwen generation is then submitted to
that same queue before the search command is waited on. Metal queue ordering
provides the dependency without returning a mutable candidate buffer to Python
between stages. The proposal is published only after generation completes, and
an evaluator failure poisons the search state after waiting for the in-flight
command.

This removes the host-side proposal wait from the critical path while preserving
the exact dense BF16 candidate and generated-token objective. The optimized
operation is intentionally limited to greedy decoding for now; sampled decoding
continues through the ordinary explicit proposal path until it has a matching
seed and lifecycle contract. Telemetry records the combined stage as
`candidate_pipeline_seconds` rather than mislabeling it as proposal time.

### 4. Keep selection on the GPU

For greedy generation, the output projection should end in a deterministic GPU reduction:

- each threadgroup scans a vocabulary tile;
- partial results store `(value, token_id)`;
- ties choose the lower token ID;
- a final reduction writes one `u32` token ID.

This avoids transferring roughly 608 KiB of FP32 logits to the CPU for every token. Keep the full-logit path for `logits`, scoring, debugging, and sampling. Apple provides tuned matrix and top-K primitives ([MPS overview](https://developer.apple.com/documentation/metalperformanceshaders), [MPS matrix-vector multiplication](https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixvectormultiplication), [MPS top-K](https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixfindtopk)), but MPS top-K should not replace the exact greedy reduction until tie behavior and end-to-end latency are measured.

### 5. Use a native tiled projection kernel

The projection path now has three shape-selected kernels in
`rust/crates/ennx/src/qwen_linear.metal`:

1. `qwen_gemv` handles decode and other small row counts;
2. `qwen_simd_gemm` handles aligned prefill tiles, staging BF16 weights
   as FP32 and accumulating with Metal's 8x8 SIMD-group matrix operations;
3. `flame_linear` remains the bounds-safe fallback for irregular shapes.

The SIMD path covers a 64x32 output tile with four SIMD groups and is selected
only when `rows >= 64`, `inside % 8 == 0`, and `outside % 32 == 0`. This keeps
partial tiles away from cooperative matrix stores. The direct native test
`simdgroup_linear_matches_bf16_reference` compares a complete tile against a
CPU BF16 reference before the kernel is used by the wheel.

The model keeps one resident BF16 weight buffer. A temporary FP16 mirror and
an unverified Objective-C MPS bridge were deliberately rejected: they add
working-set pressure and create a second numerical/ownership contract. MPS
remains a future benchmark candidate, not an unconditional dependency.

RoPE is another clear kernel issue: the current shader evaluates `pow`, `cos`, and `sin` per element. Build a reusable rotary table once per evaluator, or compute one sine/cosine pair per `(position, rotary_pair)` and share it across the pair. Benchmark both; do not make every decode element recompute the same phase.

## Implementation sequence

### Phase 0: lock correctness

- Add prefill-vs-decode parity tests at one token, several tokens, and a continuation.
- Compare final hidden state, logits, and selected token; check finite/nonzero invariants.
- Add an absolute-position RoPE test that would fail if position resets to zero.
- Keep the existing full-forward implementation as the reference path behind an explicit test helper.

### Phase 1: resident decode state

- Add `GenerationState` and a preallocated BF16 K/V cache in Rust.
- Add cache capacity checks and a hard context limit before any Metal dispatch.
- Add one-token workspace sized independently from prefill workspace.
- Preserve `logits`, `next_logits`, and `losses` semantics.

### Phase 2: one-token Metal pipeline

- Add fused decode RoPE/cache-write/online-attention kernels.
- Avoid K/V repetition by computing the GQA mapping inside attention.
- Encode all decode stages into one command buffer.
- Initially use the existing projection kernel so numerical changes are isolated to cache attention.

### Phase 3: GPU token selection

- Implemented deterministic Metal argmax for generation only; full-logit APIs are unchanged.
- Added lowest-token-ID tie handling and nonfinite-logit rejection.
- Verify it against CPU argmax for random logits and real model output on a Metal-equipped Mac.
- Keep the CPU/full-logit path available for diagnostics and non-greedy callers.

### Phase 4: measured GEMM work

- Implemented the shape-selected native SIMD-group projection kernel.
- Record GPU time, command-buffer count, cache bytes, output bandwidth, and end-to-end tokens/sec.
- Keep the direct tile reference test as a release gate; do not infer correctness from generated text alone.
- Use Metal System Trace and the Metal debugger for GPU/CPU overlap, counters, memory, and per-kernel timelines ([Apple Metal developer tools](https://developer.apple.com/metal/tools/), [Metal debugger](https://developer.apple.com/documentation/xcode/metal-debugger)).

### Phase 5: long-context and batching

- Implemented bounded chunked prefill with causal online attention and absolute RoPE positions.
- Add paged cache only when batching or parallel BO candidates demonstrates a need.
- Add batched candidate evaluation without changing single-sequence semantics.

## Non-negotiable performance and correctness gates

The optimized path is not accepted until it passes all of these:

- prefill and cached decode agree within an explicitly recorded tolerance;
- greedy token IDs match the reference, including deterministic ties;
- no NaN, infinity, stale-cache, or position-reset failures;
- one command buffer per decoded token in the steady state;
- no full-prefix recomputation after prefill;
- no `sequence * sequence` decode allocation;
- benchmark results are recorded on the target Mac rather than inferred from shader appearance.

## Bottom line

The implementation is now a contiguous BF16 KV cache, bounded online-attention prefill, one-token online-attention decode, native GPU token selection, and a verified SIMD-group GEMM path for aligned prefill projections. The irregular-shape fallback remains explicit, and decode still uses GEMV. Do not add a second weight representation or framework bridge until a measured device-specific benchmark justifies its memory and numerical trade-offs.

## Sources

- Tri Dao et al., [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691).
- Woosuk Kwon et al., [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180).
- Apple, [Metal Performance Shaders](https://developer.apple.com/documentation/metalperformanceshaders).
- Apple, [MPSMatrixVectorMultiplication](https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixvectormultiplication).
- Apple, [MPSMatrixFindTopK](https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixfindtopk).
- Apple, [Resource synchronization](https://developer.apple.com/documentation/metal/resource-synchronization).
- Apple, [Metal developer tools](https://developer.apple.com/metal/tools/).
- Apple, [Metal Performance Primitives Programming Guide](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf).
