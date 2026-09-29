# CUDA-Oxide

ENNX CUDA kernels written in Rust with CUDA-Oxide. The current target is
NVIDIA T4 (`sm_75`). CUDA is built separately from the CPU, Metal, and OpenCL code.

The target is a subsecond BO pre-training round with a million-token context
window, including generation, scoring, evaluation, and the optimizer update.
Prompt occupancy and output length are independent workload counts; a million
new tokens per second is not required. The packed executor supports at most
64K. The streamed executor allocates bounded activations and distinct KV for
each recurrent visit. Capacity support alone does not establish latency or
learned long-context capability.

The `bo-generation-t4-v1` measurement is invalid as a BO/pre-training result:
it interpreted FP16 checkpoint bits as BF16, perturbed only the readout, and
used a throughput-only reward. Its command has been removed. The historical JSON
is retained alongside its invalidation notice under `results/cuda/`.

PTX synthesis can be selected for the checkpoint executor's vocabulary readout:

```sh
ENNX_SYNTH_GEMM=32 ./ennx cuda generate-t4 --unroll 1 --out results/cuda/fresh-synth-run
```

The run checks boundary tiles against independent CPU FP16 arithmetic and
compares staging depths 8, 16, 32, and 64 with CUDA-Oxide on the same T4.
`gemm.json` records kernel comparisons; `result.json` and `tokens.json` record
generation. Default execution retains CUDA-Oxide until complete-generation
parity and timing justify a different schedule. The zero-output hierarchical
PTX stub has been removed.

T4 comparison on 2026-10-05: the replacement synthesized GEMM matched independent
CPU edge cases and CUDA-Oxide exactly on all four projection shapes. Selecting
it for readout produced identical 3,968-token outputs and accepted frontiers in
four alternating full-generation comparisons. CUDA-Oxide took 4.809–4.861 s;
synthesized readout took 4.766–4.838 s, a paired improvement of 0.4–0.9%.
Both used a 128-token zero-ID prompt and 21 repair waves at 4K. These are
execution measurements, not pre-training or text-quality results. Artifacts:
`results/cuda/generation-t4-synth-readout/{gemm,result,tokens}.json`.

The vocabulary readout now reuses an 8 MiB buffer in 512-row chunks. The former
full-context allocation would consume 16 GiB at 1,048,576 rows. On the T4,
chunking preserved all 3,968 generated tokens and 21 accepted frontiers; paired
generation times were 4.725/4.745 s versus 4.751/4.761 s with full logits.
Artifacts: `results/cuda/generation-t4-bounded-readout/`. This removes one memory
obstacle. Streamed execution now retains the accepted prefix within each
generation invocation and rebuilds KV at the start of the next invocation.

Measured on 2026-10-05: at 4K, 512-row streaming preserved all generated tokens
and accepted frontiers while reducing evaluated positions from 86,016 to
49,152 and device time from 4.122 s to 3.129 s. At 64K with a 64,512-token
zero-ID fixture prompt and 1,024 new tokens, streaming matched the packed
reference exactly: 69,632 versus 327,680 evaluated positions and 3.446 versus
15.001 s. Artifacts: `results/cuda/generation-t4-streamed-v1/` and
`results/cuda/generation-t4-streamed-64k-v1/`. These first measurements exclude
workspace initialization. Subsequent streamed wall timings include it;
checkpoint loading remains outside generation timing.

```sh
./ennx cuda generate-t4 --context 65536 --prompt 64512 --chunk 512 --unroll 1 --out results/cuda/fresh-streamed-run
```

The benchmark compares streamed tokens and frontiers to packed execution through
64K. Zero and varied token-ID prompts are numerical fixtures, not a language
quality evaluation or a complete BO round.

The combined implementation also passed on the first 64,512 positions of the
checkpoint's actual pre-training corpus, followed by 1,024 newly generated
tokens. At 64K on T4, packed execution took 45.215 s and streaming took 4.415 s
device / 4.580 s wall including workspace initialization. Both produced exactly
the same tokens and 15 accepted frontiers. Evaluated positions fell from
983,040 to 78,848. This is one paired comparison, not a latency distribution.
Artifacts and decoded output: `results/cuda/generation-t4-corpus-64k/`;
source/data hashes and the exact prompt: the adjacent `-source/` directory.
The completion remains incoherent. This is execution progress, not a trained
long-context capability or a subsecond complete BO round.

The complete causal generator also executed at 1,048,576 context positions on
T4: 1,047,552 actual corpus tokens followed by 1,024 new tokens. With 512-row
chunks, it took 59.218 s device / 60.484 s wall, including workspace setup;
checkpoint load plus generation took 65.127 s. The first forward wave consumed
58.401 s; the remaining verification work consumed 0.817 s. Fourteen waves
evaluated 1,061,376 positions. A 1,024-row comparison took 54.885 s device and
produced identical tokens and accepted frontiers. This is chunk invariance,
not an independent packed reference at 1M. The completion remains incoherent.
Artifacts: `results/cuda/generation-t4-corpus-1m/`, with prompt provenance,
source hashes, command and log in the adjacent `-source/` directory. Fresh
context processing dominates this workload; subsecond BO remains unachieved.

The 4,096-row follow-up reduced device time to 52.025 s versus 60.956 s for
its paired 512-row reference (14.7% lower). All tokens and frontiers matched
both the paired reference and the saved baseline. Reusing constructor scratch
removed duplicate allocation and RoPE construction: wall time was 52.042 s,
with 49.367 s in the first wave and 2.659 s afterward. Larger repair chunks
increased evaluated positions to 1,101,824, so prefill and repair chunk sizes
should be tuned separately. Sampled middle-chunk time was 34.4% selected
attention, 33.5% MoE, 16.6% mHC, and 6.9% indexing. These percentages describe
one sampled chunk, not a whole-context trace. Artifacts:
`results/cuda/generation-t4-corpus-1m-chunk4096/`. Profiling was enabled in both
arms; this is a single paired comparison, not a latency distribution.

Prefill and repair chunk sizes can now differ. Set
`ENNX_CUDA_REPAIR_CHUNK=512` with `--chunk 4096` to retain large first-wave
batches and use smaller suffix batches. The effective repair size is capped
at the allocated chunk size. The default retains the same size for both.
This passed exact packed-token and frontier comparisons at 4K and 64K, and
streamed chunk invariance plus saved-baseline token equality at 1M.
At 1M the 4096/512 schedule took 51.220 s device / 51.338 s wall versus a
paired 512/512 reference's 61.679 s device. It evaluated 1,061,376 positions;
the first wave took 50.369 s and subsequent verification 0.850 s. This remains
a generation-only result. Artifact: `results/cuda/generation-t4-schedule-1m/`.

Set `ENNX_CUDA_PROMPT` to a JSON token-ID array when using `generate-t4` to
replace the zero-ID fixture. Its length must equal `--prompt`, and IDs must use
the checkpoint's 8,192-token vocabulary. The Rust runner uploads the array and
records `prompt_source=token_file`; it does not repeat or pad the input.
Set `ENNX_CUDA_PROFILE=1` to record stage timings for the first, middle and last
chunks of the first streamed wave. These are sampled chunk profiles, not sums
over the whole context. Profiling is disabled by default.

Inspect current coverage on any development host:

```sh
./ennx cuda inspect
```

On a machine with the CUDA-Oxide toolchain installed, use the repository CLI:

```sh
./ennx cuda setup
./ennx cuda build
./ennx cuda parity
./ennx cuda resident
./ennx cuda bench
./ennx cuda sanitize
```

`setup` installs the pinned Linux x86-64 CUDA-Oxide toolchain on a fresh root
host with CUDA 13.0 and an NVIDIA runtime. Repository CUDA builds and runs use
the remaining `./ennx cuda` commands.

The CUDA-Oxide revision is pinned in `Cargo.toml`; Rust and LLVM versions are in
`cuda/rust-toolchain.toml` and `tools/cuda-setup`.

T4 verification on 2026-10-03: 36 parity cases passed, resident search passed,
Compute Sanitizer reported zero errors, and the 16,777,216-element perturbation
benchmark ran in 0.289 ms at 54.1 GiB/s. This measures the resident search
kernel, not the complete 64K generation and scoring loop.

The recurrent checkpoint executor and accepted-prefix generator use the same
CLI on the NVIDIA host:

```sh
./ennx cuda model-check
./ennx cuda model checkpoint.safetensors tokens.json fresh-output
./ennx cuda generate checkpoint.safetensors prompt.json fresh-generation
```

JSON inputs contain vocabulary IDs in an array. `model` predicts one token per
input row; `generate` retains the prompt and generates the remaining positions.
Both accept context, recurrent visits, temperature, and seed after the output
directory. `generate` also accepts repair unroll (default 2). Packed context
must be a power of two from 4,096 to 65,536; inference visits range from one to four.
Output directories must be new. Artifacts include token IDs and measured timing;
`model` also saves normalized FP16 hidden states for numerical comparison.
`generate` accepts a final chunk size (zero selects packed execution); streamed
chunks are powers of two from 64 through 4,096. Streamed context capacity is
bounded at 1,048,576 positions. It includes the prompt and generated output.

The checkpoint format is `ennx.fbt-pisa1-looped-mhc4-rope.v1`. The implementation
uses all 625 routed experts plus the shared expert. Repairs remain on-device
between passes in one submission batch; the host reads the accepted frontier
between batches. These commands do not yet run the Bayesian optimization loop.

T4 measurements on 2026-10-04: the seven-visit 4K analytical full-model fixture
used 496.120 ms of device time. The routed-attention slice's median was
50.847 ms device / 51.876 ms wall. The 3,968-token sampled fixture matched for
repair unroll 1 and 2. These measurements do not establish a 200 ms complete
round, 64K throughput, or checkpoint agreement with Metal.

Persistent PISA attention supports contexts through 1,048,576 tokens with a
separate KV cache and bounded query scratch:

```sh
./ennx cuda context 1048576 128 7
./ennx cuda sanitize context 1048576 128 3
```

Arguments are context capacity, query count, and timing repetitions. Query
count must be a multiple of four through 4,096; the numerical probe requires
at most half the context. The JSON report distinguishes cached attention from
complete model execution. It verifies prefix append, causal selection,
selected-support attention, and incremental tree repair against the Rust
reference shared with Metal. FP16 KV consumes 256 MiB per layer visit at 1M,
with approximately 4 MiB of tree summaries. The complete checkpoint executor
supports bounded streamed activations. Packed comparisons reach 64K; complete
1M generation has passed chunk-size invariance as measured above. See
[the context design and research](../docs/kernel-architecture-plan.md#million-token-context).

## Learned block diffusion

The new source accepts the distinct checkpoint format
`ennx.fbt-pisa1-diffusion-mhc4-rope.v1`, including learned mask and index weights:

```sh
./ennx cuda diffusion-check
./ennx cuda diffusion CHECKPOINT PROMPT_JSON FRESH_OUTPUT 4096 2 0.8 17 2 128 true
./ennx cuda sanitize diffusion-check
```

Arguments after the output directory are total context positions, recurrent
visits, temperature, seed, denoising steps, visibility block, and soft-input mode.
The context includes the prompt: 4,096 positions with a 128-token prompt produce
3,968 new tokens. The packed executor requires a power-of-two context from 4K
through 64K, aligned prompt/output blocks, and one to four recurrent visits.

Diffusion replaces exact autoregressive repairs with parallel denoising passes.
The kernels implement learned-mask input, confidence-weighted token interpolation,
block visibility, learned query projection, and sampled output confidence.
Selection is independent per query; Metal's grouped refinement and drift reuse
are not implemented here. The fixture compares sampled output with an analytical
uniform-weight model. It does not establish trained-checkpoint agreement.

The CUDA diffusion fixture passed on T4 on 2026-10-05. A trained diffusion
checkpoint and integration as a target-verified drafter remain outstanding.
Complete CUDA BO rounds still require integration.
