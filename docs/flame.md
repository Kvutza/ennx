# FLAME full-weight BO

This includes checkpoint/forward validation and an experimental full-weight CUDA
BO driver. ENNX's surrogate is unchanged. The driver defaults to dense correlated
Gaussian noise and acquisition-selected step sizes. Paired minibatch runs also
contract the radius after repeated screened deteriorations; inconclusive
comparisons leave the contraction counter unchanged. Independent Gaussian noise
with the existing TuRBO controller is the comparison baseline; legacy independent
signs remain explicitly available. Both Gaussian modes have completed full-weight
T4 smoke runs; the results and remaining memory/performance limits are below.

The public [FLAME-MoE checkpoint](https://huggingface.co/CMU-FLAME/FLAME-MoE-290M-1.3B)
needs no Hugging Face token or license-acceptance login. We pin revision
`04bded20e1eafa97c5c84cee9522798a2e83606f`, snapshot `iter_0005473`.
Its 1,300,024,320 parameters occupy approximately 2.60 GB in BF16.
The converter downloads only model-tensor byte ranges, not optimizer state or
the repository's other training snapshots. Interrupted output is intentionally
not loadable; use a fresh output directory when retrying.

## Architecture

The saved `common.pt` arguments and tensor metadata take precedence over the
model card: **top-6 routing, not top-8**. There are nine blocks, width 1024,
16 attention heads, a dense first MLP, then 64 routed experts plus a shared MLP.
Embeddings and readout are untied. Attention uses head-grouped QKV and
non-interleaved RoPE. Routing applies softmax over all experts before selecting
six, without renormalizing their probabilities. The shared branch is ungated.
Exact routing ties select lower expert IDs in both implementations. This is an
explicit reproducibility convention: Megatron's PyTorch `topk` does not promise
that ordering at ties, so equivalence to that runtime is not claimed there.

The implementation follows the [pinned Megatron source](https://github.com/yuzc19/Megatron-LM/tree/cbaf684c5d03997e0fdd5347c5e2d371c381a3d8)
used by [FLAME](https://github.com/cmu-flame/FLAME-MoE/tree/e9b2fe2df3f1abb8dbb9ec0eabde8cdfb65e5c78).
Weights remain BF16; the forward calculation deliberately uses FP32 for this
correctness milestone. This is not a BF16-throughput claim.

## Run

Use Python 3.12 or 3.13 and an isolated environment. The Colab T4 package pins
are in `ops/flame/requirements-cuda.txt`. CUDA 12 is deliberate for T4.
Run from the repository root:

```sh
python -m pip install -r ops/flame/requirements-cuda.txt
python -m pytest tests/testflame.py -q
python -m ops.flame.checkpoint --output .cache/flame-checkpoint
python -m ops.flame.parity .cache/flame-checkpoint --tokens ops/flame/tokens.json --output .cache/flame-parity.json
```

The parity command requires a T4 by default; `--allow-other-device` explicitly
permits CPU or another accelerator. The supplied token IDs are a deterministic
numerical fixture, **not a language-quality benchmark**. Provide your own JSON
array of equal-length, unpadded token sequences to check other inputs. Loss is
mean next-token negative log likelihood over the supplied sequence, without
padding masks. The reference stays on CPU; allow at least 8 GB of free host RAM
and roughly 3 GB of disk for converted weights.

`examples/colabflame.ipynb` runs the same commands. Until this code is published,
upload a local source archive to the notebook; no remote commit is assumed:

```sh
zip -r .cache/flame-source.zip ops/flame tests/testflame.py -x '*/__pycache__/*'
```

## What Parity Means

The comparator is a separate, unfused PyTorch implementation of the pinned
equations, not execution of the released Megatron/TransformerEngine runtime.
It checks logits, next-token loss, greedy token IDs, block outputs, attention,
router logits, routing weights, and selected experts. Numerical tolerances are
explicit in `ops/flame/parity.py`; greedy tokens and expert IDs must match
exactly. A disagreement fails the command rather than emitting a passing report.
Reports record the device, package versions, source-file hashes, and available
JAX allocator statistics. The notebook also saves `pip freeze`; top-level
package pins are not a complete lock of the Colab image or transitive dependencies.
Agreement between these two implementations cannot exclude a shared mistake in
interpreting the original model. Original-runtime parity remains outstanding.

Tests also cover dense perturbations of every parameter in a small architecture,
BF16 storage, causality, non-renormalized routing, checkpoint chunk coverage and
overlap, restricted metadata deserialization, and HTTP range enforcement.

## Validated Runs

On 2026-09-12, all 28 tests passed on both local CPU and Colab Tesla T4.
The full 1.3B checkpoint also passed both numerical fixtures on each device:

| Device | Batch / sequence length | Maximum logit difference | JAX peak live allocation |
| --- | --- | --- | --- |
| CPU | 1 / 8 | 0.0000186 | Not reported |
| CPU | 2 / 32 | 0.0000291 | Not reported |
| Tesla T4 | 1 / 8 | 0.0000107 | 2.93 GiB |
| Tesla T4 | 2 / 32 | 0.0000210 | 2.95 GiB |

Greedy token IDs and routed expert IDs matched exactly in all four runs.
The T4 allocator pool peaked at approximately 4.00 GiB; live allocation is not
total process VRAM. These small numerical fixtures do not establish long-context
throughput, language quality, or BO performance. CPU runs used Python 3.12.13,
JAX 0.6.2, and PyTorch 2.7.1; T4 runs used Python 3.13.15 and the CUDA pins above.

## Inspect

`model.forward(..., capture=True)` returns named per-block traces.
`model.parameter_tree(params)` exposes all weight tensors as a nested dictionary
without copying them, suitable for `treescope.display(...)` in a notebook.
No parameters are hidden behind an adapter or coefficient basis.

The JAX MoE forward evaluates gathered experts one token at a time to
bound temporary weight storage. It is a readable validation implementation,
not an optimized grouped-GEMM engine. No KV cache, padding, training backward
pass, or Metal backend is added by that reference implementation. The separate
native Metal implementation is described below.

## Native Forward

The native CUDA C++/cuBLAS forward remains experimental and opt-in. The combined
Python extension built successfully with CUDA 13.0 NVCC for `sm_75` and passed
public API parity on a Tesla T4. JAX remains the CLI default and the reference
implementation; select native explicitly with `--backend native`. The historical
BO results below use the JAX forward.

Public API parity passed 16 toy cases with maximum absolute logit difference
`1.25169754e-6`. Checks covered BO weight leases, invalid-export rejection and
recovery, producer-stream synchronization, large cross-entropy inputs, and router
recovery. On the full 1,300,024,320-parameter checkpoint, native and JAX losses
matched on two problems with maximum absolute difference `1.11e-6`. A separate
native Compute Sanitizer memcheck run reported zero errors. These numerical
fixtures do not establish optimization improvement or throughput.

Build the Python extension with the CUDA 13.0 toolkit/NVCC and the opt-in Cargo
feature `--features native-flame`. Install `ops/flame/requirements-native.txt`,
which pins CUDA 13 cuBLAS/runtime libraries alongside Torch's CUDA 12 input-upload
dependencies. The source-built extension and its runtime library loading were
verified on T4; standalone wheel installation has not been verified.

```sh
python -m pip install -r ops/flame/requirements-native.txt
CUDA_HOME=/usr/local/cuda-13.0 \
cargo +nightly-2026-09-29 oxide build --arch sm_75 \
  --device-codegen-crate ennx_cuda_kernels -- \
  -p ennx-py --locked --release --features native-flame
# Install the resulting extension into this checkout for the experiment:
cp target/release/libennx_rust.so src/ennx/ennx_rust.so
export PYTHONPATH="$PWD/src:$PWD"
python -m ops.flame.bo .cache/flame-checkpoint --backend native \
  --tokens .cache/flame-coding/train.json --output .cache/flame-native-bo \
  --evaluations 8 --candidates 4 --history 2
```

Native execution uses FP32 compute and BF16 weight storage, with no JAX imports
or Torch neural computation. It currently requires a prepared solution corpus,
correlated sampling, and paired minibatches. Omitted `--minibatch-size` becomes
2 only for native; JAX retains the fixed-objective default. Both backends share
the BO loop, paired acceptance statistics, and single final checkpoint write.

Canonical tensor names, shapes, offsets, and keys share the same definition,
but numerical layout metadata is not bit-identical between backends. Native
checkpoint preparation uses a scaled FP32 RMS calculation on CPU; JAX uses its
FP32 square/reduction path. FP32 rounding differences in RMS scales also affect
the derived metric weights. The pilot comparison matches settings
and objective work, not bitwise-identical search trajectories or model weights.

The existing CUDA-Oxide search kernels remain unchanged by the forward backend.
Custom CUDA kernels implement normalization, RoPE, routing, and masked loss;
cuBLAS implements matrix multiplication. Expert dispatch groups tokens without
dropping routes, but currently synchronizes expert counts to the host once per
MoE layer. Pilot timings below include this synchronization. The public API is
`ennx.experimental.FlameEvaluator(config, max_tokens)`, with synchronous
`losses(weights, tokens, masks)` and diagnostic `logits(weights, tokens)` methods.
Weights are borrowed as contiguous CUDA-device-0 BF16 DLPack tensors.

After building, run the parity harness in an environment that also has the pinned
JAX reference dependencies:

```sh
python -m ops.flame.native_parity --output .cache/flame-native-parity.json \
  --checkpoint .cache/flame-checkpoint --tokens .cache/flame-coding/train.json
```

### Native and JAX Pilots

The native T4 worker completed a full-checkpoint pilot on the 312-problem TRAIN
corpus: two problems per minibatch, four acquisition candidates, history two,
radius 0.01, all seeds zero, and acceptance threshold two standard errors.
Seven proposal steps plus the baseline used eight weight evaluations,
15 minibatch objective evaluations, and 30 problem forwards. The matched JAX
worker completed the same declared settings and objective work. Both runs
accepted one proposal at the seventh and final step and wrote one final
checkpoint each.

| Measurement | Native T4 | JAX T4 |
| --- | --- | --- |
| Ask, seven steps | 6.033454288 s | 6.289297226 s |
| Incumbent and candidate forwards, seven steps | 2.579431948 s | 8.261470652 s |
| Tell, seven steps | 1.102469627 s | 1.173597193 s |
| BO core, sum of the above | 9.715355863 s | 15.724365071 s |
| End-to-end worker | 70.454666912 s | 105.159829055 s |
| Peak sampled GPU memory | 12,787 MiB | 14,777 MiB |
| Minimum sampled free GPU memory | 2,126 MiB | 136 MiB |

Native forward time comprised 1.346294754 seconds for incumbents and
1.233137194 seconds for candidates.

End-to-end time includes imports, loading, baseline evaluation, and the final
save; it excludes the separate reload check. GPU memory was sampled every
100 ms and measures total device usage, not just this process; transient peaks
may be missed. The JAX/native elapsed-time ratios were `1.618506341` for BO core
and `3.202825590` for incumbent-plus-candidate forwards. These are observations
from one matched pair of runs, not evidence of training quality or general
throughput.

Selected candidate indices, radii, acceptance decisions, and minibatches matched
at all seven steps. Numerical scales and metric weights differed in 60 of 81
blocks, with maximum relative differences `1.3718597273428792e-6` and
`2.7068385569302287e-6`, respectively.
Final checkpoints were not byte-identical: 39 of 81 saved tensor files had equal
SHA256 hashes, while 42 differed. No bitwise backend or weight identity is claimed.
The maximum per-problem loss difference across the paired runs was
`4.89790415025837e-5`. This includes effects from perturbation rounding and is not
a forward-only error measure at fixed weights.

A fresh native process on T4 verified all 81 saved tensor checksums and reproduced the
original-checkpoint baseline and the saved checkpoint's final accepted minibatch
losses with zero observed difference, without importing JAX. Reload took
48.69 seconds, excluded from the pilot timing above.

The baseline reward was `-1.9311107993125916`; the final incumbent measurement
was `-2.208064556121826` on a different minibatch. These scores must not be
compared as improvement. Acceptance used paired losses on the same problems
within each step, with no validation set or confirmation batch.

The accepted step used zero-based problem indices 294 and 183:

| Problem index | Incumbent loss | Candidate loss |
| --- | --- | --- |
| 294 | 2.368304491043091 | 2.3660590648651123 |
| 183 | 2.0509634017944336 | 2.050070285797119 |

The paired mean improvement, `0.0015692710876464844`, exceeded the two-standard-error
threshold, `0.0013479688847717326`. This is a measurement on those two problems,
not an improvement measured over the whole corpus.

Run logs, parity and comparison reports, fresh-process versions, and
`verified-source.zip` are archived under `.cache/flame-native-t4/results/`.
The detailed comparison is `ennx_native_comparison.json`; the authoritative
loaded-package record is `ennx_native_process_versions.json`. Fresh-process
imports reported Python 3.13.15, NumPy 2.1.3, Torch 2.11.0+cu128, JAX 0.11.1,
Click 8.5.0, and safetensors 0.8.0. The final tested extension was 7,067,256 bytes,
SHA256 `15b1c66806cb5ccd17c1b89da72c173ba3f0442fb419fa180f2f2b61fd9924bd`,
as recorded in `ennx_native_build.json`.

The native final checkpoint is preserved locally at
`.cache/flame-native-t4/results/native-pilot/bo/best/`: all 81 tensor downloads
passed SHA256 verification, totaling 2,600,058,920 bytes. JAX checkpoint manifests
in the results archive are comparison metadata only; its tensor files were not
downloaded. The T4 session was released after verification and artifact transfer.

The final extension also passed the broader GPU search regressions:
`BF16_PARITY` reported all checks true, and `CORRELATED_PARITY` passed exact GPU
replay plus reference, acquisition, radius, and Gaussian checks. The CPU
comparison retained seven rounding-boundary mismatches within its existing
tolerance; this is not a claim of bit-exact CPU/GPU agreement. All 488 local
FLAME tests passed again in 19.52 seconds, and Ruff and `cargo fmt` checks passed.

### 300-Round Native Run

A subsequent native T4 run started from the original checkpoint with the same
312-problem TRAIN corpus, minibatch size two, four candidates, history two,
radius 0.01, all seeds zero, and acceptance threshold two standard errors.
It completed 300 proposal rounds: 301 selected weight evaluations including the
baseline, 601 minibatch objective evaluations, and 1,202 problem forwards.
There were 27 acceptances, the last at round 289, and one final checkpoint save.
The first eight events matched the earlier native pilot exactly in every
non-timing field.

Summed BO phase time was 465.05 seconds: ask 295.10, incumbent forward 61.58,
candidate forward 56.43, and tell 51.94 seconds. This is a sum of event timings,
not an outer-loop wall timer. End-to-end worker time was **552.72 seconds**,
including loading, baseline evaluation, saving, and Python GC/control overhead;
separate reload and full-corpus comparison are excluded. Total device memory
sampling at 100 ms yielded 5,515 samples, a peak of 12,787 MiB used, and a minimum
of 2,126 MiB free. Sampling can miss transient peaks.

Fresh-process reload reproduced both the original baseline and final-incumbent
minibatch losses with zero observed difference: runtime validation passed,
but optimization improvement was false. The full 312-problem
TRAIN objective **worsened**: mean loss rose from `2.4663068663615446` to
`2.480149207588954`, an increase of `0.013842341227409167` (about 0.561258%).
Loss improved on 118 problems and worsened on 194. This fixed training set is
not held out, and no generation accuracy was measured. The original checkpoint
remains preferred; the final checkpoint is not promoted.

The legacy `bo/best` directory contains the final minibatch incumbent, not the
best checkpoint on the full corpus. Its final minibatch was `[298, 291]`, with
losses `3.00500154495` and `2.42490243912` (mean `2.71495199203`). Reload and
comparison took 99.97 seconds, excluded from BO timing: 624 full-corpus problem
forwards plus four reload forwards, with no saves or JAX imports. The regressed
checkpoint is preserved for audit and reproduction under
`.cache/flame-native-300-t4/results/`: reports are `summary.json`,
`comparison.json`, and `outcome.json`, with final parameters in `bo/best/`.
All 81 downloaded tensors passed SHA256 verification, totaling 2,600,058,920
bytes; `bo/best/local-verification.json` records the check. The T4 was released
after all GPU work and artifact transfer completed.
Its final manifest SHA256 is
`594233b31689a7090eb759b9372349ff6f34d1a49b09c1f3afcac7e878895441`.

## Metal

`--backend metal` is an explicit experimental Apple-GPU backend. It uses full
BF16 model weights and FP32 computation, not the existing packed KDA-MoE path,
Torch MPS, or JAX Metal. CUDA and JAX remain available without a silent fallback.
The host ENNX acquisition reduces GPU-computed realized BF16 distances; the
four candidates share the same per-observation Thompson draw. Its supported
configuration is correlated sampling, history 2, four candidates, one selected
proposal, and paired-relative history with failure tolerance 4. Other history
sizes and controllers fail explicitly.

Forward kernels implement FP32 tiled matrix multiplication, grouped MoE dispatch,
normalization, causal attention, routing, and shifted masked cross-entropy.
Weight rows occupy separate resident buffers. The reference is allocated lazily;
the selected proposal is materialized in one reusable row. There are no model-size
host transfers per problem evaluation. Explicit `read()`/`read_best()` calls copy
weights for debugging and final saving only. Read-only incumbent views prevent
search mutation while alive; resolved or foreign proposals cannot be consumed.

Metal allocation preflight checks individual buffer limits and the recommended
working set. The Python driver reserves another 4 GiB for system/host use. These
checks are conservative estimates, not a guarantee against concurrent memory
pressure. Operation-scoped autorelease pools bound command-resource lifetimes.

For a local macOS development extension, use a Python 3.12 environment with
`ops/flame/requirements-metal.txt`. The Python-symbol linker flags are macOS-only:

```sh
PYO3_PYTHON="$PWD/.cache/flame-env/bin/python" cargo rustc \
  -p ennx-py --features metal --release --locked --lib -- \
  -C link-arg=-undefined -C link-arg=dynamic_lookup
cp target/release/libennx_rust.dylib src/ennx/ennx_rust.so
PYTHONPATH=src:. .cache/flame-env/bin/python -m ops.flame.metal_parity \
  --checkpoint .cache/flame-290m --corpus .cache/flame-paired-300-t4/train.json \
  --output .cache/flame-metal/parity.json
PYTHONPATH=src:. .cache/flame-env/bin/python -m ops.flame.bo .cache/flame-290m \
  --backend metal --tokens .cache/flame-paired-300-t4/train.json \
  --output .cache/flame-metal/run --evaluations 301
PYTHONPATH=src:. .cache/flame-env/bin/python -m ops.flame.metal_compare \
  .cache/flame-290m .cache/flame-metal/run --output .cache/flame-metal/comparison.json
```

Run/comparison output paths must not already exist. With the default
`--minibatch-refresh 1`, each normal round evaluates the incumbent and one
selected candidate on the same two problems: four problem forwards, with two
additional baseline forwards. There is no online validation batch. For a faster
exploratory run, `--minibatch-refresh 8` reuses a paired minibatch for eight
rounds, reuses accepted-candidate losses as the next incumbent measurement, and
evaluates the incumbent only at refresh boundaries. This reduces forward work
while deliberately increasing reuse of each minibatch; the selected value is
recorded in `run.json` and `events.jsonl`. Only the final incumbent is saved
once; the legacy `best/` directory name does not mean it is best on the full
corpus. The separate comparison uses every TRAIN problem and checks exact
checkpoint reload losses. It is not held-out evaluation or generated-code
accuracy.

On the 24 GiB Apple M4, 18 toy forward cases matched the independent FP32 CPU
reference with maximum absolute logit error `1.0728836e-6`. Twelve small search
rounds passed paired-history, acquisition-score, seeded replay, and CPU rounding
checks. Dedicated Rust tests also covered bounded allocation over 32 rounds.
Full-model baseline losses on TRAIN indices `[96, 21]` were
`[1.9968150854, 1.8652830124]`, versus CUDA's
`[1.9969390631, 1.8652825356]`. The maximum difference is `1.24e-4`; its cause is
not isolated, and bitwise or identical-trajectory CUDA/Metal parity is not claimed.
The two full-model forwards took 0.947 seconds with 65,262,617 workspace bytes.

An eight-round full-weight pilot completed with 34 problem forwards, no accepted
proposals, and one final checkpoint save. Timed ask/evaluate/tell phases totaled
30.575 seconds: 21.137 seconds in ask and 8.988 seconds in objective evaluation.
These are local run measurements, not an isolated hardware benchmark or evidence
of optimization efficacy. Artifacts are under `.cache/flame-metal/`.

The corrected 300-round Metal run then completed from the original checkpoint
with 32 accepted proposals, 1,202 training problem forwards, and one final save.
This historical run used contraction after all rejections, not the newer
three-outcome screening controller.
Timed BO phases totaled 1,100.718 seconds (18m21s): ask 748.051, objective
323.997, tell 28.671. The CLI process, including startup/baseline/housekeeping/save,
took 1,146.16 seconds (19m06s). macOS `time -l` reported peak process memory
footprint 18,227,386,512 bytes; this is not a GPU-only allocation measurement.
The first eight rounds exactly replayed the pilot apart from timings.

Post-run mean loss across all 312 TRAIN problems decreased from
`2.4663056689` to `2.4650071645`: a change of `-0.0012985044`, about 0.053%.
Loss improved on 204 problems and worsened on 108. Both checkpoint reload loss
checks were exact; all 81 final tensor files passed SHA256 checks, totaling
2,600,058,920 bytes. Full-corpus forward evaluation took another 161.364 seconds,
excluding reloads and checkpoint I/O. This small single-seed training improvement
is not evidence of held-out generation gains or competitiveness with gradient
methods. No checkpoint was promoted over the original.

Reports: `.cache/flame-metal/summary.json`, `comparison.json`, and
`process-timing.json`. The saved checkpoint is `bo-300/best/` under that directory;
`implementation/` contains the executable, source snapshot, and checksums.
The run used extension SHA256
`179806f8ebbd156c4ad23c537274d8a2effc6f668e981df3e4153e2c1bc2c605`.
Verification also passed 520 Python regression tests, the three opt-in Metal
binding tests on the M4, both dedicated Rust GPU harnesses, Ruff, and Cargo format
and check with and without the Metal feature. No commit or push was made.

### Three-outcome controller comparison

Two 50-round M4 runs started from the same original 1.3B-parameter checkpoint,
with identical executable, Python implementation, seeds, problem batches, and
settings except `--rejection-policy`. Both used 202 optimization problem forwards
and one final checkpoint save. The `all` run exactly replayed the numerical
trajectory of the first 50 rounds of the historical run above.

| Policy | Accepted | Inconclusive | Counted failures | Final radius | Final TRAIN loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| `all` (legacy) | 7 | 39 | 43 | 0.0001 | 2.465265144 |
| `deterioration` (default) | 6 | 35 | 9 | 0.02 | 2.447327136 |

Original TRAIN loss was 2.466305669. The new policy reduced it by 0.7695%, versus
0.0422% for the legacy policy in this paired, single-seed pilot. It never reached
the radius floor; the legacy run ended 20 rounds at that floor. Average realized
BF16 changed-weight fractions were 82.32% and 23.72%, respectively. The dense
proposal law, rounding, surrogate, and acceptance threshold were not changed.
Timed ask/evaluate/tell phases totaled 196.793 seconds for the new policy and
201.327 for legacy, excluding setup, baseline, housekeeping, saving, and comparison.
These timings are not an isolated hardware benchmark.

Post-run diagnostics used all 312 TRAIN problems, with 628 additional forwards
per run including reload checks. Those diagnostics did not select checkpoints
or change the online loop. Both reload checks were exact and all 162 saved
tensor files passed their hashes. This is training-loss evidence only, not
generated-code accuracy, a statistical guarantee, or a multi-seed result.

Artifacts and reproduction commands are in
`.cache/flame-metal/controller-ablation/README.md`; `summary.json` contains the
verified comparison. The new rejection flag is wired through both Metal and
CUDA. Verification passed 572 Python regression tests, 12 actual Metal binding
tests, seven dedicated Rust Metal search tests, and 15 extracted CUDA CPU helper
tests. Full CUDA compilation and T4 execution of the changed flag remain untested.

### Generation sanity check

The completed Metal BO checkpoint was compared with the original checkpoint on
the first eight pinned MBPP TRAIN prompts. Both models received only the prompt,
then generated with a 128-token limit; reference solution tokens were never fed
to the model. Candidate zero is greedy. Three additional candidates branch at
the first token and then decode greedily, because greedy chose a comment loop
even though a code-looking first token was in its top three. The Metal
implementation recomputes the full prefix for each step and copies only the
final-row logits, so this is a correctness sanity check rather than a cached-
generation benchmark.

All eight reference solutions passed the isolated macOS checker. The candidate
rerun produced 0/8 passing cases for both checkpoints: all 64 raw candidates
failed. Candidate branches often began with `def`, but generated incorrect
function names or bodies. Candidate zero still commonly entered a repeated
comment loop. Every branch hit the 128-token limit, and the original failures
were reported as `NameError` when the generated text contained no usable
function definition. This confirms that the lower teacher-forced TRAIN loss did
not translate to executable generation on this sample; decoding alone cannot
repair the checkpoint's task-conditioned code-generation quality.

The candidate rerun used 8,192 autoregressive problem forwards, plus four logits
parity forwards and two reload-check forwards. Generated code was not executed
automatically. The explicit `generate check` command ran all candidates only
after inspection
inside a fail-closed macOS sandbox with filesystem, network, child-process, CPU,
file-size, descriptor, and output limits. macOS did not provide a hard RSS or
aggregate disk quota; the report records that limitation. The checker is not
claimed to be tamper-proof against hostile Python. Raw outputs and report are in
`.cache/flame-metal/generation-candidates-4-rerun/`; no checkpoint was modified
or promoted.

## BO Driver

Build the CUDA extension from the same checkout, then run in a fresh process on
CUDA device 0, a T4:

```sh
XLA_PYTHON_CLIENT_ALLOCATOR=platform XLA_PYTHON_CLIENT_PREALLOCATE=false \
python -m ops.flame.bo .cache/flame-checkpoint \
  --tokens ops/flame/tokens.json --output .cache/flame-bo \
  --sampler correlated --evaluations 8 --candidates 4 --history 2 \
  --radius 0.01 --seed 0 --reference-seed 0
```

The budget includes the baseline. Each subsequent round scores four candidates
against resident BF16 history using native ENNX Thompson acquisition, then runs
one JAX forward on the selected full-weight candidate through DLPack. All 81
tensors participate; no gradients, adapters, or fixed coefficient basis are used.
Reward is negative mean next-token loss on the same supplied token batch.
This fixture tests integration, not learning quality or competitiveness with AdamW.

### Coding Solution Objective

`ops.flame.coding` prepares a fixed teacher-forcing objective from the official
MBPP **training split**, separate from the proposed BigCodeBench-Hard evaluation.
The objective is negative mean cross-entropy over reference-solution tokens and
one EOS per example. Prompt tokens and right-padding do not contribute to loss.
No gradients or generated programs are involved in candidate scoring.

```sh
python -m pip install -r ops/flame/requirements-coding.txt
mkdir -p .cache/flame-coding
python -m ops.flame.coding --output .cache/flame-coding/train.json \
  --count 8 --max-tokens 256
XLA_PYTHON_CLIENT_ALLOCATOR=platform XLA_PYTHON_CLIENT_PREALLOCATE=false \
python -m ops.flame.bo .cache/flame-checkpoint \
  --tokens .cache/flame-coding/train.json --output .cache/flame-coding/pilot \
  --sampler correlated --evaluations 8 --candidates 4 --history 2 \
  --radius 0.01 --seed 0 --reference-seed 0
```

Preparation needs only Click, Requests, and the pinned `tokenizers` package, not
model weights, Hugging Face credentials, or remote Python execution. Dataset and
tokenizer artifacts have immutable revisions and SHA256 checks. Existing output
files are not overwritten. The checkpoint's `common.pt` specifies
`HuggingFaceTokenizer` with `EleutherAI/pythia-12b`, whose tokenizer is GPT-NeoX
with EOS ID 0; a GPT-2 tokenizer would be incorrect. FLAME did not record the
tokenizer revision, so our artifact revision is an explicit reproducibility pin,
not a recovered training-time hash. Evidence URLs are in `ops/flame/coding.py`.

Examples are selected in ascending training-task ID order, subject only to the
declared token-length and tokenization-boundary checks. The preparer records all
exclusions and fails if it cannot fill the requested count. It never truncates
solutions. Prompt and solution are tokenized together; tokens crossing their
boundary are rejected. Each stored mask entry describes whether that token is a
prediction target, not whether its position's logits should be scored. EOS is
included in the length limit and loss denominator. Raw token-batch JSON remains
supported by the BO driver for earlier numerical fixtures.

The initial eight-example corpus contains 1,214 stored tokens, 1,206 next-token
targets, and 435 scored solution/EOS tokens. The longest sequence is 243 tokens;
62 of the 374 training examples exceed the 256-token cap, with zero boundary
exclusions. These are selected training examples, not a representative benchmark
score. References remain dataset-supplied code: no programs were executed or
independently verified during preparation.

The fixed-objective evaluator right-pads to the longest example and runs one compiled forward
at a time from Python. Padding is safe because attention is causal and
all padded targets are unscored. It accumulates solution loss and divides by the
total scored-token count, not an average of per-example means. Each scalar loss
is synchronized before the next example; an outer XLA loop was measured to add
roughly 0.5 GiB of temporary storage on T4. This bounds
forward workspace with respect to example count, but the padded token corpus
and masks remain device-resident in two packed buffers and are included in the
memory estimate. The compiled forward selects one row dynamically. Separate tiny
device allocations per example incurred enough allocator overhead to exhaust
T4 memory with the 312-problem pool, despite its small logical byte count.
After compilation, the driver raises the estimate if compiler temporary and
output storage plus a 64 MiB margin exceed the static forward allowance. It
rejects an oversized estimate before allocating resident BO rows. This remains
an estimate, not an arbitrary-length guarantee; a full-model T4 pilot is required
before a larger run.

`run.json` records the complete token document, objective digest, selected IDs,
token counts, normalization, provenance, and microbatch size. Lower reference
loss does not prove generated programs pass tests. BigCodeBench-Hard must remain
outside BO history and checkpoint selection; a separate generation and isolated
execution evaluator is still required to measure held-out coding performance.

On 2026-09-12, this eight-example objective completed a T4 pilot over all
1,300,024,320 BF16 weights: one baseline plus seven BO evaluations, four
candidates, history two, radius 0.01, and both seeds zero. All 351 FLAME Python
tests passed locally and on T4. The initial solution loss was 2.0803694725;
none of the seven proposals improved it. All 81 best-checkpoint tensor hashes
match the original checkpoint, and a fresh-process reload reproduced the loss.

The model process took 93.24 seconds, including loading, compilation, baseline,
BO, and saving. Recorded ask/evaluate/tell work totaled 24.63 seconds, of which
17.20 seconds were candidate objective evaluations. Peak device memory sampled
every 100 ms was 14,805 MiB, leaving approximately 108 MiB on that T4. The final
compiled per-example temporary allocation is 2,344,063,640 bytes. The initial
outer-loop version requested 2,860,204,440 bytes and failed during candidate
evaluation; its failure log is retained, not counted as a successful run.

Artifacts are in `.cache/flame-coding-t4/`, including the corpus, full best
checkpoint, events, timing, memory samples, package inventory, and failed-run
audit. This pilot validates the objective path, not optimization efficacy or a
coding benchmark score. A 300-iteration run has not been started.

### Paired Minibatch Training

Use a larger prepared TRAIN pool with `--minibatch-size 2 --minibatch-seed 0`
and the correlated sampler. With the default `--minibatch-refresh 1`, each
iteration draws a fresh uniform batch without replacement within the batch.
Candidate and incumbent use the same problems.
Only those problems are forwarded; no validation or confirmation batch is run.
This uses four problem-level forwards per normal BO step: two for the incumbent
and two for the selected candidate. The four acquisition candidates are unchanged;
only one is scored by the full model. The initial baseline uses two additional
forwards once. Two-problem comparisons have noisier uncertainty estimates than
the eight-problem batches used in the historical pilot below.

Unlike the fixed objective, minibatch rewards average each problem's mean
solution-token loss with equal problem weight. The current driver enables
`SearchState.enable_pairedrelative` before its first acquisition. ENNX history
contains an exact zero-valued incumbent anchor and paired improvements from
rejected candidates measured against that same incumbent, with the sampling
variance of each paired improvement. It does not fit absolute losses from
unrelated problem batches. The anchor occupies one of the configured history
slots, so `--history 2` retains the anchor and the latest rejected candidate.
Promotion resets history to the new zero-valued incumbent anchor; it does not
mix improvements measured against different incumbents. No additional weight
rows or objective forwards are required.

The default `--rejection-policy deterioration` distinguishes three outcomes.
Improvement above the paired standard-error threshold accepts the candidate.
Deterioration beyond the same threshold rejects it and counts a failure.
Everything else is inconclusive: the incumbent stays, the measured observation
still enters ENNX history, and the failure counter neither increments nor resets.
Both directional decisions require strict score ordering at native FP32 precision.
These are heuristic screens, not confidence guarantees with a two-problem batch.

`--failure-tolerance 4` halves the reference radius after four counted failures
since the last acceptance or contraction, bounded by `--radius-min`. Acceptance
adopts the selected candidate's radius and clears the failure counter. Contraction
neither changes the incumbent nor resets its reference direction, and there are
no TuRBO restarts. `--rejection-policy all` reproduces the previous rule, counting
every rejection, for matched-budget comparisons. Neither policy adds objective
forwards or checkpoint saves. Logs include the screening outcome and whether a
failure was counted, separately from the actual accept/reject decision.

The paired path uses `--paired-epistemic-scale 1`, interpreted as loss variance
per unit squared tensor-normalized distance, rather than the old inverse-initial-
radius-squared multiplier (10000 at radius 0.01). This is an explicit prior scale,
not an empirically calibrated uncertainty claim. The extra aleatoric floor is
zero because paired sampling variance is already supplied; the kernel retains
its numerical epsilon. `y_scale` stays one so observations and variances retain
raw loss units.

Acceptance still compares paired loss differences. The default
`--acceptance-se 2` requires improvement above twice its estimated standard
error, including the finite-population correction. This is a heuristic screen,
not a repeated-testing guarantee. Two problems cannot certify corpus improvement.
The 300-round results above used the previous absolute-history, acceptance-only
radius policy; they do not validate these changes. The legacy `tell_paired` API
is retained for reproducing that policy. The driver requires the new
`tell_pairedrelative` API and fails before model loading with an old extension.

The correction passed 510 local FLAME tests and a fresh locked CUDA-Oxide build
for T4 (`sm_75`, `native-flame`). On the T4, `ops/minibatch_parity.py` passed
zero-anchor/FIFO/recentering, batch-offset invariance under UCB and Thompson,
rejection contraction, validation, and DLPack lease checks. The existing BF16
and correlated parity suites also passed, including their documented CPU
rounding-boundary tolerance. A separate 1,848-weight synthetic native-forward BO
check completed eight rounds with 34 problem forwards including baseline,
one accepted update, and no JAX imports. These are correctness checks, not
evidence of improved 1.3B-model loss. No full-model rerun was performed.

The tested source, extension, and logs are in `.cache/flame-paired-fix-t4/`.
All 571 uploaded source hashes and the downloaded extension hash were verified;
the temporary T4 was released. Extension SHA-256:
`c5317aef4d21081bfdef9b0c859dc7694080ee081cda6978fff7d57a16fd9302`.

The accepted incumbent stays in GPU memory. Scoring it copies into existing
pending-row scratch space, not another model-sized allocation. There are no
per-step checkpoint writes: `best/` is written once after the loop. It is the
incumbent selected by noisy training comparisons, not a validated global best.
`best_reward` is its latest minibatch measurement; `final_minibatch` records the
indices and losses needed to reproduce that measurement. It must not be compared
with a full-pool loss or a previous batch's score.

`--evaluations 301` means a baseline plus 300 selected-candidate evaluations.
With the default refresh of one and changed proposals, paired scoring performs
601 objective evaluations, each on one minibatch. `--minibatch-refresh 8` keeps
the baseline batch for the first eight proposal rounds and refreshes the
incumbent at rounds 8, 16, ..., 296: the same run would perform 338 objective
evaluations and 676 two-problem forwards, assuming every proposal changes the
weights. Logs report both counts and both forward timings. Generation and
isolated execution remain a separate post-training evaluation, not a measured
outcome of this loop.

On 2026-09-12, the paired path completed a full 1,300,024,320-weight BF16 T4
pilot: 312 eligible TRAIN problems, eight problems per minibatch, four acquisition
candidates, history two, radius 0.01, all seeds zero, and acceptance threshold
two standard errors. The baseline plus seven BO steps used 15 minibatch objective
evaluations. Recorded ask/evaluate/tell work took 41.57 seconds; the model process,
including loading, compilation, and the single final save, took 98.48 seconds.
This excludes native toolchain setup and the separate reload check.

No candidate cleared the paired threshold. Three had positive measured mean
improvements but insufficient separation relative to the estimated uncertainty.
The final checkpoint's 81 tensor hashes match the original, and fresh-process
reload reproduced the last minibatch loss (approximately 2.3000025). This is not
evidence of optimization improvement or generated-code correctness.

Peak sampled GPU memory was 14,779 MiB, leaving about 134 MiB on this runtime.
The initial unpacked-corpus attempt failed at candidate scoring and is preserved
alongside the successful run in `.cache/flame-minibatch-t4/`. Both native paired
and correlated GPU parity suites passed; 439 local FLAME tests passed with
warnings treated as errors. The verified extension and full final checkpoint
are preserved locally. No validation set, confirmation batches, periodic saves,
or 300-iteration run were used.
The artifact's `source.zip` plus `verified-source/` overrides record the tested
source and Linux-resolved Cargo lockfile; that lockfile differs from the unchanged
checkout lockfile. The extension SHA256 is recorded in `minibatch-extension.json`.

### Correlated Proposals and Controller

The initial dense Gaussian reference is seeded by `--reference-seed` independently
of the PCG64 proposal/acquisition stream controlled by `--seed`. Each tensor has
a stored BF16 reference `R` and its squared RMS `q = mean(R * R)`, recomputed only
at initialization and after acceptance. Its candidate direction is
`U = rho * R / sqrt(q) + sqrt(1 - rho * rho) * E`, where `E` is fresh independent
standard Gaussian noise. Each weight's proposed change is the tensor's fixed
initial weight RMS times the candidate radius times `U`. Magnitudes and signs
both vary across weights; there are no flip bits or one-bit directions.

Conditional on any fixed, nonzero reference, `mean(U * U)` has expectation one
before numerical rounding. This holds even for an adaptively selected reference;
it does not make that reference Gaussian or prove noise-stability gains after
selection. There is no per-candidate normalization reduction. The radius is an
expected relative RMS scale, not a hard bound on realized changes, and `rho` is
a mixing coefficient, not a measured post-rounding correlation.

The native pool is exactly four candidates, with zero-based indices:

| Index | Persistence rho | Nominal radius |
| --- | --- | --- |
| 0 | 0.75 | max(radius_min, 0.5 * reference_radius) |
| 1 | 0.75 | min(radius_max, 2 * reference_radius) |
| 2 | 0 | max(radius_min, 0.5 * reference_radius) |
| 3 | 0 | min(radius_max, 2 * reference_radius) |

Indices 0/1 share one Gaussian innovation and direction; 2/3 share another. At the default
reference radius 0.01, the two step sizes are 0.005 and 0.02. The surrogate ranks
the realized BF16 candidates, not expected distances or four forward losses.

Strict reward improvement accepts the selected weights, replaces the stored
reference with BF16-rounded `U`, recomputes each tensor's `q`, and sets the reference
radius to its selected radius. Rejection (including a tie) leaves all three
unchanged but still adds the evaluation to history. The reference stores the
nominal direction, not the rounded difference between candidate and incumbent
weights. It occupies two bytes per weight; no direction chain grows with accepted
steps. This is a reference vector, not a model covariance matrix or fixed basis.

This mode has no TuRBO success/failure counters or restarts. Step size is selected
by acquisition within explicit bounds. It is a proposed heuristic, not a
convergence guarantee: rejection does not automatically shrink the radius, and
remembering an accepted direction is not evidence that it will stay useful.
Correlated mode requires `--candidates 4` and one pending evaluation, and rejects
an explicit `--failure-tolerance` rather than silently ignoring it.

For an equal-candidate baseline comparison, use a separate fresh output directory:

```sh
XLA_PYTHON_CLIENT_ALLOCATOR=platform XLA_PYTHON_CLIENT_PREALLOCATE=false \
python -m ops.flame.bo .cache/flame-checkpoint \
  --tokens ops/flame/tokens.json --output .cache/flame-bo-gaussian \
  --sampler gaussian --evaluations 8 --candidates 4 --history 2 \
  --radius 0.01 --seed 0 --failure-tolerance 4
```

All driver modes default to four candidates. `gaussian` means independent standard
Gaussian directions with the legacy TuRBO controller. `independent` means legacy
independent signs with that controller, not Gaussian noise. Both baselines allow
one to eight candidates and resolve omitted failure tolerance to 4. The native
API retains its legacy independent-sign default; the driver passes its mode
explicitly. Reference seed is logged in all modes but affects only correlated
proposals.

### Precision, Memory, and Audit

Tensor RMS scales are measured once from the original checkpoint. The distance
metric gives equal weight to each tensor's squared RMS change relative to that
fixed scale. All-zero tensors require an explicit `--zero-scale`; invalid scales
fail. BF16 proposals use round-to-nearest-even, so realized radii differ from
nominal radii and some weights remain unchanged. Native correlated selection
filters unchanged candidates whenever any non-null candidate is available,
including larger-radius candidates. An unchanged selected proposal stops the
driver without another objective evaluation, recorded as
`selected_proposal_unchanged`, not a claim of a global quantization floor. The legacy
`ParamBuffer` multi-term sampler retains its existing forced-step semantics.

The driver retains only `--history` evaluated rows, FIFO, on the GPU. The default
two-row history is a memory-limited integration baseline, not evidence of useful
surrogate accuracy at this scale. There is no SSD history or cuVS backend here.
Memory is preflighted as base, history, and one pending row (each BF16 row padded
to 128 weights). Correlated mode additionally needs `2 * parameter_count` bytes
for its dense reference: 2,600,048,640 bytes, about 2.42 GiB, for this checkpoint.
Small per-tensor/tile scratch allocations are covered by the workspace allowance,
not counted individually. Native reference allocation is deferred until the first
`ask`; the driver explicitly deletes the original flat JAX input immediately after
constructing `SearchState`, before that reference is allocated.

The incremental forward-workspace allowance is 2.25 GiB (2,304 MiB) for correlated
mode and 4 GiB for either baseline. The correlated allowance targets only the
fixed one-by-eight-token fixture after removing an observed full-weight copy.
A T4 isolation probe measured 10,035 MiB resident, 10,037 MiB after ask and DLPack
import, and 12,517 MiB after an eager reshape: exactly 2,480 MiB of additional
storage (about 2.42 GiB). Deleting that reshape returned usage to 10,037 MiB;
putting reshape and a reduction inside JIT used 10,039 MiB. The driver now passes
the borrowed batch directly and reshapes inside the jitted forward, while still
accepting the flat baseline input.

The batched forward signature is compiled after baseline evaluation and before
`SearchState` allocation. This fixes compilation ordering, but did not materially
reduce overall peak memory: the final eight-evaluation correlated run peaked at
14,763 MiB, versus 12,529 MiB for the matched Gaussian baseline. Only about
150 MiB remained available on that T4. Correlated usage between forwards was
12,535 MiB; sampled forward peaks added approximately 2,228 MiB.

Those measured runs used the 2 GiB preflight allowance. After measurement, the
guard was raised to 2.25 GiB to cover the observed forward workspace and native
context overhead. The resulting
two-history-row estimate is 14.35742 GiB, below the observed 14.46 GiB free after
JAX initialization. This guard-only change does not alter execution or the
algorithm; the measured results are not being relabelled as a run with the new
guard. This remains a narrowly fitting fixture,
not a memory-safety guarantee: transient peaks may be missed by sampling and
longer/larger token batches may OOM despite passing preflight.
Two observations cannot establish rich directional structure in 1.3B
dimensions; this remains a bounded-memory integration experiment.
Single-candidate DLPack export aliases its pending row, avoiding another 2.60 GB
copy. Versioned DLPack marks the tensor read-only. Legacy consumers such as this
JAX CUDA path receive a borrowed row; never donate or mutate it. Before `tell`
updates history, it waits for consumer streams, then a GPU pass checks every
pending weight against the seeded proposal and rejects any consumer modification.
This adds a full weight scan,
included in tell timing, without another model-sized allocation. Neither the
selected buffer nor a view may survive into `tell`.

`run.json` records configuration, tokens, checkpoint, Python source and extension
hashes, scales, sampler/controller, reference seed/storage, and completion status.
`events.jsonl` records selected seeds, scores, `candidate_index`,
`persistence`, actual selected nominal `radius`, base versions, per-tensor
changed counts and realized relative RMS, radius updates, and timings.
`reference_version` and `reference_radius` describe the nominal reference before
that proposal; `next_reference_version` and `next_reference_radius` describe it
after feedback. Versions start at zero and increment on accepted improvements.
Reference version/radius fields are null in both baseline modes. Seeds and versions
alone cannot replay an evolved reference without its intervening accepted steps.
`best/` contains checksummed BF16 tensors and a loadable manifest with optimization
provenance. Output directories must be new. These artifacts record results; exact
cross-device replay and resuming an interrupted optimizer are not implemented.

### BO Validation Status

New correlated driver integration and CPU unit tests cover settings, CLI options,
native constructor arguments, geometry diagnostics, reference-state logging,
rejections, and memory estimates. All 279 FLAME Python tests pass locally, including
195 BO-driver tests. Driver tests use mocked native search state; separate native
T4 checks validate sampling. CUDA parity covers exact same-GPU replay, acquisition
scores, reference updates/rejections, paired radii, multi-tile reference RMS,
FIFO wrap/restart copies, null-candidate filtering, and exported-buffer mutation
rejection. Compute Sanitizer reports zero errors. CPU Gaussian parity permits
only a narrow FP32 rounding-boundary allowance, not bit-exact host/GPU libm:
seven BF16 boundary differences occurred in the extended suite.

The existing packed Python CUDA API also passes its CPU comparison. The standalone
resident executable passes with `CARGO_PROFILE_RELEASE_LTO=false`; its default
release build hit CUDA-Oxide's missing `.llvmbc` LTO error. The tested Python
extension builds in release mode without this override.

On 2026-09-12 both Gaussian modes completed eight evaluations (baseline plus seven
proposals), with four candidates, two resident observations, seed/reference seed
zero, and the same one-by-eight-token fixture. Every run perturbed the full
1,300,024,320 BF16 parameters across 81 tensors.

| Measurement | Independent Gaussian / TuRBO | Correlated Gaussian / Selected Radius |
| --- | --- | --- |
| Initial next-token loss | 8.4305620 | 8.4305611 |
| Best next-token loss | 8.3352203 | 8.3983822 |
| Accepted proposals | 2 of 7 | 4 of 7 |
| Peak memory, sampled every 100 ms | 12,529 MiB | 14,763 MiB |
| Saved best reload loss | 8.3352194 | 8.3983822 |

Reloading verifies all 81 tensor checksums and matches reward within `1e-5`.
FP32 forward results can differ at the last few bits across compilations; exact
proposal replay does not imply bit-exact cross-process model loss. Moving batch
compilation before resident allocation did not materially reduce the correlated
peak: the measured run still left only about 150 MiB free. Larger batches or
history are not validated. The independent Gaussian baseline found the lower
loss here. This comparison changes both the proposal law and controller, so it
does not isolate the effect of persistence. This tiny fixed-token experiment is not evidence of language-quality
improvement, convergence, or competitiveness with gradient methods.

A separate synthetic timing comparison used 1,300,024,320 coordinates, one tensor,
four candidates, two observations, two warmup rounds, and eight measured rounds.
An identical forced acceptance/rejection schedule isolates engine overhead;
these timings exclude model forwards and checkpoint I/O.

| Sampler | Median Ask | Median Tell |
| --- | --- | --- |
| Legacy independent signs | 618 ms | 101 ms |
| Independent Gaussian | 847 ms | 148 ms |
| Correlated Gaussian | 905 ms | 211 ms |

Summing those medians, correlated overhead was about 12% above independent
Gaussian and 55% above signs. Correlated accepted/rejected `tell` medians were
253/170 ms; reference updates are not free. This is one T4 run in fixed mode
order, not a robust throughput estimate. Inspect `ops/bf16_compare.py` and
`ops/correlated_parity.py` for the reproducible checks. Logs, run manifests,
per-tensor events, the native extension, and both best checkpoints are saved
locally under `.cache/flame-gaussian-t4/`, not committed.

### Earlier Sign Baseline

The following results belong to the earlier **independent-sign, eight-candidate
baseline**, before the parallel history-copy and eager-reshape fixes.

For that baseline, 157 Python tests passed locally and on Colab T4. All 425 Rust CPU tests passed.
Native CUDA BF16 parity passed under Compute Sanitizer with zero errors, including
exact realized weights, diagnostics, single/multi-candidate exports, radius
updates, and rejection of finite, nonfinite, and delayed consumer writes.
DLPack cleanup callbacks do not acquire the GIL. The driver explicitly deletes
its borrowed JAX array before `tell`; simply dropping the Python reference left
a live lease in an instrumented test. Surviving consumer views remain prohibited.

On 2026-09-12, a four-evaluation full-checkpoint baseline run completed on a Colab T4:

| Measurement | Result |
| --- | --- |
| Parameters / tensors | 1,300,024,320 / 81 |
| Candidates / resident history | 8 / 2 |
| Initial radius / root seed | 0.01 / 0 |
| Baseline next-token loss | 8.4305620193 |
| Best next-token loss | 8.3913373947 |
| Accepted proposals | 1 of 3 |
| Peak device memory, sampled every 100 ms | 12,535 MiB |
| Steady proposal ask / forward / tell | about 1.10 / 0.063 / 1.68 seconds |

The first proposal forward took 3.94 seconds, including compilation; its `tell`
took 3.37 seconds. These are one-run observations, not robust benchmarks. The
memory measurement includes setup and best-checkpoint saving but excludes the
separate reload check. Reloading the saved best checkpoint verified every tensor
checksum and reproduced its reward exactly on that T4.

The run used the supplied one-by-eight-token fixture. It establishes full-weight
integration and memory feasibility for this configuration, not language-quality
improvement, scalability to longer batches/history, or competitiveness with
gradient methods. The local audit and best weights are in `.cache/flame-bo-t4/`;
they are not committed. These numbers must not be presented as results for the
correlated sampler or the acquisition-selected radius controller. Compare both
modes with four candidates, identical history/evaluation budgets and tokens, and
report wall time and reward progress separately.
