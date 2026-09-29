# Dense Qwen Control

ENNX keeps the FLAME-MoE experiment and the dense control as separate model
paths. The control is the pinned `Qwen/Qwen2.5-Coder-1.5B` checkpoint. Qwen's
recorded FIM format is used for the coding objective; the task is the prefix
and only the generated middle is scored.

The checkpoint is a dense Qwen2 model with 28 layers, 1,536 hidden units, 12
query heads, and 2 key/value heads. Its vocabulary is 151,936 tokens and its
weights are BF16. The architecture and checkpoint revision are validated in
`ops/qwen/config.py`; FLAME files are not reused because their tensor layout is
MoE-specific.

Install the Metal control dependencies:

```sh
python3 -m pip install -r ops/qwen/requirements-metal.txt
```

Download metadata first. This does not download model weights:

```sh
python3 -m ops.qwen download --output .cache/qwen-coder-1.5b
python3 -m ops.qwen inspect .cache/qwen-coder-1.5b
```

Download the 3.1 GB BF16 weight file only when running the control:

```sh
python3 -m ops.qwen download --output .cache/qwen-coder-1.5b --weights
python3 -m ops.qwen generate .cache/qwen-coder-1.5b \
  --prompt $'def add(a, b):\n    '
```

Prepare and run the dense code-completion BO experiment through the Qwen
tokenizer. The objective is teacher-forced solution loss; generated programs
are evaluated separately after the run. The Qwen checkpoint permits up to
32,768 total tokens; the default 512 generated-token cap is a practical
benchmark bound, and the CLI rejects requests that exceed the context.

```sh
python3 -m ops.qwen prepare \
  --checkpoint .cache/qwen-coder-1.5b \
  --output .cache/qwen-mbpp/objective.json \
  --count 8
python3 -m ops.qwen bo \
  .cache/qwen-coder-1.5b \
  .cache/qwen-mbpp/objective.json \
  --output .cache/qwen-mbpp/bo-8 \
  --evaluations 8
```

The BO command requires four candidates because that is the native correlated
proposal contract. It keeps two MBPP problems in each paired minibatch by
default and writes only the final incumbent checkpoint to the output
directory. `--minibatch-refresh 1` evaluates the incumbent on every round;
larger values trade objective freshness for fewer forwards.

Weight search retains absolute objective observations in a bounded FIFO
(`bo --history`, default 2). Acceptance moves the incumbent without clearing
history. Radius adaptation uses the CPU TuRBO controller; only a controller
restart resets history to the incumbent. The legacy paired-relative controller
is not used by `bo` or `joint`. Changing minibatches still makes historical
scores noisy and potentially incomparable; FIFO retention does not remove that
experimental limitation. Each run records the effective TuRBO tolerances and
per-round counters. With one evaluated arm, the current failure tolerance equals
the modeled parameter count, so contraction is not realistic at Qwen scale;
that behavior is retained only as the explicit comparison baseline.

The full-space BO sprint contract is recorded in
[`full-space-bo-sprint.md`](full-space-bo-sprint.md). Qwen BO reports include
the fixed LOOCV fitting objective, dense full-tensor BF16 perturbation semantics,
4096/16384/32768 context targets, the actual objective context, whether one of
those targets was reached, and the EGGROLL/hyperscale-ES comparison role in
`run.json["settings"]`. The baseline record also says whether EGGROLL was
evaluated in that run, so a declaration of the comparison regime cannot be
mistaken for comparison evidence.

Candidate materialization and teacher-forced loss evaluation are submitted to
the shared Metal queue as one pipeline. The BO loop therefore never performs
autoregressive decoding or a Python-side wait to score a candidate. Use `eval`
or `joint` when generated code and sandbox results are specifically needed.

For a generation-quality pilot, `joint` scores dense BF16 weight proposals with
generated FIM completions and the MBPP checker. It uses paired
common-random-number comparisons and writes `run.json`, `events.jsonl`, and the
final checkpoint:

```sh
python3 -m ops.qwen joint \
  .cache/qwen-coder-1.5b \
  .cache/qwen-mbpp/objective.json \
  --output .cache/qwen-mbpp/joint-8 \
  --rounds 8 \
  --tasks 8 \
  --max-new-tokens 512 \
  --seed 17
```

`events.jsonl` is the authoritative per-round record. Each event includes
proposal coverage (`changed_bf16_elements` and its fraction), score timing
split into generation, decoding, and checking, plus proposal, decision, tell,
sync, serialization, and native Metal proposal timings. `run.json` stores baseline and aggregate
timings and points to the event file instead of retaining a second in-memory
copy of every generated completion.

Perturbation records also include each candidate's realized weighted radius,
its realized cosine to the normalized persistent reference, and all six
pairwise cosines after BF16 rounding. A zero realized radius is valid when every
perturbation rounds back to the incumbent; its undefined cosines are written as
JSON `null` rather than fabricated as zero.

Decoder settings are fixed for the pilot so the run measures dense weight
proposals rather than a search over decoding knobs. Greedy decoding remains the
fixed baseline. Use `--samples` greater than one only when measuring pass@k,
since each additional sample is another full generation and checker invocation.

The forward pass is intentionally a readable reference implementation. It
implements Qwen2 grouped-query attention, RoPE, RMSNorm, SwiGLU, and tied
embeddings. The wheel-backed Metal evaluator is available from
`ops.qwen.metal.Evaluator`; it accepts `MetalProposals` directly, so the
resident correlated BO state does not copy each candidate through Python.

```python
from ops.qwen.metal import Evaluator

model = Evaluator(".cache/qwen-coder-1.5b", max_tokens=128)
losses = model.losses(model.weights, [[1, 2, 3]], [[False, True, True]])
search = model.search(-sum(losses) / len(losses), capacity=2)

prompt_ids = [1, 2, 3]
all_ids = model.generate(model.weights, prompt_ids, max_new_tokens=8)
```

`generate` keeps the autoregressive loop in Rust and uses a resident BF16 KV
cache after prompt prefill; Python only needs to tokenize the prompt and decode
the returned IDs. Prompt prefill is processed in bounded 256-token chunks, so
long prompts do not allocate a prompt-sized attention score matrix. `next_logits`
remains the full-prefix reference API for short inputs, while cached decode
keeps token selection on Metal with deterministic lowest-ID tie breaking and
must preserve greedy-token parity within the recorded BF16-cache tolerance.

The reference `logits` and `next_logits` paths reject inputs longer than 256
tokens. `losses` switches to bounded, cached prefill above that cutoff and
supports the configured context. The reference backend uses 256-token chunks;
MPS uses 2,048-token chunks. The cutoff does not change with the chunk size.

For Metal bring-up only, `ENNX_QWEN_SYNC_LAYERS=1` waits after embedding and
each transformer layer to localize a command failure. Normal evaluation keeps
the fused command-buffer path. Native commands fail after a bounded timeout
with their Metal label instead of blocking the Python process indefinitely.

For stage localization at the production 4K gate, use the existing test
primitive:

```sh
ENNX_QWEN_4K=1 \
ENNX_QWEN_CHECKPOINT="$PWD/.cache/qwen-coder-1.5b/model.safetensors" \
ENNX_QWEN_MPS=fp16 \
ENNX_QWEN_TILE_ATTN=1 \
ENNX_QWEN_STAGE_PROFILE=1 \
./ennx test
```

The profile reports GPU time for embedding, QKV projection, cached attention,
attention output, MLP expansion, MLP reduction, and final norm. It inserts
command boundaries between those stages, so use it to localize cost, not as the
production latency number. Ordinary runs do not create those boundaries. The
2026-09-18 production profile attributed 73.86% of GPU time to cached attention;
the exact measurements and next-kernel gate are recorded in
the historical Qwen experiments summarized in this document.

`ENNX_QWEN_TILE_ATTN=1` selects the exact 16-query by 16-key tiled prefill
experiment for the 128-wide production heads. Omit it to run the serial oracle;
the qualified 8-by-16 kernel remains in the parity fixture. The loss profile
records `tiled_attention`, so reports cannot silently mix serial and tiled
paths. The 4K gates measured `28,458.756 ms` total for serial, `15,243.587 ms`
for 8-by-16, and `11,991.141 ms` for 16-by-16. The 16-by-16 path preserved the
8-by-16 loss exactly at `0.004368`.

The materialized reference cutoff is fixed at 256 rows and is independent of
the cached prefill chunk. This distinction is covered at the 256/257 boundary;
tuning `MPS_CHUNK` must not change which attention algorithm scores an
objective.

The FP16 MPS prefill path also keeps gate, up, and activated MLP intermediates
in half precision between GEMMs. SiLU is evaluated in FP32 registers before its
result is rounded for the down projection. This removed redundant half-to-float
and float-to-half passes while preserving the 4K loss exactly; evaluator total
fell from `11,991.141 ms` to `11,635.314 ms`.

The native path must still be parity-tested against the reference model before
using its losses as an experiment result.

The optional PyTorch reference and generation path uses
`ops/qwen/requirements.txt`; it is not imported by the Metal evaluator. Metal
returns ordinary Python lists and does not require NumPy.

## Frozen generated-text evaluator

`./ennx tune examples/tuning/code-pretrain-evaluated.toml` uses this checkpoint
as a fixed scorer for another model's 4,096-token generation. This is separate
from Qwen weight search. Its raw teacher-likelihood reward failed the observed
repetition control; see [the generated-text reward audit](evals.md#generated-text-reward-audit-2026-09-30).

The frozen scorer selects attention explicitly in `[generation.reward]`:
`attention = "reference"` or `attention = "tiled16"`. It does not inherit
`ENNX_QWEN_TILE_ATTN`; the separate Qwen workflow retains its environment switch.
The tiled kernel reuses K/V across 16 queries without dropping context or keys.
The focused cached-attention numerical check passed on 2026-09-30.
The matched full-generation pair measured 46.32 seconds per reference round
versus 25.50 seconds with tiled attention; candidate tokens and acceptance
matched, but mean NLL was not bit-identical. The evaluated example selects
`tiled16` explicitly while the frozen-scorer default remains `reference`.
See [the comparison and qualifications](evals.md#full-power-attention-comparison).

`readout = "reference"` remains the default. The opt-in
`readout = "mps_fp32"` expands the frozen BF16 readout exactly to FP32 once and
uses MPS for rows of 32 or more. Smaller tails use the existing kernel. The
expansion costs **933,494,784 additional resident bytes** at the pinned shape;
it is never used for mutable BO candidate weights. The implementation retains
the source buffer and falls back for a different buffer.

On the full 128-by-151,936 output tile with hidden width 1,536, the synthetic
fixture observed zero logit and token-NLL differences; selected reference logits
also matched FP64 dot products within the test tolerance. Repeating that
isolated readout check with the actual pinned checkpoint weights also observed
zero logit and token-NLL differences. The activations were synthetic in both
checks, so neither is a full-model generation result. A focused end-to-end
small-model check covers mask gaps, readout tails, the 256/257 cache boundary,
and another weight buffer. This is numerical evidence, not a speedup result.

With active Low Power Mode disabled, three alternating timing pairs using the
pinned checkpoint readout measured reference times of 38.272, 38.623, and
37.783 ms, versus cached FP32 MPS times of 46.396, 47.544, and 46.656 ms.
The cached median was about 22% slower, in addition to its 934 MB memory cost
and 33.553 ms one-time preparation. This falsifies the cached-readout speedup
hypothesis for this measured tile; keep `readout = "reference"`. The opt-in
path remains available for further experiments. These are isolated readout
timings, not whole-round timings. Neither option is a demonstrated 200 ms
whole-round solution.
New generated-run metadata records device, active Low Power Mode, and student
cache context. Unknown power-policy reads are recorded as `null`; older runs
without this field cannot be assumed to have used full power.
