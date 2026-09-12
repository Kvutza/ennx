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

The reference `logits`, `next_logits`, and `losses` paths currently reject
inputs longer than 256 tokens. This is deliberate: those APIs retain the
materialized reference attention implementation; use `generate` for the
long-context cached path.

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
the [archived sprint ledger](archive/full-space-bo-sprint.md).

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
