# Coding-agent research contract

Status: normative design contract, version 1. The existing generated-text
experiments do not yet satisfy it.

## Claim under test

ENNX should turn a coherent, pretrained coding policy into a better coding
policy by selecting full-coordinate perturbations with Bayesian optimization.
The resulting model should eventually operate through an ordinary agent harness
such as Pi without a harness-specific model fork.

This claim has three independent parts:

1. the starting checkpoint can already generate coherent code;
2. ENN selection improves executable held-out outcomes over matched random
   selection at the same objective budget;
3. one complete 4,096-token BO round remains within the declared latency budget.

Passing one part is not evidence for either of the others. In particular,
subsecond generation from an untrained checkpoint is a systems result, not a
coding-model result.

## Training boundary

Causal next-token training teaches the base policy language, code, and tool
syntax. It is allowed to expose the preceding ground-truth prefix during
training. Evaluation and BO candidate scoring are free-running: a candidate
must consume only the task context and its own generated continuation.

BO is a post-training or fine-tuning mechanism. It must not be presented as a
substitute for acquiring language from an untrained billion-weight model.

The training progression is:

1. code and technical-text causal pretraining;
2. repository-aware continuation and infilling;
3. executable repair and tool-use trajectories;
4. free-running outcome optimization;
5. ENN-versus-random BO evaluation.

An elaborate repository packing strategy is not presumed superior. File-level
and repository-level mixtures must be compared with the same tokenizer,
checkpoint, token budget, context length, and evaluation set.

## `ennx.agent.v1`

`ennx.agent.v1` is the provider-neutral transcript contract. A persisted record
contains:

- a schema identifier and stable episode identifier;
- model-visible system, user, assistant, and tool messages in exact order;
- assistant text segments and tool calls as separate typed segments;
- a unique call identifier, tool name, and structured arguments for each call;
- a tool result linked to exactly one preceding call;
- truncation, cancellation, and tool-error state when they occur;
- immutable task provenance and split identity outside model-visible messages.

The public Rust types and structural validator live in
[`agent_contract.rs`](../rust/crates/ennx/src/agent_contract.rs). This first
boundary validates persisted complete or interrupted transcripts. Incremental
stream-event validation and provider adaptation remain separate work.

The first supported tool vocabulary is `read`, `write`, `edit`, and `bash`.
Tool arguments are structured data, never a JSON-looking string hidden inside a
text segment. A complete training episode has no unresolved tool call. An
inference prefix may end with one unresolved call while the harness executes it.

The transcript does not contain hidden tests, reference patches, grader source,
or post-task annotations in model-visible messages. Provider adapters may map
the transcript to Pi or an OpenAI-compatible API, but stored training data must
not depend on either wire format.

Required conformance fixtures are:

- ordinary text response;
- empty text followed by a tool call;
- multiple sequential tool calls and results;
- malformed arguments rejected before execution;
- tool failure followed by recovery;
- cancellation during text and during arguments;
- context overflow;
- arbitrary Unicode split across streaming chunks.

## `ennx.reward.v1`

Every candidate is scored from its own free-running output. Reward processing
has three layers and may not silently mix them.

The public record types, validation, and non-compensatory comparator live in
[`coding_outcome.rs`](../rust/crates/ennx/src/coding_outcome.rs). The comparator
rejects different check budgets and considers efficiency only after every
correctness field ties.

### Eligibility

A candidate is ineligible if it contains non-finite model state, cannot be
decoded according to the tokenizer contract, violates the transcript schema,
requests a forbidden tool, or exceeds the declared resource budget. An
ineligible candidate cannot be rescued by a favorable soft score.

Repetition, token diversity, identical-token runs, and syntax balance are
diagnostics for policy collapse. They are calibrated against held-out corpus
continuations and the frozen starting checkpoint. They are not positive coding
rewards and must not be tuned until a collapsed model appears superficially
diverse. If the starting checkpoint fails the collapse envelope, the experiment
stops before BO.

### Executable outcome

Eligible candidates are ordered by declared task outcomes:

1. no previously passing check regresses;
2. more task-specific failing checks become passing;
3. the requested behavior is satisfied on hidden checks;
4. the solution respects task-specific safety and scope constraints.

These outcomes are recorded separately. They must not be compressed into a
weighted scalar that permits extra failing tests to compensate for a
regression. A deterministic total order may use task completion, then
regression count, then newly passing checks, followed by predeclared
task-specific ties.

### Efficiency tie-breakers

Only candidates tied on correctness may be ordered by generated tokens, tool
calls, modified lines, or execution time. Latency is reported independently
from model quality.

Native text-overlap, reconstruction, and frozen-model likelihood rewards remain
diagnostic probes. None is a certified proxy for executable coding quality.

## Two evaluation clocks

The latency-critical BO round and the research-quality agent evaluation operate
at different frequencies.

The immediate objective runs for every candidate. It must be deterministic,
resident or prebuilt, cheap enough to remain inside the measured BO round, and
must expose an executable or protocol outcome rather than an unconstrained text
similarity. Its complete cost is included in `round_ms`.

The periodic evaluation runs outside candidate selection on frozen held-out
tasks. It may build repositories, execute full test suites, or run an agent
harness. It never supplies acceptance, acquisition, trust-region, checkpoint
selection, or early-stopping feedback. Its purpose is to detect proxy failure,
not to repair the optimization trajectory after looking at validation data.

If immediate reward improves while periodic outcomes do not, the proxy is
falsified. The experiment stops; the result is not reported as training
progress.

## `ennx.eval.v1`

A valid comparison freezes all of the following before either arm runs:

- source and starting-checkpoint identities;
- tokenizer and transcript schema;
- train, validation, and untouched test split manifests;
- task, objective-call, token, tool-call, and wall-clock budgets;
- proposal pools and domain-separated random streams;
- harness, tool implementations, sandbox, and machine power state;
- pass-to-pass and fail-to-pass checks;
- stopping and failure rules.

The minimum BO comparison is ENN selection against uniform selection from the
same candidate pool. It uses independent repetitions, alternates execution
order, preserves failures, and reports paired uncertainty. Acceptance count,
training reward, or acquisition score alone cannot establish superiority.

Model capability reports at least:

- pass-at-one under one frozen harness;
- regression-free and task-success rates;
- repository localization coverage and context efficiency;
- valid tool-call and argument rates;
- cancellation, overflow, and malformed-stream behavior;
- complete trajectory and failure-category counts.

Systems performance reports at least:

- complete candidate-round wall time;
- proposal, generation, verification, and reward times;
- generated and verifier-evaluated positions;
- speculative acceptance length and repair waves;
- host-device synchronizations and transferred bytes;
- model coordinates perturbed and realized FP16 changes;
- active experts and peak resident memory;
- machine, operating-system, build, and power state.

The 200 ms target applies to the declared immediate 4,096-token BO round. A
repository build or interactive agent episode is a separate end-to-end metric
and must not be hidden outside that round or relabeled as the same workload.

## Promotion gates

Work advances only after the preceding gate is satisfied:

| Gate | Required evidence |
| --- | --- |
| Data | Provenance, deduplication, repository-disjoint splits, and no grader leakage |
| Base policy | Coherent held-out 4,096-token code generation without collapse |
| Protocol | All `ennx.agent.v1` conformance fixtures pass |
| Immediate objective | Executable signal varies on controlled good, bad, and adversarial candidates |
| BO | ENN beats matched random selection with paired uncertainty |
| Agent | Held-out repository tasks improve under a frozen harness |
| Performance | Sustained complete-round distribution meets the declared latency target |

The present model has demonstrated a subsecond round but has not passed the base
policy gate. The next model-quality milestone is therefore a coherent pretrained
checkpoint, not a longer run of the current generated-text rewards.

Generation studies declare `purpose = "systems_probe"` or
`purpose = "coding_optimization"`. The latter is rejected before GPU allocation
unless a SHA256-bound `ennx.base_policy.v1` manifest records causal training and
complete held-out 4,096-token qualification without test feedback. Existing
untrained examples are explicitly systems probes.

## Attention and verification boundary

The target verifier is exact relative to the declared ENNX/PISA model semantics.
It is not evidence of equivalence to unrestricted dense-softmax attention.
Sublinear attention claims must state their learned-distribution assumptions,
measure failures on agent transcripts, and provide a reliability certificate or
fallback. Worst-case uniform approximation must not be implied.

Speculative generation treats text, tool selection, structured arguments, and
post-tool continuation as separate regimes. A future trained block editor may
learn from target-verifier corrections, but the target verifier remains the
authority and its work remains in the latency measurement.

## External standards used

- SWE-Bench Pro Verified: contamination-resistant executable evaluation and
  anti-reward-hacking controls.
- SWE-Explore: repository localization and context-efficiency measurement.
- KAT-Coder and KAT-Coder-V2.5: staged coding training, executable sandboxes,
  recovery trajectories, and deployment-aware reinforcement learning.
- AgentSpec and OnlineSPEC: structure-aware speculation and online learning
  from verifier feedback.
- Pi: a practical downstream transcript, streaming, and tool-use compatibility
  target, not the owner of ENNX's internal data format.
