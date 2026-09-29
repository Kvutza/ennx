# Bayesian selection audit

## Pilot results: 2026-09-29

Across three paired seeds, random selection finished with slightly lower
validation NLL in every pair. The mean difference was 0.00004677 NLL in favor of
random selection. This pilot provides no evidence that UCB selection improves
the final model over uniform selection under the shared ENN acceptance policy.

| Proposal seed | Initial NLL | ENN final NLL | Random final NLL | ENN minus random |
| --- | ---: | ---: | ---: | ---: |
| 123 | 9.01461023 | 9.01469535 | 9.01462430 | +0.00007105 |
| 1009 | 9.01461023 | 9.01448697 | 9.01442760 | +0.00005937 |
| 2027 | 9.01461023 | 9.01434815 | 9.01433825 | +0.00000989 |

ENN accepted 1, 2 and 1 proposals after initialization; random selection accepted
1, 1 and 1. Both methods improved over initialization in two pairs and worsened
in one. All six initial validation vectors and all paired initialization-round
observations, radii and decisions matched exactly.

The raw trajectory artifact is
`.cache/ennx/selection-ablation-20260929.jsonl` (192 rounds, 12 validation records,
and a completion marker). Training and validation file SHA256 values matched
the cached corpus manifest:

```text
train       b0be93d57ce24986c62bac7a7e99dcde8b6da1865654e90847e4381ba6d0fd21
validation  9fef27a887ef2d03a6090742ceaea72c4c2e767a6e83aa5a2b47cd179be5047d
```

### Identical pools

| Seed | UCB-selected index | UCB minus uniform expected NLL | UCB minus same-radius expected NLL |
| --- | ---: | ---: | ---: |
| 123 | 1 | -0.00011595 | -0.00013989 |
| 1009 | 3 | +0.00005424 | +0.00000176 |
| 2027 | 1 | +0.00001511 | +0.00002006 |

UCB beat the expectation of uniform selection in one of three pools and lost
in two. The mean difference favors UCB by about 0.00001553 NLL, driven by its
first-pool win. It picked the best candidate in that pool and the worst candidate
in the second pool. This mixed one-step result does not establish reliable
directional selection or contradict the final-trajectory result. All three UCB
choices used the larger radius; the same-radius comparison removes that radius
choice from the comparison.

Raw artifact: `.cache/ennx/pool-ablation-20260929.jsonl`. Each record includes
all 64 candidate sequence scores and the 16 incumbent sequence scores.

### Exact-distance rankings

Each row covers 45 guided pools across three seeds. The error column is the
largest relative error over every candidate/history distance in those pools.

| Coordinates | Noise | Changed winners | Changed radius | Maximum distance error |
| --- | --- | ---: | ---: | ---: |
| 4,096 | Gaussian | 22/45 | 7/45 | 5.431% |
| 4,096 | Rademacher | 14/45 | 2/45 | 4.535% |
| 65,536 | Gaussian | 17/45 | 4/45 | 1.179% |
| 65,536 | Rademacher | 23/45 | 5/45 | 1.238% |
| 1,048,576 | Gaussian | 12/45 | 4/45 | 0.329% |
| 1,048,576 | Rademacher | 19/45 | 4/45 | 0.283% |

Across all fixtures, exact distances changed 107/270 selected indices. Four
changes were tied at the CPU oracle's FP32 score precision; the other 103 had
strictly better exact acquisition scores. CPU scoring using production's
approximate distances matched all 270 GPU-selected indices. Resident-row
distance error stayed below 0.000071%, supporting the interpretation that the
nonresident approximation causes these differences.

At one million coordinates the approximation changed 31/90 choices despite
distance errors below 0.33%. However, the largest exact acquisition-score loss
there was only 0.000000621 on this synthetic objective. These are near-tie
ranking changes, not evidence of a corresponding degradation in training loss.
Higher-dimensional concentration reduces distance error without guaranteeing
identical candidate rankings.

Raw artifact: `.cache/ennx/geometry-audit-million-20260929.jsonl`, containing
270 pools with both distance matrices and acquisition scores. The earlier
`.cache/ennx/geometry-audit-20260929.jsonl` is the preliminary two-size audit.

### Assessment

The code executes full-weight derivative-free proposals at billion-parameter
scale. These measurements do not demonstrate a useful Bayesian selection
advantage. Keep the random-selection baseline and fixed validation gate when
changing proposal geometry or the surrogate. Exact-distance ranking checks
should accompany any claim that implicit history preserves acquisition;
the present approximation does not preserve it on these fixtures.

The evidence does not establish that exact history would fix training, that
longer runs would fail, or that the million-coordinate ranking-change rates
apply to the billion-weight model.

## Scope

Verification: all three GPU diagnostics completed successfully. The full
`ennx-unit` suite passed 537 tests with 18 ignored diagnostics. The existing
configuration test had two stale Gaussian assumptions after the example changed
to Rademacher; those expectations were repaired without changing optimizer
behavior. Rust formatting for the new modules, source-name checks, Python
compilation, shell syntax and artifact completeness checks also passed.
This is a dated test record, not a fresh check of later configuration edits.
Aggregate measurements are saved in
`.cache/ennx/bo-audit-summary-20260929.json`.

This diagnostic compares acquisition selection on the production 1,038,508,544
weight FP16 pretraining model and measures the implicit-history approximation
against an exact distance oracle. It does not change the production optimizer.

## Reproduce

Use an existing **resolved** pretraining `experiment.toml`, with UCB, ten neighbors,
an absolute `dataset` path and a sibling `validation.ennxptn`. Apple silicon and enough memory for the
production scorer are required.

```sh
bash tools/bo-audit /absolute/path/to/experiment.toml .cache/ennx/audits
```

The runner creates a fresh artifact directory, captures source checksums and
test logs, and writes `selection.jsonl`, `geometry.jsonl`, `pool.jsonl`,
`distance-scaling.jsonl`, and `summary.json`. Existing result files are never
overwritten. The experiments are ignored Rust tests, so normal tests do not
launch model training.

## Protocol

**Selection trajectories.** Three paired proposal seeds (123, 1009, 2027),
32 rounds per policy, four proposals and one training objective evaluation per
round. Nine initialization rounds plus the initial observation are shared.
The model initialization, corpus order, proposal seeds, fitting budget, and
acceptance policy match within each pair. The second pair reverses policy order.
Uniform selection uses a separate seeded RNG and still computes the same ENN
posterior for acceptance. This isolates acquisition selection; it is not an
ablation of the entire surrogate. Actual candidate vectors and radii may diverge
once the policies accept different incumbents.

The first eight validation batches (16 sequences, each 4,096 tokens) are scored
before and after each trajectory. Validation never enters fitting, acceptance,
radius updates, or checkpoint selection. Each arm has 33 training objective
evaluations and 16 validation objective evaluations. The diagnostic runs asks
synchronously; its wall times are not production performance measurements.

**Identical proposal pools.** For each seed, reconstruct the common initialization
history and evaluate all four candidates from its first guided pool on the same
validation batches. Compare the UCB choice with the mean loss over all four
choices, which is the expected loss of uniform random selection. Also compare
against random selection between the two candidates at the selected radius.
The incumbent is evaluated as a reference. There is no validation feedback and
no continuation chosen from these results.

**Distance-scaling proposal pools.** For each seed, reconstruct two bitwise
paired initialization histories and give global scaling and self-tuning scaling
the same first guided four-candidate pool. Score every candidate once on the
fixed validation batches, then report the NLL difference between the two
policies' selected candidates. Validation selects neither candidate and never
enters either search state. This is the direct bounded eval for
`distance_scaling = "self_tuning"`; it measures a selection decision, not a
full convergence trajectory. These records were added after the dated pilot
above and therefore have no claimed result in that section.

**History geometry.** On synthetic quadratic objectives at 4,096, 65,536 and
1,048,576 coordinates, retain every realized FP16 observation. Use the actual
Metal proposal/selection kernels with Gaussian and Rademacher perturbations,
three seeds and 24 rounds. For every guided pool, compute block-weighted
distances to all historical vectors in FP64. Rescore acquisition with those
distances and the same fitted parameters; compare its winner with production.
The audit checks the CPU score oracle against GPU scores, exact resident-row
distances, and bitwise restoration of the selected proposal after materializing
counterfactual candidates.

This isolates query-distance approximation. It does not refit the surrogate on
exact pairwise distances and does not prove billion-dimensional ranking parity.
The production selector adds the squared proposal displacement to stored base
distances for nonresident observations, omitting the directional cross term.
Only the two resident historical rows receive direct distance corrections.
The existing `distance = "full_realized_weights"` run label should therefore
not be interpreted as exact geometry to all 128 logical observations.

## Interpretation

Lower validation negative log likelihood (NLL) is better. A negative
`enn_minus_random_nll` favors ENN. Changes in acquisition ranking alone do not
establish worse objective outcomes, particularly when score gaps are small.
Three proposal seeds with one model initialization and one validation corpus
are a bounded pilot, not a general convergence or training-quality claim.

The initial experiment uses the resolved configuration from the existing
100-round artifact: Rademacher, UCB beta 1, ten neighbors, initial epistemic
scale 0.7, aleatoric scale 0.05, output scale 1, 30 fitting candidates and ten
fitting samples, trust lengths 0.01/0.0001/0.1. The diagnostic overrides the
round count and random seeds as specified above. The checked-in user-facing
pretraining example may contain different initial settings.
