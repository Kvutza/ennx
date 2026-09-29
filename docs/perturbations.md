# Dense correlated Gaussian proposals

Status: native Gaussian build and GPU validation passed, including Compute
Sanitizer with zero errors. Correlated and independent Gaussian full-FLAME T4
runs completed with eight evaluations, four candidates, and two history rows.
These establish integration for the tested configuration, not an optimization
advantage; see [FLAME validation results](flame.md#bo-validation-status).
The proposal uses dense real-valued noise across the participating weights.
ENNX's learned surrogate and acquisition remain unchanged. Check the
conditional moment algebra with `python3 scripts/check_pertmath.py`.

## Reference and proposal

Condition on everything known before generating a pool: incumbent weights $W$,
history, radius, and an arbitrary finite stored BF16 reference $R_l$ for each
tensor of size $n_l$. Each reference tensor must be nonzero. Compute

$$
q_l=\frac{1}{n_l}\sum_i R_{l,i}^2,\qquad
V_{l,i}=\frac{R_{l,i}}{\sqrt{q_l}},\qquad
\frac{1}{n_l}\|V_l\|^2=1.
$$

Compute $q_l$ from the **stored, rounded reference**, after initialization and each
accepted update. No per-candidate normalization is needed. The reference need
not be Gaussian, uniform, or independent of past rewards. Initialization uses
seeded Gaussian values rounded to BF16; reject nonfinite or zero-RMS tensors
explicitly rather than silently changing their scale.

Draw fresh independent standard Gaussian coordinates $E_{l,i}$, independent of
the entire conditioned history. For $0\leq\rho\leq1$, define

$$
U_{l,i}=\rho V_{l,i}+\sqrt{1-\rho^2}\,E_{l,i},\qquad
\Delta_{l,i}=r s_l U_{l,i},\qquad
W'_{l,i}=\operatorname{BF16}_{\mathrm{nearest}}(W_{l,i}+\Delta_{l,i}).
$$

Here $s_l$ is the RMS of the **initial checkpoint tensor**, fixed throughout the
run. An initially zero weight tensor requires an explicit positive absolute
scale. Reference normalization $q_l$ is distinct from these fixed weight scales.
For $\rho<1$ and positive $r,s_l$, the ideal proposal has full continuous support;
every coordinate receives a nonzero perturbation almost surely. BF16 rounding
can erase changes. There is no forced one-ULP move or candidate normalization.

Conditionally, $U_l$ has mean $\rho V_l$ and independent coordinates with variance
$1-\rho^2$. Thus "correlated" describes persistence with the reference, not a
learned cross-weight covariance matrix. In particular,
$\mathbb E[\langle U_l,V_l\rangle/n_l\mid R]=\rho$; this is not an exact cosine
similarity or Pearson correlation after adaptive selection or rounding.

## Conditional radius moments

Let $A_l=\|U_l\|^2/n_l$. For every fixed nonzero reference,

$$
\mathbb E[A_l\mid R]=1,\qquad
\operatorname{Var}(A_l\mid R)=\frac{2(1-\rho^4)}{n_l}.
$$

Proof: write $a=\rho V_{l,i}$ and $b^2=1-\rho^2$. The standard Gaussian moments
give $\mathbb E[(a+bE)^2]=a^2+b^2$ and
$\operatorname{Var}((a+bE)^2)=2b^4+4a^2b^2$. Sum independent coordinates and use
$\sum_i V_{l,i}^2=n_l$. Consequently

$$
\mathbb E[\|\Delta_l\|^2/n_l\mid R]=r^2s_l^2.
$$

The radius controls **expected squared RMS**, not a hard ball and not necessarily
the expectation of RMS itself. For any $t>0$, Chebyshev gives the rigorous bound
$\Pr(|A_l-1|\geq t\mid R)\leq\min(1,2(1-\rho^4)/(n_l t^2))$.
Small tensors have broad tails; the total model size does not fix that. No
stronger concentration or bounded-step claim is needed here.

## Conditional history-distance moments

For any fixed historical row $h$ and fixed nonnegative diagonal metric weights
$w_i$, flatten tensor indices and set

$$
m_i=(W_i-h_i)+r s_i\rho V_i,\qquad
\sigma_i=r s_i\sqrt{1-\rho^2},\qquad
D=\sum_i w_i(m_i+\sigma_i E_i)^2.
$$

Then, conditional on the pre-pool state,

$$
\mathbb E[D]=\sum_i w_i(m_i^2+\sigma_i^2),\qquad
\operatorname{Var}(D)=\sum_i w_i^2(2\sigma_i^4+4\sigma_i^2m_i^2).
$$

Proof: apply the same scalar Gaussian moments; independence eliminates the
cross-coordinate covariances. These statements survive arbitrary adaptive
selection of **past** references and history because we condition on them.
They apply separately to each newly generated candidate, not automatically to
the acquisition-selected winner or to distances between shared-noise candidates.

These are real-arithmetic distance moments, not a replacement for ENNX scoring.
Candidates with identical radius and persistence have identical expected distance
but different realized distances. Keep exact realized BF16 distances for ranking,
and evaluate the same BF16 candidate that was scored. Expected-distance scoring
would erase seed-dependent distinctions.

## Four-candidate proposal and controller contract

Use two independent Gaussian streams, shared within each radius pair:

| Index | Direction | Persistence $\rho$ | Radius |
| --- | --- | --- | --- |
| 0 | Persistent | 0.75 | max(radius_min, TuRBO_length / 2) |
| 1 | Same persistent direction | 0.75 | min(radius_max, 2 * TuRBO_length) |
| 2 | Fresh | 0 | max(radius_min, TuRBO_length / 2) |
| 3 | Same fresh direction | 0 | min(radius_max, 2 * TuRBO_length) |

The Qwen experiment initializes the shared TuRBO length at 0.01, so its initial
proposals use 0.005 and 0.02. Bounds are 0.0001 and 0.08. These choices are
experimental, not optimality results. Sharing noise within a pair compares step
sizes along exactly the same ideal direction. The pool's candidates are not four
independent samples.

ENNX ranks four candidates and the model evaluates only the winner. Strict reward
improvement accepts its BF16 weights, sets $R_l=\operatorname{BF16}_{\mathrm{nearest}}(U_l)$,
recomputes $q_l$ from that stored reference, and leaves radius adaptation to the
shared TuRBO controller. Store the nominal direction, **not** the rounded weight
difference or a normalized candidate. The native implementation validates the
initialized reference before the first ask and validates updated reference norms
when collecting an accepted update's result, **after** metadata, weights, and
reference have been mutated. This is not validation before commit or a rollback
guarantee. Rejection or a tie retains incumbent and reference; the observation
still enters bounded history and the controller still sees its objective value.
Fresh noise is generated next round.

### Reference failure handling

A bad updated norm fails closed: stop the search and do not treat the mutated
state as a successful result. The queued wrapper retains its queued marker when
`collect_tell` fails, so operations requiring `sync` continue to fail.

The native engine now checks stored reference inverse RMS values for finite
positivity in correlated-mode `check_ask` (once initialized), `check_tell`,
`read_reference`, and `collect_tell`. Initialization validates the real reference
before the first ask; the ask guard skips the lazy placeholder. Thus an invalid
updated norm blocks subsequent ask/tell operations even for direct engine callers,
and reference reads fail rather than returning an invalid direction.

These checks synchronize preceding work and read only per-tensor statistics; they
do not rescan model weights. This is **fail-closed continuation, not transactional
rollback**: metadata and weights may already have changed when validation fails.
The checks do not promise that every low-level diagnostic or raw weight export
rejects the mutated state. Whole-model
null-candidate filtering does not establish positive reference RMS for every tensor.

The absolute-history Metal path uses the same `TurboTrustRegion` update and
restart behavior as the CPU path. The selected radius does not directly become
the next center length. TuRBO updates from the retained sequence of objective
values, doubles length after its success tolerance, halves it after its derived
failure tolerance, and resets bounded history to the incumbent on restart.
Measured startup and restart values seed the Metal controller before its next
update. The generic failure tolerance is derived from the ambient parameter
count. The FBT round experiment explicitly overrides it to four failures, independent
of dimension; see [its runbook](turbo-enn.md). That experimental budget makes
contraction reachable without establishing that the controller is suitable for
full-space weight search. A wholly quantized-away selected proposal must not incur a duplicate
forward; stopping for it does not imply the other candidates are null. Controller
choices have no convergence or regret guarantee.

## What the research does and does not justify

For an **unselected standard Gaussian** reference $Z$ and independent Gaussian
$E$, the stationary Ornstein-Uhlenbeck coupling
$Z'=\rho Z+\sqrt{1-\rho^2}E$ preserves Gaussian marginals. For square-integrable
reward $F$ with orthonormal Hermite coefficients $\widehat F_\alpha$, its covariance
is $\sum_{|\alpha|>0}\rho^{|\alpha|}\widehat F_\alpha^2$. This motivates testing
persistence when low-order reward components matter; see the Gaussian noise
operator and Hermite preliminaries in
[O'Donnell, Servedio, and Tan](https://www.cs.cmu.edu/~odonnell/papers/fooling-gaussian-ptfs.pdf).

[Kane's Gaussian noise-sensitivity result](https://arxiv.org/abs/0912.2709)
provides dimension-independent bounds for polynomial threshold functions of fixed
degree. We have not established those assumptions for the model loss. Neither
result gives stationary reward stability for the adaptively selected, normalized,
BF16 reference above. The conditional moment proofs do not need those assumptions,
but do not prove improved reward, posterior calibration, convergence, or regret.
The Gaussian proposal law is not an ENNX posterior or surrogate covariance change.

## Numerical and cost gates

- A BF16 reference costs two bytes per weight: about **2.60 GB (2.42 GiB)** at
  1.3B weights, plus per-tensor statistics. The eight-evaluation correlated T4 run
  peaked at **14,763 MiB**. This is a narrowly fitting fixture, not a guarantee for
  larger batches or histories; see [memory measurements and limits](flame.md#precision-memory-and-audit).
- Normalize references once on initialization/acceptance, not every candidate.
  Generation adds reference reads and Gaussian arithmetic; history scoring still
  scans candidates times history times weights. Four candidates do not require
  four resident models or four forwards. Matched synthetic ask/tell medians were
  **905/211 ms correlated**, **847/148 ms independent Gaussian**, and **618/101 ms
  independent signs**, with the same forced-acceptance schedule. These timings
  isolate execution cost, not optimization effectiveness or speed equivalence.
- Replay must key Gaussian streams by seed, tensor, and coordinate, preserve paired
  streams and base/reference/scale versions, and reproduce scoring, materialization,
  and accepted-reference updates without an ever-growing seed chain. Exact GPU
  replay and extended reference tests across the 65,536-element tile boundary,
  FIFO/restart handling, and null filtering passed under Compute Sanitizer with
  zero errors; see [validation details](flame.md#bo-validation-status).
- Finite PRNG output, approximate Gaussian generation, FP32 reductions/arithmetic,
  and BF16 rounding are not the exact ideal lemmas. Recompute $q_l$ from rounded
  values, validate finite positive statistics, and measure actual BF16 changes.
  CPU/GPU rounding-envelope parity passed with **seven near-boundary mismatches**;
  this is not bitwise CPU/GPU equality or exact ideal-Gaussian sampling. CPU algebra
  checks alone do not validate a GPU generator or reduction implementation.
- Both full runs covered **1.3B weights / 81 tensors**, with eight evaluations,
  four candidates, and two history rows; best-checkpoint reloads passed at loss
  tolerance **1e-5**. From initial loss **8.43056**, correlated best loss was
  **8.3983822**, versus **8.3352203** for independent Gaussian. The latter was lower
  in this run; these tiny-fixture results establish neither a general ranking nor
  an optimization advantage for correlation. See [full results and limitations](flame.md#bo-validation-status).

The checker uses standard-library three-point Gaussian quadrature, with nodes
$-\sqrt3,0,\sqrt3$ and weights $1/6,2/3,1/6$. Its product rule integrates the
degree-four moment polynomials exactly in real arithmetic; floating-point checks
use tolerances. Small arbitrary references and endpoint/interior persistence
values check the derivations deterministically, not by Monte Carlo. This does not
test Gaussian tails, BF16 behavior, or optimization performance.
