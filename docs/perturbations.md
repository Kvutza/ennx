# Dense perturbation specification

Status: research specification with exact-arithmetic reference checks. The
production sampler and TuRBO radius controller are unchanged. Run
`python3 scripts/check_perturbation_math.py` to check the finite examples below.

## Scope and cost

Preserve ENNX's candidate ranking and real-evaluation budget. Candidate generation
must use a bounded descriptor and a fixed amount of work per weight. An extra
model forward is not required. The first comparison is independent dense signs
against correlated dense signs, with the same tensor scales and radii.

Signs are a starting point because their magnitude is fixed. Exact per-tensor
radius needs no reduction over each generated candidate. This is a computational
property, not a claim that signs optimize reward better than Gaussians.

The existing CUDA weight path already uses dense signs, not a RAASP coordinate
mask: `cuda/kernels/src/bf16.rs::bf16_seed`. The proposed work changes their scale
interpretation and optionally their joint distribution across proposals.

## Tensor-relative radius

For a fixed base model, tensor $W_l$ has $n_l$ entries. For each nonzero tensor,
compute its RMS scale once when establishing the base:

$$
s_l=\frac{\|W_l\|_F}{\sqrt{n_l}},\qquad
\Delta_{l,i}=r s_l T_{l,i},\qquad T_{l,i}\in\{-1,+1\}.
$$

Then, in real arithmetic, every candidate satisfies

$$
\|\Delta_l\|_F=r\|W_l\|_F.
$$

Proof: sum the squared entries and use $T_{l,i}^2=1$. This remains true for
correlated signs. Positive rescaling of a tensor rescales its perturbation by the
same factor, at fixed signs and radius. That is a property of the proposal, not a
claim of neural-network functional invariance.

An all-zero tensor requires an explicit positive absolute scale. Silently using
zero would freeze it; silently inserting an arbitrary floor changes the meaning
of the radius. Tiny tensors also require a documented application-level scale
policy. Per-tensor scales are not learned here. All candidate descriptors record
the base/scale version so replay cannot accidentally use a later base.

Start with radius supplied by the current controller. A radial distribution or
new controller is a separate experiment. If scales are recomputed after accepting
a new base, that is a full weight reduction once per accepted base, not free work.
It can potentially share the materialization pass; that needs implementation and
measurement. Keeping scales fixed longer instead changes their interpretation to
relative size at the reference base.

## Correlated sign law

Within one proposal family, generate an independent uniform reference sign field
$S$. For candidate $j$, generate independent flip bits $B_j$ with probability
$p_j$ of a flip, independently across weights and candidate streams:

$$
T_{j,l,i}=S_{l,i}(-1)^{B_{j,l,i}},\qquad
\rho_j=1-2p_j.
$$

A candidate needs a reference seed, flip seed, flip probability, radius, and
base/scale version. The reference field is regenerated per coordinate and is
fixed within the family. No parent chain is traversed. Each weight requires one
reference sign and one flip decision, rather than one sign in the baseline.
This is a bounded increase in generator work, not a measured timing claim.

When $p_j=1/2$, the child is an independent uniform sign field even conditional on
the reference. At $p_j=0$ or $1$, it is respectively the reference or its negative.
For $0<p_j<1$, every sign vector has positive conditional probability. Each
candidate changes every real-valued weight relative to the base when $r,s_l>0$;
two candidates can share many entries. The support is a sign cube, not every
continuous direction in weight space.

The reference is an ephemeral noise draw. This specification does not restrict
updates to a fixed low-dimensional span. It also does not yet specify how an
accepted direction becomes a new reference: recursively composing flip seeds
would violate the bounded-work requirement. Initial experiments must keep the
family reference fixed or start a fresh family explicitly.

The probability statements assume ideal independent streams. An implementation
needs domain-separated reference/flip streams keyed by tensor and local index,
a specified integer threshold for representable probabilities, and replay tests.
The existing hash function is not automatically certified by these probability
proofs.

## Correlation and distance identities

For a reference and child, $\mathbb E[S_iT_{j,i}]=\rho_j$. For distinct children
with independent flip streams,

$$
\mathbb E[T_{j,i}T_{k,i}]=\rho_j\rho_k.
$$

Consequently, for children of the same base, scales, and reference, with radii
$r_j,r_k$ and fixed positive metric weights $w_l$,

$$
\mathbb E\!\left[\sum_l w_l\|\Delta_{j,l}-\Delta_{k,l}\|_F^2\right]
=\sum_l w_l n_l s_l^2
\left(r_j^2+r_k^2-2r_jr_k\rho_j\rho_k\right).
$$

This formula is not valid for a candidate compared with itself or for reused flip
streams. Marginally, each child is uniform; correlation describes the joint law.

For an arbitrary fixed historical row $h$, condition on the reference $S$ and set
$d_i=W_i-h_i$, using the scale and metric weight of coordinate $i$'s tensor.
The actual squared-distance random variable is

$$
D=\sum_i w_i(d_i+r s_iT_i)^2.
$$

Its conditional moments are

$$
\begin{aligned}
\mathbb E[D\mid S]&=\sum_i w_i
  (d_i^2+2r s_i\rho S_i d_i+r^2s_i^2),\\
\operatorname{Var}(D\mid S)&=4r^2(1-\rho^2)
  \sum_i w_i^2s_i^2d_i^2.
\end{aligned}
$$

Proof: expand the square, use $T_i^2=1$, and use conditional independence of the
flip bits. These formulas require neither a model covariance nor gradients.

Expected distance can be scored from cached sums, but those sums cost a scan of
history and weights when the reference/base changes. More importantly, all seeds
with the same $r,\rho$ have the same expected distances. Replacing actual distance
with expectation removes their distinctions and can change neighbors, acquisition
values, and selection. This is not an approved scoring shortcut.

The current CUDA metric is the weighted squared distance between materialized
BF16 values, as implemented by `tile_distances`. Its scoring work remains
proportional to candidates times history times weights. The identities above
describe the proposed real-arithmetic law, not that quantized metric.

## Reward lemma to investigate

Fix the base, scales, radius, and evaluation convention. Let $F(S)$ be the actual
reward obtained from a sign field. On the uniform sign cube it has an expansion
in products of coordinates, with coefficient $\widehat F(A)$ for subset $A$.
This is an analysis of the reward function, not a stored weight basis.

For a uniform reference $S$ and its independently flipped child $T$,

$$
\operatorname{Cov}(F(S),F(T))
=\sum_{A\ne\varnothing}\rho^{|A|}\widehat F(A)^2.
$$

Proof: expand both functions. Independence and symmetry make products with
different coordinate subsets vanish; matching subsets contribute
$\rho^{|A|}$. This is the standard product-noise identity from Boolean function
analysis, not a new theorem. See
[Mossel, O'Donnell, and Oleszkiewicz (2010)](https://annals.math.princeton.edu/2010/171-1/p05).

For $0\leq\rho\leq1$, if at least a fraction $1-\eta$ of positive reward variance
comes from terms involving at most $q$ coordinates, reward correlation is at
least $(1-\eta)\rho^q$. Every weight can participate in such terms. This assumption
does not imply cheap optimization: even an arbitrary dense linear objective
has interaction degree one.

The identity averages over a uniform reference. Conditioning on a BO-selected
high-reward reference changes the distribution, so it does not provide a
conditional posterior or a convergence guarantee for the adaptive loop. A fresh
random family does not automatically exploit previously good directions either.

Gaussian rotation and threshold-stability results remain a comparison route,
particularly [Kane (2011)](https://arxiv.org/abs/0912.2709). They do not transfer
automatically to signs or tensor-normalized Gaussians. The invariance principle
requires additional assumptions about degree, influence, and moments.

## Cost and integration gates

| Operation | Independent signs | Correlated signs |
| --- | --- | --- |
| Candidate description | Seed, radius, base version | Two seeds, probability, radius, base version |
| Tensor scale setup | One reduction per reference base | Same |
| Real-arithmetic radius normalization | No per-candidate reduction | Same |
| Candidate materialization | One weight pass | One pass, extra flip stream |
| Auxiliary full-weight direction storage | None | None with fixed family reference |
| Exact current history scoring | Candidates x history x weights | Same asymptotic cost, larger generator constant |
| Real reward evaluations | Existing budget | Same budget |

Before a production change:

1. Specify family lifetime and reference refresh without growing state or losing
   the intended use of historical rewards.
2. Decide and test finite-precision behavior. Rounding can erase perturbations;
   forcing a neighboring representable value changes their norm and symmetry.
   The current `bf16_seed` forces such a move, so exact radius claims do not
   describe it. The T4 experiment needs an FP16 execution path.
3. Match scoring and evaluation to the same realized candidate. Neither clipping
   nor a switch to expected distances may be hidden inside an implementation.
4. Compare independent and correlated signs on T4 at identical forward budgets,
   history sizes, candidate counts, and scales. Measure materialization time,
   scoring time, memory, and actual reward per evaluation. Gaussian proposals can
   join that comparison when their generation and normalization costs are explicit.

The exact checker establishes finite-case algebra and catches mistaken claims
about correlation and expected-distance ranking. It is not a GPU benchmark,
evidence of better optimization, or a proof about a billion-weight model.
