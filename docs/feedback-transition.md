# Feedback transition contract

Status: architectural analysis for the active two-pass FBT/PISA/MoE model.
This contract determines what follows from the model equations. It makes no
model-quality claim.

## Role

Let `F` be the five-layer shared Transformer stack and `h` the raw residual
state after its first visit. The second visit computes `F(T(h))`. With no
additional requirement, repeated shared computation gives `T(h) = h`.

The active projected-sigmoid path computes

```text
n = RMSNorm(h)
u = W_state h
g = sigmoid(W_gate n)
T(h) = g * u
```

The multiplication is coordinatewise. `T(h)` becomes the raw residual state
for the second visit. The second visit separately normalizes that state before
attention. Gate magnitude therefore affects the balance between the carried
residual and the new attention update; it is not erased by normalization.

This operator is a projected feature modulator. It is not a retain-versus-update
gate because `h` is not one of the alternatives in its equation.

## Required properties

1. **Exact carry.** A finite parameter setting must satisfy `T(h) = h` for every
   finite `h`.
2. **Identity initialization.** A learned boundary must begin at exact carry.
   The first training step should add a boundary transformation rather than
   repair a random replacement of the residual state.
3. **Named semantics.** A feature modulator, a retain/replace mixture, and a
   recurrent cell are different operators. Configuration and documentation
   must identify the implemented operation.
4. **Raw-state continuity.** Normalization supplies the pre-attention view. It
   must not silently replace the raw residual state.
5. **Smooth locality.** Small state or weight changes must produce small
   transition changes. This supports gradient training and ENNX's local
   weight-space search.
6. **Path parity.** Teacher-forced scoring, candidate scoring, serial decoding,
   and accepted-prefix verification must implement the same transition in the
   same precision at the same pass boundary.
7. **Current cost envelope.** A learned transition may use two width-by-width
   projections. A third projection or a new synchronization point requires a
   separate systems argument.

## Deductions

The current operator contains exact carry:

```text
W_gate = 0
W_state = 2I
g = 0.5
T(h) = 0.5 * 2Ih = h
```

More generally, it contains every linear boundary map `A`: set `W_gate = 0`
and `W_state = 2A`. The operator therefore cannot be rejected for lacking an
identity or linear-map representation.

The active normal, Xavier, and patterned initializers do not use the identity
setting. With a zero-mean gate projection, `g` begins near one half while
`W_state` is random. The first boundary consequently replaces the first pass's
residual basis before training has assigned that replacement a role. This is an
initialization defect independent of downstream benchmark results.

The sigmoid does not express a probability of retaining `h`. Values near zero
suppress `W_state h`; values near one expose it. A retain/replace interpretation
requires the highway equation

```text
u = W_state h
g = sigmoid(W_gate RMSNorm(h))
T(h) = (1 - g) * h + g * u
```

This also contains exact carry, with `W_state = I` for any gate value. It
imposes coordinatewise interpolation between the retained and proposed states.
That constraint is useful only when retain/replace is part of the intended
model semantics.

A two-way softmax expresses the same mixture because

```text
softmax(a, b)[replace] = sigmoid(b - a)
```

Softmax over hidden coordinates defines a different operator. It forces
unrelated features to compete for fixed mass and cannot express retention. It
does not satisfy the current feedback role.

A GRU adds a reset gate to construct a recurrent candidate. ENNX already runs
the complete shared Transformer stack to construct the next representation.
No stated requirement assigns work to an additional reset mechanism, so a GRU
does not follow from the architecture contract.

## Decision

The current model specification states that the same five-layer stack receives
two visits. It does not state that an inter-pass operator must select features,
replace state, or implement an independent recurrent cell. The architecture
entailed by that specification is the identity transition:

```text
F(F(x))
```

Use `identity` for the first coherent base-policy checkpoint. This preserves
the first visit exactly, adds no parameters or kernels to the forward path, and
gives recurrent depth its literal compositional meaning.

Keep projected sigmoid as a separate feature-modulation hypothesis. Before it
is used in a new training run, initialize `W_state = 2I` and `W_gate = 0`. Its
name and documentation must describe modulation rather than retention.

The full multi-stream experiment replaces this boundary operator with
manifold-constrained residual transport at every attention and MoE update. Its
state, parameter, kernel, and measurement contracts are defined in
[Manifold-constrained residual streams](manifold-hyperconnections.md). The
single-stream identity checkpoint remains the matched base architecture for
that experiment.

Add a highway transition only after adopting an explicit retain/replace
requirement. Use a sigmoid implementation; a two-way softmax is a redundant
parameterization. Do not add coordinate softmax or a GRU under the present
contract.

## Scope

This decision applies to the active 512-wide, five-layer, two-visit model in
[`fbt_moe.rs`](../rust/crates/ennx/src/fbt_moe.rs), its scorer, and its decoder.
The older token-conditioned primitive in
[`fbt_metal.rs`](../rust/crates/ennx/src/fbt_metal.rs) has a different state
contract and does not define this model.

More than two visits introduces a separate stability question. A variable-depth
model must bound the repeated map's Jacobian or supply a convergence or halting
rule. A pass embedding may identify the recurrent step; it must remain separate
from the transition's state semantics.

The recurrent stack and step signal in the
[Universal Transformer](https://arxiv.org/abs/1807.03819), fixed-point analysis
in [Deep Equilibrium Models](https://arxiv.org/abs/1909.01377), and exact
reference preservation in
[Source-Centered State Evolution](https://arxiv.org/abs/2607.27656) address the
same structural concerns. The ENNX decision above follows from its active
equations and declared two-visit role.
