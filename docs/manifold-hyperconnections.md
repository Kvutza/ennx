# Manifold-constrained residual streams

Status: direct Metal implementation and staged experiment contract. Single-Pass
scheduling remains a measured optimization after direct-path parity.

## Decision

Implement four residual streams across every attention and MoE residual update.
The five physical layers and their mHC predictors remain shared across both
feedback visits. Remove the projected-sigmoid boundary from this architecture.

This is a new model and checkpoint revision. Existing checkpoints retain their
declared transition and cannot be loaded into the mHC model.

Select the architecture with one categorical model identifier:

- `fbt-pisa1-residual1-v1`
- `fbt-pisa1-projected-boundary-v1`
- `fbt-pisa1-hc4-v1`
- `fbt-pisa1-mhc4-v1`

The stream count, projection shape, Sinkhorn iterations, coefficient bounds,
and residual-site placement are architecture constants. They do not become
continuous TOML controls. Architecture ablations use separate model identifiers
with separate checkpoint formats.

## State transition

For stream count `n = 4` and model width `d = 512`, each token carries
`X_l in R^(4 x 512)`. Every attention or MoE residual site computes

```text
(A_l, B_l, C_l) = H_l(X_l)
z_l = A_l X_l
y_l = F_l(z_l)
X_(l+1) = B_l X_l + C_l y_l
```

`A_l` has shape `1 x 4`, `B_l` has shape `4 x 4`, and `C_l` has
shape `4 x 1`. The attention and MoE functions remain 512-wide. Residual width
therefore grows without widening attention heads or experts.

The coefficient predictor normalizes the flattened 2,048-element stream state
and emits 24 raw coefficients:

```text
raw = alpha_l * RMSNorm(vec(X_l)) W_l + bias_l
A_l = sigmoid(raw_A) / sum(sigmoid(raw_A))
C_l = 2 sigmoid(raw_C)
B_l = Sinkhorn(exp(raw_B), 20 iterations)
```

Coefficient projection and Sinkhorn arithmetic use FP32. Residual streams use
FP16. The same coefficient path must serve teacher-forced scoring, serial
decoding, accepted-prefix verification, and candidate scoring.

## Base-model equivalence

The initial state replicates the embedding into all four streams. The static
coefficient initialization is

```text
A_l = [1/4, 1/4, 1/4, 1/4]
B_l = I
C_l = [1, 1, 1, 1]^T
alpha_l = 0
raw_lambda_l = 0
```

The corresponding static raw biases are equal across `A_l` and zero for `C_l`.
The normalized sigmoid makes equal `A_l` biases exactly uniform. The raw `B_l`
values are inactive while `lambda_l = 0`.

All streams remain equal under this setting. Their mean follows the ordinary
single-stream residual model exactly. The final readout uses the stream mean.

The production parameterization must contain this finite exact setting. A
finite Sinkhorn projection only approximates the identity when all entries are
strictly positive. Use an explicit transport interpolation

```text
B_l = (1 - lambda_l) I + lambda_l Sinkhorn(exp(raw_B))
lambda_l = clamp(raw_lambda_l, 0, 1)
```

with `lambda_l = 0` in the base-equivalent initialization and
`0 <= lambda_l <= 1` enforced by the operator. `lambda_l` is a declared model
coefficient, not an unconstrained matrix entry.

Every allowed `B_l` is doubly stochastic. It is nonnegative, its rows and
columns sum to one, and its spectral norm is at most one. Products of these
matrices remain doubly stochastic. This bounds residual transport across all
ten residual applications in a visit and across repeated visits. It does not
bound the complete nonlinear update, which also contains `F_l`, `A_l`, and
`C_l`.

## Parameter layout

The model has ten unique residual sites: attention and MoE in each of five
physical layers. Their parameters are shared across the two visits.

For each site, the dynamic projection has shape `2048 x 24` and contains
49,152 weights. The 24 static biases, three dynamic gates, and transport
interpolation coefficient bring the site to 49,180 parameters. Ten sites
contain 491,800 parameters.

The current global feedback state and gate matrices contain 524,288 parameters.
Replacing them with the mHC predictors reduces the model by 32,488 parameters
before alignment. Checkpoint tensors are stored per physical layer and residual
kind:

```text
mhc_attention_predictor [5, 2048, 24]
mhc_attention_bias      [5, 24]
mhc_attention_control   [5, 4]
mhc_moe_predictor       [5, 2048, 24]
mhc_moe_bias            [5, 24]
mhc_moe_control         [5, 4]
```

The first three controls independently scale the `A`, `B`, and `C` dynamic
slices. The fourth is `raw_lambda_l`. Padding may preserve candidate-row
alignment but is excluded from the search space and parameter count.

## Metal execution

The implemented mathematically direct path executes:

1. Update the four streams from the previous block output.
2. Predict the current `A`, `B`, and `C` coefficients.
3. Run 20 FP32 Sinkhorn row/column normalizations on the 4-by-4 matrix.
4. Mix the four streams into one 512-wide block input.
5. Run the existing attention or MoE operation.

This path is the numerical reference. It keeps coefficient timing identical
between training and inference.

After reference parity, add the V4.1 schedule. Site `l` consumes `A_(l-1)`
while producing `(A_l, B_l, C_l)`. Fuse residual update, coefficient prediction,
stream mixing, input RMS normalization, and FP16 conversion into one Metal
kernel. Keep the direct path for audits.

Single-Pass mHC has a lower-bound residual traffic of `(2n + 2)d` elements per
token and site. At `n = 4`, `d = 512`, 8,192 scorer rows, and twenty executed
sites across two visits, this is 1,600 MiB. The comparable single-stream
read/write lower bound is 320 MiB. One additional four-stream state buffer adds
24 MiB beyond a single-stream buffer. These are residual-path bounds rather
than complete-model traffic measurements.

No standalone mHC dispatch is allowed in the optimized decoder. The scalar
coefficient work is too small to justify a launch at each residual site.

## Verification

The CPU reference and Metal kernels must establish the following properties:

- exact single-stream equivalence at the declared base initialization;
- row and column sums of `B_l` within `2e-5` in FP32;
- nonnegative `A_l`, `B_l`, and `C_l` with declared bounds;
- spectral norm of each transport and their product at most `1 + 2e-4`;
- finite results for adversarial raw coefficients in `[-80, 80]`;
- scorer, serial decoder, and accepted-prefix token parity;
- direct and Single-Pass schedules measured separately because their
  coefficient timing differs.

Record per-site stream RMS, stream cosine matrix, transport singular values,
row/column residuals, and coefficient extrema. These diagnostics determine
whether the extra streams specialize or collapse.

## Experiment

Use four categorical architecture arms:

1. `residual1`: the ordinary one-stream residual model.
2. `projected_boundary`: the current projected-sigmoid boundary control.
3. `hc4`: four streams with unconstrained transport, serving as the instability
   control.
4. `mhc4`: four streams with manifold-constrained transport.

The repository contains matched configurations for every arm at 1M, 4M, 16M,
and 64M causal pretraining tokens under `examples/tuning/residual-*.toml`. They
contain no resource paths and share explicit model, reference, proposal, and
acquisition seeds.

Hold corpus order, initial model seed, proposal stream, candidate count,
objectives, and evaluation tasks constant within each paired replicate.
Replicate seeds measure variance; they are not search dimensions.

Each round compares the candidate and incumbent on the same rotating corpus
batch of 8,192 positions. Their paired causal-loss change drives acceptance.
The candidate also freely generates a 4,096-token code continuation in the
round. Mean target likelihood, worst-window likelihood, reconstruction,
repetition, and positional accuracy describe the generated trajectory and do
not select weights. Checkpoints after 128, 512, 2,048, and 8,192 rounds
correspond to 1M, 4M, 16M, and 64M candidate pretraining tokens. Temperature
remains 0.8 with fixed sampling seeds. Zero-temperature decoding collapses the
untrained model to one argmax token and is not a coherence measurement.

At every checkpoint report:

- held-out mean and worst-window NLL;
- exact byte reconstruction and positional accuracy;
- Rust parse and compile rates on the fixed code suite;
- finite-candidate fraction and objective spread;
- complete-round wall time and GPU time;
- stream specialization and transport diagnostics.

The first decision point is 4M pretraining tokens. Continue an arm to 16M when its
paired held-out NLL is no worse than `residual1`, its finite-candidate rate is
at least 99.9%, and its median complete round remains below 200 ms. Continue to
64M only when either held-out NLL or compile rate improves in every paired
replicate. Run ENNX-controller studies after selecting the residual
architecture.

## Evidence boundary

DeepSeek trained mHC with gradients. ENNX searches full model weights through
forward evaluations. The Birkhoff constraint transfers as a mathematical
invariant; DeepSeek's quality gains do not transfer as evidence for ENNX.

The dedicated mHC experiment reports a 0.021 final loss reduction against its 27B
baseline, gains on seven of eight reported downstream tasks, and a composite
gain magnitude near 1.6 versus nearly 3,000 for unconstrained HC. DeepSeek-V4
reports 6.7% training wall-time overhead after systems optimization. V4.1
reports that Single-Pass scheduling halves residual activation traffic and
causes negligible, unquantified performance degradation.

Sources: [mHC](https://arxiv.org/abs/2512.24880),
[DeepSeek-V4](https://arxiv.org/abs/2606.19348), and the
[DeepSeek-V4.1-Flash report](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash).
