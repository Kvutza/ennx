# Python Integrations

BoTorch, Optuna, and Ax are installed with ENNX. They load when their adapters
are imported.

| Import | Use |
| --- | --- |
| `ennx.botorch.Model` | Use an existing `ENN` as a BoTorch model |
| `ennx.botorch.Sampler` | Sample joint function values from ENNX |
| `ennx.optuna.Sampler` | Propose Optuna trials |
| `ennx.ax.Node` | Propose Ax trials |

BoTorch accepts CPU `float32` and `float64` tensors, query batches, and multiple
outputs. It does not support input gradients, posterior transforms, bounded
outputs, or fantasy conditioning. Evaluate acquisition functions on candidate
sets; gradient-based optimization is unavailable.

Optuna and Ax support a fixed continuous search space and one objective.
Log-scaled parameters are supported; integers, categories, conditional spaces,
and stepped parameters are not. Ax also rejects constraints and fidelity
parameters. Use one process to generate trials, with `n_jobs=1` in Optuna.

Pending points are excluded from new proposals without assigning them outcomes.
A new adapter can reuse completed observations, but it does not restore the
previous random-number or trust-region state. The adapters do not share GPU
buffers with the frameworks.

`./ennx dev` tests all three integrations against each Python wheel.
