# API

| API | Use |
| --- | --- |
| Python `ennx` | Fit an `ENN` model or optimize with `create_optimizer` |
| Rust `ennx::prelude` | Models, optimizers, acquisition functions, and index types |
| `ennx.search` / `ennx::search` | Generate and evaluate candidates stored as low-bit parameters |
| `ennx.experimental` / `ennx::experimental` | Quantization, GPU buffers, and experimental model support |
| `ennx.botorch`, `ennx.optuna`, `ennx.ax` | Use ENNX with Python optimization libraries |

Optimizers use `ask` to propose a trial and `tell` to record its result.
GPU support varies by operation; Rust callers can check `ennx::capability::matrix()`.

[API changes](../CHANGELOG.md) · [Python integrations](interop.md)
