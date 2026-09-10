# API

| Surface | Purpose |
| --- | --- |
| Python `ennx` | `ENN` model and `create_optimizer` ask/tell workflow |
| Rust `ennx::prelude` | Curated Rust API |
| `ennx.search` / `ennx::search` | Encoded parameter search and optimization |
| `ennx.experimental` / `ennx::experimental` | Experimental and accelerator APIs |
| `ennx.botorch`, `ennx.optuna`, `ennx.ax` | Python framework adapters |

Keep implementation helpers private. Promote experimental APIs only with
boundary, edge-case, and failure tests. Use matching concept names in Rust and
Python; backend support depends on the operation.

[Migration](../CHANGELOG.md) · [Integration contracts](interop.md)
