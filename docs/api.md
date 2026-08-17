# API policy

ENNX has three public layers:

- `ennx::prelude` is the stable Rust API. New stable Rust examples should import
  from here first.
- `ennx.experimental` and `ennx::experimental` are staging areas for native,
  packed, accelerator, quantization, and low-level research hooks.
- Internal modules are implementation detail. They can stay public while the
  crate is still maturing, but new user-facing code should not depend on them.

The Python API follows the same rule. Top-level `ennx` should stay small and
boring. Anything that exposes native layout, packed data, hardware frontier
behavior, or low-level quantization starts in `ennx.experimental`.

Experimental native workflows that are exposed to Python should also have a
matching Rust surface in `ennx::experimental` unless the feature is purely
Python orchestration. Use the same concept names on both sides. For example,
resident parameter search is `ParamBlock`, `SearchState`, `Proposal`, and
`Proposals` in Rust, and `ParamBlock`, `SearchState`, and `Proposals` in
Python. Hardware-specific engines, storage dtypes, and kernel names stay behind
that API.

## Promotion rule

Move an API from experimental to stable only when it has:

- a clear name and shape;
- a Python contract test when Python exposes it;
- a Rust boundary test when Rust exposes it;
- deterministic tests for edge cases and failure behavior;
- no dependency on accidental module layout.

## Boundary rule

Do not add new top-level exports just because a helper is useful internally.
Use one of these instead:

- stable user workflow: `ennx::prelude` or top-level Python `ennx`;
- unstable user workflow: `experimental`;
- implementation helper: private module or crate-local item.

Existing root Rust exports are legacy surface area. Shrink them in small
compatible steps, with tests that prove the supported import path still works.
