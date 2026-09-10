# Python Integrations

ENNX ships BoTorch, Optuna, and Ax adapters as default Python dependencies.
Importing `ennx` does not import those frameworks; importing the adapter module
does.

| Adapter | Import | Role |
| --- | --- | --- |
| BoTorch | `ennx.botorch.Model` | expose an existing `ENN` as a surrogate |
| BoTorch | `ennx.botorch.Sampler` | produce ENNX function draws |
| Optuna | `ennx.optuna.Sampler` | generate trials for a fixed continuous study |
| Ax | `ennx.ax.Node` | generate candidates for an Ax experiment |

Boundaries:

- Fixed continuous spaces are supported.
- Integer, categorical, conditional, fidelity, and multiobjective adapter paths
  are not supported unless the specific adapter rejects or handles them
  explicitly.
- Adapters do not provide Metal, OpenCL, or CUDA zero-copy interoperability.
- Pending parameters are excluded from new proposals. This is not fantasy
  conditioning.
- Restarting an adapter from existing observations is a warm restart, not a
  bitwise continuation of trust-region and RNG state.
- No framework settings are inserted into the algorithm config.

Run the installed-wheel integration checks with:

```sh
./ennx build
ENNX_WHEEL_PATH=dist/ennx-...whl ./ennx test --python
```

`./ennx dev` builds each supported ABI wheel and runs the same Python test suite
against each artifact.
