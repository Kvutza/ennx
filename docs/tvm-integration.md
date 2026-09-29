# Apache TVM integration boundary

## Decision

Use TIRx as an optional CUDA kernel compiler and search surface. Do not move
ENNX's search loop or resident runtime into TVM. The first targets are the QKV,
attention-output, gate/up, down, and readout matrix operations. Keep KDA state,
routing, weight overlays, candidate evaluation, verification, and objectives
under the Rust runtime until their state and exactness contracts are explicit.

`forward_program::ModelProgram` is a data-only semantic graph with tensor
lifetimes, symbolic extents, exactness, and fusion boundaries. It has no
compiled TVM backend. `./ennx model inspect --backend tvm-cuda` and
`--backend tvm-metal` expose contraction plans with `coverage-kind` set to
`lowering-plan`. Their schedule coverage does not establish compilation or
execution; no TVM lowering has been measured yet. The production path
is the native Rust/Metal or CUDA-Oxide implementation. The CLI builds workers
through Buck2 or Bazel and starts them as processes; it has no TVM compiler,
TVM runtime, or TVM-FFI dependency today.

`./ennx model tirx-probe` checks for both compiler and runtime libraries under
`TVM_LIBRARY_PATH` (or `$TVM_HOME/build`) and reports `unavailable` when either
is absent. Finding both libraries reports only their presence: the probe does
not verify `tirx.build` registration, CUDA target support, or kernel behavior.

The published TIRx authoring and build path requires Python. ENNX does not add
Python to its runtime or CLI path. A prototype proceeds only after the Rust FFI
can construct and compile the required `PrimFunc`; otherwise TVM remains an
offline comparison compiler.

## Appropriate first slice

Author one stateless projection in TIRx, compile it to a TVM runtime module,
and call it through TVM-FFI's Rust API while ENNX retains its buffers and weight
overlay. TIRx exposes explicit thread roles, memory placement, tensor layouts,
CUDA intrinsics, and tile dispatch, so it fits the schedule work that currently
lives directly in CUDA-Oxide kernels. It also leaves a path to task composition
and megakernels after individual operations are correct.

Relax and BYOC belong later, if graph-level fusion across several proven
kernels saves measurable work. They are unnecessary for the first projection
benchmark and would add a second graph runtime before its value is known. TVM
must remain an optional compiler dependency. Planned schedules must stay
separate from compiled runtime coverage.

Relax supports symbolic shapes, but memory planning for dynamic shapes uses
explicit variable bounds. For initial deployment, bound token and routed-row
extents to the same limits ENNX validates. Relax's VM is the flexible execution
boundary; an ahead-of-time artifact is a better fit for fixed, bounded
signatures, but should only be selected after confirming the exact Relax
executor and target combination. Preserve ENNX's long-lived resident buffers
and call TVM for device work rather than moving the optimization loop into a
per-operation TVM VM.

TVM's target code generators document both CUDA and Metal. That establishes
backend codegen capability; it does not establish matching the Metal and
CUDA-Oxide kernels' numerical, synchronization, allocation, or performance
contracts. TVM's external-codegen path links generated runtime modules into a
TVM executable. Integrating that runtime wholesale would cross ENNX's current
process/build boundary. A narrow generated-kernel ABI is less invasive; if
runtime modules are needed later, evaluate Apache TVM-FFI's Rust bindings and
ABI separately from the TVM compiler integration.

## Suggested gate before adding an adapter

1. Make a static projection and its buffer/overlay inputs explicit in
   the model contract, including bounded symbolic sizes.
2. Compile it to Metal and CUDA, call it through the native owner of buffers,
   and compare full outputs against ENNX reference kernels over boundary
   shapes, perturbation seeds, and quantized weight cases.
3. Measure the whole evaluator call, including launch and synchronization,
   against native kernels. Add the adapter only if parity passes and the
   measured end-to-end path improves.

Upstream references (Apache TVM `main`, consulted 2026-10-03):

- [TVM architecture: Relax, TIR and target translation](https://github.com/apache/tvm/blob/main/docs/arch/index.rst)
- [TVM code generation: CUDA, Metal, and runtime modules](https://github.com/apache/tvm/blob/main/docs/arch/codegen.rst)
- [Relax external codegen / BYOC](https://github.com/apache/tvm/blob/main/docs/arch/external_library_dispatch.rst)
- [Relax symbolic-shape tutorial](https://github.com/apache/tvm/blob/main/docs/deep_dive/relax/learning.rst)
- [Dynamic-shape bounds for Relax memory planning](https://github.com/apache/tvm/blob/main/include/tvm/relax/transform.h)
- [Apache TVM-FFI Rust and C ABI status](https://github.com/apache/tvm-ffi)
- [TIRx overview and kernel/compiler boundary](https://tvm.apache.org/docs/tirx/overview.html)
- [TIRx installation and Python authoring requirement](https://tvm.apache.org/docs/tirx/install.html)
- [TIRx CUDA programming guide](https://tvm.apache.org/docs/tirx/native_basics.html)
