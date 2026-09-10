# Nearest-neighbor experiments

Run `./ennx tune knn tune/knn.toml` from the repository root.
Configuration is TOML, deserialized with Serde; schema version 1 is required.
Relative output paths are relative to the repository root. Output directories
are created automatically and an existing output CSV is overwritten.

`[knn]` requires `output`, positive `rounds`, and non-empty `points`.
Use `[[knn.points]]` tables or an array of inline tables. Each point has a unique
`name` (letters, digits, underscores, hyphens, or periods) and positive integer
`rows`, `queries`, `dims`, and `k`. Optional `[knn.defaults]` supplies dimensions
omitted from individual points. Explicit point values override defaults.

Comments, literal and escaped strings, multiline arrays, dotted keys, and
numeric separators use standard TOML syntax. Unknown fields and duplicate keys
are errors. All points are validated before starting the benchmark: `k` cannot
exceed `rows` or 2048, and the runner's distance-matrix limit is 256 MiB.

The configuration describes explicit benchmark cases; it does not yet expose
adaptive search, backend selection, or correctness thresholds. The existing
runner chooses available implementations and reports timing, recall, and error.

# Resident proposal experiments

Run `./ennx tune proposal tune/proposal.toml` from the repository root.
This workflow exercises resident weight-candidate search through the `trial_bench`
example and records raw per-round samples plus resolved settings.

`[proposal]` requires `output`, positive `elements`, `history`, `candidates`,
and `rounds`, and a non-empty `encoding`. `device`, `acquisition`, `neighbors`,
`edited_parameters`, `length`, `beta`, `seed`, `warmup`, and `memory_budget_mib`
are optional and validated before the benchmark starts. The CLI rejects unknown
fields and saves the captured benchmark output to the requested output file.

The current benchmark measures proposal, materialization, row readback,
evaluation, and update stages for a single resident candidate cycle. It also
records accounted transfer bytes, host allocation counts, synchronization
counts, peak accounted memory, selected backend, and source/toolchain identity.
These are accounted metrics, not device-driver counters or allocator internals.

The checked-in `tune/proposal_1b.toml` file is the explicit one-billion-element
scale check. It uses the same preflight memory-budget validation as the CLI, so
the run is rejected before launch if the estimate exceeds the budget.
