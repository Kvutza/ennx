"""Generated from Cargo metadata by tools/build-parity --sync. Do not edit."""

ENNX_CRATES = ["anstream", "anstyle", "deser", "memmap2", "ndarray", "rand", "rand_distr", "rayon", "sha2", "sobol", "thiserror"]

ENNX_TEST_CRATES = ["anstream", "anstyle", "deser", "memmap2", "ndarray", "numpy", "rand", "rand_chacha", "rand_distr", "rayon", "sha2", "sobol", "tempfile", "thiserror"]

ENNX_PROGRAMS = [
    {
        "version": "0.2.0",
        "name": "turbo-enn",
        "crate": "turbo_enn",
        "root": "src/bin/turbo-enn/main.rs",
        "crate_deps": ["anstream", "anstyle", "deser", "memmap2", "ndarray", "rand", "rand_distr", "rayon", "sha2", "sobol", "thiserror"],
        "local_deps": ["bpann", "ennx", "wire"],
        "required_features": [],
    },
    {
        "version": "0.2.0",
        "name": "turbo-enn-worker",
        "crate": "turbo_enn_worker",
        "root": "src/bin/turbo-enn-worker/main.rs",
        "crate_deps": ["anstream", "anstyle", "deser", "memmap2", "ndarray", "rand", "rand_distr", "rayon", "sha2", "sobol", "thiserror"],
        "local_deps": ["bpann", "ennx", "wire"],
        "required_features": [],
    },
    {
        "version": "0.2.0",
        "name": "cuda_trial",
        "crate": "cuda_trial",
        "root": "examples/cuda_trial.rs",
        "crate_deps": ["anstream", "anstyle", "deser", "memmap2", "ndarray", "numpy", "rand", "rand_chacha", "rand_distr", "rayon", "sha2", "sobol", "tempfile", "thiserror"],
        "local_deps": ["bpann", "ennx", "wire"],
        "required_features": ["cuda"],
    },
    {
        "version": "0.2.0",
        "name": "knn_frontier",
        "crate": "knn_frontier",
        "root": "examples/knn_frontier.rs",
        "crate_deps": ["anstream", "anstyle", "deser", "memmap2", "ndarray", "numpy", "rand", "rand_chacha", "rand_distr", "rayon", "sha2", "sobol", "tempfile", "thiserror"],
        "local_deps": ["bpann", "ennx", "wire"],
        "required_features": ["metal"],
    },
    {
        "version": "0.2.0",
        "name": "posterior_frontier",
        "crate": "posterior_frontier",
        "root": "examples/posterior_frontier.rs",
        "crate_deps": ["anstream", "anstyle", "deser", "memmap2", "ndarray", "numpy", "rand", "rand_chacha", "rand_distr", "rayon", "sha2", "sobol", "tempfile", "thiserror"],
        "local_deps": ["bpann", "ennx", "wire"],
        "required_features": [],
    },
    {
        "version": "0.2.0",
        "name": "stable_api",
        "crate": "stable_api",
        "root": "examples/stable_api.rs",
        "crate_deps": ["anstream", "anstyle", "deser", "memmap2", "ndarray", "numpy", "rand", "rand_chacha", "rand_distr", "rayon", "sha2", "sobol", "tempfile", "thiserror"],
        "local_deps": ["bpann", "ennx", "wire"],
        "required_features": [],
    },
    {
        "version": "0.2.0",
        "name": "trial_bench",
        "crate": "trial_bench",
        "root": "examples/trial_bench.rs",
        "crate_deps": ["anstream", "anstyle", "deser", "memmap2", "ndarray", "numpy", "rand", "rand_chacha", "rand_distr", "rayon", "sha2", "sobol", "tempfile", "thiserror"],
        "local_deps": ["bpann", "ennx", "wire"],
        "required_features": [],
    },
]

BPANN_CRATES = ["deser", "memmap2", "ndarray", "rand", "rayon", "thiserror"]

BPANN_TEST_CRATES = ["deser", "memmap2", "ndarray", "rand", "rand_chacha", "rayon", "tempfile", "thiserror"]

BPANN_PROGRAMS = [
]

CORPUS_CRATES = ["arrow-array", "deser", "futures", "opendal", "parquet", "parquet_opendal", "sha2", "tokio"]

CORPUS_TEST_CRATES = ["arrow-array", "deser", "futures", "opendal", "parquet", "parquet_opendal", "sha2", "tokio"]

CORPUS_PROGRAMS = [
    {
        "version": "0.2.0",
        "name": "ennx-corpus",
        "crate": "ennx_corpus",
        "root": "src/main.rs",
        "crate_deps": ["arrow-array", "deser", "futures", "opendal", "parquet", "parquet_opendal", "sha2", "tokio"],
        "local_deps": ["wire"],
        "required_features": [],
    },
]

DEV_CLI_CRATES = ["clap", "deser", "ndarray", "rand", "sha2"]

DEV_CLI_TEST_CRATES = ["clap", "deser", "ndarray", "rand", "sha2"]

DEV_CLI_PROGRAMS = [
    {
        "version": "0.2.0",
        "name": "build-graph",
        "crate": "build_graph",
        "root": "src/bin/build-graph.rs",
        "crate_deps": ["clap", "deser", "ndarray", "rand", "sha2"],
        "local_deps": ["ennx", "ptx-synth", "wire"],
        "required_features": [],
    },
    {
        "version": "0.2.0",
        "name": "ennx",
        "crate": "ennx",
        "root": "src/bin/ennx/main.rs",
        "crate_deps": ["clap", "deser", "ndarray", "rand", "sha2"],
        "local_deps": ["ennx", "ptx-synth", "wire"],
        "required_features": [],
    },
]

MODAL_RUNNER_CRATES = ["modal-rs", "tempfile", "tokio"]

MODAL_RUNNER_TEST_CRATES = ["modal-rs", "tempfile", "tokio"]

MODAL_RUNNER_PROGRAMS = [
    {
        "version": "0.2.0",
        "name": "ennx-modal",
        "crate": "ennx_modal",
        "root": "src/main.rs",
        "crate_deps": ["modal-rs", "tempfile", "tokio"],
        "local_deps": [],
        "required_features": [],
    },
]

ENNX_PY_CRATES = ["ndarray", "numpy", "pyo3", "rand"]

ENNX_PY_TEST_CRATES = ["ndarray", "numpy", "pyo3", "rand"]

ENNX_PY_PROGRAMS = [
]

WIRE_CRATES = ["deser", "deser-json", "deser-toml", "deser-value"]

WIRE_TEST_CRATES = ["deser", "deser-json", "deser-toml", "deser-value"]

WIRE_PROGRAMS = [
]

PTX_SYNTH_CRATES = []

PTX_SYNTH_TEST_CRATES = []

PTX_SYNTH_PROGRAMS = [
    {
        "version": "0.2.0",
        "name": "ptx-synth",
        "crate": "ptx_synth",
        "root": "src/main.rs",
        "crate_deps": [],
        "local_deps": ["ptx-synth"],
        "required_features": [],
    },
]

RUST_EDITION = "2024"
