use super::*;
use crate::{
    Action, Cli, DataAction, OptAction, TargetAction, TensorAction, UtilAction, completion,
    cuda_track,
};
use clap::Parser;
use clap::error::ErrorKind;

fn parse(args: &[&str]) -> Result<Action, clap::Error> {
    Cli::try_parse_from(std::iter::once("ennx").chain(args.iter().copied())).map(|cli| cli.action)
}

#[test]
fn ordinary_options() {
    assert_eq!(
        parse(&["tensor", "inspect", "weights.safetensors"]).unwrap(),
        Action::Tensor {
            command: TensorAction::Inspect {
                file: "weights.safetensors".into(),
                json: false,
            },
        }
    );
    assert_eq!(
        parse(&["eval", "suite.toml"]).unwrap(),
        Action::Eval {
            config: "suite.toml".into(),
            output: None,
        }
    );
    assert_eq!(
        parse(&["eval", "suite.toml", "--output", "run.jsonl"]).unwrap(),
        Action::Eval {
            config: "suite.toml".into(),
            output: Some("run.jsonl".into()),
        }
    );
    assert_eq!(
        parse(&["tune", "knn.toml"]).unwrap(),
        Action::Tune {
            config: "knn.toml".into(),
            prepare: false,
        }
    );
    assert_eq!(
        parse(&["tune", "proposal.toml"]).unwrap(),
        Action::Tune {
            config: "proposal.toml".into(),
            prepare: false,
        }
    );
    assert_eq!(
        parse(&[]).unwrap_err().kind(),
        ErrorKind::DisplayHelpOnMissingArgumentOrSubcommand
    );
    assert_eq!(
        parse(&["--help"]).unwrap_err().kind(),
        ErrorKind::DisplayHelp
    );
    assert_eq!(
        parse(&["--version"]).unwrap_err().kind(),
        ErrorKind::DisplayVersion
    );
}

#[test]
fn menu_dispatch() {
    assert_eq!(parse(&["menu"]).unwrap(), Action::Menu);
    assert_eq!(
        parse(&["completion", "zsh"]).unwrap(),
        Action::Completion {
            shell: completion::Shell::Zsh,
        }
    );
    assert_eq!(
        parse(&["completion", "bash"]).unwrap(),
        Action::Completion {
            shell: completion::Shell::Bash,
        }
    );
    assert_eq!(
        parse(&["completion", "fish"]).unwrap(),
        Action::Completion {
            shell: completion::Shell::Fish,
        }
    );
    assert_eq!(
        parse(&["util", "completion", "zsh"]).unwrap(),
        Action::Util {
            command: UtilAction::Completion {
                shell: completion::Shell::Zsh,
            },
        }
    );
}

#[test]
fn opt_dispatch() {
    assert_eq!(
        parse(&["target", "cuda", "inspect"]).unwrap(),
        Action::Target {
            command: TargetAction::Cuda {
                command: cuda_track::Action::Inspect,
            },
        }
    );
    assert_eq!(
        parse(&["opt", "run", "exp.toml", "--prepare"]).unwrap(),
        Action::Opt {
            command: OptAction::Run {
                config: "exp.toml".into(),
                prepare: true,
            },
        }
    );
    assert_eq!(
        parse(&["opt", "catalog", "--json"]).unwrap(),
        Action::Opt {
            command: OptAction::Catalog {
                config: None,
                out: None,
                json: true,
            },
        }
    );
    assert_eq!(
        parse(&["data", "inspect", "weights.safetensors"]).unwrap(),
        Action::Data {
            command: DataAction::Inspect {
                file: "weights.safetensors".into(),
                json: false,
            },
        }
    );
}

#[test]
fn prepare_flag() {
    assert_eq!(
        parse(&["tune", "search.toml", "--prepare"]).unwrap(),
        Action::Tune {
            config: "search.toml".into(),
            prepare: true
        },
    );
}

#[test]
fn catalog_options() {
    assert_eq!(
        parse(&["catalog"]).unwrap(),
        Action::Catalog {
            config: None,
            out: None,
            json: false,
        }
    );
    assert_eq!(
        parse(&["catalog", "base.toml", "--out", "catalog", "--json"]).unwrap(),
        Action::Catalog {
            config: Some("base.toml".into()),
            out: Some("catalog".into()),
            json: true,
        }
    );
}

#[test]
fn malformed_work() {
    for args in [
        vec!["ci"],
        vec!["deps", "--check"],
        vec!["wheel"],
        vec!["eval"],
        vec!["tune"],
        vec!["tune", "--config"],
        vec!["tune", "other", "knn.toml"],
        vec!["tune", "a", "b"],
        vec!["--help", "unexpected"],
    ] {
        assert!(parse(&args).is_err(), "accepted {args:?}");
    }
}

#[test]
fn experiment_dispatch() {
    assert_eq!(experiment_kind("version=1\n[knn]"), Ok(ExperimentKind::Knn));
    assert_eq!(
        experiment_kind(
            "version=1\n[proposal]\noutput='x'\nelements=1\nhistory=1\ncandidates=1\nrounds=1"
        ),
        Ok(ExperimentKind::Proposal)
    );
    assert_eq!(
        experiment_kind("version=1\n[proposal]\ndistribution='gaussian'"),
        Ok(ExperimentKind::TurboEnn)
    );
    assert_eq!(
        experiment_kind("version=1\nacquisition='thompson'"),
        Ok(ExperimentKind::TurboEnn)
    );
    assert_eq!(experiment_kind("version=1"), Ok(ExperimentKind::TurboEnn));
    assert!(experiment_kind(
        "version=1\n[knn]\n[proposal]\noutput='x'\nelements=1\nhistory=1\ncandidates=1\nrounds=1"
    )
    .is_err());
    let identity = "\nmodel='fbt-pisa1-legacy-v1'\ncorpus='stack-v3-python-pilot-v1'";
    assert!(is_pretrain(&format!("version=2\nexperiment='pretrain'{identity}")).unwrap());
    assert!(!is_pretrain("version=2\nexperiment='end-to-end'").unwrap());
}

#[test]
fn knn_config2() {
    let config = parse_knn(
        r#"
version = 1
[knn]
output = '/tmp/knn#results.csv' # A hash inside a string is not a comment.
rounds = 3
points = [
    { name = "ann", rows = 1_024 },
    { name = "dim", rows = 4_096, queries = 16, dims = 64, k = 8 },
]
[knn.defaults]
queries = 32
dims = 16
k = 10
"#,
    )
    .unwrap();
    assert_eq!(
        config,
        KnnTuneConfig {
            output: "/tmp/knn#results.csv".into(),
            rounds: 3,
            points: vec!["ann:1024:32:16:10".into(), "dim:4096:16:64:8".into()],
        }
    );
}

#[test]
fn knn_running() {
    let valid = r#"
version = 1
[knn]
output = "x"
rounds = 1
points = [{ name = "a", rows = 10, queries = 1, dims = 1, k = 2 }]
"#;
    assert!(parse_knn(valid).is_ok());
    for (from, to, expected) in [
        ("version = 1", "version = 2", "version"),
        ("rounds = 1", "rounds = 0", "positive rounds"),
        ("rounds = 1", "round = 1", "unknown field"),
        ("rounds = 1", "rounds = 1\nrounds = 2", "already defined"),
        ("rows = 10", "rows = 0", "positive"),
        ("rows = 10", "rows = -1", "invalid value"),
        ("rows = 10", "rows = 1", "k must not exceed"),
        ("dims = 1", "dimension = 1", "unknown field"),
        ("dims = 1,", "", "dims must be supplied"),
        ("queries = 1", "queries = 100_000_000", "256 MiB"),
        ("dims = 1", "dims = 9_223_372_036_854_775_807", "overflows"),
        ("name = \"a\"", "name = \"a:b\"", "point name"),
        ("output = \"x\"", "output = \"\"", "non-empty"),
        ("k = 2", "k = 2049", "k must not exceed"),
        (
            "version = 1",
            "version = 1\nunexpected = true",
            "unknown field",
        ),
    ] {
        let text = valid.replace(from, to);
        let error = parse_knn(&text).unwrap_err();
        assert!(error.contains(expected), "{text}: {error}");
    }
}

#[test]
fn array_names() {
    let text = r#"
version = 1
[knn]
output = "x.csv"
rounds = 2
[knn.defaults]
rows = 1_024
queries = 32
dims = 16
k = 10
[[knn.points]]
name = "small"
[[knn.points]]
name = "large"
rows = 8_192
"#;
    assert_eq!(
        parse_knn(text).unwrap().points,
        ["small:1024:32:16:10", "large:8192:32:16:10"]
    );
    assert!(
        parse_knn(&text.replace("large", "small"))
            .unwrap_err()
            .contains("unique")
    );
    assert!(
        parse_knn(&text.replace("rows = 1_024", "row = 1_024"))
            .unwrap_err()
            .contains("unknown field")
    );
}

#[test]
fn parses_config() {
    let config = parse_config(
        r#"
version = 1
[proposal]
output = "proposal.csv"
elements = 1024
history = 8
candidates = 16
rounds = 4
warmup = 2
device = "metal"
encoding = "fp4"
acquisition = "thompson"
neighbors = 4
edited_parameters = 6
seed = 99
length = 0.75
beta = 1.25
memory_budget_mib = 16
"#,
    )
    .unwrap();
    assert_eq!(
        config,
        ProposalTuneConfig {
            output: "proposal.csv".into(),
            elements: 1024,
            history: 8,
            candidates: 16,
            rounds: 4,
            warmup: 2,
            device: "metal".into(),
            encoding: "fp4".into(),
            acquisition: "thompson".into(),
            neighbors: 4,
            edited_parameters: 6,
            seed: 99,
            length: 0.75,
            beta: 1.25,
            memory_budget_mib: Some(16),
        }
    );
}

#[test]
fn parses_budget() {
    let config = parse_config(
        r#"
version = 1

[proposal]
output = "dist/proposal-1b.csv"
elements = 1_000_000_000
history = 8
candidates = 8
rounds = 1
warmup = 0
device = "auto"
encoding = "int4"
acquisition = "ucb"
neighbors = 8
edited_parameters = 64
length = 0.8
beta = 1.0
seed = 7
memory_budget_mib = 8_192
"#,
    )
    .unwrap();
    assert_eq!(config.output, "dist/proposal-1b.csv");
    assert_eq!(config.elements, 1_000_000_000);
    assert_eq!(config.edited_parameters, 64);
    assert_eq!(config.memory_budget_mib, Some(8_192));
    let estimate = estimate_bytes(&config).unwrap();
    let budget = config
        .memory_budget_mib
        .unwrap()
        .checked_mul(1024 * 1024)
        .unwrap();
    assert!(
        estimate <= budget,
        "proposal estimate {estimate} exceeds budget {budget}"
    );
}

#[test]
fn rejects_running() {
    let valid = r#"
version = 1
[proposal]
output = "proposal.csv"
elements = 1024
history = 8
candidates = 16
rounds = 4
device = "metal"
encoding = "fp4"
acquisition = "thompson"
neighbors = 4
edited_parameters = 6
length = 0.75
beta = 1.25
memory_budget_mib = 16
"#;
    assert!(parse_config(valid).is_ok());
    for (from, to, expected) in [
        ("version = 1", "version = 2", "version"),
        (
            "output = \"proposal.csv\"",
            "output = \"\"",
            "non-empty output",
        ),
        ("elements = 1024", "elements = 0", "positive elements"),
        ("history = 8", "history = 0", "positive elements"),
        ("candidates = 16", "candidates = 0", "positive elements"),
        ("rounds = 4", "rounds = 0", "positive elements"),
        (
            "device = \"metal\"",
            "device = \"\"",
            "device must be non-empty",
        ),
        (
            "encoding = \"fp4\"",
            "encoding = \"\"",
            "encoding must be non-empty",
        ),
        (
            "acquisition = \"thompson\"",
            "acquisition = \"\"",
            "acquisition must be non-empty",
        ),
        ("neighbors = 4", "neighbors = 0", "neighbors must be"),
        (
            "edited_parameters = 6",
            "edited_parameters = 1025",
            "edited_parameters",
        ),
        (
            "length = 0.75",
            "length = 0.0",
            "length must be finite and positive",
        ),
        (
            "beta = 1.25",
            "beta = -1.0",
            "beta must be finite and nonnegative",
        ),
        (
            "memory_budget_mib = 16",
            "memory_budget_mib = 0",
            "memory_budget_mib must be positive",
        ),
        (
            "version = 1",
            "version = 1\nunexpected = true",
            "unknown field",
        ),
    ] {
        let text = valid.replace(from, to);
        let error = parse_config(&text).unwrap_err();
        assert!(error.contains(expected), "{text}: {error}");
    }
}
