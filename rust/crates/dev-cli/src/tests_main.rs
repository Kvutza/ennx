use super::*;
use clap::error::ErrorKind;

fn parse(args: &[&str]) -> Result<Action, clap::Error> {
    Cli::try_parse_from(std::iter::once("ennx").chain(args.iter().copied())).map(|cli| cli.action)
}

#[test]
fn ordinary_options() {
    assert_eq!(
        parse(&["tune", "knn", "tune/knn.toml"]).unwrap(),
        Action::Tune {
            target: TuneTarget::Knn {
                config: "tune/knn.toml".into()
            }
        }
    );
    assert_eq!(
        parse(&["tune", "proposal", "tune/proposal.toml"]).unwrap(),
        Action::Tune {
            target: TuneTarget::Proposal {
                config: "tune/proposal.toml".into()
            }
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
fn malformed_work() {
    for args in [
        vec!["build"],
        vec!["ci"],
        vec!["deps", "--check"],
        vec!["dev"],
        vec!["fmt"],
        vec!["test"],
        vec!["wheel"],
        vec!["tune"],
        vec!["tune", "knn"],
        vec!["tune", "proposal"],
        vec!["tune", "knn", "--config"],
        vec!["tune", "other", "tune/knn.toml"],
        vec!["tune", "knn", "a", "b"],
        vec!["tune", "proposal", "a", "b"],
        vec!["--help", "unexpected"],
    ] {
        assert!(parse(&args).is_err(), "accepted {args:?}");
    }
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
        ("rounds = 1", "rounds = 1\nrounds = 2", "duplicate key"),
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
    assert!(parse_knn(&text.replace("large", "small"))
        .unwrap_err()
        .contains("unique"));
    assert!(parse_knn(&text.replace("rows = 1_024", "row = 1_024"))
        .unwrap_err()
        .contains("unknown field"));
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
