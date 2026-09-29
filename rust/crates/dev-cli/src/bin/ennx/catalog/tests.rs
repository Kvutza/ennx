use super::*;
use ennx::Perturbation;
use ennx::config::{GenerationReward, HistoryGeometry, TrustRegionSpec};

const PRETRAIN: &str = r#"version = 2

experiment = "pretrain"
model = "fbt-pisa1-legacy-v1"
corpus = "stack-v3-python-pilot-v1"

[run]
rounds = 1000
target-ms = 1000

[proposal]
distribution = "gaussian"

[enn]
epistemic-scale = 0.05
aleatoric-scale = 0.05

[enn.fit]
neighbors = true

[acquisition]
method = "ucb"
beta = 0.75

[trust-region]
method = "turbo"
shape = "tensor-family-learned"
initial = 0.01
min = 0.0001
max = 0.1
"#;

const GENERATED: &str = r#"version = 2

experiment = "pretrain"
model = "fbt-pisa1-legacy-v1"
corpus = "stack-v3-python-pilot-v1"

[run]
rounds = 100
target-ms = 200

[proposal]
distribution = "rademacher"

[enn]
geometry = "latent"
epistemic-scale = 0.05
aleatoric-scale = 0.05

[enn.fit]
neighbors = true

[acquisition]
method = "augmented-chebyshev"
scales = [1.0, 1.0, 1.0]
preferences = [1.0, 1.0, 1.0]
alpha = 0.0
seed-domain = "code-pretrain-objectives"
[trust-region]
method = "morbo"
regions = 4
rescalarize = "on-restart"
clip = true
shape = "tensor-family-learned"
initial = 0.001
min = 0.0005
max = 0.1

[generation]
purpose = "systems-probe"
initialization = "patterned"
max-tokens = 4096
temperature = 0.0
corpus-prompt-tokens = 128

[generation.verify]
mode = "accepted-prefix"
window = 128
max-window = 1024
passes = 2
unroll = 2

[generation.reward]
kind = "code-objectives"
critical-window = 128
"#;

const EXECUTION: &str = r#"version = 2
experiment = "generation"
model = "fbt-pisa1-legacy-v1"

[run]
rounds = 10000
target-ms = 200

[proposal]
method = "polynomial-threshold"
distribution = "rademacher"

[enn]
geometry = "latent"
epistemic-scale = 0.05
aleatoric-scale = 0.05

[enn.fit]
neighbors = true

[acquisition]
method = "augmented-chebyshev"
scales = [1.0, 1.0, 1.0]
preferences = [1.0, 1.0, 1.0]
alpha = 0.0
seed-domain = "code-execution-objectives"
[trust-region]
method = "morbo"
regions = 4
rescalarize = "on-restart"
clip = true
shape = "tensor-family-learned"
initial = 0.001
min = 0.0005
max = 0.1

[generation]
purpose = "pretrain"
initialization = "megatron"
max-tokens = 4096
temperature = 1.0
save-final-checkpoint = false

[generation.verify]
mode = "accepted-prefix"
window = 128
max-window = 1024
passes = 2
unroll = 2

[generation.reward]
kind = "code-execution"
environment = "../evals/mbpp-602.toml"
interpreter = "../../.venv/bin/python"
timeout-ms = 1000
"#;

fn catalog(text: &str) -> Catalog {
    let base = TuneSpec::parse(text).unwrap();
    enumerate(&base, &axes::axes(&base)).unwrap()
}

#[test]
fn actual_constraints() {
    let result = catalog(PRETRAIN);
    assert!(result.entries.len() > 100);
    assert!(
        result
            .rejected
            .keys()
            .any(|reason| reason.contains("latent history"))
    );
    assert!(
        result
            .rejected
            .keys()
            .any(|reason| reason.contains("polynomial-threshold"))
    );
    for entry in &result.entries {
        let spec = TuneSpec::parse(&entry.spec.to_toml().unwrap()).unwrap();
        if spec.enn.geometry == HistoryGeometry::Latent {
            assert_eq!(spec.proposal.distribution, Some(Perturbation::Rademacher));
        }
        assert!(spec.generation.is_none());
    }
    assert_eq!(
        result.attempted,
        result.entries.len() + result.duplicates + result.rejected.values().sum::<usize>()
    );
}

#[test]
fn vector_payloads() {
    let base = TuneSpec::parse(GENERATED).unwrap();
    let result = catalog(GENERATED);
    assert!(
        result.duplicates > 0,
        "equivalent trust-region paths must collapse"
    );
    assert!(
        result
            .rejected
            .keys()
            .any(|reason| reason.contains("cannot use reliability"))
    );
    assert!(
        result
            .rejected
            .keys()
            .any(|reason| reason.contains("accepted-prefix"))
    );
    for entry in &result.entries {
        let spec = &entry.spec;
        if matches!(spec.trust_region, TrustRegionSpec::Morbo { .. }) {
            assert_eq!(entry.choices["trust-region.method"], "morbo");
        }
        assert_eq!(spec.run.rounds, base.run.rounds);
        assert_eq!(spec.run.reps, base.run.reps);
        assert_eq!(spec.enn.fit.candidates, base.enn.fit.candidates);
        assert_eq!(
            spec.trust_region.bounds().initial,
            base.trust_region.bounds().initial
        );
        assert_eq!(
            spec.trust_region.bounds().min,
            base.trust_region.bounds().min
        );
        assert_eq!(spec.enn.epistemic_scale, base.enn.epistemic_scale);
        assert_eq!(spec.generation.as_ref().unwrap().max_tokens, 4096);
        assert!(matches!(
            spec.generation.as_ref().unwrap().reward,
            GenerationReward::CodeObjectives { .. }
        ));
        assert_eq!(spec.seeds.model, base.seeds.model);
        assert_eq!(spec.seeds.proposal, base.seeds.proposal);
    }
}

#[test]
fn purpose_gate() {
    let result = catalog(EXECUTION);
    assert!(
        result
            .rejected
            .keys()
            .any(|reason| reason.contains("neural initialization"))
    );
    for entry in result.entries {
        assert_ne!(
            entry.spec.generation.unwrap().initialization,
            ennx::config::ModelInitialization::Patterned
        );
    }
}

#[test]
fn checkpoint_axis() {
    let mut base = TuneSpec::parse(GENERATED).unwrap();
    base.generation.as_mut().unwrap().checkpoint = Some("model.fp16".into());
    let dimensions = axes::axes(&base);
    assert!(
        !dimensions
            .iter()
            .any(|axis| axis.name == "generation.initialization")
    );
}

#[test]
fn stable_unique() {
    let first = catalog(PRETRAIN);
    let second = catalog(PRETRAIN);
    let rows = |result: &Catalog| {
        result
            .entries
            .iter()
            .map(|entry| (entry.id.clone(), entry.spec.to_toml().unwrap()))
            .collect::<Vec<_>>()
    };
    assert_eq!(rows(&first), rows(&second));
    let unique = first
        .entries
        .iter()
        .map(|entry| entry.spec.to_toml().unwrap())
        .collect::<BTreeSet<_>>();
    assert_eq!(unique.len(), first.entries.len());
}

#[test]
fn export_paths() {
    let root = std::env::temp_dir().join(format!(
        "ennx-catalog-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let mut base = TuneSpec::parse(EXECUTION).unwrap();
    export::rebase(&mut base, Path::new("/source/examples/tuning"));
    let dimensions = vec![axes::Axis {
        name: "trust-region.method",
        choices: vec![(
            "morbo",
            axes::Choice::TrustRegion(base.trust_region.clone()),
        )],
    }];
    let result = enumerate(&base, &dimensions).unwrap();
    export::write(
        &root,
        &result,
        &json!({"valid": result.entries.len()}),
        &base,
    )
    .unwrap();
    assert!(export::write(&root, &result, &json!({}), &base).is_err());
    let text = fs::read_to_string(root.join("experiment-00001.toml")).unwrap();
    let spec = TuneSpec::parse(&text).unwrap();
    assert_eq!(
        spec.output.unwrap(),
        root.canonicalize().unwrap().join("runs/experiment-00001")
    );
    if let GenerationReward::CodeExecution {
        environment,
        interpreter,
        ..
    } = spec.generation.unwrap().reward
    {
        assert_eq!(
            environment,
            Path::new("/source/examples/tuning/../evals/mbpp-602.toml")
        );
        assert_eq!(
            interpreter,
            Path::new("/source/examples/tuning/../../.venv/bin/python")
        );
    } else {
        panic!("changed reward")
    }
    assert!(root.join("manifest.json").is_file());
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn coverage_counts() {
    let base = TuneSpec::parse(PRETRAIN).unwrap();
    let dimensions = axes::axes(&base);
    let result = enumerate(&base, &dimensions).unwrap();
    let report = coverage(&dimensions, &result);
    assert_eq!(report.len(), dimensions.len());
    for axis in &report {
        let counts = axis["valid_by_choice"].as_map().unwrap();
        let sum: usize = counts
            .values()
            .map(|count| count.as_u64().unwrap() as usize)
            .sum();
        assert_eq!(sum, result.entries.len());
    }
    let fitted = report
        .iter()
        .find(|axis| axis["field"].as_str() == Some("enn.fit.neighbors"))
        .unwrap();
    assert!(fitted["one_field_pairs"].as_u64().unwrap() > 0);
}
