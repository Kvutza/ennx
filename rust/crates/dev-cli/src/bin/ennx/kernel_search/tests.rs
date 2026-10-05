use super::*;

fn candidate() -> Candidate {
    Candidate {
        name: "register-rescale".into(),
        parent: None,
        operator: "pisa".into(),
        source: None,
        defines: vec!["PISA_DIRECT_RESCALE".into()],
        hypothesis: "remove diagonal products".into(),
        bottleneck: "accumulator rescale".into(),
        evidence: "profile artifact".into(),
        falsify: "no repeatable whole-loop win".into(),
        expected_saving_ms: 20.0,
    }
}

#[test]
fn manifest_guards() {
    for name in ["", "../production", "validation", "inputs", "bad/name"] {
        let mut branch = candidate();
        branch.name = name.into();
        assert!(branch.validate().is_err());
    }
    let mut branch = candidate();
    branch.hypothesis.clear();
    assert!(branch.validate().is_err());
}

#[test]
fn shader_contract() {
    assert!(candidate().validate().is_ok());
    let mut branch = candidate();
    branch.defines.push("FLAG\n#include <anything>".into());
    assert!(branch.validate().is_err());
    branch.defines.clear();
    branch.operator = "unknown".into();
    assert!(branch.validate().is_err());
}

#[test]
fn exclusive_dispatch() {
    assert_eq!(
        crate::tune::experiment_kind("version=1\n[kernel-search]").unwrap(),
        crate::tune::ExperimentKind::KernelSearch
    );
    assert!(crate::tune::experiment_kind("version=1\n[kernel-search]\n[pretrain]").is_err());
    assert!(crate::tune::experiment_kind("version=1\n[kernel-search]\n[knn]").is_err());
}

#[test]
fn manifest_roundtrip() {
    let config = Search {
        baseline: "baseline.toml".into(),
        output: "runs".into(),
        pairs: 3,
        validation_pairs: 2,
        timeout_seconds: 180,
        gate: Gate {
            min_speedup: 1.03,
            win_fraction: 0.8,
            sequence_atol: 1e-5,
            token_atol: 2e-3,
        },
        micro: false,
        iterations: 1000,
        candidates: vec![candidate()],
    };
    let mut document = ennx_wire::toml::Table::new();
    document.insert("version", 1);
    document.insert("kernel-search", ennx_wire::toml::to_value(&config).unwrap());
    let encoded = ennx_wire::toml::to_string(&document).unwrap();
    assert_eq!(Search::parse(&encoded).unwrap().pairs, 3);
    document
        .get_mut("kernel-search")
        .unwrap()
        .as_map_mut()
        .unwrap()
        .insert("pairs", 1);
    assert!(Search::parse(&ennx_wire::toml::to_string(&document).unwrap()).is_err());
}

#[test]
fn archive_replay() {
    let root = std::env::temp_dir().join(format!(
        "ennx-kernel-{}-{}",
        std::process::id(),
        rand::random::<u64>()
    ));
    fs::create_dir(&root).unwrap();
    let output = root.join("runs");
    let archive = Archive::create(&root, &output).unwrap();
    let candidate_path = root.join("candidate.metal");
    fs::write(&candidate_path, "// original kernel").unwrap();
    let mut branch = candidate();
    branch.source = Some("candidate.metal".into());
    let frozen = archive.freeze_candidates(&root, &[branch]).unwrap();
    fs::write(candidate_path, "// edited after snapshot").unwrap();
    let source = fs::read_to_string(frozen[0].pisa.as_ref().unwrap()).unwrap();
    assert_eq!(source, "#define PISA_DIRECT_RESCALE\n// original kernel");
    assert!(
        Archive::create(&root, &output).is_err(),
        "concurrent measurements must be excluded"
    );
    write_json(
        &archive.path.join("failure.json"),
        &json!({"error": "compiler failure", "promotion": null}),
    )
    .unwrap();
    drop(archive);
    let next = Archive::create(&root, &output).unwrap();
    assert_eq!(
        next.history().unwrap()[0]["failure"]["error"],
        "compiler failure"
    );
    drop(next);
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn trial_roundtrip() {
    let root = std::env::temp_dir().join(format!(
        "ennx-kernel-{}-{}",
        std::process::id(),
        rand::random::<u64>()
    ));
    let archive = Archive::create(&root, &root.join("runs")).unwrap();
    let mut config = ConfigOverrides::default();
    config.experiment = Some(ennx::TurboEnnExperiment::Pretrain);
    config.model = Some(ennx::config::PretrainModel::FbtPisa1MoeV1);
    config.corpus = Some(ennx::config::PretrainCorpus::StackV3PythonPilotV1);
    config.rounds = Some(12);
    config.proposal_seed = Some(i64::MAX as u64);
    config.kernel_trial = Some(KernelTrial::default());
    let path = archive.path.join("experiment.toml");
    archive.write_experiment(&path, &config).unwrap();
    let text = fs::read_to_string(path).unwrap();
    let fields: ennx_wire::toml::Value = ennx_wire::toml::from_str(&text).unwrap();
    assert_eq!(fields["seeds"]["proposal"].as_i64(), Some(i64::MAX));
    assert_eq!(fields["run"]["rounds"].as_i64(), Some(12));
    assert!(
        fields["diagnostics"]["kernels"]
            .as_map()
            .unwrap()
            .is_empty()
    );
    let restored = ennx::config::parse_tune(&text).unwrap();
    assert_eq!(restored.proposal_seed, config.proposal_seed);
    assert_eq!(restored.rounds(), config.rounds());
    assert!(restored.kernel_trial.is_some());
    drop(archive);
    fs::remove_dir_all(root).unwrap();
}
