use super::*;

const PRETRAIN: &str = "version=2\nexperiment='pretrain'\nmodel='fbt-pisa1-legacy-v1'\ncorpus='stack-v3-python-pilot-v1'\n";

#[test]
fn roundtrip() {
    macro_rules! examples {
        ($($name:literal),+ $(,)?) => {
            [$({
                #[cfg(feature = "buck2-test-data")]
                let text = include_str!(concat!("../../../turbo-enn.toml/", $name));
                #[cfg(not(feature = "buck2-test-data"))]
                let text = include_str!(concat!("../../../../../../examples/tuning/", $name));
                ($name, text)
            }),+]
        };
    }
    for (name, text) in examples!(
        "code-pretrain.toml",
        "diffusion.toml",
        "code-generation-enn.toml",
        "code-generation-random.toml",
        "fineweb-enn.toml",
        "fineweb-random-selection.toml",
        "code-pretrain-generated.toml",
        "code-pretrain-evaluated.toml",
        "code-pretrain-family-learned.toml",
        "code-pretrain-kernel.toml",
        "code-pretrain-kernel-rademacher.toml",
        "code-pretrain-perturbation-gaussian.toml",
        "code-pretrain-perturbation-rademacher.toml",
        "code-pretrain-ablation-global-reliability.toml",
        "code-pretrain-ablation-global-turbo.toml",
        "code-pretrain-ablation-local-reliability.toml",
        "code-pretrain-ablation-local-turbo.toml",
        "code-pretrain-reconstruction.toml",
        "turbo-enn-one-round.toml",
        "moe-layer.toml",
        "generation.toml",
        "generation-block.toml",
        "generation-4096.toml",
        "generation-4096-gaussian.toml",
        "scorer-stages.toml",
        "turbo-enn.toml",
        "turbo-enn-trace.toml",
    ) {
        let spec = TuneSpec::parse(text).unwrap_or_else(|error| panic!("{name}: {error}"));
        let resolved = spec.to_toml().unwrap();
        let restored = TuneSpec::parse(&resolved).unwrap();
        assert_eq!(resolved, restored.to_toml().unwrap(), "{name}");
    }
}

#[test]
fn closed() {
    for fields in [
        "neighbor=10",
        "[run]\nround=10",
        "[data]\ndataset='x'",
        "[proposal]\ncandidate=4",
        "[enn]\nneigbors=10",
        "[enn.fit]\nfit-samples=10",
        "[acquisition]\nmethod='thompson'\nbeta=1.0",
        "[trust-region]\nlength-min=0.1",
        "[trust-region]\nmethod='turbo'\nevidence-decay=0.9",
        "[trust-region]\nmethod='reliability'\nevidence-decy=0.9",
        "[objective]\nreference='imaginary'",
        "[seeds]\nproposal-seed=17",
        "[diagnostics]\nstage-sample=1",
    ] {
        assert!(
            parse_tune(&format!("{PRETRAIN}{fields}")).is_err(),
            "accepted {fields}"
        );
    }
}

#[test]
fn guards() {
    for fields in [
        "[run]\nrounds=0",
        "[run]\nrounds=true",
        "[run]\nreps=0",
        "[run]\ntarget-ms=0",
        "[proposal]\ndistribution='uniform'",
        "[proposal]\ncandidates=0",
        "[proposal]\ncandidates=8",
        "[proposal]\ncandidates=4\narms=3",
        "[enn]\nneighbors=0",
        "[enn]\nepistemic-scale=nan",
        "[enn]\nscaling='global'\nlocal-neighbors=8",
        "[enn]\ngeometry='latent'\n[proposal]\ndistribution='gaussian'",
        "[enn.fit]\nsamples=0",
        "[trust-region]\nmin=0.2\ninitial=0.1\nmax=0.3",
        "[trust-region]\nmethod='reliability'\nevidence-decay=2.0",
        "[acquisition]\nmethod='pareto'\nscales=[1.0,2.0]",
    ] {
        assert!(
            parse_tune(&format!("{PRETRAIN}{fields}")).is_err(),
            "accepted {fields}"
        );
    }
}

#[test]
fn reliability() {
    let config = parse_tune(&format!("{PRETRAIN}[trust-region]\nmethod='reliability'")).unwrap();
    assert_eq!(
        config.reliability_controller(),
        Some(crate::ReliabilityPolicy::default())
    );
    let spec = TuneSpec::from_overrides(&config).unwrap();
    let roundtrip = parse_tune(&spec.to_toml().unwrap()).unwrap();
    assert_eq!(
        roundtrip.reliability_controller(),
        config.reliability_controller()
    );
}

#[test]
fn paths() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("experiment.toml");
    std::fs::write(
        &path,
        "version=2\nexperiment='end-to-end'\noutput='results'\n[acquisition]\nmethod='thompson'",
    )
    .unwrap();
    let authored = std::fs::read_to_string(&path).unwrap();
    let (config, text) = load_tune(&path).unwrap();
    let restored = parse_tune(&text).unwrap();
    assert_eq!(
        restored.output(),
        directory.path().canonicalize().unwrap().join("results")
    );
    assert_eq!(restored.length(), config.length());
    assert_eq!(restored.acquisition_seed(), config.acquisition_seed());
    assert!(matches!(
        restored.acquisition,
        Some(AcquisitionConfig::Thompson)
    ));
    assert!(text.contains("version = 2"));
    assert!(!text.contains("length-init"));
    assert!(!text.contains("num-candidates"));
    assert_eq!(std::fs::read_to_string(path).unwrap(), authored);
}

#[test]
fn models() {
    assert_eq!(
        PretrainModel::ALL.map(PretrainModel::id),
        [
            "fbt-pisa1-legacy-v1",
            "fbt-pisa1-residual1-v1",
            "fbt-pisa1-projected-boundary-v1",
            "fbt-pisa1-hc4-v1",
            "fbt-pisa1-mhc4-v1",
            "fbt-pisa1-looped-mhc4-v1",
            "fbt-pisa1-diffusion-mhc4-v1",
            "fbt-pisa1-hnet-mhc4-v1",
        ]
    );
}

#[test]
fn budgets() {
    let config = parse_tune(&format!(
        "{PRETRAIN}[proposal]\ncandidates=4\narms=2\n[enn.fit]\ncandidates=64\nsamples=32"
    ))
    .unwrap();
    let resident = config.resident_enn(17).unwrap();
    assert_eq!(resident.pool.count(), 4);
    assert_eq!(resident.pool.arms(), 2);
    assert_eq!(resident.num_candidates, 64);
    assert_eq!(resident.num_samples, 32);
}

#[test]
fn vectors() {
    let base = "version=2\nexperiment='generation'\nmodel='fbt-pisa1-legacy-v1'\n[generation]\npurpose='systems-probe'\nmax-tokens=16\ntemperature=0.0\n[[generation.tasks]]\nprompt=[1]\n[generation.reward]\nkind='command-objectives'\nprogram='score'\nargs=[]\ntimeout-ms=10\n";
    let policy = "[acquisition]\nmethod='pareto'\nscales=[1.0,2.0]";
    let config = parse_tune(&format!("{base}{policy}")).unwrap();
    assert_eq!(
        config.objective_acquisition.unwrap().mode,
        ResidentObjectiveMode::Pareto
    );
    assert!(parse_tune(&format!("{base}{policy}\npreferences=[1.0,1.0]")).is_err());
    assert!(parse_tune(base).is_err());
}
