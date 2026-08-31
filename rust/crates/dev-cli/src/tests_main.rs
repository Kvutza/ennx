use super::*;

#[test]
fn parses_only_supported_commands() {
    assert_eq!(parse_mode(&[]).unwrap(), Mode::Help);
    assert_eq!(parse_mode(&["help".into()]).unwrap(), Mode::Help);
    assert_eq!(parse_mode(&["--help".into()]).unwrap(), Mode::Help);
    assert_eq!(parse_mode(&["version".into()]).unwrap(), Mode::Version);
    assert_eq!(
        parse_mode(&["dev".into(), "--full".into()]).unwrap(),
        Mode::Dev { full: true }
    );
    assert_eq!(parse_mode(&["ci".into()]).unwrap(), Mode::Ci);
    assert_eq!(parse_mode(&["wheel".into()]).unwrap(), Mode::Wheel);
    assert_eq!(
        parse_mode(&["tune".into(), "config.toml".into()]).unwrap(),
        Mode::Tune(PathBuf::from("config.toml"))
    );
    assert!(parse_mode(&["build".into()]).is_err());
    assert!(parse_mode(&["check".into()]).is_err());
    assert!(parse_mode(&["python".into()]).is_err());
    assert!(parse_mode(&["cuda".into()]).is_err());
}

#[test]
fn expands_reverse_dependencies_for_core_rust_targets() {
    let downstream = downstream_targets("//rust/crates/ennx:ennx");
    assert!(downstream.contains(&"//rust/crates/dev-cli:ennx".to_string()));
    assert!(downstream.contains(&"//rust/crates/ennx-py:ennx-py".to_string()));
    assert!(downstream.contains(&"//rust/crates/ennx:ennx-unit".to_string()));
    if cfg!(target_os = "macos") {
        assert!(downstream.contains(&"//rust/crates/ennx:knn_metal_test".to_string()));
    }
    assert_eq!(normalize_target("root//rust/crates/dev-cli:ennx"), "//rust/crates/dev-cli:ennx");
    assert!(is_test_target("//rust/crates/dev-cli:ennx-test"));
    assert!(is_test_target("//rust/crates/bpann:bpann-unit"));
    assert!(!is_test_target("//rust/crates/dev-cli:ennx"));
}

#[test]
fn treats_repo_manifests_as_full_graph_inputs() {
    assert!(fallback_targets(Path::new("Cargo.toml")).is_some());
    assert!(fallback_targets(Path::new("BUCK")).is_some());
    assert!(fallback_targets(Path::new("rust/crates/ennx/Cargo.toml")).is_some());
    assert!(fallback_targets(Path::new("docs/testing.md")).is_none());
}

#[test]
fn excludes_release_artifacts_from_the_dev_graph() {
    assert!(is_dev_target("//rust/crates/ennx:ennx"));
    assert!(is_dev_target("//cuda:ennx-cuda"));
    assert!(!is_dev_target("//:wheel-linux-x86_64"));
}
