use super::*;

fn parse(fields: &str) -> ConfigOverrides {
    ennx_wire::toml::from_str(&format!("[objective_acquisition]\n{fields}")).unwrap()
}

#[test]
fn explicit_vectors() {
    let pareto = parse("mode='pareto'\nscales=[1.0,2.0]");
    assert!(
        pareto
            .objective_acquisition
            .as_ref()
            .unwrap()
            .validate()
            .is_ok()
    );
    assert!(
        pareto
            .validate_objectives()
            .unwrap_err()
            .contains("scalar rewards only")
    );
    assert!(
        parse("mode='pareto'\nscales=[1.0,2.0]\nalpha=0.0")
            .objective_acquisition
            .unwrap()
            .validate()
            .is_err()
    );
    assert!(
        parse("mode='morbo'\nscales=[1.0,2.0]")
            .objective_acquisition
            .unwrap()
            .validate()
            .is_err()
    );
}

#[test]
fn morbo_preferences() {
    let config = parse(
        "mode='morbo'\nscales=[1.0,2.0]\npreferences=[2.0,1.0]\nalpha=0.05\nseed-domain='coding-objectives'\nrescalarize='on-restart'\nclip=true",
    );
    assert!(
        config
            .objective_acquisition
            .as_ref()
            .unwrap()
            .validate()
            .is_ok()
    );
    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        let first = config.resident_objectives(0).unwrap().unwrap();
        let next = config.resident_objectives(1).unwrap().unwrap();
        let crate::bf16_metal::ObjectiveAcquisition::Morbo { seed: a, .. } = first.acquisition
        else {
            panic!()
        };
        let crate::bf16_metal::ObjectiveAcquisition::Morbo { seed: b, .. } = next.acquisition
        else {
            panic!()
        };
        assert_eq!(a, config.derived_seed(0, "coding-objectives", 0));
        assert_ne!(a, b);
    }
}

#[test]
fn morbo_geometry() {
    let mut config = parse(
        "mode='morbo'\nscales=[1.0,2.0]\npreferences=[2.0,1.0]\nalpha=0.05\nseed-domain='coding-objectives'\nrescalarize='on-restart'\nclip=true",
    );
    assert!(
        config
            .validate_objectives()
            .unwrap_err()
            .contains("latent history geometry")
    );
    config.history_geometry = Some(HistoryGeometry::Latent);
    assert!(
        config
            .validate_objectives()
            .unwrap_err()
            .contains("scalar rewards only")
    );
}
