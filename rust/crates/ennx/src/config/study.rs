use serde::{Deserialize, Serialize};

/// Distance normalization applied before resident ENN neighbor ranking and weighting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DistanceScaling {
    /// Preserve the globally scaled squared metric.
    #[default]
    Global,
    /// Divide by the geometric mean of query and observation local squared radii.
    SelfTuning,
}

impl DistanceScaling {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Global => "global",
            Self::SelfTuning => "self_tuning",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrustRegionShape {
    Scalar,
    TensorFamilyStatic,
    TensorFamilyLearned,
}

impl TrustRegionShape {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::TensorFamilyStatic => "tensor_family_static",
            Self::TensorFamilyLearned => "tensor_family_learned",
        }
    }
}

fn remove_table(root: &mut toml::Table, name: &str) -> Result<Option<toml::Table>, String> {
    match root.get(name) {
        Some(toml::Value::Table(_)) => {}
        Some(_) | None => return Ok(None),
    }
    match root.remove(name) {
        Some(toml::Value::Table(table)) => Ok(Some(table)),
        _ => unreachable!("checked table value above"),
    }
}

fn take_string(table: &mut toml::Table, name: &str) -> Result<Option<String>, String> {
    match table.remove(name) {
        Some(toml::Value::String(value)) => Ok(Some(value)),
        Some(_) => Err(format!("{name} must be a string")),
        None => Ok(None),
    }
}

fn insert_if_absent(
    root: &mut toml::Table,
    key: &str,
    value: toml::Value,
    section: &str,
) -> Result<(), String> {
    if root.insert(key.into(), value).is_some() {
        return Err(format!("{section}.{key} conflicts with top-level {key}"));
    }
    Ok(())
}

fn drain_table(
    root: &mut toml::Table,
    section: &str,
    table: toml::Table,
    renames: &[(&str, &str)],
) -> Result<(), String> {
    for (key, value) in table {
        let target = renames
            .iter()
            .find_map(|(from, to)| (*from == key).then_some(*to))
            .unwrap_or(&key);
        insert_if_absent(root, target, value, section)?;
    }
    Ok(())
}

fn take_single_subtable(
    table: &mut toml::Table,
    section: &str,
) -> Result<Option<(String, toml::Table)>, String> {
    let subtables = table
        .iter()
        .filter(|(_, value)| matches!(value, toml::Value::Table(_)))
        .map(|(key, _)| key.clone())
        .collect::<Vec<_>>();
    match subtables.as_slice() {
        [] => Ok(None),
        [name] => match table.remove(name) {
            Some(toml::Value::Table(inner)) => Ok(Some((name.clone(), inner))),
            _ => unreachable!("checked table value above"),
        },
        _ => Err(format!("{section} must select exactly one primitive")),
    }
}

fn normalize_primitive(name: String) -> String {
    name.replace('-', "_")
}

fn lower_pretrain(root: &mut toml::Table) -> Result<(), String> {
    if let Some(pretrain) = remove_table(root, "pretrain")? {
        insert_if_absent(
            root,
            "study",
            toml::Value::String("pretrain".into()),
            "pretrain",
        )?;
        drain_table(root, "pretrain", pretrain, &[])?;
    }
    Ok(())
}

fn lower_study(root: &mut toml::Table) -> Result<(), String> {
    if let Some(mut study) = remove_table(root, "study")? {
        if let Some(kind) = take_string(&mut study, "kind")? {
            insert_if_absent(root, "study", toml::Value::String(kind), "study")?;
        }
        drain_table(root, "study", study, &[])?;
    }
    for (section, renames) in [
        (
            "rounds",
            &[("count", "rounds"), ("target_ms", "target_round_ms")][..],
        ),
        ("experiment", &[][..]),
    ] {
        if let Some(table) = remove_table(root, section)? {
            drain_table(root, section, table, renames)?;
        }
    }
    Ok(())
}

fn lower_perturbation(root: &mut toml::Table) -> Result<(), String> {
    if let Some(mut perturbation) = remove_table(root, "perturbation")? {
        if let Some(distribution) = take_string(&mut perturbation, "distribution")? {
            insert_if_absent(
                root,
                "perturbation",
                toml::Value::String(distribution),
                "perturbation",
            )?;
        }
        if let Some((name, inner)) = take_single_subtable(&mut perturbation, "perturbation")? {
            if !inner.is_empty() {
                return Err(format!("perturbation.{name} does not accept fields"));
            }
            insert_if_absent(
                root,
                "perturbation",
                toml::Value::String(name),
                "perturbation",
            )?;
        }
        drain_table(root, "perturbation", perturbation, &[])?;
    }
    if let Some(mut proposal) = remove_table(root, "proposal")? {
        if let Some(distribution) = take_string(&mut proposal, "distribution")? {
            insert_if_absent(
                root,
                "perturbation",
                toml::Value::String(distribution),
                "proposal",
            )?;
        }
        drain_table(root, "proposal", proposal, &[])?;
    }
    Ok(())
}

fn lower_acquisition(root: &mut toml::Table) -> Result<(), String> {
    let Some(mut acquisition) = remove_table(root, "acquisition")? else {
        return Ok(());
    };
    if let Some(method) = take_string(&mut acquisition, "method")? {
        insert_if_absent(
            root,
            "acquisition",
            toml::Value::String(method),
            "acquisition",
        )?;
    }
    if let Some((name, inner)) = take_single_subtable(&mut acquisition, "acquisition")? {
        insert_if_absent(
            root,
            "acquisition",
            toml::Value::String(name),
            "acquisition",
        )?;
        drain_table(root, "acquisition", inner, &[])?;
    }
    if let Some(kind) = take_string(&mut acquisition, "kind")? {
        insert_if_absent(
            root,
            "acquisition",
            toml::Value::String(kind),
            "acquisition",
        )?;
    }
    drain_table(root, "acquisition", acquisition, &[])
}

fn lower_enn_table(root: &mut toml::Table, mut inner: toml::Table) -> Result<(), String> {
    if let Some(fit) = remove_table(&mut inner, "fit")? {
        for key in fit.keys() {
            if !matches!(key.as_str(), "candidates" | "samples") {
                return Err(format!("unknown field surrogate.enn.fit.{key}"));
            }
        }
        drain_table(
            root,
            "surrogate.enn.fit",
            fit,
            &[("candidates", "num_candidates"), ("samples", "num_samples")],
        )?;
    }
    drain_table(
        root,
        "surrogate",
        inner,
        &[
            ("neighbors", "k_neighbors"),
            ("candidates", "num_candidates"),
            ("samples", "num_samples"),
        ],
    )
}

fn lower_surrogate(root: &mut toml::Table) -> Result<(), String> {
    let Some(mut surrogate) = remove_table(root, "surrogate")? else {
        return Ok(());
    };
    if let Some(method) = take_string(&mut surrogate, "method")? {
        if method != "enn" && method != "resident_enn" {
            return Err("surrogate.method must be 'enn'".into());
        }
    }
    for (source, target) in [
        ("fit_candidates", "num_candidates"),
        ("fit_samples", "num_samples"),
    ] {
        if let Some(value) = surrogate.remove(source) {
            insert_if_absent(root, target, value, "surrogate")?;
        }
    }
    if let Some((name, inner)) = take_single_subtable(&mut surrogate, "surrogate")? {
        if !matches!(normalize_primitive(name).as_str(), "enn" | "resident_enn") {
            return Err("surrogate must select method = 'enn'".into());
        }
        lower_enn_table(root, inner)?;
    }
    if let Some(kind) = take_string(&mut surrogate, "kind")? {
        if kind != "resident_enn" {
            return Err("surrogate.kind must be 'resident_enn'".into());
        }
    }
    lower_enn_table(root, surrogate)
}

fn take_trust_region(root: &mut toml::Table) -> Result<Option<toml::Table>, String> {
    let public = remove_table(root, "trust-region")?;
    let legacy = remove_table(root, "trust_region")?;
    if public.is_some() && legacy.is_some() {
        return Err("trust-region conflicts with trust_region".into());
    }
    Ok(public.or(legacy))
}

fn lower_trust_region(root: &mut toml::Table) -> Result<(), String> {
    let Some(mut trust_region) = take_trust_region(root)? else {
        return Ok(());
    };
    if let Some(reliability) = remove_table(&mut trust_region, "reliability")? {
        insert_if_absent(
            root,
            "reliability_controller",
            toml::Value::Table(reliability),
            "trust-region.reliability",
        )?;
    }
    if let Some(method) = take_string(&mut trust_region, "method")? {
        insert_if_absent(
            root,
            "trust_region_kind",
            toml::Value::String(normalize_primitive(method)),
            "trust-region",
        )?;
    }
    if let Some((name, inner)) = take_single_subtable(&mut trust_region, "trust_region")? {
        insert_if_absent(
            root,
            "trust_region_kind",
            toml::Value::String(normalize_primitive(name)),
            "trust_region",
        )?;
        drain_table(
            root,
            "trust_region",
            inner,
            &[("shape", "trust_region_shape")],
        )?;
    }
    for (source, target) in [
        ("kind", "trust_region_kind"),
        ("shape", "trust_region_shape"),
    ] {
        if let Some(value) = take_string(&mut trust_region, source)? {
            insert_if_absent(root, target, toml::Value::String(value), "trust_region")?;
        }
    }
    if trust_region.remove("tensor_family").is_some() {
        return Err("trust_region.tensor_family is reserved for learned/static shape values but is not configurable yet".into());
    }
    drain_table(root, "trust_region", trust_region, &[])
}

pub(super) fn lower_structured_tune(root: &mut toml::Table) -> Result<(), String> {
    lower_pretrain(root)?;
    lower_study(root)?;
    lower_perturbation(root)?;
    lower_acquisition(root)?;
    lower_surrogate(root)?;
    lower_trust_region(root)
}
