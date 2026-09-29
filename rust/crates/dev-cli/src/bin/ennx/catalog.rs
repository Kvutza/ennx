//! Library inventory and constraint-checked experiment enumeration. No execution.
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use ennx::config::TuneSpec;
use ennx_wire::json::{Value, json};

mod axes;
mod export;
#[cfg(test)]
mod tests;

struct Entry {
    id: String,
    choices: BTreeMap<String, String>,
    spec: TuneSpec,
}

struct Catalog {
    entries: Vec<Entry>,
    rejected: BTreeMap<String, usize>,
    attempted: usize,
    duplicates: usize,
}

pub(crate) fn run(
    root: &Path,
    base: Option<&Path>,
    out: Option<&Path>,
    as_json: bool,
) -> Result<(), String> {
    let inventory = inventory(root)?;
    let Some(base) = base else {
        if out.is_some() {
            return Err("catalog --out requires a baseline experiment".into());
        }
        if as_json {
            println!(
                "{}",
                ennx_wire::json::pretty_string(&inventory).map_err(|e| e.to_string())?
            );
        } else {
            println!("{}", inventory["scope"].as_str().unwrap_or(""));
            for component in inventory["component"]
                .as_seq()
                .ok_or("inventory has no components")?
            {
                println!(
                    "{} | {} | {} | {}",
                    component["id"].as_str().unwrap_or(""),
                    component["family"].as_str().unwrap_or(""),
                    component["access"].as_str().unwrap_or(""),
                    component["options"]
                        .as_seq()
                        .ok_or("component has no options")?
                        .iter()
                        .filter_map(|value| value.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }
            println!("Enumerate: ./ennx catalog BASE.toml [--out NEW_DIRECTORY] [--json]");
        }
        return Ok(());
    };
    let base = base.canonicalize().map_err(|e| e.to_string())?;
    let text = fs::read_to_string(&base).map_err(|e| e.to_string())?;
    let mut spec = TuneSpec::parse(&text)?;
    export::rebase(&mut spec, base.parent().ok_or("baseline has no parent")?);
    let dimensions = axes::axes(&spec);
    let catalog = enumerate(&spec, &dimensions)?;
    let manifest = manifest(&base, &spec, &dimensions, &catalog, inventory)?;
    if let Some(out) = out {
        export::write(out, &catalog, &manifest, &spec)?;
    }
    if as_json {
        println!(
            "{}",
            ennx_wire::json::pretty_string(&manifest).map_err(|e| e.to_string())?
        );
    } else {
        println!(
            "{} valid configurations; {} rejected; {} duplicate paths; {} categorical axes",
            catalog.entries.len(),
            catalog.rejected.values().sum::<usize>(),
            catalog.duplicates,
            dimensions.len(),
        );
        for axis in &dimensions {
            println!(
                "{}: {}",
                axis.name,
                axis.choices
                    .iter()
                    .map(|(name, _)| *name)
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
        if let Some(out) = out {
            println!(
                "Exported TOMLs and manifest to {}. No experiments executed.",
                out.display()
            );
        }
    }
    Ok(())
}

fn inventory(root: &Path) -> Result<Value, String> {
    let text = fs::read_to_string(root.join("examples/catalog/library.toml"))
        .map_err(|e| format!("read library inventory: {e}"))?;
    let value: ennx_wire::toml::Value =
        ennx_wire::toml::from_str(&text).map_err(|e| e.to_string())?;
    ennx_wire::json::to_value(&value).map_err(|e| e.to_string())
}

fn enumerate(base: &TuneSpec, dimensions: &[axes::Axis]) -> Result<Catalog, String> {
    let count = dimensions
        .iter()
        .try_fold(1usize, |count, axis| count.checked_mul(axis.choices.len()))
        .ok_or("categorical product exceeds usize")?;
    if count > 1_000_000 {
        return Err(
            "categorical product exceeds one million; split the campaign by workload".into(),
        );
    }
    let mut catalog = Catalog {
        entries: Vec::new(),
        rejected: BTreeMap::new(),
        attempted: 0,
        duplicates: 0,
    };
    let mut seen = BTreeSet::new();
    visit(base, dimensions, &BTreeMap::new(), &mut catalog, &mut seen)?;
    Ok(catalog)
}

fn visit(
    spec: &TuneSpec,
    dimensions: &[axes::Axis],
    choices: &BTreeMap<String, String>,
    catalog: &mut Catalog,
    seen: &mut BTreeSet<String>,
) -> Result<(), String> {
    if let Some((axis, tail)) = dimensions.split_first() {
        for (name, choice) in &axis.choices {
            let mut next = spec.clone();
            choice.apply(&mut next);
            let mut selected = choices.clone();
            selected.insert(axis.name.into(), (*name).into());
            visit(&next, tail, &selected, catalog, seen)?;
        }
        return Ok(());
    }
    catalog.attempted += 1;
    let resolved = spec
        .overrides()
        .and_then(|config| TuneSpec::from_overrides(&config));
    match resolved {
        Err(reason) => *catalog.rejected.entry(reason).or_default() += 1,
        Ok(spec) => {
            let text = spec.to_toml()?;
            if seen.insert(text) {
                catalog.entries.push(Entry {
                    id: format!("experiment-{:05}", catalog.entries.len() + 1),
                    choices: resolved_choices(choices, &spec),
                    spec,
                });
            } else {
                catalog.duplicates += 1;
            }
        }
    }
    Ok(())
}

fn resolved_choices(
    choices: &BTreeMap<String, String>,
    spec: &TuneSpec,
) -> BTreeMap<String, String> {
    let mut choices = choices.clone();
    if choices.contains_key("trust-region.method") {
        let method = match spec.trust_region {
            ennx::config::TrustRegionSpec::Turbo { .. } => "turbo",
            ennx::config::TrustRegionSpec::Morbo { .. } => "morbo",
            ennx::config::TrustRegionSpec::Reliability { .. } => "reliability",
        };
        choices.insert("trust-region.method".into(), method.into());
    }
    choices
}

fn manifest(
    base: &Path,
    spec: &TuneSpec,
    dimensions: &[axes::Axis],
    catalog: &Catalog,
    inventory: Value,
) -> Result<Value, String> {
    let entries = catalog
        .entries
        .iter()
        .map(|entry| {
            json!({
                "id": entry.id, "file": format!("{}.toml", entry.id),
                "choices": entry.choices,
            })
        })
        .collect::<Vec<_>>();
    let axes = dimensions
        .iter()
        .map(|axis| {
            json!({
                "field": axis.name,
                "choices": axis.choices.iter().map(|(name, _)| *name).collect::<Vec<_>>(),
            })
        })
        .collect::<Vec<_>>();
    Ok(json!({
        "schema": "ennx.catalog.v1", "base": base, "baseline": spec.to_toml()?,
        "inventory": inventory, "axes": axes, "entries": entries,
        "coverage": coverage(dimensions, catalog),
        "attempted": catalog.attempted, "valid": catalog.entries.len(),
        "duplicates": catalog.duplicates, "rejections": catalog.rejected,
        "qualification": "Rust schema validated; hardware, resources, runtime and research quality are not qualified",
        "fixed": ["experiment", "model", "corpus", "reward", "purpose", "tasks", "budget", "seeds", "numeric settings", "diagnostics"],
        "defaults": "Newly activated payloads use Rust defaults: UCB beta, reliability policy, self-tuning neighbor count. No MORBO payload is invented.",
        "comparisons": "Compare entries differing in one categorical field within this baseline. Hold numeric payloads and resources equal; initialization and feedback changes require separate capability interpretation.",
    }))
}

fn coverage(dimensions: &[axes::Axis], catalog: &Catalog) -> Vec<Value> {
    dimensions
        .iter()
        .map(|axis| {
            let mut counts: BTreeMap<String, usize> = axis
                .choices
                .iter()
                .map(|(name, _)| ((*name).into(), 0))
                .collect();
            let mut contexts: BTreeMap<BTreeMap<String, String>, BTreeSet<String>> =
                BTreeMap::new();
            for entry in &catalog.entries {
                let mut context = entry.choices.clone();
                let selected = context.remove(axis.name).unwrap();
                *counts.entry(selected.clone()).or_default() += 1;
                contexts.entry(context).or_default().insert(selected);
            }
            let groups = contexts.values().filter(|values| values.len() > 1).count();
            let pairs = contexts
                .values()
                .map(|values| values.len() * (values.len() - 1) / 2)
                .sum::<usize>();
            json!({"field": axis.name, "valid_by_choice": counts,
            "comparison_contexts": groups, "one_field_pairs": pairs})
        })
        .collect()
}
