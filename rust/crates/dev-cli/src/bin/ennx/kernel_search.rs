//! Evidence-driven kernel search using the existing full pretraining worker.
//! Candidate generation stays with the coding agent; measurement cannot edit it.

use deser::{Deserialize, Serialize};
use ennx::config::{ConfigOverrides, KernelTrial, load_tune};
use ennx_wire::json::{Value, json};
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

mod artifacts;
mod measure;
#[cfg(test)]
mod tests;
use artifacts::{Archive, write_json};
use measure::{Gate, Pair, Seeds};

#[derive(Debug, Deserialize)]
#[deser(deny_unknown_fields)]
struct Manifest {
    version: u32,
    #[deser(rename = "kernel-search")]
    search: Search,
}

#[derive(Debug, Serialize, Deserialize)]
#[deser(deny_unknown_fields, rename_all = "kebab-case")]
struct Search {
    baseline: PathBuf,
    output: PathBuf,
    pairs: usize,
    validation_pairs: usize,
    timeout_seconds: u64,
    gate: Gate,
    candidates: Vec<Candidate>,
}

#[derive(Debug, Serialize, Deserialize)]
#[deser(deny_unknown_fields, rename_all = "kebab-case")]
struct Candidate {
    name: String,
    /// Human/agent branch identity, not a command to execute.
    parent: Option<String>,
    operator: String,
    source: Option<PathBuf>,
    #[deser(default)]
    defines: Vec<String>,
    hypothesis: String,
    bottleneck: String,
    evidence: String,
    falsify: String,
    expected_saving_ms: f64,
}

impl Search {
    fn parse(text: &str) -> Result<Self, String> {
        let manifest: Manifest =
            ennx_wire::toml::from_str(text).map_err(|error| error.to_string())?;
        let search = manifest.search;
        if manifest.version != 1
            || search.pairs < 3
            || search.validation_pairs < 2
            || search.timeout_seconds == 0
        {
            return Err("kernel-search requires version 1, at least 3 search pairs, 2 validation pairs and a positive timeout-seconds".into());
        }
        if search.baseline.as_os_str().is_empty()
            || search.output.as_os_str().is_empty()
            || search.candidates.is_empty()
        {
            return Err("kernel-search requires baseline, output and candidates".into());
        }
        search.gate.validate()?;
        let mut names = BTreeSet::new();
        for candidate in &search.candidates {
            candidate.validate()?;
            if !names.insert(&candidate.name) {
                return Err(format!("duplicate kernel branch {}", candidate.name));
            }
        }
        Ok(search)
    }
}

impl Candidate {
    fn validate(&self) -> Result<(), String> {
        if self.name.is_empty()
            || self.name.len() > 80
            || matches!(self.name.as_str(), "inputs" | "validation")
            || !self
                .name
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || b"_-".contains(&c))
        {
            return Err("kernel branch names must contain only letters, digits, _ or -".into());
        }
        KernelTrial::source(&self.operator)?;
        if [
            &self.hypothesis,
            &self.bottleneck,
            &self.evidence,
            &self.falsify,
        ]
        .iter()
        .any(|field| field.trim().is_empty())
            || !self.expected_saving_ms.is_finite()
            || self.expected_saving_ms <= 0.0
        {
            return Err(format!(
                "{} requires hypothesis, bottleneck, evidence, falsify and positive expected-saving-ms",
                self.name
            ));
        }
        if self.defines.iter().any(|name| {
            name.is_empty()
                || name.starts_with(|c: char| c.is_ascii_digit())
                || !name.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_')
        }) {
            return Err(
                "shader defines must be single C identifiers (no injected directives)".into(),
            );
        }
        Ok(())
    }
}

/// Run serially: GPU measurements from different branches must not overlap.
pub(super) fn run(root: &Path, path: &Path, text: &str, prepare: bool) -> Result<(), String> {
    let search = Search::parse(text)?;
    let parent = path
        .canonicalize()
        .map_err(|error| error.to_string())?
        .parent()
        .ok_or("manifest has no parent")?
        .to_path_buf();
    let archive = Archive::create(root, &parent.join(&search.output))?;
    let result = campaign(root, &parent, &search, &archive, text, prepare);
    if let Err(error) = &result {
        let feedback_path = archive.path.join("feedback.json");
        if feedback_path.exists() {
            let bytes = fs::read(&feedback_path).map_err(|error| error.to_string())?;
            let mut feedback: Value =
                ennx_wire::json::from_slice(&bytes).map_err(|error| error.to_string())?;
            feedback["status"] = json!("incomplete");
            feedback["error"] = json!(error);
            feedback["promotion"] = Value::null();
            write_json(&feedback_path, &feedback)?;
        }
        write_json(
            &archive.path.join("failure.json"),
            &json!({
                "schema": "ennx.kernel_failure.v1", "status": "incomplete", "error": error,
                "promotion": null,
            }),
        )?;
    }
    println!("Kernel search artifacts: {}", archive.path.display());
    result
}

fn baseline(root: &Path, path: &Path) -> Result<ConfigOverrides, String> {
    let text = fs::read_to_string(path).map_err(|error| error.to_string())?;
    let parsed = ennx::config::parse_tune(&text)?;
    if parsed.experiment != Some(ennx::TurboEnnExperiment::Pretrain) || parsed.reps() != 1 {
        return Err("kernel-search baseline must be a single-repetition pretrain experiment; use search pairs for repetitions".into());
    }
    // Reject diagnostic/config side effects, rather than silently removing them.
    let fields = ennx_wire::json::to_value(&parsed).map_err(|error| error.to_string())?;
    if fields
        .as_map()
        .ok_or("baseline is not an object")?
        .iter()
        .any(|(key, value)| {
            !value.is_null()
                && (key.as_str().is_some_and(|key| key.ends_with("_pairs"))
                    || key == "scorer_stage_samples"
                    || key == "kernel_trial"
                    || (key == "trace" && value == &json!(true)))
        })
    {
        return Err(
            "kernel-search baseline cannot contain diagnostics, tracing or shader overrides".into(),
        );
    }
    let resolved = if parsed.dataset.is_some() {
        path.to_path_buf()
    } else {
        crate::experiment::resolve_pretrain(root, path)?
    };
    load_tune(&resolved).map(|(config, _)| config)
}

fn campaign(
    root: &Path,
    parent: &Path,
    search: &Search,
    archive: &Archive,
    manifest: &str,
    prepare: bool,
) -> Result<(), String> {
    fs::write(archive.path.join("search.toml"), manifest).map_err(|error| error.to_string())?;
    let mut baseline = baseline(root, &parent.join(&search.baseline))?;
    archive.freeze_dataset(&mut baseline)?;
    let trials = archive.freeze_candidates(parent, &search.candidates)?;
    let seeds = Seeds::draw(search.pairs);
    write_json(
        &archive.path.join("context.json"),
        &json!({
            "schema": "ennx.kernel_context.v1",
        "baseline": ennx_wire::toml::to_value(&baseline).map_err(|error| error.to_string())?,
        "candidates": search.candidates, "search_seeds": seeds,
            "gate": search.gate, "history": archive.history()?,
            "measurement": "paired independent full BO loops; alternating order; compilation/setup excluded; reporting and numerical capture included in loop_seconds",
            "validation": "new proposal/acquisition seeds drawn only after selecting and freezing one winner; same fixed model and corpus",
            "source_contract": "same entry points, buffer ABI, launch dimensions and workload; PISA replaces q4 attention only; MoE replaces production TensorOps source",
            "hardware": artifacts::hardware(),
        }),
    )?;
    archive.write_experiment(&archive.path.join("baseline.toml"), &baseline)?;
    if prepare {
        println!("Prepared context, prior evidence and candidate sources; no GPU trials executed.");
        return Ok(());
    }
    let worker = archive.build_worker(root)?;
    let runner = measure::Runner {
        root,
        worker: &worker,
        baseline: &baseline,
        archive,
        gate: &search.gate,
        timeout: std::time::Duration::from_secs(search.timeout_seconds),
    };
    let mut reports = Vec::new();
    write_json(
        &archive.path.join("feedback.json"),
        &json!({
            "schema": "ennx.kernel_feedback.v1", "status": "searching",
            "candidates": reports, "promotion": null,
        }),
    )?;
    for (candidate, trial) in search.candidates.iter().zip(&trials) {
        let report = evaluate(&runner, candidate, trial, &seeds)?;
        reports.push(report);
        write_json(
            &archive.path.join("feedback.json"),
            &json!({
                "schema": "ennx.kernel_feedback.v1", "status": "searching", "candidates": reports,
                "promotion": null,
            }),
        )?;
    }
    let winner = reports
        .iter()
        .enumerate()
        .filter(|(_, report)| report["decision"] == "promising")
        .min_by(|(_, left), (_, right)| {
            left["median_ratio"]
                .as_f64()
                .unwrap_or(f64::INFINITY)
                .total_cmp(&right["median_ratio"].as_f64().unwrap_or(f64::INFINITY))
        })
        .map(|(index, _)| index);
    let validation = winner
        .map(|index| {
            validate_winner(
                &runner,
                &search.candidates[index],
                &trials[index],
                search.validation_pairs,
            )
        })
        .transpose()?;
    let promotion = validation
        .as_ref()
        .filter(|report| report["decision"] == "eligible")
        .map(|report| report["candidate"].clone());
    write_json(
        &archive.path.join("feedback.json"),
        &json!({
            "schema": "ennx.kernel_feedback.v1", "status": "completed", "candidates": reports,
            "validation": validation, "promotion": promotion,
            "production_changed": false,
        }),
    )?;
    println!(
        "Decision: {}. Production sources unchanged.",
        promotion.map_or_else(
            || "retain production".into(),
            |name| format!(
                "{} eligible for promotion",
                name.as_str().unwrap_or("unknown")
            )
        )
    );
    Ok(())
}

fn evaluate(
    runner: &measure::Runner<'_>,
    candidate: &Candidate,
    trial: &KernelTrial,
    seeds: &[Seeds],
) -> Result<Value, String> {
    println!("Branch {}: {}", candidate.name, candidate.hypothesis);
    let directory = runner.archive.path.join("branches").join(&candidate.name);
    let result = runner.pairs(&directory, trial, seeds);
    let report = match result {
        Ok(pairs) => {
            let passed = runner.gate.passes(&pairs);
            json!({
                "candidate": candidate, "pairs": pairs,
                "median_ratio": measure::median(pairs.iter().map(|pair| pair.ratio)),
                "wins": pairs.iter().filter(|pair| pair.ratio < 1.0).count(),
                "required_speedup": runner.gate.min_speedup,
                "required_win_fraction": runner.gate.win_fraction,
                "decision": if passed { "promising" } else { "inconclusive_or_slower" },
                "expected_saving_ms": candidate.expected_saving_ms,
                "measured_saving_ms": measure::median(pairs.iter().map(Pair::saving_ms)),
                "saving_prediction_met": measure::median(pairs.iter().map(Pair::saving_ms)) >= candidate.expected_saving_ms,
            })
        }
        Err(error) => {
            let report = json!({"candidate": candidate,
                "decision": if error.abort { "blocked" } else { "rejected" }, "failure": error});
            write_json(&directory.join("feedback.json"), &report)?;
            if error.abort {
                return Err(error.message);
            }
            report
        }
    };
    write_json(&directory.join("feedback.json"), &report)?;
    println!(
        "{}: {}",
        candidate.name,
        report["decision"].as_str().unwrap_or("unknown")
    );
    Ok(report)
}

fn validate_winner(
    runner: &measure::Runner<'_>,
    candidate: &Candidate,
    trial: &KernelTrial,
    count: usize,
) -> Result<Value, String> {
    let directory = runner.archive.path.join("validation");
    fs::create_dir(&directory).map_err(|error| error.to_string())?;
    // Candidate bytes are already frozen. These seeds never selected the winner.
    let seeds = Seeds::draw(count);
    write_json(
        &directory.join("selection.json"),
        &json!({"candidate": candidate.name, "seeds": seeds}),
    )?;
    let result = runner.pairs(&directory, trial, &seeds);
    let report = match result {
        Ok(pairs) => json!({
            "candidate": candidate.name,
            "decision": if runner.gate.passes(&pairs) { "eligible" } else { "validation_failed" },
            "target_met": pairs.iter().all(|pair| pair.candidate_max_ms <= f64::from(runner.baseline.target_ms())
                && pair.candidate_ms <= f64::from(runner.baseline.target_ms())),
            "pairs": pairs,
        }),
        Err(error) => {
            let report = json!({"candidate": candidate.name,
                "decision": if error.abort { "blocked" } else { "validation_failed" }, "failure": error});
            write_json(&directory.join("feedback.json"), &report)?;
            if error.abort {
                return Err(error.message);
            }
            report
        }
    };
    write_json(&directory.join("feedback.json"), &report)?;
    Ok(report)
}
