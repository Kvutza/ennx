use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use sha2::{Digest, Sha256};

use crate::tune::{
    ExperimentKind, KnnTuneConfig, ProposalTuneConfig, experiment_kind, is_pretrain, parse_config,
    parse_knn,
};
use crate::{TensorAction, build_system, kernel_search};

pub(crate) fn tensor(command: TensorAction) -> Result<(), String> {
    match command {
        TensorAction::Inspect { file, json } => {
            let mapped = ennx::tensor_store::SafeTensors::open(&file)?;
            let total_bytes = mapped.tensors().iter().try_fold(0usize, |total, tensor| {
                total
                    .checked_add(mapped.bytes(tensor).len())
                    .ok_or("tensor byte count overflow")
            })?;
            if json {
                let tensors = mapped
                    .tensors()
                    .iter()
                    .map(|tensor| {
                        ennx_wire::json::json!({
                            "name": tensor.name,
                            "dtype": tensor.dtype,
                            "shape": tensor.shape,
                            "bytes": mapped.bytes(tensor).len(),
                        })
                    })
                    .collect::<Vec<_>>();
                println!(
                    "{}",
                    ennx_wire::json::pretty_string(&ennx_wire::json::json!({
                        "file": file,
                        "tensors": tensors,
                        "tensor_count": tensors.len(),
                        "data_bytes": total_bytes,
                        "metadata": mapped.metadata(),
                    }))
                    .map_err(|error| error.to_string())?
                );
            } else {
                println!(
                    "{} tensors, {} data bytes | {}",
                    mapped.tensors().len(),
                    total_bytes,
                    file.display()
                );
                for tensor in mapped.tensors() {
                    println!(
                        "{} {:?} {:?} {} bytes",
                        tensor.name,
                        tensor.dtype,
                        tensor.shape,
                        mapped.bytes(tensor).len()
                    );
                }
            }
            Ok(())
        }
    }
}

pub(crate) fn tune(root: &Path, path: &Path, prepare: bool) -> Result<(), String> {
    let text = fs::read_to_string(path)
        .map_err(|error| format!("read config {}: {error}", path.display()))?;
    let kind = experiment_kind(&text)
        .map_err(|error| format!("invalid config {}: {error}", path.display()))?;
    let pretrain = kind == ExperimentKind::TurboEnn
        && is_pretrain(&text)
            .map_err(|error| format!("invalid config {}: {error}", path.display()))?;
    if prepare && pretrain {
        let resolved = resolve_pretrain(root, path)?;
        println!("Prepared {}", resolved.display());
        return Ok(());
    }
    if prepare && kind != ExperimentKind::KernelSearch {
        return Err("--prepare requires a pretraining or kernel-search configuration".into());
    }
    match kind {
        ExperimentKind::KernelSearch => kernel_search::run(root, path, &text, prepare),
        ExperimentKind::Knn => tune_knn(root, path),
        ExperimentKind::Proposal => tune_proposal(root, path),
        ExperimentKind::TurboEnn => {
            let resolved;
            let path = if pretrain && ennx::config::parse_tune(&text)?.dataset.is_none() {
                resolved = resolve_pretrain(root, path)?;
                resolved.as_path()
            } else {
                path
            };
            let engine = build_system::active()?;
            let worker = engine.build_artifact(root, "//rust/crates/ennx:turbo-enn-worker")?;
            let mut command = engine.run_command(
                root,
                "//rust/crates/ennx:turbo-enn",
                &[path.to_str().ok_or("config path is not UTF-8")?.to_owned()],
            );
            command.env("ENNX_WORKER_EXECUTABLE", worker.path);
            crate::execute(command)
        }
    }
}

pub(crate) fn resolve_pretrain(root: &Path, path: &Path) -> Result<PathBuf, String> {
    println!("Resolving the pretraining corpus...");
    let source = path.canonicalize().map_err(|error| error.to_string())?;
    let parent = source.parent().ok_or("config has no parent directory")?;
    let text = fs::read_to_string(&source).map_err(|error| error.to_string())?;
    let mut spec = ennx::config::TuneSpec::parse(&text)?;
    if let Some(generation) = &mut spec.generation {
        generation.resolve(parent)?;
    }
    crate::protocol::resolve_corpus(root, &mut spec)?;

    let mut hash = Sha256::new();
    hash.update(b"ennx-experiment-v2\0");
    hash.update(spec.to_toml()?.as_bytes());
    let id = format!("{:x}", hash.finalize())[..20].to_owned();
    let root = root.canonicalize().map_err(|error| error.to_string())?;
    spec.output = Some(root.join(".cache/ennx/runs/pretrain").join(&id));
    let resolved = root
        .join(".cache/ennx/experiments/pretrain")
        .join(format!("{id}.toml"));
    let directory = resolved.parent().ok_or("resolved config has no parent")?;
    fs::create_dir_all(directory).map_err(|error| error.to_string())?;
    let temporary = directory.join(format!(".{id}.{}.tmp", std::process::id()));
    fs::write(&temporary, spec.to_toml()?).map_err(|error| error.to_string())?;
    fs::rename(&temporary, &resolved).map_err(|error| error.to_string())?;
    Ok(resolved)
}

fn tune_knn(root: &Path, path: &Path) -> Result<(), String> {
    let config = knn_config(path)?;
    let output = root.join(&config.output);
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent).map_err(|error| format!("create output directory: {error}"))?;
    }
    println!("Running KNN frontier tuning...");
    let rounds = config.rounds.to_string();
    let mut args = vec![config.output.clone(), rounds];
    for point in &config.points {
        args.push(point.clone());
    }
    build_system::active()?.run(root, "//rust/crates/ennx:knn_frontier", &args)
}

fn tune_proposal(root: &Path, path: &Path) -> Result<(), String> {
    let config = load_config(path)?;
    let output = root.join(&config.output);
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent).map_err(|error| format!("create output directory: {error}"))?;
    }
    let estimated_bytes = proposal_bytes(&config)?;
    println!("Running proposal benchmark...");
    let source_commit = command_output(
        root,
        "jj",
        &["log", "-r", "@", "--no-graph", "-T", "commit_id.short(12)"],
    )?
    .trim()
    .to_string();
    let dirty_paths = command_output(root, "jj", &["diff", "--summary", "--no-pager"])?;
    let dirty_count = dirty_paths.lines().count();
    let rustc_version = command_output(root, "rustc", &["--version"])?
        .trim()
        .to_string();
    let cargo_version = command_output(root, "cargo", &["--version"])?
        .trim()
        .to_string();
    let command_start = Instant::now();
    let args = vec![
        config.elements.to_string(),
        config.history.to_string(),
        config.candidates.to_string(),
        config.rounds.to_string(),
        config.warmup.to_string(),
        config.device.clone(),
        config.encoding.clone(),
        config.acquisition.clone(),
        config.length.to_string(),
        config.neighbors.to_string(),
        config.beta.to_string(),
        config.seed.to_string(),
        config.edited_parameters.to_string(),
    ];
    let build_system = build_system::active()?;
    let stdout = build_system.run_output(root, "//rust/crates/ennx:trial_bench", &args)?;
    let command_s = command_start.elapsed().as_secs_f64();
    let mut report = String::new();
    report.push_str("# ENNX proposal benchmark\n");
    report.push_str(&format!("# config={}\n", path.display()));
    report.push_str(&format!("# output={}\n", output.display()));
    report.push_str(&format!("# elements={}\n", config.elements));
    report.push_str(&format!("# history={}\n", config.history));
    report.push_str(&format!("# candidates={}\n", config.candidates));
    report.push_str(&format!("# rounds={}\n", config.rounds));
    report.push_str(&format!("# warmup={}\n", config.warmup));
    report.push_str(&format!("# device={}\n", config.device));
    report.push_str(&format!("# encoding={}\n", config.encoding));
    report.push_str(&format!("# acquisition={}\n", config.acquisition));
    report.push_str(&format!("# neighbors={}\n", config.neighbors));
    report.push_str(&format!(
        "# edited_parameters={}\n",
        config.edited_parameters
    ));
    report.push_str(&format!("# length={}\n", config.length));
    report.push_str(&format!("# beta={}\n", config.beta));
    report.push_str(&format!("# seed={}\n", config.seed));
    report.push_str(&format!("# estimated_bytes={estimated_bytes}\n"));
    report.push_str(&format!("# peak_accounted_bytes={estimated_bytes}\n"));
    if let Some(mib) = config.memory_budget_mib {
        report.push_str(&format!("# memory_budget_mib={mib}\n"));
    }
    report.push_str(&format!("# source_commit={source_commit}\n"));
    report.push_str(&format!("# source_dirty={}\n", dirty_count > 0));
    report.push_str(&format!("# source_dirty_paths={dirty_count}\n"));
    report.push_str(&format!("# rustc_version={rustc_version}\n"));
    report.push_str(&format!("# cargo_version={cargo_version}\n"));
    report.push_str(&format!("# build_system={}\n", build_system.name()));
    report.push_str(&format!("# orchestration_s={command_s:.9}\n"));
    report.push_str(&stdout);
    fs::write(&output, report).map_err(|error| format!("write {}: {error}", output.display()))?;
    println!("Proposal benchmark written to {}", output.display());
    Ok(())
}

fn proposal_bytes(config: &ProposalTuneConfig) -> Result<usize, String> {
    let estimated_bytes = estimate_bytes(config)?;
    if let Some(budget_mib) = config.memory_budget_mib {
        let budget_bytes = budget_mib
            .checked_mul(1024 * 1024)
            .ok_or("proposal memory budget overflows usize")?;
        if estimated_bytes > budget_bytes {
            return Err(format!(
                "proposal estimate {estimated_bytes} bytes exceeds the {budget_mib} MiB budget"
            ));
        }
    }
    Ok(estimated_bytes)
}

fn knn_config(path: &Path) -> Result<KnnTuneConfig, String> {
    let text = fs::read_to_string(path)
        .map_err(|error| format!("read config {}: {error}", path.display()))?;
    parse_knn(&text).map_err(|error| format!("invalid config {}: {error}", path.display()))
}

fn load_config(path: &Path) -> Result<ProposalTuneConfig, String> {
    let text = fs::read_to_string(path)
        .map_err(|error| format!("read config {}: {error}", path.display()))?;
    parse_config(&text).map_err(|error| format!("invalid config {}: {error}", path.display()))
}

fn command_output(root: &Path, program: &str, args: &[&str]) -> Result<String, String> {
    let output = Command::new(program)
        .current_dir(root)
        .args(args)
        .output()
        .map_err(|error| format!("start {program}: {error}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "{program} exited with {}{}\n{}",
            output.status,
            if stderr.trim().is_empty() {
                String::new()
            } else {
                ":".to_string()
            },
            stderr.trim()
        ));
    }
    String::from_utf8(output.stdout)
        .map_err(|error| format!("{program} stdout was not UTF-8: {error}"))
}

fn estimate_bytes(config: &ProposalTuneConfig) -> Result<usize, String> {
    let bits = proposal_bits(&config.encoding)?;
    let row_bytes = if bits == 4 {
        config.elements.div_ceil(2)
    } else {
        config.elements
    };
    let candidate_capacity = config.candidates.next_power_of_two().max(1);
    let tile_count = config.elements.div_ceil(65_536).max(1);
    let resident_rows = config
        .history
        .checked_add(2)
        .and_then(|rows| rows.checked_mul(row_bytes))
        .ok_or("proposal row allocation overflows")?;
    let scratch_small = candidate_capacity
        .checked_mul(8 + 4 + 4 + 4)
        .ok_or("proposal scratch allocation overflows")?;
    let partials = candidate_capacity
        .checked_mul(128)
        .and_then(|value| value.checked_mul(tile_count))
        .and_then(|value| value.checked_mul(4))
        .ok_or("proposal partial allocation overflows")?;
    let sparse_edits = candidate_capacity
        .checked_mul(config.edited_parameters)
        .and_then(|value| value.checked_mul(2 * std::mem::size_of::<u32>()))
        .ok_or("proposal sparse edit allocation overflows")?;
    resident_rows
        .checked_add(scratch_small)
        .and_then(|value| value.checked_add(partials))
        .and_then(|value| value.checked_add(sparse_edits))
        .ok_or_else(|| "proposal memory estimate overflows".to_string())
}

fn proposal_bits(encoding: &str) -> Result<usize, String> {
    match encoding {
        "int4" | "fp4" | "fp4_e2m1" | "e2m1" => Ok(4),
        "int8" | "fp8" | "fp8_e4m3" | "e4m3" | "fp8_e5m2" | "e5m2" => Ok(8),
        other => Err(format!("unsupported proposal encoding {other:?}")),
    }
}

#[cfg(test)]
#[path = "tests_main.rs"]
mod tests;
