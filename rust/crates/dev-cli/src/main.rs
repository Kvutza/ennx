use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};
use std::time::Instant;

use clap::{Parser, Subcommand};

mod tune;
use tune::{parse_config, parse_knn, KnnTuneConfig, ProposalTuneConfig};

const VERSION: &str = "0.2.0";
#[derive(Debug, Parser)]
#[command(
    name = "ennx",
    bin_name = "./ennx",
    version = VERSION,
    about = "Tune ENNX experiments through Buck2",
    arg_required_else_help = true
)]
struct Cli {
    #[command(subcommand)]
    action: Action,
}

#[derive(Debug, PartialEq, Eq, Subcommand)]
enum Action {
    /// Run a configured experiment.
    Tune {
        #[command(subcommand)]
        target: TuneTarget,
    },
}

#[derive(Debug, PartialEq, Eq, Subcommand)]
enum TuneTarget {
    /// Run a KNN frontier experiment.
    Knn { config: PathBuf },
    /// Run a resident proposal experiment.
    Proposal { config: PathBuf },
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("ennx: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let cli = Cli::parse();
    let root = env::current_dir().map_err(|error| error.to_string())?;
    match cli.action {
        Action::Tune {
            target: TuneTarget::Knn { config },
        } => tune_knn(&root, &config)?,
        Action::Tune {
            target: TuneTarget::Proposal { config },
        } => tune_proposal(&root, &config)?,
    }
    Ok(())
}

fn tune_knn(root: &Path, path: &Path) -> Result<(), String> {
    let config = knn_config(path)?;
    let output = root.join(&config.output);
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent).map_err(|error| format!("create output directory: {error}"))?;
    }
    println!("Running KNN frontier tuning...");
    let rounds = config.rounds.to_string();
    let mut args = vec![
        "--isolation-dir",
        "dev",
        "run",
        "//rust/crates/ennx:knn_frontier",
        "--",
        config.output.as_str(),
        rounds.as_str(),
    ];
    for point in &config.points {
        args.push(point.as_str());
    }
    command(root, "./buck2w", &args)
}

fn tune_proposal(root: &Path, path: &Path) -> Result<(), String> {
    let config = load_config(path)?;
    let output = root.join(&config.output);
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent).map_err(|error| format!("create output directory: {error}"))?;
    }
    let estimated_bytes = estimate_bytes(&config)?;
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
        "--isolation-dir".to_string(),
        "dev".to_string(),
        "run".to_string(),
        "//rust/crates/ennx:trial_bench".to_string(),
        "--".to_string(),
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
    let arg_refs = args.iter().map(String::as_str).collect::<Vec<_>>();
    let stdout = command_output(root, "./buck2w", &arg_refs)?;
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
    report.push_str(&format!("# orchestration_s={command_s:.9}\n"));
    report.push_str(&stdout);
    fs::write(&output, report).map_err(|error| format!("write {}: {error}", output.display()))?;
    println!("Proposal benchmark written to {}", output.display());
    Ok(())
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

fn command(root: &Path, program: &str, args: &[&str]) -> Result<(), String> {
    let status = Command::new(program)
        .current_dir(root)
        .args(args)
        .status()
        .map_err(|error| format!("start {program}: {error}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("{program} exited with {status}"))
    }
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
