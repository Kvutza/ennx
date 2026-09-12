use std::collections::BTreeSet;
use std::fs::{self, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use ndarray::{Array2, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileConfig {
    version: u32,
    eval: EvalConfig,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct EvalConfig {
    name: String,
    output: PathBuf,
    budget: usize,
    num_init: usize,
    seeds: Vec<u64>,
    optimizers: Vec<OptimizerSpec>,
    tasks: Vec<TaskSpec>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct OptimizerSpec {
    name: String,
    kind: OptimizerKind,
    #[serde(default = "default_neighbors")]
    neighbors: i32,
    #[serde(default = "default_candidates")]
    candidates: usize,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum OptimizerKind {
    TurboEnn,
    TurboZero,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct TaskSpec {
    name: String,
    function: Function,
    dimensions: usize,
    #[serde(default = "default_lower")]
    lower: f64,
    #[serde(default = "default_upper")]
    upper: f64,
    #[serde(default)]
    noise_std: f64,
    #[serde(default = "default_target")]
    target: f64,
    active_dimensions: Option<usize>,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum Function {
    Sphere,
    Ellipsoid,
    Rosenbrock,
    Rastrigin,
    Ackley,
    SparseSphere,
}

#[derive(Debug)]
struct TaskInstance<'a> {
    spec: &'a TaskSpec,
    center: Vec<f64>,
    seed: u64,
}

#[derive(Debug)]
struct RunSummary {
    task: String,
    optimizer: String,
    seed: u64,
    final_regret: Option<f64>,
    normalized_auc: Option<f64>,
    target_eval: Option<usize>,
    seconds: f64,
    error: Option<String>,
}

fn default_neighbors() -> i32 {
    8
}

fn default_candidates() -> usize {
    512
}

fn default_lower() -> f64 {
    -5.0
}

fn default_upper() -> f64 {
    5.0
}

fn default_target() -> f64 {
    1e-6
}

pub(crate) fn run(root: &Path, path: &Path, output_override: Option<&Path>) -> Result<(), String> {
    let text = fs::read_to_string(path)
        .map_err(|error| format!("read eval config {}: {error}", path.display()))?;
    let file: FileConfig = toml::from_str(&text)
        .map_err(|error| format!("invalid config {}: {error}", path.display()))?;
    validate(&file)?;

    let declared_output = output_override.unwrap_or(&file.eval.output);
    let output = absolute(root, declared_output);
    let summary_path = output.with_extension("md");
    if output.exists() || summary_path.exists() {
        return Err(format!(
            "eval artifacts already exist; choose a fresh --output ({} or {})",
            output.display(),
            summary_path.display()
        ));
    }
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("create output directory {}: {error}", parent.display()))?;
    }
    let file_out = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&output)
        .map_err(|error| format!("create {}: {error}", output.display()))?;
    let mut writer = BufWriter::new(file_out);
    emit(
        &mut writer,
        &json!({
            "record": "protocol",
            "schema": "ennx.optimizer_eval.v1",
            "suite": file.eval.name,
            "config": path,
            "budget": file.eval.budget,
            "num_init": file.eval.num_init,
            "seeds": file.eval.seeds,
            "pairing": "identical external LHD initialization and per-evaluation common random numbers across policies",
            "objective_direction": "minimize",
            "budget_unit": "objective_evaluations",
            "config_text": text,
        }),
    )?;

    println!(
        "Evaluating {} optimizers on {} tasks × {} seeds ({} trials)...",
        file.eval.optimizers.len(),
        file.eval.tasks.len(),
        file.eval.seeds.len(),
        file.eval.optimizers.len() * file.eval.tasks.len() * file.eval.seeds.len()
    );
    let mut runs = Vec::new();
    for task in &file.eval.tasks {
        for (seed_index, &seed) in file.eval.seeds.iter().enumerate() {
            let instance = TaskInstance::new(task, seed);
            // Alternating policy order prevents a fixed order from becoming a timing confound.
            let policies: Vec<_> = if seed_index % 2 == 0 {
                file.eval.optimizers.iter().collect()
            } else {
                file.eval.optimizers.iter().rev().collect()
            };
            for policy in policies {
                let started = Instant::now();
                let result = run_trial(&file.eval, &instance, policy, seed, &mut writer);
                let seconds = started.elapsed().as_secs_f64();
                let summary = match result {
                    Ok((final_regret, normalized_auc, target_eval)) => RunSummary {
                        task: task.name.clone(),
                        optimizer: policy.name.clone(),
                        seed,
                        final_regret: Some(final_regret),
                        normalized_auc: Some(normalized_auc),
                        target_eval,
                        seconds,
                        error: None,
                    },
                    Err(error) => RunSummary {
                        task: task.name.clone(),
                        optimizer: policy.name.clone(),
                        seed,
                        final_regret: None,
                        normalized_auc: None,
                        target_eval: None,
                        seconds,
                        error: Some(error),
                    },
                };
                emit_run(&mut writer, &summary)?;
                runs.push(summary);
            }
        }
    }

    let (table, comparisons) = emit_aggregates(&mut writer, &file.eval, &runs)?;
    emit(
        &mut writer,
        &json!({"record": "completed", "schema": "ennx.optimizer_eval.v1"}),
    )?;
    writer
        .flush()
        .map_err(|error| format!("flush {}: {error}", output.display()))?;
    write_summary(&summary_path, &file.eval, &table, &comparisons)?;
    println!("Raw eval records: {}", output.display());
    println!("Scorecard: {}", summary_path.display());
    Ok(())
}

fn run_trial(
    config: &EvalConfig,
    task: &TaskInstance<'_>,
    policy: &OptimizerSpec,
    seed: u64,
    writer: &mut impl Write,
) -> Result<(f64, f64, Option<usize>), String> {
    let trial_seed = mix(seed ^ stable_hash(&task.spec.name));
    // The Optimizer API operates in the unit hypercube. Task bounds are applied
    // only at the objective boundary.
    let bounds = Array2::from_shape_fn((task.spec.dimensions, 2), |(_, column)| {
        if column == 0 { 0.0 } else { 1.0 }
    });
    let overrides = ennx::ConfigOverrides {
        min_candidates: Some(policy.candidates),
        max_candidates: Some(policy.candidates),
        ..Default::default()
    };
    let mut construction_rng = StdRng::seed_from_u64(mix(trial_seed ^ 0x434f4e5354525543));
    let mut optimizer = match policy.kind {
        OptimizerKind::TurboEnn => ennx::optimizer_factory::enn_overrides(
            bounds.clone(),
            policy.neighbors,
            0,
            &mut construction_rng,
            Some(&overrides),
        ),
        OptimizerKind::TurboZero => ennx::optimizer_factory::create_overrides(
            bounds.clone(),
            0,
            &mut construction_rng,
            Some(&overrides),
        ),
    }
    .map_err(|error| format!("construct optimizer: {error}"))?;

    let mut best = f64::INFINITY;
    let mut initial_gap = None;
    let mut auc_sum = 0.0;
    let mut target_eval = None;
    let mut initial_rng = StdRng::seed_from_u64(mix(trial_seed ^ 0x494e495449414c));
    let initial = ennx::generate_lhd(
        config.num_init,
        task.spec.dimensions,
        &bounds.view(),
        &mut initial_rng,
    );
    let mut initial_y = Vec::with_capacity(config.num_init);
    for (row, x) in initial.rows().into_iter().enumerate() {
        let evaluation = row + 1;
        let value = task.value(x);
        let observed = value + task.noise(evaluation);
        initial_y.push(-observed);
        record_evaluation(
            writer,
            config,
            task,
            policy,
            seed,
            evaluation,
            x,
            value,
            observed,
            &mut best,
            &mut initial_gap,
            &mut auc_sum,
            &mut target_eval,
        )?;
    }
    let initial_y = Array2::from_shape_vec((config.num_init, 1), initial_y)
        .map_err(|error| format!("construct initial observations: {error}"))?;
    let mut fit_rng = StdRng::seed_from_u64(mix(trial_seed ^ 0x464954));
    optimizer
        .tell(&initial.view(), &initial_y.view(), &mut fit_rng)
        .map_err(|error| format!("tell initial design: {error}"))?;

    for evaluation in config.num_init + 1..=config.budget {
        let mut step_rng =
            StdRng::seed_from_u64(mix(trial_seed ^ (evaluation as u64).wrapping_mul(0x9e37)));
        let candidate = optimizer
            .ask(1, &mut step_rng)
            .map_err(|error| format!("ask at evaluation {evaluation}: {error}"))?;
        let x = candidate.row(0);
        let value = task.value(x);
        let observed = value + task.noise(evaluation);
        record_evaluation(
            writer,
            config,
            task,
            policy,
            seed,
            evaluation,
            x,
            value,
            observed,
            &mut best,
            &mut initial_gap,
            &mut auc_sum,
            &mut target_eval,
        )?;
        let y = Array2::from_shape_vec((1, 1), vec![-observed])
            .map_err(|error| format!("construct observation: {error}"))?;
        optimizer
            .tell(&candidate.view(), &y.view(), &mut step_rng)
            .map_err(|error| format!("tell at evaluation {evaluation}: {error}"))?;
    }
    Ok((best, auc_sum / config.budget as f64, target_eval))
}

#[allow(clippy::too_many_arguments)]
fn record_evaluation(
    writer: &mut impl Write,
    config: &EvalConfig,
    task: &TaskInstance<'_>,
    policy: &OptimizerSpec,
    seed: u64,
    evaluation: usize,
    x: ArrayView1<'_, f64>,
    value: f64,
    observed: f64,
    best: &mut f64,
    initial_gap: &mut Option<f64>,
    auc_sum: &mut f64,
    target_eval: &mut Option<usize>,
) -> Result<(), String> {
    *best = best.min(value);
    let gap = (*best).max(0.0);
    let initial = *initial_gap.get_or_insert(gap.max(1e-15));
    let normalized_regret = (gap / initial).min(1.0);
    *auc_sum += normalized_regret;
    if target_eval.is_none() && gap <= task.spec.target {
        *target_eval = Some(evaluation);
    }
    emit(
        writer,
        &json!({
            "record": "evaluation",
            "suite": config.name,
            "task": task.spec.name,
            "function": task.spec.function,
            "dimensions": task.spec.dimensions,
            "optimizer": policy.name,
            "optimizer_kind": policy.kind,
            "seed": seed,
            "evaluation": evaluation,
            "value": value,
            "observed_value": observed,
            "best_value": *best,
            "simple_regret": gap,
            "normalized_regret": normalized_regret,
            "target_hit": gap <= task.spec.target,
            "candidate_unit_l2": x.iter().map(|v| v * v).sum::<f64>().sqrt(),
            "candidate_hash": candidate_hash(x),
        }),
    )
}

impl<'a> TaskInstance<'a> {
    fn new(spec: &'a TaskSpec, seed: u64) -> Self {
        let width = spec.upper - spec.lower;
        let center = (0..spec.dimensions)
            .map(|dimension| {
                let unit = unit_uniform(mix(seed ^ dimension as u64));
                (unit - 0.5) * width * 0.2
            })
            .collect();
        Self { spec, center, seed }
    }

    fn value(&self, x: ArrayView1<'_, f64>) -> f64 {
        let width = self.spec.upper - self.spec.lower;
        let mut z: Vec<_> = x
            .iter()
            .zip(&self.center)
            .map(|(&unit, &center)| self.spec.lower + unit * width - center)
            .collect();
        match self.spec.function {
            Function::Sphere => z.iter().map(|value| value * value).sum(),
            Function::SparseSphere => z
                .iter()
                .take(self.spec.active_dimensions.unwrap_or(8).min(z.len()))
                .map(|value| value * value)
                .sum(),
            Function::Ellipsoid => {
                rotate(&mut z, self.seed);
                let denominator = (z.len().saturating_sub(1)).max(1) as f64;
                z.iter()
                    .enumerate()
                    .map(|(index, value)| {
                        1_000_000_f64.powf(index as f64 / denominator) * value * value
                    })
                    .sum()
            }
            Function::Rosenbrock => {
                for value in &mut z {
                    *value += 1.0;
                }
                z.windows(2)
                    .map(|pair| {
                        let a = pair[1] - pair[0] * pair[0];
                        let b = 1.0 - pair[0];
                        100.0 * a * a + b * b
                    })
                    .sum()
            }
            Function::Rastrigin => {
                10.0 * z.len() as f64
                    + z.iter()
                        .map(|value| {
                            value * value - 10.0 * (2.0 * std::f64::consts::PI * value).cos()
                        })
                        .sum::<f64>()
            }
            Function::Ackley => {
                let n = z.len() as f64;
                let squares = z.iter().map(|value| value * value).sum::<f64>() / n;
                let cosines = z
                    .iter()
                    .map(|value| (2.0 * std::f64::consts::PI * value).cos())
                    .sum::<f64>()
                    / n;
                -20.0 * (-0.2 * squares.sqrt()).exp() - cosines.exp() + 20.0 + std::f64::consts::E
            }
        }
    }

    fn noise(&self, evaluation: usize) -> f64 {
        if self.spec.noise_std == 0.0 {
            return 0.0;
        }
        let u1 = unit_uniform(mix(self.seed ^ evaluation as u64)).max(f64::MIN_POSITIVE);
        let u2 = unit_uniform(mix(self.seed ^ (evaluation as u64).wrapping_mul(0x9e37)));
        self.spec.noise_std * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

#[derive(Debug, Serialize)]
struct Aggregate {
    task: String,
    optimizer: String,
    completed: usize,
    failed: usize,
    median_final_regret: Option<f64>,
    median_normalized_auc: Option<f64>,
    success_rate: f64,
    median_evals_to_target: Option<f64>,
    total_seconds: f64,
}

#[derive(Debug, Serialize)]
struct Comparison {
    task: String,
    left: String,
    right: String,
    pairs: usize,
    left_wins: usize,
    ties: usize,
    left_losses: usize,
    median_final_regret_delta: Option<f64>,
    median_normalized_auc_delta: Option<f64>,
}

fn emit_aggregates(
    writer: &mut impl Write,
    config: &EvalConfig,
    runs: &[RunSummary],
) -> Result<(Vec<Aggregate>, Vec<Comparison>), String> {
    let mut table = Vec::new();
    for task in &config.tasks {
        for policy in &config.optimizers {
            let group: Vec<_> = runs
                .iter()
                .filter(|run| run.task == task.name && run.optimizer == policy.name)
                .collect();
            let final_regrets: Vec<_> = group.iter().filter_map(|run| run.final_regret).collect();
            let aucs: Vec<_> = group.iter().filter_map(|run| run.normalized_auc).collect();
            let targets: Vec<_> = group
                .iter()
                .filter_map(|run| run.target_eval.map(|value| value as f64))
                .collect();
            let completed = final_regrets.len();
            let aggregate = Aggregate {
                task: task.name.clone(),
                optimizer: policy.name.clone(),
                completed,
                failed: group.len() - completed,
                median_final_regret: median(final_regrets),
                median_normalized_auc: median(aucs),
                success_rate: targets.len() as f64 / group.len() as f64,
                median_evals_to_target: median(targets),
                total_seconds: group.iter().map(|run| run.seconds).sum(),
            };
            emit(
                writer,
                &json!({"record": "aggregate", "metrics": aggregate}),
            )?;
            table.push(aggregate);
        }
    }
    let mut comparisons = Vec::new();
    for task in &config.tasks {
        for left_index in 0..config.optimizers.len() {
            for right_index in left_index + 1..config.optimizers.len() {
                let left = &config.optimizers[left_index];
                let right = &config.optimizers[right_index];
                let mut regret_deltas = Vec::new();
                let mut auc_deltas = Vec::new();
                let mut wins = 0;
                let mut ties = 0;
                let mut losses = 0;
                for &seed in &config.seeds {
                    let left_run = runs.iter().find(|run| {
                        run.task == task.name && run.optimizer == left.name && run.seed == seed
                    });
                    let right_run = runs.iter().find(|run| {
                        run.task == task.name && run.optimizer == right.name && run.seed == seed
                    });
                    let (Some(left_run), Some(right_run)) = (left_run, right_run) else {
                        continue;
                    };
                    let (Some(left_regret), Some(right_regret)) =
                        (left_run.final_regret, right_run.final_regret)
                    else {
                        continue;
                    };
                    let delta = left_regret - right_regret;
                    regret_deltas.push(delta);
                    if let (Some(left_auc), Some(right_auc)) =
                        (left_run.normalized_auc, right_run.normalized_auc)
                    {
                        auc_deltas.push(left_auc - right_auc);
                    }
                    let tolerance = 1e-12 * left_regret.abs().max(right_regret.abs()).max(1.0);
                    if delta < -tolerance {
                        wins += 1;
                    } else if delta > tolerance {
                        losses += 1;
                    } else {
                        ties += 1;
                    }
                }
                let comparison = Comparison {
                    task: task.name.clone(),
                    left: left.name.clone(),
                    right: right.name.clone(),
                    pairs: regret_deltas.len(),
                    left_wins: wins,
                    ties,
                    left_losses: losses,
                    median_final_regret_delta: median(regret_deltas),
                    median_normalized_auc_delta: median(auc_deltas),
                };
                emit(
                    writer,
                    &json!({"record": "comparison", "metrics": comparison}),
                )?;
                comparisons.push(comparison);
            }
        }
    }
    Ok((table, comparisons))
}

fn emit_run(writer: &mut impl Write, run: &RunSummary) -> Result<(), String> {
    emit(
        writer,
        &json!({
            "record": "trial",
            "task": run.task,
            "optimizer": run.optimizer,
            "seed": run.seed,
            "final_regret": run.final_regret,
            "normalized_auc": run.normalized_auc,
            "target_eval": run.target_eval,
            "seconds": run.seconds,
            "error": run.error,
        }),
    )
}

fn emit(writer: &mut impl Write, value: &serde_json::Value) -> Result<(), String> {
    serde_json::to_writer(&mut *writer, value).map_err(|error| error.to_string())?;
    writer.write_all(b"\n").map_err(|error| error.to_string())
}

fn write_summary(
    path: &Path,
    config: &EvalConfig,
    rows: &[Aggregate],
    comparisons: &[Comparison],
) -> Result<(), String> {
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| format!("create {}: {error}", path.display()))?;
    let mut writer = BufWriter::new(file);
    writeln!(writer, "# ENNX optimizer eval: {}\n", config.name).map_err(|e| e.to_string())?;
    writeln!(
        writer,
        "Budget: {} objective evaluations per trial; {} paired seeds. Lower regret and AUC are better.\n",
        config.budget,
        config.seeds.len()
    )
    .map_err(|e| e.to_string())?;
    writeln!(writer, "| Task | Optimizer | Complete | Failed | Median final regret | Median normalized AUC | Success | Median evals to target |")
        .map_err(|e| e.to_string())?;
    writeln!(writer, "|---|---|---:|---:|---:|---:|---:|---:|").map_err(|e| e.to_string())?;
    for row in rows {
        writeln!(
            writer,
            "| {} | {} | {} | {} | {} | {} | {:.1}% | {} |",
            row.task,
            row.optimizer,
            row.completed,
            row.failed,
            display_metric(row.median_final_regret),
            display_metric(row.median_normalized_auc),
            row.success_rate * 100.0,
            display_metric(row.median_evals_to_target),
        )
        .map_err(|e| e.to_string())?;
    }
    writeln!(writer, "\n## Paired comparisons\n").map_err(|e| e.to_string())?;
    writeln!(
        writer,
        "Deltas are left minus right; negative is better for the left optimizer.\n"
    )
    .map_err(|e| e.to_string())?;
    writeln!(
        writer,
        "| Task | Left vs right | Pairs | W-T-L | Median final-regret delta | Median AUC delta |"
    )
    .map_err(|e| e.to_string())?;
    writeln!(writer, "|---|---|---:|---:|---:|---:|").map_err(|e| e.to_string())?;
    for row in comparisons {
        writeln!(
            writer,
            "| {} | {} vs {} | {} | {}-{}-{} | {} | {} |",
            row.task,
            row.left,
            row.right,
            row.pairs,
            row.left_wins,
            row.ties,
            row.left_losses,
            display_metric(row.median_final_regret_delta),
            display_metric(row.median_normalized_auc_delta),
        )
        .map_err(|e| e.to_string())?;
    }
    writer.flush().map_err(|error| error.to_string())
}

fn display_metric(value: Option<f64>) -> String {
    value.map_or_else(|| "—".into(), |metric| format!("{metric:.6e}"))
}

fn median(mut values: Vec<f64>) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    Some(if values.len() % 2 == 0 {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    })
}

fn validate(file: &FileConfig) -> Result<(), String> {
    if file.version != 1 {
        return Err(format!("unsupported eval config version {}", file.version));
    }
    let config = &file.eval;
    if config.name.trim().is_empty() || config.output.as_os_str().is_empty() {
        return Err("eval name and output must be non-empty".into());
    }
    if config.seeds.is_empty() || config.optimizers.is_empty() || config.tasks.is_empty() {
        return Err("eval requires seeds, optimizers, and tasks".into());
    }
    if config.num_init == 0 || config.budget < config.num_init {
        return Err("eval budget must be at least the positive num_init".into());
    }
    unique(
        config.optimizers.iter().map(|item| item.name.as_str()),
        "optimizer",
    )?;
    unique(config.tasks.iter().map(|item| item.name.as_str()), "task")?;
    let seed_count = config.seeds.iter().copied().collect::<BTreeSet<_>>().len();
    if seed_count != config.seeds.len() {
        return Err("eval seeds must be unique".into());
    }
    for policy in &config.optimizers {
        validate_optimizer(policy, config.num_init)?;
    }
    for task in &config.tasks {
        validate_task(task)?;
    }
    Ok(())
}

fn validate_optimizer(policy: &OptimizerSpec, num_init: usize) -> Result<(), String> {
    if policy.name.trim().is_empty() || policy.neighbors <= 0 || policy.candidates == 0 {
        return Err(format!("invalid optimizer {}", policy.name));
    }
    if matches!(policy.kind, OptimizerKind::TurboEnn) && policy.neighbors as usize > num_init {
        return Err(format!(
            "optimizer {} neighbors must not exceed num_init",
            policy.name
        ));
    }
    Ok(())
}

fn validate_task(task: &TaskSpec) -> Result<(), String> {
    if task.name.trim().is_empty()
        || task.dimensions == 0
        || task.dimensions > 100_000
        || !task.lower.is_finite()
        || !task.upper.is_finite()
        || task.lower >= task.upper
        || !task.noise_std.is_finite()
        || task.noise_std < 0.0
        || !task.target.is_finite()
        || task.target < 0.0
    {
        return Err(format!("invalid task {}", task.name));
    }
    if matches!(task.function, Function::Rosenbrock) && task.dimensions < 2 {
        return Err(format!(
            "task {} requires at least two dimensions",
            task.name
        ));
    }
    if let Some(active) = task.active_dimensions {
        if !matches!(task.function, Function::SparseSphere)
            || active == 0
            || active > task.dimensions
        {
            return Err(format!("invalid active_dimensions for task {}", task.name));
        }
    }
    Ok(())
}

fn unique<'a>(names: impl Iterator<Item = &'a str>, kind: &str) -> Result<(), String> {
    let mut seen = BTreeSet::new();
    for name in names {
        if !seen.insert(name) {
            return Err(format!("duplicate {kind} name {name:?}"));
        }
    }
    Ok(())
}

fn absolute(root: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    }
}

fn rotate(values: &mut [f64], seed: u64) {
    for index in 0..values.len().saturating_sub(1) {
        let angle = unit_uniform(mix(seed ^ index as u64)) * std::f64::consts::PI;
        let (sin, cos) = angle.sin_cos();
        let left = values[index];
        let right = values[index + 1];
        values[index] = cos * left - sin * right;
        values[index + 1] = sin * left + cos * right;
    }
}

fn stable_hash(value: &str) -> u64 {
    value.bytes().fold(0xcbf29ce484222325, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
    })
}

fn candidate_hash(values: ArrayView1<'_, f64>) -> String {
    let hash = values.iter().fold(0xcbf29ce484222325, |hash, value| {
        (hash ^ value.to_bits()).wrapping_mul(0x100000001b3)
    });
    format!("{hash:016x}")
}

fn mix(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}

fn unit_uniform(value: u64) -> f64 {
    (value >> 11) as f64 * (1.0 / ((1_u64 << 53) as f64))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sphere() -> TaskSpec {
        TaskSpec {
            name: "sphere".into(),
            function: Function::Sphere,
            dimensions: 8,
            lower: -5.0,
            upper: 5.0,
            noise_std: 0.0,
            target: 1e-6,
            active_dimensions: None,
        }
    }

    #[test]
    fn task_optima_are_zero() {
        let functions = [
            Function::Sphere,
            Function::Ellipsoid,
            Function::Rosenbrock,
            Function::Rastrigin,
            Function::Ackley,
            Function::SparseSphere,
        ];
        for function in functions {
            let mut spec = sphere();
            spec.function = function;
            let instance = TaskInstance::new(&spec, 7);
            let width = spec.upper - spec.lower;
            let unit_optimum = instance
                .center
                .iter()
                .map(|center| (center - spec.lower) / width)
                .collect();
            let optimum = Array2::from_shape_vec((1, 8), unit_optimum).unwrap();
            assert!(instance.value(optimum.row(0)).abs() < 1e-12, "{function:?}");
        }
    }

    #[test]
    fn rejects_unpaired_or_invalid_suites() {
        assert!(unique(["same", "same"].into_iter(), "optimizer").is_err());
        let policy = OptimizerSpec {
            name: "enn".into(),
            kind: OptimizerKind::TurboEnn,
            neighbors: 9,
            candidates: 16,
        };
        assert!(
            validate_optimizer(&policy, 8)
                .unwrap_err()
                .contains("num_init")
        );
    }

    #[test]
    fn deterministic_noise() {
        let mut spec = sphere();
        spec.noise_std = 0.5;
        let left = TaskInstance::new(&spec, 99);
        let right = TaskInstance::new(&spec, 99);
        assert_eq!(left.noise(3), right.noise(3));
    }

    #[test]
    fn median_handles_even_and_empty() {
        assert_eq!(median(Vec::new()), None);
        assert_eq!(median(vec![3.0, 1.0, 2.0, 4.0]), Some(2.5));
    }
}
