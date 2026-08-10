use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::Instant;

use ennx::experimental::DenseTerm;
use ennx::{
    turbo_enn_config, AcquisitionConfig, Optimizer, Strategy, TRLengthConfig, TrustRegionConfig,
};
use ndarray::array;
use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};
use toml::Value;

struct TuneConfig {
    melville: PathBuf,
    workspace: PathBuf,
    task: PathBuf,
    verifier: String,
    reset: String,
    model: String,
    turns: u32,
    rounds: usize,
    candidates: usize,
    seed: u64,
    radius: f64,
    history: usize,
}

struct Evaluator {
    child: Child,
    input: ChildStdin,
    output: BufReader<ChildStdout>,
    stopped: bool,
}

struct Evaluation {
    reward: f64,
    tensors: usize,
    elements: usize,
    perturb_ms: f64,
}

pub(super) fn run(root: &Path, args: &[String]) -> Result<u8, String> {
    let [path] = args else {
        return Err("usage: ennx tune CONFIG.toml".into());
    };
    let config = load_config(root, Path::new(path))?;
    let mut evaluator = Evaluator::spawn(&config)?;
    let baseline_started = Instant::now();
    let baseline = evaluator.evaluate(&[])?;
    println!(
        "baseline reward={:.6} tensors={} parameters={} evaluate_ms={:.3}",
        baseline.reward,
        baseline.tensors,
        baseline.elements,
        baseline_started.elapsed().as_secs_f64() * 1000.0
    );

    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut optimizer_config = turbo_enn_config();
    optimizer_config.acquisition = AcquisitionConfig::UCB { beta: 2.0 };
    optimizer_config.failure_tolerance_dim = Some(baseline.elements as f64);
    optimizer_config.trust_region = TrustRegionConfig::Turbo(TRLengthConfig::new(
        config.radius,
        config.radius / 128.0,
        config.radius * 2.0,
    ));
    let mut optimizer = Optimizer::new_with_strategy(
        array![[0.0, 1.0]],
        optimizer_config,
        Strategy::turbo(),
        &mut rng,
    )
    .map_err(|error| error.to_string())?;
    optimizer
        .enable_program(baseline.reward, config.history)
        .map_err(|error| error.to_string())?;

    for round in 1..=config.rounds {
        let seeds = (0..config.candidates)
            .map(|_| rng.next_u64())
            .collect::<Vec<_>>();
        let trial = optimizer
            .ask_program(&seeds, &mut rng)
            .map_err(|error| error.to_string())?;
        let telemetry = optimizer.telemetry().clone();
        let eval_started = Instant::now();
        let evaluation = evaluator.evaluate(&trial.terms)?;
        let evaluate_ms = eval_started.elapsed().as_secs_f64() * 1000.0;
        optimizer
            .tell_program(trial, evaluation.reward, &mut rng)
            .map_err(|error| error.to_string())?;
        let best = optimizer
            .best_program()
            .map_or(baseline.reward, |item| item.1);
        println!(
            "round={round} fit_ms={:.3} propose_ms={:.3} perturb_ms={:.3} evaluate_ms={evaluate_ms:.3} reward={:.6} best={best:.6} radius={:.8}",
            telemetry.dt_fit * 1000.0,
            telemetry.dt_gen * 1000.0,
            evaluation.perturb_ms,
            evaluation.reward,
            optimizer.tr_length(),
        );
    }
    evaluator.stop()?;
    Ok(0)
}

impl Evaluator {
    fn spawn(config: &TuneConfig) -> Result<Self, String> {
        let mut child = Command::new("./ctl")
            .args([
                "buck2",
                "run",
                "//:tune-worker",
                "--",
                config
                    .workspace
                    .to_str()
                    .ok_or("workspace path is not UTF-8")?,
                config.task.to_str().ok_or("task path is not UTF-8")?,
                &config.verifier,
                &config.reset,
                &config.model,
                &config.turns.to_string(),
            ])
            .current_dir(&config.melville)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|error| format!("start Melville evaluator: {error}"))?;
        let input = child
            .stdin
            .take()
            .ok_or("Melville evaluator stdin unavailable")?;
        let output = child
            .stdout
            .take()
            .ok_or("Melville evaluator stdout unavailable")?;
        Ok(Self {
            child,
            input,
            output: BufReader::new(output),
            stopped: false,
        })
    }

    fn evaluate(&mut self, terms: &[DenseTerm]) -> Result<Evaluation, String> {
        writeln!(self.input, "EVAL {}", terms.len()).map_err(|error| error.to_string())?;
        for term in terms {
            writeln!(self.input, "{} {}", term.seed, term.coefficient)
                .map_err(|error| error.to_string())?;
        }
        self.input.flush().map_err(|error| error.to_string())?;
        let mut line = String::new();
        loop {
            line.clear();
            if self
                .output
                .read_line(&mut line)
                .map_err(|error| error.to_string())?
                == 0
            {
                return Err("Melville evaluator exited before returning a result".into());
            }
            if line.starts_with("RESULT ") {
                return parse_result(&line);
            }
        }
    }

    fn stop(&mut self) -> Result<(), String> {
        writeln!(self.input, "STOP").map_err(|error| error.to_string())?;
        self.input.flush().map_err(|error| error.to_string())?;
        let status = self.child.wait().map_err(|error| error.to_string())?;
        self.stopped = true;
        if status.success() {
            Ok(())
        } else {
            Err(format!("Melville evaluator exited with {status}"))
        }
    }
}

impl Drop for Evaluator {
    fn drop(&mut self) {
        if !self.stopped {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

fn parse_result(line: &str) -> Result<Evaluation, String> {
    let fields = line.split_whitespace().collect::<Vec<_>>();
    if fields.len() != 10 {
        return Err(format!("invalid Melville result: {line}"));
    }
    let evaluation = Evaluation {
        reward: parse_field(fields[1], "reward")?,
        tensors: parse_field(fields[2], "tensor count")?,
        elements: parse_field(fields[3], "parameter count")?,
        perturb_ms: parse_field(fields[4], "perturbation time")?,
    };
    if !evaluation.reward.is_finite() {
        return Err("Melville reward must be finite".into());
    }
    if evaluation.tensors == 0 || evaluation.elements == 0 {
        return Err("Melville applied no model weights".into());
    }
    if !evaluation.perturb_ms.is_finite() || evaluation.perturb_ms < 0.0 {
        return Err("Melville perturbation time must be finite and nonnegative".into());
    }
    parse_flag(fields[5], "verifier result")?;
    let tests_passed = parse_field::<u32>(fields[6], "passed test count")?;
    let tests_total = parse_field::<u32>(fields[7], "total test count")?;
    if tests_passed > tests_total {
        return Err("Melville passed test count exceeds total test count".into());
    }
    parse_flag(fields[8], "workspace change flag")?;
    parse_flag(fields[9], "clean exit flag")?;
    Ok(evaluation)
}

fn parse_field<T: std::str::FromStr>(text: &str, name: &str) -> Result<T, String> {
    text.parse().map_err(|_| format!("invalid {name}: {text}"))
}

fn parse_flag(text: &str, name: &str) -> Result<(), String> {
    match text {
        "0" | "1" => Ok(()),
        _ => Err(format!("invalid {name}: {text}")),
    }
}

fn load_config(root: &Path, path: &Path) -> Result<TuneConfig, String> {
    let path = if path.is_absolute() {
        path.to_owned()
    } else {
        root.join(path)
    };
    let text = fs::read_to_string(&path).map_err(|error| error.to_string())?;
    let value = text.parse::<Value>().map_err(|error| error.to_string())?;
    let base = path.parent().unwrap_or(root);
    let config = TuneConfig {
        melville: config_path(&value, base, "melville")?,
        workspace: config_path(&value, base, "workspace")?,
        task: config_path(&value, base, "task")?,
        verifier: config_str(&value, "verifier")?.into(),
        reset: config_str(&value, "reset")?.into(),
        model: config_str(&value, "model")?.into(),
        turns: config_u32(&value, "turns")?,
        rounds: config_usize(&value, "rounds")?,
        candidates: config_usize(&value, "candidates")?,
        seed: config_u64(&value, "seed")?,
        radius: config_f64(&value, "radius")?,
        history: config_usize(&value, "history")?,
    };
    config.validate()?;
    Ok(config)
}

impl TuneConfig {
    fn validate(&self) -> Result<(), String> {
        for (name, value) in [
            ("verifier", self.verifier.as_str()),
            ("reset", self.reset.as_str()),
            ("model", self.model.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(format!("{name} must not be empty"));
            }
        }
        if self.turns == 0 {
            return Err("turns must be positive".into());
        }
        if self.rounds == 0 {
            return Err("rounds must be positive".into());
        }
        if self.candidates == 0 {
            return Err("candidates must be positive".into());
        }
        if !self.radius.is_finite() || self.radius <= 0.0 || !(self.radius * 2.0).is_finite() {
            return Err("radius must be finite, positive, and safely scalable".into());
        }
        if self.history < 2 {
            return Err("history must be at least two".into());
        }
        Ok(())
    }
}

fn config_str<'a>(value: &'a Value, key: &str) -> Result<&'a str, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("missing string {key}"))
}

fn config_u64(value: &Value, key: &str) -> Result<u64, String> {
    value
        .get(key)
        .and_then(Value::as_integer)
        .and_then(|item| u64::try_from(item).ok())
        .ok_or_else(|| format!("missing nonnegative integer {key}"))
}

fn config_u32(value: &Value, key: &str) -> Result<u32, String> {
    u32::try_from(config_u64(value, key)?).map_err(|_| format!("integer {key} exceeds u32"))
}

fn config_usize(value: &Value, key: &str) -> Result<usize, String> {
    usize::try_from(config_u64(value, key)?).map_err(|_| format!("integer {key} exceeds usize"))
}

fn config_f64(value: &Value, key: &str) -> Result<f64, String> {
    value
        .get(key)
        .and_then(Value::as_float)
        .ok_or_else(|| format!("missing float {key}"))
}

fn config_path(value: &Value, base: &Path, key: &str) -> Result<PathBuf, String> {
    let path = Path::new(config_str(value, key)?);
    if path.as_os_str().is_empty() {
        return Err(format!("path {key} must not be empty"));
    }
    Ok(if path.is_absolute() {
        path.to_owned()
    } else {
        base.join(path)
    })
}

#[cfg(test)]
#[path = "tests_tune.rs"]
mod tests;
