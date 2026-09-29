use super::artifacts::{Archive, hardware, write_json};
use deser::{Deserialize, Serialize};
use ennx::config::{ConfigOverrides, KernelTrial};
use ennx_wire::json::json;
use std::fs::{self, File};
use std::path::Path;
use std::process::Command;
use std::time::{Duration, Instant};

#[derive(Debug, Serialize, Deserialize)]
#[deser(deny_unknown_fields, rename_all = "kebab-case")]
pub(super) struct Gate {
    pub min_speedup: f64,
    pub win_fraction: f64,
    pub sequence_atol: f64,
    pub token_atol: f64,
}

impl Gate {
    pub fn validate(&self) -> Result<(), String> {
        if !self.min_speedup.is_finite()
            || self.min_speedup <= 1.0
            || !self.win_fraction.is_finite()
            || !(0.5..=1.0).contains(&self.win_fraction)
            || [self.sequence_atol, self.token_atol]
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err("kernel gate requires finite min-speedup > 1, win-fraction in [0.5, 1], and nonnegative error limits".into());
        }
        Ok(())
    }

    pub fn passes(&self, pairs: &[Pair]) -> bool {
        !pairs.is_empty()
            && pairs
                .iter()
                .all(|pair| pair.ratio.is_finite() && pair.ratio > 0.0)
            && median(pairs.iter().map(|pair| pair.ratio)) <= 1.0 / self.min_speedup
            && pairs.iter().filter(|pair| pair.ratio < 1.0).count() as f64 / pairs.len() as f64
                >= self.win_fraction
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
pub(super) struct Seeds {
    proposal: u64,
    acquisition: u64,
}

impl Seeds {
    pub fn draw(count: usize) -> Vec<Self> {
        // TOML integers are signed. No fixed seed list; actual seeds are archived.
        (0..count)
            .map(|_| Self {
                proposal: rand::random::<u64>() & i64::MAX as u64,
                acquisition: rand::random::<u64>() & i64::MAX as u64,
            })
            .collect()
    }
}

#[derive(Debug, Serialize)]
pub(super) struct Pair {
    pub baseline_ms: f64,
    pub candidate_ms: f64,
    pub candidate_max_ms: f64,
    pub ratio: f64,
    baseline_scorer_ms: f64,
    candidate_scorer_ms: f64,
    max_sequence_error: f64,
    max_token_error: f64,
}

impl Pair {
    pub fn saving_ms(&self) -> f64 {
        self.baseline_ms - self.candidate_ms
    }
}

#[derive(Debug, Serialize)]
pub(super) struct Failure {
    stage: &'static str,
    pub message: String,
    pub abort: bool,
}

impl Failure {
    fn new(stage: &'static str, message: impl Into<String>, abort: bool) -> Self {
        Self {
            stage,
            message: message.into(),
            abort,
        }
    }
}

#[derive(Serialize, Deserialize)]
struct Round {
    round: u32,
    candidate_index: u32,
    candidate_seed: u64,
    nominal_radius: f64,
    accepted: bool,
    next_length: f64,
    wall_seconds: f64,
    kernel_check: Check,
}

#[derive(Serialize, Deserialize)]
struct Check {
    sequence_nlls: Vec<f64>,
    token_nlls: Vec<f64>,
    scorer_gpu_seconds: f64,
    complete_round_seconds: f64,
}

struct Run {
    loop_seconds: f64,
    rounds: Vec<Round>,
}

impl Run {
    fn read(path: &Path, expected: u32) -> Result<Self, String> {
        let text =
            fs::read_to_string(path.join("result.toml")).map_err(|error| error.to_string())?;
        let result: ennx_wire::toml::Value =
            ennx_wire::toml::from_str(&text).map_err(|error| error.to_string())?;
        if result.get("status").and_then(|value| value.as_str()) != Some("completed") {
            return Err("worker did not complete".into());
        }
        let loop_seconds = result
            .get("loop_seconds")
            .and_then(|value| value.as_f64())
            .filter(|value| value.is_finite() && *value > 0.0)
            .ok_or("missing positive loop_seconds")?;
        let text =
            fs::read_to_string(path.join("controller.jsonl")).map_err(|error| error.to_string())?;
        let rounds = text
            .lines()
            .map(ennx_wire::json::from_str::<Round>)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| error.to_string())?;
        if rounds.len() != expected as usize
            || rounds.iter().enumerate().any(|(index, round)| {
                round.round as usize != index + 1
                    || !round.wall_seconds.is_finite()
                    || round.wall_seconds <= 0.0
                    || !round.kernel_check.complete_round_seconds.is_finite()
                    || round.kernel_check.complete_round_seconds < round.wall_seconds
                    || !round.kernel_check.scorer_gpu_seconds.is_finite()
                    || round.kernel_check.scorer_gpu_seconds <= 0.0
                    || round.kernel_check.sequence_nlls.len() != 2
                    || round.kernel_check.token_nlls.len() != 8192
            })
        {
            return Err("missing or malformed full-workload round records".into());
        }
        Ok(Self {
            loop_seconds,
            rounds,
        })
    }
}

pub(super) struct Runner<'a> {
    pub root: &'a Path,
    pub worker: &'a Path,
    pub baseline: &'a ConfigOverrides,
    pub archive: &'a Archive,
    pub gate: &'a Gate,
    pub timeout: Duration,
}

impl Runner<'_> {
    pub fn pairs(
        &self,
        directory: &Path,
        trial: &KernelTrial,
        seeds: &[Seeds],
    ) -> Result<Vec<Pair>, Failure> {
        fs::create_dir_all(directory)
            .map_err(|error| Failure::new("artifact", error.to_string(), true))?;
        let mut pairs = Vec::new();
        for (index, seed) in seeds.iter().enumerate() {
            println!(
                "  pair {}/{} | complete {}-round BO loops",
                index + 1,
                seeds.len(),
                self.baseline.rounds()
            );
            let path = directory.join(format!("pair-{:03}", index + 1));
            fs::create_dir(&path)
                .map_err(|error| Failure::new("artifact", error.to_string(), true))?;
            let mut baseline = None;
            let mut candidate = None;
            for use_candidate in [index % 2 == 1, index % 2 == 0] {
                let (name, variant) = if use_candidate {
                    ("candidate", trial.clone())
                } else {
                    ("baseline", KernelTrial::default())
                };
                let run = self
                    .execute(&path.join(name), variant, *seed)
                    .map_err(|mut error| {
                        error.abort |= !use_candidate;
                        error
                    })?;
                if use_candidate {
                    candidate = Some(run);
                } else {
                    baseline = Some(run);
                }
            }
            let pair = compare(
                baseline
                    .as_ref()
                    .ok_or_else(|| Failure::new("runner", "missing baseline", true))?,
                candidate
                    .as_ref()
                    .ok_or_else(|| Failure::new("runner", "missing candidate", true))?,
                self.gate,
            )
            .map_err(|error| Failure::new("correctness", error, false))?;
            println!(
                "  wall/round: {:.3} -> {:.3} ms | token error {:.9}",
                pair.baseline_ms, pair.candidate_ms, pair.max_token_error
            );
            pairs.push(pair);
            write_json(&directory.join("pairs.json"), &pairs)
                .map_err(|error| Failure::new("artifact", error, true))?;
        }
        Ok(pairs)
    }

    fn execute(&self, path: &Path, trial: KernelTrial, seeds: Seeds) -> Result<Run, Failure> {
        let execute = || -> Result<Run, String> {
            fs::create_dir(path).map_err(|error| error.to_string())?;
            let mut config = self.baseline.clone();
            config.kernel_trial = Some(trial);
            config.output = Some(path.to_path_buf());
            config.proposal_seed = Some(seeds.proposal);
            config.acquisition_seed = Some(seeds.acquisition);
            let experiment = path.join("experiment.toml");
            self.archive.write_experiment(&experiment, &config)?;
            write_json(&path.join("environment-before.json"), &hardware())?;
            let log = File::create(path.join("run.log")).map_err(|error| error.to_string())?;
            let mut command = Command::new(self.worker);
            command
                .current_dir(self.root)
                .arg(experiment)
                .arg(path)
                .stdout(log.try_clone().map_err(|error| error.to_string())?)
                .stderr(log);
            let status = wait_worker(&mut command, self.timeout)?;
            write_json(&path.join("environment-after.json"), &hardware())?;
            if !status.success() {
                return Err(format!(
                    "worker exited {status}; see {}/run.log",
                    path.display()
                ));
            }
            Run::read(path, config.rounds())
        };
        match execute() {
            Ok(run) => {
                write_json(
                    &path.join("status.json"),
                    &json!({"stage": "completed", "compile": "passed"}),
                )
                .map_err(|error| Failure::new("artifact", error, true))?;
                Ok(run)
            }
            Err(message) => {
                let log = fs::read_to_string(path.join("run.log")).unwrap_or_default();
                let blocked = log.contains("Low Power Mode is enabled");
                let timed_out = message.starts_with("worker timed out");
                let stage = if timed_out {
                    "timeout"
                } else if blocked {
                    "environment"
                } else if log.contains("ENNX_KERNEL_STAGE execute") {
                    "execution"
                } else if log.contains("ENNX_KERNEL_STAGE compile") {
                    "compile"
                } else {
                    "runner"
                };
                let error = Failure::new(
                    stage,
                    format!("{message}\n{}", tail(&log, 12000)),
                    blocked || timed_out || stage == "runner",
                );
                if path.is_dir() {
                    write_json(&path.join("status.json"), &error)
                        .map_err(|error| Failure::new("artifact", error, true))?;
                }
                Err(error)
            }
        }
    }
}

fn wait_worker(
    command: &mut Command,
    timeout: Duration,
) -> Result<std::process::ExitStatus, String> {
    let mut child = command.spawn().map_err(|error| error.to_string())?;
    let started = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return Ok(status),
            Ok(None) if started.elapsed() < timeout => {
                std::thread::sleep(Duration::from_millis(100))
            }
            status => {
                // Only this trial's direct worker is terminated. No process-name matching.
                let _ = child.kill();
                let _ = child.wait();
                return Err(match status {
                    Err(error) => error.to_string(),
                    _ => format!("worker timed out after {} seconds", timeout.as_secs()),
                });
            }
        }
    }
}

fn tail(text: &str, length: usize) -> &str {
    let mut start = text.len().saturating_sub(length);
    while !text.is_char_boundary(start) {
        start += 1;
    }
    &text[start..]
}

fn max_error(actual: &[f64], expected: &[f64]) -> Result<f64, String> {
    if actual.len() != expected.len() || actual.is_empty() {
        return Err("numerical output shape mismatch".into());
    }
    actual
        .iter()
        .zip(expected)
        .try_fold(0.0f64, |maximum, (a, b)| {
            if !a.is_finite() || !b.is_finite() {
                return Err("nonfinite numerical output".into());
            }
            Ok(maximum.max((a - b).abs()))
        })
}

fn compare(baseline: &Run, candidate: &Run, gate: &Gate) -> Result<Pair, String> {
    if baseline.rounds.len() != candidate.rounds.len() || baseline.rounds.is_empty() {
        return Err("different round budgets".into());
    }
    let mut sequence_error = 0.0f64;
    let mut token_error = 0.0f64;
    for (base, cand) in baseline.rounds.iter().zip(&candidate.rounds) {
        if base.round != cand.round
            || base.candidate_index != cand.candidate_index
            || base.candidate_seed != cand.candidate_seed
            || base.nominal_radius != cand.nominal_radius
            || base.accepted != cand.accepted
            || base.next_length != cand.next_length
        {
            return Err(format!(
                "round {} changed the optimizer trajectory; paired-workload timing is not comparable",
                base.round
            ));
        }
        sequence_error = sequence_error.max(max_error(
            &cand.kernel_check.sequence_nlls,
            &base.kernel_check.sequence_nlls,
        )?);
        token_error = token_error.max(max_error(
            &cand.kernel_check.token_nlls,
            &base.kernel_check.token_nlls,
        )?);
    }
    if sequence_error > gate.sequence_atol || token_error > gate.token_atol {
        return Err(format!(
            "numerical gate failed: sequence error {sequence_error}, token error {token_error}; limits {}, {}",
            gate.sequence_atol, gate.token_atol
        ));
    }
    let rounds = baseline.rounds.len() as f64;
    Ok(Pair {
        baseline_ms: baseline.loop_seconds * 1000.0 / rounds,
        candidate_ms: candidate.loop_seconds * 1000.0 / rounds,
        candidate_max_ms: candidate
            .rounds
            .iter()
            .map(|round| round.kernel_check.complete_round_seconds * 1000.0)
            .fold(0.0, f64::max),
        ratio: candidate.loop_seconds / baseline.loop_seconds,
        baseline_scorer_ms: median(
            baseline
                .rounds
                .iter()
                .map(|round| round.kernel_check.scorer_gpu_seconds * 1000.0),
        ),
        candidate_scorer_ms: median(
            candidate
                .rounds
                .iter()
                .map(|round| round.kernel_check.scorer_gpu_seconds * 1000.0),
        ),
        max_sequence_error: sequence_error,
        max_token_error: token_error,
    })
}

pub(super) fn median(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut values = values.into_iter().collect::<Vec<_>>();
    values.sort_by(f64::total_cmp);
    if values.is_empty() {
        return f64::NAN;
    }
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gate() -> Gate {
        Gate {
            min_speedup: 1.03,
            win_fraction: 0.8,
            sequence_atol: 1e-5,
            token_atol: 2e-3,
        }
    }

    fn run(seconds: f64) -> Run {
        Run {
            loop_seconds: seconds,
            rounds: vec![Round {
                round: 1,
                candidate_index: 0,
                candidate_seed: 7,
                nominal_radius: 0.01,
                accepted: false,
                next_length: 0.01,
                wall_seconds: seconds,
                kernel_check: Check {
                    sequence_nlls: vec![1.0, 1.0],
                    token_nlls: vec![1.0; 8192],
                    scorer_gpu_seconds: seconds * 0.8,
                    complete_round_seconds: seconds,
                },
            }],
        }
    }

    #[test]
    fn numerical_contract() {
        let baseline = run(1.0);
        let mut candidate = run(0.8);
        candidate.rounds[0].kernel_check.token_nlls[0] += 1e-4;
        assert!(compare(&baseline, &candidate, &gate()).is_ok());
        candidate.rounds[0].kernel_check.token_nlls[0] = 1.1;
        assert!(compare(&baseline, &candidate, &gate()).is_err());
        candidate.rounds[0].kernel_check.token_nlls[0] = f64::NAN;
        assert!(compare(&baseline, &candidate, &gate()).is_err());
    }

    #[test]
    fn trajectory_gate() {
        let baseline = run(1.0);
        let mut candidate = run(0.1);
        candidate.rounds[0].accepted = true;
        assert!(
            compare(&baseline, &candidate, &gate())
                .unwrap_err()
                .contains("trajectory")
        );
        candidate.rounds.clear();
        assert!(compare(&baseline, &candidate, &gate()).is_err());
    }

    #[test]
    fn wholeloop_gate() {
        let baseline = run(1.0);
        let mut candidate = run(1.1);
        candidate.rounds[0].wall_seconds = 0.01;
        let pair = compare(&baseline, &candidate, &gate()).unwrap();
        assert!(!gate().passes(&[pair]));
    }

    #[test]
    fn evidence_gate() {
        let pairs =
            [0.8, 0.8, 1.1].map(|seconds| compare(&run(1.0), &run(seconds), &gate()).unwrap());
        assert!(!gate().passes(&pairs));
        assert!(!gate().passes(&[]));
        let wins =
            [0.8, 0.85, 0.9].map(|seconds| compare(&run(1.0), &run(seconds), &gate()).unwrap());
        assert!(gate().passes(&wins));
    }

    #[test]
    fn threshold_gate() {
        let mut rules = gate();
        rules.token_atol = f64::NAN;
        assert!(rules.validate().is_err());
        rules = gate();
        rules.min_speedup = 1.0;
        assert!(rules.validate().is_err());
    }

    #[test]
    fn timing_edges() {
        assert_eq!(median([4.0, 1.0, 3.0, 2.0]), 2.5);
        assert_eq!(tail("abcdefé", 1), "");
        assert_eq!(tail("abcdefé", 2), "é");
    }

    #[test]
    fn worker_timeout() {
        let mut command = Command::new("sleep");
        command.arg("10");
        let started = Instant::now();
        let error = wait_worker(&mut command, Duration::from_millis(20)).unwrap_err();
        assert!(error.starts_with("worker timed out"));
        assert!(started.elapsed() < Duration::from_secs(2));
    }

    #[test]
    fn worker_coverage() {
        let directory = std::env::temp_dir().join(format!(
            "ennx-record-{}-{}",
            std::process::id(),
            rand::random::<u64>()
        ));
        fs::create_dir(&directory).unwrap();
        let mut summary = ennx_wire::toml::Table::new();
        summary.insert("status", "completed");
        summary.insert("loop_seconds", 1.1);
        fs::write(
            directory.join("result.toml"),
            ennx_wire::toml::to_string(&summary).unwrap(),
        )
        .unwrap();
        let mut sample = run(1.0);
        let write = |round: &Round| {
            fs::write(
                directory.join("controller.jsonl"),
                ennx_wire::json::to_vec(round).unwrap(),
            )
            .unwrap();
        };
        write(&sample.rounds[0]);
        assert!(Run::read(&directory, 1).is_ok());
        assert!(Run::read(&directory, 2).is_err());
        sample.rounds[0].kernel_check.token_nlls.pop();
        write(&sample.rounds[0]);
        assert!(Run::read(&directory, 1).is_err());
        fs::remove_dir_all(directory).unwrap();
    }
}
