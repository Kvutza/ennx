use std::fs;
use std::path::{Path, PathBuf};

use ennx_wire::json::{Value, json};
use ndarray::{Array1, Array2};

pub(crate) struct Arm<'a> {
    pub id: &'a str,
    pub role: &'a str,
    pub scales: &'a [f32],
}

struct Checkpoint {
    round: u32,
    objectives: Vec<f64>,
}

struct Rep {
    index: u64,
    auc: f64,
    gain_auc: f64,
    final_hypervolume: f64,
    initial: Vec<f64>,
    final_objectives: Vec<f64>,
    rounds: u64,
    accepted: u64,
    median_wall_ms: f64,
    artifact: PathBuf,
}

pub(crate) fn write(out: &Path, arms: &[Arm<'_>]) -> Result<Option<Value>, String> {
    let mut summaries = Vec::new();
    let mut complete = true;
    for arm in arms {
        let Some(run) = latest(&out.join("runs").join(arm.id))? else {
            complete = false;
            continue;
        };
        let reps = repetitions(&run, arm.scales)?;
        if reps.is_empty() {
            complete = false;
            continue;
        }
        summaries.push((arm, reps));
    }
    if !complete || summaries.len() != arms.len() {
        return Ok(None);
    }

    let arm_rows = summaries
        .iter()
        .map(|(arm, reps)| arm_row(arm, reps))
        .collect::<Vec<_>>();

    let winner = summaries
        .iter()
        .max_by(|(_, left), (_, right)| {
            mean(left.iter().map(|rep| rep.gain_auc))
                .total_cmp(&mean(right.iter().map(|rep| rep.gain_auc)))
        })
        .map(|(arm, _)| arm.id)
        .ok_or("experiment analysis has no arms")?;
    let report = json!({
        "schema": "ennx.experiment-analysis.v1",
        "primary-metric": "normalized-hypervolume-gain-auc-per-objective-call",
        "normalization": "initial-heldout-objectives-minus-configured-scales",
        "winner": winner,
        "arms": arm_rows,
    });
    let text = ennx_wire::json::pretty_string(&report).map_err(|error| error.to_string())?;
    fs::write(out.join("analysis.json"), text).map_err(|error| error.to_string())?;
    Ok(Some(report))
}

fn arm_row(arm: &Arm<'_>, reps: &[Rep]) -> Value {
    let auc = mean(reps.iter().map(|rep| rep.auc));
    let gain = mean(reps.iter().map(|rep| rep.gain_auc));
    let final_hypervolume = mean(reps.iter().map(|rep| rep.final_hypervolume));
    let repetitions = reps.iter().map(rep_row).collect::<Vec<_>>();
    json!({
        "id": arm.id,
        "role": arm.role,
        "mean-hypervolume-auc-per-objective-call": auc,
        "mean-hypervolume-gain-auc-per-objective-call": gain,
        "mean-final-hypervolume": final_hypervolume,
        "repetitions": repetitions,
    })
}

fn rep_row(rep: &Rep) -> Value {
    let deltas = rep
        .final_objectives
        .iter()
        .zip(&rep.initial)
        .map(|(final_value, initial)| final_value - initial)
        .collect::<Vec<_>>();
    json!({
        "rep": rep.index,
        "hypervolume-auc-per-objective-call": rep.auc,
        "hypervolume-gain-auc-per-objective-call": rep.gain_auc,
        "final-hypervolume": rep.final_hypervolume,
        "initial-objectives": rep.initial,
        "final-objectives": rep.final_objectives,
        "objective-deltas": deltas,
        "rounds": rep.rounds,
        "accepted": rep.accepted,
        "median-wall-ms": rep.median_wall_ms,
        "artifact": rep.artifact,
    })
}

fn latest(root: &Path) -> Result<Option<PathBuf>, String> {
    if !root.exists() {
        return Ok(None);
    }
    let mut runs = fs::read_dir(root)
        .map_err(|error| error.to_string())?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.is_dir() && completed(&path.join("result.json")))
        .collect::<Vec<_>>();
    runs.sort();
    Ok(runs.pop())
}

fn completed(path: &Path) -> bool {
    fs::read_to_string(path)
        .ok()
        .and_then(|text| ennx_wire::json::from_str::<Value>(&text).ok())
        .is_some_and(|result| result["status"].as_str() == Some("completed"))
}

fn repetitions(run: &Path, scales: &[f32]) -> Result<Vec<Rep>, String> {
    let mut roots = fs::read_dir(run)
        .map_err(|error| error.to_string())?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_dir()
                && path
                    .file_name()
                    .is_some_and(|name| name.to_string_lossy().starts_with("rep-"))
        })
        .collect::<Vec<_>>();
    if roots.is_empty() {
        roots.push(run.to_owned());
    }
    roots.sort();
    roots
        .into_iter()
        .map(|root| repetition(&root, scales))
        .collect()
}

fn repetition(root: &Path, scales: &[f32]) -> Result<Rep, String> {
    let repetition = read_json(&root.join("repetition.json"))?;
    let result = read_json(&root.join("result.json"))?;
    let checkpoints = checkpoints(&root.join("validation.jsonl"))?;
    let first = checkpoints
        .first()
        .ok_or("validation has no initial checkpoint")?;
    let last = checkpoints
        .last()
        .ok_or("validation has no final checkpoint")?;
    let rounds = result["rounds"]
        .as_u64()
        .ok_or("result is missing rounds")?;
    if u64::from(last.round) != rounds {
        return Err(format!(
            "validation ends at round {} but result has {rounds}",
            last.round
        ));
    }
    if first.round != 0 || first.objectives.len() != scales.len() {
        return Err("validation objective schema does not match acquisition scales".into());
    }
    let curve = hypervolume_curve(&checkpoints, scales)?;
    let auc = area(&curve, rounds as f64);
    let initial_hypervolume = curve.first().map(|row| row.1).unwrap_or(0.0);
    Ok(Rep {
        index: repetition["rep"]
            .as_u64()
            .ok_or("missing repetition index")?,
        auc,
        gain_auc: auc - initial_hypervolume,
        final_hypervolume: curve.last().map(|row| row.1).unwrap_or(0.0),
        initial: first.objectives.clone(),
        final_objectives: last.objectives.clone(),
        rounds,
        accepted: result["accepted"]
            .as_u64()
            .ok_or("result is missing accepted")?,
        median_wall_ms: result["median_wall_ms"]
            .as_f64()
            .ok_or("result is missing median_wall_ms")?,
        artifact: root.to_owned(),
    })
}

fn read_json(path: &Path) -> Result<Value, String> {
    let text = fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))?;
    ennx_wire::json::from_str(&text).map_err(|error| format!("{}: {error}", path.display()))
}

fn checkpoints(path: &Path) -> Result<Vec<Checkpoint>, String> {
    let text = fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let mut rows = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let row: Value = ennx_wire::json::from_str(line).map_err(|error| error.to_string())?;
            let round = row["round"].as_u64().ok_or("validation row has no round")?;
            let values = row["objective_means"]
                .as_seq()
                .ok_or("validation row has no objective_means")?
                .iter()
                .map(|value| value.as_f64().ok_or("validation objective is not numeric"))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Checkpoint {
                round: u32::try_from(round).map_err(|error| error.to_string())?,
                objectives: values,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    rows.sort_by_key(|row| row.round);
    Ok(rows)
}

fn hypervolume_curve(rows: &[Checkpoint], scales: &[f32]) -> Result<Vec<(f64, f64)>, String> {
    let initial = &rows[0].objectives;
    let reference = initial
        .iter()
        .zip(scales)
        .map(|(value, scale)| value - f64::from(*scale))
        .collect::<Vec<_>>();
    let mut points = Vec::new();
    rows.iter()
        .map(|row| {
            if row.objectives.len() != scales.len() {
                return Err("validation objective width changed".into());
            }
            points.push(
                row.objectives
                    .iter()
                    .zip(&reference)
                    .zip(scales)
                    .map(|((value, reference), scale)| (value - reference) / f64::from(*scale))
                    .collect::<Vec<_>>(),
            );
            let matrix = Array2::from_shape_vec(
                (points.len(), scales.len()),
                points.iter().flatten().copied().collect(),
            )
            .map_err(|error| error.to_string())?;
            let volume = ennx::hypervolume_max(&matrix.view(), &Array1::zeros(scales.len()).view())
                .map_err(|error| error.to_string())?;
            Ok((f64::from(row.round), volume))
        })
        .collect()
}

fn area(curve: &[(f64, f64)], calls: f64) -> f64 {
    if calls <= 0.0 {
        return 0.0;
    }
    curve
        .windows(2)
        .map(|pair| (pair[1].0 - pair[0].0) * (pair[0].1 + pair[1].1) * 0.5)
        .sum::<f64>()
        / calls
}

fn mean(values: impl Iterator<Item = f64>) -> f64 {
    let values = values.collect::<Vec<_>>();
    values.iter().sum::<f64>() / values.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn curve_auc() {
        let rows = vec![
            Checkpoint {
                round: 0,
                objectives: vec![0.0, 0.0],
            },
            Checkpoint {
                round: 10,
                objectives: vec![1.0, 0.0],
            },
        ];
        let curve = hypervolume_curve(&rows, &[1.0, 1.0]).unwrap();
        assert_eq!(curve, [(0.0, 1.0), (10.0, 2.0)]);
        assert_eq!(area(&curve, 10.0), 1.5);
    }
}
