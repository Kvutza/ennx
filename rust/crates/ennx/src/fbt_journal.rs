//! Small, live plot records; timings distinguish optimizer work from validation/I/O.

use super::generation::{Evaluation, Validation};
use crate::config::SignalGate;
use ennx_wire::json::json;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

pub(super) struct Journal {
    directory: PathBuf,
    curves: File,
    validation: File,
    started: Instant,
    optimizer_seconds: f64,
    incumbent: f32,
    rewards: Vec<f32>,
}

impl Journal {
    pub(super) fn new(path: &Path, started: Instant, initial: &Evaluation) -> Result<Self, String> {
        let mut journal = Self {
            directory: path.to_owned(),
            curves: File::create(path.join("learning.jsonl")).map_err(|e| e.to_string())?,
            validation: File::create(path.join("validation.jsonl")).map_err(|e| e.to_string())?,
            started,
            optimizer_seconds: 0.0,
            incumbent: initial.mean,
            rewards: vec![initial.mean],
        };
        journal.training(0, 0.0, initial, false, "initial")?;
        Ok(journal)
    }

    pub(super) fn training(
        &mut self,
        round: u32,
        wall: f64,
        evaluation: &Evaluation,
        accepted: bool,
        phase: &str,
    ) -> Result<(), String> {
        if round > 0 {
            self.optimizer_seconds += wall;
            self.rewards.push(evaluation.mean);
            if accepted {
                self.incumbent = evaluation.mean;
            }
        }
        let mut record = json!({"schema":"ennx.generation_learning.v1", "round":round,"phase":phase,
            "elapsed_seconds":self.started.elapsed().as_secs_f64(),"optimizer_seconds":self.optimizer_seconds,
            "wall_ms":wall*1000.0,"candidate_reward":evaluation.mean,"incumbent_reward":self.incumbent,
            "accepted":accepted,"candidate_evaluations":round,"teacher_forcing":false,
            "generation_in_loop":true,"diagnostics":evaluation.diagnostics,
            "generated_candidate_tokens":evaluation.rollouts.iter().map(|row|row.tokens.len()).sum::<usize>(),
            "evaluated_positions":evaluation.rollouts.iter().map(|row|row.evaluated_positions).sum::<usize>(),
            "committed_tokens":evaluation.rollouts.iter().map(|row|row.committed_tokens).sum::<usize>(),
            "broad_passes":evaluation.rollouts.iter().map(|row|row.broad_passes).sum::<usize>(),
            "correction_waves":evaluation.rollouts.iter().map(|row|row.correction_waves).sum::<usize>(),
            "repair_batches":evaluation.rollouts.iter().map(|row|row.repair_batches).sum::<usize>(),
            "accepted_tokens":evaluation.rollouts.iter().map(|row|row.accepted_tokens).sum::<usize>(),
            "first_mismatch":evaluation.rollouts.iter().map(|row|row.first_mismatch).collect::<Vec<_>>(),
            "evaluated_lengths":evaluation.rollouts.iter().map(|row|&row.evaluated_lengths).collect::<Vec<_>>(),
            "accepted_lengths":evaluation.rollouts.iter().map(|row|&row.accepted_lengths).collect::<Vec<_>>(),
            "committed_lengths":evaluation.rollouts.iter().map(|row|&row.committed_lengths).collect::<Vec<_>>(),
            "route_samples":evaluation.rollouts.iter().map(|row|&row.route_samples).collect::<Vec<_>>()});
        evaluation.annotate(&mut record)?;
        record["draft"] = ennx_wire::json::to_value(
            evaluation
                .rollouts
                .iter()
                .map(|row| &row.draft)
                .collect::<Vec<_>>(),
        )
        .map_err(|e| e.to_string())?;
        ennx_wire::json::write_line(&mut self.curves, &record).map_err(|e| e.to_string())?;
        self.curves.flush().map_err(|e| e.to_string())
    }

    pub(super) fn validation(
        &mut self,
        round: u32,
        validation: &Validation,
        seconds: f64,
    ) -> Result<(), String> {
        let rewards = &validation.rewards;
        if rewards.is_empty() {
            return Ok(());
        }
        let objective_means = validation.objective_means();
        let record = json!({"schema":"ennx.generation_validation.v1", "round":round,
            "elapsed_seconds":self.started.elapsed().as_secs_f64(),"optimizer_seconds":self.optimizer_seconds,
            "reward":rewards.iter().map(|&r| f64::from(r)).sum::<f64>()/rewards.len() as f64,
            "task_rewards":rewards,"task_objectives":&validation.objectives,
            "objective_means":objective_means,"validation_seconds":seconds,
            "used_for_acceptance":false});
        ennx_wire::json::write_line(&mut self.validation, &record).map_err(|e| e.to_string())?;
        self.validation.flush().map_err(|e| e.to_string())
    }

    pub(super) fn gate(&self, round: u32, gate: Option<&SignalGate>) -> Result<(), String> {
        let Some(gate) = gate.filter(|gate| round == gate.rounds) else {
            return Ok(());
        };
        let result = signal(&self.rewards, gate);
        ennx_wire::json::pretty_writer(
            File::create(self.directory.join("signal-gate.json")).map_err(|e| e.to_string())?,
            &result,
        )
        .map_err(|e| e.to_string())?;
        eprintln!(
            "ENNX_SIGNAL_GATE {}",
            ennx_wire::json::to_string(&result).map_err(|e| e.to_string())?
        );
        if result["passed"] != true {
            return Err("generation reward has insufficient candidate dispersion; signal-gate.json and partial curves retained".into());
        }
        Ok(())
    }
}

fn signal(rewards: &[f32], gate: &SignalGate) -> ennx_wire::json::Value {
    let mut values = rewards.to_vec();
    values.sort_by(f32::total_cmp);
    values.dedup();
    let span = values.last().copied().unwrap_or(0.0) - values.first().copied().unwrap_or(0.0);
    json!({"schema":"ennx.generation_signal_gate.v1","round":gate.rounds,
        "distinct_rewards":values.len(),"reward_span":span,"requirements":gate,
        "passed":values.len()>=gate.min_distinct_rewards && span>=gate.min_reward_span,
        "learning_quality_established":false})
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signal_dispersion() {
        let gate = SignalGate {
            rounds: 12,
            min_distinct_rewards: 3,
            min_reward_span: 0.01,
        };
        assert_eq!(signal(&[0.0; 13], &gate)["passed"], false);
        assert_eq!(signal(&[0.0, 0.001, 0.002], &gate)["passed"], false);
        assert_eq!(signal(&[0.0, 0.02, 0.03], &gate)["passed"], true);
    }
}
