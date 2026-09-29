//! Held-out corpus-prefix NLL on accepted weights, never an optimizer observation.

use super::*;

pub(super) struct Scorer<'a> {
    pub runtime: &'a Runtime,
    pub pipelines: &'a Pipelines,
    pub tensorops: &'a TensorOpsPipelines,
    pub pisa1: &'a Pisa1,
    pub buffers: &'a Buffers,
}

impl Scorer<'_> {
    pub(super) fn observe(
        &self,
        dataset: &crate::pretrain_data::PretrainDataset,
        weights: CandidateRow<'_>,
        round: u32,
        objective_calls: u32,
        started: Instant,
    ) -> Result<ennx_wire::json::Value, String> {
        let start = Instant::now();
        let mut nll = 0.0f64;
        for batch in 0..dataset.batches() {
            pretrain_batch(self.buffers, dataset, batch)?;
            let command = self.runtime.queue.new_command_buffer();
            objective_fused(
                command,
                self.pipelines,
                self.tensorops,
                self.pisa1,
                self.buffers,
                weights,
            )?;
            complete(command)?;
            nll -= f64::from(sequence_objective(self.buffers)?.reward);
        }
        nll /= f64::from(dataset.batches());
        if !nll.is_finite() {
            return Err("held-out scorer produced nonfinite NLL".into());
        }
        let elapsed = started.elapsed().as_secs_f64();
        eprintln!(
            "ENNX_PRETRAIN_VALIDATION round={round} nll={nll:.9} elapsed_seconds={elapsed:.3} used_for_acceptance=false"
        );
        Ok(ennx_wire::json::json!({
            "round":round, "nll":nll, "learning_seconds":elapsed,
            "validation_seconds":start.elapsed().as_secs_f64(),
            "candidate_evaluations":round, "training_objective_calls":objective_calls,
            "training_token_positions":u64::from(objective_calls) * u64::from(ROWS),
            "training_causal_targets":u64::from(objective_calls) * u64::from(BATCH) * u64::from(CONTEXT - 1),
            "heldout_causal_targets":u64::from(dataset.sequences()) * u64::from(CONTEXT - 1),
        }))
    }
}
