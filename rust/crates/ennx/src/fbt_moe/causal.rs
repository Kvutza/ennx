use super::*;
use deser::Serialize;
use std::io::Write;

#[derive(Clone, Copy, Serialize)]
pub(super) struct CausalEvidence {
    pub(super) batch: u32,
    pub(super) candidate_nll: f32,
    pub(super) incumbent_nll: f32,
    pub(super) improvement: f32,
    pub(super) optimizer_value: f32,
    pub(super) variance: f32,
    pub(super) gpu_seconds: f64,
}

pub(super) struct CausalScorer {
    pipelines: Pipelines,
    tensorops: TensorOpsPipelines,
    pisa1: Pisa1,
    buffers: Buffers,
    dataset: crate::pretrain_data::PretrainDataset,
    validation: crate::pretrain_data::PretrainDataset,
}

impl CausalScorer {
    pub(super) fn new(runtime: &Runtime, path: &std::path::Path) -> Result<Self, String> {
        let validation_path = path.with_file_name("validation.ennxptn");
        Ok(Self {
            pipelines: Pipelines::new(runtime)?,
            tensorops: TensorOpsPipelines::new(runtime)?,
            pisa1: Pisa1::new(runtime)?,
            buffers: Buffers::new(runtime),
            dataset: crate::pretrain_data::PretrainDataset::load(path)?,
            validation: crate::pretrain_data::PretrainDataset::load(&validation_path)?,
        })
    }

    pub(super) fn batches(&self) -> u32 {
        self.dataset.batches()
    }

    fn score_on(
        &self,
        runtime: &Runtime,
        row: CandidateRow<'_>,
        batch: u32,
        dataset: &crate::pretrain_data::PretrainDataset,
    ) -> Result<(ObjectiveStats, f64), String> {
        pretrain_batch(&self.buffers, dataset, batch)?;
        let command = runtime.queue.new_command_buffer();
        objective_fused(
            command,
            &self.pipelines,
            &self.tensorops,
            &self.pisa1,
            &self.buffers,
            row,
        )?;
        let gpu_seconds = complete(command)?;
        Ok((sequence_objective(&self.buffers)?, gpu_seconds))
    }

    fn score(
        &self,
        runtime: &Runtime,
        row: CandidateRow<'_>,
        batch: u32,
    ) -> Result<(ObjectiveStats, f64), String> {
        self.score_on(runtime, row, batch, &self.dataset)
    }

    pub(super) fn initial(
        &self,
        runtime: &Runtime,
        row: CandidateRow<'_>,
    ) -> Result<CausalEvidence, String> {
        let (score, gpu_seconds) = self.score(runtime, row, 0)?;
        Ok(CausalEvidence {
            batch: 0,
            candidate_nll: -score.reward,
            incumbent_nll: -score.reward,
            improvement: 0.0,
            optimizer_value: score.reward,
            variance: score.variance,
            gpu_seconds,
        })
    }

    pub(super) fn paired(
        &self,
        runtime: &Runtime,
        candidate: CandidateRow<'_>,
        incumbent: CandidateRow<'_>,
        batch: u32,
        baseline: f32,
    ) -> Result<CausalEvidence, String> {
        let (candidate, candidate_gpu) = self.score(runtime, candidate, batch)?;
        let (incumbent, incumbent_gpu) = self.score(runtime, incumbent, batch)?;
        let (improvement, variance) = paired_improvement(&candidate, &incumbent);
        Ok(CausalEvidence {
            batch,
            candidate_nll: -candidate.reward,
            incumbent_nll: -incumbent.reward,
            improvement,
            optimizer_value: baseline + improvement,
            variance,
            gpu_seconds: candidate_gpu + incumbent_gpu,
        })
    }

    fn validation(
        &self,
        runtime: &Runtime,
        row: CandidateRow<'_>,
        round: u32,
    ) -> Result<CausalValidation, String> {
        let mut nll = 0.0f64;
        let mut gpu_seconds = 0.0;
        for batch in 0..self.validation.batches() {
            let (score, gpu) = self.score_on(runtime, row, batch, &self.validation)?;
            nll -= f64::from(score.reward);
            gpu_seconds += gpu;
        }
        nll /= f64::from(self.validation.batches());
        Ok(CausalValidation {
            round,
            nll,
            batches: self.validation.batches(),
            causal_targets: u64::from(self.validation.sequences())
                * u64::from(crate::pretrain_data::CONTEXT - 1),
            gpu_seconds,
        })
    }
}

#[derive(Serialize)]
struct CausalValidation {
    round: u32,
    nll: f64,
    batches: u32,
    causal_targets: u64,
    gpu_seconds: f64,
}

pub(super) struct CausalLog {
    file: std::fs::File,
}

impl CausalLog {
    pub(super) fn new(output: &std::path::Path) -> Result<Self, String> {
        Ok(Self {
            file: std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(output.join("causal-validation.jsonl"))
                .map_err(|error| error.to_string())?,
        })
    }

    pub(super) fn observe(
        &mut self,
        scorer: &CausalScorer,
        runtime: &Runtime,
        row: CandidateRow<'_>,
        round: u32,
    ) -> Result<(), String> {
        let result = scorer.validation(runtime, row, round)?;
        ennx_wire::json::write_line(&mut self.file, &result).map_err(|error| error.to_string())?;
        self.file.flush().map_err(|error| error.to_string())?;
        eprintln!(
            "ENNX_CAUSAL_VALIDATION round={} nll={:.9} gpu_seconds={:.6} used_for_acceptance=false",
            result.round, result.nll, result.gpu_seconds
        );
        Ok(())
    }
}
