use deser::Serialize;

#[derive(Debug, Serialize)]
pub(super) struct Aggregate {
    pub(super) task: String,
    pub(super) optimizer: String,
    pub(super) completed: usize,
    pub(super) failed: usize,
    pub(super) median_final_regret: Option<f64>,
    pub(super) median_normalized_auc: Option<f64>,
    pub(super) success_rate: f64,
    pub(super) median_evals_to_target: Option<f64>,
    pub(super) total_seconds: f64,
}

#[derive(Debug, Serialize)]
pub(super) struct Comparison {
    pub(super) task: String,
    pub(super) left: String,
    pub(super) right: String,
    pub(super) pairs: usize,
    pub(super) left_wins: usize,
    pub(super) ties: usize,
    pub(super) left_losses: usize,
    pub(super) median_final_regret_delta: Option<f64>,
    pub(super) median_normalized_auc_delta: Option<f64>,
}
