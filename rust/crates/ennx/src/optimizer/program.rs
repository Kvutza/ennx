use std::collections::VecDeque;
use std::time::Instant;

use ndarray::{Array1, Array2};
use rand::RngCore;

use crate::acquisition::{ParetoAcquisition, ThompsonAcquisition, UCBAcquisition};
use crate::config::{AcquisitionConfig, OptimizerConfig, SurrogateConfig};
use crate::dense::DenseTerm;
use crate::error::ENNError;
use crate::surrogate::{ENNSurrogate, Surrogate};

use super::Optimizer;

#[derive(Debug, Clone, PartialEq)]
pub struct ProgramTrial {
    pub id: u64,
    pub seed: u64,
    pub score: f64,
    pub terms: Vec<DenseTerm>,
}

#[derive(Debug, Clone)]
struct Observation {
    terms: Vec<DenseTerm>,
    reward: f64,
}

pub(super) struct ProgramState {
    history: VecDeque<Observation>,
    capacity: usize,
    incumbent: Observation,
    pending: Option<ProgramTrial>,
    next_id: u64,
    observations: usize,
}

impl ProgramState {
    pub(super) fn new(base_reward: f64, capacity: usize) -> Result<Self, ENNError> {
        if !base_reward.is_finite() {
            return Err(invalid("program baseline reward must be finite"));
        }
        if capacity < 2 {
            return Err(invalid("program history capacity must be at least two"));
        }
        let incumbent = Observation {
            terms: Vec::new(),
            reward: base_reward,
        };
        Ok(Self {
            history: VecDeque::from([incumbent.clone()]),
            capacity,
            incumbent,
            pending: None,
            next_id: 0,
            observations: 1,
        })
    }

    fn candidates(&self, seeds: &[u64], radius: f32) -> Result<Vec<ProgramTrial>, ENNError> {
        seeds
            .iter()
            .copied()
            .map(|seed| {
                let mut terms = self.incumbent.terms.clone();
                terms.push(DenseTerm::new(seed, radius).map_err(invalid)?);
                Ok(ProgramTrial {
                    id: self.next_id,
                    seed,
                    score: 0.0,
                    terms,
                })
            })
            .collect()
    }

    fn remember(&mut self, trial: ProgramTrial, reward: f64) {
        let observation = Observation {
            terms: trial.terms,
            reward,
        };
        if reward > self.incumbent.reward {
            self.incumbent = observation.clone();
        }
        if self.history.len() == self.capacity {
            self.history.pop_front();
        }
        self.history.push_back(observation);
        self.observations += 1;
    }
}

impl Optimizer {
    pub fn enable_program(&mut self, base_reward: f64, capacity: usize) -> Result<(), ENNError> {
        self.program = Some(ProgramState::new(base_reward, capacity)?);
        self.trust_region_mut().set_turbo_prev_num_obs(1);
        Ok(())
    }

    pub fn ask_program(
        &mut self,
        seeds: &[u64],
        rng: &mut dyn RngCore,
    ) -> Result<ProgramTrial, ENNError> {
        let started = Instant::now();
        if seeds.is_empty() {
            return Err(invalid("program ask requires candidate seeds"));
        }
        let radius = self.tr_length() as f32;
        let config = self.config.clone();
        let state = self
            .program
            .as_ref()
            .ok_or_else(|| invalid("program space is disabled"))?;
        if state.pending.is_some() {
            return Err(invalid("program tell must finish the pending trial"));
        }
        let mut candidates = state.candidates(seeds, radius)?;
        let fit_started = Instant::now();
        let prediction = predict(state, &candidates, &config, rng)?;
        self.telemetry.dt_fit = fit_started.elapsed().as_secs_f64();
        let select_started = Instant::now();
        let index = select(&prediction.0, &prediction.1, config.acquisition, rng)?;
        self.telemetry.dt_sel = select_started.elapsed().as_secs_f64();
        self.telemetry.num_candidates = candidates.len();
        let mut trial = candidates.swap_remove(index);
        trial.score = score_at(&prediction, index, config.acquisition);
        let state = self.program.as_mut().expect("program enabled");
        state.next_id = state.next_id.wrapping_add(1);
        state.pending = Some(trial.clone());
        self.telemetry.dt_gen = started.elapsed().as_secs_f64();
        Ok(trial)
    }

    pub fn tell_program(
        &mut self,
        trial: ProgramTrial,
        reward: f64,
        rng: &mut dyn RngCore,
    ) -> Result<(), ENNError> {
        let started = Instant::now();
        if !reward.is_finite() {
            return Err(invalid("program reward must be finite"));
        }
        let state = self
            .program
            .as_mut()
            .ok_or_else(|| invalid("program space is disabled"))?;
        let pending = state
            .pending
            .take()
            .ok_or_else(|| invalid("program trial is not pending"))?;
        if pending != trial {
            state.pending = Some(pending);
            return Err(invalid("program trial does not match pending ask"));
        }
        state.remember(trial, reward);
        let count = state.observations;
        let incumbent = state.incumbent.reward;
        let y = Array2::from_shape_vec((1, 1), vec![reward])?;
        let best = Array1::from_vec(vec![incumbent]);
        self.trust_region_mut().set_num_arms(1);
        self.trust_region_mut()
            .tell_update_new_batch(&y.view(), &best.view(), count)?;
        if self.trust_region().needs_restart() {
            self.trust_region_mut().restart(Some(rng));
            self.increment_restart_generation();
        }
        self.telemetry.dt_tell = started.elapsed().as_secs_f64();
        Ok(())
    }

    pub fn best_program(&self) -> Option<(&[DenseTerm], f64)> {
        self.program
            .as_ref()
            .map(|state| (state.incumbent.terms.as_slice(), state.incumbent.reward))
    }
}

fn predict(
    state: &ProgramState,
    candidates: &[ProgramTrial],
    config: &OptimizerConfig,
    rng: &mut dyn RngCore,
) -> Result<(Array2<f64>, Array2<f64>), ENNError> {
    if state.history.len() < 2 || matches!(config.acquisition, AcquisitionConfig::Random) {
        let rows = candidates.len();
        return Ok((Array2::zeros((rows, 1)), Array2::ones((rows, 1))));
    }
    let columns = seed_columns(state, candidates);
    let x = encode_history(state, &columns);
    let y = Array2::from_shape_vec(
        (state.history.len(), 1),
        state.history.iter().map(|item| item.reward).collect(),
    )?;
    let query = encode_candidates(candidates, &columns);
    let SurrogateConfig::ENN(mut enn_config) = config.surrogate.clone() else {
        return Ok((
            Array2::zeros((query.nrows(), 1)),
            Array2::ones((query.nrows(), 1)),
        ));
    };
    enn_config.k = enn_config.k.max(1).min(state.history.len() as i32);
    enn_config.num_fit_samples = enn_config.num_fit_samples.min(state.history.len()).max(1);
    let mut surrogate = ENNSurrogate::new(enn_config);
    surrogate.fit(&x.view(), &y.view(), None, rng)?;
    let prediction = surrogate.predict(&query.view())?;
    Ok((prediction.mu, prediction.se))
}

fn seed_columns(state: &ProgramState, candidates: &[ProgramTrial]) -> Vec<u64> {
    let mut columns = Vec::new();
    for terms in state
        .history
        .iter()
        .map(|item| item.terms.as_slice())
        .chain(candidates.iter().map(|item| item.terms.as_slice()))
    {
        for term in terms {
            if !columns.contains(&term.seed) {
                columns.push(term.seed);
            }
        }
    }
    columns
}

fn encode_history(state: &ProgramState, columns: &[u64]) -> Array2<f64> {
    let mut encoded = Array2::zeros((state.history.len(), columns.len()));
    for (row, item) in state.history.iter().enumerate() {
        encode_terms(&mut encoded, row, &item.terms, columns);
    }
    encoded
}

fn encode_candidates(candidates: &[ProgramTrial], columns: &[u64]) -> Array2<f64> {
    let mut encoded = Array2::zeros((candidates.len(), columns.len()));
    for (row, item) in candidates.iter().enumerate() {
        encode_terms(&mut encoded, row, &item.terms, columns);
    }
    encoded
}

fn encode_terms(matrix: &mut Array2<f64>, row: usize, terms: &[DenseTerm], columns: &[u64]) {
    for term in terms {
        if let Some(column) = columns.iter().position(|seed| *seed == term.seed) {
            matrix[[row, column]] += f64::from(term.coefficient);
        }
    }
}

fn select(
    mean: &Array2<f64>,
    se: &Array2<f64>,
    acquisition: AcquisitionConfig,
    rng: &mut dyn RngCore,
) -> Result<usize, ENNError> {
    let selected = match acquisition {
        AcquisitionConfig::UCB { beta } => {
            UCBAcquisition::new(beta).select(&mean.column(0), &se.column(0), 1, rng)
        }
        AcquisitionConfig::Thompson => {
            ThompsonAcquisition::new().select(&mean.column(0), &se.column(0), 1, rng)
        }
        AcquisitionConfig::Random => {
            crate::acquisition::RandomAcquisition::new().select(mean.nrows(), 1, rng)
        }
        AcquisitionConfig::Pareto => {
            ParetoAcquisition::new().select(&mean.view(), &se.view(), 1, rng)
        }
    }
    .map_err(|error| invalid(error.to_string()))?;
    selected
        .into_iter()
        .next()
        .ok_or_else(|| invalid("acquisition selected no candidate"))
}

fn score_at(
    prediction: &(Array2<f64>, Array2<f64>),
    index: usize,
    acquisition: AcquisitionConfig,
) -> f64 {
    match acquisition {
        AcquisitionConfig::UCB { beta } => {
            prediction.0[[index, 0]] + beta * prediction.1[[index, 0]]
        }
        _ => prediction.0[[index, 0]],
    }
}

fn invalid(message: impl Into<String>) -> ENNError {
    ENNError::InvalidParameter(message.into())
}

#[cfg(test)]
#[path = "tests_program.rs"]
mod tests;
