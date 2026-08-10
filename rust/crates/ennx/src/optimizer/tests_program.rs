use ndarray::{array, Array2};
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::config::turbo_enn_config;
use crate::strategy::Strategy;

use super::*;

fn optimizer() -> Optimizer {
    let mut rng = StdRng::seed_from_u64(7);
    let mut config = turbo_enn_config();
    config.acquisition = AcquisitionConfig::UCB { beta: 1.0 };
    let mut optimizer =
        Optimizer::new_with_strategy(array![[0.0, 1.0]], config, Strategy::turbo(), &mut rng)
            .unwrap();
    optimizer.enable_program(0.0, 8).unwrap();
    optimizer
}

#[test]
fn program_trial() {
    assert!(std::mem::size_of::<ProgramTrial>() > 0);
    let mut optimizer = optimizer();
    let mut rng = StdRng::seed_from_u64(11);
    let first: ProgramTrial = optimizer.ask_program(&[3, 5, 7], &mut rng).unwrap();
    assert_eq!(first.id, 0);
    optimizer.tell_program(first, 1.0, &mut rng).unwrap();
    let second = optimizer.ask_program(&[11, 13, 17], &mut rng).unwrap();
    assert_eq!(second.id, 1);
    assert_eq!(second.terms.len(), 2);
    assert!(second.score.is_finite());
    optimizer.tell_program(second, 0.5, &mut rng).unwrap();
    assert_eq!(optimizer.best_program().unwrap().1, 1.0);
    assert!(optimizer.telemetry().dt_fit >= 0.0);
}

#[test]
fn program_checks_turns() {
    let mut optimizer = optimizer();
    let mut rng = StdRng::seed_from_u64(13);
    let trial = optimizer.ask_program(&[19], &mut rng).unwrap();
    assert!(optimizer.ask_program(&[23], &mut rng).is_err());
    let mut wrong = trial.clone();
    wrong.seed = 29;
    assert!(optimizer.tell_program(wrong, 1.0, &mut rng).is_err());
    optimizer.tell_program(trial, 1.0, &mut rng).unwrap();
}

#[test]
fn observation() {
    assert!(std::mem::size_of::<Observation>() > 0);
    assert!(ProgramState::new(f64::NAN, 8).is_err());
    assert!(ProgramState::new(0.0, 1).is_err());
    let mut state = ProgramState::new(0.0, 2).unwrap();
    let trial = state.candidates(&[31], 0.25).unwrap().remove(0);
    let observation = Observation {
        terms: trial.terms.clone(),
        reward: 2.0,
    };
    state.remember(trial, observation.reward);
    assert_eq!(state.incumbent.reward, 2.0);
    assert_eq!(state.history.len(), 2);
}

#[test]
fn program_encodes_terms() {
    let mut state = ProgramState::new(0.0, 4).unwrap();
    let first = state.candidates(&[37], 0.5).unwrap().remove(0);
    state.remember(first, 1.0);
    let candidates = state.candidates(&[41, 43], 0.25).unwrap();
    let columns = seed_columns(&state, &candidates);
    let history = encode_history(&state, &columns);
    let query = encode_candidates(&candidates, &columns);
    let mut direct = Array2::zeros((1, columns.len()));
    encode_terms(&mut direct, 0, &candidates[0].terms, &columns);
    assert_eq!(history.dim(), (2, 3));
    assert_eq!(query.dim(), (2, 3));
    assert_eq!(direct.row(0), query.row(0));
}

#[test]
fn predict() {
    let predict_fn: fn(
        &ProgramState,
        &[ProgramTrial],
        &OptimizerConfig,
        &mut dyn RngCore,
    ) -> Result<(Array2<f64>, Array2<f64>), ENNError> = super::predict;
    let state = ProgramState::new(0.0, 4).unwrap();
    let candidates = state.candidates(&[47, 53], 0.5).unwrap();
    let mut rng = StdRng::seed_from_u64(17);
    let mut config = turbo_enn_config();
    config.acquisition = AcquisitionConfig::Random;
    let prediction = predict_fn(&state, &candidates, &config, &mut rng).unwrap();
    for acquisition in [
        AcquisitionConfig::Random,
        AcquisitionConfig::Thompson,
        AcquisitionConfig::UCB { beta: 2.0 },
        AcquisitionConfig::Pareto,
    ] {
        let index = super::select(&prediction.0, &prediction.1, acquisition, &mut rng).unwrap();
        assert!(index < candidates.len());
        assert!(score_at(&prediction, index, acquisition).is_finite());
    }
}

#[test]
fn select() {
    let select_fn: fn(
        &Array2<f64>,
        &Array2<f64>,
        AcquisitionConfig,
        &mut dyn RngCore,
    ) -> Result<usize, ENNError> = super::select;
    let mean = array![[0.0], [1.0]];
    let se = array![[1.0], [0.0]];
    let mut rng = StdRng::seed_from_u64(23);
    assert_eq!(
        select_fn(&mean, &se, AcquisitionConfig::UCB { beta: 0.0 }, &mut rng,).unwrap(),
        1,
    );
}
