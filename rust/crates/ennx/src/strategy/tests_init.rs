use super::{select_arms, select_indices, InitStrategy, Strategy};
use crate::config::{lhd_only, turbo_enn, turbo_zero, AcquisitionConfig};
use crate::optimizer::Optimizer;
use ndarray::array;
use rand::rngs::StdRng;
use rand::SeedableRng;

#[test]
fn test_selectbyindices() {
    let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    let indices = vec![0, 2];
    let selected = select_indices(&x.view(), &indices);

    assert_eq!(selected.nrows(), 2);
    assert_eq!(selected[[0, 0]], 1.0);
    assert_eq!(selected[[0, 1]], 2.0);
    assert_eq!(selected[[1, 0]], 5.0);
    assert_eq!(selected[[1, 1]], 6.0);
}

#[test]
fn test_002() {
    let bounds = array![[0.0, 1.0], [0.0, 1.0]];
    let mut rng = StdRng::seed_from_u64(7);
    let strategy = Strategy::init(InitStrategy::Random, 4);
    let mut optimizer = Optimizer::new_strategy(bounds, turbo_zero(), strategy, &mut rng).unwrap();
    let x = optimizer.ask(2, &mut rng).unwrap();
    assert_eq!(x.nrows(), 2);
    assert!(optimizer.init_progress().is_some());
    let y = array![[1.0], [0.5]];
    optimizer.tell(&x.view(), &y.view(), &mut rng).unwrap();
    let (done, total) = optimizer.init_progress().unwrap();
    assert_eq!(done, 2);
    assert_eq!(total, 4);
}

#[test]
fn test_003() {
    let bounds = array![[0.0, 1.0], [0.0, 1.0]];
    let mut rng = StdRng::seed_from_u64(11);
    let strategy = Strategy::hybrid(InitStrategy::LHD, 2);
    let mut optimizer = Optimizer::new_strategy(bounds, turbo_zero(), strategy, &mut rng).unwrap();
    let x0 = optimizer.ask(2, &mut rng).unwrap();
    let y0 = array![[0.1], [0.2]];
    optimizer.tell(&x0.view(), &y0.view(), &mut rng).unwrap();
    assert!(optimizer.init_progress().is_none());
    let x1 = optimizer.ask(2, &mut rng).unwrap();
    assert_eq!(x1.nrows(), 2);
    let y1 = array![[0.3], [0.4]];
    optimizer.tell(&x1.view(), &y1.view(), &mut rng).unwrap();
}

#[test]
fn test_004() {
    let bounds = array![[0.0, 1.0], [0.0, 1.0]];
    let mut rng = StdRng::seed_from_u64(13);
    let mut optimizer =
        Optimizer::new_strategy(bounds, lhd_only(), Strategy::turbo(), &mut rng).unwrap();

    let x = optimizer.ask(2, &mut rng).unwrap();
    let y = array![[1.0], [1.1]];
    optimizer.tell(&x.view(), &y.view(), &mut rng).unwrap();
    assert!(optimizer.tr_length() > 0.0);
}

#[test]
fn test_strategyinitlhdpath() {
    let bounds = array![[0.0, 1.0], [0.0, 1.0]];
    let mut rng = StdRng::seed_from_u64(31);
    let strategy = Strategy::init(InitStrategy::LHD, 3);
    let mut optimizer = Optimizer::new_strategy(bounds, turbo_zero(), strategy, &mut rng).unwrap();
    let x = optimizer.ask(2, &mut rng).unwrap();
    let y = array![[0.2], [0.1]];
    optimizer.tell(&x.view(), &y.view(), &mut rng).unwrap();
    assert_eq!(optimizer.init_progress().unwrap(), (2, 3));
}

#[test]
fn test_006() {
    let x_cand = array![[0.1, 0.1], [0.9, 0.9], [0.5, 0.5], [0.2, 0.8]];
    let bounds = array![[0.0, 1.0], [0.0, 1.0]];
    let mut rng = StdRng::seed_from_u64(41);

    let mut cfg_random = turbo_zero();
    cfg_random.acquisition = AcquisitionConfig::Random;
    let opt_random =
        Optimizer::new_strategy(bounds.clone(), cfg_random, Strategy::turbo(), &mut rng).unwrap();
    let out_random = select_arms(&opt_random, &x_cand.view(), 2, &mut rng).unwrap();
    assert_eq!(out_random.nrows(), 2);

    let mut cfg_ucb = turbo_enn();
    cfg_ucb.acquisition = AcquisitionConfig::UCB { beta: 1.0 };
    let mut opt_ucb =
        Optimizer::new_strategy(bounds.clone(), cfg_ucb, Strategy::turbo(), &mut rng).unwrap();
    let x_fit = array![[0.0, 0.0], [1.0, 1.0], [0.2, 0.8], [0.8, 0.2]];
    let y_fit = array![[0.0], [1.0], [0.5], [0.4]];
    opt_ucb
        .tell(&x_fit.view(), &y_fit.view(), &mut rng)
        .unwrap();
    let out_ucb = select_arms(&opt_ucb, &x_cand.view(), 2, &mut rng).unwrap();
    assert_eq!(out_ucb.nrows(), 2);

    let mut cfg_ts = turbo_enn();
    cfg_ts.acquisition = AcquisitionConfig::Thompson;
    let mut opt_ts =
        Optimizer::new_strategy(bounds.clone(), cfg_ts, Strategy::turbo(), &mut rng).unwrap();
    opt_ts.tell(&x_fit.view(), &y_fit.view(), &mut rng).unwrap();
    let out_ts = select_arms(&opt_ts, &x_cand.view(), 2, &mut rng).unwrap();
    assert_eq!(out_ts.nrows(), 2);

    let mut cfg_pareto = turbo_enn();
    cfg_pareto.acquisition = AcquisitionConfig::Pareto;
    let mut opt_pareto =
        Optimizer::new_strategy(bounds, cfg_pareto, Strategy::turbo(), &mut rng).unwrap();
    opt_pareto
        .tell(&x_fit.view(), &y_fit.view(), &mut rng)
        .unwrap();
    let out_pareto = select_arms(&opt_pareto, &x_cand.view(), 2, &mut rng).unwrap();
    assert_eq!(out_pareto.nrows(), 2);
}
