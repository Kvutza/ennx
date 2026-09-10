use super::{
    ask_hybrid, ask_init, ask_turbo, select_pareto, select_random, select_segment, select_thompson,
    select_ucb, tell_common, tell_init, tell_turbo, CandidateSegment, InitStrategy,
    InitStrategyState, Strategy, TurboStrategyState,
};
use crate::config::{turbo_enn, turbo_zero, AcquisitionConfig};
use crate::optimizer::{Optimizer, Telemetry};
use ndarray::array;
use rand::rngs::StdRng;
use rand::SeedableRng;

#[test]
fn test_001() {
    let bounds = array![[0.0, 1.0], [0.0, 1.0]];
    let mut rng = StdRng::seed_from_u64(77);
    let mut optimizer =
        Optimizer::new_strategy(bounds, turbo_zero(), Strategy::turbo(), &mut rng).unwrap();

    let init_state = InitStrategyState::new(InitStrategy::Random, 3);
    let x_init = ask_init(&init_state, &mut optimizer, 2, &mut rng).unwrap();
    assert_eq!(x_init.nrows(), 2);

    let mut telemetry = Telemetry::default();
    let init_state_h = InitStrategyState::new(InitStrategy::LHD, 3);
    let x_init_h = ask_hybrid(&init_state_h, &mut optimizer, 2, &mut rng).unwrap();
    assert_eq!(x_init_h.nrows(), 2);

    let mut init_state2 = InitStrategyState::new(InitStrategy::LHD, 3);
    let y_init = array![[0.1], [0.2]];
    tell_init(
        &mut init_state2,
        &mut optimizer,
        &x_init_h.view(),
        &y_init.view(),
        None,
        &mut rng,
    )
    .unwrap();
    assert_eq!(init_state2.completed, 2);

    let x_turbo = ask_turbo(&mut optimizer, 2, &mut telemetry, &mut rng).unwrap();
    assert_eq!(x_turbo.nrows(), 2);
    let y_turbo = array![[0.3], [0.4]];
    tell_turbo(
        &mut optimizer,
        &x_turbo.view(),
        &y_turbo.view(),
        None,
        &mut telemetry,
        &mut rng,
    )
    .unwrap();
}

#[test]
fn test_002() {
    let bounds = array![[0.0, 1.0], [0.0, 1.0]];
    let mut rng = StdRng::seed_from_u64(42);

    let mut cfg = turbo_enn();
    cfg.acquisition = AcquisitionConfig::Thompson;
    let mut optimizer = Optimizer::new_strategy(bounds, cfg, Strategy::turbo(), &mut rng).unwrap();

    let x_fit = array![[0.0, 0.0], [1.0, 1.0], [0.2, 0.8], [0.8, 0.2], [0.5, 0.5]];
    let y_fit = array![[0.0], [1.0], [0.5], [0.4], [0.6]];
    optimizer
        .tell(&x_fit.view(), &y_fit.view(), &mut rng)
        .unwrap();

    let tel_after_tell = optimizer.telemetry();
    assert!(
        tel_after_tell.dt_fit > 0.0,
        "dt_fit should be populated after surrogate fitting (regression test for B3)"
    );

    let tel_before_ask = optimizer.telemetry().clone();
    let _candidates = optimizer.ask(2, &mut rng).unwrap();
    let tel_after_ask = optimizer.telemetry();
    assert!(
        tel_after_ask.dt_sel > 0.0 || tel_after_ask.dt_sel != tel_before_ask.dt_sel,
        "dt_sel should be populated after arm selection (regression test for B3)"
    );

    let mut rng2 = StdRng::seed_from_u64(42);
    let mut cfg2 = turbo_enn();
    cfg2.acquisition = AcquisitionConfig::Thompson;
    let mut optimizer2 = Optimizer::new_strategy(
        array![[0.0, 1.0], [0.0, 1.0]],
        cfg2,
        Strategy::turbo(),
        &mut rng2,
    )
    .unwrap();
    optimizer2
        .tell(&x_fit.view(), &y_fit.view(), &mut rng2)
        .unwrap();

    let _candidates1 = optimizer.ask(2, &mut rng).unwrap();
    let _candidates2 = optimizer2.ask(2, &mut rng2).unwrap();
}

#[test]
fn test_003() {
    let bounds = array![[0.0, 1.0], [0.0, 1.0]];
    let mut rng = StdRng::seed_from_u64(123);

    let strategy = Strategy::hybrid(InitStrategy::Random, 4);
    let mut optimizer = Optimizer::new_strategy(bounds, turbo_zero(), strategy, &mut rng).unwrap();

    let x1 = optimizer.ask(2, &mut rng).unwrap();
    assert_eq!(x1.nrows(), 2);

    let y1 = array![[0.1], [0.2]];
    optimizer.tell(&x1.view(), &y1.view(), &mut rng).unwrap();

    let progress = optimizer.init_progress();
    assert!(progress.is_some(), "Should be in init phase");
    let (done, total) = progress.unwrap();
    assert_eq!(done, 2);
    assert_eq!(total, 4);

    let x2 = optimizer.ask(2, &mut rng).unwrap();
    let y2 = array![[0.3], [0.4]];
    optimizer.tell(&x2.view(), &y2.view(), &mut rng).unwrap();

    assert!(
        optimizer.init_progress().is_none(),
        "Should have exited init phase"
    );

    let x3 = optimizer.ask(2, &mut rng).unwrap();
    assert_eq!(x3.nrows(), 2);
}

#[test]
fn failed_state() {
    let bounds = array![[0.0, 1.0], [0.0, 1.0]];
    let mut rng = StdRng::seed_from_u64(124);
    let strategy = Strategy::hybrid(InitStrategy::LHD, 3);
    let mut opt = Optimizer::new_strategy(bounds, turbo_enn(), strategy, &mut rng).unwrap();

    let x = opt.ask(1, &mut rng).unwrap();
    opt.tell(&x.view(), &array![[0.1, 0.2]].view(), &mut rng)
        .unwrap();
    assert_eq!(opt.init_progress(), Some((1, 3)));
    assert_eq!(opt.obs_count(), 1);

    let x = opt.ask(1, &mut rng).unwrap();
    assert!(opt
        .tell(&x.view(), &array![[0.3]].view(), &mut rng)
        .is_err());
    assert_eq!(opt.init_progress(), Some((1, 3)));
    assert_eq!(opt.obs_count(), 1);
}

#[test]
fn test_005() {
    let bounds = array![[0.0, 1.0], [0.0, 1.0]];
    let mut rng = StdRng::seed_from_u64(99);

    let mut cfg = turbo_enn();
    cfg.acquisition = AcquisitionConfig::UCB { beta: 2.0 };
    let mut optimizer = Optimizer::new_strategy(bounds, cfg, Strategy::turbo(), &mut rng).unwrap();

    let tel0 = optimizer.telemetry();
    assert_eq!(tel0.dt_fit, 0.0);
    assert_eq!(tel0.dt_sel, 0.0);

    let x_fit = array![[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]];
    let y_fit = array![[0.0], [1.0], [0.5]];
    optimizer
        .tell(&x_fit.view(), &y_fit.view(), &mut rng)
        .unwrap();

    let tel1 = optimizer.telemetry();
    assert!(
        tel1.dt_fit > 0.0,
        "dt_fit should be > 0 after surrogate fitting, got {}",
        tel1.dt_fit
    );

    let _candidates = optimizer.ask(2, &mut rng).unwrap();

    let tel2 = optimizer.telemetry();
    assert!(
        tel2.dt_sel > 0.0,
        "dt_sel should be > 0 after arm selection, got {}",
        tel2.dt_sel
    );
}

#[test]
fn turbo_paths() {
    let _: TurboStrategyState = TurboStrategyState;

    let bounds = array![[0.0, 1.0], [0.0, 1.0]];
    let mut rng = StdRng::seed_from_u64(200);
    let mut opt_no_sur =
        Optimizer::new_strategy(bounds.clone(), turbo_zero(), Strategy::turbo(), &mut rng).unwrap();
    let x = array![[0.2, 0.3]];
    let y = array![[0.1]];
    tell_common(&mut opt_no_sur, &x.view(), &y.view(), None, None, &mut rng).unwrap();

    let mut opt_enn =
        Optimizer::new_strategy(bounds, turbo_enn(), Strategy::turbo(), &mut rng).unwrap();
    let x2 = array![[0.0, 0.0], [1.0, 1.0]];
    let y2 = array![[0.0], [1.0]];
    let mut tel = Telemetry::default();
    tell_common(
        &mut opt_enn,
        &x2.view(),
        &y2.view(),
        None,
        Some(&mut tel),
        &mut rng,
    )
    .unwrap();
    assert!(tel.dt_fit > 0.0);
}

#[test]
fn tr_trip() {
    let bounds = array![[0.0, 1.0], [0.0, 1.0]];
    let mut rng = StdRng::seed_from_u64(203);
    let strategy = Strategy::experimental(2, 3, 2, &mut rng).unwrap();
    let mut opt = Optimizer::new_strategy(bounds, turbo_zero(), strategy, &mut rng).unwrap();

    let x_init = opt.ask(2, &mut rng).unwrap();
    assert_eq!(x_init.nrows(), 2);
    let y_init = array![[0.0], [1.0]];
    opt.tell(&x_init.view(), &y_init.view(), &mut rng).unwrap();
    assert!(opt.init_progress().is_none());

    let x_next = opt.ask(2, &mut rng).unwrap();
    assert_eq!(x_next.nrows(), 2);

    let y_next = array![[1.5], [2.0]];
    opt.tell(&x_next.view(), &y_next.view(), &mut rng).unwrap();
    assert!(opt.telemetry().num_candidates > 0);
}

#[test]
fn ask_pool() {
    let bounds = array![[0.0, 1.0], [0.0, 1.0]];
    let mut rng = StdRng::seed_from_u64(201);
    let mut config = turbo_enn();
    let num_candidates = 37usize;
    config.candidates.num_candidates_factor = 1.0;
    config.candidates.num_candidates_per_arm = Some(num_candidates);
    let num_arms = 2usize;
    let expected_pool = config.candidates.num_candidates(2, num_arms);
    let mut opt = Optimizer::new_strategy(bounds, config, Strategy::turbo(), &mut rng).unwrap();
    let xf = array![[0.0, 0.0], [1.0, 1.0], [0.2, 0.8]];
    let yf = array![[0.0], [1.0], [0.5]];
    opt.tell(&xf.view(), &yf.view(), &mut rng).unwrap();
    let _ = opt.ask(num_arms, &mut rng).unwrap();
    assert_eq!(
        opt.telemetry().num_candidates,
        expected_pool,
        "ask must score the full configured RAASP pool (no silent cap)"
    );
    assert!(
        expected_pool >= num_candidates * num_arms,
        "pool size must honor per-arm setting (got {expected_pool}, want at least {})",
        num_candidates * num_arms
    );
}

#[test]
fn select_smoke() {
    let bounds = array![[0.0, 1.0], [0.0, 1.0]];
    let mut rng = StdRng::seed_from_u64(202);
    let x_cand = array![[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]];

    let out_r = select_random(&x_cand.view(), 2, &mut rng).unwrap();
    assert_eq!(out_r.nrows(), 2);

    let mut opt =
        Optimizer::new_strategy(bounds, turbo_enn(), Strategy::turbo(), &mut rng).unwrap();
    let xf = array![[0.0, 0.0], [1.0, 1.0], [0.2, 0.8]];
    let yf = array![[0.0], [1.0], [0.5]];
    opt.tell(&xf.view(), &yf.view(), &mut rng).unwrap();
    let sur = opt.surrogate().expect("enn surrogate");

    let out_ts = select_thompson(&opt, sur, &x_cand.view(), 2, &mut rng).unwrap();
    assert_eq!(out_ts.nrows(), 2);

    let out_ucb = select_ucb(&opt, sur, &x_cand.view(), 2, 1.0, &mut rng).unwrap();
    assert_eq!(out_ucb.nrows(), 2);

    let out_pf = select_pareto(sur, &x_cand.view(), 2, &mut rng).unwrap();
    assert_eq!(out_pf.nrows(), 2);
}

#[test]
fn segmented_acquisition() {
    let candidates = array![
        [0.1, 0.2],
        [0.3, 0.4],
        [0.5, 0.6],
        [0.7, 0.8],
        [0.9, 0.1],
        [0.2, 0.9]
    ];
    let segments = [
        CandidateSegment {
            start: 0,
            end: 3,
            arms: 1,
            region: 0,
        },
        CandidateSegment {
            start: 3,
            end: 6,
            arms: 2,
            region: 1,
        },
    ];
    let fit_x = array![[0.0, 0.0], [1.0, 1.0], [0.2, 0.8]];
    let fit_y = array![[0.0], [1.0], [0.5]];
    let bounds = array![[0.0, 1.0], [0.0, 1.0]];
    let acquisitions = [
        AcquisitionConfig::Random,
        AcquisitionConfig::Thompson,
        AcquisitionConfig::UCB { beta: 1.0 },
        AcquisitionConfig::Pareto,
    ];

    for (seed, acquisition) in acquisitions.into_iter().enumerate() {
        let mut rng = StdRng::seed_from_u64(seed as u64 + 500);
        let mut config = turbo_enn();
        config.acquisition = acquisition;
        let mut optimizer =
            Optimizer::new_strategy(bounds.clone(), config, Strategy::turbo(), &mut rng).unwrap();
        optimizer
            .tell(&fit_x.view(), &fit_y.view(), &mut rng)
            .unwrap();
        let selected = select_segment(&optimizer, &candidates.view(), &segments, &mut rng).unwrap();
        assert_eq!(selected.len(), 3);
        assert!(selected[0] < 3);
        assert!(selected[1..].iter().all(|index| *index >= 3));
    }
}
