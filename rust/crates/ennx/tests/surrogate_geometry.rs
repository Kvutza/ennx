//! Model limitations plus regression coverage for the repaired fitting defects.
//! These use the actual Exact ENN backend; no inference model or GPU is needed.

use ennx::acquisition::UCBAcquisition;
use ennx::{
    ENN, ENNFitter, ENNParams, IndexDriver, PosteriorComputation, PosteriorFlags, row_loglik,
    subsample_loglik,
};
use ndarray::{Array2, array};
use rand::{SeedableRng, rngs::StdRng};

fn model(x: Array2<f64>, y: Array2<f64>) -> ENN {
    ENN::new(x, y, None, false, IndexDriver::Exact).unwrap()
}

fn flags() -> PosteriorFlags {
    PosteriorFlags::new().tie_neighbors(false)
}

#[test]
fn reflection() {
    let model = model(array![[-1.0, 0.0], [1.0, 0.0]], array![[-1.0], [1.0]]);
    let queries = array![[0.25, 1.0], [0.25, -1.0]];
    let params = ENNParams::new(2, 1.0, 0.0).unwrap();
    let post = model.posterior(&queries.view(), &params, &flags()).unwrap();
    let (draws, _) = model
        .posterior_draw(&queries.view(), &params, &[1, 2, 3, 4], &flags())
        .unwrap();
    assert_eq!(post.mu[[0, 0]], post.mu[[1, 0]]);
    assert_eq!(post.se[[0, 0]], post.se[[1, 0]]);
    for seed in 0..4 {
        assert_eq!(draws[[seed, 0, 0]], draws[[seed, 1, 0]]);
    }
    // Both training labels fit f(x,y)=x+y, but query rewards are 1.25 and -0.75.
    eprintln!("reflection: identical ENN distributions; compatible linear rewards differ by 2");
}

#[test]
fn shared_draw() {
    let model = model(array![[0.0, 0.0]], array![[0.0]]);
    let queries = array![[1.0, 0.0], [0.0, 2.0], [-3.0, 0.0]];
    let params = ENNParams::new(1, 1.0, 0.0).unwrap();
    let post = model.posterior(&queries.view(), &params, &flags()).unwrap();
    let (draws, _) = model
        .posterior_draw(&queries.view(), &params, &[1, 2, 3, 4], &flags())
        .unwrap();
    for seed in 0..4 {
        let z = draws[[seed, 0, 0]] / post.se[[0, 0]];
        for query in 0..3 {
            assert_eq!(post.mu[[query, 0]], 0.0);
            assert!((draws[[seed, query, 0]] / post.se[[query, 0]] - z).abs() < 1e-12);
        }
    }
    eprintln!("one anchor: all standardized Thompson draws identical across queries");
}

#[test]
fn dup_precision() {
    let one = model(array![[0.0]], array![[0.0]]);
    let repeated = model(Array2::zeros((16, 1)), Array2::zeros((16, 1)));
    let params = ENNParams::new(16, 1.0, 0.0).unwrap();
    let query = array![[1.0]];
    let first = one.posterior(&query.view(), &params, &flags()).unwrap();
    let second = repeated
        .posterior(&query.view(), &params, &flags())
        .unwrap();
    let ratio = first.se[[0, 0]] / second.se[[0, 0]];
    assert!((ratio - 4.0).abs() < 1e-12);
    eprintln!(
        "duplicate records: 16 copies reduce remote SE by {ratio}x without new spatial information"
    );
}

#[test]
fn convex_mean() {
    let model = model(array![[-1.0], [1.0]], array![[-1.0], [1.0]]);
    let params = ENNParams::new(2, 1.0, 0.0).unwrap();
    let post = model
        .posterior(&array![[2.0]].view(), &params, &flags())
        .unwrap();
    assert!((post.mu[[0, 0]] - 0.8).abs() < 1e-8);
    assert!(post.mu[[0, 0]] < 1.0);
    eprintln!("linear extrapolation: f(2)=2, ENN mean={}", post.mu[[0, 0]]);
}

#[test]
fn neighbor_jump() {
    let model = model(array![[-1.0], [1.0]], array![[-1.0], [1.0]]);
    let params = ENNParams::new(1, 1.0, 0.0).unwrap();
    let queries = array![[-1e-8], [1e-8]];
    let post = model.posterior(&queries.view(), &params, &flags()).unwrap();
    assert_eq!(post.mu[[0, 0]], -1.0);
    assert_eq!(post.mu[[1, 0]], 1.0);
    eprintln!("neighbor boundary: query separation 2e-8, mean jump 2 for labels fitting f(x)=x");
}

#[test]
fn draw_conditioning() {
    let mut model = model(array![[0.0, 0.0]], array![[0.0]]);
    let params = ENNParams::new(2, 1.0, 0.0).unwrap();
    let queries = array![[1.0, 0.0], [0.0, 1.0]];
    let (before, _) = model
        .posterior_draw(&queries.view(), &params, &[1, 2, 3, 4], &flags())
        .unwrap();
    for seed in 0..4 {
        assert_eq!(before[[seed, 0, 0]], before[[seed, 1, 0]]);
    }
    // The previous draw law makes f(a)=f(b) almost surely. Conditioning on an
    // exact observation f(a)=0 would also fix f(b)=0 under that same law.
    model
        .add(&array![[1.0, 0.0]].view(), &array![[0.0]].view(), None)
        .unwrap();
    let after = model.posterior(&queries.view(), &params, &flags()).unwrap();
    assert!(after.se[[1, 0]] > 0.8);
    eprintln!(
        "conditioning: identical pre-update random variables, post-add remote SE={}",
        after.se[[1, 0]]
    );
}

#[test]
fn label_permutation() {
    let x = array![[0.0], [1.0], [2.0], [3.0]];
    let first = model(x.clone(), array![[-3.0], [-1.0], [1.0], [3.0]]);
    let second = model(x, array![[3.0], [-1.0], [1.0], [-3.0]]);
    let params = ENNParams::new(2, 1.0, 0.0).unwrap();
    let query = array![[0.25]];
    let a = first.posterior(&query.view(), &params, &flags()).unwrap();
    let b = second.posterior(&query.view(), &params, &flags()).unwrap();
    assert!((a.mu[[0, 0]] - b.mu[[0, 0]]).abs() > 1.0);
    assert_eq!(a.se[[0, 0]], b.se[[0, 0]]);
    eprintln!(
        "local label disagreement: mean changes, SE unchanged at fixed params/global output scale"
    );
}

#[test]
fn invalid_fit() {
    let x = array![[-1.0], [1.0]];
    let y = array![[-1.0], [1.0]];
    let model = model(x.clone(), y.clone());
    let params = [
        ENNParams::new(1, 1.0, 0.0).unwrap(),
        ENNParams::new(1, f64::MAX, 0.0).unwrap(),
    ];
    let fit_flags = flags().exclude_nearest(true).observation_noise(true);
    let invalid = model.posterior(&x.view(), &params[1], &fit_flags).unwrap();
    assert!(
        invalid
            .mu
            .iter()
            .chain(invalid.se.iter())
            .any(|v| !v.is_finite())
    );
    let scores = subsample_loglik(
        &model,
        &x.view(),
        &y.view(),
        &params,
        2,
        &mut StdRng::seed_from_u64(42),
        None,
    )
    .unwrap();
    assert!(scores[0].is_finite() && scores[0] < 0.0);
    assert_eq!(scores[1], f64::NEG_INFINITY);
    assert!(scores[1] < scores[0]);
    let mut fitter = ENNFitter::new(1, false);
    fitter.tell(&x.view(), &y.view(), None).unwrap();
    let selected = fitter
        .ask(
            &model,
            8,
            2,
            Some(&params[1]),
            &mut StdRng::seed_from_u64(42),
        )
        .unwrap();
    assert!(selected.epistemic_scale < f64::MAX);
    let previous = *fitter.params().unwrap();
    assert!(
        fitter
            .ask(
                &model,
                0,
                2,
                Some(&params[1]),
                &mut StdRng::seed_from_u64(42)
            )
            .is_err()
    );
    assert_eq!(*fitter.params().unwrap(), previous);
    eprintln!(
        "fit scores: valid={}, invalid={}; invalid candidates excluded",
        scores[0], scores[1]
    );
}

#[test]
fn loo_leak() {
    let x = Array2::zeros((3, 1));
    let y = array![[0.0], [5.0], [10.0]];
    let variances = array![[1.0], [4.0], [9.0]];
    let model = ENN::new(
        x.clone(),
        y.clone(),
        Some(variances.clone()),
        false,
        IndexDriver::Exact,
    )
    .unwrap();
    let params: Vec<_> = [1, 2, 8]
        .into_iter()
        .map(|k| ENNParams::new(k, 1.0, 1.0).unwrap())
        .collect();
    let scale = array![1.0];
    let mut expected = vec![0.0; params.len()];
    for held in [2, 0, 1] {
        let actual = row_loglik(
            &model,
            &[held],
            &params,
            1,
            &mut StdRng::seed_from_u64(42),
            Some(&scale.view()),
        )
        .unwrap();
        for (i, p) in params.iter().enumerate() {
            let others: Vec<_> = (0..3)
                .filter(|&id| id != held)
                .take(p.k_neighbors as usize)
                .collect();
            let precision: Vec<_> = others
                .iter()
                .map(|&id| 1.0 / (1e-9 + 1.0 + variances[[id, 0]] / (50.0 / 3.0)))
                .collect();
            let total: f64 = precision.iter().sum();
            let mean: f64 = others
                .iter()
                .zip(&precision)
                .map(|(&id, w)| w * y[[id, 0]] / total)
                .sum();
            let noise: f64 = others
                .iter()
                .zip(&precision)
                .map(|(&id, w)| w / total * (50.0 / 3.0 + variances[[id, 0]]))
                .sum();
            let variance = (50.0 / 3.0) / total + noise;
            let score = -0.5
                * ((2.0 * std::f64::consts::PI * variance).ln()
                    + (y[[held, 0]] - mean).powi(2) / variance);
            assert!(
                (actual[i] - score).abs() < 1e-12,
                "held={held}, k={}: {} != {score}",
                p.k_neighbors,
                actual[i]
            );
            expected[i] += score;
        }
    }
    let scores = subsample_loglik(
        &model,
        &x.view(),
        &y.view(),
        &params,
        3,
        &mut StdRng::seed_from_u64(42),
        Some(&scale.view()),
    )
    .unwrap();
    for (a, b) in scores.iter().zip(expected) {
        assert!((a - b).abs() < 1e-12);
    }
    eprintln!(
        "LOO: explicit exclusion matches independent likelihoods for every duplicate row and K"
    );
}

#[test]
fn adaptive_bias() {
    let params = ENNParams::new(1, 1.0, 1.0).unwrap();
    let query = array![[1.0]];
    let mut adaptive_sum = 0.0;
    let mut fixed_sum = 0.0;
    // Enumerate all equiprobable independent +/-1 noises for f(x)=0.
    // Only the second location depends on the first observed outcome.
    for first_noise in [-1.0, 1.0] {
        for second_noise in [-1.0, 1.0] {
            let y = array![[first_noise], [second_noise]];
            let adaptive = model(array![[0.0], [first_noise]], y.clone());
            let fixed = model(array![[0.0], [-1.0]], y);
            adaptive_sum += adaptive
                .posterior(&query.view(), &params, &flags())
                .unwrap()
                .mu[[0, 0]];
            fixed_sum += fixed
                .posterior(&query.view(), &params, &flags())
                .unwrap()
                .mu[[0, 0]];
        }
    }
    assert_eq!(adaptive_sum / 4.0, -0.5);
    assert_eq!(fixed_sum / 4.0, 0.0);
    eprintln!("adaptive design: exact expected prediction=-0.5 for f=0; fixed design=0");
}

#[test]
fn spatial_bias() {
    for k in [1, 4, 16, 64] {
        let x = Array2::from_shape_fn((k, 2), |(i, j)| {
            if j == 0 {
                1.0
            } else {
                0.01 * i as f64 / k as f64
            }
        });
        // All distinct observations fit the 1-Lipschitz linear objective f(x)=-x[0].
        let model = model(x, Array2::from_elem((k, 1), -1.0));
        let params = ENNParams::new(k as i32, 1.0, 0.0).unwrap();
        let post = model
            .posterior(&array![[0.0, 0.0]].view(), &params, &flags())
            .unwrap();
        let error = post.mu[[0, 0]].abs();
        let se = post.se[[0, 0]];
        assert!((error - 1.0).abs() < 1e-12);
        assert!(error / se > 0.999 * (k as f64).sqrt());
        if k > 1 {
            assert!(post.mu[[0, 0]] + se < 0.0);
        }
        eprintln!(
            "one-sided distinct neighbors: K={k}, error={error}, SE={se}, error/SE={}",
            error / se
        );
    }
}

#[test]
fn ucb_trap() {
    // Continuous piecewise-linear f on [0,1], optimum f(0.25)=2;
    // all points in [0.5,1] have rewards <=1. No trust-region restriction.
    let objective = |x: f64| {
        if x <= 0.25 {
            8.0 * x
        } else if x <= 0.5 {
            4.0 - 8.0 * x
        } else {
            2.0 * x - 1.0
        }
    };
    let mut model = model(array![[0.0], [0.5], [1.0]], array![[0.0], [0.0], [1.0]]);
    let params = ENNParams::new(1, 1.0, 0.0).unwrap();
    let candidates = Array2::from_shape_fn((101, 1), |(i, _)| i as f64 / 100.0);
    let mut rng = StdRng::seed_from_u64(42);
    let acquisition = UCBAcquisition::new(1.0);
    for _ in 0..32 {
        let post = model
            .posterior(&candidates.view(), &params, &flags())
            .unwrap();
        for i in 0..50 {
            assert!(post.mu[[i, 0]] + post.se[[i, 0]] < 0.26);
        }
        let means = post
            .mu
            .view()
            .into_dimensionality::<ndarray::Ix2>()
            .unwrap();
        let errors = post
            .se
            .view()
            .into_dimensionality::<ndarray::Ix2>()
            .unwrap();
        let selected = acquisition
            .select(&means.column(0), &errors.column(0), 1, &mut rng)
            .unwrap()[0];
        let x = candidates[[selected, 0]];
        assert!(x > 0.5);
        assert!(objective(x) <= 1.0);
        model
            .add(&array![[x]].view(), &array![[objective(x)]].view(), None)
            .unwrap();
    }
    assert_eq!(objective(candidates[[25, 0]]), 2.0);
    eprintln!("fixed UCB: 32 rounds ignore available optimum x=0.25, f=2; best observed=1");
}
