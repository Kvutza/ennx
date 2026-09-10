//! Golden tests for mathy paths listed in `weak_tests.md` (integration crate tests).

use ennx::{
    conditional_internals, ENNParams, IndexDriver, ParetoAcquisition, PosteriorComputation,
    PosteriorFlags, ENN,
};
use ndarray::array;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn unit_model() -> ENN {
    let train_x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
    let train_y = array![[0.0], [1.0], [1.0], [2.0]];
    ENN::new(train_x, train_y, None, false, IndexDriver::Exact).unwrap()
}

#[test]
fn weak_order() {
    let pareto = ParetoAcquisition::new();
    let mu = array![[1.0], [1.0], [2.0]];
    let se = array![[0.01], [10.0], [1.5]];
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let selected = pareto.select(&mu.view(), &se.view(), 1, &mut rng).unwrap();
    assert_eq!(selected, vec![2]);
}

#[test]
fn weak_whatif() {
    let train_x = array![[0.0, 0.0], [10.0, 10.0]];
    let train_y = array![[0.0], [10.0]];
    let model = ENN::new(train_x, train_y, None, false, IndexDriver::Exact).unwrap();
    let params = ENNParams::new(2, 1.0, 0.1).unwrap();
    let flags = PosteriorFlags::new();
    let internals = conditional_internals(
        &model,
        &array![[1.0, 1.0]].view(),
        &array![[2.0, 2.0]].view(),
        &array![[5.0]].view(),
        &params,
        &flags,
    )
    .unwrap();
    assert!(internals.idx[0].contains(&2));
}

#[test]
fn weak_golden() {
    let model = unit_model();
    let params = ENNParams::new(2, 1.0, 0.1).unwrap();
    let flags = PosteriorFlags::new();
    let query = array![[0.5, 0.5]];
    let seeds = [7i64];
    let (draws, _) = model
        .posterior_draw(&query.view(), &params, &seeds, &flags)
        .unwrap();
    assert!((draws[[0, 0, 0]] - 0.223_806_179_572_216_43).abs() < 1e-12);
}
