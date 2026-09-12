use ennx::{
    ENN, ENNParams, EnnStorage, IndexDriver, ModelOptions, row_loglik, subsample_loglik,
    subsample_model,
};
use ndarray::{Array2, array};
use rand::{SeedableRng, rngs::StdRng};

#[test]
fn row_guards() {
    let x = array![[0.0], [1.0], [2.0]];
    let y = array![[0.0], [1.0], [2.0]];
    let model = ENN::new(x.clone(), y.clone(), None, false, IndexDriver::Exact).unwrap();
    let params = [ENNParams::new(2, 1.0, 0.1).unwrap()];
    let mut rng = StdRng::seed_from_u64(42);
    assert!(row_loglik(&model, &[3], &params, 1, &mut rng, None).is_err());
    assert!(row_loglik(&model, &[0], &params, 0, &mut rng, None).is_err());
    assert!(row_loglik(&model, &[0], &[], 1, &mut rng, None).is_err());
    assert!(
        row_loglik(
            &model,
            &[0],
            &params,
            1,
            &mut rng,
            Some(&array![1.0, 2.0].view())
        )
        .is_err()
    );
    assert!(subsample_model(&model, &[], 1, &mut rng, None).is_err());
    assert!(subsample_model(&model, &params, 0, &mut rng, None).is_err());
    let reordered = array![[2.0], [1.0], [0.0]];
    assert!(
        subsample_loglik(
            &model,
            &reordered.view(),
            &reordered.view(),
            &params,
            3,
            &mut rng,
            None
        )
        .is_err()
    );
    assert!(
        subsample_loglik(
            &model,
            &array![[0.0]].view(),
            &array![[0.0]].view(),
            &params,
            1,
            &mut rng,
            None
        )
        .is_err()
    );
    let scale = array![1.0];
    let ordered = row_loglik(
        &model,
        &[0, 1, 2],
        &params,
        3,
        &mut rng,
        Some(&scale.view()),
    )
    .unwrap();
    let shuffled = row_loglik(
        &model,
        &[2, 0, 1],
        &params,
        3,
        &mut rng,
        Some(&scale.view()),
    )
    .unwrap();
    assert!((ordered[0] - shuffled[0]).abs() < 1e-12);
}

#[test]
fn disk_loo() {
    let dir = tempfile::TempDir::new().unwrap();
    let x = Array2::zeros((4, 1));
    let y = array![[0.0], [5.0], [10.0], [15.0]];
    let yvar = array![[1.0], [2.0], [3.0], [4.0]];
    let disk = ENN::new_storage(
        x.clone(),
        y.clone(),
        Some(yvar.clone()),
        false,
        IndexDriver::BpAnnDisk,
        EnnStorage::Disk,
        Some(dir.path().to_path_buf()),
    )
    .unwrap();
    let memory = ENN::new(x, y, Some(yvar), false, IndexDriver::Exact).unwrap();
    let params = [ENNParams::new(3, 0.7, 0.1).unwrap()];
    for row in 0..4 {
        let mut rng = StdRng::seed_from_u64(42);
        let a = row_loglik(&disk, &[row], &params, 1, &mut rng, None).unwrap();
        let b = row_loglik(&memory, &[row], &params, 1, &mut rng, None).unwrap();
        assert!((a[0] - b[0]).abs() < 1e-12);
    }
}

#[test]
fn bounded_loo() {
    let x = array![[0.0], [0.0], [0.0]];
    let y = array![[0.2], [0.5], [0.8]];
    let model = ENN::new_options(
        x.clone(),
        y.clone(),
        None,
        ModelOptions {
            y_bounds: Some(array![[0.0, 1.0]]),
            ..Default::default()
        },
    )
    .unwrap();
    let params = [ENNParams::new(2, 0.7, 0.1).unwrap()];
    let scale = array![1.0];
    let mut rng = StdRng::seed_from_u64(42);
    let a = row_loglik(
        &model,
        &[0, 1, 2],
        &params,
        3,
        &mut rng,
        Some(&scale.view()),
    )
    .unwrap();
    let b = subsample_loglik(
        &model,
        &x.view(),
        &y.view(),
        &params,
        3,
        &mut rng,
        Some(&scale.view()),
    )
    .unwrap();
    assert!(a[0].is_finite());
    assert!((a[0] - b[0]).abs() < 1e-12);
}
