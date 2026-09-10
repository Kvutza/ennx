use ndarray::array;
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::config::turbo_enn;
use crate::fitter::ENNFitter;
use crate::index::IndexDriver;
use crate::mbtrregn::{MorboTRSettings, Rescalarize};
use crate::model::ENN;
use crate::optimizer::{ObservationDelta, Optimizer};
use crate::strategy::Strategy;
use crate::surrogate::{ENNSurrogate, ENNSurrogateConfig, Surrogate};
use crate::trregncfg::TrustRegionConfig;
use crate::trust_region::TRLengthConfig;

#[test]
fn x_add() {
    let train_x = array![[0.0, 0.0], [1.0, 0.0]];
    let train_y = array![[0.0], [1.0]];
    let mut model = ENN::new(train_x, train_y, None, false, IndexDriver::Exact).unwrap();
    model.ensure_sync().unwrap();
    assert!(!model.index_access().is_stale());
    let x_add = array![[0.5, 0.5]];
    let y_add = array![[0.5]];
    model.add(&x_add.view(), &y_add.view(), None).unwrap();
    assert!(!model.index_access().is_stale());
    model.ensure_sync().unwrap();
    assert_eq!(model.num_obs(), 3);
    assert_eq!(model.index_access().len(), 3);
}

#[test]
fn x_succeeds() {
    let train_x = array![[0.0, 0.0], [1.0, 0.0]];
    let train_y = array![[0.0], [1.0]];
    let mut model = ENN::new(train_x, train_y, None, true, IndexDriver::Exact).unwrap();
    model.ensure_sync().unwrap();
    let x_add = array![[0.5, 0.5]];
    let y_add = array![[0.5]];
    model
        .add(&x_add.view(), &y_add.view(), None)
        .expect("scale_x=true append to non-empty model must succeed");
    model.ensure_sync().unwrap();
    assert_eq!(model.num_obs(), 3);
    assert_eq!(model.index_access().len(), 3);
}

#[test]
fn x_add2() {
    let train_x = array![[0.0, 0.0], [1.0, 0.0]];
    let train_y = array![[0.0], [1.0]];
    let mut model = ENN::new(train_x, train_y, None, true, IndexDriver::Exact).unwrap();
    model.ensure_sync().unwrap();
    assert!(!model.index_access().is_stale());
    let x_add = array![[0.5, 0.5]];
    let y_add = array![[0.5]];
    model.add(&x_add.view(), &y_add.view(), None).unwrap();
    assert!(
        model.index_access().is_stale(),
        "scale_x append must mark index stale before ensure_sync"
    );
    model.ensure_sync().unwrap();
    assert!(!model.index_access().is_stale());
}

#[test]
fn enn_obs() {
    let train_x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
    let train_y = array![[0.0], [1.0], [1.0], [2.0]];
    let model = ENN::new(train_x, train_y, None, false, IndexDriver::Exact).unwrap();
    let mut fitter = ENNFitter::new(2, true);
    let all: Vec<usize> = (0..model.len()).collect();
    let (_, ty, _) = model.rows().train_rows(&all).unwrap();
    fitter.y_stats(&ty.view());
    let mut rng = StdRng::seed_from_u64(99);
    let p = fitter.ask(&model, 4, 3, None, &mut rng).unwrap();
    assert_eq!(p.k_neighbors, 2);
    assert!(p.epistemic_scale > 0.0);
}

#[test]
fn observation_views() {
    let delta = ObservationDelta {
        old_n: 2,
        new_n: 4,
        x_new: array![[0.1, 0.2], [0.3, 0.4]],
        y_new: array![[1.0], [2.0]],
    };
    assert_eq!(delta.x_view().nrows(), 2);
    assert_eq!(delta.y_view().nrows(), 2);
}

#[test]
fn enn_model() {
    let config = ENNSurrogateConfig {
        k: 2,
        num_candidates: 4,
        num_samples: 2,
        ..Default::default()
    };
    let mut sur = ENNSurrogate::new(config);
    let mut rng = StdRng::seed_from_u64(42);
    let x0 = array![[0.0, 0.0], [1.0, 0.0]];
    let y0 = array![[0.0], [1.0]];
    sur.fit_append(&x0.view(), &y0.view(), None, &mut rng)
        .unwrap();
    let x1 = array![[0.5, 0.5]];
    let y1 = array![[1.5]];
    sur.fit_append(&x1.view(), &y1.view(), None, &mut rng)
        .unwrap();
    assert_eq!(sur.model().unwrap().num_obs(), 3);
}

#[test]
fn tell_metrics() {
    let bounds = array![[0.0, 1.0], [0.0, 1.0]];
    let mut rng = StdRng::seed_from_u64(88);
    let cfg = turbo_enn();
    let mut opt = Optimizer::new(bounds, cfg, &mut rng).unwrap();
    let x0 = array![[0.2, 0.3]];
    let y0 = array![[1.0]];
    opt.tell(&x0.view(), &y0.view(), &mut rng).unwrap();
    let x1 = array![[0.4, 0.5], [0.6, 0.7]];
    let y1 = array![[2.0, 3.0], [4.0, 5.0]];
    let err = opt.tell(&x1.view(), &y1.view(), &mut rng);
    assert!(err.is_err());
    let msg = err.unwrap_err().to_string();
    assert!(
        msg.contains("unsupported"),
        "expected unsupported metrics change error, got: {msg}"
    );
}

#[test]
fn morbo_ranges() {
    let bounds = array![[0.0, 1.0], [0.0, 1.0]];
    let mut rng = StdRng::seed_from_u64(77);
    let mut cfg = turbo_enn();
    cfg.trust_region = TrustRegionConfig::Morbo(MorboTRSettings {
        num_metrics: 2,
        alpha: 0.05,
        length: TRLengthConfig::default(),
        rescalarize: Rescalarize::OnRestart,
        noise_aware: false,
    });
    let mut opt = Optimizer::new_strategy(bounds, cfg, Strategy::turbo(), &mut rng).unwrap();
    let x0 = array![[0.1, 0.2], [0.3, 0.4]];
    let y0 = array![[1.0, 0.5], [0.2, 0.9]];
    opt.tell(&x0.view(), &y0.view(), &mut rng).unwrap();
    let morbo = opt.trust_region().morbo().expect("morbo");
    let ymin_before = morbo.y_min().expect("y_min").to_owned();
    let ymax_before = morbo.y_max().expect("y_max").to_owned();
    let _ = opt.ask(2, &mut rng).unwrap();
    let morbo_after = opt.trust_region().morbo().expect("morbo");
    assert!(morbo_after
        .y_min()
        .unwrap()
        .iter()
        .zip(ymin_before.iter())
        .all(|(a, b)| (a - b).abs() < 1e-12));
    assert!(morbo_after
        .y_max()
        .unwrap()
        .iter()
        .zip(ymax_before.iter())
        .all(|(a, b)| (a - b).abs() < 1e-12));
}
