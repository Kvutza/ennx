use super::*;

#[test]
fn upstream_rng() {
    let mut rows = RowReservoir::new(4, 1, 1, 719).unwrap();
    for i in 0..20 {
        rows.push_row(&[i as f64], &[-(i as f64)]);
    }
    // NumPy Generator(PCG64(719)), Algorithm R, capacity 4.
    assert_eq!(rows.x(), [6.0, 15.0, 4.0, 18.0]);
    assert_eq!(rows.y(), [-6.0, -15.0, -4.0, -18.0]);
    assert_eq!(rows.num_seen(), 20);
    let mut rng = crate::metric_rng::MetricRng::from_seed(719);
    let got = [5, 6, 7, 1 << 32, (1 << 32) + 3, 1 << 48, 17].map(|high| rng.integers_high(high));
    assert_eq!(got, [2, 4, 0, 1822184769, 2849654795, 112494928035177, 13]);
}

#[test]
fn metric_validation() {
    assert!(AutoMetric::new(2, 1, vec![vec![0], vec![0]], 719).is_err());
    assert!(AutoMetric::new(2, 1, vec![vec![]], 719).is_err());
    assert!(auto_weights(&[], 100, 2, &[], 1, 10, &[]).is_err());
    let mut metric = AutoMetric::new(2, 1, vec![], 719).unwrap();
    assert!(metric.configure(Some(f64::NAN), None, None, None).is_err());
    assert!(metric.observe(&[f64::NAN, 0.0], &[1.0], 1).is_err());
    assert_eq!(metric.num_seen(), 0);
    for gain in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0, 0.0] {
        assert!(!accept_gain(gain));
    }
    assert!(accept_gain(1e-12));
}

#[test]
fn tied_weights() {
    let n = 128;
    let x = (0..n)
        .flat_map(|i| {
            [
                (i % 2) as f64,
                (1 - i % 2) as f64,
                (i * 17 % 97) as f64 / 97.0,
            ]
        })
        .collect::<Vec<_>>();
    let y = (0..n).map(|i| (i % 2) as f64).collect::<Vec<_>>();
    let w = dependence_weights(&x, n, 3, &y, 1, &[vec![0, 1]], 1e-6).unwrap();
    assert_eq!(w[0], w[1]);
    assert!(w[0] > w[2] && w.iter().all(|v| v.is_finite() && *v > 0.0));
    let (w, gain) = auto_weights(&x, n, 3, &y, 1, 10, &[vec![0, 1]]).unwrap();
    assert!(gain > 0.0 && w[0] == w[1]);
    let mut metric = AutoMetric::new(3, 1, vec![vec![0, 1]], 719).unwrap();
    assert!(metric.observe(&x, &y, n).unwrap().is_some());
    assert!(metric.snapshot().uses_learned);
    assert_eq!(metric.snapshot().counters.num_refits, 1);
    // The next scheduled refit is at ceil(1.5*128)=192 seen rows.
    metric.observe(&x[..3], &y[..1], 1).unwrap();
    assert_eq!(metric.snapshot().counters.num_refits, 1);
}

#[test]
fn gate_fallback() {
    let n = 128;
    let x = (0..n)
        .flat_map(|i| [i as f64 / 128.0, (i * 17 % 97) as f64 / 97.0])
        .collect::<Vec<_>>();
    let y = vec![1.0; n];
    let mut metric = AutoMetric::new(2, 1, vec![], 719).unwrap();
    metric.observe(&x[..2 * 99], &y[..99], 99).unwrap();
    assert_eq!(metric.snapshot().counters.num_refits, 0);
    metric.observe(&x[2 * 99..], &y[99..], n - 99).unwrap();
    assert_eq!(metric.snapshot().weights, [1.0, 1.0]);
    assert!(!metric.snapshot().uses_learned);
}
