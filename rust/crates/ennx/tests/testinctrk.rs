use ennx::incumbent_tracker::{enn_k, tracker_surrogate, IncumbentTracker};
use ndarray::array;

#[test]
fn incumbent_paths() {
    let names: &[&str] = &[
        "enn_k",
        "tracker_surrogate",
        "IncumbentTracker",
        "reset_tracker",
        "sync_incumbent_tracker_from_obs",
        "push_m",
        "sorted_indices",
    ];
    assert!(!names.is_empty());
    assert_eq!(enn_k(3), 3);
    let _ = tracker_surrogate();

    let mut noiseless = IncumbentTracker::new(5, false, 1);
    noiseless.tell(0, &array![1.0]);
    noiseless.tell(1, &array![3.0]);
    let _ = noiseless.ask();
    noiseless.reset();

    let mut noisy = IncumbentTracker::new(2, true, 1);
    noisy.tell(0, &array![1.0]);
    noisy.tell(1, &array![2.0]);
    noisy.tell(2, &array![3.0]);
    let _ = noisy.ask();
    noisy.rebuild(&array![[4.0], [1.0]].view());

    let mut multi = IncumbentTracker::new(2, false, 2);
    multi.tell(0, &array![1.0, 0.0]);
    multi.tell(1, &array![0.0, 2.0]);
    let _ = multi.ask();

    let mut all = IncumbentTracker::new(tracker_surrogate(), false, 1);
    all.tell(0, &array![0.1]);
    let _ = all.ask();
}
