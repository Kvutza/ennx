use super::*;

fn estimate(mean: f32) -> ObjectiveEstimate {
    ObjectiveEstimate {
        mean,
        variance: 0.0,
    }
}

fn vector(mean: f32) -> ObjectiveObservation {
    ObjectiveObservation::new(&[estimate(mean), estimate(-mean)], estimate(mean * 2.0)).unwrap()
}

#[test]
fn scalar_bits() {
    for (mean, variance) in [
        (-0.0, 0.0),
        (f32::from_bits(1), f32::from_bits(1)),
        (-17.5, 0.25),
    ] {
        let observation = ObjectiveObservation::scalar(mean, variance).unwrap();
        assert_eq!(observation.estimates().len(), 1);
        assert_eq!(observation.control().mean.to_bits(), mean.to_bits());
        assert_eq!(observation.control().variance.to_bits(), variance.to_bits());
        assert_eq!(observation.estimates()[0].mean.to_bits(), mean.to_bits());
        assert_eq!(
            observation.estimates()[0].variance.to_bits(),
            variance.to_bits()
        );
    }
}

#[test]
fn checked_variances() {
    assert!(ObjectiveObservation::new(&[], estimate(0.0)).is_err());
    assert!(
        ObjectiveObservation::new(&[estimate(0.0); MAX_OBJECTIVES + 1], estimate(0.0)).is_err()
    );
    for invalid in [
        ObjectiveEstimate {
            mean: f32::NAN,
            variance: 0.0,
        },
        ObjectiveEstimate {
            mean: f32::INFINITY,
            variance: 0.0,
        },
        ObjectiveEstimate {
            mean: 0.0,
            variance: -1.0,
        },
        ObjectiveEstimate {
            mean: 0.0,
            variance: f32::INFINITY,
        },
    ] {
        assert!(ObjectiveObservation::new(&[estimate(0.0), invalid], estimate(0.0)).is_err());
        assert!(ObjectiveObservation::new(&[estimate(0.0)], invalid).is_err());
    }
}

#[test]
fn explicit_control() {
    let observation = vector(3.0);
    assert_eq!(observation.estimates(), &[estimate(3.0), estimate(-3.0)]);
    assert_eq!(observation.control(), estimate(6.0));
}

#[test]
fn fifo_retention() {
    let mut window = ObjectiveWindow::new(3);
    for identity in 1..=6 {
        let observation = vector(identity as f32);
        window.validate(observation).unwrap();
        window.record(identity, observation, identity == 1);
    }
    assert_eq!(
        window.rows().map(|row| row.identity).collect::<Vec<_>>(),
        [4, 5, 6]
    );
    assert_eq!(window.incumbent().unwrap().identity, 1);
    window.retain_incumbent();
    assert_eq!(window.rows().len(), 1);
    assert_eq!(window.rows().next().unwrap().observation, vector(1.0));
    assert_eq!(window.rows().next().unwrap().identity, 1);
}

#[test]
fn window_capacity() {
    let mut window = ObjectiveWindow::new(OBJECTIVE_CAPACITY);
    for identity in 0..OBJECTIVE_CAPACITY as i64 {
        window.record(identity, vector(identity as f32), identity == 0);
    }
    assert_eq!(window.rows().len(), OBJECTIVE_CAPACITY);
    assert_eq!(window.rows().next().unwrap().identity, 0);
    assert_eq!(window.rows().last().unwrap().identity, 127);
    assert!(std::mem::size_of::<ObjectiveObservation>() <= 80);
    assert!(std::mem::size_of::<ObjectiveWindow>() <= 13_000);
}

#[test]
fn schema_retention() {
    let mut window = ObjectiveWindow::new(2);
    window.record(7, vector(1.0), true);
    assert!(
        window
            .validate(ObjectiveObservation::scalar(1.0, 0.0).unwrap())
            .is_err()
    );
    assert_eq!(window.rows().len(), 1);
    assert_eq!(window.rows().next().unwrap().identity, 7);
    assert_eq!(window.incumbent().unwrap().observation, vector(1.0));
}
