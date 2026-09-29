use super::*;
use crate::objective_observation::{ObjectiveEstimate, ObjectiveObservation};

fn state() -> SearchState {
    let base = [0x3e80; 17];
    let estimate = |mean| ObjectiveEstimate {
        mean,
        variance: 0.0,
    };
    SearchState::new_objectives(
        &base,
        ObjectiveObservation::new(&[estimate(1.0), estimate(-1.0)], estimate(0.0)).unwrap(),
        vec![ParamBlock::new(71, 0, base.len(), 0.25, 1.0).unwrap()],
        3,
        1,
        TRLengthConfig::new(0.1, 0.001, 0.4),
    )
    .unwrap()
}

fn config(acquisition: ObjectiveAcquisition) -> ObjectivePolicy {
    ObjectivePolicy {
        acquisition,
        scales: vec![1.0, 1.0],
    }
}

#[test]
fn parameter_layout() {
    assert_eq!(size_of::<ObjectiveParams>(), 12464);
    assert_eq!(size_of::<ObjectiveSelectionReport>(), 176);
    assert!(config(ObjectiveAcquisition::Pareto).validate().is_ok());
    assert!(
        ObjectivePolicy {
            acquisition: ObjectiveAcquisition::Pareto,
            scales: vec![1.0]
        }
        .validate()
        .is_err()
    );
    assert!(
        config(ObjectiveAcquisition::Morbo {
            weights: vec![1.0],
            regions: 4,
            alpha: 0.1,
            seed: 7,
            rescalarize: Rescalarize::OnPropose,
            clip: false
        })
        .validate()
        .is_err()
    );
}

fn nondominated(report: &ObjectiveSelectionReport, candidate: usize) -> bool {
    !(0..4).any(|other| {
        other != candidate
            && report.values[other][..2]
                .iter()
                .zip(&report.values[candidate][..2])
                .all(|(other, candidate)| other >= candidate)
            && report.values[other][..2]
                .iter()
                .zip(&report.values[candidate][..2])
                .any(|(other, candidate)| other > candidate)
    })
}

#[test]
fn pareto_control() {
    autoreleasepool(|| {
        let mut search = state();
        search.correlate(41).unwrap();
        search
            .configure_objectives(Some(config(ObjectiveAcquisition::Pareto)))
            .unwrap();
        assert!(search.objective_report().unwrap().is_none());
        let proposal = search.ask_round(1, 4, 29, Ask::default()).unwrap();
        let report = search.objective_report().unwrap().unwrap();
        assert_eq!(report.width, 2);
        assert_eq!(proposal.index, report.selected as usize);
        assert_ne!(report.nondominated_mask & (1 << proposal.index), 0);
        for candidate in 0..4 {
            assert!(
                report.values[candidate][..2]
                    .iter()
                    .all(|value| value.is_finite())
            );
            assert_eq!(
                report.nondominated_mask & (1 << candidate) != 0,
                nondominated(&report, candidate)
            );
        }
        assert_eq!(proposal.score, 0.0);
        assert_eq!(proposal.incumbent_mean, 0.0);
        assert_eq!(search.best().unwrap(), 0.0);
    });
}

#[test]
fn morbo_chebyshev() {
    autoreleasepool(|| {
        let mut search = state();
        search.correlate(41).unwrap();
        search
            .configure_objectives(Some(config(ObjectiveAcquisition::Morbo {
                weights: vec![1.0, 2.0],
                regions: 4,
                alpha: 0.1,
                seed: 31,
                rescalarize: Rescalarize::OnRestart,
                clip: true,
            })))
            .unwrap();
        let proposal = search.ask_round(1, 4, 29, Ask::default()).unwrap();
        assert_eq!(search.morbo_region(), Some(0));
        assert_eq!(search.morbo_regions(), 4);
        let report = search.objective_report().unwrap().unwrap();
        assert!((report.weights[..2].iter().sum::<f32>() - 1.0).abs() < 1e-6);
        // The initial window has constant ranges, matching MORBO's documented 0.5 convention.
        let expected = report.weights[..2]
            .iter()
            .map(|weight| 0.5 * weight)
            .fold(f32::INFINITY, f32::min)
            + 0.1
                * report.weights[..2]
                    .iter()
                    .map(|weight| 0.5 * weight)
                    .sum::<f32>();
        assert!((proposal.score - expected).abs() < 1e-6);
        assert_eq!(search.best().unwrap(), 0.0);
    });
}

#[test]
fn morbo_state() {
    autoreleasepool(|| {
        let mut search = state();
        search.correlate(41).unwrap();
        search
            .configure_objectives(Some(config(ObjectiveAcquisition::Morbo {
                weights: vec![1.0, 1.0],
                regions: 4,
                alpha: 0.1,
                seed: 31,
                rescalarize: Rescalarize::OnRestart,
                clip: true,
            })))
            .unwrap();
        let estimate = |mean| ObjectiveEstimate {
            mean,
            variance: 0.0,
        };
        let proposal = search.ask_round(1, 4, 29, Ask::default()).unwrap();
        let dominates =
            ObjectiveObservation::new(&[estimate(2.0), estimate(0.0)], estimate(-1_000.0)).unwrap();
        let decision = search.noisy_objectives(&proposal, dominates).unwrap();
        assert!(
            decision.accepted,
            "MORBO vector must override scalar control"
        );
        assert_eq!(decision.threshold, 0.0);
        assert_eq!(search.sync().unwrap(), [true]);
        assert_eq!(search.best().unwrap(), -1_000.0);

        let proposal = search.ask_round(1, 4, 37, Ask::default()).unwrap();
        assert_eq!(search.morbo_region(), Some(1));
        let dominated =
            ObjectiveObservation::new(&[estimate(0.0), estimate(-2.0)], estimate(1_000.0)).unwrap();
        let decision = search.noisy_objectives(&proposal, dominated).unwrap();
        assert!(!decision.accepted, "scalar control must not override MORBO");
        assert_eq!(search.sync().unwrap(), [false]);
        assert_eq!(search.best().unwrap(), 0.0);
        assert!(search.controller_info().unwrap().failure_counter > 0);
        assert_eq!(search.finalize_morbo().unwrap(), Some(0));
        assert_eq!(search.best().unwrap(), -1_000.0);
    });
}

#[test]
fn morbo_noise() {
    autoreleasepool(|| {
        let mut search = state();
        search.correlate(41).unwrap();
        search
            .configure_objectives(Some(config(ObjectiveAcquisition::Morbo {
                weights: vec![1.0, 1.0],
                regions: 2,
                alpha: 0.1,
                seed: 31,
                rescalarize: Rescalarize::OnRestart,
                clip: true,
            })))
            .unwrap();
        let proposal = search.ask_round(1, 4, 29, Ask::default()).unwrap();
        let noisy = |mean| ObjectiveEstimate {
            mean,
            variance: 100.0,
        };
        let observation = ObjectiveObservation::new(&[noisy(2.0), noisy(0.0)], noisy(1.0)).unwrap();
        let decision = search.noisy_objectives(&proposal, observation).unwrap();
        assert!(!decision.accepted);
        assert_eq!(search.sync().unwrap(), [false]);
    });
}

#[test]
fn morbo_resample() {
    autoreleasepool(|| {
        let mut search = state();
        let error = search
            .configure_objectives(Some(config(ObjectiveAcquisition::Morbo {
                weights: vec![1.0, 1.0],
                regions: 4,
                alpha: 0.1,
                seed: 31,
                rescalarize: Rescalarize::OnPropose,
                clip: true,
            })))
            .unwrap_err();
        assert!(error.contains("rematerializing"));
    });
}

#[test]
fn scalar_optout() {
    autoreleasepool(|| {
        let mut baseline = state();
        let mut toggled = state();
        baseline.correlate(41).unwrap();
        toggled.correlate(41).unwrap();
        toggled
            .configure_objectives(Some(config(ObjectiveAcquisition::Pareto)))
            .unwrap();
        toggled.configure_objectives(None).unwrap();
        let first = baseline.ask_round(1, 4, 29, Ask::default()).unwrap();
        let second = toggled.ask_round(1, 4, 29, Ask::default()).unwrap();
        assert_eq!(first.index, second.index);
        assert_eq!(first.seed, second.seed);
        assert_eq!(first.score.to_bits(), second.score.to_bits());
        assert_eq!(
            first.predicted_mean.to_bits(),
            second.predicted_mean.to_bits()
        );
        assert_eq!(
            first.predicted_standard_error.to_bits(),
            second.predicted_standard_error.to_bits()
        );
        assert_eq!(
            read::<u16>(&baseline.proposal, 17),
            read::<u16>(&toggled.proposal, 17)
        );
    });
}

#[test]
fn shared_geometry() {
    autoreleasepool(|| {
        let mut scalar = state();
        let mut vector = state();
        scalar.correlate(41).unwrap();
        vector.correlate(41).unwrap();
        prime_geometry(&mut scalar);
        prime_geometry(&mut vector);
        vector
            .configure_objectives(Some(config(ObjectiveAcquisition::Pareto)))
            .unwrap();
        scalar.begin_forced(29, Ask::default(), 2).unwrap();
        vector.begin_forced(29, Ask::default(), 2).unwrap();
        let first = scalar.finish_ask().unwrap();
        let second = vector.finish_ask().unwrap();
        assert_eq!(
            first.predicted_mean.to_bits(),
            second.predicted_mean.to_bits()
        );
        assert_eq!(
            first.predicted_standard_error.to_bits(),
            second.predicted_standard_error.to_bits()
        );
        assert_eq!(
            first.incumbent_mean.to_bits(),
            second.incumbent_mean.to_bits()
        );
        assert_eq!(
            first.incumbent_standard_error.to_bits(),
            second.incumbent_standard_error.to_bits()
        );
    });
}

fn prime_geometry(search: &mut SearchState) {
    for root in [31, 37] {
        let proposal = search.ask_round(1, 4, root, Ask::default()).unwrap();
        let estimate = |mean| ObjectiveEstimate {
            mean,
            variance: 0.01,
        };
        let observation =
            ObjectiveObservation::new(&[estimate(2.0), estimate(-2.0)], estimate(-1.0)).unwrap();
        search.noisy_objectives(&proposal, observation).unwrap();
        search.sync().unwrap();
    }
    search.distance_scaling = crate::config::DistanceScaling::SelfTuning;
    search.local_scale_neighbors = 2;
}
