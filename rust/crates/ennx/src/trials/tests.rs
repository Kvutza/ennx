use super::*;
use ndarray::{array, Axis};
use tempfile::TempDir;

fn leaves() -> Vec<Leaf> {
    vec![
        Leaf::new(0, 5, 4, 0.25, 1.0, 0.75).unwrap(),
        Leaf::new(5, 4, 8, 0.5, 0.5, 1.0).unwrap(),
    ]
}

#[test]
fn cpu_search() {
    let base = [0x76, 0x98, 0x0a, 100, 120, 140, 160];
    let mut left = Search::new(&base, 1.0, leaves(), 4, ComputeBackend::Cpu).unwrap();
    let mut right = Search::new(&base, 1.0, leaves(), 4, ComputeBackend::Cpu).unwrap();
    let config = Ask {
        neighbors: 1,
        length: 1.0,
        ..Ask::default()
    };
    let a = left.ask(&[7, 11, 13], config).unwrap();
    let b = right.ask(&[7, 11, 13], config).unwrap();
    assert_eq!(a, b);
    let row = left.row(a).unwrap();
    assert_eq!(row, right.row(b).unwrap());
    assert_ne!(&row[..3], &base[..3]);
    assert_ne!(&row[3..], &base[3..]);
}

#[test]
fn accepted_center() {
    let base = [0x76, 0x98, 0x0a, 100, 120, 140, 160];
    let mut search = Search::new(&base, 0.0, leaves(), 2, ComputeBackend::Cpu).unwrap();
    let config = Ask {
        neighbors: 1,
        length: 1.0,
        ..Ask::default()
    };
    let first = search.ask(&[5], config).unwrap();
    let first_row = search.row(first).unwrap();
    search.tell(first, 1.0, true).unwrap();
    let second = search.ask(&[9], config).unwrap();
    let second_row = search.row(second).unwrap();
    assert_ne!(first_row, second_row);
    assert_eq!(search.history_len(), 2);
}

fn lazy_match(backend: ComputeBackend) {
    let base = [0x76, 0x98, 0x0a, 100, 120, 140, 160];
    let mut eager = Search::new(&base, 0.0, leaves(), 3, backend).unwrap();
    let mut lazy = Search::new(&base, 0.0, leaves(), 3, backend).unwrap();
    let config = Ask {
        neighbors: 1,
        length: 0.65,
        ..Ask::default()
    };
    let eager_trial = eager.ask(&[5, 7, 11], config).unwrap();
    let lazy_trial = lazy.ask_lazy(&[5, 7, 11], config).unwrap();
    assert_eq!(lazy_trial, eager_trial);
    assert!(lazy.row(lazy_trial).is_err());
    eager.tell(eager_trial, 1.0, true).unwrap();
    lazy.tell(lazy_trial, 1.0, true).unwrap();
    let eager_next = eager.ask(&[13, 17], config).unwrap();
    let lazy_next = lazy.ask(&[13, 17], config).unwrap();
    assert_eq!(lazy_next, eager_next);
    assert_eq!(lazy.row(lazy_next).unwrap(), eager.row(eager_next).unwrap());
}

#[test]
fn lazy_history() {
    lazy_match(ComputeBackend::Cpu);
}

#[test]
fn sparse_rows() {
    let base = [0x76, 0x98, 0x0a, 100, 120, 140, 160];
    let leaves = leaves();
    let edits = sparse::make_edits(&[7], &leaves, 1).unwrap();
    let history = [(0, 1.0)];
    let config = Ask {
        neighbors: 1,
        length: 1.0,
        ..Ask::default()
    };
    let (index, _) =
        sparse::sparse_select(&base, &[&base], &history, &[7], &edits, 1, &leaves, config);
    assert_eq!(index, 0);
    assert_ne!(
        sparse::sparse_materialize(&base, 7, &edits, &leaves, 1.0),
        base
    );
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn metal_history() {
    lazy_match(ComputeBackend::Metal);
    lazy_match(ComputeBackend::Agx);
}

#[cfg(feature = "opencl")]
#[test]
fn opencl_history() {
    let base = [0x76, 0x98, 0x0a, 100, 120, 140, 160];
    match Search::new(&base, 0.0, leaves(), 3, ComputeBackend::OpenCl) {
        Ok(_) => lazy_match(ComputeBackend::OpenCl),
        Err(error) if error.contains("no OpenCL GPU or CPU device") => {}
        Err(error) => panic!("{error}"),
    }
}

#[test]
fn rejected_center() {
    let base = [0x76, 0x98, 0x0a, 100, 120, 140, 160];
    let mut search = Search::new(&base, 0.0, leaves(), 2, ComputeBackend::Cpu).unwrap();
    let mut control = Search::new(&base, 0.0, leaves(), 2, ComputeBackend::Cpu).unwrap();
    let config = Ask {
        neighbors: 1,
        length: 1.0,
        ..Ask::default()
    };
    let rejected = search.ask(&[5], config).unwrap();
    search.tell(rejected, -1.0, false).unwrap();
    let next = search.ask(&[5], config).unwrap();
    let expected = control.ask(&[5], config).unwrap();
    assert_eq!(search.row(next).unwrap(), control.row(expected).unwrap());
}

#[test]
fn history_score() {
    let base = [0x76, 0x98, 0x0a, 100, 120, 140, 160];
    let mut search = Search::new(&base, 0.0, leaves(), 3, ComputeBackend::Cpu).unwrap();
    let rows = [
        0x11, 0x22, 0x03, 10, 20, 30, 40, 0x44, 0x55, 0x06, 70, 80, 90, 100,
    ];
    search.replace_history(&rows, &[3.0, 7.0]).unwrap();
    assert_eq!(search.history_len(), 2);
    assert_eq!(search.history_capacity(), 3);
    let trial = search
        .ask(
            &[17, 23],
            Ask {
                neighbors: 1,
                length: 1.0,
                ..Ask::default()
            },
        )
        .unwrap();
    assert_eq!(search.row(trial).unwrap().len(), base.len());
    search.tell(trial, 9.0, false).unwrap();
    let next = search
        .ask(
            &[17],
            Ask {
                neighbors: 1,
                length: 1.0,
                ..Ask::default()
            },
        )
        .unwrap();
    let mut control = Search::new(&base, 0.0, leaves(), 3, ComputeBackend::Cpu).unwrap();
    control.replace_history(&rows, &[3.0, 7.0]).unwrap();
    let expected = control
        .ask(
            &[17],
            Ask {
                neighbors: 1,
                length: 1.0,
                ..Ask::default()
            },
        )
        .unwrap();
    assert_eq!(search.row(next).unwrap(), control.row(expected).unwrap());
}

#[test]
fn history_state() {
    let base = [0x76, 0x98, 0x0a, 100, 120, 140, 160];
    let mut search = Search::new(&base, 0.0, leaves(), 2, ComputeBackend::Cpu).unwrap();
    assert!(search.replace_history(&[], &[]).is_err());
    assert!(search.replace_history(&base, &[1.0, 2.0]).is_err());
    let trial = search
        .ask(
            &[7],
            Ask {
                neighbors: 1,
                ..Ask::default()
            },
        )
        .unwrap();
    assert!(search.replace_history(&base, &[1.0]).is_err());
    search.tell(trial, 1.0, false).unwrap();
}

#[test]
fn indexed_row() {
    let base = [0x76, 0x98, 0x0a, 100, 120, 140, 160];
    let rows = [
        [0x11, 0x22, 0x03, 10, 20, 30, 40],
        [0x44, 0x55, 0x06, 70, 80, 90, 100],
    ];
    let observations = [
        IndexedObservation {
            id: ObservationId(1),
            value: 3.0,
        },
        IndexedObservation {
            id: ObservationId(0),
            value: 7.0,
        },
    ];
    let mut resolved = Vec::new();
    let mut search = Search::new(&base, 0.0, leaves(), 2, ComputeBackend::Cpu).unwrap();
    search
        .replace_indexed_history(&observations, |id| {
            resolved.push(id);
            Ok(rows[id.0 as usize].to_vec())
        })
        .unwrap();
    assert_eq!(resolved, vec![ObservationId(1), ObservationId(0)]);
    assert_eq!(search.history_len(), 2);
    assert!(search
        .ask(
            &[31],
            Ask {
                neighbors: 2,
                ..Ask::default()
            }
        )
        .is_ok());
}

#[test]
fn indexed_search() {
    let base = [0x76, 0x98, 0x0a, 100, 120, 140, 160];
    let archive = [
        [0x11, 0x22, 0x03, 10, 20, 30, 40],
        [0x44, 0x55, 0x06, 70, 80, 90, 100],
        [0x77, 0x88, 0x09, 110, 120, 130, 140],
    ];
    let descriptors = array![[0.0, 0.0], [1.0, 0.0], [4.0, 0.0]];
    let dir = TempDir::new().unwrap();
    let mut history = BpannHistory::new(dir.path().to_path_buf(), 2).unwrap();
    for (index, descriptor) in descriptors.axis_iter(Axis(0)).enumerate() {
        history
            .append(&descriptor, (index as f32 + 1.0) * 10.0)
            .unwrap();
    }
    let candidate_descriptors = array![[0.1, 0.0], [3.9, 0.0]];
    let mut resolved = Vec::new();
    let mut search = Search::new(&base, 0.0, leaves(), 3, ComputeBackend::Cpu).unwrap();
    let trial = search
        .ask_indexed(
            &history,
            &candidate_descriptors.view(),
            1,
            &[17, 23],
            Ask {
                neighbors: 1,
                length: 1.0,
                ..Ask::default()
            },
            |id| {
                resolved.push(id);
                Ok(archive[id.0 as usize].to_vec())
            },
        )
        .unwrap();
    assert_eq!(resolved, vec![ObservationId(0), ObservationId(2)]);
    assert_eq!(search.history_len(), 2);
    assert!(trial.index < 2);
    assert_eq!(search.row(trial).unwrap().len(), base.len());
}
