use super::*;
use ndarray::{array, Axis};
#[cfg(feature = "opencl")]
use opencl3::memory::ClMem;
use tempfile::TempDir;

fn same_candidate(left: Trial, right: Trial) {
    assert_ne!(left, right);
    assert_eq!(
        (left.index, left.seed, left.score),
        (right.index, right.seed, right.score)
    );
}

fn leaves() -> Vec<Parameter> {
    vec![
        Parameter::new(0, 5, 4, 0.25, 1.0, 0.75).unwrap(),
        Parameter::new(5, 4, 8, 0.5, 0.5, 1.0).unwrap(),
    ]
}

fn sparse_history(base: &[u8]) -> (Vec<u8>, Vec<f32>) {
    let mut second = base.to_vec();
    second[0] ^= 0x01;
    let mut rows = Vec::with_capacity(base.len() * 2);
    rows.extend_from_slice(base);
    rows.extend_from_slice(&second);
    (rows, vec![1.0, 2.0])
}

#[test]
fn cpu_search() {
    let base = [0x76, 0x98, 0x0a, 100, 120, 140, 160];
    let mut left = Search::new(&base, 1.0, leaves(), 4, ComputeDevice::Cpu).unwrap();
    let mut right = Search::new(&base, 1.0, leaves(), 4, ComputeDevice::Cpu).unwrap();
    assert_eq!(left.device(), ComputeDevice::Cpu);
    let config = Ask {
        neighbors: 1,
        length: 1.0,
        ..Ask::default()
    };
    let a = left.ask(&[7, 11, 13], config).unwrap();
    let b = right.ask(&[7, 11, 13], config).unwrap();
    same_candidate(a, b);
    let row = left.row(a).unwrap();
    assert_eq!(row, right.row(b).unwrap());
    assert_ne!(&row[..3], &base[..3]);
    assert_ne!(&row[3..], &base[3..]);
}

#[test]
fn accepted_center() {
    let base = [0x76, 0x98, 0x0a, 100, 120, 140, 160];
    let mut search = Search::new(&base, 0.0, leaves(), 2, ComputeDevice::Cpu).unwrap();
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

fn lazy_match(device: ComputeDevice) {
    let base = [0x76, 0x98, 0x0a, 100, 120, 140, 160];
    let mut eager = Search::new(&base, 0.0, leaves(), 3, device).unwrap();
    let mut lazy = Search::new(&base, 0.0, leaves(), 3, device).unwrap();
    let config = Ask {
        neighbors: 1,
        length: 0.65,
        ..Ask::default()
    };
    let eager_trial = eager.ask(&[5, 7, 11], config).unwrap();
    let lazy_trial = lazy.ask_lazy(&[5, 7, 11], config).unwrap();
    same_candidate(lazy_trial, eager_trial);
    assert!(lazy.row(lazy_trial).is_err());
    eager.tell(eager_trial, 1.0, true).unwrap();
    lazy.tell(lazy_trial, 1.0, true).unwrap();
    let eager_next = eager.ask(&[13, 17], config).unwrap();
    let lazy_next = lazy.ask(&[13, 17], config).unwrap();
    same_candidate(lazy_next, eager_next);
    assert_eq!(lazy.row(lazy_next).unwrap(), eager.row(eager_next).unwrap());
}

fn stream_match(device: ComputeDevice, acquisition: crate::weights::AcquisitionKind) {
    let base = [0x76, 0x98, 0x0a, 100, 120, 140, 160];
    let mut explicit = Search::new(&base, 0.0, leaves(), 3, device).unwrap();
    let mut streamed = Search::new(&base, 0.0, leaves(), 3, device).unwrap();
    let config = Ask {
        acquisition,
        neighbors: 1,
        length: 0.65,
        seed: 0xfeed_beef,
        ..Ask::default()
    };
    let seeds = cpu::seed_stream(0x1234_5678_9abc_def0, 5);
    let left = explicit.ask(&seeds, config).unwrap();
    let right = streamed
        .ask_stream(0x1234_5678_9abc_def0, seeds.len(), config)
        .unwrap();
    assert_eq!(left.index, right.index);
    assert_eq!(left.seed, right.seed);
    assert_eq!(left.score, right.score);
    assert_eq!(explicit.row(left).unwrap(), streamed.row(right).unwrap());
}

fn stream_parity(device: ComputeDevice, acquisition: crate::weights::AcquisitionKind) {
    let base = [0x76, 0x98, 0x0a, 100, 120, 140, 160];
    let mut cpu = Search::new(&base, 0.0, leaves(), 3, ComputeDevice::Cpu).unwrap();
    let mut gpu = Search::new(&base, 0.0, leaves(), 3, device).unwrap();
    let config = Ask {
        acquisition,
        neighbors: 1,
        length: 0.65,
        seed: 0xfeed_beef,
        ..Ask::default()
    };
    let left = cpu.ask_stream(0x1234_5678_9abc_def0, 7, config).unwrap();
    let right = gpu.ask_stream(0x1234_5678_9abc_def0, 7, config).unwrap();
    assert_eq!(left.index, right.index);
    assert_eq!(left.seed, right.seed);
    assert_eq!(left.score.to_bits(), right.score.to_bits());
    assert_eq!(cpu.row(left).unwrap(), gpu.row(right).unwrap());
}

fn stream_all(device: ComputeDevice) {
    for acquisition in [
        crate::weights::AcquisitionKind::Ucb,
        crate::weights::AcquisitionKind::Thompson,
        crate::weights::AcquisitionKind::Pareto,
    ] {
        stream_match(device, acquisition);
        if device != ComputeDevice::Cpu {
            stream_parity(device, acquisition);
        }
    }
}

fn sparse_stream(device: ComputeDevice) {
    let base = [0x76, 0x98, 0x0a, 100, 120, 140, 160];
    let (rows, values) = sparse_history(&base);
    let config = Ask {
        neighbors: 1,
        length: 1.0,
        ..Ask::default()
    };
    let seeds = cpu::seed_stream(0x1234_5678_9abc_def0, 5);
    let mut explicit = Search::new(&base, 0.0, leaves(), 4, device).unwrap();
    explicit.replace_history(&rows, &values).unwrap();
    let left = explicit.ask_sparse(&seeds, 2, config).unwrap();
    let mut streamed = Search::new(&base, 0.0, leaves(), 4, device).unwrap();
    streamed.replace_history(&rows, &values).unwrap();
    let right = streamed
        .sparse_stream(0x1234_5678_9abc_def0, seeds.len(), 2, config)
        .unwrap();
    assert_eq!(left.index, right.index);
    assert_eq!(left.seed, right.seed);
    assert!((left.score - right.score).abs() < 1e-5);
    assert_eq!(explicit.row(left).unwrap(), streamed.row(right).unwrap());
}

fn regions_stream(device: ComputeDevice) {
    let base = [0x76, 0x98, 0x0a, 100, 120, 140, 160];
    let (rows, values) = sparse_history(&base);
    let config = Ask {
        neighbors: 1,
        length: 0.65,
        ..Ask::default()
    };
    let seeds = cpu::seed_stream(0x1234_5678_9abc_def0, 6);
    let mut explicit = Search::new(&base, 0.0, leaves(), 4, device).unwrap();
    explicit.replace_history(&rows, &values).unwrap();
    let left = explicit.ask_regions(2, 3, &seeds, config).unwrap();
    let mut streamed = Search::new(&base, 0.0, leaves(), 4, device).unwrap();
    streamed.replace_history(&rows, &values).unwrap();
    let right = streamed
        .regions_stream(2, 3, 0x1234_5678_9abc_def0, config)
        .unwrap();
    assert_eq!(left, right);
}

fn centers_stream(device: ComputeDevice) {
    let base = [0x76, 0x98, 0x0a, 100, 120, 140, 160];
    let (rows, values) = sparse_history(&base);
    let config = Ask {
        neighbors: 1,
        length: 0.65,
        ..Ask::default()
    };
    let centers = [
        Center {
            parent: None,
            seed: 3,
        },
        Center {
            parent: Some(0),
            seed: 5,
        },
    ];
    let region_centers = [0, 1];
    let seeds = cpu::seed_stream(0x1234_5678_9abc_def0, 6);
    let mut explicit = Search::new(&base, 0.0, leaves(), 4, device).unwrap();
    explicit.replace_history(&rows, &values).unwrap();
    let left = explicit
        .ask_centers(2, 3, &centers, &region_centers, &seeds, config)
        .unwrap();
    let mut streamed = Search::new(&base, 0.0, leaves(), 4, device).unwrap();
    streamed.replace_history(&rows, &values).unwrap();
    let right = streamed
        .centers_stream(
            2,
            3,
            &centers,
            &region_centers,
            0x1234_5678_9abc_def0,
            config,
        )
        .unwrap();
    assert_eq!(left, right);
}

fn batch_stream(device: ComputeDevice) {
    let base = [0x76, 0x98, 0x0a, 100, 120, 140, 160];
    let (rows, values) = sparse_history(&base);
    let config = Ask {
        neighbors: 1,
        length: 1.0,
        ..Ask::default()
    };
    let mut search = Search::new_batch(&base, 0.0, leaves(), 4, 2, device).unwrap();
    search.replace_history(&rows, &values).unwrap();
    let trials = search
        .batch_stream(0x1234_5678_9abc_def0, 2, 3, 2, config)
        .unwrap();
    assert_eq!(trials.len(), 2);
    assert_ne!(trials[0].id, trials[1].id);
    assert_ne!(trials[0].seed, trials[1].seed);
    assert!(trials.iter().all(|trial| trial.index < 3));
}

#[test]
fn cpu_stream() {
    stream_all(ComputeDevice::Cpu);
}

#[test]
fn cpu_thompson() {
    stream_match(
        ComputeDevice::Cpu,
        crate::weights::AcquisitionKind::Thompson,
    );
}

#[test]
fn cpu_sparse2() {
    sparse_stream(ComputeDevice::Cpu);
}

#[test]
fn cpu_regions() {
    regions_stream(ComputeDevice::Cpu);
}

#[test]
fn cpu_centers() {
    centers_stream(ComputeDevice::Cpu);
}

#[test]
fn cpu_batch2() {
    batch_stream(ComputeDevice::Cpu);
}

#[test]
fn lazy_history() {
    lazy_match(ComputeDevice::Cpu);
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

#[test]
fn cpu_sparse() {
    let base = [0x76, 0x98, 0x0a, 100, 120, 140, 160];
    let (rows, values) = sparse_history(&base);
    let mut search = Search::new(&base, 0.0, leaves(), 4, ComputeDevice::Cpu).unwrap();
    search.replace_history(&rows, &values).unwrap();
    let trial = search
        .ask_sparse(
            &[7, 11, 13],
            2,
            Ask {
                neighbors: 1,
                length: 1.0,
                ..Ask::default()
            },
        )
        .unwrap();
    assert!(trial.index < 3);
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn metal_unavailable(error: &str) -> bool {
    error.contains("no default Metal device found")
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn metal_sparse() {
    let base = [0x76, 0x98, 0x0a, 100, 120, 140, 160];
    let (rows, values) = sparse_history(&base);
    let config = Ask {
        neighbors: 1,
        length: 1.0,
        ..Ask::default()
    };
    let mut cpu = Search::new(&base, 0.0, leaves(), 4, ComputeDevice::Cpu).unwrap();
    cpu.replace_history(&rows, &values).unwrap();
    let cpu_trial = cpu.ask_sparse(&[7, 11, 13], 2, config).unwrap();
    let mut search = match Search::new(&base, 0.0, leaves(), 4, ComputeDevice::Metal) {
        Ok(search) => search,
        Err(error) if metal_unavailable(&error) => return,
        Err(error) => panic!("{error}"),
    };
    search.replace_history(&rows, &values).unwrap();
    let trial = search.ask_sparse(&[7, 11, 13], 2, config).unwrap();
    assert_eq!(trial.index, cpu_trial.index);
    assert_eq!(trial.seed, cpu_trial.seed);
    assert!((trial.score - cpu_trial.score).abs() < 1e-5);
    assert_eq!(search.row(trial).unwrap(), cpu.row(cpu_trial).unwrap());
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn metal_resident() {
    let base = [0x76, 0x98, 0x0a, 100, 120, 140, 160];
    let mut search = match Search::new(&base, 0.0, leaves(), 4, ComputeDevice::Metal) {
        Ok(search) => search,
        Err(error) if metal_unavailable(&error) => return,
        Err(error) => panic!("{error}"),
    };
    let trial = search
        .ask_sparse(
            &[7, 11, 13],
            2,
            Ask {
                neighbors: 1,
                length: 1.0,
                ..Ask::default()
            },
        )
        .unwrap();
    let row = search.row(trial).unwrap();
    let resident = search.device_view(trial).unwrap();
    if let Some((buffer, offset)) = resident.as_metal() {
        assert_eq!(resident.row_bytes(), row.len());
        assert!(offset + resident.row_bytes() <= buffer.length() as usize);
        let batch = device_views(&search, &[trial]).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(batch[0].row_bytes(), row.len());
        let expected = row.iter().fold(0u64, |sum, &byte| sum + u64::from(byte));
        assert_eq!(search.byte_sum(trial).unwrap(), expected);
    } else {
        panic!("expected Metal resident row");
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn metal_stream() {
    let base = [0x76, 0x98, 0x0a, 100, 120, 140, 160];
    match Search::new(&base, 0.0, leaves(), 3, ComputeDevice::Metal) {
        Ok(_) => stream_all(ComputeDevice::Metal),
        Err(error) if metal_unavailable(&error) => {}
        Err(error) => panic!("{error}"),
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn metal_thompson() {
    let base = [0x76, 0x98, 0x0a, 100, 120, 140, 160];
    match Search::new(&base, 0.0, leaves(), 3, ComputeDevice::Metal) {
        Ok(_) => stream_parity(
            ComputeDevice::Metal,
            crate::weights::AcquisitionKind::Thompson,
        ),
        Err(error) if metal_unavailable(&error) => {}
        Err(error) => panic!("{error}"),
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn metal_sparse2() {
    let base = [0x76, 0x98, 0x0a, 100, 120, 140, 160];
    match Search::new(&base, 0.0, leaves(), 4, ComputeDevice::Metal) {
        Ok(_) => sparse_stream(ComputeDevice::Metal),
        Err(error) if metal_unavailable(&error) => {}
        Err(error) => panic!("{error}"),
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn metal_regions2() {
    let base = [0x76, 0x98, 0x0a, 100, 120, 140, 160];
    match Search::new(&base, 0.0, leaves(), 4, ComputeDevice::Metal) {
        Ok(_) => regions_stream(ComputeDevice::Metal),
        Err(error) if metal_unavailable(&error) => {}
        Err(error) => panic!("{error}"),
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn metal_centers2() {
    let base = [0x76, 0x98, 0x0a, 100, 120, 140, 160];
    match Search::new(&base, 0.0, leaves(), 4, ComputeDevice::Metal) {
        Ok(_) => centers_stream(ComputeDevice::Metal),
        Err(error) if metal_unavailable(&error) => {}
        Err(error) => panic!("{error}"),
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn metal_batch2() {
    let base = [0x76, 0x98, 0x0a, 100, 120, 140, 160];
    match Search::new_batch(&base, 0.0, leaves(), 4, 2, ComputeDevice::Metal) {
        Ok(_) => batch_stream(ComputeDevice::Metal),
        Err(error) if metal_unavailable(&error) => {}
        Err(error) => panic!("{error}"),
    }
}

#[cfg(feature = "opencl")]
#[test]
fn opencl_sparse() {
    let base = [0x76, 0x98, 0x0a, 100, 120, 140, 160];
    let (rows, values) = sparse_history(&base);
    let config = Ask {
        neighbors: 1,
        length: 1.0,
        ..Ask::default()
    };
    let mut cpu = Search::new(&base, 0.0, leaves(), 4, ComputeDevice::Cpu).unwrap();
    cpu.replace_history(&rows, &values).unwrap();
    let cpu_trial = cpu.ask_sparse(&[7, 11, 13], 2, config).unwrap();
    let mut search = match Search::new(&base, 0.0, leaves(), 4, ComputeDevice::OpenCl) {
        Ok(search) => search,
        Err(error) if opencl_unavailable(&error) => return,
        Err(error) => panic!("{error}"),
    };
    search.replace_history(&rows, &values).unwrap();
    let trial = search.ask_sparse(&[7, 11, 13], 2, config).unwrap();
    assert_eq!(trial.index, cpu_trial.index);
    assert_eq!(trial.seed, cpu_trial.seed);
    assert!((trial.score - cpu_trial.score).abs() < 1e-5);
    assert_eq!(search.row(trial).unwrap(), cpu.row(cpu_trial).unwrap());
}

#[cfg(feature = "opencl")]
#[test]
fn opencl_resident() {
    let base = [0x76, 0x98, 0x0a, 100, 120, 140, 160];
    let mut search = match Search::new(&base, 0.0, leaves(), 4, ComputeDevice::OpenCl) {
        Ok(search) => search,
        Err(error) if opencl_unavailable(&error) => return,
        Err(error) => panic!("{error}"),
    };
    let trial = search
        .ask_sparse(
            &[7, 11, 13],
            2,
            Ask {
                neighbors: 1,
                length: 1.0,
                ..Ask::default()
            },
        )
        .unwrap();
    let row = search.row(trial).unwrap();
    let resident = search.device_view(trial).unwrap();
    if let Some((_, _, buffer, offset)) = resident.as_opencl() {
        let allocation_len = buffer.size().unwrap() as usize;
        assert_eq!(resident.row_bytes(), row.len());
        assert!(offset + resident.row_bytes() <= allocation_len);
        let batch = device_views(&search, &[trial]).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(batch[0].row_bytes(), row.len());
        let expected = row.iter().fold(0u64, |sum, &byte| sum + u64::from(byte));
        assert_eq!(search.byte_sum(trial).unwrap(), expected);
    } else {
        panic!("expected OpenCL resident row");
    }
}

#[cfg(feature = "opencl")]
#[test]
fn opencl_stream() {
    let base = [0x76, 0x98, 0x0a, 100, 120, 140, 160];
    match Search::new(&base, 0.0, leaves(), 3, ComputeDevice::OpenCl) {
        Ok(_) => stream_all(ComputeDevice::OpenCl),
        Err(error) if opencl_unavailable(&error) => {}
        Err(error) => panic!("{error}"),
    }
}

#[cfg(feature = "opencl")]
#[test]
fn opencl_thompson() {
    let base = [0x76, 0x98, 0x0a, 100, 120, 140, 160];
    match Search::new(&base, 0.0, leaves(), 3, ComputeDevice::OpenCl) {
        Ok(_) => stream_parity(
            ComputeDevice::OpenCl,
            crate::weights::AcquisitionKind::Thompson,
        ),
        Err(error) if opencl_unavailable(&error) => {}
        Err(error) => panic!("{error}"),
    }
}

#[cfg(feature = "opencl")]
#[test]
fn opencl_sparse2() {
    let base = [0x76, 0x98, 0x0a, 100, 120, 140, 160];
    match Search::new(&base, 0.0, leaves(), 4, ComputeDevice::OpenCl) {
        Ok(_) => sparse_stream(ComputeDevice::OpenCl),
        Err(error) if opencl_unavailable(&error) => {}
        Err(error) => panic!("{error}"),
    }
}

#[cfg(feature = "opencl")]
#[test]
fn opencl_regions2() {
    let base = [0x76, 0x98, 0x0a, 100, 120, 140, 160];
    match Search::new(&base, 0.0, leaves(), 4, ComputeDevice::OpenCl) {
        Ok(_) => regions_stream(ComputeDevice::OpenCl),
        Err(error) if opencl_unavailable(&error) => {}
        Err(error) => panic!("{error}"),
    }
}

#[cfg(feature = "opencl")]
#[test]
fn opencl_centers2() {
    let base = [0x76, 0x98, 0x0a, 100, 120, 140, 160];
    match Search::new(&base, 0.0, leaves(), 4, ComputeDevice::OpenCl) {
        Ok(_) => centers_stream(ComputeDevice::OpenCl),
        Err(error) if opencl_unavailable(&error) => {}
        Err(error) => panic!("{error}"),
    }
}

#[cfg(feature = "opencl")]
#[test]
fn opencl_batch2() {
    let base = [0x76, 0x98, 0x0a, 100, 120, 140, 160];
    match Search::new_batch(&base, 0.0, leaves(), 4, 2, ComputeDevice::OpenCl) {
        Ok(_) => batch_stream(ComputeDevice::OpenCl),
        Err(error) if opencl_unavailable(&error) => {}
        Err(error) => panic!("{error}"),
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn metal_history() {
    let base = [0x76, 0x98, 0x0a, 100, 120, 140, 160];
    let mut eager = match Search::new(&base, 0.0, leaves(), 3, ComputeDevice::Metal) {
        Ok(search) => search,
        Err(error) if metal_unavailable(&error) => return,
        Err(error) => panic!("{error}"),
    };
    let mut lazy = match Search::new(&base, 0.0, leaves(), 3, ComputeDevice::Metal) {
        Ok(search) => search,
        Err(error) if metal_unavailable(&error) => return,
        Err(error) => panic!("{error}"),
    };
    let config = Ask {
        neighbors: 1,
        length: 0.65,
        ..Ask::default()
    };
    let eager_trial = eager.ask(&[5, 7, 11], config).unwrap();
    let lazy_trial = lazy.ask_lazy(&[5, 7, 11], config).unwrap();
    same_candidate(lazy_trial, eager_trial);
    assert!(lazy.row(lazy_trial).is_err());
    eager.tell(eager_trial, 1.0, true).unwrap();
    lazy.tell(lazy_trial, 1.0, true).unwrap();
    let eager_next = eager.ask(&[13, 17], config).unwrap();
    let lazy_next = lazy.ask(&[13, 17], config).unwrap();
    same_candidate(lazy_next, eager_next);
    assert_eq!(lazy.row(lazy_next).unwrap(), eager.row(eager_next).unwrap());
}

#[cfg(feature = "opencl")]
fn opencl_unavailable(error: &str) -> bool {
    error.contains("no OpenCL GPU or CPU device")
        || error.contains("CL_PLATFORM_NOT_FOUND_KHR")
        || error.contains("failed to enumerate OpenCL")
}

#[cfg(feature = "opencl")]
#[test]
fn opencl_history() {
    let base = [0x76, 0x98, 0x0a, 100, 120, 140, 160];
    match Search::new(&base, 0.0, leaves(), 3, ComputeDevice::OpenCl) {
        Ok(_) => lazy_match(ComputeDevice::OpenCl),
        Err(error) if opencl_unavailable(&error) => {}
        Err(error) => panic!("{error}"),
    }
}

#[test]
fn rejected_center() {
    let base = [0x76, 0x98, 0x0a, 100, 120, 140, 160];
    let mut search = Search::new(&base, 0.0, leaves(), 2, ComputeDevice::Cpu).unwrap();
    let mut control = Search::new(&base, 0.0, leaves(), 2, ComputeDevice::Cpu).unwrap();
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
    let mut search = Search::new(&base, 0.0, leaves(), 3, ComputeDevice::Cpu).unwrap();
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
    let mut control = Search::new(&base, 0.0, leaves(), 3, ComputeDevice::Cpu).unwrap();
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
    let mut search = Search::new(&base, 0.0, leaves(), 2, ComputeDevice::Cpu).unwrap();
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
    let mut search = Search::new(&base, 0.0, leaves(), 2, ComputeDevice::Cpu).unwrap();
    search
        .indexed_history(&observations, |id| {
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
    let mut search = Search::new(&base, 0.0, leaves(), 3, ComputeDevice::Cpu).unwrap();
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
