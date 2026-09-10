#[cfg(all(target_os = "macos", feature = "metal"))]
use ennx::experimental::BpannHistory;
use ennx::experimental::{
    apply_dense, dense_linear, AcquisitionKind, ComputeDevice, DenseLeaf, DenseLinear, DenseTerm,
    DenseView, SearchCenter, SearchConfig,
};
use ennx::search::Parameter;
use ennx::search::Search;
#[cfg(all(target_os = "macos", feature = "metal"))]
use ndarray::{array, Axis};
#[cfg(all(target_os = "macos", feature = "metal"))]
use tempfile::TempDir;

fn leaves() -> Vec<Parameter> {
    vec![
        Parameter::new(0, 257, 4, 0.25, 1.0, 0.75).unwrap(),
        Parameter::new(257, 263, 8, 0.5, 0.5, 1.0).unwrap(),
    ]
}

fn base() -> Vec<u8> {
    let row_bytes = 257usize.div_ceil(2) + 263;
    (0..row_bytes)
        .map(|index| (index.wrapping_mul(37).wrapping_add(11) & 0xff) as u8)
        .collect()
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn metal_unavailable(error: &str) -> bool {
    error.contains("no default Metal device found")
}

#[cfg(feature = "opencl")]
fn opencl_unavailable(error: &str) -> bool {
    error.contains("no OpenCL GPU or CPU device")
        || error.contains("CL_PLATFORM_NOT_FOUND_KHR")
        || error.contains("failed to enumerate OpenCL GPU devices")
        || error.contains("failed to enumerate OpenCL")
        || error.contains("CL_INVALID_VALUE")
}

fn altered_row(base: &[u8], tweak: u8) -> Vec<u8> {
    base.iter()
        .enumerate()
        .map(|(index, value)| value.wrapping_add(tweak.wrapping_add((index % 7) as u8)))
        .collect()
}

fn encoded_rows(base: &[u8], left_tweak: u8, right_tweak: u8) -> Vec<u8> {
    let mut rows = Vec::with_capacity(base.len() * 2);
    rows.extend_from_slice(&altered_row(base, left_tweak));
    rows.extend_from_slice(&altered_row(base, right_tweak));
    rows
}

fn dense_input() -> (Vec<f32>, Vec<DenseLeaf>, Vec<DenseTerm>) {
    (
        vec![0.5, -1.0, 2.0, 0.25, 4.0, -2.0, 0.75, -0.125],
        vec![
            DenseLeaf::new(11, 0, 4, 0.5).unwrap(),
            DenseLeaf::new(29, 4, 4, 1.25).unwrap(),
        ],
        vec![
            DenseTerm::new(0x1234_5678_9abc_def0, 0.01).unwrap(),
            DenseTerm::new(91, -0.0025).unwrap(),
        ],
    )
}

fn linear_input() -> (
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    DenseView,
    DenseView,
    Vec<DenseTerm>,
) {
    (
        vec![0.25, -0.5, 1.5, 2.0],
        vec![0.5, -1.0, 0.75, 0.25, -0.5, 2.0, 1.25, -0.75],
        vec![0.125, -0.25],
        DenseView::new(11, 0, 0.02).unwrap(),
        DenseView::new(29, 0, 0.01).unwrap(),
        vec![
            DenseTerm::new(0x1234_5678_9abc_def0, 0.5).unwrap(),
            DenseTerm::new(91, -0.125).unwrap(),
        ],
    )
}

fn ask(
    device: ComputeDevice,
    acquisition: AcquisitionKind,
) -> Result<(usize, f32, Vec<u8>), String> {
    let mut search = Search::new(&base(), 0.25, leaves(), 4, device)?;
    let warm = search.ask(
        &[17],
        SearchConfig {
            neighbors: 1,
            length: 1.0,
            ..SearchConfig::default()
        },
    )?;
    search.tell(warm, 0.75, true)?;
    let trial = search.ask(
        &[19, 23, 29, 31],
        SearchConfig {
            neighbors: 2,
            length: 0.65,
            beta: 1.3,
            acquisition,
            seed: 41,
            ..SearchConfig::default()
        },
    )?;
    Ok((trial.index, trial.score, search.row(trial)?))
}

fn sparse_updates(device: ComputeDevice) -> Result<Vec<(usize, f32, Vec<u8>)>, String> {
    let base = base();
    let mut search = Search::new(&base, 0.25, leaves(), 4, device)?;

    let history_a = encoded_rows(&base, 17, 31);
    search.replace_history(&history_a, &[1.0, 2.0])?;
    let trial_a = search.ask_sparse(
        &[7, 11, 13],
        2,
        SearchConfig {
            neighbors: 1,
            length: 1.0,
            ..SearchConfig::default()
        },
    )?;
    let row_a = search.row(trial_a)?;
    search.tell(trial_a, 0.75, true)?;

    let history_b = encoded_rows(&base, 43, 59);
    search.replace_history(&history_b, &[3.0, 4.0])?;
    let trial_b = search.ask_sparse(
        &[17, 19, 23],
        2,
        SearchConfig {
            neighbors: 1,
            length: 0.65,
            beta: 1.3,
            seed: 41,
            ..SearchConfig::default()
        },
    )?;
    let row_b = search.row(trial_b)?;

    Ok(vec![
        (trial_a.index, trial_a.score, row_a),
        (trial_b.index, trial_b.score, row_b),
    ])
}

fn tree_updates(device: ComputeDevice) -> Result<Vec<(usize, f32)>, String> {
    let base = base();
    let mut search = Search::new(&base, 0.25, leaves(), 4, device)?;
    let warm = search.ask(
        &[17],
        SearchConfig {
            neighbors: 1,
            length: 1.0,
            ..SearchConfig::default()
        },
    )?;
    search.tell(warm, 0.75, true)?;

    let seeds = [19, 23, 29, 31, 37, 41];
    let first = search.ask_centers(
        2,
        3,
        &[
            SearchCenter {
                parent: None,
                seed: 101,
            },
            SearchCenter {
                parent: Some(0),
                seed: 103,
            },
        ],
        &[0, 1],
        &seeds,
        SearchConfig {
            neighbors: 1,
            length: 0.65,
            acquisition: AcquisitionKind::Ucb,
            seed: 43,
            ..SearchConfig::default()
        },
    )?;
    let second = search.ask_centers(
        2,
        3,
        &[
            SearchCenter {
                parent: None,
                seed: 151,
            },
            SearchCenter {
                parent: Some(0),
                seed: 157,
            },
            SearchCenter {
                parent: Some(1),
                seed: 163,
            },
        ],
        &[1, 2],
        &seeds,
        SearchConfig {
            neighbors: 1,
            length: 0.8,
            acquisition: AcquisitionKind::Ucb,
            seed: 43,
            ..SearchConfig::default()
        },
    )?;

    let mut result = first;
    result.extend(second);
    Ok(result)
}

#[test]
fn cpu_repeatable() {
    for acquisition in [
        AcquisitionKind::Ucb,
        AcquisitionKind::Thompson,
        AcquisitionKind::Pareto,
    ] {
        let left = ask(ComputeDevice::Cpu, acquisition).unwrap();
        let right = ask(ComputeDevice::Cpu, acquisition).unwrap();
        assert_eq!(left, right);
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn metal_bytes() {
    for acquisition in [
        AcquisitionKind::Ucb,
        AcquisitionKind::Thompson,
        AcquisitionKind::Pareto,
    ] {
        let cpu = ask(ComputeDevice::Cpu, acquisition).unwrap();
        let metal = match ask(ComputeDevice::Metal, acquisition) {
            Ok(value) => value,
            Err(error) if metal_unavailable(&error) => return,
            Err(error) => panic!("{error}"),
        };
        assert_eq!(metal.0, cpu.0);
        assert!((metal.1 - cpu.1).abs() <= 1.0e-5, "{metal:?} != {cpu:?}");
        assert_eq!(metal.2, cpu.2);
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn metal_updates() {
    let cpu_sparse = sparse_updates(ComputeDevice::Cpu).unwrap();
    let metal_sparse = match sparse_updates(ComputeDevice::Metal) {
        Ok(value) => value,
        Err(error) if metal_unavailable(&error) => return,
        Err(error) => panic!("{error}"),
    };
    assert_eq!(metal_sparse.len(), cpu_sparse.len());
    for ((index, score, row), (cpu_index, cpu_score, cpu_row)) in
        metal_sparse.into_iter().zip(cpu_sparse)
    {
        assert_eq!(index, cpu_index);
        assert!(
            (score - cpu_score).abs() <= 5.0e-5,
            "sparse resident score mismatch at index {index}: metal={score} cpu={cpu_score}"
        );
        assert_eq!(row, cpu_row);
    }

    let cpu_tree = tree_updates(ComputeDevice::Cpu).unwrap();
    let metal_tree = match tree_updates(ComputeDevice::Metal) {
        Ok(value) => value,
        Err(error) if metal_unavailable(&error) => return,
        Err(error) => panic!("{error}"),
    };
    assert_eq!(metal_tree.len(), cpu_tree.len());
    for ((index, score), (cpu_index, cpu_score)) in metal_tree.into_iter().zip(cpu_tree) {
        assert_eq!(index, cpu_index);
        assert!(
            (score - cpu_score).abs() <= 1.0e-5,
            "tree resident score mismatch at index {index}: metal={score} cpu={cpu_score}"
        );
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn metal_reduction() {
    for count in [1usize, 31, 32, 33, 63, 64, 65, 257] {
        let seeds: Vec<u64> = (0..count).map(|index| 10_001 + index as u64).collect();
        for acquisition in [
            AcquisitionKind::Ucb,
            AcquisitionKind::Thompson,
            AcquisitionKind::Pareto,
        ] {
            let run = |device| -> Result<(usize, f32), String> {
                let mut search = Search::new(&base(), 0.25, leaves(), 4, device)?;
                let trial = search.ask(
                    &seeds,
                    SearchConfig {
                        neighbors: 1,
                        length: 0.65,
                        beta: 1.3,
                        acquisition,
                        seed: 41,
                        ..SearchConfig::default()
                    },
                )?;
                Ok((trial.index, trial.score))
            };
            let cpu = run(ComputeDevice::Cpu).unwrap();
            let metal = match run(ComputeDevice::Metal) {
                Ok(value) => value,
                Err(error) if metal_unavailable(&error) => return,
                Err(error) => panic!("{error}"),
            };
            assert_eq!(metal.0, cpu.0, "candidate count {count}");
            assert!(
                (metal.1 - cpu.1).abs() <= 1.0e-5,
                "candidate count {count}: {metal:?} != {cpu:?}"
            );
        }
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn metal_history() {
    let mut cpu = Search::new(&base(), 0.25, leaves(), 4, ComputeDevice::Cpu).unwrap();
    let mut metal = match Search::new(&base(), 0.25, leaves(), 4, ComputeDevice::Metal) {
        Ok(search) => search,
        Err(error) if metal_unavailable(&error) => return,
        Err(error) => panic!("{error}"),
    };
    for round in 0..9 {
        let seeds: Vec<u64> = (0..7)
            .map(|candidate| 100 + round * 7 + candidate)
            .collect();
        let config = SearchConfig {
            neighbors: (round as usize + 1).min(4),
            length: if round % 3 == 0 { 0.65 } else { 0.8 },
            acquisition: if round % 2 == 0 {
                AcquisitionKind::Ucb
            } else {
                AcquisitionKind::Thompson
            },
            seed: 900 + round,
            ..SearchConfig::default()
        };
        let cpu_trial = cpu.ask(&seeds, config).unwrap();
        let metal_trial = metal.ask(&seeds, config).unwrap();
        assert_eq!(metal_trial.index, cpu_trial.index);
        assert!((metal_trial.score - cpu_trial.score).abs() <= 1.0e-5);
        assert_eq!(metal.row(metal_trial).unwrap(), cpu.row(cpu_trial).unwrap());
        let value = round as f32 * 0.125;
        let accept = round % 2 == 0;
        cpu.tell(cpu_trial, value, accept).unwrap();
        metal.tell(metal_trial, value, accept).unwrap();
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn dense_directions() {
    let (base, leaves, terms) = dense_input();
    let cpu = apply_dense(&base, &leaves, &terms, ComputeDevice::Cpu).unwrap();
    let device = ComputeDevice::Metal;
    let gpu = match apply_dense(&base, &leaves, &terms, device) {
        Ok(result) => result,
        Err(error) if metal_unavailable(&error) => return,
        Err(error) => panic!("{error}"),
    };
    assert_eq!(gpu.changed, base.len());
    for (left, right) in gpu.values.iter().zip(&cpu.values) {
        assert!((left - right).abs() <= f32::EPSILON);
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn linear_eval() {
    let (input, weight, bias, weight_view, bias_view, terms) = linear_input();
    let run = |device| {
        dense_linear(
            &input,
            &weight,
            Some(&bias),
            weight_view,
            Some(bias_view),
            &terms,
            device,
        )
    };
    let cpu = run(ComputeDevice::Cpu).unwrap();
    let device = ComputeDevice::Metal;
    let gpu = match run(device) {
        Ok(result) => result,
        Err(error) if metal_unavailable(&error) => return,
        Err(error) => panic!("{error}"),
    };
    for (left, right) in gpu.iter().zip(&cpu) {
        assert!((left - right).abs() <= 1.0e-5);
    }
    let mut resident = DenseLinear::new(
        weight.clone(),
        input.len(),
        Some(bias.clone()),
        weight_view,
        Some(bias_view),
        device,
    )
    .unwrap();
    for (left, right) in resident.eval(&input, &terms).unwrap().iter().zip(&gpu) {
        assert!((left - right).abs() <= 1.0e-5);
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn metal_shortlist() {
    let packed_rows: Vec<u8> = [base(), base()]
        .into_iter()
        .flatten()
        .enumerate()
        .map(|(index, value)| value.wrapping_add((index % 7) as u8))
        .collect();
    let values = [0.5, 1.5];
    let run = |device| -> Result<(usize, f32, Vec<u8>), String> {
        let mut search = Search::new(&base(), 0.25, leaves(), 4, device)?;
        search.replace_history(&packed_rows, &values)?;
        let trial = search.ask(
            &[19, 23, 29, 31],
            SearchConfig {
                neighbors: 2,
                length: 0.65,
                beta: 1.3,
                seed: 41,
                ..SearchConfig::default()
            },
        )?;
        Ok((trial.index, trial.score, search.row(trial)?))
    };
    let cpu = run(ComputeDevice::Cpu).unwrap();
    let metal = match run(ComputeDevice::Metal) {
        Ok(value) => value,
        Err(error) if metal_unavailable(&error) => return,
        Err(error) => panic!("{error}"),
    };
    assert_eq!(metal.0, cpu.0);
    assert!((metal.1 - cpu.1).abs() <= 1.0e-5, "{metal:?} != {cpu:?}");
    assert_eq!(metal.2, cpu.2);
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn metal_regions() {
    let seeds = [19, 23, 29, 31, 37, 41];
    let config = SearchConfig {
        neighbors: 1,
        length: 0.65,
        beta: 1.3,
        acquisition: AcquisitionKind::Ucb,
        seed: 43,
        ..SearchConfig::default()
    };
    let warm = |device| -> Result<Search, String> {
        let mut search = Search::new(&base(), 0.25, leaves(), 4, device)?;
        let trial = search.ask(
            &[17],
            SearchConfig {
                neighbors: 1,
                ..SearchConfig::default()
            },
        )?;
        search.tell(trial, 0.75, true)?;
        Ok(search)
    };

    let mut expected = Vec::new();
    for (region, region_seeds) in seeds.chunks_exact(3).enumerate() {
        let trial = warm(ComputeDevice::Cpu)
            .unwrap()
            .ask(region_seeds, config)
            .unwrap();
        expected.push((region * 3 + trial.index, trial.score));
    }

    let device = ComputeDevice::Metal;
    let actual = match warm(device) {
        Ok(mut search) => search.ask_regions(2, 3, &seeds, config),
        Err(error) if metal_unavailable(&error) => return,
        Err(error) => panic!("{error}"),
    }
    .unwrap();
    assert_eq!(actual.len(), expected.len());
    for ((actual_index, actual_score), &(expected_index, expected_score)) in
        actual.into_iter().zip(&expected)
    {
        assert_eq!(actual_index, expected_index);
        assert!((actual_score - expected_score).abs() <= 1.0e-5);
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn metal_centers() {
    let centers = [
        SearchCenter {
            parent: None,
            seed: 101,
        },
        SearchCenter {
            parent: Some(0),
            seed: 103,
        },
    ];
    let region_centers = [0, 1];
    let seeds = [19, 23, 29, 31, 37, 41];
    let config = SearchConfig {
        neighbors: 1,
        length: 0.65,
        acquisition: AcquisitionKind::Ucb,
        seed: 43,
        ..SearchConfig::default()
    };
    let run = |device| {
        Search::new(&base(), 0.25, leaves(), 4, device).and_then(|mut search| {
            search.ask_centers(2, 3, &centers, &region_centers, &seeds, config)
        })
    };

    let cpu = run(ComputeDevice::Cpu).unwrap();
    let metal = match run(ComputeDevice::Metal) {
        Ok(value) => value,
        Err(error) if metal_unavailable(&error) => return,
        Err(error) => panic!("{error}"),
    };
    assert_eq!(metal.len(), cpu.len());
    for ((index, score), &(cpu_index, cpu_score)) in metal.into_iter().zip(&cpu) {
        assert_eq!(index, cpu_index);
        assert!((score - cpu_score).abs() <= 1.0e-5);
    }
}

#[cfg(feature = "opencl")]
#[test]
fn opencl_regions() {
    let seeds = [19, 23, 29, 31, 37, 41];
    let config = SearchConfig {
        neighbors: 1,
        length: 0.65,
        beta: 1.3,
        acquisition: AcquisitionKind::Ucb,
        seed: 43,
        ..SearchConfig::default()
    };
    let warm = |device| -> Result<Search, String> {
        let mut search = Search::new(&base(), 0.25, leaves(), 4, device)?;
        let trial = search.ask(
            &[17],
            SearchConfig {
                neighbors: 1,
                ..SearchConfig::default()
            },
        )?;
        search.tell(trial, 0.75, true)?;
        Ok(search)
    };

    let mut expected = Vec::new();
    for (region, region_seeds) in seeds.chunks_exact(3).enumerate() {
        let trial = warm(ComputeDevice::Cpu)
            .unwrap()
            .ask(region_seeds, config)
            .unwrap();
        expected.push((region * 3 + trial.index, trial.score));
    }

    let actual = match warm(ComputeDevice::OpenCl) {
        Ok(mut search) => search.ask_regions(2, 3, &seeds, config),
        Err(error) if opencl_unavailable(&error) => return,
        Err(error) => panic!("{error}"),
    }
    .unwrap();
    assert_eq!(actual.len(), expected.len());
    for ((actual_index, actual_score), &(expected_index, expected_score)) in
        actual.into_iter().zip(&expected)
    {
        assert_eq!(actual_index, expected_index);
        assert!((actual_score - expected_score).abs() <= 1.0e-5);
    }
}

#[cfg(feature = "opencl")]
#[test]
fn opencl_centers() {
    let centers = [
        SearchCenter {
            parent: None,
            seed: 101,
        },
        SearchCenter {
            parent: Some(0),
            seed: 103,
        },
    ];
    let region_centers = [0, 1];
    let seeds = [19, 23, 29, 31, 37, 41];
    let config = SearchConfig {
        neighbors: 1,
        length: 0.65,
        acquisition: AcquisitionKind::Ucb,
        seed: 43,
        ..SearchConfig::default()
    };
    let run = |device| {
        Search::new(&base(), 0.25, leaves(), 4, device).and_then(|mut search| {
            search.ask_centers(2, 3, &centers, &region_centers, &seeds, config)
        })
    };

    let cpu = run(ComputeDevice::Cpu).unwrap();
    let opencl = match run(ComputeDevice::OpenCl) {
        Ok(value) => value,
        Err(error) if opencl_unavailable(&error) => return,
        Err(error) => panic!("{error}"),
    };
    assert_eq!(opencl.len(), cpu.len());
    for ((opencl_index, opencl_score), (cpu_index, cpu_score)) in opencl.into_iter().zip(cpu) {
        assert_eq!(opencl_index, cpu_index);
        assert!((opencl_score - cpu_score).abs() <= 1.0e-5);
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn bpann_ask(device: ComputeDevice) -> Result<(usize, f32, Vec<u8>), String> {
    let archive = [
        base(),
        base()
            .into_iter()
            .map(|value| value.wrapping_add(17))
            .collect(),
        base()
            .into_iter()
            .map(|value| value.wrapping_add(31))
            .collect(),
    ];
    let descriptors = array![[0.0, 0.0], [1.0, 0.0], [4.0, 0.0]];
    let dir = TempDir::new().map_err(|error| error.to_string())?;
    let mut history = BpannHistory::new(dir.path().to_path_buf(), 2)?;
    for (index, descriptor) in descriptors.axis_iter(Axis(0)).enumerate() {
        history.append(&descriptor, (index as f32 + 1.0) * 10.0)?;
    }

    let candidate_descriptors = array![[0.1, 0.0], [3.9, 0.0]];
    let mut search = Search::new(&base(), 0.25, leaves(), 4, device)?;
    let trial = search.ask_indexed(
        &history,
        &candidate_descriptors.view(),
        1,
        &[19, 23],
        SearchConfig {
            neighbors: 1,
            length: 0.65,
            beta: 1.3,
            seed: 41,
            ..SearchConfig::default()
        },
        |id| Ok(archive[id.0 as usize].clone()),
    )?;
    Ok((trial.index, trial.score, search.row(trial)?))
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn metal_bpann() {
    let cpu = bpann_ask(ComputeDevice::Cpu).unwrap();
    let metal = match bpann_ask(ComputeDevice::Metal) {
        Ok(value) => value,
        Err(error) if metal_unavailable(&error) => return,
        Err(error) => panic!("{error}"),
    };
    assert_eq!(metal.0, cpu.0);
    assert!((metal.1 - cpu.1).abs() <= 1.0e-5, "{metal:?} != {cpu:?}");
    assert_eq!(metal.2, cpu.2);
}

#[cfg(feature = "opencl")]
#[test]
fn opencl_cpu() {
    for acquisition in [
        AcquisitionKind::Ucb,
        AcquisitionKind::Thompson,
        AcquisitionKind::Pareto,
    ] {
        let cpu = ask(ComputeDevice::Cpu, acquisition).unwrap();
        let opencl = match ask(ComputeDevice::OpenCl, acquisition) {
            Ok(value) => value,
            Err(error) if opencl_unavailable(&error) => return,
            Err(error) => panic!("{error}"),
        };
        assert_eq!(opencl.0, cpu.0);
        assert!((opencl.1 - cpu.1).abs() <= 1.0e-5);
        assert_eq!(opencl.2, cpu.2);
    }
}

#[cfg(feature = "opencl")]
#[test]
fn opencl_updates() {
    let cpu_sparse = sparse_updates(ComputeDevice::Cpu).unwrap();
    let opencl_sparse = match sparse_updates(ComputeDevice::OpenCl) {
        Ok(value) => value,
        Err(error) if opencl_unavailable(&error) => return,
        Err(error) => panic!("{error}"),
    };
    assert_eq!(opencl_sparse.len(), cpu_sparse.len());
    for ((index, score, row), (cpu_index, cpu_score, cpu_row)) in
        opencl_sparse.into_iter().zip(cpu_sparse)
    {
        assert_eq!(index, cpu_index);
        assert!((score - cpu_score).abs() <= 1.0e-5);
        assert_eq!(row, cpu_row);
    }

    let cpu_tree = tree_updates(ComputeDevice::Cpu).unwrap();
    let opencl_tree = match tree_updates(ComputeDevice::OpenCl) {
        Ok(value) => value,
        Err(error) if opencl_unavailable(&error) => return,
        Err(error) => panic!("{error}"),
    };
    assert_eq!(opencl_tree.len(), cpu_tree.len());
    for ((index, score), (cpu_index, cpu_score)) in opencl_tree.into_iter().zip(cpu_tree) {
        assert_eq!(index, cpu_index);
        assert!((score - cpu_score).abs() <= 1.0e-5);
    }
}

#[cfg(feature = "opencl")]
#[test]
fn opencl_linear() {
    let (input, weight, bias, weight_view, bias_view, terms) = linear_input();
    let run = |device| {
        dense_linear(
            &input,
            &weight,
            Some(&bias),
            weight_view,
            Some(bias_view),
            &terms,
            device,
        )
    };
    let cpu = run(ComputeDevice::Cpu).unwrap();
    let opencl = match run(ComputeDevice::OpenCl) {
        Ok(value) => value,
        Err(error) if opencl_unavailable(&error) => return,
        Err(error) => panic!("{error}"),
    };
    for (left, right) in opencl.iter().zip(cpu) {
        assert!((left - right).abs() <= 1.0e-5);
    }
    let mut resident = DenseLinear::new(
        weight,
        input.len(),
        Some(bias),
        weight_view,
        Some(bias_view),
        ComputeDevice::OpenCl,
    )
    .unwrap();
    for (left, right) in resident.eval(&input, &terms).unwrap().iter().zip(opencl) {
        assert!((left - right).abs() <= 1.0e-5);
    }
}

#[cfg(feature = "opencl")]
#[test]
fn opencl_directions() {
    let (base, leaves, terms) = dense_input();
    let cpu = apply_dense(&base, &leaves, &terms, ComputeDevice::Cpu).unwrap();
    let opencl = match apply_dense(&base, &leaves, &terms, ComputeDevice::OpenCl) {
        Ok(result) => result,
        Err(error) if opencl_unavailable(&error) => return,
        Err(error) => panic!("{error}"),
    };
    assert_eq!(opencl.changed, base.len());
    for (left, right) in opencl.values.iter().zip(cpu.values) {
        assert!((left - right).abs() <= f32::EPSILON);
    }
}
