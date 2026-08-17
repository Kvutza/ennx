use ennx::experimental::{
    apply_dense, dense_linear, AcquisitionKind, BpannHistory, ComputeDevice, DenseLeaf,
    DenseLinear, DenseTerm, DenseView, PackedLeaf, PackedSearch, SearchCenter, SearchConfig,
};
use ndarray::{array, Axis};
use tempfile::TempDir;

fn leaves() -> Vec<PackedLeaf> {
    vec![
        PackedLeaf::new(0, 257, 4, 0.25, 1.0, 0.75).unwrap(),
        PackedLeaf::new(257, 263, 8, 0.5, 0.5, 1.0).unwrap(),
    ]
}

fn base() -> Vec<u8> {
    let row_bytes = 257usize.div_ceil(2) + 263;
    (0..row_bytes)
        .map(|index| (index.wrapping_mul(37).wrapping_add(11) & 0xff) as u8)
        .collect()
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
    let mut search = PackedSearch::new(&base(), 0.25, leaves(), 4, device)?;
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

#[test]
fn cpu_trial_search_is_repeatable() {
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
fn metal_matches_cpu_trial_bytes() {
    for acquisition in [
        AcquisitionKind::Ucb,
        AcquisitionKind::Thompson,
        AcquisitionKind::Pareto,
    ] {
        let cpu = ask(ComputeDevice::Cpu, acquisition).unwrap();
        let metal = ask(ComputeDevice::Metal, acquisition).unwrap();
        let agx = ask(ComputeDevice::Agx, acquisition).unwrap();
        assert_eq!(metal.0, cpu.0);
        assert!((metal.1 - cpu.1).abs() <= 1.0e-5, "{metal:?} != {cpu:?}");
        assert_eq!(metal.2, cpu.2);
        assert_eq!(agx.0, cpu.0);
        assert!((agx.1 - cpu.1).abs() <= 1.0e-5, "{agx:?} != {cpu:?}");
        assert_eq!(agx.2, cpu.2);
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn agx_acquisition_reductions_match_cpu_across_simd_boundaries() {
    for count in [1usize, 31, 32, 33, 63, 64, 65, 257] {
        let seeds: Vec<u64> = (0..count).map(|index| 10_001 + index as u64).collect();
        for acquisition in [
            AcquisitionKind::Ucb,
            AcquisitionKind::Thompson,
            AcquisitionKind::Pareto,
        ] {
            let run = |device| -> Result<(usize, f32), String> {
                let mut search = PackedSearch::new(&base(), 0.25, leaves(), 4, device)?;
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
            let agx = run(ComputeDevice::Agx).unwrap();
            assert_eq!(agx.0, cpu.0, "candidate count {count}");
            assert!(
                (agx.1 - cpu.1).abs() <= 1.0e-5,
                "candidate count {count}: {agx:?} != {cpu:?}"
            );
        }
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn agx_resident_state_matches_cpu_across_rolling_history() {
    let mut cpu = PackedSearch::new(&base(), 0.25, leaves(), 4, ComputeDevice::Cpu).unwrap();
    let mut agx = PackedSearch::new(&base(), 0.25, leaves(), 4, ComputeDevice::Agx).unwrap();
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
        let agx_trial = agx.ask(&seeds, config).unwrap();
        assert_eq!(agx_trial.index, cpu_trial.index);
        assert!((agx_trial.score - cpu_trial.score).abs() <= 1.0e-5);
        assert_eq!(agx.row(agx_trial).unwrap(), cpu.row(cpu_trial).unwrap());
        let value = round as f32 * 0.125;
        let accept = round % 2 == 0;
        cpu.tell(cpu_trial, value, accept).unwrap();
        agx.tell(agx_trial, value, accept).unwrap();
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn dense_directions_match_cpu() {
    let (base, leaves, terms) = dense_input();
    let cpu = apply_dense(&base, &leaves, &terms, ComputeDevice::Cpu).unwrap();
    for device in [ComputeDevice::Metal, ComputeDevice::Agx] {
        let gpu = apply_dense(&base, &leaves, &terms, device).unwrap();
        assert_eq!(gpu.changed, base.len());
        for (left, right) in gpu.values.iter().zip(&cpu.values) {
            assert!((left - right).abs() <= f32::EPSILON);
        }
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn dense_linear_matches_cpu() {
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
        .unwrap()
    };
    let cpu = run(ComputeDevice::Cpu);
    for device in [ComputeDevice::Metal, ComputeDevice::Agx] {
        let gpu = run(device);
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
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn metal_matches_cpu_with_external_history_shortlist() {
    let packed_rows: Vec<u8> = [base(), base()]
        .into_iter()
        .flatten()
        .enumerate()
        .map(|(index, value)| value.wrapping_add((index % 7) as u8))
        .collect();
    let values = [0.5, 1.5];
    let run = |device| -> Result<(usize, f32, Vec<u8>), String> {
        let mut search = PackedSearch::new(&base(), 0.25, leaves(), 4, device)?;
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
    let metal = run(ComputeDevice::Metal).unwrap();
    assert_eq!(metal.0, cpu.0);
    assert!((metal.1 - cpu.1).abs() <= 1.0e-5, "{metal:?} != {cpu:?}");
    assert_eq!(metal.2, cpu.2);
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn metal_multi_tr_matches_independent_cpu_regions() {
    let seeds = [19, 23, 29, 31, 37, 41];
    let config = SearchConfig {
        neighbors: 1,
        length: 0.65,
        beta: 1.3,
        acquisition: AcquisitionKind::Ucb,
        seed: 43,
        ..SearchConfig::default()
    };
    let warm = |device| -> Result<PackedSearch, String> {
        let mut search = PackedSearch::new(&base(), 0.25, leaves(), 4, device)?;
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

    for device in [ComputeDevice::Metal, ComputeDevice::Agx] {
        let actual = warm(device)
            .unwrap()
            .ask_multi_tr(2, 3, &seeds, config)
            .unwrap();
        assert_eq!(actual.len(), expected.len());
        for ((actual_index, actual_score), &(expected_index, expected_score)) in
            actual.into_iter().zip(&expected)
        {
            assert_eq!(actual_index, expected_index);
            assert!((actual_score - expected_score).abs() <= 1.0e-5);
        }
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn metal_multi_tr_resolves_perturbation_tree() {
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
        PackedSearch::new(&base(), 0.25, leaves(), 4, device)
            .unwrap()
            .ask_multi_tr_tree(2, 3, &centers, &region_centers, &seeds, config)
            .unwrap()
    };

    let cpu = run(ComputeDevice::Cpu);
    let metal = run(ComputeDevice::Metal);
    let agx = run(ComputeDevice::Agx);
    for result in [metal, agx] {
        assert_eq!(result.len(), cpu.len());
        for ((index, score), &(cpu_index, cpu_score)) in result.into_iter().zip(&cpu) {
            assert_eq!(index, cpu_index);
            assert!((score - cpu_score).abs() <= 1.0e-5);
        }
    }
}

#[cfg(feature = "opencl")]
#[test]
fn opencl_multi_tr_resolves_perturbation_tree() {
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
        PackedSearch::new(&base(), 0.25, leaves(), 4, device)
            .unwrap()
            .ask_multi_tr_tree(2, 3, &centers, &region_centers, &seeds, config)
            .unwrap()
    };

    let cpu = run(ComputeDevice::Cpu);
    let opencl = run(ComputeDevice::OpenCl);
    assert_eq!(opencl.len(), cpu.len());
    for ((opencl_index, opencl_score), (cpu_index, cpu_score)) in opencl.into_iter().zip(cpu) {
        assert_eq!(opencl_index, cpu_index);
        assert!((opencl_score - cpu_score).abs() <= 1.0e-5);
    }
}

fn ask_with_bpann(device: ComputeDevice) -> Result<(usize, f32, Vec<u8>), String> {
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
    let mut search = PackedSearch::new(&base(), 0.25, leaves(), 4, device)?;
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
fn bpann_shortlist_to_metal_exact_search_matches_cpu() {
    let cpu = ask_with_bpann(ComputeDevice::Cpu).unwrap();
    let metal = ask_with_bpann(ComputeDevice::Metal).unwrap();
    assert_eq!(metal.0, cpu.0);
    assert!((metal.1 - cpu.1).abs() <= 1.0e-5, "{metal:?} != {cpu:?}");
    assert_eq!(metal.2, cpu.2);
}

#[cfg(feature = "opencl")]
#[test]
fn opencl_matches_cpu_when_a_device_exists() {
    for acquisition in [
        AcquisitionKind::Ucb,
        AcquisitionKind::Thompson,
        AcquisitionKind::Pareto,
    ] {
        let cpu = ask(ComputeDevice::Cpu, acquisition).unwrap();
        let opencl = match ask(ComputeDevice::OpenCl, acquisition) {
            Ok(value) => value,
            Err(error) if error.contains("no OpenCL GPU or CPU device") => return,
            Err(error) => panic!("{error}"),
        };
        assert_eq!(opencl.0, cpu.0);
        assert!((opencl.1 - cpu.1).abs() <= 1.0e-5);
        assert_eq!(opencl.2, cpu.2);
    }
}

#[cfg(feature = "opencl")]
#[test]
fn opencl_dense_linear_matches_cpu_when_a_device_exists() {
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
        Err(error)
            if error.contains("no OpenCL GPU or CPU device")
                || error.contains("failed to enumerate OpenCL GPU devices") =>
        {
            return;
        }
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
fn opencl_dense_directions_match_cpu_when_a_device_exists() {
    let (base, leaves, terms) = dense_input();
    let cpu = apply_dense(&base, &leaves, &terms, ComputeDevice::Cpu).unwrap();
    let opencl = match apply_dense(&base, &leaves, &terms, ComputeDevice::OpenCl) {
        Ok(result) => result,
        Err(error) if error.contains("no OpenCL GPU or CPU device") => return,
        Err(error) => panic!("{error}"),
    };
    assert_eq!(opencl.changed, base.len());
    for (left, right) in opencl.values.iter().zip(cpu.values) {
        assert!((left - right).abs() <= f32::EPSILON);
    }
}
