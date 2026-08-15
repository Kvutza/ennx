use ennx::experimental::{
    AcquisitionKind, ComputeBackend, EncodingType, WeightAsk, WeightCenter, WeightLeaf,
    WeightSearch, WeightTrial,
};

fn main() {
    if let Err(error) = run() {
        eprintln!("ERROR: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    test_basic_ask_tell()?;
    test_lazy_ask()?;
    test_all_acquisitions()?;
    test_all_encodings()?;
    test_replace_history()?;
    test_batch_candidates()?;
    test_multi_regions()?;
    test_compact_center_tree()?;
    println!("CUDA_TRIAL ok=true suites=8 target=sm_75");
    Ok(())
}

fn test_basic_ask_tell() -> Result<(), String> {
    let leaves = vec![
        WeightLeaf::new(0, 257, 4, 0.125, 1.0, 0.75)?,
        WeightLeaf::new(257, 259, 8, 0.03125, 1.0, 0.5)?,
    ];
    let row_bytes = 257_usize.div_ceil(2) + 259;
    let base = make_base(row_bytes);
    let mut cpu = WeightSearch::new(&base, -0.75, leaves.clone(), 6, ComputeBackend::Cpu)?;
    let mut cuda = WeightSearch::new(&base, -0.75, leaves, 6, ComputeBackend::Cuda)?;

    let mut config = WeightAsk {
        length: 0.8,
        neighbors: 1,
        ..WeightAsk::default()
    };
    let first_seeds = [3_u64, 17, 0xdead_beef_cafe_babe, u64::MAX - 9];
    compare_round(&mut cpu, &mut cuda, &first_seeds, config, 1.25, true)?;

    config.neighbors = 2;
    config.seed = 23;
    let second_seeds = [5_u64, 29, 0x0123_4567_89ab_cdef, u64::MAX - 3];
    compare_round(&mut cpu, &mut cuda, &second_seeds, config, 0.5, false)?;

    config.neighbors = 3;
    let third_seeds = [11_u64, 31, 77, 99];
    compare_round(&mut cpu, &mut cuda, &third_seeds, config, 2.0, true)?;

    Ok(())
}

fn test_lazy_ask() -> Result<(), String> {
    let leaves = vec![
        WeightLeaf::new(0, 128, 4, 0.25, 1.0, 0.8)?,
        WeightLeaf::new(128, 128, 8, 0.05, 1.0, 0.6)?,
    ];
    let row_bytes = 128_usize.div_ceil(2) + 128;
    let base = make_base(row_bytes);
    let mut cpu = WeightSearch::new(&base, 0.0, leaves.clone(), 4, ComputeBackend::Cpu)?;
    let mut cuda = WeightSearch::new(&base, 0.0, leaves, 4, ComputeBackend::Cuda)?;

    let config = WeightAsk {
        length: 0.75,
        neighbors: 1,
        ..WeightAsk::default()
    };
    let seeds = [7_u64, 19, 43, 89];
    let cpu_trial = cpu.ask_lazy(&seeds, config)?;
    let cuda_trial = cuda.ask_lazy(&seeds, config)?;
    compare_trial(cpu_trial, cuda_trial)?;

    // Both should error when requesting row before materialization
    assert!(cpu.row(cpu_trial).is_err());
    assert!(cuda.row(cuda_trial).is_err());

    cpu.materialize_pending(cpu_trial)?;
    cuda.materialize_pending(cuda_trial)?;

    let cpu_row = cpu.row(cpu_trial)?;
    let cuda_row = cuda.row(cuda_trial)?;
    assert_eq!(cpu_row, cuda_row);

    cpu.tell(cpu_trial, 1.5, true)?;
    cuda.tell(cuda_trial, 1.5, true)?;

    let next_trial_cpu = cpu.ask(&seeds, config)?;
    let next_trial_cuda = cuda.ask(&seeds, config)?;
    compare_trial(next_trial_cpu, next_trial_cuda)?;
    assert_eq!(cpu.row(next_trial_cpu)?, cuda.row(next_trial_cuda)?);

    Ok(())
}

fn test_all_acquisitions() -> Result<(), String> {
    let leaves = vec![
        WeightLeaf::new(0, 64, 4, 0.125, 1.0, 0.5)?,
        WeightLeaf::new(64, 64, 8, 0.05, 1.0, 0.5)?,
    ];
    let row_bytes = 64_usize.div_ceil(2) + 64;
    let base = make_base(row_bytes);
    let seeds = [13_u64, 27, 41, 59, 83];

    for acq in [
        AcquisitionKind::Ucb,
        AcquisitionKind::Thompson,
        AcquisitionKind::Pareto,
    ] {
        let mut cpu = WeightSearch::new(&base, 0.5, leaves.clone(), 4, ComputeBackend::Cpu)?;
        let mut cuda = WeightSearch::new(&base, 0.5, leaves.clone(), 4, ComputeBackend::Cuda)?;
        let config = WeightAsk {
            length: 0.6,
            neighbors: 1,
            acquisition: acq,
            seed: 42,
            beta: 1.5,
            ..WeightAsk::default()
        };
        compare_round(&mut cpu, &mut cuda, &seeds, config, 1.0, true)?;

        let next_config = WeightAsk {
            length: 0.6,
            neighbors: 2,
            acquisition: acq,
            seed: 99,
            beta: 2.0,
            ..WeightAsk::default()
        };
        compare_round(&mut cpu, &mut cuda, &seeds, next_config, 0.8, false)?;
    }
    Ok(())
}

fn test_all_encodings() -> Result<(), String> {
    let encodings = [
        (EncodingType::Int4, 4),
        (EncodingType::Int8, 8),
        (EncodingType::Fp4E2M1, 4),
        (EncodingType::Fp8E4M3, 8),
        (EncodingType::Fp8E5M2, 8),
    ];
    for (encoding, bits) in encodings {
        let leaf = WeightLeaf::new_with_encoding(0, 128, bits, encoding, 0.125, 1.0, 0.5)?;
        let row_bytes = if bits == 4 { 64 } else { 128 };
        let base = make_base(row_bytes);
        let mut cpu = WeightSearch::new(&base, 1.0, vec![leaf], 4, ComputeBackend::Cpu)?;
        let mut cuda = WeightSearch::new(&base, 1.0, vec![leaf], 4, ComputeBackend::Cuda)?;

        let config = WeightAsk {
            length: 0.8,
            neighbors: 1,
            ..WeightAsk::default()
        };
        let seeds = [5_u64, 17, 33, 65];
        compare_round(&mut cpu, &mut cuda, &seeds, config, 2.0, true)?;
    }
    Ok(())
}

fn test_replace_history() -> Result<(), String> {
    let leaves = vec![
        WeightLeaf::new(0, 64, 4, 0.25, 1.0, 0.75)?,
        WeightLeaf::new(64, 64, 8, 0.5, 0.5, 1.0)?,
    ];
    let row_bytes = 64_usize.div_ceil(2) + 64;
    let base = make_base(row_bytes);
    let mut cpu = WeightSearch::new(&base, 0.0, leaves.clone(), 4, ComputeBackend::Cpu)?;
    let mut cuda = WeightSearch::new(&base, 0.0, leaves, 4, ComputeBackend::Cuda)?;

    let mut replacement = Vec::new();
    for i in 0..2 {
        replacement.extend(
            make_base(row_bytes)
                .into_iter()
                .map(|b| b.wrapping_add(i as u8 * 50)),
        );
    }
    let values = [3.0_f32, 7.0];
    cpu.replace_history(&replacement, &values)?;
    cuda.replace_history(&replacement, &values)?;

    let config = WeightAsk {
        neighbors: 2,
        length: 1.0,
        ..WeightAsk::default()
    };
    let seeds = [17_u64, 23, 47];
    compare_round(&mut cpu, &mut cuda, &seeds, config, 9.0, false)?;
    Ok(())
}

fn test_batch_candidates() -> Result<(), String> {
    let leaves = vec![
        WeightLeaf::new(0, 512, 4, 0.125, 1.0, 0.75)?,
        WeightLeaf::new(512, 512, 8, 0.03125, 1.0, 0.5)?,
    ];
    let row_bytes = 512_usize.div_ceil(2) + 512;
    let base = make_base(row_bytes);
    let mut cpu = WeightSearch::new(&base, 0.0, leaves.clone(), 8, ComputeBackend::Cpu)?;
    let mut cuda = WeightSearch::new(&base, 0.0, leaves, 8, ComputeBackend::Cuda)?;

    let candidate_seeds = (1..=64)
        .map(|i| (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15))
        .collect::<Vec<_>>();
    let mut config = WeightAsk {
        length: 0.8,
        neighbors: 1,
        ..WeightAsk::default()
    };
    for round in 1..=4 {
        config.neighbors = round;
        compare_round(
            &mut cpu,
            &mut cuda,
            &candidate_seeds,
            config,
            round as f32 * 1.5,
            true,
        )?;
    }
    Ok(())
}

fn test_multi_regions() -> Result<(), String> {
    let leaves = vec![
        WeightLeaf::new(0, 257, 4, 0.125, 1.0, 0.75)?,
        WeightLeaf::new(257, 259, 8, 0.03125, 1.0, 0.5)?,
    ];
    let row_bytes = 257_usize.div_ceil(2) + 259;
    let base = make_base(row_bytes);
    let mut cpu = WeightSearch::new(&base, 0.25, leaves.clone(), 4, ComputeBackend::Cpu)?;
    let mut cuda = WeightSearch::new(&base, 0.25, leaves, 4, ComputeBackend::Cuda)?;
    let seeds = [19_u64, 23, 29, 31, 37, 41, 43, 47, 53];
    let config = WeightAsk {
        neighbors: 1,
        length: 0.65,
        beta: 1.3,
        ..WeightAsk::default()
    };
    let expected = cpu.ask_multi_tr(3, 3, &seeds, config)?;
    let actual = cuda.ask_multi_tr(3, 3, &seeds, config)?;
    compare_regions(&expected, &actual)
}

fn test_compact_center_tree() -> Result<(), String> {
    let leaves = vec![
        WeightLeaf::new(0, 257, 4, 0.125, 1.0, 0.75)?,
        WeightLeaf::new(257, 259, 8, 0.03125, 1.0, 0.5)?,
    ];
    let row_bytes = 257_usize.div_ceil(2) + 259;
    let base = make_base(row_bytes);
    let mut cpu = WeightSearch::new(&base, 0.25, leaves.clone(), 4, ComputeBackend::Cpu)?;
    let mut cuda = WeightSearch::new(&base, 0.25, leaves, 4, ComputeBackend::Cuda)?;
    let centers = [
        WeightCenter {
            parent: None,
            seed: 101,
        },
        WeightCenter {
            parent: Some(0),
            seed: 103,
        },
    ];
    let region_centers = [0, 1];
    let seeds = [19_u64, 23, 29, 31, 37, 41];
    let config = WeightAsk {
        neighbors: 1,
        length: 0.65,
        beta: 1.3,
        ..WeightAsk::default()
    };
    let expected = cpu.ask_multi_tr_tree(2, 3, &centers, &region_centers, &seeds, config)?;
    let actual = cuda.ask_multi_tr_tree(2, 3, &centers, &region_centers, &seeds, config)?;
    compare_regions(&expected, &actual)
}

fn compare_regions(expected: &[(usize, f32)], actual: &[(usize, f32)]) -> Result<(), String> {
    if expected.len() != actual.len() {
        return Err(format!(
            "CUDA returned {} regions, expected {}",
            actual.len(),
            expected.len()
        ));
    }
    for (region, (&(expected_index, expected_score), &(actual_index, actual_score))) in
        expected.iter().zip(actual).enumerate()
    {
        if expected_index != actual_index {
            return Err(format!(
                "CUDA region {region} selected {actual_index}, expected {expected_index}"
            ));
        }
        let tolerance = 2.0e-5 * expected_score.abs().max(1.0);
        if (expected_score - actual_score).abs() > tolerance {
            return Err(format!(
                "CUDA region {region} score {actual_score} differs from CPU score {expected_score} by more than {tolerance}"
            ));
        }
    }
    Ok(())
}

fn compare_round(
    cpu: &mut WeightSearch,
    cuda: &mut WeightSearch,
    seeds: &[u64],
    config: WeightAsk,
    reward: f32,
    accept: bool,
) -> Result<(), String> {
    let cpu_trial = cpu.ask(seeds, config)?;
    let cuda_trial = cuda.ask(seeds, config)?;
    compare_trial(cpu_trial, cuda_trial)?;
    let cpu_row = cpu.row(cpu_trial)?;
    let cuda_row = cuda.row(cuda_trial)?;
    if cpu_row != cuda_row {
        let mismatch = cpu_row
            .iter()
            .zip(&cuda_row)
            .position(|(left, right)| left != right)
            .unwrap_or(cpu_row.len().min(cuda_row.len()));
        return Err(format!("CUDA trial row mismatch at byte {mismatch}"));
    }
    cpu.tell(cpu_trial, reward, accept)?;
    cuda.tell(cuda_trial, reward, accept)?;
    Ok(())
}

fn compare_trial(cpu: WeightTrial, cuda: WeightTrial) -> Result<(), String> {
    if cpu.index != cuda.index || cpu.seed != cuda.seed {
        return Err(format!(
            "CUDA selected index={} seed={}, CPU selected index={} seed={}",
            cuda.index, cuda.seed, cpu.index, cpu.seed
        ));
    }
    let tolerance = 2.0e-5 * cpu.score.abs().max(1.0);
    if (cpu.score - cuda.score).abs() > tolerance {
        return Err(format!(
            "CUDA score {} differs from CPU score {} by more than {tolerance}",
            cuda.score, cpu.score
        ));
    }
    Ok(())
}

fn make_base(bytes: usize) -> Vec<u8> {
    (0..bytes)
        .map(|index| (index.wrapping_mul(37).wrapping_add(11) & 0xff) as u8)
        .collect()
}
