use ennx::search::Parameter;
use ennx::search::Search;
use std::time::Instant;

use ennx::experimental::{AcquisitionKind, ComputeDevice, EncodingType, SearchConfig};

#[derive(Debug)]
struct CycleSample {
    round: usize,
    proposal_s: f64,
    materialize_s: f64,
    objective_read_s: f64,
    evaluation_s: f64,
    update_s: f64,
    transfer_bytes: usize,
    host_allocations: usize,
    sync_points: usize,
    index: usize,
    seed: u64,
    trial_score: f32,
    reward: f32,
    accept: bool,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("ERROR: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let elements = arg_usize(&args, 1, 16 * 1024 * 1024)?;
    let history = arg_usize(&args, 2, 10)?;
    let candidates = arg_usize(&args, 3, 8)?;
    let rounds = arg_usize(&args, 4, 10)?;
    let warmup = arg_usize(&args, 5, history.saturating_sub(1))?;
    let device = parse_device(args.get(6).map(String::as_str).unwrap_or("auto"))?;
    let encoding_name = args.get(7).map(String::as_str).unwrap_or("int4");
    let acquisition = AcquisitionKind::parse(args.get(8).map(String::as_str).unwrap_or("ucb"))?;
    let length = arg_f32(&args, 9, 0.8)?;
    let neighbors = arg_usize(&args, 10, history.min(10))?;
    let beta = arg_f32(&args, 11, 1.0)?;
    let seed = arg_u64(&args, 12, 0)?;
    let edited_parameters = arg_usize(&args, 13, 0)?;

    if elements == 0 || history == 0 || candidates == 0 || rounds == 0 {
        return Err("elements, history, candidates, and rounds must be positive".to_string());
    }
    if neighbors == 0 {
        return Err("neighbors must be positive".to_string());
    }
    if edited_parameters > elements {
        return Err("edited_parameters must not exceed elements".to_string());
    }

    let (encoding, bits) = parse_encoding(encoding_name)?;
    let row_bytes = if bits == 4 {
        elements.div_ceil(2)
    } else {
        elements
    };
    let base: Vec<u8> = (0..row_bytes)
        .map(|index| (index.wrapping_mul(37).wrapping_add(11) & 0xff) as u8)
        .collect();
    let leaves = vec![Parameter::encoded(
        0, elements, bits, encoding, 0.125, 1.0, 0.25,
    )?];

    let setup_start = Instant::now();
    let mut search = Search::new(&base, 0.0, leaves, history, device)?;
    let setup_s = setup_start.elapsed().as_secs_f64();
    let selected_device = search.device();
    let benchmark_mode = if edited_parameters == 0 {
        "dense"
    } else {
        "sparse"
    };
    let peak_accounted_bytes =
        estimate_bytes2(elements, history, candidates, bits, edited_parameters)?;

    let proposal_config = SearchConfig {
        length,
        neighbors,
        beta,
        acquisition,
        seed,
        ..SearchConfig::default()
    };

    let warmup_start = Instant::now();
    let mut min_effective_neighbors = usize::MAX;
    let mut max_effective_neighbors = 0usize;
    for step in 0..warmup {
        let cycle_seed = seed.wrapping_add(step as u64);
        let candidate_seeds = seeds_round(candidates, cycle_seed);
        let mut config = proposal_config;
        let effective_neighbors = proposal_config.neighbors.min(search.history_len());
        min_effective_neighbors = min_effective_neighbors.min(effective_neighbors);
        max_effective_neighbors = max_effective_neighbors.max(effective_neighbors);
        config.neighbors = effective_neighbors;
        config.seed = cycle_seed;
        let _ = execute_cycle(
            &mut search,
            step,
            &candidate_seeds,
            edited_parameters,
            config,
            true,
        )?;
    }
    let warmup_s = warmup_start.elapsed().as_secs_f64();

    let mut samples = Vec::with_capacity(rounds);
    let bench_start = Instant::now();
    for round in 0..rounds {
        let cycle_seed = seed.wrapping_add(0x9e37_79b9_u64.wrapping_mul(round as u64 + 1));
        let candidate_seeds = seeds_round(candidates, cycle_seed);
        let mut config = proposal_config;
        let effective_neighbors = proposal_config.neighbors.min(search.history_len());
        min_effective_neighbors = min_effective_neighbors.min(effective_neighbors);
        max_effective_neighbors = max_effective_neighbors.max(effective_neighbors);
        config.neighbors = effective_neighbors;
        config.seed = cycle_seed;
        samples.push(execute_cycle(
            &mut search,
            round,
            &candidate_seeds,
            edited_parameters,
            config,
            round % 2 == 0,
        )?);
    }
    let bench_s = bench_start.elapsed().as_secs_f64();

    let proposal_median = median(samples.iter().map(|sample| sample.proposal_s).collect());
    let materialize_median = median(samples.iter().map(|sample| sample.materialize_s).collect());
    let objective_read_median = median(
        samples
            .iter()
            .map(|sample| sample.objective_read_s)
            .collect(),
    );
    let evaluation_median = median(samples.iter().map(|sample| sample.evaluation_s).collect());
    let update_median = median(samples.iter().map(|sample| sample.update_s).collect());

    println!("# ENNX proposal benchmark");
    println!("# elements={elements}");
    println!("# row_bytes={row_bytes}");
    println!("# history={history}");
    println!("# warmup={warmup}");
    println!("# candidates={candidates}");
    println!("# rounds={rounds}");
    println!("# requested_device={device:?}");
    println!("# selected_device={selected_device:?}");
    println!("# benchmark_mode={benchmark_mode}");
    println!("# encoding={encoding_name}");
    println!("# acquisition={acquisition:?}");
    println!("# neighbors={neighbors}");
    println!("# effective_neighbors_min={min_effective_neighbors}");
    println!("# effective_neighbors_max={max_effective_neighbors}");
    println!("# length={length}");
    println!("# beta={beta}");
    println!("# seed={seed}");
    println!("# edited_parameters={edited_parameters}");
    println!("# setup_s={setup_s:.9}");
    println!("# warmup_s={warmup_s:.9}");
    println!("# bench_s={bench_s:.9}");
    println!("# peak_accounted_bytes={peak_accounted_bytes}");
    println!(
        "# peak_accounted_bytes_note=includes resident rows, scratch, and a bounded proposal estimate; excludes driver-private caches and allocator fragmentation"
    );
    println!("# proposal_median_s={proposal_median:.9}");
    println!("# materialize_median_s={materialize_median:.9}");
    println!("# objective_read_median_s={objective_read_median:.9}");
    println!("# evaluation_median_s={evaluation_median:.9}");
    println!("# update_median_s={update_median:.9}");
    println!("# transfer_bytes_note=counts benchmark-owned input payloads and scalar evaluation-result readback; it does not include device-driver staging, allocator fragments, or resident row bytes that stay on device");
    println!("# host_allocations_note=counts explicit benchmark-side vector allocations; it does not include internal device buffer growth");
    println!("# sync_points_note=counts the proposal, materialize, objective-read, and update stage waits observed by the benchmark");
    println!("round,proposal_s,materialize_s,objective_read_s,evaluation_s,update_s,full_cycle_s,transfer_bytes,host_allocations,sync_points,index,seed,trial_score,reward,accept");
    for sample in samples {
        let full_cycle_s = sample.proposal_s
            + sample.materialize_s
            + sample.objective_read_s
            + sample.evaluation_s
            + sample.update_s;
        println!(
            "{},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{},{},{},{},{},{:.9},{:.9},{}",
            sample.round,
            sample.proposal_s,
            sample.materialize_s,
            sample.objective_read_s,
            sample.evaluation_s,
            sample.update_s,
            full_cycle_s,
            sample.transfer_bytes,
            sample.host_allocations,
            sample.sync_points,
            sample.index,
            sample.seed,
            sample.trial_score,
            sample.reward,
            sample.accept,
        );
    }
    Ok(())
}

fn execute_cycle(
    search: &mut Search,
    round: usize,
    seeds: &[u64],
    edited_parameters: usize,
    config: SearchConfig,
    accept: bool,
) -> Result<CycleSample, String> {
    let proposal_start = Instant::now();
    let trial = if edited_parameters == 0 {
        search.ask_lazy(seeds, config)?
    } else {
        search.ask_sparse(seeds, edited_parameters, config)?
    };
    let proposal_s = proposal_start.elapsed().as_secs_f64();

    let materialize_start = Instant::now();
    search.materialize_pending(trial)?;
    let materialize_s = materialize_start.elapsed().as_secs_f64();

    let scalar_start = Instant::now();
    let row_sum = search.byte_sum(trial)?;
    let objective_read_s = scalar_start.elapsed().as_secs_f64();

    let evaluation_start = Instant::now();
    let reward = evaluate_sum(row_sum, search.row_bytes(), trial.seed, round);
    let evaluation_s = evaluation_start.elapsed().as_secs_f64();

    let history_len = search.history_len();
    let update_start = Instant::now();
    search.tell(trial, reward, accept)?;
    let update_s = update_start.elapsed().as_secs_f64();

    let transfer_bytes = accounted_bytes(
        std::mem::size_of::<u64>(),
        seeds.len(),
        history_len,
        edited_parameters,
    );
    let host_allocations = 1;
    let sync_points = 4;

    Ok(CycleSample {
        round,
        proposal_s,
        materialize_s,
        objective_read_s,
        evaluation_s,
        update_s,
        transfer_bytes,
        host_allocations,
        sync_points,
        index: trial.index,
        seed: trial.seed,
        trial_score: trial.score,
        reward,
        accept,
    })
}

fn evaluate_sum(row_sum: u64, row_bytes: usize, seed: u64, round: usize) -> f32 {
    (row_sum as f32 / row_bytes.max(1) as f32)
        + ((seed & 0xffff) as f32) * 1.0e-6
        + round as f32 * 0.01
}

fn accounted_bytes(
    evaluation_result_bytes: usize,
    candidates: usize,
    history: usize,
    edited_parameters: usize,
) -> usize {
    let history_bytes = history.saturating_mul(std::mem::size_of::<u32>());
    let outcomes_bytes = history.saturating_mul(std::mem::size_of::<f32>());
    let seeds_bytes = candidates.saturating_mul(std::mem::size_of::<u64>());
    let draws_bytes = candidates.saturating_mul(std::mem::size_of::<f32>());
    let edits_bytes = if edited_parameters == 0 {
        0
    } else {
        candidates
            .saturating_mul(edited_parameters)
            .saturating_mul(2 * std::mem::size_of::<u32>())
    };
    history_bytes
        + outcomes_bytes
        + seeds_bytes
        + draws_bytes
        + edits_bytes
        + evaluation_result_bytes
}

fn estimate_bytes2(
    elements: usize,
    history: usize,
    candidates: usize,
    bits: u8,
    edited_parameters: usize,
) -> Result<usize, String> {
    let row_bytes = if bits == 4 {
        elements.div_ceil(2)
    } else {
        elements
    };
    let candidate_capacity = candidates.next_power_of_two().max(1);
    let tile_count = elements.div_ceil(65_536).max(1);
    let resident_rows = history
        .checked_add(2)
        .and_then(|rows| rows.checked_mul(row_bytes))
        .ok_or("proposal row allocation overflows")?;
    let scratch_small = candidate_capacity
        .checked_mul(8 + 4 + 4 + 4)
        .ok_or("proposal scratch allocation overflows")?;
    let partials = candidate_capacity
        .checked_mul(128)
        .and_then(|value| value.checked_mul(tile_count))
        .and_then(|value| value.checked_mul(4))
        .ok_or("proposal partial allocation overflows")?;
    let sparse_edits = candidate_capacity
        .checked_mul(edited_parameters)
        .and_then(|value| value.checked_mul(2 * std::mem::size_of::<u32>()))
        .ok_or("proposal sparse edit allocation overflows")?;
    resident_rows
        .checked_add(scratch_small)
        .and_then(|value| value.checked_add(partials))
        .and_then(|value| value.checked_add(sparse_edits))
        .ok_or("proposal memory estimate overflows".to_string())
}

fn seeds_round(count: usize, seed: u64) -> Vec<u64> {
    (0..count)
        .map(|candidate| seed.wrapping_add(10_000 + candidate as u64))
        .collect()
}

fn parse_device(name: &str) -> Result<ComputeDevice, String> {
    ComputeDevice::parse(name)
}

fn parse_encoding(name: &str) -> Result<(EncodingType, u8), String> {
    match name.trim().to_ascii_lowercase().as_str() {
        "int4" => Ok((EncodingType::Int4, 4)),
        "int8" => Ok((EncodingType::Int8, 8)),
        "fp4" | "fp4_e2m1" | "e2m1" => Ok((EncodingType::Fp4E2M1, 4)),
        "fp8" | "fp8_e4m3" | "e4m3" => Ok((EncodingType::Fp8E4M3, 8)),
        "fp8_e5m2" | "e5m2" => Ok((EncodingType::Fp8E5M2, 8)),
        other => Err(format!(
            "unknown encoding {other:?}; expected int4, int8, fp4, fp4_e2m1, e2m1, fp8, fp8_e4m3, e4m3, fp8_e5m2, or e5m2"
        )),
    }
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn arg_usize(args: &[String], index: usize, default: usize) -> Result<usize, String> {
    args.get(index)
        .map(|value| {
            value
                .parse()
                .map_err(|error| format!("invalid argument {index}: {error}"))
        })
        .unwrap_or(Ok(default))
}

fn arg_u64(args: &[String], index: usize, default: u64) -> Result<u64, String> {
    args.get(index)
        .map(|value| {
            value
                .parse()
                .map_err(|error| format!("invalid argument {index}: {error}"))
        })
        .unwrap_or(Ok(default))
}

fn arg_f32(args: &[String], index: usize, default: f32) -> Result<f32, String> {
    args.get(index)
        .map(|value| {
            value
                .parse()
                .map_err(|error| format!("invalid argument {index}: {error}"))
        })
        .unwrap_or(Ok(default))
}
