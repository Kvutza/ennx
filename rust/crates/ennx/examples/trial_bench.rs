use std::time::Instant;

use ennx::experimental::{ComputeDevice, PackedLeaf, PackedSearch, SearchConfig};

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let elements = arg(&args, 1, 16 * 1024 * 1024)?;
    let history = arg(&args, 2, 10)?;
    let candidates = arg(&args, 3, 8)?;
    let rounds = arg(&args, 4, 10)?;
    let regions = arg(&args, 6, 1)?;
    let device = match args.get(5).map(String::as_str).unwrap_or("metal") {
        "cpu" => ComputeDevice::Cpu,
        "metal" => ComputeDevice::Metal,
        "agx" => ComputeDevice::Agx,
        "auto" => ComputeDevice::Auto,
        "opencl" => ComputeDevice::OpenCl,
        "cuda" => ComputeDevice::Cuda,
        value => return Err(format!("unknown device {value:?}")),
    };
    let row_bytes = elements.div_ceil(2);
    let base: Vec<u8> = (0..row_bytes)
        .map(|index| (index.wrapping_mul(37).wrapping_add(11) & 0xff) as u8)
        .collect();
    let leaves = vec![PackedLeaf::new(0, elements, 4, 0.125, 1.0, 0.25)?];
    let mut search = PackedSearch::new(&base, 0.0, leaves, history, device)?;
    let ask = SearchConfig {
        length: 0.8,
        neighbors: history.min(10),
        beta: 1.0,
        ..SearchConfig::default()
    };
    for round in 1..history {
        let trial = search.ask(
            &[round as u64],
            SearchConfig {
                neighbors: round.min(10),
                ..ask
            },
        )?;
        search.tell(trial, round as f32, true)?;
    }

    let mut times = Vec::with_capacity(rounds);
    for round in 0..rounds {
        let total = candidates
            .checked_mul(regions)
            .ok_or("candidate count overflow")?;
        let seeds: Vec<u64> = (0..total)
            .map(|candidate| 10_000 + (round * total + candidate) as u64)
            .collect();
        let start = Instant::now();
        let config = SearchConfig {
            seed: round as u64,
            ..ask
        };
        let trial = if regions == 1 {
            Some(search.ask(&seeds, config)?)
        } else {
            search.ask_multi_tr(regions, candidates, &seeds, config)?;
            None
        };
        times.push(start.elapsed().as_secs_f64());
        if let Some(trial) = trial {
            search.tell(trial, round as f32, round % 2 == 0)?;
        }
    }
    times.sort_by(f64::total_cmp);
    let median = times[times.len() / 2];
    let min = times[0];
    println!(
        "elements={elements} row_bytes={row_bytes} history={history} regions={regions} \
         candidates_per_region={candidates} rounds={rounds} min_ms={:.3} median_ms={:.3}",
        min * 1_000.0,
        median * 1_000.0
    );
    Ok(())
}

fn arg(args: &[String], index: usize, default: usize) -> Result<usize, String> {
    args.get(index)
        .map(|value| {
            value
                .parse()
                .map_err(|error| format!("invalid argument {index}: {error}"))
        })
        .unwrap_or(Ok(default))
}
