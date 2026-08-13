use std::time::Instant;

use ennx::experimental::{ComputeBackend, WeightAsk, WeightLeaf, WeightSearch};

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let elements = arg(&args, 1, 16 * 1024 * 1024)?;
    let history = arg(&args, 2, 10)?;
    let candidates = arg(&args, 3, 8)?;
    let rounds = arg(&args, 4, 10)?;
    let regions = arg(&args, 6, 1)?;
    let backend = match args.get(5).map(String::as_str).unwrap_or("metal") {
        "cpu" => ComputeBackend::Cpu,
        "metal" => ComputeBackend::Metal,
        "agx" => ComputeBackend::Agx,
        "auto" => ComputeBackend::Auto,
        "opencl" => ComputeBackend::OpenCl,
        value => return Err(format!("unknown backend {value:?}")),
    };
    let row_bytes = elements.div_ceil(2);
    let base: Vec<u8> = (0..row_bytes)
        .map(|index| (index.wrapping_mul(37).wrapping_add(11) & 0xff) as u8)
        .collect();
    let leaves = vec![WeightLeaf::new(0, elements, 4, 0.125, 1.0, 0.25)?];
    let mut search = WeightSearch::new(&base, 0.0, leaves, history, backend)?;
    let ask = WeightAsk {
        length: 0.8,
        neighbors: history.min(10),
        beta: 1.0,
        ..WeightAsk::default()
    };
    for round in 1..history {
        let trial = search.ask(
            &[round as u64],
            WeightAsk {
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
        let config = WeightAsk {
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
