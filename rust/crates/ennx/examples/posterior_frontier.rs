use std::time::{Duration, Instant};

use ennx::{ENNParams, IndexDriver, PosteriorComputation, PosteriorFlags, ENN};
use ndarray::Array2;

fn main() -> Result<(), String> {
    let args = std::env::args().collect::<Vec<_>>();
    let rows = arg(&args, 1, 8_192)?;
    let queries = arg(&args, 2, 1_024)?;
    let dimensions = arg(&args, 3, 32)?;
    let metrics = arg(&args, 4, 32)?;
    let neighbors = arg(&args, 5, 16)?;
    let rounds = arg(&args, 6, 15)?;
    let train_x = Array2::from_shape_fn((rows, dimensions), |(i, j)| {
        ((i * 37 + j * 17) % 1_009) as f64 / 1_009.0
    });
    let train_y = Array2::from_shape_fn((rows, metrics), |(i, j)| {
        ((i * 11 + j * 29) % 1_013) as f64 / 173.0
    });
    let query = Array2::from_shape_fn((queries, dimensions), |(i, j)| {
        ((i * 29 + j * 11 + 3) % 1_013) as f64 / 1_013.0
    });
    let params = ENNParams::new(neighbors as i32, 0.7, 0.13).map_err(|e| e.to_string())?;
    let flags = PosteriorFlags::new().tie_neighbors(false);
    println!(
        "rows={rows} queries={queries} dimensions={dimensions} metrics={metrics} \
         neighbors={neighbors} rounds={rounds}"
    );
    println!("backend,posterior_us");
    for driver in [IndexDriver::Metal, IndexDriver::Exact] {
        let model = ENN::new(train_x.clone(), train_y.clone(), None, false, driver)
            .map_err(|e| e.to_string())?;
        let run = || {
            model
                .posterior(&query.view(), &params, &flags)
                .map(|_| ())
                .map_err(|e| e.to_string())
        };
        run()?;
        let elapsed = median(rounds, run)?;
        println!("{driver:?},{}", elapsed.as_micros());
    }
    Ok(())
}

fn median(rounds: usize, mut run: impl FnMut() -> Result<(), String>) -> Result<Duration, String> {
    if rounds == 0 {
        return Err("rounds must be positive".to_string());
    }
    let mut samples = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        let start = Instant::now();
        run()?;
        samples.push(start.elapsed());
    }
    samples.sort_unstable();
    Ok(samples[rounds / 2])
}

fn arg(args: &[String], at: usize, default: usize) -> Result<usize, String> {
    args.get(at)
        .map(|value| {
            value
                .parse()
                .map_err(|error| format!("argument {at}: {error}"))
        })
        .unwrap_or(Ok(default))
}
