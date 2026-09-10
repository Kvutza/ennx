use std::time::{Duration, Instant};

use ennx::experimental::apple_gpu_info;
use ennx::{ENNIndex, IndexDriver};
use ndarray::{Array1, Array2};

fn main() -> Result<(), String> {
    let args: Vec<_> = std::env::args().collect();
    let rows = arg(&args, 1, 2_048)?;
    let queries = arg(&args, 2, 256)?;
    let dimensions = arg(&args, 3, 32)?;
    let neighbors = arg(&args, 4, 16)?;
    let rounds = arg(&args, 5, 10)?;

    let train = Array2::from_shape_fn((rows, dimensions), |(row, column)| {
        ((row * 37 + column * 17) % 1_009) as f64 / 1_009.0
    });
    let query = Array2::from_shape_fn((queries, dimensions), |(row, column)| {
        ((row * 29 + column * 11 + 3) % 1_013) as f64 / 1_013.0
    });
    let scale = Array1::ones(dimensions);

    let cold_start = Instant::now();
    let cold = index(&train, &scale, IndexDriver::Metal)?;
    let cold_setup = cold_start.elapsed();
    let _ = cold
        .search(&query.view(), neighbors as i32, false)
        .map_err(|error| error.to_string())?;

    let (metal_setup, metal_search) = measure(
        rounds,
        || index(&train, &scale, IndexDriver::Metal),
        |index| {
            index
                .search(&query.view(), neighbors as i32, false)
                .map(|_| ())
                .map_err(|error| error.to_string())
        },
    )?;
    let (cpu_setup, cpu_search) = measure(
        rounds,
        || index(&train, &scale, IndexDriver::Exact),
        |index| {
            index
                .search(&query.view(), neighbors as i32, false)
                .map(|_| ())
                .map_err(|error| error.to_string())
        },
    )?;

    let device = apple_gpu_info()?;
    println!(
        "device={:?} target={:?} rows={rows} queries={queries} dimensions={dimensions} \
         neighbors={neighbors} rounds={rounds}",
        device.name, device.target
    );
    println!("backend,setup_us,search_us");
    println!("metal_cold,{},-", cold_setup.as_micros());
    println!(
        "metal_warm,{},{}",
        metal_setup.as_micros(),
        metal_search.as_micros()
    );
    println!("cpu,{},{}", cpu_setup.as_micros(), cpu_search.as_micros());
    Ok(())
}

fn index(
    train: &Array2<f64>,
    scale: &Array1<f64>,
    driver: IndexDriver,
) -> Result<ENNIndex, String> {
    ENNIndex::new(
        train.to_owned(),
        train.ncols(),
        scale.to_owned(),
        false,
        driver,
    )
    .map_err(|error| error.to_string())
}

fn measure(
    rounds: usize,
    mut setup: impl FnMut() -> Result<ENNIndex, String>,
    mut run: impl FnMut(&ENNIndex) -> Result<(), String>,
) -> Result<(Duration, Duration), String> {
    if rounds == 0 {
        return Err("rounds must be positive".to_string());
    }
    let setup_start = Instant::now();
    let index = setup()?;
    let setup_time = setup_start.elapsed();
    run(&index)?;
    let mut run_times = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        let start = Instant::now();
        run(&index)?;
        run_times.push(start.elapsed());
    }
    run_times.sort_unstable();
    Ok((setup_time, run_times[rounds / 2]))
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
