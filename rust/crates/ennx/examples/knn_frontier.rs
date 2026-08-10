use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::{Duration, Instant};

use ennx::experimental::{KnnIndex, KnnPlan};
use ennx::IndexDriver;
use ndarray::Array2;

struct Point {
    axis: String,
    rows: usize,
    queries: usize,
    dims: usize,
    k: usize,
}

fn main() -> Result<(), String> {
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() < 4 {
        return Err("usage: knn-frontier OUT ROUNDS POINT...".to_string());
    }
    let rounds = args[2]
        .parse::<usize>()
        .map_err(|error| format!("rounds: {error}"))?;
    if rounds == 0 {
        return Err("rounds must be positive".to_string());
    }
    let file = File::create(&args[1]).map_err(|error| error.to_string())?;
    let mut out = BufWriter::new(file);
    writeln!(
        out,
        "axis,backend,request,plan,rows,queries,dims,k,search_ms,gpu_ms,scan_ms,select_ms,reduce_ms,rank_match,recall_at_k,max_abs_error,distance_mib"
    )
    .map_err(|error| error.to_string())?;
    for spec in &args[3..] {
        bench(&mut out, &parse(spec)?, rounds)?;
    }
    Ok(())
}

fn bench(out: &mut impl Write, point: &Point, rounds: usize) -> Result<(), String> {
    let train = values(point.rows, point.dims, point.rows as u32);
    let query = values(point.queries, point.dims, point.queries as u32);
    let exact = KnnIndex::new(&train.view(), IndexDriver::Exact).map_err(|e| e.to_string())?;
    let (cpu_ms, cpu_dist, cpu_ids) = run(&exact, &query, point.k, rounds)?;
    writeln!(
        out,
        "{},exact,exact,{},{},{},{},{},{:.6},0,0,0,0,1,1,0,0",
        point.axis,
        exact.plan(),
        point.rows,
        point.queries,
        point.dims,
        point.k,
        cpu_ms.as_secs_f64() * 1_000.0
    )
    .map_err(|error| error.to_string())?;

    for (name, driver) in [("metal", IndexDriver::Metal), ("agx", IndexDriver::Agx)] {
        let plans: &[KnnPlan] = if point.k <= 16 {
            &[
                KnnPlan::Measured,
                KnnPlan::Split,
                KnnPlan::Fused,
                KnnPlan::Tree,
                KnnPlan::Tiled,
                KnnPlan::Simd,
                KnnPlan::Gram,
            ]
        } else {
            &[KnnPlan::Measured, KnnPlan::Split, KnnPlan::Wide]
        };
        let indices = plans
            .iter()
            .map(|&plan| KnnIndex::with_plan(&train.view(), driver, plan))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| error.to_string())?;
        for ((&request, index), (elapsed, dist, ids)) in plans
            .iter()
            .zip(&indices)
            .zip(run_many(&indices, &query, point.k, rounds)?)
        {
            let matches = cpu_ids
                .iter()
                .zip(ids.iter())
                .filter(|(left, right)| left == right)
                .count();
            let match_rate = matches as f64 / cpu_ids.len() as f64;
            let recall = recall(&cpu_ids, &ids);
            let max_error = cpu_dist
                .iter()
                .zip(dist.iter())
                .map(|(left, right)| (left - right).abs())
                .fold(0.0_f64, f64::max);
            let distance_mib = if index.plan() == "split" {
                (point.queries.next_power_of_two() * 1_024 * size_of::<f32>()) as f64
                    / (1_024 * 1_024) as f64
            } else {
                0.0
            };
            let profile = index.profile();
            let millis = |duration: Duration| duration.as_secs_f64() * 1_000.0;
            let (gpu_ms, scan_ms, select_ms, reduce_ms) = profile
                .map(|profile| {
                    (
                        millis(profile.gpu),
                        millis(profile.scan),
                        millis(profile.select),
                        millis(profile.reduce),
                    )
                })
                .unwrap_or_default();
            writeln!(
                out,
                "{},{name},{},{},{},{},{},{},{:.6},{gpu_ms:.6},{scan_ms:.6},{select_ms:.6},{reduce_ms:.6},{match_rate:.9},{recall:.9},{max_error:.9},{distance_mib:.6}",
                point.axis,
                plan_name(request),
                index.plan(),
                point.rows,
                point.queries,
                point.dims,
                point.k,
                elapsed.as_secs_f64() * 1_000.0
            )
            .map_err(|error| error.to_string())?;
        }
    }
    Ok(())
}

fn plan_name(plan: KnnPlan) -> &'static str {
    match plan {
        KnnPlan::Measured => "measured",
        KnnPlan::Split => "split",
        KnnPlan::Fused => "fused",
        KnnPlan::Tree => "tree",
        KnnPlan::Tiled => "tiled",
        KnnPlan::Simd => "simd",
        KnnPlan::Gram => "gram",
        KnnPlan::Wide => "wide",
    }
}

fn run_many(
    indices: &[KnnIndex],
    query: &Array2<f64>,
    k: usize,
    rounds: usize,
) -> Result<Vec<(Duration, Array2<f64>, Array2<i64>)>, String> {
    for index in indices {
        index.search(&query.view(), k).map_err(|e| e.to_string())?;
    }
    let mut times = (0..indices.len())
        .map(|_| Vec::with_capacity(rounds))
        .collect::<Vec<_>>();
    let mut results = (0..indices.len()).map(|_| None).collect::<Vec<_>>();
    for round in 0..rounds {
        for offset in 0..indices.len() {
            let index = (round + offset) % indices.len();
            let start = Instant::now();
            results[index] = Some(
                indices[index]
                    .search(&query.view(), k)
                    .map_err(|e| e.to_string())?,
            );
            times[index].push(start.elapsed());
        }
    }
    times
        .into_iter()
        .zip(results)
        .map(|(mut times, result)| {
            times.sort_unstable();
            let (distances, ids) = result.expect("positive rounds");
            Ok((times[times.len() / 2], distances, ids))
        })
        .collect()
}

fn run(
    index: &KnnIndex,
    query: &Array2<f64>,
    k: usize,
    rounds: usize,
) -> Result<(Duration, Array2<f64>, Array2<i64>), String> {
    index.search(&query.view(), k).map_err(|e| e.to_string())?;
    let mut times = Vec::with_capacity(rounds);
    let mut result = None;
    for _ in 0..rounds {
        let start = Instant::now();
        result = Some(index.search(&query.view(), k).map_err(|e| e.to_string())?);
        times.push(start.elapsed());
    }
    times.sort_unstable();
    let (distances, ids) = result.expect("positive rounds");
    Ok((times[times.len() / 2], distances, ids))
}

fn values(rows: usize, cols: usize, mut state: u32) -> Array2<f64> {
    Array2::from_shape_fn((rows, cols), |_| {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((state >> 8) as f32 / (1_u32 << 24) as f32) as f64
    })
}

fn parse(spec: &str) -> Result<Point, String> {
    let fields = spec.split(':').collect::<Vec<_>>();
    if fields.len() != 5 {
        return Err("point must be AXIS:ROWS:QUERIES:DIMS:K".to_string());
    }
    let number = |index: usize| {
        fields[index]
            .parse::<usize>()
            .map_err(|error| format!("{spec}: {error}"))
    };
    let point = Point {
        axis: fields[0].to_string(),
        rows: number(1)?,
        queries: number(2)?,
        dims: number(3)?,
        k: number(4)?,
    };
    if point.axis.is_empty()
        || point.rows == 0
        || point.queries == 0
        || point.dims == 0
        || point.k == 0
        || point.k > point.rows
        || point.k > 2_048
    {
        return Err(format!("invalid point {spec}"));
    }
    let distance_bytes = point
        .rows
        .checked_mul(point.queries)
        .and_then(|count| count.checked_mul(size_of::<f32>()))
        .ok_or_else(|| format!("distance matrix overflows: {spec}"))?;
    if distance_bytes > 256 * 1_024 * 1_024 {
        return Err(format!("distance matrix exceeds 256 MiB: {spec}"));
    }
    Ok(point)
}

fn recall(expected: &Array2<i64>, actual: &Array2<i64>) -> f64 {
    let mut matches = 0;
    for (left, right) in expected.rows().into_iter().zip(actual.rows()) {
        let mut left = left.to_vec();
        let mut right = right.to_vec();
        left.sort_unstable();
        right.sort_unstable();
        let mut i = 0;
        let mut j = 0;
        while i < left.len() && j < right.len() {
            match left[i].cmp(&right[j]) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    matches += 1;
                    i += 1;
                    j += 1;
                }
            }
        }
    }
    matches as f64 / expected.len() as f64
}
