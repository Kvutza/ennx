use std::error::Error;
use std::io;
use std::time::Instant;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig};
use ennx_cuda::{
    Ask as ResidentAsk, Leaf as ResidentLeaf, MAX_HISTORY, Tile as ResidentTile, TrialEngine,
};
use ennx_cuda_kernels::trials;

type AppResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

#[derive(Debug, Clone, Copy)]
struct Leaf {
    offset: usize,
    length: usize,
    bits: u8,
    scale: f32,
    radius: f32,
}

impl Leaf {
    fn new(offset: usize, length: usize, bits: u8, scale: f32, radius: f32) -> AppResult<Self> {
        if length == 0 {
            return Err(io::Error::other("leaf length must be positive").into());
        }
        if bits != 4 && bits != 8 {
            return Err(io::Error::other("leaf bits must be 4 or 8").into());
        }
        if !scale.is_finite() || scale <= 0.0 || !radius.is_finite() || radius <= 0.0 {
            return Err(io::Error::other("leaf scale and radius must be positive").into());
        }
        Ok(Self {
            offset,
            length,
            bits,
            scale,
            radius,
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct LaunchLeaf {
    byte_offset: u32,
    element_offset: u32,
    length: u32,
    bits: u32,
    whole: u32,
    threshold: u32,
}

impl LaunchLeaf {
    fn new(leaf: Leaf, byte_offset: usize, length: f32) -> AppResult<Self> {
        let max_code = (1_u32 << leaf.bits) - 1;
        let amplitude = (length * leaf.radius / leaf.scale).clamp(0.0, max_code as f32);
        let whole = amplitude.floor() as u32;
        let threshold = if whole == max_code {
            0
        } else {
            ((amplitude - whole as f32) * (u32::MAX as f32)) as u32
        };
        Ok(Self {
            byte_offset: byte_offset.try_into()?,
            element_offset: leaf.offset.try_into()?,
            length: leaf.length.try_into()?,
            bits: u32::from(leaf.bits),
            whole,
            threshold,
        })
    }

    fn work_items(self) -> u32 {
        if self.bits == 4 {
            self.length.div_ceil(2)
        } else {
            self.length
        }
    }

    fn row_bytes(self) -> usize {
        self.work_items() as usize
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("ERROR: {error}");
        std::process::exit(1);
    }
}

fn run() -> AppResult<()> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str).unwrap_or("parity") {
        "parity" => parity(),
        "resident" => resident_parity(),
        "bench" => benchmark(&args[2..]),
        "trial-bench" => trial_benchmark(&args[2..]),
        command => Err(io::Error::other(format!(
            "unknown command {command:?}; expected parity, resident, bench, or trial-bench"
        ))
        .into()),
    }
}

fn resident_parity() -> AppResult<()> {
    let leaves = vec![
        Leaf::new(0, 257, 4, 0.125, 0.75)?,
        Leaf::new(257, 259, 8, 0.03125, 0.5)?,
    ];
    let base = make_base(row_bytes(&leaves));
    let length = 0.8;
    let steps = resident_steps(&leaves, length)?;
    let tiles = steps
        .iter()
        .enumerate()
        .map(|(leaf, step)| ResidentTile {
            leaf: leaf as u32,
            start: 0,
            length: step.length,
            pad: 0,
        })
        .collect::<Vec<_>>();
    let mut engine = TrialEngine::new(&base, &steps, &tiles, 8)
        .map_err(|error| io::Error::other(format!("resident init: {error}")))?;

    let history_seeds = [11_u64, 0x0123_4567_89ab_cdef, 42_u64];
    let mut history_rows = vec![base.clone()];
    for (slot, seed) in history_seeds.into_iter().enumerate() {
        engine
            .materialize(0, slot + 1, seed, &steps)
            .map_err(|error| io::Error::other(format!("resident history write: {error}")))?;
        let actual = engine
            .read(slot + 1)
            .map_err(|error| io::Error::other(format!("resident history read: {error}")))?;
        let expected = expected_row(&base, &leaves, length, seed)?;
        compare_rows(&expected, &actual, seed, length)?;
        history_rows.push(expected);
    }

    let history_slots = [0_u32, 1, 2, 3];
    let outcomes = [-0.75_f32, 1.25, 0.5, 2.0];
    let seeds = [3_u64, 17, 0xdead_beef_cafe_babe, u64::MAX - 9, 99, 1001];
    let draws = [0.1_f32, -0.5, 0.8, -0.2, 0.0, 0.3];

    for acquisition in [0_u32, 1, 2] {
        let config = ResidentAsk {
            neighbors: 3,
            acquisition,
            epistemic_scale: 0.7,
            aleatoric_scale: 0.05,
            y_scale: 1.0,
            beta: 1.2,
        };
        let expected_scores = seeds
            .iter()
            .enumerate()
            .map(|(i, &seed)| {
                cpu_resident_score(
                    &base,
                    &history_rows,
                    &outcomes,
                    &leaves,
                    &steps,
                    seed,
                    draws[i],
                    config,
                )
            })
            .collect::<Vec<_>>();
        let expected_choice = expected_scores
            .iter()
            .enumerate()
            .max_by(|(left_index, left), (right_index, right)| {
                left.total_cmp(right)
                    .then_with(|| right_index.cmp(left_index))
            })
            .map(|(index, _)| index)
            .expect("resident parity has candidates");
        let (choice, scores) = engine
            .ask_with_scores(
                0,
                &history_slots,
                &outcomes,
                4,
                &seeds,
                &draws,
                &steps,
                config,
                true,
            )
            .map_err(|error| io::Error::other(format!("resident ask: {error}")))?;
        if choice != expected_choice {
            return Err(io::Error::other(format!(
                "resident choice mismatch for acq={acquisition}: expected {expected_choice}, got {choice}; expected_scores={expected_scores:?}, scores={scores:?}"
            ))
            .into());
        }
        for (index, (&expected, &actual)) in expected_scores.iter().zip(&scores).enumerate() {
            let tolerance = 2.0e-5 * expected.abs().max(1.0);
            if (expected - actual).abs() > tolerance {
                return Err(io::Error::other(format!(
                    "resident score {index} mismatch for acq={acquisition}: expected {expected}, got {actual}, tolerance={tolerance}"
                ))
                .into());
            }
        }
        let actual = engine
            .read(4)
            .map_err(|error| io::Error::other(format!("resident trial read: {error}")))?;
        let expected = expected_row(&base, &leaves, length, seeds[choice])?;
        compare_rows(&expected, &actual, seeds[choice], length)?;
    }

    println!(
        "RESIDENT ok=true target=sm_75 history={} candidates={} choice=passed",
        history_rows.len(),
        seeds.len()
    );
    Ok(())
}

fn resident_steps(leaves: &[Leaf], length: f32) -> AppResult<Vec<ResidentLeaf>> {
    let mut byte_offset = 0;
    leaves
        .iter()
        .map(|&leaf| {
            let step = LaunchLeaf::new(leaf, byte_offset, length)?;
            byte_offset += step.row_bytes();
            Ok(ResidentLeaf {
                byte_offset: step.byte_offset,
                element_offset: step.element_offset,
                length: step.length,
                bits: step.bits,
                encoding: if step.bits == 4 { 0 } else { 1 },
                scale: leaf.scale,
                weight: 1.0,
                whole: step.whole,
                threshold: step.threshold,
            })
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn cpu_resident_score(
    base: &[u8],
    history_rows: &[Vec<u8>],
    outcomes: &[f32],
    leaves: &[Leaf],
    steps: &[ResidentLeaf],
    seed: u64,
    draw: f32,
    config: ResidentAsk,
) -> f32 {
    let mut nearest = vec![(f32::INFINITY, 0_usize); config.neighbors];
    for (history_index, observation) in history_rows.iter().enumerate() {
        let distance = cpu_resident_distance(base, observation, leaves, steps, seed);
        let insert_at = nearest.iter().position(|&(other, other_index)| {
            distance < other || (distance == other && history_index < other_index)
        });
        if let Some(insert_at) = insert_at {
            nearest.insert(insert_at, (distance, history_index));
            nearest.pop();
        }
    }
    let mut weight_sum = 0.0_f32;
    let mut weighted_value = 0.0_f32;
    for &(distance, history_index) in &nearest {
        let variance = 1.0e-9 + config.epistemic_scale * distance + config.aleatoric_scale;
        let weight = 1.0 / variance.max(1.0e-12);
        weight_sum += weight;
        weighted_value += weight * outcomes[history_index];
    }
    let mean = weighted_value / weight_sum.max(1.0e-12);
    let se = (1.0 / weight_sum.max(1.0e-12)).sqrt() * config.y_scale;
    match config.acquisition {
        1 => mean + se * draw,
        2 => mean + se,
        _ => mean + config.beta * se,
    }
}

fn cpu_resident_distance(
    base: &[u8],
    observation: &[u8],
    leaves: &[Leaf],
    steps: &[ResidentLeaf],
    seed: u64,
) -> f32 {
    let mut distance = 0.0_f32;
    for (&leaf, &step) in leaves.iter().zip(steps) {
        for element in 0..leaf.length {
            let byte =
                step.byte_offset as usize + if leaf.bits == 4 { element / 2 } else { element };
            let shift = if leaf.bits == 4 { (element & 1) * 4 } else { 0 };
            let mask = if leaf.bits == 4 { 0x0f } else { 0xff };
            let code = u32::from((base[byte] >> shift) & mask);
            let candidate = cpu_perturb(
                code,
                seed,
                leaf.offset as u32 + element as u32,
                step.bits,
                step.whole,
                step.threshold,
            );
            let observed = u32::from((observation[byte] >> shift) & mask);
            let delta = (candidate as f32 - observed as f32) * step.scale;
            distance = delta.mul_add(delta * step.weight, distance);
        }
    }
    distance
}

fn parity() -> AppResult<()> {
    let client = tracy_client::Client::start();
    let _zone = client
        .clone()
        .span(tracy_client::span_location!("ennx.cuda.parity"), 0);
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    // SAFETY: the generated bindings load the matching embedded kernel artifact.
    let module = unsafe { trials::load(&context) }?;
    let layouts = [
        vec![Leaf::new(0, 257, 4, 0.125, 0.75)?],
        vec![Leaf::new(0, 259, 8, 0.03125, 0.5)?],
        vec![
            Leaf::new(0, 257, 4, 0.125, 0.75)?,
            Leaf::new(257, 259, 8, 0.03125, 0.5)?,
        ],
    ];
    let mut cases = 0;
    for leaves in layouts {
        let base = make_base(row_bytes(&leaves));
        for length in [0.0, 0.8, 8.0] {
            for seed in [0_u64, 1, 0x0123_4567_89ab_cdef, u64::MAX] {
                check_case(&context, &stream, &module, &base, &leaves, length, seed)?;
                cases += 1;
            }
        }
    }
    println!("PARITY ok=true cases={cases} target=sm_75");
    Ok(())
}

fn benchmark(args: &[String]) -> AppResult<()> {
    let elements = parse_arg(args, 0, 16 * 1024 * 1024, "elements")?;
    let iterations = parse_arg(args, 1, 100, "iterations")?;
    let bits: u8 = parse_arg(args, 2, 4, "bits")?;
    if iterations == 0 {
        return Err(io::Error::other("iterations must be positive").into());
    }
    if bits != 4 && bits != 8 {
        return Err(io::Error::other("bits must be 4 or 8").into());
    }

    let scale = if bits == 4 { 0.125 } else { 0.03125 };
    let leaves = vec![Leaf::new(0, elements, bits, scale, 0.75)?];
    let base = make_base(row_bytes(&leaves));
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    // SAFETY: the generated bindings load the matching embedded kernel artifact.
    let module = unsafe { trials::load(&context) }?;
    let base_device = DeviceBuffer::from_host(&stream, &base)?;
    let mut output_device = DeviceBuffer::<u8>::zeroed(&stream, base.len())?;
    let length = 0.8;

    for seed in 0..5 {
        launch_materialize(
            &module,
            &stream,
            &base_device,
            &mut output_device,
            &leaves,
            length,
            seed,
        )?;
    }
    stream.synchronize()?;

    let client = tracy_client::Client::start();
    let _zone = client.clone().span(
        tracy_client::span_location!("ennx.cuda.materialize.bench"),
        0,
    );
    let event_flags = cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT;
    let start = stream.record_event(Some(event_flags))?;
    for iteration in 0..iterations {
        launch_materialize(
            &module,
            &stream,
            &base_device,
            &mut output_device,
            &leaves,
            length,
            10_000 + iteration as u64,
        )?;
    }
    let end = stream.record_event(Some(event_flags))?;
    let total_ms = f64::from(start.elapsed_ms(&end)?);
    let output = output_device.to_host_vec(&stream)?;
    let final_seed = 10_000 + iterations as u64 - 1;
    let expected = expected_row(&base, &leaves, length, final_seed)?;
    compare_rows(&expected, &output, final_seed, length)?;

    let kernel_ms = total_ms / iterations as f64;
    let bytes = (base.len() as f64) * 2.0 * iterations as f64;
    let gib_s = bytes / (total_ms / 1_000.0) / (1024.0 * 1024.0 * 1024.0);
    println!(
        "BENCH ok=true target=sm_75 elements={elements} bits={bits} iterations={iterations} \
         row_bytes={} kernel_ms={kernel_ms:.6} gib_s={gib_s:.3}",
        base.len()
    );
    Ok(())
}

fn trial_benchmark(args: &[String]) -> AppResult<()> {
    let candidates: usize = parse_arg(args, 0, 1_024, "candidates")?;
    let history: usize = parse_arg(args, 1, 32, "history")?;
    let elements: usize = parse_arg(args, 2, 8_192, "elements")?;
    let iterations: usize = parse_arg(args, 3, 50, "iterations")?;
    if candidates == 0 || elements == 0 || iterations == 0 {
        return Err(
            io::Error::other("candidates, elements, and iterations must be positive").into(),
        );
    }
    if history == 0 || history > MAX_HISTORY {
        return Err(io::Error::other(format!("history must be in 1..={MAX_HISTORY}")).into());
    }

    let leaves = vec![Leaf::new(0, elements, 4, 0.125, 0.75)?];
    let steps = resident_steps(&leaves, 0.8)?;
    let tiles = steps
        .iter()
        .enumerate()
        .map(|(leaf, step)| ResidentTile {
            leaf: leaf as u32,
            start: 0,
            length: step.length,
            pad: 0,
        })
        .collect::<Vec<_>>();
    let base = make_base(row_bytes(&leaves));
    let trial_slot = history;
    let mut engine = TrialEngine::new(&base, &steps, &tiles, history + 1)
        .map_err(|error| io::Error::other(format!("trial benchmark init: {error}")))?;
    for slot in 1..history {
        engine
            .materialize(0, slot, 0x1_0000 + slot as u64, &steps)
            .map_err(|error| io::Error::other(format!("trial benchmark history: {error}")))?;
    }

    let history_slots = (0..history as u32).collect::<Vec<_>>();
    let outcomes = (0..history)
        .map(|index| ((index.wrapping_mul(37) % 101) as f32 - 50.0) / 25.0)
        .collect::<Vec<_>>();
    let seeds = (0..candidates)
        .map(|index| 0x9e37_79b9_7f4a_7c15_u64.wrapping_mul(index as u64 + 1))
        .collect::<Vec<_>>();
    let draws = (0..candidates)
        .map(|index| ((index.wrapping_mul(29) % 97) as f32 - 48.0) / 24.0)
        .collect::<Vec<_>>();
    let config = ResidentAsk {
        neighbors: history.min(8),
        acquisition: 0,
        epistemic_scale: 0.7,
        aleatoric_scale: 0.05,
        y_scale: 1.0,
        beta: 1.2,
    };
    engine.set_profiling(true);

    for _ in 0..3 {
        engine
            .ask(
                0,
                &history_slots,
                &outcomes,
                trial_slot,
                &seeds,
                &draws,
                &steps,
                config,
                true,
            )
            .map_err(|error| io::Error::other(format!("trial benchmark warmup: {error}")))?;
    }

    let started = Instant::now();
    let mut score_ms = 0.0_f64;
    let mut pick_ms = 0.0_f64;
    let mut materialize_ms = 0.0_f64;
    let mut gpu_ms = 0.0_f64;
    for _ in 0..iterations {
        let (choice, score) = engine
            .ask(
                0,
                &history_slots,
                &outcomes,
                trial_slot,
                &seeds,
                &draws,
                &steps,
                config,
                true,
            )
            .map_err(|error| io::Error::other(format!("trial benchmark ask: {error}")))?;
        if choice >= candidates || !score.is_finite() {
            return Err(io::Error::other(format!(
                "trial benchmark returned choice={choice}, score={score}"
            ))
            .into());
        }
        let profile = engine
            .last_profile()
            .ok_or_else(|| io::Error::other("trial benchmark profile was not recorded"))?;
        score_ms += f64::from(profile.score_ms);
        pick_ms += f64::from(profile.pick_ms);
        materialize_ms += f64::from(profile.materialize_ms);
        gpu_ms += f64::from(profile.total_ms);
    }
    let wall_seconds = started.elapsed().as_secs_f64();
    let count = iterations as f64;
    let asks_s = count / wall_seconds;
    let candidates_s = candidates as f64 * asks_s;
    println!(
        "TRIAL_BENCH ok=true target=sm_75 candidates={candidates} history={history} \
         elements={elements} iterations={iterations} score_ms={:.6} pick_ms={:.6} \
         materialize_ms={:.6} gpu_ms={:.6} wall_ms={:.6} asks_s={asks_s:.3} \
         candidates_s={candidates_s:.3}",
        score_ms / count,
        pick_ms / count,
        materialize_ms / count,
        gpu_ms / count,
        wall_seconds * 1_000.0 / count,
    );
    Ok(())
}

fn check_case(
    context: &CudaContext,
    stream: &CudaStream,
    module: &trials::LoadedModule,
    base: &[u8],
    leaves: &[Leaf],
    length: f32,
    seed: u64,
) -> AppResult<()> {
    let base_device = DeviceBuffer::from_host(stream, base)?;
    let mut output_device = DeviceBuffer::<u8>::zeroed(stream, base.len())?;
    launch_materialize(
        module,
        stream,
        &base_device,
        &mut output_device,
        leaves,
        length,
        seed,
    )?;
    let output = output_device.to_host_vec(stream)?;
    let expected = expected_row(base, leaves, length, seed)?;
    compare_rows(&expected, &output, seed, length)?;
    context.check_err()?;
    Ok(())
}

fn launch_materialize(
    module: &trials::LoadedModule,
    stream: &CudaStream,
    base: &DeviceBuffer<u8>,
    output: &mut DeviceBuffer<u8>,
    leaves: &[Leaf],
    length: f32,
    seed: u64,
) -> AppResult<()> {
    let mut byte_offset = 0;
    for &leaf in leaves {
        let launch = LaunchLeaf::new(leaf, byte_offset, length)?;
        // SAFETY: there is one thread per leaf byte, leaf byte ranges do not overlap,
        // and the input and output buffers contain the complete encoded row.
        unsafe {
            module.materialize(
                stream,
                LaunchConfig::for_num_elems(launch.work_items()),
                base,
                output,
                launch.byte_offset,
                launch.element_offset,
                launch.length,
                launch.bits,
                seed as u32,
                (seed >> 32) as u32,
                launch.whole,
                launch.threshold,
            )
        }?;
        byte_offset += launch.row_bytes();
    }
    Ok(())
}

fn expected_row(base: &[u8], leaves: &[Leaf], length: f32, seed: u64) -> AppResult<Vec<u8>> {
    let mut row = vec![0_u8; base.len()];
    let mut byte_offset = 0;
    for &leaf in leaves {
        let step = LaunchLeaf::new(leaf, byte_offset, length)?;
        for element in 0..leaf.length {
            let byte = byte_offset + if leaf.bits == 4 { element / 2 } else { element };
            let shift = if leaf.bits == 4 { (element & 1) * 4 } else { 0 };
            let mask = if leaf.bits == 4 { 0x0f } else { 0xff };
            let code = u32::from((base[byte] >> shift) & mask);
            let value = cpu_perturb(
                code,
                seed,
                leaf.offset as u32 + element as u32,
                step.bits,
                step.whole,
                step.threshold,
            );
            row[byte] |= (value as u8) << shift;
        }
        byte_offset += step.row_bytes();
    }
    Ok(row)
}

fn cpu_perturb(code: u32, seed: u64, element: u32, bits: u32, whole: u32, threshold: u32) -> u32 {
    let random = cpu_hash(seed, element);
    let amount = whole + u32::from((random >> 1) < (threshold >> 1));
    if amount == 0 {
        return code;
    }
    let max_code = (1_u32 << bits) - 1;
    if random & 1 == 0 {
        if code >= amount {
            code - amount
        } else {
            (code + amount).min(max_code)
        }
    } else if code + amount <= max_code {
        code + amount
    } else {
        code.saturating_sub(amount)
    }
}

fn cpu_hash(seed: u64, element: u32) -> u32 {
    let mut value = (seed as u32) ^ element.wrapping_mul(0x9e37_79b9);
    value ^= value >> 16;
    value = value.wrapping_mul(0x7feb_352d);
    value ^= (seed >> 32) as u32;
    value = value.wrapping_mul(0x846c_a68b);
    value ^ (value >> 15)
}

fn compare_rows(expected: &[u8], actual: &[u8], seed: u64, length: f32) -> AppResult<()> {
    if expected == actual {
        return Ok(());
    }
    let mismatch = expected
        .iter()
        .zip(actual)
        .position(|(expected, actual)| expected != actual)
        .unwrap_or(expected.len().min(actual.len()));
    Err(io::Error::other(format!(
        "CUDA parity failed at byte {mismatch} for seed={seed} length={length}: \
         expected={:?} actual={:?}",
        expected.get(mismatch),
        actual.get(mismatch)
    ))
    .into())
}

fn row_bytes(leaves: &[Leaf]) -> usize {
    leaves
        .iter()
        .map(|leaf| {
            if leaf.bits == 4 {
                leaf.length.div_ceil(2)
            } else {
                leaf.length
            }
        })
        .sum()
}

fn make_base(bytes: usize) -> Vec<u8> {
    (0..bytes)
        .map(|index| (index.wrapping_mul(37).wrapping_add(11) & 0xff) as u8)
        .collect()
}

fn parse_arg<T>(args: &[String], index: usize, default: T, name: &str) -> AppResult<T>
where
    T: std::str::FromStr,
    T::Err: Error + Send + Sync + 'static,
{
    args.get(index)
        .map(|value| value.parse().map_err(Into::into))
        .unwrap_or(Ok(default))
        .map_err(|error: Box<dyn Error + Send + Sync>| {
            io::Error::other(format!("invalid {name}: {error}")).into()
        })
}
