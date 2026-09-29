use std::ffi::c_void;
use std::fmt::Write as _;
use std::fs::OpenOptions;
use std::io::Write as _;
use std::process::Command;

use cuda_core::{CudaContext, CudaEvent, DeviceBuffer, simt::launch_kernel_on_stream};
use ptx_synth::{
    ProbeOp, ProbeShape, synthesize_turing_fused_rademacher, synthesize_turing_probe,
    synthesize_turing_vector_scale,
};

use crate::AppResult;

const THREADS: u32 = 256;
const RADIUS: f32 = 0.03125;
const SEED: u64 = 0x17c0_ffee_5eed_beef;
const SASS_BUNDLE: &str = "/tmp/ptx-synth.sass";
const CLOCK_OPERATIONS: u32 = 256;
const CLOCK_SAMPLES: usize = 31;

struct Compiled {
    cubin: String,
    sass: String,
    ptx_hash: u64,
    cubin_hash: u64,
    sass_hash: u64,
    registers: u32,
    spill_load_bytes: u32,
    spill_store_bytes: u32,
    instructions: u32,
}

struct Probe {
    op: ProbeOp,
    shape: ProbeShape,
    baseline: Compiled,
    kernel: Compiled,
    baseline_cycles: u64,
    total_cycles: u64,
}

fn positive(value: Option<&String>, name: &str, fallback: u32) -> AppResult<u32> {
    let parsed = value.map_or(Ok(fallback), |value| value.parse::<u32>())?;
    if parsed == 0 {
        return Err(format!("{name} must be positive").into());
    }
    Ok(parsed)
}

fn hash(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325_u64, |state, byte| {
        (state ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    })
}

fn before(text: &str, suffix: &[&str]) -> u32 {
    let words = text.split_whitespace().collect::<Vec<_>>();
    words
        .windows(suffix.len() + 1)
        .find(|window| window[1..] == *suffix)
        .and_then(|window| window[0].parse().ok())
        .unwrap_or(0)
}

fn compile(ptx: &str, name: &str) -> AppResult<Compiled> {
    let ptx_path = format!("/tmp/{name}.ptx");
    let cubin = format!("/tmp/{name}.cubin");
    std::fs::write(&ptx_path, ptx)?;
    let assembled = Command::new("ptxas")
        .args([
            "-arch=sm_75",
            "-O3",
            "--warn-on-spills",
            "-v",
            &ptx_path,
            "-o",
            &cubin,
        ])
        .output()?;
    if !assembled.status.success() {
        return Err(format!(
            "ptxas rejected {name}: {}",
            String::from_utf8_lossy(&assembled.stderr)
        )
        .into());
    }
    let log = String::from_utf8_lossy(&assembled.stderr);
    let disassembled = Command::new("nvdisasm").arg(&cubin).output()?;
    if !disassembled.status.success() {
        return Err(format!(
            "nvdisasm rejected {name}: {}",
            String::from_utf8_lossy(&disassembled.stderr)
        )
        .into());
    }
    let sass = String::from_utf8_lossy(&disassembled.stdout);
    let instructions = sass
        .lines()
        .filter(|line| line.contains("/*") && line.contains("*/") && line.contains(';'))
        .count() as u32;
    let mut bundle = OpenOptions::new()
        .create(true)
        .append(true)
        .open(SASS_BUNDLE)?;
    writeln!(bundle, "\n===== {name} =====")?;
    bundle.write_all(&disassembled.stdout)?;
    let cubin_bytes = std::fs::read(&cubin)?;
    Ok(Compiled {
        cubin,
        sass: sass.into_owned(),
        ptx_hash: hash(ptx.as_bytes()),
        cubin_hash: hash(&cubin_bytes),
        sass_hash: hash(disassembled.stdout.as_slice()),
        registers: before(&log, &["registers,"]),
        spill_load_bytes: before(&log, &["bytes", "spill", "loads"]),
        spill_store_bytes: before(&log, &["bytes", "spill", "stores,"]),
        instructions,
    })
}

unsafe fn clock_launch(
    function: &cuda_core::CudaFunction,
    stream: &cuda_core::CudaStream,
    output: &DeviceBuffer<u64>,
    slot: usize,
    rounds: u32,
) -> AppResult<()> {
    let mut output_ptr = output.cu_deviceptr() + (slot * 2 * size_of::<u64>()) as u64;
    let mut seed = 0x6a09_e667_u32.wrapping_add(slot as u32);
    let mut rounds = rounds;
    let mut parameters = [
        (&mut output_ptr as *mut u64).cast::<c_void>(),
        (&mut seed as *mut u32).cast::<c_void>(),
        (&mut rounds as *mut u32).cast::<c_void>(),
    ];
    unsafe {
        launch_kernel_on_stream(function, (1, 1, 1), (1, 1, 1), 0, stream, &mut parameters)?;
    }
    Ok(())
}

fn clock_cycles(
    stream: &cuda_core::CudaStream,
    function: &cuda_core::CudaFunction,
    rounds: u32,
) -> AppResult<u64> {
    let output = DeviceBuffer::<u64>::zeroed(stream, CLOCK_SAMPLES * 2)?;
    unsafe { clock_launch(function, stream, &output, 0, rounds)? };
    stream.synchronize()?;
    for slot in 0..CLOCK_SAMPLES {
        unsafe { clock_launch(function, stream, &output, slot, rounds)? };
    }
    stream.synchronize()?;
    let mut host = vec![0_u64; CLOCK_SAMPLES * 2];
    output.copy_to_host(stream, &mut host)?;
    stream.synchronize()?;
    let mut cycles = host.chunks_exact(2).map(|pair| pair[0]).collect::<Vec<_>>();
    cycles.sort_unstable();
    Ok(cycles[cycles.len() / 2])
}

fn calibrate(
    context: &std::sync::Arc<CudaContext>,
    stream: &cuda_core::CudaStream,
    op: ProbeOp,
    shape: ProbeShape,
) -> AppResult<Probe> {
    let stem = format!("ennx_turing_{}_{}", op.name(), shape.name());
    let baseline_name = format!("{stem}_baseline");
    let baseline_ptx = synthesize_turing_probe(&baseline_name, op, shape, false);
    let kernel_ptx = synthesize_turing_probe(&stem, op, shape, true);
    let baseline = compile(&baseline_ptx, &baseline_name)?;
    let kernel = compile(&kernel_ptx, &stem)?;
    let baseline_module = context.load_module_from_file(&baseline.cubin)?;
    let baseline_function = baseline_module.load_function(&baseline_name)?;
    let kernel_module = context.load_module_from_file(&kernel.cubin)?;
    let kernel_function = kernel_module.load_function(&stem)?;
    let rounds = CLOCK_OPERATIONS
        / match shape {
            ProbeShape::Chain => 1,
            ProbeShape::Ilp8 => 8,
        };
    let baseline_cycles = clock_cycles(stream, &baseline_function, rounds)?;
    let total_cycles = clock_cycles(stream, &kernel_function, rounds)?;
    Ok(Probe {
        op,
        shape,
        baseline,
        kernel,
        baseline_cycles,
        total_cycles,
    })
}

fn sass_mnemonic_count(code: &str, prefix: &str) -> u32 {
    code.lines()
        .filter_map(|line| {
            line.split_once("*/")
                .map(|(_, instruction)| instruction.trim())
        })
        .filter_map(|instruction| {
            let mut words = instruction.split_whitespace();
            let first = words.next()?;
            if first.starts_with('@') {
                words.next()
            } else {
                Some(first)
            }
        })
        .filter(|mnemonic| mnemonic.starts_with(prefix))
        .count() as u32
}

fn target_sass(probe: &Probe) -> u32 {
    let prefix = match probe.op {
        ProbeOp::Iadd => "IADD",
        ProbeOp::Imul => "IMAD",
        ProbeOp::Xor => "LOP3",
        ProbeOp::Shf => "SHF",
        ProbeOp::Fadd => "FADD",
        ProbeOp::Fmul => "FMUL",
    };
    sass_mnemonic_count(&probe.kernel.sass, prefix)
        .saturating_sub(sass_mnemonic_count(&probe.baseline.sass, prefix))
}

fn probe_valid(probe: &Probe) -> bool {
    probe.total_cycles >= probe.baseline_cycles
        && target_sass(probe) > 0
        && probe.kernel.spill_load_bytes == 0
        && probe.kernel.spill_store_bytes == 0
}

fn probes_json(probes: &[Probe]) -> String {
    let mut json = String::from("[");
    for (index, probe) in probes.iter().enumerate() {
        if index > 0 {
            json.push(',');
        }
        let net = probe.total_cycles.saturating_sub(probe.baseline_cycles);
        let target_sass = target_sass(probe);
        let valid = probe_valid(probe);
        write!(
            json,
            "{{\"op\":\"{}\",\"shape\":\"{}\",\"method\":\"runtime-loop-minus-matched-baseline\",\"ptx_operations\":{},\"samples\":{},\"baseline_cycles\":{},\"total_cycles\":{},\"net_cycles\":{},\"cycles_per_ptx_op\":{:.6},\"static_target_sass_instructions\":{},\"valid\":{},\"ptx_hash\":\"{:016x}\",\"cubin_hash\":\"{:016x}\",\"sass_hash\":\"{:016x}\",\"baseline_sass_hash\":\"{:016x}\",\"registers\":{},\"spill_load_bytes\":{},\"spill_store_bytes\":{},\"sass_instructions\":{}}}",
            probe.op.name(),
            probe.shape.name(),
            CLOCK_OPERATIONS,
            CLOCK_SAMPLES,
            probe.baseline_cycles,
            probe.total_cycles,
            net,
            net as f64 / f64::from(CLOCK_OPERATIONS),
            target_sass,
            valid,
            probe.kernel.ptx_hash,
            probe.kernel.cubin_hash,
            probe.kernel.sass_hash,
            probe.baseline.sass_hash,
            probe.kernel.registers,
            probe.kernel.spill_load_bytes,
            probe.kernel.spill_store_bytes,
            probe.kernel.instructions,
        )
        .unwrap();
    }
    json.push(']');
    json
}

fn blocks(elements: u32) -> u32 {
    elements.div_ceil(4).div_ceil(THREADS)
}

unsafe fn vector(
    function: &cuda_core::CudaFunction,
    stream: &cuda_core::CudaStream,
    input: &DeviceBuffer<f32>,
    output: &DeviceBuffer<f32>,
    scale: f32,
    elements: u32,
) -> AppResult<()> {
    let mut input_ptr = input.cu_deviceptr();
    let mut output_ptr = output.cu_deviceptr();
    let mut scale = scale;
    let mut elements = elements;
    let mut parameters = [
        (&mut input_ptr as *mut u64).cast::<c_void>(),
        (&mut output_ptr as *mut u64).cast::<c_void>(),
        (&mut scale as *mut f32).cast::<c_void>(),
        (&mut elements as *mut u32).cast::<c_void>(),
    ];
    unsafe {
        launch_kernel_on_stream(
            function,
            (blocks(elements), 1, 1),
            (THREADS, 1, 1),
            0,
            stream,
            &mut parameters,
        )?;
    }
    Ok(())
}

unsafe fn rademacher(
    function: &cuda_core::CudaFunction,
    stream: &cuda_core::CudaStream,
    input: &DeviceBuffer<f32>,
    output: &DeviceBuffer<f32>,
    radius: f32,
    seed: u64,
    elements: u32,
) -> AppResult<()> {
    let mut input_ptr = input.cu_deviceptr();
    let mut output_ptr = output.cu_deviceptr();
    let mut radius = radius;
    let mut seed = seed;
    let mut elements = elements;
    let mut parameters = [
        (&mut input_ptr as *mut u64).cast::<c_void>(),
        (&mut output_ptr as *mut u64).cast::<c_void>(),
        (&mut radius as *mut f32).cast::<c_void>(),
        (&mut seed as *mut u64).cast::<c_void>(),
        (&mut elements as *mut u32).cast::<c_void>(),
    ];
    unsafe {
        launch_kernel_on_stream(
            function,
            (blocks(elements), 1, 1),
            (THREADS, 1, 1),
            0,
            stream,
            &mut parameters,
        )?;
    }
    Ok(())
}

fn events(context: &std::sync::Arc<CudaContext>) -> AppResult<(CudaEvent, CudaEvent)> {
    let flags = cuda_core::simt::sys::CUevent_flags_enum_CU_EVENT_DEFAULT;
    Ok((
        context.new_event(Some(flags))?,
        context.new_event(Some(flags))?,
    ))
}

fn time_vector(
    context: &std::sync::Arc<CudaContext>,
    stream: &cuda_core::CudaStream,
    function: &cuda_core::CudaFunction,
    input: &DeviceBuffer<f32>,
    output: &DeviceBuffer<f32>,
    elements: u32,
    iterations: u32,
) -> AppResult<f32> {
    let (start, end) = events(context)?;
    start.record(stream)?;
    for _ in 0..iterations {
        unsafe { vector(function, stream, input, output, 1.75, elements)? };
    }
    end.record(stream)?;
    Ok(start.elapsed_ms(&end)? / iterations as f32)
}

fn time_rademacher(
    context: &std::sync::Arc<CudaContext>,
    stream: &cuda_core::CudaStream,
    function: &cuda_core::CudaFunction,
    input: &DeviceBuffer<f32>,
    output: &DeviceBuffer<f32>,
    elements: u32,
    iterations: u32,
) -> AppResult<f32> {
    let (start, end) = events(context)?;
    start.record(stream)?;
    for _ in 0..iterations {
        unsafe { rademacher(function, stream, input, output, RADIUS, SEED, elements)? };
    }
    end.record(stream)?;
    Ok(start.elapsed_ms(&end)? / iterations as f32)
}

fn step(index: u32) -> f32 {
    let seed_lo = SEED as u32;
    let seed_hi = (SEED >> 32) as u32;
    let mut value = seed_lo ^ index.wrapping_mul(0x9e37_79b9);
    value ^= value >> 16;
    value = value.wrapping_mul(0x7feb_352d);
    value ^= seed_hi;
    value = value.wrapping_mul(0x846c_a68b);
    value ^= value >> 15;
    if value & 1 == 0 { -RADIUS } else { RADIUS }
}

fn bandwidth(elements: u32, milliseconds: f32) -> f64 {
    (f64::from(elements) * 8.0 / 1.0e6) / f64::from(milliseconds)
}

pub fn run(args: &[String]) -> AppResult<()> {
    let elements = positive(args.first(), "elements", 1_048_576)?;
    let iterations = positive(args.get(1), "iterations", 100)?;
    if elements % 4 != 0 {
        return Err("elements must be divisible by four for the PTX hardware gate".into());
    }
    match std::fs::remove_file(SASS_BUNDLE) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }

    let vector_ptx = synthesize_turing_vector_scale("ennx_turing_vector_scale");
    let rademacher_ptx = synthesize_turing_fused_rademacher("ennx_turing_rademacher");
    let vector_code = compile(&vector_ptx, "ennx_turing_vector_scale")?;
    let rademacher_code = compile(&rademacher_ptx, "ennx_turing_rademacher")?;
    let context = CudaContext::new(0)?;
    let device = context.device_name()?;
    let (major, minor) = context.compute_capability()?;
    let stream = context.new_stream()?;
    let vector_module = context.load_module_from_file(&vector_code.cubin)?;
    let vector_function = vector_module.load_function("ennx_turing_vector_scale")?;
    let rademacher_module = context.load_module_from_file(&rademacher_code.cubin)?;
    let rademacher_function = rademacher_module.load_function("ennx_turing_rademacher")?;
    let host = (0..elements)
        .map(|index| (index as f32 - elements as f32 * 0.5) / elements as f32)
        .collect::<Vec<_>>();
    let input = DeviceBuffer::from_host(&stream, &host)?;
    let output = DeviceBuffer::<f32>::zeroed(&stream, elements as usize)?;

    unsafe { vector(&vector_function, &stream, &input, &output, 1.75, elements)? };
    stream.synchronize()?;
    let vector_ms = time_vector(
        &context,
        &stream,
        &vector_function,
        &input,
        &output,
        elements,
        iterations,
    )?;
    let mut observed = vec![0.0_f32; elements as usize];
    output.copy_to_host(&stream, &mut observed)?;
    let vector_error = host
        .iter()
        .zip(&observed)
        .map(|(input, output)| (input * 1.75 - output).abs())
        .fold(0.0_f32, f32::max);
    let vector_parity = vector_error <= f32::EPSILON;

    unsafe {
        rademacher(
            &rademacher_function,
            &stream,
            &input,
            &output,
            RADIUS,
            SEED,
            elements,
        )?
    };
    stream.synchronize()?;
    let rademacher_ms = time_rademacher(
        &context,
        &stream,
        &rademacher_function,
        &input,
        &output,
        elements,
        iterations,
    )?;
    output.copy_to_host(&stream, &mut observed)?;
    let mut rademacher_error = 0.0_f32;
    let mut mismatches = 0_u32;
    let mut positive = 0_u32;
    for (index, (&input, &output)) in host.iter().zip(&observed).enumerate() {
        let expected_step = step(index as u32);
        positive += u32::from(expected_step > 0.0);
        let error = (input + expected_step - output).abs();
        rademacher_error = rademacher_error.max(error);
        mismatches += u32::from(error > f32::EPSILON);
    }
    let rademacher_parity = mismatches == 0;
    let all_parity = vector_parity && rademacher_parity;
    let mut probes = Vec::with_capacity(12);
    for shape in [ProbeShape::Chain, ProbeShape::Ilp8] {
        for op in [
            ProbeOp::Iadd,
            ProbeOp::Imul,
            ProbeOp::Xor,
            ProbeOp::Shf,
            ProbeOp::Fadd,
            ProbeOp::Fmul,
        ] {
            probes.push(calibrate(&context, &stream, op, shape)?);
        }
    }
    let calibration = probes_json(&probes);
    let calibration_valid = probes.iter().all(probe_valid);
    println!(
        "{{\"schema\":\"ennx.ptx-synth.t4-probe.v6\",\"hardware\":\"{}\",\"compute_capability\":\"{}.{}\",\"elements\":{},\"iterations\":{},\"vector\":{{\"kernel\":\"ennx_turing_vector_scale\",\"ptx_hash\":\"{:016x}\",\"cubin_hash\":\"{:016x}\",\"sass_hash\":\"{:016x}\",\"registers\":{},\"spill_load_bytes\":{},\"spill_store_bytes\":{},\"sass_instructions\":{},\"kernel_ms\":{:.6},\"effective_gb_s\":{:.6},\"max_abs_error\":{},\"parity\":{}}},\"rademacher\":{{\"kernel\":\"ennx_turing_rademacher\",\"ptx_hash\":\"{:016x}\",\"cubin_hash\":\"{:016x}\",\"sass_hash\":\"{:016x}\",\"registers\":{},\"spill_load_bytes\":{},\"spill_store_bytes\":{},\"sass_instructions\":{},\"radius\":{},\"seed\":{},\"kernel_ms\":{:.6},\"effective_gb_s\":{:.6},\"max_abs_error\":{},\"mismatches\":{},\"positive_steps\":{},\"negative_steps\":{},\"parity\":{}}},\"calibration\":{},\"calibration_valid\":{},\"all_parity\":{}}}",
        device.replace('"', ""),
        major,
        minor,
        elements,
        iterations,
        vector_code.ptx_hash,
        vector_code.cubin_hash,
        vector_code.sass_hash,
        vector_code.registers,
        vector_code.spill_load_bytes,
        vector_code.spill_store_bytes,
        vector_code.instructions,
        vector_ms,
        bandwidth(elements, vector_ms),
        vector_error,
        vector_parity,
        rademacher_code.ptx_hash,
        rademacher_code.cubin_hash,
        rademacher_code.sass_hash,
        rademacher_code.registers,
        rademacher_code.spill_load_bytes,
        rademacher_code.spill_store_bytes,
        rademacher_code.instructions,
        RADIUS,
        SEED,
        rademacher_ms,
        bandwidth(elements, rademacher_ms),
        rademacher_error,
        mismatches,
        positive,
        elements - positive,
        rademacher_parity,
        calibration,
        calibration_valid,
        all_parity,
    );
    if all_parity && calibration_valid {
        Ok(())
    } else if !all_parity {
        Err("PTX hardware gate failed numerical parity".into())
    } else {
        Err("PTX hardware gate failed SASS calibration validity".into())
    }
}
