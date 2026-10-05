//! Isolated micro-kernel benchmarking testbed for Metal operator mutations.

use deser::Serialize;
use metal::objc::rc::autoreleasepool;
use metal::{Buffer, BufferRef, MTLCommandBufferStatus};

use crate::apple_gpu::{Runtime, gpu_seconds, thread_group};
use crate::config::KernelTrial;
use crate::fbt_moe::decode::Decoder;
use crate::fbt_pisa1::Pisa1;

const WARMUP_ROUNDS: usize = 30;

#[derive(Debug, Clone, Serialize)]
pub struct MicroMetrics {
    pub operator: String,
    pub candidate_name: String,
    pub iterations: usize,
    pub baseline_median_us: f64,
    pub candidate_median_us: f64,
    pub speedup: f64,
    pub bytes_transferred: usize,
    pub achieved_bw_gbs: f64,
    pub roofline_pct: f64,
    pub max_abs_error: f64,
    pub passed_gate: bool,
}

fn peak_bw(name: &str) -> f64 {
    if name.contains("Ultra") {
        800.0
    } else if name.contains("Max") {
        if name.contains("M4") { 546.0 } else { 400.0 }
    } else if name.contains("Pro") {
        if name.contains("M1") { 200.0 } else { 150.0 }
    } else if name.contains("M4") {
        120.0
    } else {
        100.0
    }
}

fn complete_cmd(cmd: &metal::CommandBufferRef) -> Result<f64, String> {
    cmd.commit();
    cmd.wait_until_completed();
    if cmd.status() != MTLCommandBufferStatus::Completed {
        return Err(format!("Metal command failed: {:?}", cmd.status()));
    }
    gpu_seconds(cmd).ok_or_else(|| "missing GPU seconds".into())
}

fn median_latency(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    if values.is_empty() {
        return 0.0;
    }
    values[values.len() / 2]
}

fn filled_buffer(runtime: &Runtime, count: usize, fill: u16) -> Buffer {
    let buf = runtime.buffer::<u16>(count);
    let slice = unsafe { std::slice::from_raw_parts_mut(buf.contents().cast::<u16>(), count) };
    slice.fill(fill);
    buf
}

fn max_difference(left: &BufferRef, right: &BufferRef, count: usize) -> f64 {
    let left_slice = unsafe { std::slice::from_raw_parts(left.contents().cast::<u16>(), count) };
    let right_slice = unsafe { std::slice::from_raw_parts(right.contents().cast::<u16>(), count) };
    let mut max_err = 0.0f64;
    for (&l, &r) in left_slice.iter().zip(right_slice.iter()) {
        let l_f = f32::from_bits(u32::from(l) << 16) as f64;
        let r_f = f32::from_bits(u32::from(r) << 16) as f64;
        let diff = (l_f - r_f).abs();
        if diff > max_err {
            max_err = diff;
        }
    }
    max_err
}

fn time_gemv(
    runtime: &Runtime,
    decoder: &Decoder,
    input: &BufferRef,
    weight: &BufferRef,
    output: &BufferRef,
    k: u32,
    n: u32,
    iterations: usize,
) -> Result<f64, String> {
    for _ in 0..WARMUP_ROUNDS {
        let cmd = runtime.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        decoder.gemv(&enc, (input, 0), (weight, 0), (output, 0), k, n, 0);
        enc.end_encoding();
        complete_cmd(cmd)?;
    }
    let mut times = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let cmd = runtime.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        decoder.gemv(&enc, (input, 0), (weight, 0), (output, 0), k, n, 0);
        enc.end_encoding();
        times.push(complete_cmd(cmd)? * 1e6);
    }
    Ok(median_latency(times))
}

fn time_pisa(
    runtime: &Runtime,
    pisa: &Pisa1,
    qkv: &BufferRef,
    iterations: usize,
) -> Result<f64, String> {
    for _ in 0..WARMUP_ROUNDS {
        let cmd = runtime.queue.new_command_buffer();
        pisa.encode_layer(cmd, qkv);
        complete_cmd(cmd)?;
    }
    let mut times = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let cmd = runtime.queue.new_command_buffer();
        pisa.encode_layer(cmd, qkv);
        times.push(complete_cmd(cmd)? * 1e6);
    }
    Ok(median_latency(times))
}

fn probe_decode(
    runtime: &Runtime,
    trial: Option<&KernelTrial>,
    name: &str,
    iterations: usize,
    atol: f64,
    min_speedup: f64,
) -> Result<MicroMetrics, String> {
    let k: u32 = 512;
    let n: u32 = 640;
    let bytes_trans = 2 * (k as usize + (k * n) as usize + n as usize);
    let input = filled_buffer(runtime, k as usize, 0x3c00);
    let weight = filled_buffer(runtime, (k * n) as usize, 0x3800);
    let base_out = filled_buffer(runtime, n as usize, 0);
    let cand_out = filled_buffer(runtime, n as usize, 0);

    let base_decoder = Decoder::new(runtime)?;
    let cand_decoder = Decoder::for_trial(runtime, 4096, false, 1, trial)?;

    let base_med = time_gemv(
        runtime,
        &base_decoder,
        &input,
        &weight,
        &base_out,
        k,
        n,
        iterations,
    )?;
    let cand_med = time_gemv(
        runtime,
        &cand_decoder,
        &input,
        &weight,
        &cand_out,
        k,
        n,
        iterations,
    )?;

    let speedup = if cand_med > 0.0 {
        base_med / cand_med
    } else {
        1.0
    };
    let max_err = max_difference(&base_out, &cand_out, n as usize);
    let achieved_bw = if cand_med > 0.0 {
        (bytes_trans as f64 / 1e9) / (cand_med * 1e-6)
    } else {
        0.0
    };
    let dev_name = runtime.info().name.as_str();
    let roofline = (achieved_bw / peak_bw(dev_name)) * 100.0;
    let passed = speedup >= min_speedup && max_err <= atol;

    Ok(MicroMetrics {
        operator: "decode".into(),
        candidate_name: name.into(),
        iterations,
        baseline_median_us: base_med,
        candidate_median_us: cand_med,
        speedup,
        bytes_transferred: bytes_trans,
        achieved_bw_gbs: achieved_bw,
        roofline_pct: roofline,
        max_abs_error: max_err,
        passed_gate: passed,
    })
}

fn probe_pisa(
    runtime: &Runtime,
    trial: Option<&KernelTrial>,
    name: &str,
    iterations: usize,
    atol: f64,
    min_speedup: f64,
) -> Result<MicroMetrics, String> {
    let qkv_count = 2 * 4096 * 640;
    let bytes_trans = qkv_count * 2;
    let qkv = filled_buffer(runtime, qkv_count, 0x3c00);

    let base_pisa = Pisa1::new(runtime)?;
    let cand_pisa = Pisa1::for_trial(runtime, trial)?;

    let base_med = time_pisa(runtime, &base_pisa, &qkv, iterations)?;
    let cand_med = time_pisa(runtime, &cand_pisa, &qkv, iterations)?;

    let speedup = if cand_med > 0.0 {
        base_med / cand_med
    } else {
        1.0
    };
    let max_err = max_difference(base_pisa.output(), cand_pisa.output(), 2 * 4096 * 512);
    let achieved_bw = if cand_med > 0.0 {
        (bytes_trans as f64 / 1e9) / (cand_med * 1e-6)
    } else {
        0.0
    };
    let dev_name = runtime.info().name.as_str();
    let roofline = (achieved_bw / peak_bw(dev_name)) * 100.0;
    let passed = speedup >= min_speedup && max_err <= atol;

    Ok(MicroMetrics {
        operator: "pisa".into(),
        candidate_name: name.into(),
        iterations,
        baseline_median_us: base_med,
        candidate_median_us: cand_med,
        speedup,
        bytes_transferred: bytes_trans,
        achieved_bw_gbs: achieved_bw,
        roofline_pct: roofline,
        max_abs_error: max_err,
        passed_gate: passed,
    })
}

#[allow(clippy::too_many_arguments)]
fn time_mhc(
    runtime: &Runtime,
    pipeline: &metal::ComputePipelineState,
    streams: &BufferRef,
    branch: &BufferRef,
    coefficients: &BufferRef,
    output: &BufferRef,
    shape: &[u32; 4],
    elements: u32,
    iterations: usize,
) -> Result<f64, String> {
    for _ in 0..WARMUP_ROUNDS {
        let cmd = runtime.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipeline);
        for (slot, buffer) in [streams, branch, coefficients, output].iter().enumerate() {
            enc.set_buffer(slot as u64, Some(buffer), 0);
        }
        enc.set_bytes(
            4,
            std::mem::size_of_val(shape) as u64,
            shape.as_ptr().cast(),
        );
        enc.dispatch_threads(thread_group(u64::from(elements)), thread_group(256));
        enc.end_encoding();
        complete_cmd(cmd)?;
    }
    let mut times = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let cmd = runtime.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipeline);
        for (slot, buffer) in [streams, branch, coefficients, output].iter().enumerate() {
            enc.set_buffer(slot as u64, Some(buffer), 0);
        }
        enc.set_bytes(
            4,
            std::mem::size_of_val(shape) as u64,
            shape.as_ptr().cast(),
        );
        enc.dispatch_threads(thread_group(u64::from(elements)), thread_group(256));
        enc.end_encoding();
        times.push(complete_cmd(cmd)? * 1e6);
    }
    Ok(median_latency(times))
}

fn probe_mhc(
    runtime: &Runtime,
    trial: Option<&KernelTrial>,
    name: &str,
    iterations: usize,
    atol: f64,
    min_speedup: f64,
) -> Result<MicroMetrics, String> {
    let rows: u32 = 4096;
    let width: u32 = 512;
    let mhc_input: usize = 2048;
    let mhc_coeffs: usize = 24;
    let shape = [rows, width, 2, 0];
    let elements = rows * width / 4;
    let bytes_trans = (rows as usize * mhc_input * 2) * 2
        + (rows as usize * width as usize * 2)
        + (rows as usize * mhc_coeffs * 4);
    let streams = filled_buffer(runtime, rows as usize * mhc_input, 0x3800);
    let branch = filled_buffer(runtime, rows as usize * width as usize, 0x3000);
    let coeffs_buf = runtime.buffer_with(
        &(0..rows as usize * mhc_coeffs)
            .map(|i| ((i * 17 % 257) as f32 - 128.0) / 128.0)
            .collect::<Vec<_>>(),
    );
    let base_out = filled_buffer(runtime, rows as usize * mhc_input, 0);
    let cand_out = filled_buffer(runtime, rows as usize * mhc_input, 0);
    let base_pipeline = runtime.pipeline(
        include_str!("fbt_moe.metal"),
        "mHC baseline update",
        "fbt_mhc_update_rows",
    )?;
    let cand_source = match trial.and_then(|t| t.mhc.as_ref()) {
        Some(path) => {
            std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?
        }
        None => include_str!("fbt_moe.metal").to_string(),
    };
    let cand_pipeline =
        runtime.pipeline(&cand_source, "mHC candidate update", "fbt_mhc_update_rows")?;
    let base_med = time_mhc(
        runtime,
        &base_pipeline,
        &streams,
        &branch,
        &coeffs_buf,
        &base_out,
        &shape,
        elements,
        iterations,
    )?;
    let cand_med = time_mhc(
        runtime,
        &cand_pipeline,
        &streams,
        &branch,
        &coeffs_buf,
        &cand_out,
        &shape,
        elements,
        iterations,
    )?;
    let speedup = if cand_med > 0.0 {
        base_med / cand_med
    } else {
        1.0
    };
    let max_err = max_difference(&base_out, &cand_out, rows as usize * mhc_input);
    let achieved_bw = if cand_med > 0.0 {
        (bytes_trans as f64 / 1e9) / (cand_med * 1e-6)
    } else {
        0.0
    };
    let dev_name = runtime.info().name.as_str();
    let roofline = (achieved_bw / peak_bw(dev_name)) * 100.0;
    let passed = speedup >= min_speedup && max_err <= atol;
    Ok(MicroMetrics {
        operator: "mhc".into(),
        candidate_name: name.into(),
        iterations,
        baseline_median_us: base_med,
        candidate_median_us: cand_med,
        speedup,
        bytes_transferred: bytes_trans,
        achieved_bw_gbs: achieved_bw,
        roofline_pct: roofline,
        max_abs_error: max_err,
        passed_gate: passed,
    })
}

pub fn benchmark_micro(
    operator: &str,
    trial: Option<&KernelTrial>,
    name: &str,
    iterations: usize,
    atol: f64,
    min_speedup: f64,
) -> Result<MicroMetrics, String> {
    autoreleasepool(|| {
        let runtime = Runtime::shared()?;
        match operator {
            "decode" => probe_decode(&runtime, trial, name, iterations, atol, min_speedup),
            "pisa" => probe_pisa(&runtime, trial, name, iterations, atol, min_speedup),
            "mhc" => probe_mhc(&runtime, trial, name, iterations, atol, min_speedup),
            _ => Err(format!(
                "micro-benchmark operator {operator:?} not supported yet on Metal"
            )),
        }
    })
}
