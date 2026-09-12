//! Fixed-shape PISA-1 attention for the proposed 4K subsecond scorer.

use crate::apple_gpu::{Runtime, gpu_seconds, thread_group};
use metal::{
    Buffer, BufferRef, CommandBufferRef, ComputePipelineState, MTLCommandBufferStatus, MTLSize,
};
use std::time::Instant;

const CONTEXT: u32 = 4096;
const BATCH: u32 = 2;
const ROWS: u32 = BATCH * CONTEXT;
const QUERY_HEADS: u32 = 8;
const HEAD_DIM: u32 = 64;
const QUERY_WIDTH: u32 = QUERY_HEADS * HEAD_DIM;
const QKV_WIDTH: u32 = QUERY_WIDTH + 2 * HEAD_DIM;
const BLOCK: u32 = 64;
const LEAVES: u32 = CONTEXT / BLOCK;
const NODES: u32 = 2 * LEAVES - 1;
const SELECTED: u32 = 8;

#[derive(Debug, Clone, Copy)]
pub(crate) struct Pisa1Probe {
    pub pyramid_gpu_seconds: f64,
    pub selection_gpu_seconds: f64,
    pub attention_gpu_seconds: f64,
    pub layer_gpu_seconds: f64,
    pub layer_wall_seconds: f64,
    pub max_abs_error: f64,
}

struct Pipelines {
    leaf_means: ComputePipelineState,
    upper_means: ComputePipelineState,
    select: ComputePipelineState,
    attention: ComputePipelineState,
    select_attention: ComputePipelineState,
    select_attention_q4: ComputePipelineState,
}

pub(crate) struct Pisa1 {
    pipelines: Pipelines,
    pyramid: Buffer,
    blocks: Buffer,
    output: Buffer,
    oracle_output: Buffer,
}

#[derive(Clone, Copy)]
enum Stage {
    Pyramid,
    Selection,
    Attention,
}

fn dispatch(
    command: &CommandBufferRef,
    pipeline: &ComputePipelineState,
    buffers: &[&BufferRef],
    groups: MTLSize,
    threads: u64,
) {
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    for (index, buffer) in buffers.iter().enumerate() {
        encoder.set_buffer(index as u64, Some(buffer), 0);
    }
    encoder.dispatch_thread_groups(groups, thread_group(threads));
    encoder.end_encoding();
}

fn complete(command: &CommandBufferRef) -> Result<f64, String> {
    command.commit();
    command.wait_until_completed();
    if command.status() != MTLCommandBufferStatus::Completed {
        return Err(format!("PISA-1 command failed: {:?}", command.status()));
    }
    gpu_seconds(command).ok_or("Metal did not report PISA-1 GPU timing".into())
}

impl Pipelines {
    fn new(runtime: &Runtime) -> Result<Self, String> {
        let source = include_str!("fbt_pisa1.metal");
        let compile = |name| runtime.precise(source, "PISA-1 attention", name);
        let result = Self {
            leaf_means: compile("fbt_pisa1_leaf_means")?,
            upper_means: compile("fbt_pisa1_upper_means")?,
            select: compile("fbt_pisa1_select")?,
            attention: compile("fbt_pisa1_attention")?,
            select_attention: compile("fbt_pisa1_select_attention")?,
            select_attention_q4: compile("fbt_pisa1_select_attention_q4")?,
        };
        if result.select_attention_q4.thread_execution_width() != 32
            || result
                .select_attention_q4
                .max_total_threads_per_threadgroup()
                < 128
            || result
                .select_attention_q4
                .static_threadgroup_memory_length()
                > runtime.device.max_threadgroup_memory_length()
        {
            return Err("PISA-1 attention exceeds this GPU's SIMD or threadgroup limits".into());
        }
        Ok(result)
    }
}

impl Pisa1 {
    pub(crate) fn new(runtime: &Runtime) -> Result<Self, String> {
        Ok(Self {
            pipelines: Pipelines::new(runtime)?,
            pyramid: runtime.buffer::<u16>((BATCH * NODES * HEAD_DIM) as usize),
            blocks: runtime.buffer_with(&vec![u32::MAX; (ROWS * SELECTED) as usize + 64]),
            output: runtime.buffer_with(&vec![0x7e00u16; (ROWS * QUERY_WIDTH) as usize + 64]),
            oracle_output: runtime
                .buffer_with(&vec![0x7e00u16; (ROWS * QUERY_WIDTH) as usize + 64]),
        })
    }

    fn encode_pyramid(&self, command: &CommandBufferRef, qkv: &BufferRef) {
        dispatch(
            command,
            &self.pipelines.leaf_means,
            &[qkv, &self.pyramid],
            MTLSize {
                width: u64::from(LEAVES),
                height: u64::from(BATCH),
                depth: 1,
            },
            64,
        );
        dispatch(
            command,
            &self.pipelines.upper_means,
            &[&self.pyramid],
            thread_group(u64::from(BATCH)),
            64,
        );
    }

    fn encode_selection(&self, command: &CommandBufferRef, qkv: &BufferRef) {
        dispatch(
            command,
            &self.pipelines.select,
            &[qkv, &self.pyramid, &self.blocks],
            thread_group(u64::from(ROWS)),
            32,
        );
    }

    fn encode_attention(&self, command: &CommandBufferRef, qkv: &BufferRef) {
        dispatch(
            command,
            &self.pipelines.select_attention_q4,
            &[qkv, &self.pyramid, &self.blocks, &self.output],
            thread_group(u64::from(ROWS / 4)),
            128,
        );
    }

    fn encode_oracle(&self, command: &CommandBufferRef, qkv: &BufferRef) {
        dispatch(
            command,
            &self.pipelines.select_attention,
            &[qkv, &self.pyramid, &self.blocks, &self.oracle_output],
            thread_group(u64::from(ROWS)),
            128,
        );
    }

    fn encode(&self, command: &CommandBufferRef, qkv: &BufferRef, stage: Option<Stage>) {
        if stage.is_none_or(|value| matches!(value, Stage::Pyramid)) {
            self.encode_pyramid(command, qkv);
        }
        if matches!(stage, Some(Stage::Selection)) {
            self.encode_selection(command, qkv);
        }
        if stage.is_none_or(|value| matches!(value, Stage::Attention)) {
            self.encode_attention(command, qkv);
        }
    }

    pub(crate) fn encode_layer(&self, command: &CommandBufferRef, qkv: &BufferRef) {
        self.encode(command, qkv, None);
    }

    pub(crate) fn output(&self) -> &BufferRef {
        &self.output
    }
}

fn decode_half(bits: u16) -> f32 {
    let exponent = (bits >> 10) & 31;
    let fraction = f32::from(bits & 1023);
    let magnitude = match exponent {
        0 => fraction * 2.0f32.powi(-24),
        31 => f32::NAN,
        _ => (1024.0 + fraction) * 2.0f32.powi(i32::from(exponent) - 25),
    };
    if bits & 0x8000 == 0 {
        magnitude
    } else {
        -magnitude
    }
}

fn half_values(buffer: &BufferRef, elements: usize) -> &[u16] {
    unsafe { std::slice::from_raw_parts(buffer.contents().cast::<u16>(), elements) }
}

fn expected_blocks(qkv: &[u16], pyramid: &[u16], row: u32) -> [u32; SELECTED as usize] {
    let mut query = [0.0f32; HEAD_DIM as usize];
    for dim in 0..HEAD_DIM {
        for head in 0..QUERY_HEADS {
            query[dim as usize] +=
                decode_half(qkv[(row * QKV_WIDTH + head * HEAD_DIM + dim) as usize]);
        }
    }
    let position = row % CONTEXT;
    let current = position / BLOCK;
    let previous = current.saturating_sub(1);
    let sequence = row / CONTEXT;
    let mut candidates = [u32::MAX; 16];
    for (candidate, value) in candidates.iter_mut().enumerate() {
        *value = candidate as u32;
    }
    let mut chosen = [u32::MAX; SELECTED as usize];
    for level in (0..=2u32).rev() {
        let offset = match level {
            2 => 96,
            1 => 64,
            _ => 0,
        };
        let mut scores = [f32::NEG_INFINITY; 16];
        for (slot, &node) in candidates.iter().enumerate() {
            if node == u32::MAX {
                continue;
            }
            let begin = node << level;
            let end = begin + (1 << level) - 1;
            let forced = [0, previous, current]
                .into_iter()
                .any(|leaf| leaf >= begin && leaf <= end);
            if forced {
                scores[slot] = f32::INFINITY;
            } else if end < current {
                let base = ((sequence * NODES + offset + node) * HEAD_DIM) as usize;
                scores[slot] = query
                    .iter()
                    .enumerate()
                    .map(|(dim, &value)| value * decode_half(pyramid[base + dim]))
                    .sum();
            }
        }
        for output in &mut chosen {
            let best = scores
                .iter()
                .enumerate()
                .filter(|(_, score)| **score > f32::NEG_INFINITY)
                .max_by(|(left_index, left), (right_index, right)| {
                    left.total_cmp(right)
                        .then_with(|| candidates[*right_index].cmp(&candidates[*left_index]))
                })
                .map(|(index, _)| index);
            if let Some(best) = best {
                *output = candidates[best];
                scores[best] = f32::NEG_INFINITY;
            } else {
                *output = u32::MAX;
            }
        }
        if level > 0 {
            for (slot, &node) in chosen.iter().enumerate() {
                candidates[2 * slot] = if node == u32::MAX { u32::MAX } else { 2 * node };
                candidates[2 * slot + 1] = if node == u32::MAX {
                    u32::MAX
                } else {
                    2 * node + 1
                };
            }
        }
    }
    chosen
}

fn validate_selection(pisa: &Pisa1, qkv: &BufferRef) -> Result<(), String> {
    let qkv = half_values(qkv, (ROWS * QKV_WIDTH) as usize);
    let pyramid = half_values(&pisa.pyramid, (BATCH * NODES * HEAD_DIM) as usize);
    let actual = unsafe {
        std::slice::from_raw_parts(
            pisa.blocks.contents().cast::<u32>(),
            (ROWS * SELECTED) as usize + 64,
        )
    };
    if !actual[(ROWS * SELECTED) as usize..]
        .iter()
        .all(|&value| value == u32::MAX)
    {
        return Err("PISA-1 selection overwrote its canary".into());
    }
    for row in [0, 63, 64, 511, 1024, 4095, 4096, 8191] {
        let expected = expected_blocks(qkv, pyramid, row);
        let start = (row * SELECTED) as usize;
        if actual[start..start + SELECTED as usize] != expected {
            return Err(format!(
                "PISA-1 selection mismatch at row {row}: {:?} != {expected:?}",
                &actual[start..start + SELECTED as usize]
            ));
        }
    }
    for row in 0..ROWS {
        let position = row % CONTEXT;
        let current = position / BLOCK;
        let expected_count = SELECTED.min(current + 1) as usize;
        let selected = &actual[(row * SELECTED) as usize..((row + 1) * SELECTED) as usize];
        let valid = selected
            .iter()
            .copied()
            .filter(|&block| block != u32::MAX)
            .collect::<Vec<_>>();
        if valid.len() != expected_count
            || valid.iter().any(|&block| block > current)
            || (0..valid.len()).any(|left| valid[left + 1..].contains(&valid[left]))
            || !valid.contains(&0)
            || !valid.contains(&current)
            || (current > 0 && !valid.contains(&(current - 1)))
        {
            return Err(format!(
                "PISA-1 invalid forced/causal selection at row {row}"
            ));
        }
    }
    let mut selections = 0usize;
    let mut unique = 0usize;
    for sequence in 0..BATCH {
        for first in (0..CONTEXT).step_by(4) {
            let mut union = [u32::MAX; 4 * SELECTED as usize];
            let mut union_len = 0usize;
            for local in 0..4 {
                let row = sequence * CONTEXT + first + local;
                for &block in &actual[(row * SELECTED) as usize..((row + 1) * SELECTED) as usize] {
                    if block != u32::MAX {
                        selections += 1;
                        if !union[..union_len].contains(&block) {
                            union[union_len] = block;
                            union_len += 1;
                        }
                    }
                }
            }
            unique += union_len;
        }
    }
    eprintln!(
        "TURBO_ENN_PISA1_QUERY_TILE tile=4 selections={} unique_blocks={} kv_reuse={:.3}",
        selections,
        unique,
        selections as f64 / unique as f64,
    );
    Ok(())
}

fn validate_attention(pisa: &Pisa1, qkv: &BufferRef) -> Result<f64, String> {
    let qkv = half_values(qkv, (ROWS * QKV_WIDTH) as usize);
    let blocks = unsafe {
        std::slice::from_raw_parts(
            pisa.blocks.contents().cast::<u32>(),
            (ROWS * SELECTED) as usize,
        )
    };
    let output = half_values(&pisa.output, (ROWS * QUERY_WIDTH) as usize + 64);
    if !output[(ROWS * QUERY_WIDTH) as usize..]
        .iter()
        .all(|&value| value == 0x7e00)
    {
        return Err("PISA-1 attention overwrote its canary".into());
    }
    let mut maximum = 0.0f64;
    for row in [0, 63, 64, 511, 1024, 4095, 4096, 8191] {
        let sequence = row / CONTEXT;
        let position = row % CONTEXT;
        for head in 0..QUERY_HEADS {
            let mut logits = Vec::with_capacity((SELECTED * BLOCK) as usize);
            let mut tokens = Vec::with_capacity((SELECTED * BLOCK) as usize);
            for &block in &blocks[(row * SELECTED) as usize..((row + 1) * SELECTED) as usize] {
                if block == u32::MAX {
                    continue;
                }
                for local in 0..BLOCK {
                    let token_position = block * BLOCK + local;
                    if token_position > position {
                        continue;
                    }
                    let source = sequence * CONTEXT + token_position;
                    let mut score = 0.0f32;
                    for dim in 0..HEAD_DIM {
                        score +=
                            decode_half(qkv[(row * QKV_WIDTH + head * HEAD_DIM + dim) as usize])
                                * decode_half(
                                    qkv[(source * QKV_WIDTH + QUERY_WIDTH + dim) as usize],
                                );
                    }
                    logits.push(score * 0.125);
                    tokens.push(source);
                }
            }
            let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let weights = logits
                .iter()
                .map(|&logit| (logit - max).exp())
                .collect::<Vec<_>>();
            let total: f32 = weights.iter().sum();
            for dim in 0..HEAD_DIM {
                let expected = weights
                    .iter()
                    .zip(&tokens)
                    .map(|(&weight, &token)| {
                        weight
                            * decode_half(
                                qkv[(token * QKV_WIDTH + QUERY_WIDTH + HEAD_DIM + dim) as usize],
                            )
                    })
                    .sum::<f32>()
                    / total;
                let index = (row * QUERY_WIDTH + head * HEAD_DIM + dim) as usize;
                let actual = decode_half(output[index]);
                if !actual.is_finite() {
                    return Err(format!("PISA-1 produced non-finite output at {index}"));
                }
                maximum = maximum.max(f64::from((expected - actual).abs()));
            }
        }
    }
    if maximum > 0.002 {
        return Err(format!(
            "PISA-1 attention max sampled absolute error {maximum:.9} exceeds 0.002"
        ));
    }
    Ok(maximum)
}

fn validate_q4_oracle(pisa: &Pisa1) -> Result<f64, String> {
    let elements = (ROWS * QUERY_WIDTH) as usize;
    let q4 = half_values(&pisa.output, elements);
    let q1 = half_values(&pisa.oracle_output, elements);
    let mut maximum = 0.0f64;
    let mut maximum_index = 0usize;
    for index in 0..elements {
        let error = f64::from((decode_half(q4[index]) - decode_half(q1[index])).abs());
        if !error.is_finite() {
            return Err(format!("PISA-1 parity found a non-finite value at {index}"));
        }
        if error > maximum {
            maximum = error;
            maximum_index = index;
        }
    }
    if maximum > 0.002 {
        return Err(format!(
            "PISA-1 Q4 versus Q1 max absolute error {maximum:.9} at {maximum_index} exceeds 0.002"
        ));
    }
    eprintln!(
        "TURBO_ENN_PISA1_PARITY candidate=q4 oracle=q1 elements={elements} max_abs_error={maximum:.9} index={maximum_index}"
    );
    Ok(maximum)
}

fn stage_median(
    runtime: &Runtime,
    pisa: &Pisa1,
    qkv: &BufferRef,
    stage: Stage,
) -> Result<f64, String> {
    let mut samples = Vec::with_capacity(3);
    for _ in 0..3 {
        let command = runtime.queue.new_command_buffer();
        pisa.encode(command, qkv, Some(stage));
        samples.push(complete(command)?);
    }
    samples.sort_by(f64::total_cmp);
    Ok(samples[1])
}

fn check_parity(runtime: &Runtime, pisa: &Pisa1, qkv: &BufferRef) -> Result<f64, String> {
    let command = runtime.queue.new_command_buffer();
    pisa.encode(command, qkv, None);
    complete(command)?;
    validate_selection(pisa, qkv)?;
    let selected = unsafe {
        std::slice::from_raw_parts(
            pisa.blocks.contents().cast::<u32>(),
            (ROWS * SELECTED) as usize,
        )
        .to_vec()
    };
    let command = runtime.queue.new_command_buffer();
    pisa.encode_oracle(command, qkv);
    complete(command)?;
    let oracle = unsafe {
        std::slice::from_raw_parts(
            pisa.blocks.contents().cast::<u32>(),
            (ROWS * SELECTED) as usize,
        )
    };
    if selected != oracle {
        return Err("PISA-1 Q4 selection differs from Q1".into());
    }
    Ok(validate_attention(pisa, qkv)?.max(validate_q4_oracle(pisa)?))
}

pub(crate) fn run_pisa1_probe(
    runtime: &Runtime,
    qkv: &BufferRef,
    rounds: u32,
) -> Result<Pisa1Probe, String> {
    if rounds == 0 {
        return Err("PISA-1 probe requires positive rounds".into());
    }
    let pisa = Pisa1::new(runtime)?;
    let tied = runtime.buffer_with(&vec![0u16; (ROWS * QKV_WIDTH) as usize]);
    check_parity(runtime, &pisa, &tied)?;
    let max_abs_error = check_parity(runtime, &pisa, qkv)?;
    let pyramid_gpu_seconds = stage_median(runtime, &pisa, qkv, Stage::Pyramid)?;
    let selection_gpu_seconds = stage_median(runtime, &pisa, qkv, Stage::Selection)?;
    let attention_gpu_seconds = stage_median(runtime, &pisa, qkv, Stage::Attention)?;
    for _ in 0..3 {
        let command = runtime.queue.new_command_buffer();
        pisa.encode(command, qkv, None);
        complete(command)?;
    }
    let mut gpu = Vec::with_capacity(rounds as usize);
    let mut wall = Vec::with_capacity(rounds as usize);
    for _ in 0..rounds {
        let command = runtime.queue.new_command_buffer();
        pisa.encode(command, qkv, None);
        let start = Instant::now();
        gpu.push(complete(command)?);
        wall.push(start.elapsed().as_secs_f64());
    }
    gpu.sort_by(f64::total_cmp);
    wall.sort_by(f64::total_cmp);
    Ok(Pisa1Probe {
        pyramid_gpu_seconds,
        selection_gpu_seconds,
        attention_gpu_seconds,
        layer_gpu_seconds: gpu[gpu.len() / 2],
        layer_wall_seconds: wall[wall.len() / 2],
        max_abs_error,
    })
}
