//! Batched resident diagonal-metric scoring on the shared Apple GPU.
use super::*;
use crate::apple_gpu::thread_group;
use crate::params::ENNParams;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::index::sample;

const METRIC_SOURCE: &str = include_str!("bf16_metric.metal");
const METRIC_CANDIDATES: usize = 9;
const _: () = assert!(MAX_HISTORY == 128 && FAMILIES == 4);
#[cfg(test)]
#[path = "bf16_metal/metric_tests.rs"]
mod tests;

#[repr(C)]
#[derive(Clone, Copy)]
struct MetricParams {
    rows: u32,
    candidates: u32,
    samples: u32,
    neighbors: u32,
    local: u32,
    pad: u32,
    epistemic: f32,
    aleatoric: f32,
}

pub(super) struct MetricGpu {
    runtime: Arc<Runtime>,
    components: Buffer,
    weights: Buffer,
    distances: Buffer,
    radii: Buffer,
    outcomes: Buffer,
    selected: Buffer,
    scores: Buffer,
    distance_pipeline: ComputePipelineState,
    radius_pipeline: ComputePipelineState,
    score_pipeline: ComputePipelineState,
}

impl MetricGpu {
    pub fn new(runtime: Arc<Runtime>) -> Result<Self, String> {
        let distance_pipeline = runtime.precise(METRIC_SOURCE, "Metric fit", "metric_distances")?;
        let radius_pipeline = runtime.precise(METRIC_SOURCE, "Metric fit", "metric_radii")?;
        let score_pipeline = runtime.precise(METRIC_SOURCE, "Metric fit", "metric_scores")?;
        for pipeline in [&radius_pipeline, &score_pipeline] {
            if pipeline.max_total_threads_per_threadgroup() < MAX_HISTORY as u64
                || pipeline.thread_execution_width() != 32
            {
                return Err("Metric fit requires 128 threads and 32-lane SIMD groups".into());
            }
        }
        Ok(Self {
            components: runtime.buffer::<[f32; FAMILIES]>(MAX_HISTORY * MAX_HISTORY),
            weights: runtime.buffer::<[f32; FAMILIES]>(METRIC_CANDIDATES),
            distances: runtime.buffer::<f32>(METRIC_CANDIDATES * MAX_HISTORY * MAX_HISTORY),
            radii: runtime.buffer::<f32>(METRIC_CANDIDATES * MAX_HISTORY),
            outcomes: runtime.buffer::<[f32; 2]>(MAX_HISTORY),
            selected: runtime.buffer::<u32>(MAX_HISTORY),
            scores: runtime.buffer::<f32>(METRIC_CANDIDATES * MAX_HISTORY),
            runtime,
            distance_pipeline,
            radius_pipeline,
            score_pipeline,
        })
    }

    pub fn bytes(&self) -> u64 {
        self.buffers().iter().map(|b| b.length()).sum()
    }

    fn buffers(&self) -> [&Buffer; 7] {
        [
            &self.components,
            &self.weights,
            &self.distances,
            &self.radii,
            &self.outcomes,
            &self.selected,
            &self.scores,
        ]
    }

    pub fn score(
        &mut self,
        family: &FamilyHistory,
        candidates: &[[f32; FAMILIES]],
        y: &ndarray::ArrayView2<f64>,
        variance: &ndarray::ArrayView2<f64>,
        params: ENNParams,
        samples: usize,
        seed: u64,
        local: Option<usize>,
    ) -> Result<Vec<f64>, String> {
        let n = y.nrows();
        check_metric(family, candidates, y, variance, params, samples, local)?;
        let selected = if samples >= n {
            (0..n).collect::<Vec<_>>()
        } else {
            sample(&mut StdRng::seed_from_u64(seed), n, samples)
                .into_iter()
                .collect()
        };
        let outcomes = metric_outcomes(y, variance)?;
        upload_metric(&self.components, &family.components[..n * MAX_HISTORY]);
        upload_metric(&self.weights, candidates);
        upload_metric(&self.outcomes, &outcomes);
        upload_metric(
            &self.selected,
            &selected.iter().map(|&i| i as u32).collect::<Vec<_>>(),
        );
        let p = MetricParams {
            rows: n as u32,
            candidates: candidates.len() as u32,
            samples: selected.len() as u32,
            neighbors: params.k_neighbors as u32,
            local: local.unwrap_or(0).min(n - 1) as u32,
            pad: 0,
            epistemic: params.epistemic_scale as f32,
            aleatoric: params.aleatoric_scale as f32,
        };
        let command = self.runtime.queue.new_command_buffer();
        self.encode(
            command,
            &self.distance_pipeline,
            &[&self.components, &self.weights, &self.distances],
            p,
            (candidates.len() * n * n).div_ceil(128),
            128,
        );
        if p.local > 0 {
            self.encode(
                command,
                &self.radius_pipeline,
                &[&self.distances, &self.radii],
                p,
                candidates.len() * n,
                128,
            );
        }
        self.encode(
            command,
            &self.score_pipeline,
            &[
                &self.distances,
                &self.radii,
                &self.outcomes,
                &self.selected,
                &self.scores,
            ],
            p,
            candidates.len() * selected.len(),
            128,
        );
        finish(command)?;
        Ok(read::<f32>(&self.scores, candidates.len() * selected.len())
            .chunks_exact(selected.len())
            .map(|rows| rows.iter().map(|&v| f64::from(v)).sum())
            .collect())
    }

    fn encode(
        &self,
        command: &CommandBufferRef,
        pipeline: &ComputePipelineState,
        buffers: &[&Buffer],
        params: MetricParams,
        groups: usize,
        threads: usize,
    ) {
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        for (index, buffer) in buffers.iter().enumerate() {
            encoder.set_buffer(index as u64, Some(buffer), 0);
        }
        encoder.set_bytes(
            buffers.len() as u64,
            size_of::<MetricParams>() as u64,
            (&params as *const MetricParams).cast(),
        );
        encoder.dispatch_thread_groups(thread_group(groups as u64), thread_group(threads as u64));
        encoder.end_encoding();
    }
}

fn upload_metric<T: Copy>(buffer: &Buffer, values: &[T]) {
    assert!(std::mem::size_of_val(values) <= buffer.length() as usize);
    // The caller waits for its preceding command; buffers are private fit scratch.
    unsafe {
        std::ptr::copy_nonoverlapping(values.as_ptr(), buffer.contents().cast(), values.len());
    }
}

fn metric_outcomes(
    y: &ndarray::ArrayView2<f64>,
    variance: &ndarray::ArrayView2<f64>,
) -> Result<Vec<[f32; 2]>, String> {
    let std = y.std_axis(ndarray::Axis(0), 0.0)[0];
    let scale = if std.is_finite() && std > 0.0 {
        std
    } else {
        1.0
    };
    let values = (0..y.nrows())
        .map(|i| {
            [
                (y[[i, 0]] / scale) as f32,
                (variance[[i, 0]] / (scale * scale)) as f32,
            ]
        })
        .collect::<Vec<_>>();
    if values.iter().flatten().any(|v| !v.is_finite()) {
        return Err("Metric outcomes exceed FP32 compute range".into());
    }
    Ok(values)
}

fn check_metric(
    family: &FamilyHistory,
    candidates: &[[f32; FAMILIES]],
    y: &ndarray::ArrayView2<f64>,
    variance: &ndarray::ArrayView2<f64>,
    params: ENNParams,
    samples: usize,
    local: Option<usize>,
) -> Result<(), String> {
    if !(2..=MAX_HISTORY).contains(&y.nrows())
        || y.ncols() != 1
        || y.raw_dim() != variance.raw_dim()
        || samples == 0
        || local == Some(0)
        || candidates.is_empty()
        || candidates.len() > METRIC_CANDIDATES
        || y.iter().any(|v| !v.is_finite())
        || variance.iter().any(|v| !v.is_finite() || *v < 0.0)
        || candidates
            .iter()
            .flatten()
            .any(|v| !v.is_finite() || *v <= 0.0)
        || family.components.len() != MAX_HISTORY * MAX_HISTORY
        || family
            .components
            .chunks_exact(MAX_HISTORY)
            .take(y.nrows())
            .flat_map(|row| row[..y.nrows()].iter().flatten())
            .any(|v| !v.is_finite() || *v < 0.0)
        || !((params.epistemic_scale as f32).is_finite()
            && (params.aleatoric_scale as f32).is_finite())
    {
        return Err("Invalid resident metric fit shape, values, or budget".into());
    }
    ENNParams::new(
        params.k_neighbors,
        params.epistemic_scale,
        params.aleatoric_scale,
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}
