use std::ffi::c_void;
use std::sync::Arc;
use std::time::{Duration, Instant};

extern crate metal as metal_crate;

use metal_crate::{ComputePipelineState, MTLSize};
use ndarray::{Array2, ArrayView1, ArrayView2};

use crate::apple_gpu::Runtime;

const SOURCE: &str = include_str!("posterior.metal");
const NAMES: [&str; 3] = ["posterior_1", "posterior_2", "posterior_4"];
const THREADS: u64 = 256;

#[repr(C)]
#[derive(Clone, Copy)]
struct Params {
    queries: u32,
    neighbors: u32,
    metrics: u32,
    epistemic_scale: f32,
    aleatoric_scale: f32,
}

pub(super) struct Output {
    pub mu: Array2<f64>,
    pub se: Array2<f64>,
}

struct PosteriorEngine {
    runtime: Arc<Runtime>,
    pipelines: Vec<ComputePipelineState>,
}

pub(super) fn compute(
    distances: &ArrayView2<f64>,
    indices: &ArrayView2<i64>,
    outcomes: &ArrayView2<f64>,
    y_scale: &ArrayView1<f64>,
    epistemic_scale: f64,
    aleatoric_scale: f64,
) -> Result<Output, String> {
    PosteriorEngine::new()?.compute(
        distances,
        indices,
        outcomes,
        y_scale,
        epistemic_scale,
        aleatoric_scale,
    )
}

impl PosteriorEngine {
    fn new() -> Result<Self, String> {
        let runtime = Runtime::shared()?;
        let pipelines = NAMES
            .iter()
            .map(|name| runtime.agx_pipeline(SOURCE, "posterior", name))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { runtime, pipelines })
    }

    fn compute(
        &self,
        distances: &ArrayView2<f64>,
        indices: &ArrayView2<i64>,
        outcomes: &ArrayView2<f64>,
        y_scale: &ArrayView1<f64>,
        epistemic_scale: f64,
        aleatoric_scale: f64,
    ) -> Result<Output, String> {
        let queries = distances.nrows();
        let neighbors = distances.ncols();
        let metrics = outcomes.ncols();
        let distances = distances.iter().map(|&v| v as f32).collect::<Vec<_>>();
        let indices = indices
            .iter()
            .map(|&v| u32::try_from(v).map_err(|_| "negative posterior neighbor"))
            .collect::<Result<Vec<_>, _>>()?;
        let outcomes = outcomes.iter().map(|&v| v as f32).collect::<Vec<_>>();
        let y_scale = y_scale.iter().map(|&v| v as f32).collect::<Vec<_>>();
        let params = Params {
            queries: u32::try_from(queries).map_err(|_| "too many posterior queries")?,
            neighbors: u32::try_from(neighbors).map_err(|_| "too many posterior neighbors")?,
            metrics: u32::try_from(metrics).map_err(|_| "too many posterior metrics")?,
            epistemic_scale: epistemic_scale as f32,
            aleatoric_scale: aleatoric_scale as f32,
        };
        let inputs = Inputs {
            distances: self.runtime.buffer_with(&distances),
            indices: self.runtime.buffer_with(&indices),
            outcomes: self.runtime.buffer_with(&outcomes),
            y_scale: self.runtime.buffer_with(&y_scale),
            mu: self.runtime.buffer::<f32>(queries * metrics),
            se: self.runtime.buffer::<f32>(queries * metrics),
            params,
            elements: queries * metrics,
        };
        let schedule = self.schedule(neighbors, metrics, &inputs)?;
        self.dispatch(schedule, &inputs)?;
        Ok(Output {
            mu: read_matrix(&inputs.mu, queries, metrics)?,
            se: read_matrix(&inputs.se, queries, metrics)?,
        })
    }

    fn schedule(&self, neighbors: usize, metrics: usize, inputs: &Inputs) -> Result<usize, String> {
        self.dispatch(0, inputs)?;
        let reference_mu = values(&inputs.mu, inputs.elements).to_vec();
        let reference_se = values(&inputs.se, inputs.elements).to_vec();
        self.runtime.schedule(
            "posterior.reduce",
            &[neighbors, metrics],
            self.pipelines.len(),
            |candidate| {
                self.dispatch(candidate, inputs)?;
                if !close(&reference_mu, values(&inputs.mu, inputs.elements))
                    || !close(&reference_se, values(&inputs.se, inputs.elements))
                {
                    return Ok(None);
                }
                let mut samples = [Duration::ZERO; 3];
                for elapsed in &mut samples {
                    let start = Instant::now();
                    self.dispatch(candidate, inputs)?;
                    *elapsed = start.elapsed();
                }
                samples.sort_unstable();
                Ok(Some(samples[1]))
            },
        )
    }

    fn dispatch(&self, schedule: usize, inputs: &Inputs) -> Result<(), String> {
        let command = self.runtime.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipelines[schedule]);
        for (slot, buffer) in [
            &inputs.distances,
            &inputs.indices,
            &inputs.outcomes,
            &inputs.y_scale,
            &inputs.mu,
            &inputs.se,
        ]
        .iter()
        .enumerate()
        {
            encoder.set_buffer(slot as u64, Some(buffer), 0);
        }
        encoder.set_bytes(
            6,
            size_of::<Params>() as u64,
            (&inputs.params as *const Params).cast::<c_void>(),
        );
        encoder.dispatch_threads(
            MTLSize::new(inputs.elements as u64, 1, 1),
            MTLSize::new(THREADS, 1, 1),
        );
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        let status = command.status();
        if status == metal_crate::MTLCommandBufferStatus::Completed {
            Ok(())
        } else {
            Err(format!("posterior AGX command failed: {status:?}"))
        }
    }
}

struct Inputs {
    distances: metal_crate::Buffer,
    indices: metal_crate::Buffer,
    outcomes: metal_crate::Buffer,
    y_scale: metal_crate::Buffer,
    mu: metal_crate::Buffer,
    se: metal_crate::Buffer,
    params: Params,
    elements: usize,
}

fn read_matrix(
    buffer: &metal_crate::Buffer,
    rows: usize,
    columns: usize,
) -> Result<Array2<f64>, String> {
    Array2::from_shape_vec(
        (rows, columns),
        values(buffer, rows * columns)
            .iter()
            .map(|&value| value as f64)
            .collect(),
    )
    .map_err(|error| error.to_string())
}

fn values(buffer: &metal_crate::Buffer, len: usize) -> &[f32] {
    unsafe { std::slice::from_raw_parts(buffer.contents().cast::<f32>(), len) }
}

fn close(reference: &[f32], actual: &[f32]) -> bool {
    reference
        .iter()
        .zip(actual)
        .all(|(&a, &b)| (a - b).abs() <= 2.0e-5 * (1.0 + a.abs()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_builds_and_reads_shared_buffer() {
        let engine = PosteriorEngine::new().unwrap();
        assert_eq!(engine.pipelines.len(), NAMES.len());
        let buffer = engine.runtime.buffer_with(&[1.0_f32, 2.0, 3.0, 4.0]);
        assert_eq!(
            read_matrix(&buffer, 2, 2).unwrap(),
            Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap()
        );
    }
}
