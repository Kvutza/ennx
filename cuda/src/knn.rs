use std::convert::Infallible;
use std::fmt;
use std::mem::{size_of, size_of_val};
use std::sync::OnceLock;

use cuda_core::{CudaEvent, CudaStream, DeviceBuffer, DeviceCopy, LaunchConfig1D};
use cuda_host::kernel_family::{
    KernelFamily, KernelProblem, KernelSelector, KernelVariant, NoKernelSelectionCache,
    SelectionMode,
};
use ennx_cuda_kernels::{
    BatchParams, BatchValue, DrawParams, KNN_MAX_K, KNN_ROW_TILE, KNN_WARP_TILE, KnnParams,
    MergeParams, PosteriorParams, THREADS, WeightedParams,
};

use super::{CudaResult, Runtime, copy_prefix, cuda_error, read_prefix, timing_event, to_u32};

const WARP_DIM: usize = 32;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ScanKind {
    Rows,
    Warps,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ScanMeta {
    tile_rows: u32,
}

#[derive(Clone, Copy, Debug)]
struct ScanProblem {
    rows: usize,
    dims: usize,
    queries: usize,
    neighbors: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ScanReject(&'static str);

impl fmt::Display for ScanReject {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.0)
    }
}

impl std::error::Error for ScanReject {}

type ScanVariant = KernelVariant<ScanKind, ScanKind, ScanMeta>;
type ScanFamily = KernelFamily<ScanKind, ScanKind, ScanMeta, 2>;

impl KernelProblem<ScanVariant> for ScanProblem {
    type Rejection = ScanReject;

    fn validate(&self, variant: &ScanVariant) -> Result<(), Self::Rejection> {
        if self.rows == 0 || self.dims == 0 || self.queries == 0 {
            return Err(ScanReject("CUDA index shape must be non-zero"));
        }
        if self.neighbors == 0 || self.neighbors > KNN_MAX_K || self.neighbors > self.rows {
            return Err(ScanReject("CUDA index neighbor count is unsupported"));
        }
        if variant.metadata().tile_rows == 0 {
            return Err(ScanReject("CUDA index tile cannot be empty"));
        }
        Ok(())
    }
}

struct ScanSelector;

impl KernelSelector<ScanProblem, ScanVariant, ScanKind> for ScanSelector {
    type Error = Infallible;

    fn select(
        &mut self,
        _family: cuda_host::kernel_family::KernelFamilyId,
        problem: &ScanProblem,
        _eligible: &[&ScanVariant],
    ) -> Result<ScanKind, Self::Error> {
        Ok(if problem.dims >= WARP_DIM {
            ScanKind::Warps
        } else {
            ScanKind::Rows
        })
    }
}

fn scan_family() -> CudaResult<&'static ScanFamily> {
    static FAMILY: OnceLock<Result<ScanFamily, String>> = OnceLock::new();
    FAMILY
        .get_or_init(|| {
            KernelFamily::try_new(
                "ennx/knn-scan",
                1,
                [
                    KernelVariant::new(
                        ScanKind::Rows,
                        ScanKind::Rows,
                        ScanMeta {
                            tile_rows: KNN_ROW_TILE,
                        },
                    ),
                    KernelVariant::new(
                        ScanKind::Warps,
                        ScanKind::Warps,
                        ScanMeta {
                            tile_rows: KNN_WARP_TILE,
                        },
                    ),
                ],
            )
            .map_err(|error| error.to_string())
        })
        .as_ref()
        .map_err(Clone::clone)
}

#[derive(Clone, Copy, Debug, Default)]
pub struct KnnProfile {
    pub rows: usize,
    pub queries: usize,
    pub dims: usize,
    pub neighbors: usize,
    pub scan_ms: f32,
    pub merge_ms: f32,
    pub posterior_ms: f32,
    pub total_ms: f32,
    pub kind: &'static str,
}

#[derive(Clone, Copy, Debug)]
pub struct PosteriorSpec {
    pub metrics: usize,
    pub input_k: usize,
    pub used_k: usize,
    pub skip: usize,
    pub epistemic_scale: f32,
    pub aleatoric_scale: f32,
    pub epsilon: f32,
}

#[derive(Debug)]
pub struct PosteriorOutput {
    pub means: Vec<f32>,
    pub errors: Vec<f32>,
    pub indices: Vec<u32>,
}

#[derive(Clone, Copy, Debug)]
pub struct WeightedSpec {
    pub metrics: usize,
    pub input_k: usize,
    pub used_k: usize,
    pub skip: usize,
    pub epistemic_scale: f32,
    pub aleatoric_scale: f32,
    pub epsilon: f32,
    pub observation_noise: bool,
}

#[derive(Debug)]
pub struct WeightedOutput {
    pub weights: Vec<f32>,
    pub l2: Vec<f32>,
    pub means: Vec<f32>,
    pub errors: Vec<f32>,
    pub epistemic: Vec<f32>,
    pub aleatoric: Vec<f32>,
    pub indices: Vec<u32>,
}

#[derive(Clone, Copy, Debug)]
pub struct BatchSpec {
    pub metrics: usize,
    pub input_k: usize,
    pub epsilon: f32,
    pub observation_noise: bool,
}

#[derive(Debug)]
pub struct BatchOutput {
    pub means: Vec<f32>,
    pub errors: Vec<f32>,
    pub epistemic: Vec<f32>,
    pub aleatoric: Vec<f32>,
}

#[derive(Debug)]
pub struct DrawOutput {
    pub draws: Vec<f64>,
    pub indices: Vec<u32>,
}

struct SearchStage {
    kind: ScanKind,
    input_a: bool,
    queries: usize,
    neighbors: usize,
    start: CudaEvent,
    scan_end: CudaEvent,
    merge_end: CudaEvent,
}

struct KnnScratch {
    query_capacity: usize,
    partial_capacity: usize,
    queries: DeviceBuffer<f32>,
    distances_a: DeviceBuffer<f32>,
    distances_b: DeviceBuffer<f32>,
    indices_a: DeviceBuffer<u32>,
    indices_b: DeviceBuffer<u32>,
}

impl KnnScratch {
    fn new(stream: &CudaStream) -> CudaResult<Self> {
        Ok(Self {
            query_capacity: 1,
            partial_capacity: 1,
            queries: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            distances_a: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            distances_b: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            indices_a: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            indices_b: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
        })
    }

    fn ensure(&mut self, stream: &CudaStream, queries: usize, partials: usize) -> CudaResult<()> {
        let query_capacity = next_capacity(queries)?;
        let partial_capacity = next_capacity(partials)?;
        if query_capacity > self.query_capacity {
            self.queries = DeviceBuffer::zeroed(stream, query_capacity).map_err(cuda_error)?;
            self.query_capacity = query_capacity;
        }
        if partial_capacity > self.partial_capacity {
            self.distances_a =
                DeviceBuffer::zeroed(stream, partial_capacity).map_err(cuda_error)?;
            self.distances_b =
                DeviceBuffer::zeroed(stream, partial_capacity).map_err(cuda_error)?;
            self.indices_a = DeviceBuffer::zeroed(stream, partial_capacity).map_err(cuda_error)?;
            self.indices_b = DeviceBuffer::zeroed(stream, partial_capacity).map_err(cuda_error)?;
            self.partial_capacity = partial_capacity;
        }
        Ok(())
    }
}

struct PosteriorScratch {
    outcome_capacity: usize,
    variance_capacity: usize,
    scale_capacity: usize,
    value_capacity: usize,
    weight_capacity: usize,
    param_capacity: usize,
    seed_capacity: usize,
    draw_capacity: usize,
    index_capacity: usize,
    outcomes: DeviceBuffer<f32>,
    variances: DeviceBuffer<f32>,
    scales: DeviceBuffer<f32>,
    weights: DeviceBuffer<f32>,
    params: DeviceBuffer<BatchValue>,
    seeds: DeviceBuffer<u64>,
    draws: DeviceBuffer<f64>,
    l2: DeviceBuffer<f32>,
    means: DeviceBuffer<f32>,
    errors: DeviceBuffer<f32>,
    epistemic: DeviceBuffer<f32>,
    aleatoric: DeviceBuffer<f32>,
    indices: DeviceBuffer<u32>,
}

impl PosteriorScratch {
    fn new(stream: &CudaStream) -> CudaResult<Self> {
        Ok(Self {
            outcome_capacity: 1,
            variance_capacity: 1,
            scale_capacity: 1,
            value_capacity: 1,
            weight_capacity: 1,
            param_capacity: 1,
            seed_capacity: 1,
            draw_capacity: 1,
            index_capacity: 1,
            outcomes: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            variances: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            scales: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            weights: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            params: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            seeds: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            draws: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            l2: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            means: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            errors: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            epistemic: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            aleatoric: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
            indices: DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?,
        })
    }

    fn ensure(
        &mut self,
        stream: &CudaStream,
        outcomes: usize,
        variances: usize,
        scales: usize,
        values: usize,
        weights: usize,
        indices: usize,
    ) -> CudaResult<()> {
        let outcome_capacity = next_capacity(outcomes)?;
        let variance_capacity = next_capacity(variances)?;
        let scale_capacity = next_capacity(scales)?;
        let value_capacity = next_capacity(values)?;
        let weight_capacity = next_capacity(weights)?;
        let index_capacity = next_capacity(indices)?;
        if outcome_capacity > self.outcome_capacity {
            self.outcomes = DeviceBuffer::zeroed(stream, outcome_capacity).map_err(cuda_error)?;
            self.outcome_capacity = outcome_capacity;
        }
        if variance_capacity > self.variance_capacity {
            self.variances = DeviceBuffer::zeroed(stream, variance_capacity).map_err(cuda_error)?;
            self.variance_capacity = variance_capacity;
        }
        if scale_capacity > self.scale_capacity {
            self.scales = DeviceBuffer::zeroed(stream, scale_capacity).map_err(cuda_error)?;
            self.scale_capacity = scale_capacity;
        }
        if value_capacity > self.value_capacity {
            self.l2 = DeviceBuffer::zeroed(stream, value_capacity).map_err(cuda_error)?;
            self.means = DeviceBuffer::zeroed(stream, value_capacity).map_err(cuda_error)?;
            self.errors = DeviceBuffer::zeroed(stream, value_capacity).map_err(cuda_error)?;
            self.epistemic = DeviceBuffer::zeroed(stream, value_capacity).map_err(cuda_error)?;
            self.aleatoric = DeviceBuffer::zeroed(stream, value_capacity).map_err(cuda_error)?;
            self.value_capacity = value_capacity;
        }
        if weight_capacity > self.weight_capacity {
            self.weights = DeviceBuffer::zeroed(stream, weight_capacity).map_err(cuda_error)?;
            self.weight_capacity = weight_capacity;
        }
        if index_capacity > self.index_capacity {
            self.indices = DeviceBuffer::zeroed(stream, index_capacity).map_err(cuda_error)?;
            self.index_capacity = index_capacity;
        }
        Ok(())
    }

    fn sync(&mut self, stream: &CudaStream, outcomes: &[f32], scales: &[f32]) -> CudaResult<()> {
        copy_prefix(&self.outcomes, outcomes, stream)?;
        copy_prefix(&self.scales, scales, stream)
    }

    fn sync_variances(&mut self, stream: &CudaStream, variances: Option<&[f32]>) -> CudaResult<()> {
        if let Some(values) = variances {
            copy_prefix(&self.variances, values, stream)?;
        }
        Ok(())
    }

    fn ensure_params(&mut self, stream: &CudaStream, count: usize) -> CudaResult<()> {
        let capacity = next_capacity(count)?;
        if capacity > self.param_capacity {
            self.params = DeviceBuffer::zeroed(stream, capacity).map_err(cuda_error)?;
            self.param_capacity = capacity;
        }
        Ok(())
    }

    fn sync_params(&mut self, stream: &CudaStream, values: &[BatchValue]) -> CudaResult<()> {
        copy_prefix(&self.params, values, stream)
    }

    fn ensure_draws(
        &mut self,
        stream: &CudaStream,
        seed_count: usize,
        draw_count: usize,
    ) -> CudaResult<()> {
        let seed_capacity = next_capacity(seed_count)?;
        let draw_capacity = next_capacity(draw_count)?;
        if seed_capacity > self.seed_capacity {
            self.seeds = DeviceBuffer::zeroed(stream, seed_capacity).map_err(cuda_error)?;
            self.seed_capacity = seed_capacity;
        }
        if draw_capacity > self.draw_capacity {
            self.draws = DeviceBuffer::zeroed(stream, draw_capacity).map_err(cuda_error)?;
            self.draw_capacity = draw_capacity;
        }
        Ok(())
    }

    fn sync_seeds(&mut self, stream: &CudaStream, seeds: &[u64]) -> CudaResult<()> {
        copy_prefix(&self.seeds, seeds, stream)
    }
}

pub struct CudaIndex {
    runtime: Runtime,
    rows: DeviceBuffer<f32>,
    row_capacity: usize,
    row_count: usize,
    dims: usize,
    scratch: KnnScratch,
    posterior: PosteriorScratch,
    profile: Option<KnnProfile>,
}

impl CudaIndex {
    pub fn new(dims: usize, rows: &[f32]) -> CudaResult<Self> {
        if dims == 0 || rows.len() % dims != 0 {
            return Err("CUDA index rows do not match the dimension".to_string());
        }
        if rows.iter().any(|value| !value.is_finite()) {
            return Err("CUDA index rows must be finite".to_string());
        }
        let runtime = Runtime::new()?;
        let scratch = KnnScratch::new(&runtime.stream)?;
        let posterior = PosteriorScratch::new(&runtime.stream)?;
        let row_capacity = next_capacity(rows.len())?;
        let row_buffer = DeviceBuffer::zeroed(&runtime.stream, row_capacity).map_err(cuda_error)?;
        copy_prefix(&row_buffer, rows, &runtime.stream)?;
        runtime.stream.synchronize().map_err(cuda_error)?;
        Ok(Self {
            runtime,
            rows: row_buffer,
            row_capacity,
            row_count: rows.len() / dims,
            dims,
            scratch,
            posterior,
            profile: None,
        })
    }

    pub fn len(&self) -> usize {
        self.row_count
    }

    pub fn is_empty(&self) -> bool {
        self.row_count == 0
    }

    pub fn memory_bytes(&self) -> usize {
        self.row_count
            .saturating_mul(self.dims)
            .saturating_mul(size_of::<f32>())
    }

    pub fn profile(&self) -> Option<KnnProfile> {
        self.profile
    }

    pub fn rebuild(&mut self, rows: &[f32]) -> CudaResult<()> {
        self.check_rows(rows)?;
        let capacity = next_capacity(rows.len())?;
        if capacity != self.row_capacity {
            self.rows = DeviceBuffer::zeroed(&self.runtime.stream, capacity).map_err(cuda_error)?;
            self.row_capacity = capacity;
        }
        copy_prefix(&self.rows, rows, &self.runtime.stream)?;
        self.runtime.stream.synchronize().map_err(cuda_error)?;
        self.row_count = rows.len() / self.dims;
        Ok(())
    }

    pub fn add(&mut self, rows: &[f32]) -> CudaResult<()> {
        self.check_rows(rows)?;
        if rows.is_empty() {
            return Ok(());
        }
        let used = self
            .row_count
            .checked_mul(self.dims)
            .ok_or("CUDA index row size overflow")?;
        let required = used
            .checked_add(rows.len())
            .ok_or("CUDA index row size overflow")?;
        if required > self.row_capacity {
            let capacity = next_capacity(required)?;
            let next = DeviceBuffer::zeroed(&self.runtime.stream, capacity).map_err(cuda_error)?;
            if used > 0 {
                let used_bytes = used
                    .checked_mul(size_of::<f32>())
                    .ok_or("CUDA index copy size overflow")?;
                unsafe {
                    cuda_core::memory::memcpy_dtod_async(
                        next.cu_deviceptr(),
                        self.rows.cu_deviceptr(),
                        used_bytes,
                        self.runtime.stream.cu_stream(),
                    )
                    .map_err(cuda_error)?;
                }
            }
            self.rows = next;
            self.row_capacity = capacity;
        }
        copy_at(&self.rows, used, rows, &self.runtime.stream)?;
        self.runtime.stream.synchronize().map_err(cuda_error)?;
        self.row_count = self
            .row_count
            .checked_add(rows.len() / self.dims)
            .ok_or("CUDA index row count overflow")?;
        Ok(())
    }

    pub fn search(
        &mut self,
        queries: &[f32],
        query_count: usize,
        neighbors: usize,
    ) -> CudaResult<(Vec<f32>, Vec<u32>)> {
        let stage = self.stage_search(queries, query_count, neighbors)?;
        let output = self.read_neighbors(&stage)?;
        self.set_profile(&stage, &stage.merge_end)?;
        Ok(output)
    }

    pub fn posterior(
        &mut self,
        queries: &[f32],
        query_count: usize,
        outcomes: &[f32],
        scales: &[f32],
        spec: PosteriorSpec,
    ) -> CudaResult<PosteriorOutput> {
        self.check_posterior(outcomes, scales, spec)?;
        let value_count = query_count
            .checked_mul(spec.metrics)
            .ok_or("CUDA posterior output size overflow")?;
        let index_count = query_count
            .checked_mul(spec.used_k)
            .ok_or("CUDA posterior index size overflow")?;
        self.posterior.ensure(
            &self.runtime.stream,
            outcomes.len(),
            1,
            scales.len(),
            value_count,
            1,
            index_count,
        )?;
        self.posterior
            .sync(&self.runtime.stream, outcomes, scales)?;
        let stage = self.stage_search(queries, query_count, spec.input_k)?;
        self.launch_posterior(&stage, spec)?;
        let end = timing_event(&self.runtime.stream)?;
        self.runtime.context.check_err().map_err(cuda_error)?;
        let output = PosteriorOutput {
            means: read_prefix(&self.posterior.means, &self.runtime.stream, value_count)?,
            errors: read_prefix(&self.posterior.errors, &self.runtime.stream, value_count)?,
            indices: read_prefix(&self.posterior.indices, &self.runtime.stream, index_count)?,
        };
        self.set_profile(&stage, &end)?;
        Ok(output)
    }

    pub fn weighted(
        &mut self,
        queries: &[f32],
        query_count: usize,
        outcomes: &[f32],
        variances: Option<&[f32]>,
        scales: &[f32],
        spec: WeightedSpec,
    ) -> CudaResult<WeightedOutput> {
        self.check_weighted(outcomes, variances, scales, spec)?;
        let value_count = query_count
            .checked_mul(spec.metrics)
            .ok_or("CUDA weighted output size overflow")?;
        let weight_count = value_count
            .checked_mul(spec.used_k)
            .ok_or("CUDA weighted weight size overflow")?;
        let index_count = query_count
            .checked_mul(spec.used_k)
            .ok_or("CUDA weighted index size overflow")?;
        self.posterior.ensure(
            &self.runtime.stream,
            outcomes.len(),
            variances.map_or(1, <[f32]>::len),
            scales.len(),
            value_count,
            weight_count,
            index_count,
        )?;
        self.posterior
            .sync(&self.runtime.stream, outcomes, scales)?;
        self.posterior
            .sync_variances(&self.runtime.stream, variances)?;
        let stage = self.stage_search(queries, query_count, spec.input_k)?;
        self.launch_weighted(&stage, spec, variances.is_some())?;
        let end = timing_event(&self.runtime.stream)?;
        self.runtime.context.check_err().map_err(cuda_error)?;
        let output = WeightedOutput {
            weights: read_prefix(&self.posterior.weights, &self.runtime.stream, weight_count)?,
            l2: read_prefix(&self.posterior.l2, &self.runtime.stream, value_count)?,
            means: read_prefix(&self.posterior.means, &self.runtime.stream, value_count)?,
            errors: read_prefix(&self.posterior.errors, &self.runtime.stream, value_count)?,
            epistemic: read_prefix(&self.posterior.epistemic, &self.runtime.stream, value_count)?,
            aleatoric: read_prefix(&self.posterior.aleatoric, &self.runtime.stream, value_count)?,
            indices: read_prefix(&self.posterior.indices, &self.runtime.stream, index_count)?,
        };
        self.set_profile(&stage, &end)?;
        Ok(output)
    }

    pub fn batch(
        &mut self,
        queries: &[f32],
        query_count: usize,
        outcomes: &[f32],
        variances: Option<&[f32]>,
        scales: &[f32],
        values: &[BatchValue],
        spec: BatchSpec,
    ) -> CudaResult<BatchOutput> {
        self.check_batch(outcomes, variances, scales, values, spec)?;
        let output_count = values
            .len()
            .checked_mul(query_count)
            .and_then(|count| count.checked_mul(spec.metrics))
            .ok_or("CUDA batch output size overflow")?;
        self.posterior.ensure(
            &self.runtime.stream,
            outcomes.len(),
            variances.map_or(1, <[f32]>::len),
            scales.len(),
            output_count,
            1,
            1,
        )?;
        self.posterior
            .ensure_params(&self.runtime.stream, values.len())?;
        self.posterior
            .sync(&self.runtime.stream, outcomes, scales)?;
        self.posterior
            .sync_variances(&self.runtime.stream, variances)?;
        self.posterior.sync_params(&self.runtime.stream, values)?;
        let stage = self.stage_search(queries, query_count, spec.input_k)?;
        self.launch_batch(&stage, spec, values.len(), variances.is_some())?;
        let end = timing_event(&self.runtime.stream)?;
        self.runtime.context.check_err().map_err(cuda_error)?;
        let output = BatchOutput {
            means: read_prefix(&self.posterior.means, &self.runtime.stream, output_count)?,
            errors: read_prefix(&self.posterior.errors, &self.runtime.stream, output_count)?,
            epistemic: read_prefix(
                &self.posterior.epistemic,
                &self.runtime.stream,
                output_count,
            )?,
            aleatoric: read_prefix(
                &self.posterior.aleatoric,
                &self.runtime.stream,
                output_count,
            )?,
        };
        self.set_profile(&stage, &end)?;
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn draws(
        &mut self,
        queries: &[f32],
        query_count: usize,
        outcomes: &[f32],
        variances: Option<&[f32]>,
        scales: &[f32],
        seeds: &[u64],
        spec: WeightedSpec,
    ) -> CudaResult<DrawOutput> {
        if seeds.is_empty() {
            return Err("CUDA draws require at least one seed".to_string());
        }
        self.check_weighted(outcomes, variances, scales, spec)?;
        let value_count = query_count
            .checked_mul(spec.metrics)
            .ok_or("CUDA draw value size overflow")?;
        let weight_count = value_count
            .checked_mul(spec.used_k)
            .ok_or("CUDA draw weight size overflow")?;
        let index_count = query_count
            .checked_mul(spec.used_k)
            .ok_or("CUDA draw index size overflow")?;
        let draw_count = seeds
            .len()
            .checked_mul(value_count)
            .ok_or("CUDA draw output size overflow")?;
        self.posterior.ensure(
            &self.runtime.stream,
            outcomes.len(),
            variances.map_or(1, <[f32]>::len),
            scales.len(),
            value_count,
            weight_count,
            index_count,
        )?;
        self.posterior
            .ensure_draws(&self.runtime.stream, seeds.len(), draw_count)?;
        self.posterior
            .sync(&self.runtime.stream, outcomes, scales)?;
        self.posterior
            .sync_variances(&self.runtime.stream, variances)?;
        self.posterior.sync_seeds(&self.runtime.stream, seeds)?;
        let stage = self.stage_search(queries, query_count, spec.input_k)?;
        self.launch_weighted(&stage, spec, variances.is_some())?;
        self.launch_draw(&stage, spec, seeds.len())?;
        let end = timing_event(&self.runtime.stream)?;
        self.runtime.context.check_err().map_err(cuda_error)?;
        let output = DrawOutput {
            draws: read_prefix(&self.posterior.draws, &self.runtime.stream, draw_count)?,
            indices: read_prefix(&self.posterior.indices, &self.runtime.stream, index_count)?,
        };
        self.set_profile(&stage, &end)?;
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn conditional(
        &mut self,
        queries: &[f32],
        query_count: usize,
        extra_rows: &[f32],
        outcomes: &[f32],
        variances: Option<&[f32]>,
        scales: &[f32],
        spec: WeightedSpec,
    ) -> CudaResult<WeightedOutput> {
        let base_rows = self.row_count;
        self.add(extra_rows)?;
        let result = self.weighted(queries, query_count, outcomes, variances, scales, spec);
        self.row_count = base_rows;
        result
    }

    #[allow(clippy::too_many_arguments)]
    pub fn conditional_draws(
        &mut self,
        queries: &[f32],
        query_count: usize,
        extra_rows: &[f32],
        outcomes: &[f32],
        variances: Option<&[f32]>,
        scales: &[f32],
        seeds: &[u64],
        spec: WeightedSpec,
    ) -> CudaResult<DrawOutput> {
        let base_rows = self.row_count;
        self.add(extra_rows)?;
        let result = self.draws(
            queries,
            query_count,
            outcomes,
            variances,
            scales,
            seeds,
            spec,
        );
        self.row_count = base_rows;
        result
    }

    fn stage_search(
        &mut self,
        queries: &[f32],
        query_count: usize,
        neighbors: usize,
    ) -> CudaResult<SearchStage> {
        let problem = self.check_search(queries, query_count, neighbors)?;
        let family = scan_family()?;
        let mut selector = ScanSelector;
        let mut cache = NoKernelSelectionCache;
        let selected = family
            .select(&problem, SelectionMode::Auto, &mut selector, &mut cache)
            .map_err(|error| error.to_string())?;
        let kind = *selected.variant().entry();
        let tile_rows = selected.variant().metadata().tile_rows as usize;
        let lists = self.row_count.div_ceil(tile_rows);
        let partials = query_count
            .checked_mul(lists)
            .and_then(|value| value.checked_mul(neighbors))
            .ok_or("CUDA index scratch size overflow")?;
        self.scratch
            .ensure(&self.runtime.stream, queries.len(), partials)?;
        copy_prefix(&self.scratch.queries, queries, &self.runtime.stream)?;

        let params = KnnParams {
            rows: to_u32(self.row_count, "index row count")?,
            dims: to_u32(self.dims, "index dimension")?,
            queries: to_u32(query_count, "index query count")?,
            lists: to_u32(lists, "index list count")?,
            neighbors: to_u32(neighbors, "index neighbor count")?,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let total_lists = params
            .queries
            .checked_mul(params.lists)
            .ok_or("CUDA index grid size overflow")?;
        let start = timing_event(&self.runtime.stream)?;
        self.launch_scan(kind, total_lists, params)?;
        let scan_end = timing_event(&self.runtime.stream)?;
        let input_a = self.launch_merge(query_count, lists, neighbors)?;
        let merge_end = timing_event(&self.runtime.stream)?;
        self.runtime.context.check_err().map_err(cuda_error)?;
        Ok(SearchStage {
            kind,
            input_a,
            queries: query_count,
            neighbors,
            start,
            scan_end,
            merge_end,
        })
    }

    fn read_neighbors(&self, stage: &SearchStage) -> CudaResult<(Vec<f32>, Vec<u32>)> {
        let output_count = stage
            .queries
            .checked_mul(stage.neighbors)
            .ok_or("CUDA index output size overflow")?;
        if stage.input_a {
            Ok((
                read_prefix(
                    &self.scratch.distances_a,
                    &self.runtime.stream,
                    output_count,
                )?,
                read_prefix(&self.scratch.indices_a, &self.runtime.stream, output_count)?,
            ))
        } else {
            Ok((
                read_prefix(
                    &self.scratch.distances_b,
                    &self.runtime.stream,
                    output_count,
                )?,
                read_prefix(&self.scratch.indices_b, &self.runtime.stream, output_count)?,
            ))
        }
    }

    fn set_profile(&mut self, stage: &SearchStage, end: &CudaEvent) -> CudaResult<()> {
        self.profile = Some(KnnProfile {
            rows: self.row_count,
            queries: stage.queries,
            dims: self.dims,
            neighbors: stage.neighbors,
            scan_ms: stage
                .start
                .elapsed_ms(&stage.scan_end)
                .map_err(cuda_error)?,
            merge_ms: stage
                .scan_end
                .elapsed_ms(&stage.merge_end)
                .map_err(cuda_error)?,
            posterior_ms: stage.merge_end.elapsed_ms(end).map_err(cuda_error)?,
            total_ms: stage.start.elapsed_ms(end).map_err(cuda_error)?,
            kind: match stage.kind {
                ScanKind::Rows => "cuda_rows",
                ScanKind::Warps => "cuda_warps",
            },
        });
        Ok(())
    }

    fn launch_posterior(&mut self, stage: &SearchStage, spec: PosteriorSpec) -> CudaResult<()> {
        let params = PosteriorParams {
            queries: to_u32(stage.queries, "posterior query count")?,
            input_k: to_u32(spec.input_k, "posterior input neighbors")?,
            used_k: to_u32(spec.used_k, "posterior used neighbors")?,
            skip: to_u32(spec.skip, "posterior skip")?,
            metrics: to_u32(spec.metrics, "posterior metric count")?,
            epistemic_scale: spec.epistemic_scale,
            aleatoric_scale: spec.aleatoric_scale,
            epsilon: spec.epsilon,
        };
        let launch = self
            .runtime
            .module
            .prepare_posterior_light(LaunchConfig1D::new(params.queries, THREADS, 0))
            .map_err(cuda_error)?;
        if stage.input_a {
            self.runtime
                .module
                .posterior_light(
                    &self.runtime.stream,
                    &launch,
                    &self.scratch.distances_a,
                    &self.scratch.indices_a,
                    &self.posterior.outcomes,
                    &self.posterior.scales,
                    &mut self.posterior.means,
                    &mut self.posterior.errors,
                    &mut self.posterior.indices,
                    params,
                )
                .map_err(cuda_error)
        } else {
            self.runtime
                .module
                .posterior_light(
                    &self.runtime.stream,
                    &launch,
                    &self.scratch.distances_b,
                    &self.scratch.indices_b,
                    &self.posterior.outcomes,
                    &self.posterior.scales,
                    &mut self.posterior.means,
                    &mut self.posterior.errors,
                    &mut self.posterior.indices,
                    params,
                )
                .map_err(cuda_error)
        }
    }

    fn launch_weighted(
        &mut self,
        stage: &SearchStage,
        spec: WeightedSpec,
        has_yvar: bool,
    ) -> CudaResult<()> {
        let params = WeightedParams {
            queries: to_u32(stage.queries, "weighted query count")?,
            input_k: to_u32(spec.input_k, "weighted input neighbors")?,
            used_k: to_u32(spec.used_k, "weighted used neighbors")?,
            skip: to_u32(spec.skip, "weighted skip")?,
            metrics: to_u32(spec.metrics, "weighted metric count")?,
            has_yvar: u32::from(has_yvar),
            observation_noise: u32::from(spec.observation_noise),
            pad: 0,
            epistemic_scale: spec.epistemic_scale,
            aleatoric_scale: spec.aleatoric_scale,
            epsilon: spec.epsilon,
            padf: 0.0,
        };
        let blocks = params
            .queries
            .checked_mul(params.metrics)
            .ok_or("CUDA weighted grid size overflow")?;
        let launch = self
            .runtime
            .module
            .prepare_posterior_full(LaunchConfig1D::new(blocks, THREADS, 0))
            .map_err(cuda_error)?;
        if stage.input_a {
            self.runtime.module.posterior_full(
                &self.runtime.stream,
                &launch,
                &self.scratch.distances_a,
                &self.scratch.indices_a,
                &self.posterior.outcomes,
                &self.posterior.variances,
                &self.posterior.scales,
                &mut self.posterior.weights,
                &mut self.posterior.l2,
                &mut self.posterior.means,
                &mut self.posterior.errors,
                &mut self.posterior.epistemic,
                &mut self.posterior.aleatoric,
                &mut self.posterior.indices,
                params,
            )
        } else {
            self.runtime.module.posterior_full(
                &self.runtime.stream,
                &launch,
                &self.scratch.distances_b,
                &self.scratch.indices_b,
                &self.posterior.outcomes,
                &self.posterior.variances,
                &self.posterior.scales,
                &mut self.posterior.weights,
                &mut self.posterior.l2,
                &mut self.posterior.means,
                &mut self.posterior.errors,
                &mut self.posterior.epistemic,
                &mut self.posterior.aleatoric,
                &mut self.posterior.indices,
                params,
            )
        }
        .map_err(cuda_error)
    }

    fn launch_batch(
        &mut self,
        stage: &SearchStage,
        spec: BatchSpec,
        param_count: usize,
        has_yvar: bool,
    ) -> CudaResult<()> {
        let params = BatchParams {
            queries: to_u32(stage.queries, "batch query count")?,
            input_k: to_u32(spec.input_k, "batch input neighbors")?,
            metrics: to_u32(spec.metrics, "batch metric count")?,
            param_count: to_u32(param_count, "batch parameter count")?,
            has_yvar: u32::from(has_yvar),
            observation_noise: u32::from(spec.observation_noise),
            pad0: 0,
            pad1: 0,
            epsilon: spec.epsilon,
            padf0: 0.0,
            padf1: 0.0,
            padf2: 0.0,
        };
        let blocks = params
            .param_count
            .checked_mul(params.queries)
            .and_then(|count| count.checked_mul(params.metrics))
            .ok_or("CUDA batch grid size overflow")?;
        let launch = self
            .runtime
            .module
            .prepare_posterior_batch(LaunchConfig1D::new(blocks, THREADS, 0))
            .map_err(cuda_error)?;
        if stage.input_a {
            self.runtime.module.posterior_batch(
                &self.runtime.stream,
                &launch,
                &self.scratch.distances_a,
                &self.scratch.indices_a,
                &self.posterior.outcomes,
                &self.posterior.variances,
                &self.posterior.scales,
                &self.posterior.params,
                &mut self.posterior.means,
                &mut self.posterior.errors,
                &mut self.posterior.epistemic,
                &mut self.posterior.aleatoric,
                params,
            )
        } else {
            self.runtime.module.posterior_batch(
                &self.runtime.stream,
                &launch,
                &self.scratch.distances_b,
                &self.scratch.indices_b,
                &self.posterior.outcomes,
                &self.posterior.variances,
                &self.posterior.scales,
                &self.posterior.params,
                &mut self.posterior.means,
                &mut self.posterior.errors,
                &mut self.posterior.epistemic,
                &mut self.posterior.aleatoric,
                params,
            )
        }
        .map_err(cuda_error)
    }

    fn launch_draw(
        &mut self,
        stage: &SearchStage,
        spec: WeightedSpec,
        seed_count: usize,
    ) -> CudaResult<()> {
        let params = DrawParams {
            queries: to_u32(stage.queries, "draw query count")?,
            neighbors: to_u32(spec.used_k, "draw neighbor count")?,
            metrics: to_u32(spec.metrics, "draw metric count")?,
            seed_count: to_u32(seed_count, "draw seed count")?,
        };
        let outputs = params
            .seed_count
            .checked_mul(params.queries)
            .and_then(|count| count.checked_mul(params.metrics))
            .ok_or("CUDA draw grid size overflow")?;
        let blocks = outputs.div_ceil(THREADS);
        let launch = self
            .runtime
            .module
            .prepare_posterior_draw(LaunchConfig1D::new(blocks, THREADS, 0))
            .map_err(cuda_error)?;
        self.runtime
            .module
            .posterior_draw(
                &self.runtime.stream,
                &launch,
                &self.posterior.weights,
                &self.posterior.l2,
                &self.posterior.means,
                &self.posterior.errors,
                &self.posterior.indices,
                &self.posterior.seeds,
                &mut self.posterior.draws,
                params,
            )
            .map_err(cuda_error)
    }

    fn check_posterior(
        &self,
        outcomes: &[f32],
        scales: &[f32],
        spec: PosteriorSpec,
    ) -> CudaResult<()> {
        let expected = self
            .row_count
            .checked_mul(spec.metrics)
            .ok_or("CUDA posterior outcome size overflow")?;
        let used_end = spec
            .skip
            .checked_add(spec.used_k)
            .ok_or("CUDA posterior neighbor range overflow")?;
        if spec.metrics == 0
            || spec.input_k == 0
            || spec.used_k == 0
            || spec.input_k > KNN_MAX_K
            || used_end > spec.input_k
        {
            return Err("CUDA posterior shape is unsupported".to_string());
        }
        if outcomes.len() != expected || scales.len() != spec.metrics {
            return Err("CUDA posterior outcomes or scales have the wrong shape".to_string());
        }
        if outcomes.iter().any(|value| !value.is_finite())
            || scales.iter().any(|value| !value.is_finite())
            || !spec.epistemic_scale.is_finite()
            || !spec.aleatoric_scale.is_finite()
            || !spec.epsilon.is_finite()
            || spec.epsilon <= 0.0
        {
            return Err("CUDA posterior values must be finite".to_string());
        }
        Ok(())
    }

    fn check_weighted(
        &self,
        outcomes: &[f32],
        variances: Option<&[f32]>,
        scales: &[f32],
        spec: WeightedSpec,
    ) -> CudaResult<()> {
        self.check_posterior(
            outcomes,
            scales,
            PosteriorSpec {
                metrics: spec.metrics,
                input_k: spec.input_k,
                used_k: spec.used_k,
                skip: spec.skip,
                epistemic_scale: spec.epistemic_scale,
                aleatoric_scale: spec.aleatoric_scale,
                epsilon: spec.epsilon,
            },
        )?;
        let expected = self
            .row_count
            .checked_mul(spec.metrics)
            .ok_or("CUDA weighted variance size overflow")?;
        if let Some(values) = variances {
            if values.len() != expected || values.iter().any(|value| !value.is_finite()) {
                return Err("CUDA weighted variances have the wrong shape or values".to_string());
            }
        }
        if scales.iter().any(|value| *value <= 0.0) {
            return Err("CUDA weighted scales must be positive".to_string());
        }
        Ok(())
    }

    fn check_batch(
        &self,
        outcomes: &[f32],
        variances: Option<&[f32]>,
        scales: &[f32],
        values: &[BatchValue],
        spec: BatchSpec,
    ) -> CudaResult<()> {
        let Some(first) = values.first() else {
            return Err("CUDA batch requires parameters".to_string());
        };
        self.check_weighted(
            outcomes,
            variances,
            scales,
            WeightedSpec {
                metrics: spec.metrics,
                input_k: spec.input_k,
                used_k: first.used_k as usize,
                skip: first.skip as usize,
                epistemic_scale: first.epistemic_scale,
                aleatoric_scale: first.aleatoric_scale,
                epsilon: spec.epsilon,
                observation_noise: spec.observation_noise,
            },
        )?;
        for value in values {
            let end = (value.skip as usize)
                .checked_add(value.used_k as usize)
                .ok_or("CUDA batch neighbor range overflow")?;
            if value.used_k == 0
                || end > spec.input_k
                || !value.epistemic_scale.is_finite()
                || !value.aleatoric_scale.is_finite()
            {
                return Err("CUDA batch parameters are unsupported".to_string());
            }
        }
        Ok(())
    }

    fn launch_scan(&mut self, kind: ScanKind, lists: u32, params: KnnParams) -> CudaResult<()> {
        match kind {
            ScanKind::Rows => {
                let launch = self
                    .runtime
                    .module
                    .prepare_scan_rows(LaunchConfig1D::new(lists, THREADS, 0))
                    .map_err(cuda_error)?;
                self.runtime
                    .module
                    .scan_rows(
                        &self.runtime.stream,
                        &launch,
                        &self.rows,
                        &self.scratch.queries,
                        &mut self.scratch.distances_a,
                        &mut self.scratch.indices_a,
                        params,
                    )
                    .map_err(cuda_error)
            }
            ScanKind::Warps => {
                let launch = self
                    .runtime
                    .module
                    .prepare_scan_warps(LaunchConfig1D::new(lists, THREADS, 0))
                    .map_err(cuda_error)?;
                self.runtime
                    .module
                    .scan_warps(
                        &self.runtime.stream,
                        &launch,
                        &self.rows,
                        &self.scratch.queries,
                        &mut self.scratch.distances_a,
                        &mut self.scratch.indices_a,
                        params,
                    )
                    .map_err(cuda_error)
            }
        }
    }

    fn launch_merge(
        &mut self,
        queries: usize,
        mut input_lists: usize,
        neighbors: usize,
    ) -> CudaResult<bool> {
        let mut input_a = true;
        while input_lists > 1 {
            let output_lists = input_lists.div_ceil(2);
            let params = MergeParams {
                queries: to_u32(queries, "merge query count")?,
                input_lists: to_u32(input_lists, "merge input list count")?,
                output_lists: to_u32(output_lists, "merge output list count")?,
                neighbors: to_u32(neighbors, "merge neighbor count")?,
            };
            let blocks = params
                .queries
                .checked_mul(params.output_lists)
                .ok_or("CUDA merge grid size overflow")?;
            let launch = self
                .runtime
                .module
                .prepare_merge_topk(LaunchConfig1D::new(blocks, THREADS, 0))
                .map_err(cuda_error)?;
            if input_a {
                self.runtime
                    .module
                    .merge_topk(
                        &self.runtime.stream,
                        &launch,
                        &self.scratch.distances_a,
                        &self.scratch.indices_a,
                        &mut self.scratch.distances_b,
                        &mut self.scratch.indices_b,
                        params,
                    )
                    .map_err(cuda_error)?;
            } else {
                self.runtime
                    .module
                    .merge_topk(
                        &self.runtime.stream,
                        &launch,
                        &self.scratch.distances_b,
                        &self.scratch.indices_b,
                        &mut self.scratch.distances_a,
                        &mut self.scratch.indices_a,
                        params,
                    )
                    .map_err(cuda_error)?;
            }
            input_a = !input_a;
            input_lists = output_lists;
        }
        Ok(input_a)
    }

    fn check_rows(&self, rows: &[f32]) -> CudaResult<()> {
        if rows.len() % self.dims != 0 {
            Err("CUDA index rows do not match the dimension".to_string())
        } else if rows.iter().any(|value| !value.is_finite()) {
            Err("CUDA index rows must be finite".to_string())
        } else {
            Ok(())
        }
    }

    fn check_search(
        &self,
        queries: &[f32],
        query_count: usize,
        neighbors: usize,
    ) -> CudaResult<ScanProblem> {
        if self.is_empty() || query_count == 0 {
            return Err("CUDA index search requires rows and queries".to_string());
        }
        let expected = query_count
            .checked_mul(self.dims)
            .ok_or("CUDA index query size overflow")?;
        if queries.len() != expected || queries.iter().any(|value| !value.is_finite()) {
            return Err("CUDA index queries do not match the finite input shape".to_string());
        }
        let problem = ScanProblem {
            rows: self.row_count,
            dims: self.dims,
            queries: query_count,
            neighbors,
        };
        problem
            .validate(
                scan_family()?
                    .variants()
                    .first()
                    .expect("CUDA scan family is non-empty"),
            )
            .map_err(|error| error.to_string())?;
        Ok(problem)
    }
}

fn copy_at<T: DeviceCopy>(
    buffer: &DeviceBuffer<T>,
    offset: usize,
    values: &[T],
    stream: &CudaStream,
) -> CudaResult<()> {
    let end = offset
        .checked_add(values.len())
        .ok_or("CUDA copy range overflow")?;
    if end > buffer.len() {
        return Err("CUDA copy exceeds device allocation".to_string());
    }
    if values.is_empty() {
        return Ok(());
    }
    let byte_offset = offset
        .checked_mul(size_of::<T>())
        .ok_or("CUDA copy byte offset overflow")?;
    unsafe {
        cuda_core::memory::memcpy_htod_async(
            buffer.cu_deviceptr() + byte_offset as u64,
            values.as_ptr(),
            size_of_val(values),
            stream.cu_stream(),
        )
        .map_err(cuda_error)
    }
}

fn next_capacity(value: usize) -> CudaResult<usize> {
    value
        .max(1)
        .checked_next_power_of_two()
        .ok_or("CUDA index capacity overflow".to_string())
}
