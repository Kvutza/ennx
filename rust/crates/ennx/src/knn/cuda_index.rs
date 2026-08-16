use std::time::Duration;

use ndarray::{Array2, Array3, ArrayView1, ArrayView2};

use super::{
    arr2_rows_to_f32, pad_neighbor_cols_to_search_k, CudaParam, KnnBatch, KnnPosterior, KnnProfile,
};
use crate::draw::DrawInternals;
use crate::index::IndexError;

pub(crate) struct CudaIndex {
    inner: ennx_cuda::CudaIndex,
    dims: usize,
}

impl CudaIndex {
    pub(crate) fn new(dims: usize, train: &ArrayView2<f64>) -> Result<Self, IndexError> {
        check_shape(dims, train)?;
        Ok(Self {
            inner: ennx_cuda::CudaIndex::new(dims, &arr2_rows_to_f32(train))
                .map_err(index_error)?,
            dims,
        })
    }

    pub(crate) fn len(&self) -> usize {
        self.inner.len()
    }

    pub(crate) fn memory_bytes(&self) -> usize {
        self.inner.memory_bytes()
    }

    pub(crate) fn rebuild(&mut self, train: &ArrayView2<f64>) -> Result<(), IndexError> {
        check_shape(self.dims, train)?;
        self.inner
            .rebuild(&arr2_rows_to_f32(train))
            .map_err(index_error)
    }

    pub(crate) fn add(&mut self, rows: &ArrayView2<f64>, start_key: u64) -> Result<(), IndexError> {
        check_shape(self.dims, rows)?;
        if start_key != self.len() as u64 {
            return Err(IndexError::InvalidParameter(format!(
                "CUDA index expected start key {}, got {start_key}",
                self.len()
            )));
        }
        self.inner.add(&arr2_rows_to_f32(rows)).map_err(index_error)
    }

    pub(crate) fn search(
        &mut self,
        queries: &ArrayView2<f64>,
        k_eff: usize,
        search_k: usize,
    ) -> Result<(Array2<f64>, Array2<i64>), IndexError> {
        check_shape(self.dims, queries)?;
        if k_eff == 0 {
            return Err(IndexError::InvalidParameter(format!(
                "CUDA index requires at least one neighbor, got {k_eff}"
            )));
        }
        if queries.nrows() == 0 || self.len() == 0 {
            return Ok(pad_neighbor_cols_to_search_k(
                Array2::from_elem((queries.nrows(), 0), f64::INFINITY),
                Array2::zeros((queries.nrows(), 0)),
                search_k,
            ));
        }
        let k_eff = k_eff.min(self.len());
        if k_eff > ennx_cuda::KNN_MAX_K {
            return Err(IndexError::InvalidParameter(format!(
                "CUDA index supports at most {} neighbors, got {k_eff}",
                ennx_cuda::KNN_MAX_K
            )));
        }

        let (distances, indices) = self
            .inner
            .search(&arr2_rows_to_f32(queries), queries.nrows(), k_eff)
            .map_err(index_error)?;
        let distances = Array2::from_shape_vec(
            (queries.nrows(), k_eff),
            distances.into_iter().map(f64::from).collect(),
        )
        .map_err(|error| IndexError::InvalidParameter(error.to_string()))?;
        let indices = Array2::from_shape_vec(
            (queries.nrows(), k_eff),
            indices.into_iter().map(i64::from).collect(),
        )
        .map_err(|error| IndexError::InvalidParameter(error.to_string()))?;
        Ok(pad_neighbor_cols_to_search_k(distances, indices, search_k))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn posterior(
        &mut self,
        queries: &ArrayView2<f64>,
        outcomes: &ArrayView2<f64>,
        scales: &ArrayView1<f64>,
        input_k: usize,
        used_k: usize,
        skip: usize,
        epistemic_scale: f64,
        aleatoric_scale: f64,
    ) -> Result<KnnPosterior, IndexError> {
        check_shape(self.dims, queries)?;
        if outcomes.nrows() != self.len() || outcomes.ncols() != scales.len() {
            return Err(IndexError::InvalidParameter(
                "CUDA posterior outcomes and scales have incompatible shapes".to_string(),
            ));
        }
        let output = self
            .inner
            .posterior(
                &arr2_rows_to_f32(queries),
                queries.nrows(),
                &arr2_rows_to_f32(outcomes),
                &scales.iter().map(|&value| value as f32).collect::<Vec<_>>(),
                ennx_cuda::PosteriorSpec {
                    metrics: outcomes.ncols(),
                    input_k,
                    used_k,
                    skip,
                    epistemic_scale: epistemic_scale as f32,
                    aleatoric_scale: aleatoric_scale as f32,
                    epsilon: crate::error::EPS_VAR as f32,
                },
            )
            .map_err(index_error)?;
        let shape = (queries.nrows(), outcomes.ncols());
        let mu = Array2::from_shape_vec(shape, output.means.into_iter().map(f64::from).collect())
            .map_err(|error| IndexError::InvalidParameter(error.to_string()))?;
        let se = Array2::from_shape_vec(shape, output.errors.into_iter().map(f64::from).collect())
            .map_err(|error| IndexError::InvalidParameter(error.to_string()))?;
        let idx = Array2::from_shape_vec(
            (queries.nrows(), used_k),
            output.indices.into_iter().map(i64::from).collect(),
        )
        .map_err(|error| IndexError::InvalidParameter(error.to_string()))?;
        Ok((mu, se.clone(), se, Array2::zeros(shape), idx))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn weighted(
        &mut self,
        queries: &ArrayView2<f64>,
        outcomes: &ArrayView2<f64>,
        variances: Option<&ArrayView2<f64>>,
        scales: &ArrayView1<f64>,
        input_k: usize,
        used_k: usize,
        skip: usize,
        epistemic_scale: f64,
        aleatoric_scale: f64,
        observation_noise: bool,
    ) -> Result<DrawInternals, IndexError> {
        check_shape(self.dims, queries)?;
        if outcomes.nrows() != self.len() || outcomes.ncols() != scales.len() {
            return Err(IndexError::InvalidParameter(
                "CUDA weighted outcomes and scales have incompatible shapes".to_string(),
            ));
        }
        if variances.is_some_and(|values| values.dim() != outcomes.dim()) {
            return Err(IndexError::InvalidParameter(
                "CUDA weighted variances have an incompatible shape".to_string(),
            ));
        }
        let outcomes_f32 = arr2_rows_to_f32(outcomes);
        let variances_f32 = variances.map(arr2_rows_to_f32);
        let scales_f32: Vec<f32> = scales.iter().map(|&value| value as f32).collect();
        let output = self
            .inner
            .weighted(
                &arr2_rows_to_f32(queries),
                queries.nrows(),
                &outcomes_f32,
                variances_f32.as_deref(),
                &scales_f32,
                ennx_cuda::WeightedSpec {
                    metrics: outcomes.ncols(),
                    input_k,
                    used_k,
                    skip,
                    epistemic_scale: epistemic_scale as f32,
                    aleatoric_scale: aleatoric_scale as f32,
                    epsilon: crate::error::EPS_VAR as f32,
                    observation_noise,
                },
            )
            .map_err(index_error)?;
        weighted_output(output, queries.nrows(), outcomes.ncols(), used_k)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn batch(
        &mut self,
        queries: &ArrayView2<f64>,
        outcomes: &ArrayView2<f64>,
        variances: Option<&ArrayView2<f64>>,
        scales: &ArrayView1<f64>,
        input_k: usize,
        skip: usize,
        values: &[CudaParam],
        observation_noise: bool,
    ) -> Result<KnnBatch, IndexError> {
        check_shape(self.dims, queries)?;
        if outcomes.nrows() != self.len() || outcomes.ncols() != scales.len() {
            return Err(IndexError::InvalidParameter(
                "CUDA batch outcomes and scales have incompatible shapes".to_string(),
            ));
        }
        if variances.is_some_and(|array| array.dim() != outcomes.dim()) {
            return Err(IndexError::InvalidParameter(
                "CUDA batch variances have an incompatible shape".to_string(),
            ));
        }
        let outcomes_f32 = arr2_rows_to_f32(outcomes);
        let variances_f32 = variances.map(arr2_rows_to_f32);
        let scales_f32: Vec<f32> = scales.iter().map(|&value| value as f32).collect();
        let cuda_values: Vec<ennx_cuda::BatchValue> = values
            .iter()
            .map(|value| ennx_cuda::BatchValue {
                used_k: value.used_k as u32,
                skip: skip as u32,
                epistemic_scale: value.epistemic_scale as f32,
                aleatoric_scale: value.aleatoric_scale as f32,
            })
            .collect();
        let output = self
            .inner
            .batch(
                &arr2_rows_to_f32(queries),
                queries.nrows(),
                &outcomes_f32,
                variances_f32.as_deref(),
                &scales_f32,
                &cuda_values,
                ennx_cuda::BatchSpec {
                    metrics: outcomes.ncols(),
                    input_k,
                    epsilon: crate::error::EPS_VAR as f32,
                    observation_noise,
                },
            )
            .map_err(index_error)?;
        let shape = (values.len(), queries.nrows(), outcomes.ncols());
        let array = |data: Vec<f32>| {
            Array3::from_shape_vec(shape, data.into_iter().map(f64::from).collect())
                .map_err(|error| IndexError::InvalidParameter(error.to_string()))
        };
        Ok((
            array(output.means)?,
            array(output.errors)?,
            array(output.epistemic)?,
            array(output.aleatoric)?,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn draws(
        &mut self,
        queries: &ArrayView2<f64>,
        outcomes: &ArrayView2<f64>,
        variances: Option<&ArrayView2<f64>>,
        scales: &ArrayView1<f64>,
        input_k: usize,
        used_k: usize,
        skip: usize,
        epistemic_scale: f64,
        aleatoric_scale: f64,
        observation_noise: bool,
        seeds: &[i64],
    ) -> Result<(Array3<f64>, Vec<Vec<usize>>), IndexError> {
        check_shape(self.dims, queries)?;
        if outcomes.nrows() != self.len() || outcomes.ncols() != scales.len() {
            return Err(IndexError::InvalidParameter(
                "CUDA draw outcomes and scales have incompatible shapes".to_string(),
            ));
        }
        if variances.is_some_and(|array| array.dim() != outcomes.dim()) {
            return Err(IndexError::InvalidParameter(
                "CUDA draw variances have an incompatible shape".to_string(),
            ));
        }
        let outcomes_f32 = arr2_rows_to_f32(outcomes);
        let variances_f32 = variances.map(arr2_rows_to_f32);
        let scales_f32: Vec<f32> = scales.iter().map(|&value| value as f32).collect();
        let seeds_u64: Vec<u64> = seeds.iter().map(|&seed| seed as u64).collect();
        let output = self
            .inner
            .draws(
                &arr2_rows_to_f32(queries),
                queries.nrows(),
                &outcomes_f32,
                variances_f32.as_deref(),
                &scales_f32,
                &seeds_u64,
                ennx_cuda::WeightedSpec {
                    metrics: outcomes.ncols(),
                    input_k,
                    used_k,
                    skip,
                    epistemic_scale: epistemic_scale as f32,
                    aleatoric_scale: aleatoric_scale as f32,
                    epsilon: crate::error::EPS_VAR as f32,
                    observation_noise,
                },
            )
            .map_err(index_error)?;
        draw_output(
            output,
            seeds.len(),
            queries.nrows(),
            outcomes.ncols(),
            used_k,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn conditional(
        &mut self,
        queries: &ArrayView2<f64>,
        extra_rows: &ArrayView2<f64>,
        outcomes: &ArrayView2<f64>,
        variances: Option<&ArrayView2<f64>>,
        scales: &ArrayView1<f64>,
        input_k: usize,
        used_k: usize,
        skip: usize,
        epistemic_scale: f64,
        aleatoric_scale: f64,
        observation_noise: bool,
    ) -> Result<DrawInternals, IndexError> {
        check_shape(self.dims, queries)?;
        check_shape(self.dims, extra_rows)?;
        let total_rows = self
            .len()
            .checked_add(extra_rows.nrows())
            .ok_or_else(|| IndexError::InvalidParameter("CUDA row count overflow".to_string()))?;
        if outcomes.nrows() != total_rows || outcomes.ncols() != scales.len() {
            return Err(IndexError::InvalidParameter(
                "CUDA conditional outcomes and scales have incompatible shapes".to_string(),
            ));
        }
        if variances.is_some_and(|array| array.dim() != outcomes.dim()) {
            return Err(IndexError::InvalidParameter(
                "CUDA conditional variances have an incompatible shape".to_string(),
            ));
        }
        let outcomes_f32 = arr2_rows_to_f32(outcomes);
        let variances_f32 = variances.map(arr2_rows_to_f32);
        let scales_f32: Vec<f32> = scales.iter().map(|&value| value as f32).collect();
        let output = self
            .inner
            .conditional(
                &arr2_rows_to_f32(queries),
                queries.nrows(),
                &arr2_rows_to_f32(extra_rows),
                &outcomes_f32,
                variances_f32.as_deref(),
                &scales_f32,
                ennx_cuda::WeightedSpec {
                    metrics: outcomes.ncols(),
                    input_k,
                    used_k,
                    skip,
                    epistemic_scale: epistemic_scale as f32,
                    aleatoric_scale: aleatoric_scale as f32,
                    epsilon: crate::error::EPS_VAR as f32,
                    observation_noise,
                },
            )
            .map_err(index_error)?;
        weighted_output(output, queries.nrows(), outcomes.ncols(), used_k)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn conditional_draws(
        &mut self,
        queries: &ArrayView2<f64>,
        extra_rows: &ArrayView2<f64>,
        outcomes: &ArrayView2<f64>,
        variances: Option<&ArrayView2<f64>>,
        scales: &ArrayView1<f64>,
        input_k: usize,
        used_k: usize,
        skip: usize,
        epistemic_scale: f64,
        aleatoric_scale: f64,
        observation_noise: bool,
        seeds: &[i64],
    ) -> Result<(Array3<f64>, Vec<Vec<usize>>), IndexError> {
        check_shape(self.dims, queries)?;
        check_shape(self.dims, extra_rows)?;
        let total_rows = self
            .len()
            .checked_add(extra_rows.nrows())
            .ok_or_else(|| IndexError::InvalidParameter("CUDA row count overflow".to_string()))?;
        if outcomes.nrows() != total_rows || outcomes.ncols() != scales.len() {
            return Err(IndexError::InvalidParameter(
                "CUDA conditional draw outcomes have an incompatible shape".to_string(),
            ));
        }
        if variances.is_some_and(|array| array.dim() != outcomes.dim()) {
            return Err(IndexError::InvalidParameter(
                "CUDA conditional draw variances have an incompatible shape".to_string(),
            ));
        }
        let outcomes_f32 = arr2_rows_to_f32(outcomes);
        let variances_f32 = variances.map(arr2_rows_to_f32);
        let scales_f32: Vec<f32> = scales.iter().map(|&value| value as f32).collect();
        let seeds_u64: Vec<u64> = seeds.iter().map(|&seed| seed as u64).collect();
        let output = self
            .inner
            .conditional_draws(
                &arr2_rows_to_f32(queries),
                queries.nrows(),
                &arr2_rows_to_f32(extra_rows),
                &outcomes_f32,
                variances_f32.as_deref(),
                &scales_f32,
                &seeds_u64,
                ennx_cuda::WeightedSpec {
                    metrics: outcomes.ncols(),
                    input_k,
                    used_k,
                    skip,
                    epistemic_scale: epistemic_scale as f32,
                    aleatoric_scale: aleatoric_scale as f32,
                    epsilon: crate::error::EPS_VAR as f32,
                    observation_noise,
                },
            )
            .map_err(index_error)?;
        draw_output(
            output,
            seeds.len(),
            queries.nrows(),
            outcomes.ncols(),
            used_k,
        )
    }

    pub(crate) fn profile(&self) -> Option<KnnProfile> {
        self.inner.profile().map(|profile| KnnProfile {
            rows: profile.rows,
            queries: profile.queries,
            dims: profile.dims,
            k: profile.neighbors,
            plan: profile.kind,
            gpu: duration_ms(profile.total_ms),
            scan: duration_ms(profile.scan_ms),
            select: Duration::ZERO,
            reduce: duration_ms(profile.merge_ms + profile.posterior_ms),
        })
    }
}

fn check_shape(dims: usize, rows: &ArrayView2<f64>) -> Result<(), IndexError> {
    if rows.ncols() != dims {
        Err(IndexError::InvalidShape {
            expected: dims,
            got: rows.ncols(),
        })
    } else if rows.iter().any(|value| !value.is_finite()) {
        Err(IndexError::InvalidParameter(
            "CUDA index values must be finite".to_string(),
        ))
    } else {
        Ok(())
    }
}

fn index_error(error: String) -> IndexError {
    IndexError::InvalidParameter(error)
}

fn duration_ms(milliseconds: f32) -> Duration {
    Duration::from_secs_f64(f64::from(milliseconds) / 1_000.0)
}

fn weighted_output(
    output: ennx_cuda::WeightedOutput,
    queries: usize,
    metrics: usize,
    neighbors: usize,
) -> Result<DrawInternals, IndexError> {
    let value_shape = (queries, metrics);
    let weights = Array3::from_shape_vec(
        (queries, neighbors, metrics),
        output.weights.into_iter().map(f64::from).collect(),
    )
    .map_err(|error| IndexError::InvalidParameter(error.to_string()))?;
    let values = |data: Vec<f32>| {
        Array2::from_shape_vec(value_shape, data.into_iter().map(f64::from).collect())
            .map_err(|error| IndexError::InvalidParameter(error.to_string()))
    };
    let indices = output
        .indices
        .chunks_exact(neighbors)
        .map(|row| row.iter().map(|&value| value as usize).collect())
        .collect();
    Ok(DrawInternals::new(
        indices,
        weights,
        values(output.l2)?,
        values(output.means)?,
        values(output.errors)?,
        values(output.epistemic)?,
        values(output.aleatoric)?,
    ))
}

fn draw_output(
    output: ennx_cuda::DrawOutput,
    seeds: usize,
    queries: usize,
    metrics: usize,
    neighbors: usize,
) -> Result<(Array3<f64>, Vec<Vec<usize>>), IndexError> {
    let draws = Array3::from_shape_vec((seeds, queries, metrics), output.draws)
        .map_err(|error| IndexError::InvalidParameter(error.to_string()))?;
    let indices = output
        .indices
        .chunks_exact(neighbors)
        .map(|row| row.iter().map(|&value| value as usize).collect())
        .collect();
    Ok((draws, indices))
}
