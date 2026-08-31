//! Posterior computation for ENN model.

mod draw_compute;
mod light;
#[cfg(all(target_os = "macos", feature = "metal"))]
mod metal;
mod neighbor;
pub mod neighbor_dist;
mod tie_break;

use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2};

use self::draw_compute::draw_from_internals;
#[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
use self::light::search_k;
use self::light::{compute_posterior_light, idx_nested_to_array2};
use self::neighbor::{get_conditional_neighbor_data, get_neighbor_data};
use crate::draw::DrawInternals;
use crate::error::{ENNError, EPS_VAR};
use crate::index::IndexDriver;
#[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
use crate::knn::CudaParam;
use crate::model::EpistemicNearestNeighbors;
use crate::params::{ENNNormal, ENNParams, PosteriorFlags};
use crate::stats::WeightedStats;
use crate::traits::PosteriorComputation;

/// Split total variance into `se`, `se_epi`, and `se_ale` per the EPS_VAR floor rule.
pub(crate) fn se_from_variance_components(
    epistemic_var: f64,
    aleatoric_var: f64,
    y_scale: f64,
) -> (f64, f64, f64) {
    let sum = epistemic_var + aleatoric_var;
    if sum < EPS_VAR {
        let se = EPS_VAR.sqrt() * y_scale;
        (se, se, 0.0)
    } else {
        let se = sum.sqrt() * y_scale;
        let se_epi = epistemic_var.sqrt() * y_scale;
        let se_ale = aleatoric_var.sqrt() * y_scale;
        (se, se_epi, se_ale)
    }
}

impl PosteriorComputation for EpistemicNearestNeighbors {
    fn posterior(
        &self,
        x: &ArrayView2<f64>,
        params: &ENNParams,
        flags: &PosteriorFlags,
    ) -> Result<ENNNormal, ENNError> {
        let (mu, se, se_epi, se_ale, idx) = if !flags.observation_noise && !self.has_yvar() {
            compute_posterior_light(self, x, params, flags)?
        } else {
            let internals = compute_posterior_internals(self, x, params, flags)?;
            (
                internals.mu,
                internals.se,
                internals.se_epi,
                internals.se_ale,
                idx_nested_to_array2(&internals.idx),
            )
        };
        let mut mu = mu;
        let mut se = se;
        let mut se_epi = se_epi;
        let mut se_ale = se_ale;
        if self.has_bounded_outputs() {
            crate::y_bounds::naturalize(
                &mut mu,
                &mut se,
                &mut se_epi,
                &mut se_ale,
                self.y_bounds(),
            );
        }
        Ok(ENNNormal::new(
            mu.into_dyn(),
            se.into_dyn(),
            se_epi.into_dyn(),
            se_ale.into_dyn(),
            Some(idx),
        ))
    }

    fn batch_posterior(
        &self,
        x: &ArrayView2<f64>,
        paramss: &[ENNParams],
        flags: &PosteriorFlags,
    ) -> Result<ENNNormal, ENNError> {
        if paramss.is_empty() {
            return Err(ENNError::InvalidParameter(
                "paramss must be non-empty".to_string(),
            ));
        }

        let batch_size = x.nrows();
        let num_params = paramss.len();

        #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
        if self.backend_driver() == IndexDriver::Cuda && self.num_obs() > 0 && !x.is_empty() {
            let skip = usize::from(flags.exclude_nearest);
            if flags.exclude_nearest && self.num_obs() <= 1 {
                return Err(ENNError::InvalidParameter(format!(
                    "exclude_nearest=True requires at least 2 observations, got {}",
                    self.num_obs()
                )));
            }
            let input_k = paramss
                .iter()
                .map(|params| params.k_num_neighbors as usize + skip)
                .max()
                .unwrap_or(0)
                .min(self.num_obs());
            let capacity = self.num_obs().saturating_sub(skip);
            let values: Vec<CudaParam> = paramss
                .iter()
                .map(|params| CudaParam {
                    used_k: (params.k_num_neighbors as usize).min(capacity),
                    epistemic_scale: params.epistemic_variance_scale,
                    aleatoric_scale: params.aleatoric_variance_scale,
                })
                .collect();
            self.ensure_index_sync()?;
            if let (Some(index), Some(train_y)) =
                (self.backend.in_memory_index(), self.train_y_view_opt())
            {
                let train_yvar = self.backend.memory_yvar();
                let (mut mu, mut se, mut se_epi, mut se_ale) = index
                    .cuda_batch(
                        x,
                        &train_y,
                        train_yvar.as_ref(),
                        &self.y_scale().view(),
                        input_k,
                        skip,
                        &values,
                        flags.observation_noise,
                    )
                    .map_err(|error| ENNError::InvalidParameter(error.to_string()))?;
                if self.has_bounded_outputs() {
                    crate::y_bounds::naturalize_batch(
                        &mut mu,
                        &mut se,
                        &mut se_epi,
                        &mut se_ale,
                        self.y_bounds(),
                    );
                }
                return Ok(ENNNormal::new(
                    mu.into_dyn(),
                    se.into_dyn(),
                    se_epi.into_dyn(),
                    se_ale.into_dyn(),
                    None,
                ));
            }
        }

        let mut mu_all = Array3::zeros((num_params, batch_size, self.num_metrics()));
        let mut se_all = Array3::zeros((num_params, batch_size, self.num_metrics()));
        let mut se_epi_all = Array3::zeros((num_params, batch_size, self.num_metrics()));
        let mut se_ale_all = Array3::zeros((num_params, batch_size, self.num_metrics()));

        let k_values: std::collections::HashSet<i32> =
            paramss.iter().map(|p| p.k_num_neighbors).collect();

        if k_values.len() == 1 && self.num_obs() > 0 {
            shared_batch(
                self,
                x,
                paramss,
                flags,
                &mut mu_all,
                &mut se_all,
                &mut se_epi_all,
                &mut se_ale_all,
            )?;
        } else {
            separate_batch(
                self,
                x,
                paramss,
                flags,
                &mut mu_all,
                &mut se_all,
                &mut se_epi_all,
                &mut se_ale_all,
            )?;
        }

        if self.has_bounded_outputs() {
            crate::y_bounds::naturalize_batch(
                &mut mu_all,
                &mut se_all,
                &mut se_epi_all,
                &mut se_ale_all,
                self.y_bounds(),
            );
        }
        Ok(ENNNormal::new(
            mu_all.into_dyn(),
            se_all.into_dyn(),
            se_epi_all.into_dyn(),
            se_ale_all.into_dyn(),
            None,
        ))
    }

    fn posterior_function_draw(
        &self,
        x: &ArrayView2<f64>,
        params: &ENNParams,
        function_seeds: &[i64],
        flags: &PosteriorFlags,
    ) -> Result<(Array3<f64>, Vec<Vec<usize>>), ENNError> {
        #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
        if self.backend_driver() == IndexDriver::Cuda
            && self.num_obs() > 0
            && !x.is_empty()
            && !function_seeds.is_empty()
        {
            let (input_k, used_k, skip) = cuda_shape(self, params, flags.exclude_nearest)?;
            self.ensure_index_sync()?;
            if let (Some(index), Some(train_y)) =
                (self.backend.in_memory_index(), self.train_y_view_opt())
            {
                let train_yvar = self.backend.memory_yvar();
                let (mut draws, indices) = index
                    .cuda_draws(
                        x,
                        &train_y,
                        train_yvar.as_ref(),
                        &self.y_scale().view(),
                        input_k,
                        used_k,
                        skip,
                        params.epistemic_variance_scale,
                        params.aleatoric_variance_scale,
                        flags.observation_noise,
                        function_seeds,
                    )
                    .map_err(|error| ENNError::InvalidParameter(error.to_string()))?;
                if self.has_bounded_outputs() {
                    crate::y_bounds::inverse_draws(&mut draws, self.y_bounds());
                }
                return Ok((draws, indices));
            }
        }
        let internals = compute_posterior_internals(self, x, params, flags)?;
        let mut draws = draw_from_internals(self, &internals, function_seeds)?;
        if self.has_bounded_outputs() {
            crate::y_bounds::inverse_draws(&mut draws, self.y_bounds());
        }
        Ok((draws, internals.idx))
    }

    fn conditional_posterior(
        &self,
        x_whatif: &ArrayView2<f64>,
        y_whatif: &ArrayView2<f64>,
        x: &ArrayView2<f64>,
        params: &ENNParams,
        flags: &PosteriorFlags,
    ) -> Result<ENNNormal, ENNError> {
        let y_whatif = if self.has_bounded_outputs() {
            crate::y_bounds::warp_y(*y_whatif, self.y_bounds())?
        } else {
            y_whatif.to_owned()
        };
        let internals = compute_conditional_posterior_internals(
            self,
            x,
            x_whatif,
            &y_whatif.view(),
            params,
            flags,
        )?;
        let mut mu = internals.mu;
        let mut se = internals.se;
        let mut se_epi = internals.se_epi;
        let mut se_ale = internals.se_ale;
        if self.has_bounded_outputs() {
            crate::y_bounds::naturalize(
                &mut mu,
                &mut se,
                &mut se_epi,
                &mut se_ale,
                self.y_bounds(),
            );
        }
        Ok(ENNNormal::new(
            mu.into_dyn(),
            se.into_dyn(),
            se_epi.into_dyn(),
            se_ale.into_dyn(),
            Some(idx_nested_to_array2(&internals.idx)),
        ))
    }

    fn conditional_posterior_function_draw(
        &self,
        x_whatif: &ArrayView2<f64>,
        y_whatif: &ArrayView2<f64>,
        x: &ArrayView2<f64>,
        params: &ENNParams,
        function_seeds: &[i64],
        flags: &PosteriorFlags,
    ) -> Result<(Array3<f64>, Vec<Vec<usize>>), ENNError> {
        let y_whatif = if self.has_bounded_outputs() {
            crate::y_bounds::warp_y(*y_whatif, self.y_bounds())?
        } else {
            y_whatif.to_owned()
        };
        if x_whatif.nrows() == 0 {
            return self.posterior_function_draw(x, params, function_seeds, flags);
        }
        check_conditional(self, x, x_whatif, &y_whatif.view())?;
        #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
        if self.backend_driver() == IndexDriver::Cuda && !x.is_empty() && !function_seeds.is_empty()
        {
            let (input_k, used_k, skip) =
                condition_shape(self, x_whatif.nrows(), params, flags.exclude_nearest)?;
            self.ensure_index_sync()?;
            if let Some(index) = self.backend.in_memory_index() {
                let (outcomes, variances, scales) = cuda_rows(self, &y_whatif.view())?;
                let variance_view = variances.as_ref().map(Array2::view);
                let (mut draws, indices) = index
                    .condition_draws(
                        x,
                        x_whatif,
                        &outcomes.view(),
                        variance_view.as_ref(),
                        &scales.view(),
                        input_k,
                        used_k,
                        skip,
                        params.epistemic_variance_scale,
                        params.aleatoric_variance_scale,
                        flags.observation_noise,
                        function_seeds,
                    )
                    .map_err(|error| ENNError::InvalidParameter(error.to_string()))?;
                if self.has_bounded_outputs() {
                    crate::y_bounds::inverse_draws(&mut draws, self.y_bounds());
                }
                return Ok((draws, indices));
            }
        }
        let internals = compute_conditional_posterior_internals(
            self,
            x,
            x_whatif,
            &y_whatif.view(),
            params,
            flags,
        )?;
        let mut draws = draw_from_internals(self, &internals, function_seeds)?;
        if self.has_bounded_outputs() {
            crate::y_bounds::inverse_draws(&mut draws, self.y_bounds());
        }
        Ok((draws, internals.idx))
    }
}

pub(crate) fn index_search(
    model: &EpistemicNearestNeighbors,
    x: &ArrayView2<f64>,
    search_k: i32,
    exclude_nearest: bool,
    tie_break_neighbors: bool,
) -> Result<(Array2<f64>, Array2<i64>), ENNError> {
    if !model.backend.defer_index_sync_for_search() {
        model.ensure_index_sync()?;
    }
    if model.backend_driver() == IndexDriver::Exact {
        neighbor::exact_f64_batch_topk(model, x, search_k, exclude_nearest, tie_break_neighbors)
    } else {
        let (_, idx) = model.backend_search(x, search_k, exclude_nearest)?;
        let dist2s = neighbor::dist2s_for_neighbor_indices(model, x, &idx);
        Ok((dist2s, idx))
    }
}

mod batch;
use batch::{separate_batch, shared_batch};
/// Data for weighted posterior computation.
pub struct WeightedPosteriorData<'a> {
    pub dist2s: &'a ArrayView2<'a, f64>,
    pub idx: &'a [Vec<usize>],
    pub y_neighbors: &'a ArrayView2<'a, f64>,
    pub params: &'a ENNParams,
    pub observation_noise: bool,
    /// When set, use this instead of gathering from model (for conditional with whatif).
    pub yvar_neighbors_override: Option<&'a Array2<f64>>,
}

pub fn compute_weighted_posterior(
    model: &EpistemicNearestNeighbors,
    data: WeightedPosteriorData<'_>,
    y_scale_override: Option<&ArrayView1<'_, f64>>,
) -> Result<DrawInternals, ENNError> {
    let y_scale: Array1<f64> = y_scale_override
        .map(|v| v.to_owned())
        .unwrap_or_else(|| model.y_scale().clone());

    let yvar_neighbors: Option<Array2<f64>> = if let Some(ov) = data.yvar_neighbors_override {
        Some(ov.clone())
    } else if model.has_yvar() {
        let n_query = data.dist2s.nrows();
        // Handle empty query case (n_query == 0 or data.idx is empty)
        let k = if data.idx.is_empty() {
            0
        } else {
            data.idx[0].len()
        };
        let n_train = model.num_obs();
        let mut yvar_neighbors = Array2::zeros((n_query * k, model.num_metrics()));
        for i in 0..n_query {
            for (j, &neighbor_idx) in data.idx[i].iter().enumerate() {
                if neighbor_idx < n_train {
                    let yvar_row = model
                        .rows()
                        .row_yvar(neighbor_idx)
                        .expect("row_yvar")
                        .expect("yvar");
                    for m in 0..model.num_metrics() {
                        yvar_neighbors[[i * k + j, m]] = yvar_row[m];
                    }
                }
                // else: whatif point, keep 0
            }
        }
        Some(yvar_neighbors)
    } else {
        None
    };

    let stats = compute_weighted_stats_impl(
        data.dist2s,
        data.y_neighbors,
        yvar_neighbors.as_ref().map(|v| v.view()),
        data.params,
        data.observation_noise,
        &y_scale.view(),
    )?;

    Ok(DrawInternals::new(
        data.idx.to_vec(),
        stats.w_normalized,
        stats.l2,
        stats.mu,
        stats.se,
        stats.se_epi,
        stats.se_ale,
    ))
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn stats_from_neighbor_weights(
    w: &Array2<f64>,
    y_neighbors: &ArrayView2<f64>,
    yvar_neighbors: Option<ArrayView2<f64>>,
    n_query: usize,
    k: usize,
    num_metrics: usize,
    observation_noise: bool,
    aleatoric_scale: f64,
    y_scale: &ArrayView1<f64>,
    y_scale_sq: &[f64],
) -> (
    Array3<f64>,
    Array2<f64>,
    Array2<f64>,
    Array2<f64>,
    Array2<f64>,
    Array2<f64>,
) {
    let yvar_ref = yvar_neighbors.as_ref();
    let mut w_normalized = Array3::zeros((n_query, k, num_metrics));
    let mut l2 = Array2::zeros((n_query, num_metrics));
    let mut mu = Array2::zeros((n_query, num_metrics));
    let mut se = Array2::zeros((n_query, num_metrics));
    let mut se_epi = Array2::zeros((n_query, num_metrics));
    let mut se_ale = Array2::zeros((n_query, num_metrics));

    for i in 0..n_query {
        for m in 0..num_metrics {
            let base_idx = i * k;
            let norm: f64 = (0..k).map(|j| w[[base_idx + j, m]]).sum();
            let inv_norm = 1.0 / norm;

            let mut l2_sq = 0.0;
            let mut mu_val = 0.0;

            for j in 0..k {
                let w_norm = w[[base_idx + j, m]] * inv_norm;
                w_normalized[[i, j, m]] = w_norm;
                l2_sq += w_norm * w_norm;
                mu_val += w_norm * y_neighbors[[base_idx + j, m]];
            }

            l2[[i, m]] = l2_sq.sqrt();
            mu[[i, m]] = mu_val;

            let epistemic_var = inv_norm;

            let aleatoric_var = if observation_noise {
                let mut sum = 0.0;
                for j in 0..k {
                    let var_ale_j = aleatoric_scale
                        + if let Some(yv) = yvar_ref {
                            yv[[base_idx + j, m]] / y_scale_sq[m]
                        } else {
                            0.0
                        };
                    sum += w_normalized[[i, j, m]] * var_ale_j;
                }
                sum
            } else {
                0.0
            };

            let (se_val, se_epi_val, se_ale_val) =
                se_from_variance_components(epistemic_var, aleatoric_var, y_scale[m]);
            se[[i, m]] = se_val;
            se_epi[[i, m]] = se_epi_val;
            se_ale[[i, m]] = se_ale_val;
        }
    }

    (w_normalized, l2, mu, se, se_epi, se_ale)
}

pub fn compute_weighted_stats_impl(
    dist2s: &ArrayView2<f64>,
    y_neighbors: &ArrayView2<f64>,
    yvar_neighbors: Option<ArrayView2<f64>>,
    params: &ENNParams,
    observation_noise: bool,
    y_scale: &ArrayView1<f64>,
) -> Result<WeightedStats, ENNError> {
    let n_query = dist2s.nrows();
    let k = dist2s.ncols();
    let num_metrics = y_scale.len();

    // Hoist constants outside loops
    let epistemic_scale = params.epistemic_variance_scale;
    let aleatoric_scale = params.aleatoric_variance_scale;
    let y_scale_sq: Vec<f64> = y_scale.iter().map(|&v| v * v).collect();

    // Pre-compute var_epi using iterator-based zip for better cache efficiency
    let mut var_epi = Array2::zeros((n_query, k));
    for (i, mut row) in var_epi.rows_mut().into_iter().enumerate() {
        let dist_row = dist2s.row(i);
        for (j, v) in row.iter_mut().enumerate() {
            *v = epistemic_scale * dist_row[j];
        }
    }

    // Compute weights w with hoisted constants and pre-allocated storage
    let mut w = Array2::zeros((n_query * k, num_metrics));
    let yvar_ref = yvar_neighbors.as_ref();

    for i in 0..n_query {
        let var_epi_row = var_epi.row(i);
        for j in 0..k {
            let var_epi_ij = var_epi_row[j];
            let idx = i * k + j;
            for m in 0..num_metrics {
                let var_y = if let Some(yv) = yvar_ref {
                    yv[[idx, m]] / y_scale_sq[m]
                } else {
                    0.0
                };
                let var_total = EPS_VAR + var_epi_ij + aleatoric_scale + var_y;
                w[[idx, m]] = 1.0 / var_total;
            }
        }
    }

    let (w_normalized, l2, mu, se, se_epi, se_ale) = stats_from_neighbor_weights(
        &w,
        y_neighbors,
        yvar_neighbors,
        n_query,
        k,
        num_metrics,
        observation_noise,
        aleatoric_scale,
        y_scale,
        &y_scale_sq,
    );

    Ok(WeightedStats::new(w_normalized, l2, mu, se, se_epi, se_ale))
}

pub fn compute_posterior_internals(
    model: &EpistemicNearestNeighbors,
    x: &ArrayView2<f64>,
    params: &ENNParams,
    flags: &PosteriorFlags,
) -> Result<DrawInternals, ENNError> {
    if x.ncols() != model.num_dim() {
        return Err(ENNError::InvalidShape {
            expected: vec![x.nrows(), model.num_dim()],
            got: x.shape().to_vec(),
        });
    }

    let batch_size = x.nrows();

    if model.num_obs() == 0 {
        return Ok(empty_posterior_internals(model, batch_size));
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
    if model.backend_driver() == IndexDriver::Cuda && !x.is_empty() {
        let (input_k, used_k, skip) = cuda_shape(model, params, flags.exclude_nearest)?;
        if used_k > 0 {
            model.ensure_index_sync()?;
            if let (Some(index), Some(train_y)) =
                (model.backend.in_memory_index(), model.train_y_view_opt())
            {
                let train_yvar = model.backend.memory_yvar();
                return index
                    .cuda_weighted(
                        x,
                        &train_y,
                        train_yvar.as_ref(),
                        &model.y_scale().view(),
                        input_k,
                        used_k,
                        skip,
                        params.epistemic_variance_scale,
                        params.aleatoric_variance_scale,
                        flags.observation_noise,
                    )
                    .map_err(|error| ENNError::InvalidParameter(error.to_string()));
            }
        }
    }

    let neighbor_data = get_neighbor_data(
        model,
        x,
        params,
        flags.exclude_nearest,
        flags.tie_break_neighbors,
    )?;

    if let Some(data) = neighbor_data {
        let wp_data = WeightedPosteriorData {
            dist2s: &data.dist2s.view(),
            idx: &data.idx,
            y_neighbors: &data.y_neighbors.view(),
            params,
            observation_noise: flags.observation_noise,
            yvar_neighbors_override: None,
        };
        compute_weighted_posterior(model, wp_data, None)
    } else {
        Ok(empty_posterior_internals(model, batch_size))
    }
}

#[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
fn cuda_shape(
    model: &EpistemicNearestNeighbors,
    params: &ENNParams,
    exclude_nearest: bool,
) -> Result<(usize, usize, usize), ENNError> {
    let input_k = search_k(model, params, exclude_nearest)?;
    let skip = usize::from(exclude_nearest);
    let used_k = (params.k_num_neighbors as usize).min(input_k.saturating_sub(skip));
    Ok((input_k, used_k, skip))
}

fn compute_scale_for_conditional(
    train_y: &ArrayView2<f64>,
    y_whatif: &ArrayView2<f64>,
) -> Array1<f64> {
    let n1 = train_y.nrows();
    let n2 = y_whatif.nrows();
    let m = train_y.ncols();
    let mut stacked = Array2::zeros((n1 + n2, m));
    stacked.slice_mut(ndarray::s![..n1, ..]).assign(train_y);
    stacked.slice_mut(ndarray::s![n1.., ..]).assign(y_whatif);

    if stacked.nrows() < 2 {
        return Array1::ones(stacked.ncols());
    }
    let mut scale = Array1::zeros(stacked.ncols());
    for j in 0..stacked.ncols() {
        let col = stacked.column(j);
        let var = col.var(0.0);
        let std = var.sqrt();
        scale[j] = if std.is_finite() && std > 0.0 {
            std
        } else {
            1.0
        };
    }
    scale
}

pub fn compute_conditional_posterior_internals(
    model: &EpistemicNearestNeighbors,
    x: &ArrayView2<f64>,
    x_whatif: &ArrayView2<f64>,
    y_whatif: &ArrayView2<f64>,
    params: &ENNParams,
    flags: &PosteriorFlags,
) -> Result<DrawInternals, ENNError> {
    if x_whatif.nrows() == 0 {
        return compute_posterior_internals(model, x, params, flags);
    }

    check_conditional(model, x, x_whatif, y_whatif)?;

    let batch_size = x.nrows();
    #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
    if model.backend_driver() == IndexDriver::Cuda && !x.is_empty() {
        let (input_k, used_k, skip) =
            condition_shape(model, x_whatif.nrows(), params, flags.exclude_nearest)?;
        model.ensure_index_sync()?;
        if let Some(index) = model.backend.in_memory_index() {
            let (outcomes, variances, scales) = cuda_rows(model, y_whatif)?;
            let variance_view = variances.as_ref().map(Array2::view);
            return index
                .cuda_conditional(
                    x,
                    x_whatif,
                    &outcomes.view(),
                    variance_view.as_ref(),
                    &scales.view(),
                    input_k,
                    used_k,
                    skip,
                    params.epistemic_variance_scale,
                    params.aleatoric_variance_scale,
                    flags.observation_noise,
                )
                .map_err(|error| ENNError::InvalidParameter(error.to_string()));
        }
    }
    let neighbor_data = get_conditional_neighbor_data(model, x, x_whatif, y_whatif, params, flags)?;

    if let Some(data) = neighbor_data {
        let all_y_indices: Vec<usize> = (0..model.num_obs()).collect();
        let (_, train_y_all, _) = model.rows().train_rows_at(&all_y_indices)?;
        let y_scale_cond = compute_scale_for_conditional(&train_y_all.view(), y_whatif);
        let wp_data = WeightedPosteriorData {
            dist2s: &data.dist2s.view(),
            idx: &data.idx,
            y_neighbors: &data.y_neighbors.view(),
            params,
            observation_noise: flags.observation_noise,
            yvar_neighbors_override: None,
        };
        compute_weighted_posterior(model, wp_data, Some(&y_scale_cond.view()))
    } else {
        Ok(empty_posterior_internals(model, batch_size))
    }
}

fn check_conditional(
    model: &EpistemicNearestNeighbors,
    x: &ArrayView2<f64>,
    x_whatif: &ArrayView2<f64>,
    y_whatif: &ArrayView2<f64>,
) -> Result<(), ENNError> {
    if x.iter().any(|value| !value.is_finite())
        || x_whatif.iter().any(|value| !value.is_finite())
        || y_whatif.iter().any(|value| !value.is_finite())
    {
        return Err(ENNError::InvalidParameter(
            "NaN or Inf not allowed in x, x_whatif, or y_whatif".to_string(),
        ));
    }
    if x.ncols() != model.num_dim() {
        return Err(ENNError::InvalidShape {
            expected: vec![x.nrows(), model.num_dim()],
            got: x.shape().to_vec(),
        });
    }
    if x_whatif.ncols() != model.num_dim() {
        return Err(ENNError::InvalidShape {
            expected: vec![x_whatif.nrows(), model.num_dim()],
            got: x_whatif.shape().to_vec(),
        });
    }
    if y_whatif.ncols() != model.num_metrics() {
        return Err(ENNError::InvalidShape {
            expected: vec![y_whatif.nrows(), model.num_metrics()],
            got: y_whatif.shape().to_vec(),
        });
    }
    if x_whatif.nrows() != y_whatif.nrows() {
        return Err(ENNError::InvalidParameter(
            "x_whatif and y_whatif must have same number of rows".to_string(),
        ));
    }
    Ok(())
}

#[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
fn condition_shape(
    model: &EpistemicNearestNeighbors,
    whatif_rows: usize,
    params: &ENNParams,
    exclude_nearest: bool,
) -> Result<(usize, usize, usize), ENNError> {
    let total = model
        .num_obs()
        .checked_add(whatif_rows)
        .ok_or_else(|| ENNError::InvalidParameter("conditional row count overflow".to_string()))?;
    if exclude_nearest && total <= 1 {
        return Err(ENNError::InvalidParameter(format!(
            "exclude_nearest=True requires at least 2 observations, got {total}"
        )));
    }
    let skip = usize::from(exclude_nearest);
    let input_k = (params.k_num_neighbors as usize + skip).min(total);
    let used_k = (params.k_num_neighbors as usize).min(input_k.saturating_sub(skip));
    Ok((input_k, used_k, skip))
}

#[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
fn cuda_rows(
    model: &EpistemicNearestNeighbors,
    y_whatif: &ArrayView2<f64>,
) -> Result<(Array2<f64>, Option<Array2<f64>>, Array1<f64>), ENNError> {
    let train_y = model.train_y_view_opt().ok_or_else(|| {
        ENNError::InvalidParameter("CUDA conditional rows require in-memory storage".to_string())
    })?;
    let train_rows = train_y.nrows();
    let total_rows = train_rows
        .checked_add(y_whatif.nrows())
        .ok_or_else(|| ENNError::InvalidParameter("conditional row count overflow".to_string()))?;
    let mut outcomes = Array2::zeros((total_rows, model.num_metrics()));
    outcomes
        .slice_mut(ndarray::s![..train_rows, ..])
        .assign(&train_y);
    outcomes
        .slice_mut(ndarray::s![train_rows.., ..])
        .assign(y_whatif);
    let variances = model.backend.memory_yvar().map(|train_yvar| {
        let mut values = Array2::zeros((total_rows, model.num_metrics()));
        values
            .slice_mut(ndarray::s![..train_rows, ..])
            .assign(&train_yvar);
        values
    });
    let scales = compute_scale_for_conditional(&train_y, y_whatif);
    Ok((outcomes, variances, scales))
}

pub fn empty_posterior_internals(
    model: &EpistemicNearestNeighbors,
    batch_size: usize,
) -> DrawInternals {
    DrawInternals::new(
        vec![vec![]; batch_size],
        Array3::zeros((batch_size, 0, model.num_metrics())),
        Array2::ones((batch_size, model.num_metrics())),
        Array2::zeros((batch_size, model.num_metrics())),
        Array2::ones((batch_size, model.num_metrics())),
        Array2::ones((batch_size, model.num_metrics())),
        Array2::zeros((batch_size, model.num_metrics())),
    )
}

#[cfg(test)]
#[path = "posterior/tests_core.rs"]
mod tests_core;

#[cfg(test)]
#[path = "posterior/tests_conditional.rs"]
mod tests_conditional;
