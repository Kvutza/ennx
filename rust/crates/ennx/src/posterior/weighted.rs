use super::*;

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

pub fn compute_posterior(
    model: &ENN,
    data: WeightedPosteriorData<'_>,
    y_scale_override: Option<&ArrayView1<'_, f64>>,
) -> Result<DrawInternals, ENNError> {
    let y_scale: Array1<f64> = y_scale_override
        .map(|v| v.to_owned())
        .unwrap_or_else(|| model.output_scale().clone());

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
                if neighbor_idx >= n_train {
                    continue;
                }
                {
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

    let stats = compute_impl(
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
fn stats_weights(
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
                se_components(epistemic_var, aleatoric_var, y_scale[m]);
            se[[i, m]] = se_val;
            se_epi[[i, m]] = se_epi_val;
            se_ale[[i, m]] = se_ale_val;
        }
    }

    (w_normalized, l2, mu, se, se_epi, se_ale)
}

pub fn compute_impl(
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
    let epistemic_scale = params.epistemic_scale;
    let aleatoric_scale = params.aleatoric_scale;
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

    let (w_normalized, l2, mu, se, se_epi, se_ale) = stats_weights(
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
