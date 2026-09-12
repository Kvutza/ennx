//! Parameter fitting for ENN models via subsample log-likelihood.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rand::Rng;
use rand::seq::index::sample;

use crate::error::ENNError;
use crate::model::ENN;
use crate::params::ENNParams;
use crate::posterior::{WeightedPosteriorData, compute_posterior, index_search};

const LOCAL_SCALE_FLOOR: f64 = 1.0e-12;

/// Normalize a square matrix of squared distances by per-row local radii.
///
/// Each radius is the `k`th positive-or-zero off-diagonal distance. The
/// geometric mean keeps the result dimensionless while preserving symmetry.
pub fn self_tuned_distances(
    distances: &ArrayView2<f64>,
    k: usize,
) -> Result<Array2<f64>, ENNError> {
    let n = distances.nrows();
    if distances.ncols() != n || k == 0 {
        return Err(ENNError::InvalidParameter(
            "self-tuning distances require a square matrix and positive local k".into(),
        ));
    }
    if distances
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(ENNError::InvalidParameter(
            "self-tuning distances must be finite and nonnegative".into(),
        ));
    }
    if n <= 1 {
        return Ok(distances.to_owned());
    }
    let scales = (0..n)
        .map(|row| {
            let mut values = (0..n)
                .filter(|&column| column != row)
                .map(|column| distances[[row, column]])
                .collect::<Vec<_>>();
            values.sort_by(f64::total_cmp);
            values[k.min(values.len()) - 1].max(LOCAL_SCALE_FLOOR)
        })
        .collect::<Vec<_>>();
    Ok(Array2::from_shape_fn((n, n), |(row, column)| {
        if row == column {
            0.0
        } else {
            distances[[row, column]] / (scales[row].sqrt() * scales[column].sqrt())
        }
    }))
}

/// Validates subsample log-likelihood inputs.
fn subsample_inputs(
    x: &ArrayView2<f64>,
    y: &ArrayView2<f64>,
    p: usize,
    num_params: usize,
) -> Result<(), ENNError> {
    if x.ndim() != 2 {
        return Err(ENNError::InvalidShape {
            expected: vec![x.nrows(), x.ncols()],
            got: x.shape().to_vec(),
        });
    }
    if y.ndim() != 2 {
        return Err(ENNError::InvalidShape {
            expected: vec![y.nrows(), y.ncols()],
            got: y.shape().to_vec(),
        });
    }
    if x.nrows() != y.nrows() {
        return Err(ENNError::InvalidParameter(format!(
            "x and y must have same number of rows: {} vs {}",
            x.nrows(),
            y.nrows()
        )));
    }
    if p == 0 {
        return Err(ENNError::InvalidParameter(
            "P (num_samples) must be > 0".to_string(),
        ));
    }
    if num_params == 0 {
        return Err(ENNError::InvalidParameter(
            "paramss must be non-empty".to_string(),
        ));
    }
    Ok(())
}

/// Compute single log-likelihood given scaled predictions.
fn compute_loglik(
    y_scaled: &ArrayView2<f64>,
    mu_i: &ArrayView2<f64>,
    se_i: &ArrayView2<f64>,
) -> f64 {
    // Check for non-finite values
    if !mu_i.iter().all(|v| v.is_finite()) || !se_i.iter().all(|v| v.is_finite()) {
        return f64::NEG_INFINITY;
    }
    if se_i.iter().any(|&v| v <= 0.0) {
        return f64::NEG_INFINITY;
    }

    let mut loglik = 0.0;
    for i in 0..y_scaled.nrows() {
        for j in 0..y_scaled.ncols() {
            let y_ij = y_scaled[[i, j]];
            let mu_ij = mu_i[[i, j]];
            let se_ij = se_i[[i, j]];
            let residual = (y_ij - mu_ij) / se_ij;
            loglik +=
                -0.5 * (2.0 * std::f64::consts::PI).ln() - se_ij.ln() - 0.5 * residual * residual;
        }
    }

    if loglik.is_finite() {
        loglik
    } else {
        f64::NEG_INFINITY
    }
}

/// Subsample model rows for leave-one-out likelihood without full-dataset views.
/// Numerically invalid predictions receive negative infinity, never a valid score.
pub fn subsample_model<R: Rng>(
    model: &ENN,
    paramss: &[ENNParams],
    p: usize,
    rng: &mut R,
    y_std: Option<&ArrayView1<f64>>,
) -> Result<Vec<f64>, ENNError> {
    let n = model.len();
    if p == 0 || paramss.is_empty() {
        return Err(ENNError::InvalidParameter(
            "nonempty parameters and p > 0 required".into(),
        ));
    }
    if n <= 1 {
        return Ok(vec![0.0; paramss.len()]);
    }
    let p_actual = p.min(n);
    let indices: Vec<usize> = if p_actual == n {
        (0..n).collect()
    } else {
        sample(rng, n, p_actual).into_iter().collect()
    };
    score_rows(model, &indices, paramss, y_std)
}

/// Score a subsample of the complete training data in model row order.
/// Subsets and reordered observations must use `row_loglik` with explicit IDs.
pub fn subsample_loglik<R: Rng>(
    model: &ENN,
    x: &ArrayView2<f64>,
    y: &ArrayView2<f64>,
    paramss: &[ENNParams],
    p: usize,
    rng: &mut R,
    y_std: Option<&ArrayView1<f64>>,
) -> Result<Vec<f64>, ENNError> {
    subsample_inputs(x, y, p, paramss.len())?;

    if x.nrows() != model.len() || x.ncols() != model.num_dim() || y.ncols() != model.num_metrics()
    {
        return Err(ENNError::InvalidParameter(
            "x and y must be complete ordered training rows; use row_loglik for subsets".into(),
        ));
    }
    let n = x.nrows();
    if n == 0 || model.num_obs() <= 1 {
        return Ok(vec![0.0; paramss.len()]);
    }

    // Check for non-finite y values
    if !y.iter().all(|v| v.is_finite()) {
        return Err(ENNError::InvalidParameter(
            "training targets must be finite".into(),
        ));
    }

    let p_actual = p.min(n);

    let indices: Vec<usize> = if p_actual == n {
        (0..n).collect()
    } else {
        sample(rng, n, p_actual).into_iter().collect()
    };

    let (stored_x, stored_y, _) = model.natural_rows(&indices)?;
    for (i, &id) in indices.iter().enumerate() {
        if x.row(id) != stored_x.row(i)
            || y.row(id)
                .iter()
                .zip(stored_y.row(i))
                .any(|(&a, &b)| (a - b).abs() > 16.0 * f64::EPSILON * (1.0 + b.abs()))
        {
            return Err(ENNError::InvalidParameter(
                "sampled rows do not match model row order; use row_loglik with explicit IDs"
                    .into(),
            ));
        }
    }
    let scale = y_std
        .map(|s| s.to_owned())
        .unwrap_or_else(|| y.std_axis(Axis(0), 0.0));
    score_rows(model, &indices, paramss, Some(&scale.view()))
}

/// Leave-one-out likelihood for explicit model row IDs, including duplicates.
pub fn row_loglik<R: Rng>(
    model: &ENN,
    rows: &[usize],
    paramss: &[ENNParams],
    p: usize,
    rng: &mut R,
    y_std: Option<&ArrayView1<f64>>,
) -> Result<Vec<f64>, ENNError> {
    if p == 0 || paramss.is_empty() || rows.iter().any(|&id| id >= model.len()) {
        return Err(ENNError::InvalidParameter(
            "valid row IDs, nonempty parameters and p > 0 required".into(),
        ));
    }
    let indices = if p >= rows.len() {
        rows.to_vec()
    } else {
        sample(rng, rows.len(), p).iter().map(|i| rows[i]).collect()
    };
    score_rows(model, &indices, paramss, y_std)
}

/// Leave-one-out likelihood from an exact pairwise squared-distance matrix.
///
/// This is equivalent to `row_loglik` for backends that retain observation
/// geometry but cannot afford to materialize every high-dimensional row.
fn validate_distance_loglik(
    distances: &ArrayView2<f64>,
    y: &ArrayView2<f64>,
    yvar: Option<&ArrayView2<f64>>,
    paramss: &[ENNParams],
    p: usize,
) -> Result<(), ENNError> {
    let n = y.nrows();
    if distances.nrows() != n || distances.ncols() != n {
        return Err(ENNError::InvalidShape {
            expected: vec![n, n],
            got: distances.shape().to_vec(),
        });
    }
    if p == 0 || paramss.is_empty() {
        return Err(ENNError::InvalidParameter(
            "nonempty parameters and p > 0 required".into(),
        ));
    }
    if y.ncols() == 0
        || y.iter().any(|value| !value.is_finite())
        || distances
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(ENNError::InvalidParameter(
            "distance fitting requires finite outcomes and finite nonnegative distances".into(),
        ));
    }
    if yvar.is_some_and(|values| {
        values.raw_dim() != y.raw_dim()
            || values
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
    }) {
        return Err(ENNError::InvalidParameter(
            "distance fitting variances must match outcomes and be finite and nonnegative".into(),
        ));
    }
    for params in paramss {
        ENNParams::new(
            params.k_neighbors,
            params.epistemic_scale,
            params.aleatoric_scale,
        )
        .map_err(|error| ENNError::InvalidParameter(error.to_string()))?;
    }
    Ok(())
}

fn sampled_rows<R: Rng>(n: usize, p: usize, rng: &mut R) -> Vec<usize> {
    if p >= n {
        (0..n).collect()
    } else {
        sample(rng, n, p).into_iter().collect()
    }
}

fn distance_neighbors(
    distances: &ArrayView2<f64>,
    selected: &[usize],
    max_k: usize,
) -> Vec<Vec<(f64, usize)>> {
    selected
        .iter()
        .map(|&held| {
            let mut row = (0..distances.nrows())
                .filter(|&candidate| candidate != held)
                .map(|candidate| (distances[[held, candidate]], candidate))
                .collect::<Vec<_>>();
            row.sort_by(|left, right| left.0.total_cmp(&right.0).then(left.1.cmp(&right.1)));
            row.truncate(max_k);
            row
        })
        .collect()
}

fn output_scale(y: &ArrayView2<f64>, y_std: Option<&ArrayView1<f64>>) -> Array1<f64> {
    y_std
        .map(|values| values.to_owned())
        .unwrap_or_else(|| y.std_axis(Axis(0), 0.0))
        .mapv(|value| {
            if value.is_finite() && value > 0.0 {
                value
            } else {
                1.0
            }
        })
}

fn distance_targets(y: &ArrayView2<f64>, selected: &[usize]) -> Array2<f64> {
    Array2::from_shape_fn((selected.len(), y.ncols()), |(row, metric)| {
        y[[selected[row], metric]]
    })
}

struct DistanceScore<'view, 'data> {
    y: &'view ArrayView2<'data, f64>,
    yvar: Option<&'view ArrayView2<'data, f64>>,
    selected: &'view [usize],
    neighbors: &'view [Vec<(f64, usize)>],
    scale: &'view ndarray::Array1<f64>,
    targets_scaled: &'view Array2<f64>,
    max_k: usize,
}

impl DistanceScore<'_, '_> {
    fn score(&self, params: &ENNParams) -> Result<f64, ENNError> {
        let used = (params.k_neighbors as usize).min(self.max_k);
        let dist2s = Array2::from_shape_fn((self.selected.len(), used), |(row, neighbor)| {
            self.neighbors[row][neighbor].0
        });
        let labels = Array2::from_shape_fn(
            (self.selected.len() * used, self.y.ncols()),
            |(flat, metric)| self.y[[self.neighbors[flat / used][flat % used].1, metric]],
        );
        let variances = self.yvar.map(|values| {
            Array2::from_shape_fn(
                (self.selected.len() * used, self.y.ncols()),
                |(flat, metric)| values[[self.neighbors[flat / used][flat % used].1, metric]],
            )
        });
        let stats = crate::posterior::compute_impl(
            &dist2s.view(),
            &labels.view(),
            variances.as_ref().map(|values| values.view()),
            params,
            true,
            &self.scale.view(),
        )?;
        Ok(compute_loglik(
            &self.targets_scaled.view(),
            &(&stats.mu / self.scale).view(),
            &(&stats.se / self.scale).view(),
        ))
    }
}

pub fn distance_loglik<R: Rng>(
    distances: &ArrayView2<f64>,
    y: &ArrayView2<f64>,
    yvar: Option<&ArrayView2<f64>>,
    paramss: &[ENNParams],
    p: usize,
    rng: &mut R,
    y_std: Option<&ArrayView1<f64>>,
) -> Result<Vec<f64>, ENNError> {
    let n = y.nrows();
    validate_distance_loglik(distances, y, yvar, paramss, p)?;
    if n <= 1 {
        return Ok(vec![0.0; paramss.len()]);
    }
    let selected = sampled_rows(n, p, rng);
    let max_k = paramss
        .iter()
        .map(|params| params.k_neighbors as usize)
        .max()
        .unwrap()
        .min(n - 1);
    let neighbors = distance_neighbors(distances, &selected, max_k);
    let scale = output_scale(y, y_std);
    if scale.len() != y.ncols() {
        return Err(ENNError::InvalidParameter(
            "y_std must match the number of outputs".into(),
        ));
    }
    let targets = distance_targets(y, &selected);
    let targets_scaled = &targets / &scale;
    let scorer = DistanceScore {
        y,
        yvar,
        selected: &selected,
        neighbors: &neighbors,
        scale: &scale,
        targets_scaled: &targets_scaled,
        max_k,
    };
    paramss.iter().map(|params| scorer.score(params)).collect()
}

fn loo_neighbors(
    model: &ENN,
    rows: &[usize],
    x: &ArrayView2<f64>,
    k: usize,
) -> Result<Vec<Vec<(f64, usize)>>, ENNError> {
    // One bounded lookup for all parameter candidates; never escalate tied rows
    // to a full dataset scan. Exclude identity even if ANN did not return self first.
    let request = i32::try_from(k + 1).map_err(|_| {
        ENNError::InvalidParameter("neighbor request exceeds index capacity".into())
    })?;
    let (distances, ids) = index_search(model, x, request, false, false)?;
    let mut neighbors = Vec::with_capacity(rows.len());
    for (i, &held) in rows.iter().enumerate() {
        let mut row: Vec<(f64, usize)> = ids
            .row(i)
            .iter()
            .zip(distances.row(i))
            .filter_map(|(&id, &d)| {
                (id >= 0 && (id as usize) < model.len() && id as usize != held)
                    .then_some((d, id as usize))
            })
            .collect();
        row.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        row.dedup_by_key(|v| v.1);
        if row.len() < k {
            return Err(ENNError::InvalidParameter(
                "not enough distinct neighbors after held-out exclusion".into(),
            ));
        }
        row.truncate(k);
        neighbors.push(row);
    }
    Ok(neighbors)
}

fn score_rows(
    model: &ENN,
    rows: &[usize],
    paramss: &[ENNParams],
    y_std: Option<&ArrayView1<f64>>,
) -> Result<Vec<f64>, ENNError> {
    for params in paramss {
        ENNParams::new(
            params.k_neighbors,
            params.epistemic_scale,
            params.aleatoric_scale,
        )
        .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
    }
    if rows.is_empty() || model.len() <= 1 {
        return Ok(vec![0.0; paramss.len()]);
    }
    let (x, y, _) = model.natural_rows(rows)?;
    let scale = y_std
        .map(|s| s.to_owned())
        .unwrap_or_else(|| y.std_axis(Axis(0), 0.0));
    if scale.len() != y.ncols() {
        return Err(ENNError::InvalidParameter(
            "y_std must match the number of outputs".into(),
        ));
    }
    let scale = scale.mapv(|v| if v.is_finite() && v > 0.0 { v } else { 1.0 });
    let y_scaled = &y / &scale;
    let k = paramss
        .iter()
        .map(|p| p.k_neighbors as usize)
        .max()
        .unwrap()
        .min(model.len() - 1);
    let neighbors = loo_neighbors(model, rows, &x.view(), k)?;
    let mut scores = Vec::with_capacity(paramss.len());
    let mut cache = std::collections::HashMap::new();
    for params in paramss {
        let used = (params.k_neighbors as usize).min(k);
        if let std::collections::hash_map::Entry::Vacant(entry) = cache.entry(used) {
            let dist2s = Array2::from_shape_fn((rows.len(), used), |(i, j)| neighbors[i][j].0);
            let idx: Vec<Vec<usize>> = neighbors
                .iter()
                .map(|r| r[..used].iter().map(|v| v.1).collect())
                .collect();
            let mut labels = Array2::zeros((rows.len() * used, y.ncols()));
            for (i, row) in idx.iter().enumerate() {
                for (j, &id) in row.iter().enumerate() {
                    labels
                        .row_mut(i * used + j)
                        .assign(&model.rows().row_y(id)?);
                }
            }
            entry.insert(crate::draw::NeighborData::new(dist2s, idx, labels, used));
        }
        let data = &cache[&used];
        let mut post = compute_posterior(
            model,
            WeightedPosteriorData {
                dist2s: &data.dist2s.view(),
                idx: &data.idx,
                y_neighbors: &data.y_neighbors.view(),
                params,
                observation_noise: true,
                yvar_neighbors_override: None,
            },
            None,
        )?;
        if model.bounded_outputs() {
            crate::y_bounds::naturalize(
                &mut post.mu,
                &mut post.se,
                &mut post.se_epi,
                &mut post.se_ale,
                model.y_bounds(),
            );
        }
        scores.push(compute_loglik(
            &y_scaled.view(),
            &(&post.mu / &scale).view(),
            &(&post.se / &scale).view(),
        ));
    }
    Ok(scores)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fitter::ENNFitter;
    use crate::index::IndexDriver;
    use crate::test_helpers::test_model as create_test_model;
    use ndarray::array;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn self_tuning_is_symmetric_finite_and_uses_local_radii() {
        let distances = array![[0.0, 1.0, 100.0], [1.0, 0.0, 81.0], [100.0, 81.0, 0.0]];
        let tuned = self_tuned_distances(&distances.view(), 1).unwrap();
        assert_eq!(tuned[[0, 1]], 1.0);
        assert!((tuned[[0, 2]] - 100.0 / 9.0).abs() < 1.0e-12);
        assert_eq!(tuned[[0, 2]], tuned[[2, 0]]);
        assert!(tuned.iter().all(|value| value.is_finite()));

        let duplicates = array![[0.0, 0.0], [0.0, 0.0]];
        let tuned = self_tuned_distances(&duplicates.view(), 1).unwrap();
        assert!(tuned.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn test_001() {
        let model = create_test_model();
        let (x, y, _) = model
            .natural_rows(&(0..model.len()).collect::<Vec<_>>())
            .unwrap();
        let params = ENNParams::new(2, 1.0, 0.1).unwrap();
        let paramss = vec![params];
        let mut rng = StdRng::seed_from_u64(42);

        let logliks =
            subsample_loglik(&model, &x.view(), &y.view(), &paramss, 2, &mut rng, None).unwrap();

        assert_eq!(logliks.len(), 1);
        assert!(logliks[0].is_finite());
    }

    #[test]
    fn distance_likelihood_matches_rows() {
        let x = array![[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [2.0, -0.5]];
        let y = array![[0.2], [-0.4], [1.1], [0.7]];
        let yvar = array![[0.01], [0.04], [0.02], [0.03]];
        let model = ENN::new(
            x.clone(),
            y.clone(),
            Some(yvar.clone()),
            false,
            IndexDriver::Exact,
        )
        .unwrap();
        let distances = Array2::from_shape_fn((x.nrows(), x.nrows()), |(left, right)| {
            x.row(left)
                .iter()
                .zip(x.row(right))
                .map(|(a, b)| (a - b) * (a - b))
                .sum()
        });
        let paramss = [
            ENNParams::new(2, 0.3, 0.05).unwrap(),
            ENNParams::new(3, 1.7, 0.2).unwrap(),
        ];
        let rows = (0..x.nrows()).collect::<Vec<_>>();
        let scale = y.std_axis(Axis(0), 0.0);
        let mut row_rng = StdRng::seed_from_u64(9);
        let mut distance_rng = StdRng::seed_from_u64(9);
        let expected = row_loglik(
            &model,
            &rows,
            &paramss,
            rows.len(),
            &mut row_rng,
            Some(&scale.view()),
        )
        .unwrap();
        let actual = distance_loglik(
            &distances.view(),
            &y.view(),
            Some(&yvar.view()),
            &paramss,
            rows.len(),
            &mut distance_rng,
            Some(&scale.view()),
        )
        .unwrap();
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-12, "{actual} != {expected}");
        }
    }

    #[test]
    fn test_002() {
        let model = create_test_model();
        let params = ENNParams::new(2, 1.0, 0.1).unwrap();
        let paramss = vec![params];
        let mut rng = StdRng::seed_from_u64(99);
        let via_model = subsample_model(&model, &paramss, 2, &mut rng, None).unwrap();
        let mut rng2 = StdRng::seed_from_u64(99);
        let all: Vec<usize> = (0..model.len()).collect();
        let (full_x, full_y, _) = model.rows().train_rows(&all).unwrap();
        let via_views = subsample_loglik(
            &model,
            &full_x.view(),
            &full_y.view(),
            &paramss,
            2,
            &mut rng2,
            None,
        )
        .unwrap();
        assert_eq!(via_model.len(), via_views.len());
        assert!(via_model[0].is_finite());
    }

    #[test]
    fn test_003() {
        let train_x = array![[0.0, 0.0]];
        let train_y = array![[0.0]];
        let model = ENN::new(train_x, train_y, None, false, IndexDriver::Exact).unwrap();

        let x = array![[0.5, 0.5]];
        let y = array![[1.0]];
        let params = ENNParams::new(2, 1.0, 0.1).unwrap();
        let paramss = vec![params];
        let mut rng = StdRng::seed_from_u64(42);

        let logliks =
            subsample_loglik(&model, &x.view(), &y.view(), &paramss, 2, &mut rng, None).unwrap();

        assert_eq!(logliks, vec![0.0]);
    }

    #[test]
    fn test_ennfitteraskbasic() {
        let model = create_test_model();
        let mut rng = StdRng::seed_from_u64(42);
        let mut fitter = ENNFitter::new(2, true);
        let all: Vec<usize> = (0..model.len()).collect();
        let (_, ty, _) = model.rows().train_rows(&all).unwrap();
        fitter.y_stats(&ty.view());

        let result = fitter.ask(&model, 5, 3, None, &mut rng).unwrap();

        assert_eq!(result.k_neighbors, 2);
        assert!(result.epistemic_scale > 0.0);
        assert!(result.aleatoric_scale >= 0.0);
    }

    #[test]
    fn test_005() {
        let model = create_test_model();
        let mut rng = StdRng::seed_from_u64(42);
        let mut fitter = ENNFitter::new(2, true);
        let all: Vec<usize> = (0..model.len()).collect();
        let (_, ty, _) = model.rows().train_rows(&all).unwrap();
        fitter.y_stats(&ty.view());

        let warm_start = ENNParams::new(2, 1.5, 0.2).unwrap();

        let result = fitter
            .ask(&model, 5, 3, Some(&warm_start), &mut rng)
            .unwrap();

        assert_eq!(result.k_neighbors, 2);
        assert!(result.epistemic_scale > 0.0);
    }

    #[test]
    fn test_006() {
        let model = create_test_model();
        let mut rng = StdRng::seed_from_u64(42);
        let mut fitter = ENNFitter::new(2, false);
        let all: Vec<usize> = (0..model.len()).collect();
        let (_, ty, _) = model.rows().train_rows(&all).unwrap();
        fitter.y_stats(&ty.view());

        let result = fitter.ask(&model, 5, 3, None, &mut rng).unwrap();

        assert_eq!(result.k_neighbors, 2);
        assert!(result.epistemic_scale > 0.0);
        assert_eq!(result.aleatoric_scale, 0.0);
    }

    #[test]
    fn test_007() {
        let train_x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]];
        let train_y = array![[0.0, 1.0], [1.0, 2.0], [1.0, 0.0], [2.0, 1.0], [1.0, 1.5]];
        let model = ENN::new(train_x, train_y, None, false, IndexDriver::Exact).unwrap();

        let mut rng = StdRng::seed_from_u64(42);
        let mut fitter = ENNFitter::new(2, true);
        let all: Vec<usize> = (0..model.len()).collect();
        let (_, ty, _) = model.rows().train_rows(&all).unwrap();
        fitter.y_stats(&ty.view());

        let result = fitter.ask(&model, 5, 3, None, &mut rng).unwrap();

        assert_eq!(result.k_neighbors, 2);
        assert!(result.epistemic_scale > 0.0);
    }

    #[test]
    fn test_008() {
        let model = create_test_model();
        let x = array![[0.5, 0.5]];
        let y = array![[1.0]];
        let params = ENNParams::new(2, 1.0, 0.1).unwrap();
        let paramss = vec![params];
        let mut rng = StdRng::seed_from_u64(42);

        let result = subsample_loglik(&model, &x.view(), &y.view(), &paramss, 0, &mut rng, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_009() {
        let model = create_test_model();
        let x = array![[0.5, 0.5], [0.2, 0.8]]; // 2 rows
        let y = array![[1.0]]; // 1 row
        let params = ENNParams::new(2, 1.0, 0.1).unwrap();
        let paramss = vec![params];
        let mut rng = StdRng::seed_from_u64(42);

        let result = subsample_loglik(&model, &x.view(), &y.view(), &paramss, 2, &mut rng, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_010() {
        let x = array![[0.5, 0.5]];
        let y = array![[1.0]];
        let err = subsample_inputs(&x.view(), &y.view(), 1, 0).unwrap_err();
        assert!(err.to_string().contains("paramss must be non-empty"));
    }

    #[test]
    fn test_011() {
        let y_scaled = array![[1.0]];
        let mu_i = array![[0.0]];
        let se_i = array![[1.0]];
        let ll = compute_loglik(&y_scaled.view(), &mu_i.view(), &se_i.view());
        let expected = -0.5 * ((2.0 * std::f64::consts::PI).ln() + 1.0);
        assert!((ll - expected).abs() < 1e-12);
    }

    #[test]
    fn test_012() {
        let model = create_test_model();
        let (x, y, _) = model
            .natural_rows(&(0..model.len()).collect::<Vec<_>>())
            .unwrap();
        let params = ENNParams::new(2, 1.0, 0.1).unwrap();
        let y_std = array![2.0];
        let mut rng = StdRng::seed_from_u64(11);
        let paramss = vec![params];
        let logliks = subsample_loglik(
            &model,
            &x.view(),
            &y.view(),
            &paramss,
            2,
            &mut rng,
            Some(&y_std.view()),
        )
        .unwrap();
        assert_eq!(logliks.len(), 1);
        assert!(logliks[0].is_finite());
        let again = subsample_loglik(
            &model,
            &x.view(),
            &y.view(),
            &paramss,
            2,
            &mut StdRng::seed_from_u64(11),
            Some(&y_std.view()),
        )
        .unwrap();
        assert!((logliks[0] - again[0]).abs() < 1e-12);
    }

    #[test]
    fn test_013() {
        let y = array![[1.0]];
        let mu_bad = array![[f64::NAN]];
        let se_ok = array![[1.0]];
        assert_eq!(
            compute_loglik(&y.view(), &mu_bad.view(), &se_ok.view()),
            f64::NEG_INFINITY
        );

        let mu_ok = array![[1.0]];
        let se_bad = array![[0.0]];
        assert_eq!(
            compute_loglik(&y.view(), &mu_ok.view(), &se_bad.view()),
            f64::NEG_INFINITY
        );
    }

    #[test]
    fn likelihood_limits() {
        let value = array![[0.0]];
        for se in [1e-200, 1e200] {
            let actual = compute_loglik(&value.view(), &value.view(), &array![[se]].view());
            let expected = -0.5 * (2.0 * std::f64::consts::PI).ln() - se.ln();
            assert!(actual.is_finite());
            assert_eq!(actual, expected);
        }
        for se in [f64::NAN, f64::INFINITY, -1.0, 0.0] {
            assert_eq!(
                compute_loglik(&value.view(), &value.view(), &array![[se]].view()),
                f64::NEG_INFINITY
            );
        }
    }
}
