//! Fused posterior fast path when yvar is absent and observation_noise is off.

use ndarray::{Array2, ArrayView2, Axis};

use crate::error::{ENNError, EPS_VAR};
use crate::model::ENN;
use crate::params::{ENNParams, PosteriorFlags};

use super::{empty_internals, index_search};

pub(crate) type PosteriorLightOut = (
    Array2<f64>,
    Array2<f64>,
    Array2<f64>,
    Array2<f64>,
    Array2<i64>,
);

pub(crate) fn index_array(idx: &[Vec<usize>]) -> Array2<i64> {
    let n_query = idx.len();
    let k = idx.first().map(|r| r.len()).unwrap_or(0);
    let mut out = Array2::from_elem((n_query, k), -1i64);
    for (i, row) in idx.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            out[[i, j]] = v as i64;
        }
    }
    out
}

pub(super) fn search_k(
    model: &ENN,
    params: &ENNParams,
    exclude_nearest: bool,
) -> Result<usize, ENNError> {
    if exclude_nearest && model.num_obs() <= 1 {
        return Err(ENNError::InvalidParameter(format!(
            "exclude_nearest=True requires at least 2 observations, got {}",
            model.num_obs()
        )));
    }
    Ok(if exclude_nearest {
        (params.k_neighbors as usize + 1).min(model.num_obs())
    } else {
        (params.k_neighbors as usize).min(model.num_obs())
    })
}

fn fuse_se(
    model: &ENN,
    dist2s: &ArrayView2<f64>,
    idx: &ArrayView2<i64>,
    params: &ENNParams,
) -> PosteriorLightOut {
    let n_query = dist2s.nrows();
    let k = dist2s.ncols();
    let num_metrics = model.num_metrics();
    let y_scale = model.output_scale();
    let epistemic_scale = params.epistemic_scale;
    let aleatoric_scale = params.aleatoric_scale;

    let mut mu = Array2::zeros((n_query, num_metrics));
    let mut se = Array2::zeros((n_query, num_metrics));
    let mut se_epi = Array2::zeros((n_query, num_metrics));
    let mut se_ale = Array2::zeros((n_query, num_metrics));
    let mut idx_out = Array2::from_elem((n_query, k), -1i64);
    let mut w = vec![0.0f64; k];

    for i in 0..n_query {
        let dist_row = dist2s.row(i);
        let idx_row = idx.row(i);

        let mut norm = 0.0;
        for j in 0..k {
            let var_total = EPS_VAR + epistemic_scale * dist_row[j] + aleatoric_scale;
            w[j] = 1.0 / var_total;
            norm += w[j];
        }
        let inv_norm = 1.0 / norm;
        let se_base = inv_norm.max(EPS_VAR).sqrt();

        for j in 0..k {
            idx_out[[i, j]] = idx_row[j];
        }

        for m in 0..num_metrics {
            let mut mu_val = 0.0;
            let y_scale_m = y_scale[m];
            if let Some(y_view) = model.y_opt() {
                for j in 0..k {
                    let w_norm = w[j] * inv_norm;
                    mu_val += w_norm * y_view[[idx_row[j] as usize, m]];
                }
            } else {
                for j in 0..k {
                    let w_norm = w[j] * inv_norm;
                    let y_row = model.rows().row_y(idx_row[j] as usize).expect("row_y");
                    mu_val += w_norm * y_row[m];
                }
            }
            mu[[i, m]] = mu_val;
            let se_val = se_base * y_scale_m;
            se[[i, m]] = se_val;
            se_epi[[i, m]] = se_val;
            se_ale[[i, m]] = 0.0;
        }
    }

    (mu, se, se_epi, se_ale, idx_out)
}

/// Fused index_search + mu/se for the no-yvar, no-observation-noise posterior path.
pub(crate) fn compute_light(
    model: &ENN,
    x: &ArrayView2<f64>,
    params: &ENNParams,
    flags: &PosteriorFlags,
) -> Result<PosteriorLightOut, ENNError> {
    if x.ncols() != model.num_dim() {
        return Err(ENNError::InvalidShape {
            expected: vec![x.nrows(), model.num_dim()],
            got: x.shape().to_vec(),
        });
    }

    let batch_size = x.nrows();
    if model.num_obs() == 0 {
        let internals = empty_internals(model, batch_size);
        return Ok((
            internals.mu,
            internals.se,
            internals.se_epi,
            internals.se_ale,
            index_array(&internals.idx),
        ));
    }

    let search_k = search_k(model, params, flags.exclude_nearest)?;
    if search_k == 0 {
        let internals = empty_internals(model, batch_size);
        return Ok((
            internals.mu,
            internals.se,
            internals.se_epi,
            internals.se_ale,
            index_array(&internals.idx),
        ));
    }

    let available_k = if flags.exclude_nearest {
        search_k.saturating_sub(1)
    } else {
        search_k
    };
    let k = (params.k_neighbors as usize).min(available_k);
    if k == 0 {
        let internals = empty_internals(model, batch_size);
        return Ok((
            internals.mu,
            internals.se,
            internals.se_epi,
            internals.se_ale,
            index_array(&internals.idx),
        ));
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
    if model.backend_driver() == crate::index::IndexDriver::Cuda && !x.is_empty() {
        model.ensure_sync()?;
        if let (Some(index), Some(train_y)) = (model.backend.memory_index(), model.y_opt()) {
            return index
                .cuda_posterior(
                    x,
                    &train_y,
                    &model.output_scale().view(),
                    search_k,
                    k,
                    usize::from(flags.exclude_nearest),
                    params.epistemic_scale,
                    params.aleatoric_scale,
                )
                .map_err(|error| ENNError::InvalidParameter(error.to_string()));
        }
    }

    let (dist2s_full, idx_full) = {
        index_search(
            model,
            x,
            search_k as i32,
            flags.exclude_nearest,
            flags.tie_neighbors,
        )?
    };

    let dist2s = dist2s_full.slice_axis(Axis(1), ndarray::Slice::from(..k));
    let idx = idx_full.slice_axis(Axis(1), ndarray::Slice::from(..k));
    Ok(fuse_se(model, &dist2s, &idx, params))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::IndexDriver;
    use crate::model::ENN;
    use crate::posterior::compute_internals;
    use crate::test_helpers::test_model as create_test_model;
    use ndarray::array;

    #[test]
    fn test_001() {
        let n = 32;
        let d = 4;
        let m = 2;
        let train_x = Array2::from_shape_fn((n, d), |(i, j)| (i as f64 * 0.1) + (j as f64 * 0.01));
        let train_y = Array2::from_shape_fn((n, m), |(i, j)| (i as f64) + (j as f64));
        let model = ENN::new(train_x.clone(), train_y, None, false, IndexDriver::Exact).unwrap();
        let params = ENNParams::new(5, 1.0, 0.1).unwrap();
        let flags = PosteriorFlags::new();

        let (mu_light, se_light, se_epi_light, se_ale_light, idx_light) =
            compute_light(&model, &train_x.view(), &params, &flags).unwrap();
        let full = compute_internals(&model, &train_x.view(), &params, &flags).unwrap();

        assert_eq!(mu_light.shape(), full.mu.shape());
        assert_eq!(se_light.shape(), full.se.shape());
        assert!((mu_light - &full.mu)
            .mapv(f64::abs)
            .iter()
            .all(|&d| d < 1e-12));
        assert!((se_light - &full.se)
            .mapv(f64::abs)
            .iter()
            .all(|&d| d < 1e-12));
        assert!((se_epi_light - &full.se_epi)
            .mapv(f64::abs)
            .iter()
            .all(|&d| d < 1e-12));
        assert!((se_ale_light - &full.se_ale)
            .mapv(f64::abs)
            .iter()
            .all(|&d| d < 1e-12));
        assert_eq!(idx_light, index_array(&full.idx));
    }

    #[test]
    fn search_rules() {
        let model = ENN::new(
            array![[0.0, 0.0]],
            array![[1.0]],
            None,
            false,
            IndexDriver::Exact,
        )
        .unwrap();
        let params = ENNParams::new(1, 1.0, 0.1).unwrap();
        assert!(search_k(&model, &params, true).is_err());
        assert_eq!(search_k(&model, &params, false).unwrap(), 1);
    }

    #[test]
    fn test_003() {
        let nested = vec![vec![0usize, 2], vec![1, 3]];
        let arr = index_array(&nested);
        assert_eq!(arr[[0, 0]], 0);
        assert_eq!(arr[[1, 1]], 3);
        let empty = index_array(&[]);
        assert_eq!(empty.shape(), &[0, 0]);
    }

    #[test]
    fn test_004() {
        let model = create_test_model();
        let params = ENNParams::new(2, 1.0, 0.1).unwrap();
        let flags = PosteriorFlags::new();
        let empty_query: Array2<f64> = Array2::zeros((0, 2));
        let (mu, se, _se_epi, _se_ale, idx) =
            compute_light(&model, &empty_query.view(), &params, &flags).unwrap();
        assert_eq!(mu.nrows(), 0);
        assert_eq!(se.nrows(), 0);
        assert!(idx.is_empty());
    }

    #[test]
    fn test_005() {
        let model = create_test_model();
        let params = ENNParams::new(2, 1.0, 0.1).unwrap();
        let flags = PosteriorFlags::new();
        let bad_query = array![[0.5]];
        assert!(compute_light(&model, &bad_query.view(), &params, &flags).is_err());
    }

    #[test]
    fn test_006() {
        let model = ENN::new(
            Array2::zeros((0, 2)),
            Array2::zeros((0, 1)),
            None,
            false,
            IndexDriver::Exact,
        )
        .unwrap();
        let params = ENNParams::new(2, 1.0, 0.1).unwrap();
        let flags = PosteriorFlags::new();
        let query = array![[0.5, 0.5]];
        let (mu, se, _se_epi, _se_ale, idx) =
            compute_light(&model, &query.view(), &params, &flags).unwrap();
        assert_eq!(mu.nrows(), 1);
        assert_eq!(se.nrows(), 1);
        assert_eq!(idx.nrows(), 1);
    }

    #[test]
    fn test_007() {
        let train_x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let train_y = array![[1.0], [2.0], [3.0], [4.0]];
        let model = ENN::new(train_x, train_y, None, false, IndexDriver::Exact).unwrap();
        let params = ENNParams::new(2, 1.0, 0.1).unwrap();
        let flags = PosteriorFlags::new();
        let all: Vec<usize> = (0..model.len()).collect();
        let (tx, _, _) = model.rows().train_rows(&all).unwrap();
        let (mu_light, se_light, _se_epi_light, _se_ale_light, idx_light) =
            compute_light(&model, &tx.view(), &params, &flags).unwrap();
        let dist2s = array![[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [1.0, 2.0]];
        let idx = array![[0, 1], [1, 0], [2, 3], [3, 2]];
        let (mu, se, _se_epi, _se_ale, idx_out) =
            fuse_se(&model, &dist2s.view(), &idx.view(), &params);
        assert_eq!(mu.nrows(), 4);
        assert_eq!(se.nrows(), 4);
        assert_eq!(idx_out.shape(), &[4, 2]);
        assert!(mu.iter().all(|v| v.is_finite()));
        assert!(se.iter().all(|&v| v > 0.0));
        assert_eq!(mu_light.nrows(), 4);
        assert_eq!(se_light.nrows(), 4);
        assert_eq!(idx_light.nrows(), 4);
    }

    #[test]
    fn test_008() {
        let train_x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let train_y = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let model = ENN::new(train_x, train_y, None, false, IndexDriver::Exact).unwrap();
        let params = ENNParams::new(2, 1.0, 0.1).unwrap();
        let flags = PosteriorFlags::new();
        let all: Vec<usize> = (0..model.len()).collect();
        let (tx, _, _) = model.rows().train_rows(&all).unwrap();
        let (mu, se, _se_epi, _se_ale, idx) =
            compute_light(&model, &tx.view(), &params, &flags).unwrap();
        assert_eq!(mu.ncols(), 2);
        assert_eq!(se.ncols(), 2);
        assert_eq!(idx.ncols(), 2);
        assert!(mu.iter().all(|v| v.is_finite()));
        assert!(se.iter().all(|&v| v > 0.0));
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn metal_cpu2() {
        fn metal_unavailable4(error: &str) -> bool {
            error.contains("no default Metal device found")
        }

        let train_x =
            Array2::from_shape_fn((257, 7), |(i, j)| ((i * 37 + j * 13) % 503) as f64 / 503.0);
        let train_y =
            Array2::from_shape_fn((257, 3), |(i, j)| ((i * 11 + j * 29) % 101) as f64 / 17.0);
        let query = Array2::from_shape_fn((65, 7), |(i, j)| {
            ((i * 19 + j * 7 + 3) % 509) as f64 / 509.0
        });
        let cpu = ENN::new(
            train_x.clone(),
            train_y.clone(),
            None,
            false,
            IndexDriver::Exact,
        )
        .unwrap();
        let metal = match ENN::new(train_x, train_y, None, false, IndexDriver::Metal) {
            Ok(model) => model,
            Err(error) if metal_unavailable4(&error.to_string()) => return,
            Err(error) => panic!("{error}"),
        };
        let params = ENNParams::new(17, 0.7, 0.13).unwrap();
        let flags = PosteriorFlags::new().tie_neighbors(false);
        let cpu = compute_light(&cpu, &query.view(), &params, &flags).unwrap();
        let metal = compute_light(&metal, &query.view(), &params, &flags).unwrap();
        assert_eq!(cpu.4, metal.4);
        for (reference, actual) in cpu.0.iter().zip(metal.0.iter()) {
            assert!((reference - actual).abs() <= 2.0e-5 * (1.0 + reference.abs()));
        }
        for (reference, actual) in cpu.1.iter().zip(metal.1.iter()) {
            assert!((reference - actual).abs() <= 2.0e-5 * (1.0 + reference.abs()));
        }
    }
}
