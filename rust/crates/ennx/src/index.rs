use ndarray::{Array1, Array2, ArrayView2, Axis};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::knn::KnnBackend;
#[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
use crate::knn::{CudaParam, KnnBatch, KnnPosterior};

#[derive(Error, Debug, Clone, PartialEq)]
pub enum IndexError {
    #[error("Invalid shape: expected {expected} dims, got {got}")]
    InvalidShape { expected: usize, got: usize },
    #[error("Invalid search parameter: {0}")]
    InvalidParameter(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IndexDriver {
    #[default]
    Exact,
    /// Exact CPU/accelerator race calibrated from real query shapes.
    Auto,
    /// Device-native Apple GPU binary-archive execution.
    Agx,
    /// USearch HNSW shortlist with deterministic exact-distance reranking.
    #[serde(rename = "usearch")]
    USearch,
    /// B+ANN disk index (`EnnStorage::Disk` + `work_dir`).
    #[serde(rename = "bp_ann_disk")]
    BpAnnDisk,
    /// Apple Metal backend for native quantized-weight paths.
    Metal,
    /// OpenCL backend for native quantized-weight paths.
    #[serde(rename = "opencl")]
    OpenCl,
    /// NVIDIA CUDA exact index implemented with CUDA-Oxide.
    #[serde(rename = "cuda")]
    Cuda,
}

pub fn is_disk_index_driver(driver: IndexDriver) -> bool {
    matches!(driver, IndexDriver::BpAnnDisk)
}

use std::sync::Mutex;

pub struct ENNIndex {
    inner: KnnBackend,
    num_dim: usize,
    x_scale: Mutex<Array1<f64>>,
    scale_x: bool,
    driver: IndexDriver,
}

impl ENNIndex {
    pub fn new(
        train_x_scaled: Array2<f64>,
        num_dim: usize,
        x_scale: Array1<f64>,
        scale_x: bool,
        driver: IndexDriver,
    ) -> Result<Self, IndexError> {
        let inner = KnnBackend::new(num_dim, driver, &train_x_scaled.view())?;
        Ok(Self {
            inner,
            num_dim,
            x_scale: Mutex::new(x_scale),
            scale_x,
            driver,
        })
    }

    pub fn driver(&self) -> IndexDriver {
        self.driver
    }

    pub fn rebuild_from_scaled(
        &self,
        train_x_scaled: Array2<f64>,
        x_scale: Array1<f64>,
    ) -> Result<(), IndexError> {
        self.inner.rebuild(&train_x_scaled.view())?;
        *self.x_scale.lock().expect("x_scale mutex poisoned") = x_scale;
        Ok(())
    }

    pub fn add(&self, x: &ArrayView2<f64>) -> Result<(), IndexError> {
        if x.ncols() != self.num_dim {
            return Err(IndexError::InvalidShape {
                expected: self.num_dim,
                got: x.ncols(),
            });
        }
        let x_scale = self.x_scale.lock().expect("x_scale mutex poisoned").clone();
        let x_scaled: Array2<f64> = if self.scale_x {
            x / &x_scale.view().insert_axis(Axis(0))
        } else {
            x.to_owned()
        };
        let start_key = self.len() as u64;
        self.inner.add(&x_scaled.view(), start_key)
    }

    pub fn search(
        &self,
        x: &ArrayView2<f64>,
        search_k: i32,
        exclude_nearest: bool,
    ) -> Result<(Array2<f64>, Array2<i64>), IndexError> {
        if search_k <= 0 {
            return Err(IndexError::InvalidParameter(format!(
                "search_k must be > 0, got {search_k}"
            )));
        }
        if x.ncols() != self.num_dim {
            return Err(IndexError::InvalidShape {
                expected: self.num_dim,
                got: x.ncols(),
            });
        }

        let n_train = self.inner.len();
        let n_query = x.nrows();
        let search_k = search_k as usize;

        let x_scale = self.x_scale.lock().expect("x_scale mutex poisoned").clone();
        let x_scaled: Array2<f64> = if self.scale_x {
            x / &x_scale.view().insert_axis(Axis(0))
        } else {
            x.to_owned()
        };

        let (mut dist2s, mut indices) = if n_train == 0 {
            (
                Array2::from_elem((n_query, search_k), f64::INFINITY),
                Array2::zeros((n_query, search_k)),
            )
        } else {
            let k_eff = search_k.min(n_train);
            self.inner.search(&x_scaled.view(), k_eff, search_k)?
        };

        if exclude_nearest {
            let nc = dist2s.ncols();
            if nc <= 1 {
                dist2s = Array2::zeros((n_query, nc.saturating_sub(1)));
                indices = Array2::zeros((n_query, nc.saturating_sub(1)));
            } else {
                dist2s = dist2s
                    .slice_axis(Axis(1), ndarray::Slice::from(1..))
                    .to_owned();
                indices = indices
                    .slice_axis(Axis(1), ndarray::Slice::from(1..))
                    .to_owned();
            }
        }

        Ok((dist2s, indices))
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cuda_posterior(
        &self,
        x: &ArrayView2<f64>,
        outcomes: &ArrayView2<f64>,
        scales: &ndarray::ArrayView1<f64>,
        input_k: usize,
        used_k: usize,
        skip: usize,
        epistemic_scale: f64,
        aleatoric_scale: f64,
    ) -> Result<KnnPosterior, IndexError> {
        if x.ncols() != self.num_dim {
            return Err(IndexError::InvalidShape {
                expected: self.num_dim,
                got: x.ncols(),
            });
        }
        let x_scale = self.x_scale.lock().expect("x_scale mutex poisoned").clone();
        let x_scaled: Array2<f64> = if self.scale_x {
            x / &x_scale.view().insert_axis(Axis(0))
        } else {
            x.to_owned()
        };
        self.inner.cuda_posterior(
            &x_scaled.view(),
            outcomes,
            scales,
            input_k,
            used_k,
            skip,
            epistemic_scale,
            aleatoric_scale,
        )
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cuda_weighted(
        &self,
        x: &ArrayView2<f64>,
        outcomes: &ArrayView2<f64>,
        variances: Option<&ArrayView2<f64>>,
        scales: &ndarray::ArrayView1<f64>,
        input_k: usize,
        used_k: usize,
        skip: usize,
        epistemic_scale: f64,
        aleatoric_scale: f64,
        observation_noise: bool,
    ) -> Result<crate::draw::DrawInternals, IndexError> {
        if x.ncols() != self.num_dim {
            return Err(IndexError::InvalidShape {
                expected: self.num_dim,
                got: x.ncols(),
            });
        }
        let x_scale = self.x_scale.lock().expect("x_scale mutex poisoned").clone();
        let x_scaled: Array2<f64> = if self.scale_x {
            x / &x_scale.view().insert_axis(Axis(0))
        } else {
            x.to_owned()
        };
        self.inner.cuda_weighted(
            &x_scaled.view(),
            outcomes,
            variances,
            scales,
            input_k,
            used_k,
            skip,
            epistemic_scale,
            aleatoric_scale,
            observation_noise,
        )
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cuda_batch(
        &self,
        x: &ArrayView2<f64>,
        outcomes: &ArrayView2<f64>,
        variances: Option<&ArrayView2<f64>>,
        scales: &ndarray::ArrayView1<f64>,
        input_k: usize,
        skip: usize,
        values: &[CudaParam],
        observation_noise: bool,
    ) -> Result<KnnBatch, IndexError> {
        if x.ncols() != self.num_dim {
            return Err(IndexError::InvalidShape {
                expected: self.num_dim,
                got: x.ncols(),
            });
        }
        let x_scale = self.x_scale.lock().expect("x_scale mutex poisoned").clone();
        let x_scaled: Array2<f64> = if self.scale_x {
            x / &x_scale.view().insert_axis(Axis(0))
        } else {
            x.to_owned()
        };
        self.inner.cuda_batch(
            &x_scaled.view(),
            outcomes,
            variances,
            scales,
            input_k,
            skip,
            values,
            observation_noise,
        )
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cuda_draws(
        &self,
        x: &ArrayView2<f64>,
        outcomes: &ArrayView2<f64>,
        variances: Option<&ArrayView2<f64>>,
        scales: &ndarray::ArrayView1<f64>,
        input_k: usize,
        used_k: usize,
        skip: usize,
        epistemic_scale: f64,
        aleatoric_scale: f64,
        observation_noise: bool,
        seeds: &[i64],
    ) -> Result<(ndarray::Array3<f64>, Vec<Vec<usize>>), IndexError> {
        if x.ncols() != self.num_dim {
            return Err(IndexError::InvalidShape {
                expected: self.num_dim,
                got: x.ncols(),
            });
        }
        let x_scale = self.x_scale.lock().expect("x_scale mutex poisoned").clone();
        let x_scaled: Array2<f64> = if self.scale_x {
            x / &x_scale.view().insert_axis(Axis(0))
        } else {
            x.to_owned()
        };
        self.inner.cuda_draws(
            &x_scaled.view(),
            outcomes,
            variances,
            scales,
            input_k,
            used_k,
            skip,
            epistemic_scale,
            aleatoric_scale,
            observation_noise,
            seeds,
        )
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cuda_conditional(
        &self,
        x: &ArrayView2<f64>,
        x_whatif: &ArrayView2<f64>,
        outcomes: &ArrayView2<f64>,
        variances: Option<&ArrayView2<f64>>,
        scales: &ndarray::ArrayView1<f64>,
        input_k: usize,
        used_k: usize,
        skip: usize,
        epistemic_scale: f64,
        aleatoric_scale: f64,
        observation_noise: bool,
    ) -> Result<crate::draw::DrawInternals, IndexError> {
        if x.ncols() != self.num_dim || x_whatif.ncols() != self.num_dim {
            return Err(IndexError::InvalidShape {
                expected: self.num_dim,
                got: x.ncols().max(x_whatif.ncols()),
            });
        }
        let x_scale = self.x_scale.lock().expect("x_scale mutex poisoned").clone();
        let (x_scaled, whatif_scaled): (Array2<f64>, Array2<f64>) = if self.scale_x {
            (
                x / &x_scale.view().insert_axis(Axis(0)),
                x_whatif / &x_scale.view().insert_axis(Axis(0)),
            )
        } else {
            (x.to_owned(), x_whatif.to_owned())
        };
        self.inner.cuda_conditional(
            &x_scaled.view(),
            &whatif_scaled.view(),
            outcomes,
            variances,
            scales,
            input_k,
            used_k,
            skip,
            epistemic_scale,
            aleatoric_scale,
            observation_noise,
        )
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn condition_draws(
        &self,
        x: &ArrayView2<f64>,
        x_whatif: &ArrayView2<f64>,
        outcomes: &ArrayView2<f64>,
        variances: Option<&ArrayView2<f64>>,
        scales: &ndarray::ArrayView1<f64>,
        input_k: usize,
        used_k: usize,
        skip: usize,
        epistemic_scale: f64,
        aleatoric_scale: f64,
        observation_noise: bool,
        seeds: &[i64],
    ) -> Result<(ndarray::Array3<f64>, Vec<Vec<usize>>), IndexError> {
        if x.ncols() != self.num_dim || x_whatif.ncols() != self.num_dim {
            return Err(IndexError::InvalidShape {
                expected: self.num_dim,
                got: x.ncols().max(x_whatif.ncols()),
            });
        }
        let x_scale = self.x_scale.lock().expect("x_scale mutex poisoned").clone();
        let (x_scaled, whatif_scaled): (Array2<f64>, Array2<f64>) = if self.scale_x {
            (
                x / &x_scale.view().insert_axis(Axis(0)),
                x_whatif / &x_scale.view().insert_axis(Axis(0)),
            )
        } else {
            (x.to_owned(), x_whatif.to_owned())
        };
        self.inner.condition_draws(
            &x_scaled.view(),
            &whatif_scaled.view(),
            outcomes,
            variances,
            scales,
            input_k,
            used_k,
            skip,
            epistemic_scale,
            aleatoric_scale,
            observation_noise,
            seeds,
        )
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Approximate RAM used by the KNN index (not on-disk checkpoint size).
    pub fn memory_usage_bytes(&self) -> usize {
        self.inner.memory_usage_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knn::faiss_backend::{faiss_spec_for_test, make_faiss_for_test};
    use ndarray::array;
    use ndarray::Array2;

    fn index_unit(train_x: Array2<f64>, driver: IndexDriver) -> ENNIndex {
        let x_scale = array![1.0, 1.0];
        ENNIndex::new(train_x, 2, x_scale, false, driver).unwrap()
    }

    fn run_exact_regression_test(make: fn(Array2<f64>) -> ENNIndex) {
        let n = 10usize;
        let train_x = Array2::from_shape_fn((n, 2), |(i, j)| {
            if j == 0 {
                i as f64
            } else {
                (i * 13) as f64 % 5.0
            }
        });
        let index = make(train_x);
        let query = array![[0.25_f64, 0.25]];
        let k = n as i32;
        let (_dist2s, indices) = index.search(&query.view(), k, false).unwrap();
        assert_eq!(indices.ncols(), n);
        for j in 0..n {
            let id = indices[[0, j]];
            assert!(id >= 0 && (id as usize) < n, "invalid id={id} at j={j}");
        }
    }

    #[test]
    fn test_index_creation() {
        let train_x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let index = index_unit(train_x, IndexDriver::Exact);
        assert_eq!(index.len(), 3);
        assert!(!index.is_empty());
    }

    #[test]
    fn test_index_search() {
        let train_x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let index = index_unit(train_x, IndexDriver::Exact);
        let query = array![[0.0, 0.0]];
        let (dist2s, indices) = index.search(&query.view(), 2, false).unwrap();
        assert_eq!(indices[[0, 0]], 0);
        assert!(dist2s[[0, 0]] < 1e-6);
        assert!(dist2s[[0, 1]] > 0.0);
    }

    #[test]
    fn test_index_search_exclude_nearest() {
        let train_x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let index = index_unit(train_x, IndexDriver::Exact);
        let query = array![[0.0, 0.0]];
        let (dist2s, indices) = index.search(&query.view(), 2, true).unwrap();
        assert_eq!(dist2s.ncols(), 1);
        assert_ne!(indices[[0, 0]], 0);
    }

    #[test]
    fn test_index_add() {
        let index = index_unit(array![[0.0, 0.0]], IndexDriver::Exact);
        index.add(&array![[1.0, 1.0]].view()).unwrap();
        assert_eq!(index.len(), 2);
    }

    #[test]
    fn test_invalid_search_k() {
        let index = index_unit(array![[0.0, 0.0]], IndexDriver::Exact);
        let result = index.search(&array![[0.0, 0.0]].view(), 0, false);
        assert!(matches!(result, Err(IndexError::InvalidParameter(_))));
    }

    #[test]
    fn test_invalid_dimensions() {
        let index = index_unit(array![[0.0, 0.0]], IndexDriver::Exact);
        let result = index.search(&array![[0.0, 0.0, 0.0]].view(), 1, false);
        assert!(matches!(
            result,
            Err(IndexError::InvalidShape {
                expected: 2,
                got: 3
            })
        ));
    }

    #[test]
    fn test_exact_search_regression_all_indices_valid_for_k_equals_n() {
        run_exact_regression_test(|train_x| index_unit(train_x, IndexDriver::Exact));
    }

    #[cfg(any(feature = "usearch", feature = "usearch-native"))]
    #[test]
    fn test_usearch_search_and_incremental_add() {
        let index = index_unit(
            array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            IndexDriver::USearch,
        );
        index.add(&array![[0.1, 0.1]].view()).unwrap();
        let (distances, indices) = index
            .search(&array![[0.09, 0.09]].view(), 2, false)
            .unwrap();
        assert_eq!(indices[[0, 0]], 3);
        assert!(distances[[0, 0]] < distances[[0, 1]]);
    }

    #[test]
    fn helpers_preserve() {
        use crate::knn::{arr2_rows_to_f32, pad_neighbor_cols_to_search_k, unpack_batch_search};
        assert_eq!(faiss_spec_for_test(IndexDriver::Exact), "Flat");
        let rows = array![[1.0, 2.0], [3.0, 4.0]];
        let f32v = arr2_rows_to_f32(&rows.view());
        assert_eq!(f32v.len(), 4);
        let index = make_faiss_for_test(2, IndexDriver::Exact, &rows.view()).unwrap();
        assert_eq!(index.len(), 2);
        let (d, _i) = pad_neighbor_cols_to_search_k(array![[1.0, 2.0]], array![[0i64, 1]], 3);
        assert_eq!(d.ncols(), 3);
        let (d2, i2) = unpack_batch_search(1, 2, &[0.5f32, 1.5], &[0, 1]);
        assert_eq!(d2.shape(), &[1, 2]);
        assert_eq!(i2.shape(), &[1, 2]);
    }

    #[test]
    fn test_scaled_search() {
        let x_scale = array![2.0, 2.0];
        let index = ENNIndex::new(
            array![[0.0, 0.0], [1.0, 1.0]],
            2,
            x_scale,
            true,
            IndexDriver::Exact,
        )
        .unwrap();
        let query = array![[2.0, 2.0]];
        let (dist2s, indices) = index.search(&query.view(), 1, false).unwrap();
        assert_eq!(indices[[0, 0]], 1);
        assert!(dist2s[[0, 0]] < 0.0001);
    }

    #[test]
    fn index_driver_rebuild_and_memory() {
        let train_x = array![[0.0, 0.0], [1.0, 1.0]];
        let index = ENNIndex::new(train_x, 2, array![1.0, 1.0], false, IndexDriver::Exact).unwrap();
        assert_eq!(index.driver(), IndexDriver::Exact);
        index
            .rebuild_from_scaled(array![[0.0, 0.0], [1.0, 1.0]], array![1.0, 1.0])
            .unwrap();
        assert!(index.memory_usage_bytes() > 0);
    }
}
