//! Index implementations behind [`crate::index::ENNIndex`].

mod exact_backend;
mod mmap_store;
pub use mmap_store::MmapColumnStore;

#[cfg(any(feature = "usearch", feature = "usearch-native"))]
mod usearch_backend;

#[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
mod cuda_index;
#[cfg(all(target_os = "macos", feature = "metal"))]
mod metal_index;
#[cfg(all(target_os = "macos", feature = "metal"))]
mod metal_plan;
#[cfg(feature = "opencl")]
mod opencl_index;

use ndarray::{Array2, ArrayView2};
#[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
use ndarray::{Array3, ArrayView1};
#[cfg(all(target_os = "macos", feature = "metal"))]
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Duration;
#[cfg(all(target_os = "macos", feature = "metal"))]
use std::time::Instant;

use crate::index::{IndexDriver, IndexError};

use exact_backend::ExactBackend;

/// KNN execution diagram used by the experimental parity surface.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KnnPlan {
    /// Validate equivalent diagrams and retain the fastest Tracy-measured plan.
    Measured,
    /// Materialize tile distances before selecting local neighbors.
    Split,
    /// Fuse tile distance and local top-k computation.
    Fused,
    /// Batch fused tile scans and reduce their neighbor lists as a tree.
    Tree,
    /// Use shared query tiles inside the fused tree scan.
    Tiled,
    /// Cooperatively reduce dimensions before tree top-k selection.
    Simd,
    /// Tile the Gram identity across query and row blocks before tree selection.
    Gram,
    /// Fuse general tile top-k and pairwise-reduce lists through `k=2048`.
    Wide,
}

/// Device-stage timing summary for the latest accelerated KNN search.
#[derive(Clone, Debug)]
pub struct KnnProfile {
    pub rows: usize,
    pub queries: usize,
    pub dims: usize,
    pub k: usize,
    pub plan: &'static str,
    pub gpu: Duration,
    pub scan: Duration,
    pub select: Duration,
    pub reduce: Duration,
}

#[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
pub(crate) type KnnPosterior = (
    Array2<f64>,
    Array2<f64>,
    Array2<f64>,
    Array2<f64>,
    Array2<i64>,
);

#[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
pub(crate) type KnnBatch = (Array3<f64>, Array3<f64>, Array3<f64>, Array3<f64>);

#[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
#[derive(Clone, Copy, Debug)]
pub(crate) struct CudaParam {
    pub used_k: usize,
    pub epistemic_scale: f64,
    pub aleatoric_scale: f64,
}

/// Unstable low-level KNN surface for accelerator parity and performance work.
pub struct KnnIndex(KnnBackend);

impl KnnIndex {
    /// Builds an index using the measured execution plan.
    pub fn new(train: &ArrayView2<f64>, driver: IndexDriver) -> Result<Self, IndexError> {
        Ok(Self(KnnBackend::new(train.ncols(), driver, train)?))
    }

    /// Builds an index with a requested execution plan.
    pub fn with_plan(
        train: &ArrayView2<f64>,
        driver: IndexDriver,
        plan: KnnPlan,
    ) -> Result<Self, IndexError> {
        Ok(Self(KnnBackend::new_plan(
            train.ncols(),
            driver,
            train,
            plan,
        )?))
    }

    /// Returns the number of indexed rows.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Finds the `k` exact nearest neighbors for each query row.
    pub fn search(
        &self,
        queries: &ArrayView2<f64>,
        k: usize,
    ) -> Result<(Array2<f64>, Array2<i64>), IndexError> {
        self.0.search(queries, k, k)
    }

    /// Returns the execution plan selected by the last search.
    pub fn plan(&self) -> &'static str {
        self.0.plan()
    }

    /// Returns device-stage timings from the latest accelerated search.
    pub fn profile(&self) -> Option<KnnProfile> {
        self.0.profile()
    }
}

#[cfg(any(feature = "usearch", feature = "usearch-native"))]
use usearch_backend::USearchBackend;

#[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
use cuda_index::CudaIndex;
#[cfg(all(target_os = "macos", feature = "metal"))]
use metal_index::MetalIndex;
#[cfg(feature = "opencl")]
use opencl_index::OpenClIndex;

/// In-memory exact and accelerator-backed index implementations.
pub(crate) enum KnnBackend {
    Exact(Mutex<ExactBackend>),
    #[cfg(all(target_os = "macos", feature = "metal"))]
    Auto(Mutex<AutoIndex>),
    #[cfg(any(feature = "usearch", feature = "usearch-native"))]
    USearch(Mutex<USearchBackend>),
    #[cfg(all(target_os = "macos", feature = "metal"))]
    Metal(Mutex<MetalIndex>),
    #[cfg(feature = "opencl")]
    OpenCl(Mutex<OpenClIndex>),
    #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
    Cuda(Mutex<CudaIndex>),
}

#[cfg(all(target_os = "macos", feature = "metal"))]
pub(crate) struct AutoIndex {
    cpu: ExactBackend,
    gpu: MetalIndex,
    decisions: HashMap<u32, bool>,
    num_dim: usize,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl AutoIndex {
    fn new(num_dim: usize, train: &ArrayView2<f64>) -> Result<Self, IndexError> {
        Ok(Self {
            cpu: ExactBackend::new(num_dim, train)?,
            gpu: MetalIndex::new_agx(num_dim, train)
                .or_else(|_| MetalIndex::new(num_dim, train))?,
            decisions: HashMap::new(),
            num_dim,
        })
    }

    fn bucket(&self, queries: usize) -> u32 {
        self.cpu
            .len()
            .saturating_mul(queries)
            .saturating_mul(self.num_dim)
            .max(1)
            .ilog2()
    }

    fn search(
        &mut self,
        queries: &ArrayView2<f64>,
        k_eff: usize,
        search_k: usize,
    ) -> Result<(Array2<f64>, Array2<i64>), IndexError> {
        let bucket = self.bucket(queries.nrows());
        if let Some(&gpu) = self.decisions.get(&bucket) {
            return if gpu {
                self.gpu.search(queries, k_eff, search_k)
            } else {
                self.cpu.search(queries, k_eff, search_k)
            };
        }

        let cpu_start = Instant::now();
        let cpu = self.cpu.search(queries, k_eff, search_k)?;
        let cpu_time = cpu_start.elapsed();
        let gpu_start = Instant::now();
        let gpu = self.gpu.search(queries, k_eff, search_k)?;
        let gpu_time = gpu_start.elapsed();
        let use_gpu = cpu.1 == gpu.1 && gpu_time < cpu_time;
        self.decisions.insert(bucket, use_gpu);
        Ok(if use_gpu { gpu } else { cpu })
    }
}

impl KnnBackend {
    pub(crate) fn new(
        num_dim: usize,
        driver: IndexDriver,
        train_scaled: &ArrayView2<f64>,
    ) -> Result<Self, IndexError> {
        Self::new_plan(num_dim, driver, train_scaled, KnnPlan::Measured)
    }

    fn new_plan(
        num_dim: usize,
        driver: IndexDriver,
        train_scaled: &ArrayView2<f64>,
        plan: KnnPlan,
    ) -> Result<Self, IndexError> {
        if plan != KnnPlan::Measured && !matches!(driver, IndexDriver::Metal | IndexDriver::Agx) {
            return Err(IndexError::InvalidParameter(
                "explicit KNN plans require the Metal or AGX driver".to_string(),
            ));
        }
        match driver {
            IndexDriver::Exact => Ok(Self::Exact(Mutex::new(ExactBackend::new(
                num_dim,
                train_scaled,
            )?))),
            IndexDriver::Auto => {
                #[cfg(all(target_os = "macos", feature = "metal"))]
                {
                    Ok(Self::Auto(Mutex::new(AutoIndex::new(
                        num_dim,
                        train_scaled,
                    )?)))
                }
                #[cfg(not(all(target_os = "macos", feature = "metal")))]
                {
                    Ok(Self::Exact(Mutex::new(ExactBackend::new(
                        num_dim,
                        train_scaled,
                    )?)))
                }
            }
            IndexDriver::Agx => {
                #[cfg(all(target_os = "macos", feature = "metal"))]
                {
                    Ok(Self::Metal(Mutex::new(MetalIndex::new_agx_plan(
                        num_dim,
                        train_scaled,
                        plan,
                    )?)))
                }
                #[cfg(not(all(target_os = "macos", feature = "metal")))]
                {
                    Err(IndexError::InvalidParameter(
                        "AGX index is unavailable; build on macOS with the metal feature"
                            .to_string(),
                    ))
                }
            }
            IndexDriver::USearch => {
                #[cfg(any(feature = "usearch", feature = "usearch-native"))]
                {
                    return Ok(Self::USearch(Mutex::new(USearchBackend::new(
                        num_dim,
                        train_scaled,
                    )?)));
                }
                #[cfg(not(any(feature = "usearch", feature = "usearch-native")))]
                {
                    Err(IndexError::InvalidParameter(
                        "USearch index is unavailable; build with the usearch or usearch-native feature".to_string(),
                    ))
                }
            }
            IndexDriver::Metal => {
                #[cfg(all(target_os = "macos", feature = "metal"))]
                {
                    return Ok(Self::Metal(Mutex::new(MetalIndex::new_plan(
                        num_dim,
                        train_scaled,
                        plan,
                    )?)));
                }
                #[cfg(not(all(target_os = "macos", feature = "metal")))]
                {
                    Err(IndexError::InvalidParameter(
                        "Metal index is unavailable; build on macOS with the metal feature"
                            .to_string(),
                    ))
                }
            }
            IndexDriver::OpenCl => {
                #[cfg(feature = "opencl")]
                {
                    return Ok(Self::OpenCl(Mutex::new(OpenClIndex::new(
                        num_dim,
                        train_scaled,
                    )?)));
                }
                #[cfg(not(feature = "opencl"))]
                {
                    Err(IndexError::InvalidParameter(
                        "OpenCL index is unavailable; build with the opencl feature".to_string(),
                    ))
                }
            }
            IndexDriver::Cuda => {
                #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
                {
                    return Ok(Self::Cuda(Mutex::new(CudaIndex::new(
                        num_dim,
                        train_scaled,
                    )?)));
                }
                #[cfg(not(all(target_os = "linux", target_arch = "x86_64", feature = "cuda")))]
                {
                    Err(IndexError::InvalidParameter(
                        "CUDA index is unavailable; build on Linux x86_64 with the cuda feature"
                            .to_string(),
                    ))
                }
            }
            IndexDriver::BpAnnDisk => Err(IndexError::InvalidParameter(
                "IndexDriver::BpAnnDisk is disk-only; use DiskBpannEnnBackend".to_string(),
            )),
        }
    }

    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Exact(inner) => inner.lock().expect("knn mutex poisoned").len(),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Self::Auto(inner) => inner.lock().expect("knn mutex poisoned").cpu.len(),
            #[cfg(any(feature = "usearch", feature = "usearch-native"))]
            Self::USearch(inner) => inner.lock().expect("knn mutex poisoned").len(),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Self::Metal(inner) => inner.lock().expect("knn mutex poisoned").len(),
            #[cfg(feature = "opencl")]
            Self::OpenCl(inner) => inner.lock().expect("knn mutex poisoned").len(),
            #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
            Self::Cuda(inner) => inner.lock().expect("knn mutex poisoned").len(),
        }
    }

    fn plan(&self) -> &'static str {
        match self {
            Self::Exact(_) => "exact",
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Self::Auto(_) => "auto",
            #[cfg(any(feature = "usearch", feature = "usearch-native"))]
            Self::USearch(_) => "usearch",
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Self::Metal(inner) => inner.lock().expect("knn mutex poisoned").plan(),
            #[cfg(feature = "opencl")]
            Self::OpenCl(_) => "opencl",
            #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
            Self::Cuda(inner) => inner
                .lock()
                .expect("knn mutex poisoned")
                .profile()
                .map_or("cuda", |profile| profile.plan),
        }
    }

    fn profile(&self) -> Option<KnnProfile> {
        match self {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Self::Auto(inner) => inner.lock().expect("knn mutex poisoned").gpu.profile(),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Self::Metal(inner) => inner.lock().expect("knn mutex poisoned").profile(),
            #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
            Self::Cuda(inner) => inner.lock().expect("knn mutex poisoned").profile(),
            _ => None,
        }
    }

    pub(crate) fn memory_usage_bytes(&self) -> usize {
        match self {
            Self::Exact(inner) => inner
                .lock()
                .expect("knn mutex poisoned")
                .memory_usage_bytes(),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Self::Auto(inner) => {
                let inner = inner.lock().expect("knn mutex poisoned");
                inner.cpu.memory_usage_bytes() + inner.gpu.memory_usage_bytes()
            }
            #[cfg(any(feature = "usearch", feature = "usearch-native"))]
            Self::USearch(inner) => inner
                .lock()
                .expect("knn mutex poisoned")
                .memory_usage_bytes(),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Self::Metal(inner) => inner
                .lock()
                .expect("knn mutex poisoned")
                .memory_usage_bytes(),
            #[cfg(feature = "opencl")]
            Self::OpenCl(inner) => inner
                .lock()
                .expect("knn mutex poisoned")
                .memory_usage_bytes(),
            #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
            Self::Cuda(inner) => inner.lock().expect("knn mutex poisoned").memory_bytes(),
        }
    }

    pub(crate) fn rebuild(&self, train_scaled: &ArrayView2<f64>) -> Result<(), IndexError> {
        match self {
            Self::Exact(inner) => inner
                .lock()
                .expect("knn mutex poisoned")
                .rebuild(train_scaled),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Self::Auto(inner) => {
                let mut inner = inner.lock().expect("knn mutex poisoned");
                inner.cpu.rebuild(train_scaled)?;
                inner.gpu.rebuild(train_scaled)?;
                inner.decisions.clear();
                Ok(())
            }
            #[cfg(any(feature = "usearch", feature = "usearch-native"))]
            Self::USearch(inner) => inner
                .lock()
                .expect("knn mutex poisoned")
                .rebuild(train_scaled),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Self::Metal(inner) => inner
                .lock()
                .expect("knn mutex poisoned")
                .rebuild(train_scaled),
            #[cfg(feature = "opencl")]
            Self::OpenCl(inner) => inner
                .lock()
                .expect("knn mutex poisoned")
                .rebuild(train_scaled),
            #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
            Self::Cuda(inner) => inner
                .lock()
                .expect("knn mutex poisoned")
                .rebuild(train_scaled),
        }
    }

    pub(crate) fn add(
        &self,
        rows_scaled: &ArrayView2<f64>,
        start_key: u64,
    ) -> Result<(), IndexError> {
        match self {
            Self::Exact(inner) => inner
                .lock()
                .expect("knn mutex poisoned")
                .add(rows_scaled, start_key),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Self::Auto(inner) => {
                let mut inner = inner.lock().expect("knn mutex poisoned");
                inner.cpu.add(rows_scaled, start_key)?;
                inner.gpu.add(rows_scaled, start_key)?;
                Ok(())
            }
            #[cfg(any(feature = "usearch", feature = "usearch-native"))]
            Self::USearch(inner) => inner
                .lock()
                .expect("knn mutex poisoned")
                .add(rows_scaled, start_key),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Self::Metal(inner) => inner
                .lock()
                .expect("knn mutex poisoned")
                .add(rows_scaled, start_key),
            #[cfg(feature = "opencl")]
            Self::OpenCl(inner) => inner
                .lock()
                .expect("knn mutex poisoned")
                .add(rows_scaled, start_key),
            #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
            Self::Cuda(inner) => inner
                .lock()
                .expect("knn mutex poisoned")
                .add(rows_scaled, start_key),
        }
    }

    pub(crate) fn search(
        &self,
        queries_scaled: &ArrayView2<f64>,
        k_eff: usize,
        search_k: usize,
    ) -> Result<(Array2<f64>, Array2<i64>), IndexError> {
        let span = crate::tracy::zone(tracy_client::span_location!("knn.search"));
        span.emit_value(queries_scaled.nrows() as u64);
        match self {
            Self::Exact(inner) => {
                inner
                    .lock()
                    .expect("knn mutex poisoned")
                    .search(queries_scaled, k_eff, search_k)
            }
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Self::Auto(inner) => {
                inner
                    .lock()
                    .expect("knn mutex poisoned")
                    .search(queries_scaled, k_eff, search_k)
            }
            #[cfg(any(feature = "usearch", feature = "usearch-native"))]
            Self::USearch(inner) => {
                inner
                    .lock()
                    .expect("knn mutex poisoned")
                    .search(queries_scaled, k_eff, search_k)
            }
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Self::Metal(inner) => {
                inner
                    .lock()
                    .expect("knn mutex poisoned")
                    .search(queries_scaled, k_eff, search_k)
            }
            #[cfg(feature = "opencl")]
            Self::OpenCl(inner) => {
                inner
                    .lock()
                    .expect("knn mutex poisoned")
                    .search(queries_scaled, k_eff, search_k)
            }
            #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
            Self::Cuda(inner) => {
                inner
                    .lock()
                    .expect("knn mutex poisoned")
                    .search(queries_scaled, k_eff, search_k)
            }
        }
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cuda_posterior(
        &self,
        queries: &ArrayView2<f64>,
        outcomes: &ArrayView2<f64>,
        scales: &ArrayView1<f64>,
        input_k: usize,
        used_k: usize,
        skip: usize,
        epistemic_scale: f64,
        aleatoric_scale: f64,
    ) -> Result<KnnPosterior, IndexError> {
        let Self::Cuda(inner) = self else {
            return Err(IndexError::InvalidParameter(
                "CUDA posterior requires IndexDriver::Cuda".to_string(),
            ));
        };
        inner.lock().expect("knn mutex poisoned").posterior(
            queries,
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
    ) -> Result<crate::draw::DrawInternals, IndexError> {
        let Self::Cuda(inner) = self else {
            return Err(IndexError::InvalidParameter(
                "CUDA weighted posterior requires IndexDriver::Cuda".to_string(),
            ));
        };
        inner.lock().expect("knn mutex poisoned").weighted(
            queries,
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
        queries: &ArrayView2<f64>,
        outcomes: &ArrayView2<f64>,
        variances: Option<&ArrayView2<f64>>,
        scales: &ArrayView1<f64>,
        input_k: usize,
        skip: usize,
        values: &[CudaParam],
        observation_noise: bool,
    ) -> Result<KnnBatch, IndexError> {
        let Self::Cuda(inner) = self else {
            return Err(IndexError::InvalidParameter(
                "CUDA batch posterior requires IndexDriver::Cuda".to_string(),
            ));
        };
        inner.lock().expect("knn mutex poisoned").batch(
            queries,
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
        let Self::Cuda(inner) = self else {
            return Err(IndexError::InvalidParameter(
                "CUDA draws require IndexDriver::Cuda".to_string(),
            ));
        };
        inner.lock().expect("knn mutex poisoned").draws(
            queries,
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
    ) -> Result<crate::draw::DrawInternals, IndexError> {
        let Self::Cuda(inner) = self else {
            return Err(IndexError::InvalidParameter(
                "CUDA conditional posterior requires IndexDriver::Cuda".to_string(),
            ));
        };
        inner.lock().expect("knn mutex poisoned").conditional(
            queries,
            extra_rows,
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
        let Self::Cuda(inner) = self else {
            return Err(IndexError::InvalidParameter(
                "CUDA conditional draws require IndexDriver::Cuda".to_string(),
            ));
        };
        inner.lock().expect("knn mutex poisoned").conditional_draws(
            queries,
            extra_rows,
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
}

pub(crate) fn arr2_rows_to_f32(a: &ArrayView2<f64>) -> Vec<f32> {
    a.iter().map(|v| *v as f32).collect()
}

pub(crate) fn pad_neighbor_cols_to_search_k(
    dist2s: Array2<f64>,
    idx: Array2<i64>,
    search_k: usize,
) -> (Array2<f64>, Array2<i64>) {
    use ndarray::{concatenate, Axis};
    let k_eff = dist2s.ncols();
    if k_eff >= search_k {
        return (dist2s, idx);
    }
    let n_query = dist2s.nrows();
    if k_eff == 0 {
        return (
            Array2::from_elem((n_query, search_k), f64::INFINITY),
            Array2::zeros((n_query, search_k)),
        );
    }
    let pad_w = search_k - k_eff;
    let pad_dist = Array2::from_elem((n_query, pad_w), f64::INFINITY);
    let far = idx.slice(ndarray::s![.., k_eff - 1..k_eff]).to_owned();
    let mut pad_idx = Array2::zeros((n_query, pad_w));
    for j in 0..pad_w {
        pad_idx.column_mut(j).assign(&far.column(0));
    }
    (
        concatenate![Axis(1), dist2s.view(), pad_dist.view()],
        concatenate![Axis(1), idx.view(), pad_idx.view()],
    )
}

pub(crate) fn unpack_batch_search(
    n_query: usize,
    k: usize,
    distances: &[f32],
    labels: &[i64],
) -> (Array2<f64>, Array2<i64>) {
    let mut dist2s = Array2::zeros((n_query, k));
    let mut indices = Array2::zeros((n_query, k));
    for i in 0..n_query {
        for j in 0..k {
            let o = i * k + j;
            dist2s[[i, j]] = f64::from(distances[o]);
            indices[[i, j]] = labels[o];
        }
    }
    (dist2s, indices)
}

#[cfg(test)]
mod knn_backend_tests {
    use super::*;
    use crate::index::IndexDriver;
    use ndarray::array;

    #[test]
    fn knn_backend_exact() {
        let train = array![[0.0, 0.0], [1.0, 1.0]];
        let backend = KnnBackend::new(2, IndexDriver::Exact, &train.view()).unwrap();
        assert_eq!(backend.len(), 2);
        backend.add(&array![[2.0, 2.0]].view(), 2).unwrap();
        assert_eq!(backend.len(), 3);
        let (_d, i) = backend.search(&array![[0.0, 0.0]].view(), 2, 2).unwrap();
        assert_eq!(i[[0, 0]], 0);
        backend.rebuild(&train.view()).unwrap();
    }

    #[test]
    fn knn_backend_auto_preserves_exact_contract() {
        let train = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let mut auto = AutoIndex::new(2, &train.view()).unwrap();
            let (_distances, indices) = auto.search(&array![[0.1, 0.1]].view(), 2, 2).unwrap();
            assert_eq!(indices.row(0).to_vec(), vec![0, 1]);
        }
        let backend = KnnBackend::new(2, IndexDriver::Auto, &train.view()).unwrap();
        assert!(backend.memory_usage_bytes() > 0);
        assert_eq!(arr2_rows_to_f32(&train.view()).len(), 6);
        let (_distances, indices) = backend.search(&array![[0.1, 0.1]].view(), 2, 2).unwrap();
        assert_eq!(indices.row(0).to_vec(), vec![0, 1]);
        backend.add(&array![[0.05, 0.05]].view(), 3).unwrap();
        let (_distances, indices) = backend.search(&array![[0.1, 0.1]].view(), 2, 2).unwrap();
        assert_eq!(indices[[0, 0]], 3);
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn knn_backend_agx_matches_exact() {
        let train = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [0.5, 0.5]];
        let query = array![[0.1, 0.1], [1.8, 1.8]];
        let exact = KnnBackend::new(2, IndexDriver::Exact, &train.view()).unwrap();
        let agx = KnnBackend::new(2, IndexDriver::Agx, &train.view()).unwrap();
        let expected = exact.search(&query.view(), 3, 3).unwrap();
        let actual = agx.search(&query.view(), 3, 3).unwrap();
        assert_eq!(actual.1, expected.1);
        for (left, right) in actual.0.iter().zip(expected.0.iter()) {
            assert!((left - right).abs() <= 1e-5);
        }
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn knn_backend_agx_handles_schedule_tails() {
        for dimensions in [3, 5, 32, 33] {
            let train = Array2::from_shape_fn((67, dimensions), |(row, col)| {
                ((row * 17 + col * 11) % 101) as f64 / 101.0
            });
            let query = Array2::from_shape_fn((7, dimensions), |(row, col)| {
                ((row * 13 + col * 19 + 3) % 103) as f64 / 103.0
            });
            let exact = KnnBackend::new(dimensions, IndexDriver::Exact, &train.view()).unwrap();
            let agx = KnnBackend::new(dimensions, IndexDriver::Agx, &train.view()).unwrap();
            let expected = exact.search(&query.view(), 8, 8).unwrap();
            let actual = agx.search(&query.view(), 8, 8).unwrap();
            assert_eq!(actual.1, expected.1);
            for (left, right) in actual.0.iter().zip(expected.0.iter()) {
                assert!((left - right).abs() <= 1e-4);
            }
        }
    }

    #[test]
    fn knn_backend_bpann_disk_driver_errors() {
        let train = array![[0.0, 0.0], [1.0, 0.0]];
        match KnnBackend::new(2, IndexDriver::BpAnnDisk, &train.view()) {
            Err(e) => assert!(e.to_string().contains("disk-only")),
            Ok(_) => panic!("expected BpAnnDisk on KnnBackend to error"),
        }
    }

    #[cfg(any(feature = "usearch", feature = "usearch-native"))]
    #[test]
    fn knn_backend_usearch() {
        let train = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let backend = KnnBackend::new(2, IndexDriver::USearch, &train.view()).unwrap();
        assert_eq!(backend.len(), 3);
        assert!(matches!(backend, KnnBackend::USearch(_)));
    }

    #[cfg(not(any(feature = "usearch", feature = "usearch-native")))]
    #[test]
    fn knn_backend_usearch_requires_feature() {
        let train = array![[0.0, 0.0]];
        let error = match KnnBackend::new(2, IndexDriver::USearch, &train.view()) {
            Ok(_) => panic!("expected USearch feature error"),
            Err(error) => error,
        };
        assert!(error
            .to_string()
            .contains("usearch or usearch-native feature"));
    }

    #[test]
    fn pad_and_unpack_helpers() {
        let (d, i) = pad_neighbor_cols_to_search_k(array![[1.0]], array![[0i64]], 3);
        assert_eq!(d.ncols(), 3);
        assert_eq!(i.ncols(), 3);
        let (d2, i2) = unpack_batch_search(1, 2, &[0.5, 1.5], &[0, 1]);
        assert_eq!(d2[[0, 1]], 1.5);
        assert_eq!(i2[[0, 1]], 1);
    }

    #[cfg(any(all(target_os = "macos", feature = "metal"), feature = "opencl"))]
    fn check_device_backend(device: KnnBackend) {
        let train = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 2.0]];
        let query = array![[0.2, 0.1], [1.1, 0.2]];
        let exact = KnnBackend::new(2, IndexDriver::Exact, &train.view()).unwrap();
        let expected = exact.search(&query.view(), 3, 3).unwrap();
        let actual = device.search(&query.view(), 3, 3).unwrap();
        assert_eq!(actual.1, expected.1);
        for (actual, expected) in actual.0.iter().zip(expected.0.iter()) {
            assert!((actual - expected).abs() < 1.0e-5, "{actual} != {expected}");
        }

        let large_train = Array2::from_shape_fn((1050, 2), |(row, col)| {
            if col == 0 {
                row as f64 * 0.01
            } else {
                (row % 17) as f64 * 0.1
            }
        });
        let large_query = array![[2.345, 0.7], [7.891, 1.1]];
        let large_exact = KnnBackend::new(2, IndexDriver::Exact, &large_train.view()).unwrap();
        let large_expected = large_exact.search(&large_query.view(), 10, 10).unwrap();
        device.rebuild(&large_train.view()).unwrap();
        let large_actual = device.search(&large_query.view(), 10, 10).unwrap();
        assert_eq!(large_actual.1, large_expected.1);
        for (actual, expected) in large_actual.0.iter().zip(large_expected.0.iter()) {
            assert!((actual - expected).abs() < 1.0e-4, "{actual} != {expected}");
        }

        device.add(&array![[0.2, 0.2]].view(), 4).unwrap();
        assert_eq!(device.len(), 1051);
        device.rebuild(&train.view()).unwrap();
        assert_eq!(device.len(), 4);
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn metal_index_matches_exact() {
        let train = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 2.0]];
        let device = match KnnBackend::new(2, IndexDriver::Metal, &train.view()) {
            Ok(index) => index,
            Err(error) => {
                eprintln!("Metal runtime unavailable: {error}");
                return;
            }
        };
        check_device_backend(device);
    }

    #[cfg(feature = "opencl")]
    #[test]
    fn opencl_index_matches_exact() {
        let train = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 2.0]];
        let device = match KnnBackend::new(2, IndexDriver::OpenCl, &train.view()) {
            Ok(index) => index,
            Err(error) => {
                eprintln!("OpenCL runtime unavailable: {error}");
                return;
            }
        };
        check_device_backend(device);
    }
}
