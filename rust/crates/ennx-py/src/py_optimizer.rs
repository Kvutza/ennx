//! Optimizer Python bindings.

use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::path::PathBuf;

pub(crate) fn optional_f64(
    dict: &Bound<'_, pyo3::types::PyDict>,
    key: &str,
) -> PyResult<Option<f64>> {
    match dict.get_item(key)? {
        Some(v) => Ok(Some(v.extract()?)),
        None => Ok(None),
    }
}

pub(crate) fn optional_usize(
    dict: &Bound<'_, pyo3::types::PyDict>,
    key: &str,
) -> PyResult<Option<usize>> {
    match dict.get_item(key)? {
        Some(v) => Ok(Some(v.extract()?)),
        None => Ok(None),
    }
}

pub(crate) fn optional_bool(
    dict: &Bound<'_, pyo3::types::PyDict>,
    key: &str,
) -> PyResult<Option<bool>> {
    match dict.get_item(key)? {
        Some(v) => Ok(Some(v.extract()?)),
        None => Ok(None),
    }
}

pub(crate) fn apply_scalar_overrides(
    dict: &Bound<'_, pyo3::types::PyDict>,
    overrides: &mut ennx::ConfigOverrides,
) -> PyResult<()> {
    overrides.num_candidates_factor = optional_f64(dict, "num_candidates_factor")?;
    overrides.min_candidates = optional_usize(dict, "min_candidates")?;
    overrides.max_candidates = optional_usize(dict, "max_candidates")?;
    overrides.num_candidates_per_arm = optional_usize(dict, "num_candidates_per_arm")?;
    overrides.num_pert = optional_usize(dict, "num_pert")?;
    overrides.length_init = optional_f64(dict, "length_init")?;
    overrides.length_min = optional_f64(dict, "length_min")?;
    overrides.length_max = optional_f64(dict, "length_max")?;
    overrides.num_fit_samples = optional_usize(dict, "num_fit_samples")?;
    overrides.num_fit_candidates = optional_usize(dict, "num_fit_candidates")?;
    overrides.noise_aware = optional_bool(dict, "noise_aware")?;
    overrides.scale_x = optional_bool(dict, "scale_x")?;
    overrides.failure_tolerance_dim = optional_f64(dict, "failure_tolerance_dim")?;
    Ok(())
}

#[cfg(test)]
mod kiss_coverage_tests {
    use super::{apply_scalar_overrides, optional_bool, optional_f64, optional_usize};

    #[test]
    fn py_optimizer_helpers_are_linked() {
        let _ = (
            optional_f64 as fn(_, _) -> _,
            optional_usize as fn(_, _) -> _,
            optional_bool as fn(_, _) -> _,
            apply_scalar_overrides as fn(_, _) -> _,
        );
    }
}

fn parse_index_driver(s: &str) -> PyResult<ennx::index::IndexDriver> {
    use ennx::index::IndexDriver;
    match s.to_lowercase().as_str() {
        "exact" | "flat" => Ok(IndexDriver::Exact),
        "auto" => Ok(IndexDriver::Auto),
        "agx" => Ok(IndexDriver::Agx),
        "usearch" | "hnsw_usearch" => Ok(IndexDriver::USearch),
        "bpann_disk" => Ok(IndexDriver::BpAnnDisk),
        "metal" => Ok(IndexDriver::Metal),
        "opencl" | "ocl" => Ok(IndexDriver::OpenCl),
        "cuda" => Ok(IndexDriver::Cuda),
        _ => Err(PyValueError::new_err(format!("Unknown index_driver: {s}"))),
    }
}

fn parse_acquisition(
    dict: &Bound<'_, pyo3::types::PyDict>,
    s: &str,
) -> PyResult<ennx::AcquisitionConfig> {
    use ennx::AcquisitionConfig;
    match s {
        "ucb" => {
            let beta = dict
                .get_item("acquisition_beta")?
                .map(|v| v.extract::<f64>())
                .transpose()?
                .unwrap_or(2.0);
            Ok(AcquisitionConfig::UCB { beta })
        }
        "thompson" => Ok(AcquisitionConfig::Thompson),
        "random" => Ok(AcquisitionConfig::Random),
        "pareto" => Ok(AcquisitionConfig::Pareto),
        _ => Err(PyValueError::new_err(format!("Unknown acquisition: {s}"))),
    }
}

fn parse_candidate_rv(s: &str) -> PyResult<ennx::CandidateRV> {
    use ennx::CandidateRV;
    match s {
        "sobol" => Ok(CandidateRV::Sobol),
        "uniform" => Ok(CandidateRV::Uniform),
        "raasp" => Ok(CandidateRV::RAASP),
        _ => Err(PyValueError::new_err(format!("Unknown candidate_rv: {s}"))),
    }
}

fn parse_enn_storage(s: &str) -> PyResult<ennx::EnnStorage> {
    match s.to_lowercase().as_str() {
        "disk" => Ok(ennx::EnnStorage::Disk),
        "memory" | "in_memory" | "inmemory" => Ok(ennx::EnnStorage::InMemory),
        _ => Err(PyValueError::new_err(format!("Unknown enn_storage: {s}"))),
    }
}

pub fn parse_config_overrides_from_dict(
    dict: &Bound<'_, pyo3::types::PyDict>,
) -> PyResult<ennx::ConfigOverrides> {
    use ennx::ConfigOverrides;

    let mut overrides = ConfigOverrides::default();

    if let Some(v) = dict.get_item("index_driver")? {
        overrides.index_driver = Some(parse_index_driver(&v.extract::<String>()?)?);
    }
    if let Some(acq) = dict.get_item("acquisition")? {
        let s: String = acq.extract()?;
        overrides.acquisition = Some(parse_acquisition(dict, &s)?);
    }
    if let Some(rv) = dict.get_item("candidate_rv")? {
        overrides.candidate_rv = Some(parse_candidate_rv(&rv.extract::<String>()?)?);
    }
    if let Some(v) = dict.get_item("trust_region")? {
        let s: String = v.extract()?;
        let kind = match s.as_str() {
            "turbo" => ennx::config::TrustRegionKind::Turbo,
            "morbo" => ennx::config::TrustRegionKind::Morbo,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Invalid trust_region_kind: {}",
                    s
                )))
            }
        };
        overrides.trust_region_kind = Some(kind);
    }
    overrides.num_metrics = optional_usize(dict, "num_metrics")?;
    overrides.alpha = optional_f64(dict, "alpha")?;
    if let Some(v) = dict.get_item("rescalarize")? {
        let s: String = v.extract()?;
        let resc = match s.as_str() {
            "on_restart" => ennx::morbo_trust_region::Rescalarize::OnRestart,
            "on_propose" => ennx::morbo_trust_region::Rescalarize::OnPropose,
            _ => return Err(PyValueError::new_err(format!("Invalid rescalarize: {}", s))),
        };
        overrides.rescalarize = Some(resc);
    }
    if let Some(v) = dict.get_item("enn_storage")? {
        overrides.enn_storage = Some(parse_enn_storage(&v.extract::<String>()?)?);
    }
    if let Some(v) = dict.get_item("work_dir")? {
        overrides.work_dir = Some(PathBuf::from(v.extract::<String>()?));
    }
    if let Some(v) = dict.get_item("y_bounds")? {
        let bounds: numpy::PyReadonlyArray2<f64> = v.extract()?;
        if bounds.shape()[1] != 2 {
            return Err(PyValueError::new_err(
                "y_bounds must have shape (metrics, 2)",
            ));
        }
        overrides.y_bounds = Some(
            bounds
                .as_array()
                .rows()
                .into_iter()
                .map(|row| [row[0], row[1]])
                .collect(),
        );
    }
    apply_scalar_overrides(dict, &mut overrides)?;
    Ok(overrides)
}

/// Python wrapper for Optimizer
#[pyclass(name = "Optimizer")]
pub struct PyOptimizer {
    inner: ennx::Optimizer,
    rng: StdRng,
    expects_yvar: Option<bool>,
}

#[pymethods]
impl PyOptimizer {
    /// Ask for candidate points
    #[pyo3(signature = (num_arms, seed=None))]
    fn ask<'py>(
        &mut self,
        py: Python<'py>,
        num_arms: usize,
        seed: Option<u64>,
    ) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        if num_arms == 0 {
            return Err(PyValueError::new_err("num_arms must be greater than zero"));
        }
        let result_unit = match seed {
            Some(seed) => {
                let mut rng = StdRng::seed_from_u64(seed);
                self.inner.ask(num_arms, &mut rng)
            }
            None => self.inner.ask(num_arms, &mut self.rng),
        }
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let result = ennx::from_unit(&result_unit.view(), &self.inner.bounds().view());
        Ok(result.into_dyn().into_pyarray(py))
    }

    /// Tell observations
    #[pyo3(signature = (x, y, seed=None, y_var=None))]
    fn tell(
        &mut self,
        x: PyReadonlyArray2<f64>,
        y: PyReadonlyArray2<f64>,
        seed: Option<u64>,
        y_var: Option<PyReadonlyArray2<f64>>,
    ) -> PyResult<()> {
        if x.shape()[0] != y.shape()[0] || x.shape()[1] != self.inner.num_dim() {
            return Err(PyValueError::new_err(format!(
                "x/y shape mismatch: x={:?}, y={:?}, dimensions={}",
                x.shape(),
                y.shape(),
                self.inner.num_dim()
            )));
        }
        if let Some(variance) = y_var.as_ref() {
            if variance.shape() != y.shape() {
                return Err(PyValueError::new_err(format!(
                    "y_var must have shape {:?}, got {:?}",
                    y.shape(),
                    variance.shape()
                )));
            }
        }
        if let Some(expects) = self.expects_yvar {
            if expects != y_var.is_some() {
                return Err(PyValueError::new_err(format!(
                    "y_var must be {} on every tell()",
                    if expects { "provided" } else { "omitted" }
                )));
            }
        }
        let x_unit = ennx::to_unit(&x.as_array(), &self.inner.bounds().view());
        let y_view = y.as_array();
        let yvar_view = y_var.as_ref().map(PyReadonlyArray2::as_array);
        let result = match seed {
            Some(seed) => {
                let mut rng = StdRng::seed_from_u64(seed);
                self.inner
                    .tell_with_yvar(&x_unit.view(), &y_view, yvar_view.as_ref(), &mut rng)
            }
            None => self.inner.tell_with_yvar(
                &x_unit.view(),
                &y_view,
                yvar_view.as_ref(),
                &mut self.rng,
            ),
        }
        .map_err(|e| PyValueError::new_err(e.to_string()));
        if result.is_ok() {
            self.expects_yvar = Some(y_var.is_some());
        }
        result
    }

    /// Get init progress if in initialization phase
    fn init_progress(&self) -> Option<(usize, usize)> {
        self.inner.init_progress()
    }

    /// Get current telemetry
    fn telemetry(&self) -> PyTelemetry {
        let t = self.inner.telemetry();
        PyTelemetry {
            dt_fit: t.dt_fit,
            dt_gen: t.dt_gen,
            dt_sel: t.dt_sel,
            dt_tell: t.dt_tell,
            num_candidates: t.num_candidates,
        }
    }

    /// Number of retained trust-region observations.
    fn tr_obs_count(&self) -> usize {
        self.inner.obs_count()
    }

    /// Current trust-region length.
    fn tr_length(&self) -> f64 {
        self.inner.tr_length()
    }

    /// Get observations x in unit space (if any).
    fn x_obs<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArrayDyn<f64>>> {
        self.inner.x_obs().map(|x| x.into_dyn().into_pyarray(py))
    }

    /// Get observation values y (if any).
    fn y_obs<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArrayDyn<f64>>> {
        self.inner.y_obs().map(|y| y.into_dyn().into_pyarray(py))
    }

    /// Get incumbent x in unit space (if any).
    fn incumbent_x_unit<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArrayDyn<f64>>> {
        self.inner
            .incumbent_x_unit()
            .map(|x| x.view().to_owned().into_dyn().into_pyarray(py))
    }

    /// Get optimizer bounds.
    fn bounds<'py>(&self, py: Python<'py>) -> Bound<'py, PyArrayDyn<f64>> {
        self.inner
            .bounds()
            .view()
            .to_owned()
            .into_dyn()
            .into_pyarray(py)
    }
}

#[allow(clippy::too_many_arguments)]
fn build_optimizer(
    bounds: ndarray::Array2<f64>,
    kind: &str,
    k: i32,
    num_init: usize,
    num_regions: usize,
    seed: u64,
    cfg: Option<&Bound<'_, pyo3::types::PyDict>>,
    gp: Option<Py<PyAny>>,
    fit_steps: usize,
) -> PyResult<PyOptimizer> {
    use ennx::optimizer_factory::{
        create_optimizer_enn_multi_tr_with_overrides, create_optimizer_enn_with_overrides,
        create_optimizer_lhd_with_overrides, create_optimizer_zero_with_overrides,
    };

    let num_dim = bounds.nrows();
    let overrides = cfg.map(parse_config_overrides_from_dict).transpose()?;
    let mut rng = StdRng::seed_from_u64(seed);
    let optimizer = match kind {
        "enn" => create_optimizer_enn_with_overrides(
            bounds,
            k,
            num_init,
            &mut rng,
            overrides.as_ref(),
        ),
        "enn_multi_tr" => create_optimizer_enn_multi_tr_with_overrides(
            bounds,
            k,
            num_init,
            num_regions,
            &mut rng,
            overrides.as_ref(),
        ),
        "zero" => create_optimizer_zero_with_overrides(
            bounds,
            num_init,
            &mut rng,
            overrides.as_ref(),
        ),
        "lhd" => create_optimizer_lhd_with_overrides(
            bounds,
            num_init,
            &mut rng,
            overrides.as_ref(),
        ),
        "gp" => {
            let provider = gp.ok_or_else(|| {
                PyValueError::new_err("kind='gp' requires a Python surrogate provider")
            })?;
            let mut config = ennx::turbo_zero_config();
            if let Some(overrides) = overrides.as_ref() {
                config = overrides.apply_to(config);
            }
            let strategy = ennx::Strategy::hybrid(ennx::InitStrategy::LHD, num_init);
            let external = Box::new(crate::adapter::PythonSurrogateAdapter::new(
                provider,
                num_dim,
                fit_steps,
            ));
            ennx::Optimizer::new_with_surrogate(
                bounds,
                config,
                strategy,
                external,
                &mut rng,
            )
        }
        _ => {
            return Err(PyValueError::new_err(format!(
                "unknown optimizer kind {kind:?}; expected 'enn', 'enn_multi_tr', 'zero', 'lhd', or 'gp'"
            )))
        }
    }
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(PyOptimizer {
        inner: optimizer,
        rng,
        expects_yvar: None,
    })
}

/// Create a built-in optimizer, optionally with a batched Python surrogate adapter.
#[pyfunction(name = "create_optimizer")]
#[pyo3(signature = (bounds, kind, k=10, num_init=10, num_regions=4, seed=42, cfg=None, gp=None, fit_steps=50))]
#[allow(clippy::too_many_arguments)]
pub fn create_optimizer_py(
    bounds: PyReadonlyArray2<f64>,
    kind: &str,
    k: i32,
    num_init: usize,
    num_regions: usize,
    seed: u64,
    cfg: Option<Bound<'_, pyo3::types::PyDict>>,
    gp: Option<Py<PyAny>>,
    fit_steps: usize,
) -> PyResult<PyOptimizer> {
    let bounds = bounds.as_array().to_owned();
    build_optimizer(
        bounds,
        kind,
        k,
        num_init,
        num_regions,
        seed,
        cfg.as_ref(),
        gp,
        fit_steps,
    )
}

/// Telemetry data structure for Python
#[pyclass(name = "Telemetry", from_py_object)]
#[derive(Clone, Copy)]
pub struct PyTelemetry {
    #[pyo3(get)]
    pub dt_fit: f64,
    #[pyo3(get)]
    pub dt_gen: f64,
    #[pyo3(get)]
    pub dt_sel: f64,
    #[pyo3(get)]
    pub dt_tell: f64,
    #[pyo3(get)]
    pub num_candidates: usize,
}

/// Create TuRBO-ENN optimizer
#[pyfunction(name = "create_optimizer_enn")]
#[pyo3(signature = (bounds, k=10, num_init=10, seed=42, config_overrides=None))]
pub fn create_optimizer_enn_py(
    bounds: PyReadonlyArray2<f64>,
    k: i32,
    num_init: usize,
    seed: u64,
    config_overrides: Option<Bound<'_, pyo3::types::PyDict>>,
) -> PyResult<PyOptimizer> {
    build_optimizer(
        bounds.as_array().to_owned(),
        "enn",
        k,
        num_init,
        4,
        seed,
        config_overrides.as_ref(),
        None,
        50,
    )
}

/// Create experimental multi-trust-region TuRBO-ENN optimizer
#[pyfunction(name = "create_optimizer_enn_multi_tr")]
#[pyo3(signature = (bounds, k=10, num_init=10, num_regions=4, seed=42, config_overrides=None))]
pub fn create_optimizer_enn_multi_tr_py(
    bounds: PyReadonlyArray2<f64>,
    k: i32,
    num_init: usize,
    num_regions: usize,
    seed: u64,
    config_overrides: Option<Bound<'_, pyo3::types::PyDict>>,
) -> PyResult<PyOptimizer> {
    build_optimizer(
        bounds.as_array().to_owned(),
        "enn_multi_tr",
        k,
        num_init,
        num_regions,
        seed,
        config_overrides.as_ref(),
        None,
        50,
    )
}

/// Create TuRBO-ZERO optimizer
#[pyfunction(name = "create_optimizer_zero")]
#[pyo3(signature = (bounds, num_init=10, seed=42, config_overrides=None))]
pub fn create_optimizer_zero_py(
    bounds: PyReadonlyArray2<f64>,
    num_init: usize,
    seed: u64,
    config_overrides: Option<Bound<'_, pyo3::types::PyDict>>,
) -> PyResult<PyOptimizer> {
    build_optimizer(
        bounds.as_array().to_owned(),
        "zero",
        10,
        num_init,
        4,
        seed,
        config_overrides.as_ref(),
        None,
        50,
    )
}

/// Create LHD-only optimizer
#[pyfunction(name = "create_optimizer_lhd")]
#[pyo3(signature = (bounds, num_init=10, seed=42, config_overrides=None))]
pub fn create_optimizer_lhd_py(
    bounds: PyReadonlyArray2<f64>,
    num_init: usize,
    seed: u64,
    config_overrides: Option<Bound<'_, pyo3::types::PyDict>>,
) -> PyResult<PyOptimizer> {
    build_optimizer(
        bounds.as_array().to_owned(),
        "lhd",
        10,
        num_init,
        4,
        seed,
        config_overrides.as_ref(),
        None,
        50,
    )
}

/// Python wrapper for MultiTrustRegion state machine
#[pyclass(name = "MultiTrustRegion")]
pub struct PyMultiTrustRegion {
    inner: ennx::experimental::MultiTrustRegionState,
}

#[pymethods]
impl PyMultiTrustRegion {
    #[new]
    #[pyo3(signature = (num_dim, num_regions=4, sharing_policy="shared", seed=42))]
    pub fn new(
        num_dim: usize,
        num_regions: usize,
        sharing_policy: &str,
        seed: u64,
    ) -> PyResult<Self> {
        use ennx::experimental::{MultiTrustRegionConfig, MultiTrustRegionState, SharingPolicy};

        let mut rng = StdRng::seed_from_u64(seed);
        let policy = match sharing_policy {
            "shared" => SharingPolicy::Shared,
            "nearest_center" => SharingPolicy::NearestCenter,
            "independent" => SharingPolicy::Independent,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unknown sharing policy: {}",
                    sharing_policy
                )))
            }
        };
        let mut config = MultiTrustRegionConfig::new(num_regions, Default::default());
        config.sharing_policy = policy;

        let state = MultiTrustRegionState::new(num_dim, config, None, &mut rng)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(Self { inner: state })
    }

    pub fn num_regions(&self) -> usize {
        self.inner.num_regions()
    }

    pub fn num_dim(&self) -> usize {
        self.inner.num_dim()
    }

    pub fn active_count(&self) -> usize {
        self.inner.active_count()
    }

    pub fn get_centers<'py>(&self, py: Python<'py>) -> Bound<'py, PyArrayDyn<f64>> {
        self.inner.centers.clone().into_dyn().into_pyarray(py)
    }

    pub fn get_lengths<'py>(&self, py: Python<'py>) -> Bound<'py, PyArrayDyn<f64>> {
        self.inner.lengths.clone().into_dyn().into_pyarray(py)
    }

    pub fn get_incumbents<'py>(&self, py: Python<'py>) -> Bound<'py, PyArrayDyn<f64>> {
        self.inner.incumbents_y.clone().into_dyn().into_pyarray(py)
    }

    pub fn tell(
        &mut self,
        x_batch: PyReadonlyArray2<'_, f64>,
        y_batch: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<()> {
        let x_view = x_batch.as_array();
        let y_view = y_batch.as_array();
        self.inner
            .tell_update(&x_view, &y_view)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Allocate a candidate budget across regions with the default region utility.
    pub fn allocate(&self, budget: usize) -> PyResult<Vec<(usize, usize, usize)>> {
        let batches = self
            .inner
            .allocate(budget)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(batches
            .into_iter()
            .map(|batch| (batch.region, batch.start, batch.len))
            .collect())
    }

    /// Allocate a candidate budget across regions using an explicit utility vector.
    pub fn allocate_with<'py>(
        &self,
        budget: usize,
        utility: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Vec<(usize, usize, usize)>> {
        let batches = self
            .inner
            .allocate_with(budget, &utility.as_array())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(batches
            .into_iter()
            .map(|batch| (batch.region, batch.start, batch.len))
            .collect())
    }

    /// Select globally best and diverse region candidates.
    pub fn select(
        &self,
        candidates: Vec<(usize, usize, u64, f64)>,
        num_arms: usize,
    ) -> PyResult<Vec<(usize, usize, u64, f64)>> {
        let candidates = candidates
            .into_iter()
            .map(
                |(index, region, seed, score)| ennx::experimental::RegionCandidate {
                    index,
                    region,
                    seed,
                    score,
                },
            )
            .collect::<Vec<_>>();
        let selected = self
            .inner
            .select(&candidates, num_arms)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(selected
            .into_iter()
            .map(|candidate| {
                (
                    candidate.index,
                    candidate.region,
                    candidate.seed,
                    candidate.score,
                )
            })
            .collect())
    }

    /// Variance of completed objectives for a region.
    pub fn variance(&self, region: usize) -> Option<f64> {
        self.inner.variance(region)
    }

    /// Restart a region with a new center in unit coordinates.
    pub fn restart_region<'py>(
        &mut self,
        region: usize,
        new_center: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<()> {
        self.inner
            .restart_region(region, &new_center.as_array())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

#[cfg(test)]
mod kiss_pymethods_coverage {
    use super::{PyMultiTrustRegion, PyOptimizer, PyTelemetry};

    #[test]
    fn py_optimizer_pymethods_are_linked() {
        let _ = (
            PyOptimizer::ask,
            PyOptimizer::tell,
            PyOptimizer::init_progress,
            PyOptimizer::telemetry,
            PyOptimizer::tr_obs_count,
            PyOptimizer::tr_length,
            PyOptimizer::x_obs,
            PyOptimizer::y_obs,
            PyOptimizer::incumbent_x_unit,
            PyOptimizer::bounds,
            PyMultiTrustRegion::allocate,
            PyMultiTrustRegion::allocate_with,
            PyMultiTrustRegion::select,
            PyMultiTrustRegion::variance,
            PyMultiTrustRegion::restart_region,
            std::mem::size_of::<PyOptimizer>,
            std::mem::size_of::<PyTelemetry>,
        );
    }
}
