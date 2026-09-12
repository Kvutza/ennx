use std::cell::Cell;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use ennx::TRLengthConfig;
use ennx::experimental::{AcquisitionKind, ParamBlock, Proposals as CoreProposals, SearchState};
use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::{PyBufferError, PyValueError};
use pyo3::prelude::*;

type PyObject = Py<PyAny>;

fn err(error: String) -> PyErr {
    PyValueError::new_err(error)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Sampler {
    Independent,
    Gaussian,
    Correlated,
}

fn check_sampler(
    sampler: &str,
    max_pending: usize,
    failure_tolerance: Option<usize>,
) -> Result<Sampler, String> {
    let sampler = match sampler {
        "independent" => Sampler::Independent,
        "gaussian" => Sampler::Gaussian,
        "correlated" => Sampler::Correlated,
        _ => return Err("sampler must be 'independent', 'gaussian', or 'correlated'".into()),
    };
    if sampler == Sampler::Correlated && (max_pending != 1 || failure_tolerance.is_some()) {
        return Err("correlated sampling requires max_pending=1 and failure_tolerance=None".into());
    }
    if failure_tolerance.is_some_and(|value| value == 0 || u32::try_from(value).is_err()) {
        return Err("failure_tolerance must be a positive u32".into());
    }
    Ok(sampler)
}

#[pyclass(name = "ParamBlock", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyParamBlock {
    pub(crate) inner: ParamBlock,
}

#[pymethods]
impl PyParamBlock {
    #[new]
    #[pyo3(signature=(key,offset,length,scale,weight=1.0))]
    fn new(key: u64, offset: usize, length: usize, scale: f32, weight: f32) -> PyResult<Self> {
        Ok(Self {
            inner: ParamBlock::new(key, offset, length, scale, weight).map_err(err)?,
        })
    }

    #[getter]
    fn key(&self) -> u64 {
        self.inner.key
    }

    #[getter]
    fn offset(&self) -> usize {
        self.inner.offset
    }

    #[getter]
    fn length(&self) -> usize {
        self.inner.len
    }

    #[getter]
    fn scale(&self) -> f32 {
        self.inner.scale
    }

    #[getter]
    fn weight(&self) -> f32 {
        self.inner.weight
    }
}

/// CUDA BF16 search. Independent sign (default) and Gaussian sampling use TuRBO.
/// Correlated Gaussian sampling requires one arm and four candidates;
/// its accepted-radius controller replaces TuRBO failure/success counters.
#[pyclass(name = "SearchState", unsendable)]
pub struct PySearchState {
    inner: SearchState,
    exports: Arc<AtomicUsize>,
    scratch_generation: u64,
    consumer_sync: Cell<bool>,
}

#[pymethods]
impl PySearchState {
    #[new]
    #[pyo3(signature=(base,base_value,blocks,capacity,max_pending=1,base_variance=0.0,length_init=0.8,length_min=0.0078125,length_max=1.6,failure_tolerance=None,sampler="independent",reference_seed=0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        base: &Bound<'_, PyAny>,
        base_value: f32,
        blocks: Vec<PyRef<'_, PyParamBlock>>,
        capacity: usize,
        max_pending: usize,
        base_variance: f32,
        length_init: f64,
        length_min: f64,
        length_max: f64,
        failure_tolerance: Option<usize>,
        sampler: &str,
        reference_seed: u64,
    ) -> PyResult<Self> {
        let sampler = check_sampler(sampler, max_pending, failure_tolerance).map_err(err)?;
        let input = crate::dlpack::Input::new(base)?;
        let blocks = blocks.iter().map(|block| block.inner).collect();
        let mut inner = unsafe {
            SearchState::from_device(
                input.pointer,
                input.len,
                base_value,
                base_variance,
                blocks,
                capacity,
                max_pending,
                TRLengthConfig::new(length_init, length_min, length_max),
            )
        }
        .map_err(err)?;
        match sampler {
            Sampler::Independent => {}
            Sampler::Gaussian => inner.enable_gaussian().map_err(err)?,
            Sampler::Correlated => inner.enable_correlated(reference_seed).map_err(err)?,
        }
        if let Some(tolerance) = failure_tolerance {
            inner.set_tolerance(tolerance).map_err(err)?;
        }
        Ok(Self {
            inner,
            exports: Arc::new(AtomicUsize::new(0)),
            scratch_generation: 0,
            consumer_sync: Cell::new(false),
        })
    }

    #[pyo3(signature=(arms,candidates,neighbors,seed,epistemic_scale=0.7,aleatoric_scale=0.05,y_scale=1.0,beta=1.0,acquisition="ucb",draw_seed=0))]
    #[allow(clippy::too_many_arguments)]
    fn ask(
        mut slf: PyRefMut<'_, Self>,
        arms: usize,
        candidates: usize,
        neighbors: usize,
        seed: u64,
        epistemic_scale: f32,
        aleatoric_scale: f32,
        y_scale: f32,
        beta: f32,
        acquisition: &str,
        draw_seed: u64,
    ) -> PyResult<PyProposals> {
        slf.ensure_idle()?;
        slf.scratch_generation = slf.scratch_generation.wrapping_add(1);
        let inner = slf
            .inner
            .ask_round(
                arms,
                candidates,
                seed,
                ennx::experimental::SearchConfig {
                    length: 0.0,
                    neighbors,
                    epistemic_scale,
                    aleatoric_scale,
                    y_scale,
                    beta,
                    acquisition: AcquisitionKind::parse(acquisition).map_err(err)?,
                    seed: draw_seed,
                },
            )
            .map_err(err)?;
        let py = slf.py();
        Ok(PyProposals {
            owner: slf.into_pyobject(py).unwrap().into_any().unbind(),
            inner,
        })
    }

    /// Snapshot the incumbent into unused pending storage, with shape (1, N).
    /// The returned exporter is single-use; release the consumer before ask.
    fn incumbent(mut slf: PyRefMut<'_, Self>) -> PyResult<PyIncumbent> {
        slf.ensure_idle()?;
        slf.inner.snapshot_incumbent().map_err(err)?;
        slf.scratch_generation = slf.scratch_generation.wrapping_add(1);
        let generation = slf.scratch_generation;
        let py = slf.py();
        Ok(PyIncumbent {
            owner: slf.into_pyobject(py).unwrap().into_any().unbind(),
            generation,
            exported: false,
        })
    }

    /// Enable zero-anchored paired-relative history before the first round.
    #[pyo3(name = "enable_paired_relative", signature=(failure_tolerance=4))]
    fn enable_relative(&mut self, failure_tolerance: usize) -> PyResult<()> {
        self.ensure_idle()?;
        self.inner.enable_relative(failure_tolerance).map_err(err)
    }

    #[pyo3(name = "tell_paired_relative", signature=(proposals,value,variance,incumbent_value,incumbent_variance,improvement,improvement_variance,accept,*,reject_is_failure=true))]
    #[allow(clippy::too_many_arguments)]
    fn tell_relative(
        &mut self,
        proposals: PyRef<'_, PyProposals>,
        value: f64,
        variance: f64,
        incumbent_value: f64,
        incumbent_variance: f64,
        improvement: f64,
        improvement_variance: f64,
        accept: bool,
        reject_is_failure: bool,
    ) -> PyResult<()> {
        self.ensure_idle()?;
        let [
            value,
            variance,
            incumbent_value,
            incumbent_variance,
            improvement,
            improvement_variance,
        ] = check_scores(
            value,
            variance,
            incumbent_value,
            incumbent_variance,
            improvement,
            improvement_variance,
        )
        .map_err(err)?;
        self.inner
            .tell_relative(
                &proposals.inner,
                value,
                variance,
                incumbent_value,
                incumbent_variance,
                improvement,
                improvement_variance,
                accept,
                reject_is_failure,
            )
            .map_err(err)
    }

    /// Queue an explicit paired decision; collect its result with sync().
    #[pyo3(signature=(proposals,value,variance,incumbent_value,incumbent_variance,accept))]
    #[allow(clippy::too_many_arguments)]
    fn tell_paired(
        &mut self,
        proposals: PyRef<'_, PyProposals>,
        value: f64,
        variance: f64,
        incumbent_value: f64,
        incumbent_variance: f64,
        accept: bool,
    ) -> PyResult<()> {
        self.ensure_idle()?;
        let [value, variance, incumbent_value, incumbent_variance] =
            check_pair(value, variance, incumbent_value, incumbent_variance).map_err(err)?;
        self.inner
            .tell_paired(
                &proposals.inner,
                value,
                variance,
                incumbent_value,
                incumbent_variance,
                accept,
            )
            .map_err(err)
    }

    #[pyo3(signature=(proposals,values,variances=None))]
    fn tell(
        &mut self,
        proposals: PyRef<'_, PyProposals>,
        values: &Bound<'_, PyAny>,
        variances: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        self.ensure_idle()?;
        if values.hasattr("__dlpack__")? {
            let values = crate::dlpack::Input::f32(values)?;
            if values.len != proposals.inner.arms() {
                return Err(PyValueError::new_err(
                    "device rewards must match the proposal count",
                ));
            }
            let variances = variances.map(crate::dlpack::Input::f32).transpose()?;
            if variances
                .as_ref()
                .is_some_and(|input| input.len != proposals.inner.arms())
            {
                return Err(PyValueError::new_err(
                    "device variances must match the proposal count",
                ));
            }
            unsafe {
                self.inner.finish_round(
                    &proposals.inner,
                    values.pointer,
                    variances.as_ref().map(|input| input.pointer),
                )
            }
            .map_err(err)?;
            return Ok(());
        }
        let values = values.extract::<Vec<f32>>()?;
        let variances = variances
            .map(|input| input.extract::<Vec<f32>>())
            .transpose()?
            .unwrap_or_else(|| vec![0.0; values.len()]);
        self.inner
            .queue_round(&proposals.inner, &values, &variances)
            .map_err(err)?;
        Ok(())
    }

    fn sync(&mut self) -> PyResult<Vec<bool>> {
        self.ensure_idle()?;
        self.inner.sync().map_err(err)
    }

    fn profile(&mut self, enabled: bool) -> PyResult<()> {
        self.ensure_idle()?;
        self.inner.set_profiling(enabled);
        Ok(())
    }

    /// Copy the accepted model to host memory as raw BF16 bits.
    fn read_best<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<u16>>> {
        self.ensure_idle()?;
        Ok(self.inner.read_best().map_err(err)?.into_pyarray(py))
    }

    /// Copy the correlated reference to host as raw BF16 bits (a model-sized copy).
    /// Other sampler modes return an error. Intended for explicit validation only.
    fn read_reference<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<u16>>> {
        self.ensure_idle()?;
        Ok(self.inner.read_reference().map_err(err)?.into_pyarray(py))
    }

    #[getter]
    fn last_profile(&self) -> Option<(f32, f32, f32, f32)> {
        self.inner.last_profile().map(|profile| {
            (
                profile.score_ms,
                profile.pick_ms,
                profile.materialize_ms,
                profile.total_ms,
            )
        })
    }

    #[getter]
    fn length(&mut self) -> PyResult<f64> {
        self.ensure_idle()?;
        self.inner.length().map_err(err)
    }

    #[getter]
    fn best(&mut self) -> PyResult<f32> {
        self.ensure_idle()?;
        self.inner.best().map_err(err)
    }

    #[getter]
    fn best_variance(&mut self) -> PyResult<f32> {
        self.ensure_idle()?;
        self.inner.best_variance().map_err(err)
    }

    #[getter]
    fn restarts(&mut self) -> PyResult<usize> {
        self.ensure_idle()?;
        self.inner.restarts().map_err(err)
    }

    #[getter]
    fn history_len(&mut self) -> PyResult<usize> {
        self.ensure_idle()?;
        self.inner.history_len().map_err(err)
    }

    #[getter]
    fn len(&self) -> usize {
        self.inner.len()
    }
}

#[pyclass(name = "Incumbent", unsendable)]
pub struct PyIncumbent {
    owner: PyObject,
    generation: u64,
    exported: bool,
}

#[pymethods]
impl PyIncumbent {
    fn __dlpack_device__(&self) -> (i32, i32) {
        (2, 0)
    }

    #[pyo3(signature=(stream=None,max_version=None,dl_device=None,copy=None))]
    fn __dlpack__(
        mut slf: PyRefMut<'_, Self>,
        stream: Option<i64>,
        max_version: Option<(u32, u32)>,
        dl_device: Option<(i32, i32)>,
        copy: Option<bool>,
    ) -> PyResult<PyObject> {
        check_export(stream, dl_device, copy)?;
        let py = slf.py();
        let (pointer, rows, columns, lease) = {
            let search = slf.owner.bind(py).extract::<PyRef<'_, PySearchState>>()?;
            search.ensure_idle()?;
            check_handle(slf.exported, slf.generation, search.scratch_generation).map_err(err)?;
            let (pointer, rows, columns) = search.inner.device_incumbent(stream).map_err(err)?;
            search.consumer_sync.set(true);
            (pointer, rows, columns, Arc::clone(&search.exports))
        };
        // Even a released legacy consumer could have written to scratch. Never
        // export the same snapshot twice; a fresh incumbent() copies row zero again.
        slf.exported = true;
        let owner = slf.into_pyobject(py).unwrap().into_any().unbind();
        export_batch(py, owner, lease, pointer, rows, columns, max_version)
    }
}

fn check_handle(exported: bool, generation: u64, current: u64) -> Result<(), String> {
    if exported || generation != current {
        return Err("Stale or already exported incumbent; request a fresh incumbent()".into());
    }
    Ok(())
}

fn check_pair(
    value: f64,
    variance: f64,
    incumbent_value: f64,
    incumbent_variance: f64,
) -> Result<[f32; 4], String> {
    let inputs = [value, variance, incumbent_value, incumbent_variance];
    if variance < 0.0
        || incumbent_variance < 0.0
        || inputs
            .iter()
            .any(|value| !value.is_finite() || !(*value as f32).is_finite())
    {
        return Err(
            "Paired rewards must be finite FP32 values and both variances finite and nonnegative"
                .into(),
        );
    }
    Ok(inputs.map(|value| value as f32))
}

fn check_scores(
    value: f64,
    variance: f64,
    incumbent_value: f64,
    incumbent_variance: f64,
    improvement: f64,
    improvement_variance: f64,
) -> Result<[f32; 6], String> {
    let [value, variance, incumbent_value, incumbent_variance] =
        check_pair(value, variance, incumbent_value, incumbent_variance)?;
    if improvement_variance < 0.0
        || [improvement, improvement_variance]
            .iter()
            .any(|value| !value.is_finite() || !(*value as f32).is_finite())
    {
        return Err(
            "Paired improvement must be finite FP32 and its variance finite and nonnegative".into(),
        );
    }
    Ok([
        value,
        variance,
        incumbent_value,
        incumbent_variance,
        improvement as f32,
        improvement_variance as f32,
    ])
}

fn check_export(
    stream: Option<i64>,
    dl_device: Option<(i32, i32)>,
    copy: Option<bool>,
) -> PyResult<()> {
    if copy == Some(true) {
        return Err(PyBufferError::new_err("BF16 search does not export copies"));
    }
    if dl_device.is_some_and(|device| device != (2, 0)) {
        return Err(PyBufferError::new_err(
            "BF16 search cannot export to another device",
        ));
    }
    if stream.is_some_and(|value| value == 0 || value < -1) {
        return Err(PyValueError::new_err("invalid DLPack CUDA stream"));
    }
    Ok(())
}

fn export_batch(
    py: Python<'_>,
    owner: PyObject,
    lease: Arc<AtomicUsize>,
    pointer: u64,
    rows: usize,
    columns: usize,
    max_version: Option<(u32, u32)>,
) -> PyResult<PyObject> {
    lease.fetch_add(1, Ordering::AcqRel);
    let result = crate::dlpack::export_batch(
        py,
        owner,
        Arc::clone(&lease),
        pointer,
        rows,
        columns,
        max_version,
    );
    if result.is_err() {
        // Capsule construction failure already runs its deleter, whereas shape
        // validation failure does not. Release only a lease still held here.
        let _ = lease.compare_exchange(1, 0, Ordering::AcqRel, Ordering::Acquire);
    }
    result
}

#[pyclass(name = "Proposals", unsendable)]
pub struct PyProposals {
    owner: PyObject,
    inner: CoreProposals,
}

#[pymethods]
impl PyProposals {
    /// Selected candidate index and Gaussian persistence per arm.
    /// Persistence is 0.75 for correlated indices 0/1 and 0 otherwise.
    fn geometry(&self, py: Python<'_>) -> PyResult<Vec<(usize, f32)>> {
        let search = self.owner.bind(py).extract::<PyRef<'_, PySearchState>>()?;
        search.inner.geometry(&self.inner).map_err(err)
    }

    /// Seed, acquisition score, radius, and (changed count, squared L2) per block.
    fn describe(&self, py: Python<'_>) -> PyResult<Vec<(u64, f32, f32, Vec<(u64, f64)>)>> {
        let search = self.owner.bind(py).extract::<PyRef<'_, PySearchState>>()?;
        search.inner.describe(&self.inner).map_err(err)
    }

    #[getter]
    fn arms(&self) -> usize {
        self.inner.arms()
    }

    fn __dlpack_device__(&self) -> (i32, i32) {
        (2, 0)
    }

    #[pyo3(signature=(stream=None,max_version=None,dl_device=None,copy=None))]
    fn __dlpack__(
        slf: PyRef<'_, Self>,
        stream: Option<i64>,
        max_version: Option<(u32, u32)>,
        dl_device: Option<(i32, i32)>,
        copy: Option<bool>,
    ) -> PyResult<PyObject> {
        check_export(stream, dl_device, copy)?;
        let py = slf.py();
        let (pointer, rows, columns, lease) = {
            let mut search = slf
                .owner
                .bind(py)
                .extract::<PyRefMut<'_, PySearchState>>()?;
            search.ensure_idle()?;
            let (pointer, rows, columns) =
                search.inner.device_round(&slf.inner, stream).map_err(err)?;
            search.consumer_sync.set(true);
            (pointer, rows, columns, Arc::clone(&search.exports))
        };
        let owner = slf.into_pyobject(py).unwrap().into_any().unbind();
        export_batch(py, owner, lease, pointer, rows, columns, max_version)
    }
}

impl PySearchState {
    fn ensure_idle(&self) -> PyResult<()> {
        if self.exports.load(Ordering::Acquire) == 0 {
            if self.consumer_sync.get() {
                self.inner.sync_consumers().map_err(err)?;
                self.consumer_sync.set(false);
            }
            Ok(())
        } else {
            Err(PyValueError::new_err(
                "release live JAX proposals or incumbent exports before mutating the search",
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Sampler, check_handle, check_pair, check_sampler, check_scores};

    #[test]
    fn relative_scores() {
        let inputs = [-1.0, 0.25, -2.0, 0.5, 0.125, 0.0625];
        let check = |x: [f64; 6]| check_scores(x[0], x[1], x[2], x[3], x[4], x[5]);
        assert_eq!(check(inputs).unwrap(), inputs.map(|value| value as f32));
        for index in 0..6 {
            for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1e100, -1e100] {
                let mut bad = inputs;
                bad[index] = invalid;
                assert!(check(bad).is_err(), "index {index}, value {invalid}");
            }
        }
        for index in [1, 3, 5] {
            for invalid in [-1.0, -1e-100] {
                let mut bad = inputs;
                bad[index] = invalid;
                assert!(check(bad).is_err());
            }
        }
        // Improvement is supplied directly, not reconstructed from rounded absolute scores.
        assert_eq!(
            check_scores(1e8, 0.0, 1e8, 0.0, -1e-6, 0.0).unwrap()[4],
            -1e-6_f32
        );
    }

    #[test]
    fn pair_scores() {
        assert!(check_pair(-1.0, 0.0, -2.0, 0.5).is_ok());
        for variance in [-1e-100, -1.0, f64::NAN, f64::INFINITY, 1e100] {
            assert!(check_pair(0.0, variance, 0.0, 0.0).is_err());
            assert!(check_pair(0.0, 0.0, 0.0, variance).is_err());
        }
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1e100] {
            assert!(check_pair(value, 0.0, 0.0, 0.0).is_err());
            assert!(check_pair(0.0, 0.0, value, 0.0).is_err());
        }
    }

    #[test]
    fn handle_guards() {
        assert!(check_handle(false, 1, 1).is_ok());
        assert!(check_handle(true, 1, 1).is_err());
        assert!(check_handle(false, 1, 2).is_err());
        assert!(check_handle(true, 1, 2).is_err());
    }

    #[test]
    fn config_guard() {
        assert_eq!(
            check_sampler("independent", 1, None),
            Ok(Sampler::Independent)
        );
        assert_eq!(
            check_sampler("independent", 8, Some(4)),
            Ok(Sampler::Independent)
        );
        assert_eq!(check_sampler("gaussian", 1, None), Ok(Sampler::Gaussian));
        assert_eq!(check_sampler("gaussian", 8, Some(4)), Ok(Sampler::Gaussian));
        assert_eq!(
            check_sampler("correlated", 1, None),
            Ok(Sampler::Correlated)
        );
        assert!(check_sampler("unknown", 1, None).is_err());
        for pending in [0, 2, 32] {
            assert!(check_sampler("correlated", pending, None).is_err());
        }
        for tolerance in [0, 1, 4, usize::MAX] {
            assert!(check_sampler("correlated", 1, Some(tolerance)).is_err());
        }
        for sampler in ["independent", "gaussian"] {
            assert!(check_sampler(sampler, 1, Some(0)).is_err());
            assert!(check_sampler(sampler, 1, Some(u32::MAX as usize)).is_ok());
            if usize::BITS > 32 {
                assert!(check_sampler(sampler, 1, Some(usize::MAX)).is_err());
            }
        }
    }
}
