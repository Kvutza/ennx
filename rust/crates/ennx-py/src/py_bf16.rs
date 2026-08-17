use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use ennx::experimental::{AcquisitionKind, Bf16Block, Bf16Search, Bf16Trial as CoreTrial};
use ennx::TRLengthConfig;
use pyo3::exceptions::{PyBufferError, PyValueError};
use pyo3::prelude::*;

type PyObject = Py<PyAny>;

fn err(error: String) -> PyErr {
    PyValueError::new_err(error)
}

#[pyclass(name = "Bf16Trial", frozen)]
#[derive(Clone)]
pub struct PyBf16Trial {
    inner: CoreTrial,
}

#[pymethods]
impl PyBf16Trial {
    #[getter]
    fn index(&self) -> usize {
        self.inner.index
    }

    #[getter]
    fn seed(&self) -> u64 {
        self.inner.seed
    }

    #[getter]
    fn score(&self) -> f32 {
        self.inner.score
    }

    #[getter]
    fn length(&self) -> f32 {
        self.inner.length
    }
}

#[pyclass(name = "Bf16Search", unsendable)]
pub struct PyBf16Search {
    inner: Bf16Search,
    exports: Arc<AtomicUsize>,
}

#[pymethods]
impl PyBf16Search {
    #[new]
    #[pyo3(signature=(base,base_value,blocks,capacity,max_pending=1,base_variance=0.0,length_init=0.8,length_min=0.0078125,length_max=1.6))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        base: &Bound<'_, PyAny>,
        base_value: f32,
        blocks: Vec<(u64, usize, usize, f32, f32)>,
        capacity: usize,
        max_pending: usize,
        base_variance: f32,
        length_init: f64,
        length_min: f64,
        length_max: f64,
    ) -> PyResult<Self> {
        let input = crate::dlpack::Input::new(base)?;
        let blocks = bf16_blocks(blocks)?;
        let inner = unsafe {
            Bf16Search::from_device(
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
        Ok(Self {
            inner,
            exports: Arc::new(AtomicUsize::new(0)),
        })
    }

    #[pyo3(signature=(seeds,neighbors,epistemic_scale=0.7,aleatoric_scale=0.05,y_scale=1.0,beta=1.0,acquisition="ucb",seed=0))]
    #[allow(clippy::too_many_arguments)]
    fn ask_batch(
        &mut self,
        seeds: Vec<Vec<u64>>,
        neighbors: usize,
        epistemic_scale: f32,
        aleatoric_scale: f32,
        y_scale: f32,
        beta: f32,
        acquisition: &str,
        seed: u64,
    ) -> PyResult<Vec<PyBf16Trial>> {
        self.ensure_idle()?;
        let arms = seeds.len();
        let candidates = seeds.first().map_or(0, Vec::len);
        if candidates == 0 || seeds.iter().any(|row| row.len() != candidates) {
            return Err(PyValueError::new_err(
                "BF16 seed rows must have equal non-zero length",
            ));
        }
        let seeds = seeds.into_iter().flatten().collect::<Vec<_>>();
        self.inner
            .ask_batch(
                &seeds,
                arms,
                ennx::experimental::SearchConfig {
                    length: 0.0,
                    neighbors,
                    epistemic_scale,
                    aleatoric_scale,
                    y_scale,
                    beta,
                    acquisition: AcquisitionKind::parse(acquisition).map_err(err)?,
                    seed,
                },
            )
            .map_err(err)
            .map(|trials| {
                trials
                    .into_iter()
                    .map(|inner| PyBf16Trial { inner })
                    .collect()
            })
    }

    #[pyo3(signature=(trials,values,variances=None))]
    fn tell_batch(
        &mut self,
        trials: Vec<PyRef<'_, PyBf16Trial>>,
        values: &Bound<'_, PyAny>,
        variances: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Vec<bool>> {
        self.ensure_idle()?;
        let inner = trials.iter().map(|trial| trial.inner).collect::<Vec<_>>();
        if values.hasattr("__dlpack__")? {
            let values = crate::dlpack::Input::f32(values)?;
            if values.len != inner.len() {
                return Err(PyValueError::new_err(
                    "BF16 device rewards must match the trial count",
                ));
            }
            let variances = variances.map(crate::dlpack::Input::f32).transpose()?;
            if variances
                .as_ref()
                .is_some_and(|input| input.len != inner.len())
            {
                return Err(PyValueError::new_err(
                    "BF16 device variances must match the trial count",
                ));
            }
            return unsafe {
                self.inner.tell_device(
                    &inner,
                    values.pointer,
                    variances.as_ref().map(|input| input.pointer),
                )
            }
            .map_err(err);
        }
        let values = values.extract::<Vec<f32>>()?;
        let variances = variances
            .map(|input| input.extract::<Vec<f32>>())
            .transpose()?
            .unwrap_or_else(|| vec![0.0; values.len()]);
        self.inner
            .tell_batch(&inner, &values, &variances)
            .map_err(err)
    }

    fn row(slf: PyRef<'_, Self>, trial: PyRef<'_, PyBf16Trial>) -> PyBf16View {
        let py = slf.py();
        PyBf16View {
            owner: slf.into_py(py),
            trial: trial.inner,
        }
    }

    fn rows(slf: PyRef<'_, Self>, trials: Vec<PyRef<'_, PyBf16Trial>>) -> Vec<PyBf16View> {
        let py = slf.py();
        let owner = slf.into_py(py);
        trials
            .into_iter()
            .map(|trial| PyBf16View {
                owner: owner.clone_ref(py),
                trial: trial.inner,
            })
            .collect()
    }

    fn profile(&mut self, enabled: bool) {
        self.inner.set_profiling(enabled);
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
    fn length(&self) -> f64 {
        self.inner.length()
    }

    #[getter]
    fn best(&self) -> f32 {
        self.inner.best()
    }

    #[getter]
    fn best_variance(&self) -> f32 {
        self.inner.best_variance()
    }

    #[getter]
    fn restarts(&self) -> usize {
        self.inner.restarts()
    }

    #[getter]
    fn history_len(&self) -> usize {
        self.inner.history_len()
    }

    #[getter]
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl PyBf16Search {
    fn ensure_idle(&self) -> PyResult<()> {
        if self.exports.load(Ordering::Acquire) == 0 {
            Ok(())
        } else {
            Err(PyValueError::new_err(
                "release live JAX BF16 rows before mutating the search",
            ))
        }
    }
}

#[pyclass(name = "Bf16View", unsendable)]
pub struct PyBf16View {
    owner: PyObject,
    trial: CoreTrial,
}

#[pymethods]
impl PyBf16View {
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
        if copy == Some(true) {
            return Err(PyBufferError::new_err("Bf16View does not export copies"));
        }
        if dl_device.is_some_and(|device| device != (2, 0)) {
            return Err(PyBufferError::new_err(
                "Bf16View cannot export to another device",
            ));
        }
        if stream.is_some_and(|value| value == 0 || value < -1) {
            return Err(PyValueError::new_err("invalid DLPack CUDA stream"));
        }
        let py = slf.py();
        let (pointer, len, lease) = {
            let search = slf.owner.bind(py).extract::<PyRef<'_, PyBf16Search>>()?;
            let (pointer, _, _) = search.inner.device_row(slf.trial, stream).map_err(err)?;
            let len = search.inner.len();
            let lease = Arc::clone(&search.exports);
            (pointer, len, lease)
        };
        lease.fetch_add(1, Ordering::AcqRel);
        let owner = slf.into_py(py);
        match crate::dlpack::export_count(py, owner, Arc::clone(&lease), pointer, len, max_version)
        {
            Ok(capsule) => Ok(capsule),
            Err(error) => {
                lease.fetch_sub(1, Ordering::AcqRel);
                Err(error)
            }
        }
    }
}

fn bf16_blocks(raw: Vec<(u64, usize, usize, f32, f32)>) -> PyResult<Vec<Bf16Block>> {
    raw.into_iter()
        .map(|(key, offset, len, scale, weight)| {
            Bf16Block::new(key, offset, len, scale, weight).map_err(err)
        })
        .collect()
}
