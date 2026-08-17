use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use ennx::experimental::{AcquisitionKind, ParamBlock, Proposals as CoreProposals, SearchState};
use ennx::TRLengthConfig;
use pyo3::exceptions::{PyBufferError, PyValueError};
use pyo3::prelude::*;

type PyObject = Py<PyAny>;

fn err(error: String) -> PyErr {
    PyValueError::new_err(error)
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

#[pyclass(name = "SearchState", unsendable)]
pub struct PySearchState {
    inner: SearchState,
    exports: Arc<AtomicUsize>,
}

#[pymethods]
impl PySearchState {
    #[new]
    #[pyo3(signature=(base,base_value,blocks,capacity,max_pending=1,base_variance=0.0,length_init=0.8,length_min=0.0078125,length_max=1.6))]
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
    ) -> PyResult<Self> {
        let input = crate::dlpack::Input::new(base)?;
        let blocks = blocks.iter().map(|block| block.inner).collect();
        let inner = unsafe {
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
        Ok(Self {
            inner,
            exports: Arc::new(AtomicUsize::new(0)),
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
        self.inner.sync().map_err(err)
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
    fn length(&mut self) -> PyResult<f64> {
        self.inner.length().map_err(err)
    }

    #[getter]
    fn best(&mut self) -> PyResult<f32> {
        self.inner.best().map_err(err)
    }

    #[getter]
    fn best_variance(&mut self) -> PyResult<f32> {
        self.inner.best_variance().map_err(err)
    }

    #[getter]
    fn restarts(&mut self) -> PyResult<usize> {
        self.inner.restarts().map_err(err)
    }

    #[getter]
    fn history_len(&mut self) -> PyResult<usize> {
        self.inner.history_len().map_err(err)
    }

    #[getter]
    fn len(&self) -> usize {
        self.inner.len()
    }
}

#[pyclass(name = "Proposals", unsendable)]
pub struct PyProposals {
    owner: PyObject,
    inner: CoreProposals,
}

#[pymethods]
impl PyProposals {
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
        if copy == Some(true) {
            return Err(PyBufferError::new_err("Proposals does not export copies"));
        }
        if dl_device.is_some_and(|device| device != (2, 0)) {
            return Err(PyBufferError::new_err(
                "Proposals cannot export to another device",
            ));
        }
        if stream.is_some_and(|value| value == 0 || value < -1) {
            return Err(PyValueError::new_err("invalid DLPack CUDA stream"));
        }
        let py = slf.py();
        let (pointer, rows, columns, lease) = {
            let mut search = slf
                .owner
                .bind(py)
                .extract::<PyRefMut<'_, PySearchState>>()?;
            search.ensure_idle()?;
            let (pointer, rows, columns) =
                search.inner.device_round(&slf.inner, stream).map_err(err)?;
            (pointer, rows, columns, Arc::clone(&search.exports))
        };
        lease.fetch_add(1, Ordering::AcqRel);
        let owner = slf.into_pyobject(py).unwrap().into_any().unbind();
        match crate::dlpack::export_batch(
            py,
            owner,
            Arc::clone(&lease),
            pointer,
            rows,
            columns,
            max_version,
        ) {
            Ok(capsule) => Ok(capsule),
            Err(error) => {
                lease.fetch_sub(1, Ordering::AcqRel);
                Err(error)
            }
        }
    }
}

impl PySearchState {
    fn ensure_idle(&self) -> PyResult<()> {
        if self.exports.load(Ordering::Acquire) == 0 {
            Ok(())
        } else {
            Err(PyValueError::new_err(
                "release live JAX proposals before mutating the search",
            ))
        }
    }
}
