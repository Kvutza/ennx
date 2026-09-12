//! Synchronous resident Metal buffers; no DLPack or framework-owned GPU tensors.

use std::cell::Cell;
use std::rc::Rc;

use ennx::TRLengthConfig;
use ennx::experimental::{
    AcquisitionKind, MetalFlameEvaluator, MetalParamBlock, MetalProposals, MetalSearchState,
    SearchConfig,
};
use metal::Buffer;
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::{PyBufferError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict};

use crate::py_flameconfig::configuration;
use crate::py_qwen::PyMetalQwenEvaluator;

fn err(message: String) -> PyErr {
    PyValueError::new_err(message)
}

#[pyclass(name = "MetalWeights", module = "ennx.experimental", unsendable)]
pub struct PyMetalWeights {
    buffer: Buffer,
    len: usize,
    lease: Option<Rc<Cell<usize>>>,
}

impl PyMetalWeights {
    pub(crate) fn owned(buffer: Buffer, len: usize) -> Self {
        Self {
            buffer,
            len,
            lease: None,
        }
    }
}

impl Drop for PyMetalWeights {
    fn drop(&mut self) {
        if let Some(lease) = &self.lease {
            lease.set(lease.get() - 1);
        }
    }
}

#[pymethods]
impl PyMetalWeights {
    #[new]
    fn new(bits: PyReadonlyArray1<'_, u16>) -> PyResult<Self> {
        let values = bits.as_slice()?;
        Ok(Self {
            buffer: ennx::experimental::metal_flame_upload(values).map_err(err)?,
            len: values.len(),
            lease: None,
        })
    }

    #[staticmethod]
    fn device_info(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
        let device = ennx::experimental::metal_flame_memory_info().map_err(err)?;
        if !device.has_unified_memory {
            return Err(PyValueError::new_err("Metal FLAME requires unified memory"));
        }
        let info = PyDict::new(py);
        info.set_item("name", device.name)?;
        info.set_item(
            "recommended_working_set_bytes",
            device.recommended_max_working_set_size,
        )?;
        info.set_item("allocated_bytes", device.current_allocated_size)?;
        info.set_item("max_buffer_bytes", device.max_buffer_length)?;
        Ok(info)
    }

    #[getter]
    fn size(&self) -> usize {
        self.len
    }

    fn read<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u16>> {
        // All producers synchronize before publishing a view, and leases prevent mutation.
        let values =
            unsafe { std::slice::from_raw_parts(self.buffer.contents().cast::<u16>(), self.len) };
        values.to_vec().into_pyarray(py)
    }
}

#[pyclass(name = "MetalFlameEvaluator", module = "ennx.experimental", unsendable)]
pub struct PyMetalFlameEvaluator {
    inner: MetalFlameEvaluator,
}

#[pymethods]
impl PyMetalFlameEvaluator {
    #[new]
    #[pyo3(signature=(config,max_tokens))]
    fn new(config: &Bound<'_, PyDict>, max_tokens: &Bound<'_, PyAny>) -> PyResult<Self> {
        if max_tokens.is_instance_of::<PyBool>() {
            return Err(PyTypeError::new_err(
                "max_tokens must be an integer, not bool",
            ));
        }
        Ok(Self {
            inner: MetalFlameEvaluator::new(configuration(config)?, max_tokens.extract()?)
                .map_err(err)?,
        })
    }

    #[getter]
    fn weights_len(&self) -> usize {
        self.inner.weights_len()
    }

    #[getter]
    fn workspace_bytes(&self) -> u64 {
        self.inner.workspace_bytes()
    }

    fn logits<'py>(
        &mut self,
        py: Python<'py>,
        weights: &Bound<'py, PyAny>,
        tokens: Vec<i32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        self.inner.check_tokens(&tokens).map_err(err)?;
        let buffer = weight_buffer(weights)?;
        let logits = self.inner.logits(&buffer, &tokens).map_err(err)?;
        let array = Array2::from_shape_vec((tokens.len(), self.inner.vocab()), logits)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(array.into_pyarray(py))
    }

    fn next_logits<'py>(
        &mut self,
        py: Python<'py>,
        weights: &Bound<'py, PyAny>,
        tokens: Vec<i32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        self.inner.check_tokens(&tokens).map_err(err)?;
        let buffer = weight_buffer(weights)?;
        Ok(self
            .inner
            .next_logits(&buffer, &tokens)
            .map_err(err)?
            .into_pyarray(py))
    }

    fn losses<'py>(
        &mut self,
        py: Python<'py>,
        weights: &Bound<'py, PyAny>,
        tokens: Vec<Vec<i32>>,
        masks: Vec<Vec<bool>>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        self.inner.check_batch(&tokens, &masks).map_err(err)?;
        let buffer = weight_buffer(weights)?;
        Ok(self
            .inner
            .losses(&buffer, &tokens, &masks)
            .map_err(err)?
            .into_pyarray(py))
    }
}

pub(crate) fn weight_buffer(weights: &Bound<'_, PyAny>) -> PyResult<Buffer> {
    if let Ok(view) = weights.extract::<PyRef<'_, PyMetalWeights>>() {
        return Ok(view.buffer.clone());
    }
    if let Ok(proposal) = weights.extract::<PyRef<'_, PyMetalProposals>>() {
        let owner = proposal.owner.bind(weights.py()).try_borrow()?;
        return owner.inner.propose_buffer(&proposal.inner).map_err(err);
    }
    Err(PyTypeError::new_err(
        "Expected resident MetalWeights or a live Metal proposal",
    ))
}

#[pyclass(name = "MetalParamBlock", module = "ennx.experimental", frozen)]
pub struct PyMetalParamBlock {
    inner: MetalParamBlock,
}

#[pymethods]
impl PyMetalParamBlock {
    #[new]
    #[pyo3(signature=(key,offset,length,scale,weight=1.0))]
    fn new(key: u64, offset: usize, length: usize, scale: f32, weight: f32) -> PyResult<Self> {
        Ok(Self {
            inner: MetalParamBlock::new(key, offset, length, scale, weight).map_err(err)?,
        })
    }
}

#[pyclass(name = "MetalSearchState", module = "ennx.experimental", unsendable)]
pub struct PyMetalSearchState {
    inner: MetalSearchState,
    leases: Rc<Cell<usize>>,
}

impl PyMetalSearchState {
    fn unleased(&self) -> PyResult<()> {
        if self.leases.get() != 0 {
            return Err(PyBufferError::new_err(
                "Release incumbent Metal weight views before mutating search",
            ));
        }
        Ok(())
    }
}

#[pymethods]
impl PyMetalSearchState {
    #[new]
    #[pyo3(signature=(base,base_value,blocks,capacity,max_pending=1,base_variance=0.0,length_init=0.01,length_min=0.0001,length_max=0.08,failure_tolerance=None,sampler="correlated",reference_seed=0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        base: PyRef<'_, PyMetalWeights>,
        base_value: f32,
        blocks: Vec<PyRef<'_, PyMetalParamBlock>>,
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
        if sampler != "correlated" || failure_tolerance.is_some() {
            return Err(PyValueError::new_err(
                "Metal search requires sampler='correlated' and failure_tolerance=None; the shared TuRBO controller derives its tolerance from dimension",
            ));
        }
        let values =
            unsafe { std::slice::from_raw_parts(base.buffer.contents().cast::<u16>(), base.len) };
        let mut inner = MetalSearchState::new(
            values,
            base_value,
            base_variance,
            blocks.iter().map(|b| b.inner).collect(),
            capacity,
            max_pending,
            TRLengthConfig::new(length_init, length_min, length_max),
        )
        .map_err(err)?;
        inner.correlate(reference_seed).map_err(err)?;
        Ok(Self {
            inner,
            leases: Rc::new(Cell::new(0)),
        })
    }

    #[pyo3(name = "enable_paired_relative", signature=(failure_tolerance=4))]
    fn enable_relative(&mut self, failure_tolerance: usize) -> PyResult<()> {
        self.unleased()?;
        self.inner.enable_relative(failure_tolerance).map_err(err)
    }

    #[pyo3(signature=(arms,candidates,neighbors,seed,epistemic_scale=1.0,aleatoric_scale=0.0,y_scale=1.0,beta=1.0,acquisition="thompson",draw_seed=0))]
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
    ) -> PyResult<PyMetalProposals> {
        slf.unleased()?;
        let inner = slf
            .inner
            .ask_round(
                arms,
                candidates,
                seed,
                SearchConfig {
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
        Ok(PyMetalProposals {
            owner: slf.into(),
            inner,
        })
    }

    #[pyo3(signature=(evaluator,prompts,max_new_tokens,arms=1,candidates=4,neighbors=2,seed=0,draw_seed=0))]
    #[allow(clippy::too_many_arguments)]
    fn ask_generate(
        mut slf: PyRefMut<'_, Self>,
        mut evaluator: PyRefMut<'_, PyMetalQwenEvaluator>,
        prompts: Vec<Vec<i32>>,
        max_new_tokens: usize,
        arms: usize,
        candidates: usize,
        neighbors: usize,
        seed: u64,
        draw_seed: u64,
    ) -> PyResult<(PyMetalProposals, Vec<Vec<i32>>)> {
        slf.unleased()?;
        let buffer = slf
            .inner
            .begin_ask(
                arms,
                candidates,
                seed,
                ennx::experimental::SearchConfig {
                    length: 0.0,
                    neighbors,
                    epistemic_scale: 1.0,
                    aleatoric_scale: 0.0,
                    y_scale: 1.0,
                    beta: 1.0,
                    acquisition: AcquisitionKind::Thompson,
                    seed: draw_seed,
                },
            )
            .map_err(err)?;
        let generated = match evaluator
            .inner
            .generate_batch(&buffer, &prompts, max_new_tokens)
        {
            Ok(generated) => generated,
            Err(error) => {
                slf.inner.abort_ask();
                return Err(err(error));
            }
        };
        let inner = slf.inner.finish_ask().map_err(err)?;
        Ok((
            PyMetalProposals {
                owner: slf.into(),
                inner,
            },
            generated,
        ))
    }

    #[pyo3(signature=(evaluator,tokens,masks,arms=1,candidates=4,neighbors=2,seed=0,draw_seed=0))]
    #[allow(clippy::too_many_arguments)]
    fn ask_losses(
        mut slf: PyRefMut<'_, Self>,
        mut evaluator: PyRefMut<'_, PyMetalQwenEvaluator>,
        tokens: Vec<Vec<i32>>,
        masks: Vec<Vec<bool>>,
        arms: usize,
        candidates: usize,
        neighbors: usize,
        seed: u64,
        draw_seed: u64,
    ) -> PyResult<(PyMetalProposals, Vec<f32>)> {
        slf.unleased()?;
        let buffer = slf
            .inner
            .begin_ask(
                arms,
                candidates,
                seed,
                ennx::experimental::SearchConfig {
                    length: 0.0,
                    neighbors,
                    epistemic_scale: 1.0,
                    aleatoric_scale: 0.0,
                    y_scale: 1.0,
                    beta: 1.0,
                    acquisition: AcquisitionKind::Thompson,
                    seed: draw_seed,
                },
            )
            .map_err(err)?;
        let losses = match evaluator.inner.losses(&buffer, &tokens, &masks) {
            Ok(losses) => losses,
            Err(error) => {
                slf.inner.abort_ask();
                return Err(err(error));
            }
        };
        let inner = slf.inner.finish_ask().map_err(err)?;
        Ok((
            PyMetalProposals {
                owner: slf.into(),
                inner,
            },
            losses,
        ))
    }

    fn incumbent(&self) -> PyResult<PyMetalWeights> {
        // The returned buffer is read-only in Python and retains its allocation.
        self.inner.best().map_err(err)?;
        self.leases.set(self.leases.get() + 1);
        Ok(PyMetalWeights {
            buffer: self.inner.base_buffer(),
            len: self.inner.len(),
            lease: Some(self.leases.clone()),
        })
    }

    #[pyo3(signature=(proposals, value, variance, incumbent_value, incumbent_variance, accept))]
    fn tell_paired(
        &mut self,
        proposals: PyRef<'_, PyMetalProposals>,
        value: f64,
        variance: f64,
        incumbent_value: f64,
        incumbent_variance: f64,
        accept: bool,
    ) -> PyResult<()> {
        self.unleased()?;
        if [value, variance, incumbent_value, incumbent_variance]
            .iter()
            .any(|v| !v.is_finite() || !(*v as f32).is_finite())
            || variance < 0.0
            || incumbent_variance < 0.0
        {
            return Err(PyValueError::new_err(
                "Paired values must be finite FP32; variances must be nonnegative",
            ));
        }
        self.inner
            .tell_paired(
                &proposals.inner,
                value as f32,
                variance as f32,
                incumbent_value as f32,
                incumbent_variance as f32,
                accept,
            )
            .map_err(err)
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(name = "tell_paired_relative", signature=(proposals, value, variance, incumbent_value, incumbent_variance, improvement, improvement_variance, accept, *, reject_is_failure=true))]
    fn tell_relative(
        &mut self,
        proposals: PyRef<'_, PyMetalProposals>,
        value: f64,
        variance: f64,
        incumbent_value: f64,
        incumbent_variance: f64,
        improvement: f64,
        improvement_variance: f64,
        accept: bool,
        reject_is_failure: bool,
    ) -> PyResult<()> {
        self.unleased()?;
        let values = [
            value,
            variance,
            incumbent_value,
            incumbent_variance,
            improvement,
            improvement_variance,
        ];
        if values
            .iter()
            .any(|v| !v.is_finite() || !(*v as f32).is_finite())
            || [variance, incumbent_variance, improvement_variance]
                .iter()
                .any(|v| *v < 0.0)
        {
            return Err(PyValueError::new_err(
                "Paired values must be finite FP32; variances must be nonnegative",
            ));
        }
        self.inner
            .tell_relative(
                &proposals.inner,
                value as f32,
                variance as f32,
                incumbent_value as f32,
                incumbent_variance as f32,
                improvement as f32,
                improvement_variance as f32,
                accept,
                reject_is_failure,
            )
            .map_err(err)
    }

    fn sync(&mut self) -> PyResult<Vec<bool>> {
        self.unleased()?;
        self.inner.sync().map_err(err)
    }

    fn set_profiling(&mut self, enabled: bool) -> PyResult<()> {
        self.unleased()?;
        self.inner.set_profiling(enabled);
        Ok(())
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

    fn memory_info<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let memory = self.inner.memory_info();
        let info = PyDict::new(py);
        info.set_item("row_bytes", memory.row_bytes)?;
        info.set_item("search_resident_bytes", memory.resident_bytes)?;
        info.set_item("device_allocated_bytes", memory.current_allocated_size)?;
        info.set_item(
            "recommended_working_set_bytes",
            memory.recommended_max_working_set_size,
        )?;
        info.set_item("max_buffer_bytes", memory.max_buffer_length)?;
        Ok(info)
    }

    fn controller_info<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let controller = self.inner.controller_info().map_err(err)?;
        let info = PyDict::new(py);
        info.set_item("dimensions", controller.dimensions)?;
        info.set_item("evaluated_arms", controller.evaluated_arms)?;
        info.set_item("length", controller.length)?;
        info.set_item("length_min", controller.length_min)?;
        info.set_item("length_max", controller.length_max)?;
        info.set_item("success_tolerance", controller.success_tolerance)?;
        info.set_item("failure_tolerance", controller.failure_tolerance)?;
        info.set_item("success_counter", controller.success_counter)?;
        info.set_item("failure_counter", controller.failure_counter)?;
        info.set_item("restarts", controller.restarts)?;
        if let Some(reliability) = self.inner.reliability_info().map_err(err)? {
            info.set_item("reliability_action", format!("{:?}", reliability.action))?;
            info.set_item("rank_concordance", reliability.concordance)?;
            info.set_item("rank_coverage", reliability.coverage)?;
            info.set_item("reliability_mean", reliability.reliability_mean)?;
            info.set_item("reliability_lower", reliability.reliability_lower)?;
            info.set_item("reliability_evidence", reliability.evidence)?;
            info.set_item("progress", reliability.progress)?;
            info.set_item("center_radius", reliability.center_radius)?;
            info.set_item("normalized_step", reliability.normalized_step)?;
            info.set_item("radius_conversion", reliability.conversion)?;
            info.set_item("escape_remaining", reliability.escape_remaining)?;
            info.set_item("escapes", reliability.escapes)?;
        }
        Ok(info)
    }

    fn read_best<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<u16>>> {
        Ok(self.inner.read_best().map_err(err)?.into_pyarray(py))
    }

    fn read_reference<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<u16>>> {
        self.unleased()?;
        Ok(self.inner.read_reference().map_err(err)?.into_pyarray(py))
    }

    #[getter]
    fn length(&self) -> PyResult<f64> {
        self.inner.length().map_err(err)
    }
    #[getter]
    fn best(&self) -> PyResult<f32> {
        self.inner.best().map_err(err)
    }
    #[getter]
    fn best_variance(&self) -> PyResult<f32> {
        self.inner.best_variance().map_err(err)
    }
    #[getter]
    fn history_len(&self) -> PyResult<usize> {
        self.inner.history_len().map_err(err)
    }
    #[getter]
    fn restarts(&self) -> PyResult<usize> {
        self.inner.restarts().map_err(err)
    }
}

#[pyclass(name = "MetalProposals", module = "ennx.experimental", unsendable)]
pub struct PyMetalProposals {
    owner: Py<PyMetalSearchState>,
    inner: MetalProposals,
}

#[pymethods]
impl PyMetalProposals {
    fn describe(&self, py: Python<'_>) -> PyResult<Vec<(u64, f32, f32, Vec<(u64, f64)>)>> {
        self.owner
            .bind(py)
            .try_borrow()?
            .inner
            .describe(&self.inner)
            .map_err(err)
    }

    fn geometry(&self, py: Python<'_>) -> PyResult<Vec<(usize, f32)>> {
        self.owner
            .bind(py)
            .try_borrow()?
            .inner
            .geometry(&self.inner)
            .map_err(err)
    }

    fn base_id(&self, py: Python<'_>) -> PyResult<i64> {
        self.owner
            .bind(py)
            .try_borrow()?
            .inner
            .base_id(&self.inner)
            .map_err(err)
    }

    fn history_dists(&self, py: Python<'_>) -> PyResult<Vec<(i64, f32)>> {
        self.owner
            .bind(py)
            .try_borrow()?
            .inner
            .history_dists(&self.inner)
            .map_err(err)
    }

    fn pool(&self, py: Python<'_>) -> PyResult<Vec<(usize, u64, f32, f32, Vec<(i64, f32)>)>> {
        self.owner
            .bind(py)
            .try_borrow()?
            .inner
            .pool(&self.inner)
            .map_err(err)
    }

    fn pool_geometry(
        &self,
        py: Python<'_>,
    ) -> PyResult<(Vec<f32>, Vec<(usize, usize, Option<f32>)>, Vec<Option<f32>>)> {
        self.owner
            .bind(py)
            .try_borrow()?
            .inner
            .pool_geometry(&self.inner)
            .map_err(err)
    }

    fn read<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<u16>>> {
        let owner = self.owner.bind(py).try_borrow()?;
        let buffer = owner.inner.propose_buffer(&self.inner).map_err(err)?;
        let values = unsafe {
            std::slice::from_raw_parts(buffer.contents().cast::<u16>(), owner.inner.len())
        };
        Ok(values.to_vec().into_pyarray(py))
    }
}
