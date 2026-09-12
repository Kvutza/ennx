//! Synchronous, borrowed-DLPack access to the native CUDA FLAME evaluator.

use crate::py_flameconfig::configuration;
use ennx::experimental::FlameEvaluator;
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict};

fn err(message: String) -> PyErr {
    PyValueError::new_err(message)
}

/// Native FP32 forward computation from current BF16 weights on CUDA device zero.
/// No weights are retained or converted across calls. Input exports remain alive
/// until all kernels finish, including failure paths.
#[pyclass(name = "FlameEvaluator", module = "ennx.experimental", unsendable)]
pub struct PyFlameEvaluator {
    inner: FlameEvaluator,
}

#[pymethods]
impl PyFlameEvaluator {
    #[new]
    #[pyo3(signature=(config,max_tokens))]
    fn new(config: &Bound<'_, PyDict>, max_tokens: &Bound<'_, PyAny>) -> PyResult<Self> {
        if max_tokens.is_instance_of::<PyBool>() {
            return Err(PyTypeError::new_err(
                "max_tokens must be an integer, not bool",
            ));
        }
        Ok(Self {
            inner: FlameEvaluator::new(configuration(config)?, max_tokens.extract()?)
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

    #[pyo3(signature=(weights,tokens,masks))]
    fn losses<'py>(
        &mut self,
        py: Python<'py>,
        weights: &Bound<'py, PyAny>,
        tokens: Vec<Vec<i32>>,
        masks: Vec<Vec<bool>>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        self.inner.check_batch(&tokens, &masks).map_err(err)?;
        let input = crate::dlpack::Input::new(weights)?;
        // Input owns the export lease; the native call synchronizes before return.
        let losses =
            unsafe { self.inner.losses(input.pointer, input.len, &tokens, &masks) }.map_err(err)?;
        Ok(losses.into_pyarray(py))
    }

    #[pyo3(signature=(weights,tokens))]
    fn logits<'py>(
        &mut self,
        py: Python<'py>,
        weights: &Bound<'py, PyAny>,
        tokens: Vec<i32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        self.inner.check_tokens(&tokens).map_err(err)?;
        let input = crate::dlpack::Input::new(weights)?;
        let logits =
            unsafe { self.inner.logits(input.pointer, input.len, &tokens) }.map_err(err)?;
        let array = Array2::from_shape_vec((tokens.len(), self.inner.vocab()), logits)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(array.into_pyarray(py))
    }
}
