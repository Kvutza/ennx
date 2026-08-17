//! Python surrogate adapter for the Rust optimizer.

use ennx::error::ENNError;
use ennx::surrogate::{Surrogate, SurrogatePrediction};
use ndarray::{s, Array1, Array2, Array3, ArrayView2};
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::RngCore;

fn surrogate_error(error: impl std::fmt::Display) -> ENNError {
    ENNError::InvalidParameter(format!("Python surrogate: {error}"))
}

fn append_rows(current: &Array2<f64>, rows: &ArrayView2<'_, f64>) -> Result<Array2<f64>, ENNError> {
    if current.nrows() == 0 {
        return Ok(rows.to_owned());
    }
    if current.ncols() != rows.ncols() {
        return Err(ENNError::InvalidShape {
            expected: vec![rows.nrows(), current.ncols()],
            got: rows.shape().to_vec(),
        });
    }
    let split = current.nrows();
    let mut combined = Array2::zeros((split + rows.nrows(), current.ncols()));
    combined.slice_mut(s![..split, ..]).assign(current);
    combined.slice_mut(s![split.., ..]).assign(rows);
    Ok(combined)
}

/// Adapter for a Python object implementing batched `fit`, `predict`, and `draw`.
///
/// Calls cross the language boundary once per full observation or candidate
/// batch. Observation ownership stays here so the Rust optimizer remains the
/// sole state machine even when the numerical model is implemented in Python.
pub struct PythonSurrogateAdapter {
    provider: Py<PyAny>,
    x_obs: Array2<f64>,
    y_obs: Array2<f64>,
    yvar: Option<Array2<f64>>,
    has_yvar: Option<bool>,
    lengthscales: Option<Array1<f64>>,
    num_steps: usize,
}

impl PythonSurrogateAdapter {
    pub fn new(provider: Py<PyAny>, num_dim: usize, num_steps: usize) -> Self {
        Self {
            provider,
            x_obs: Array2::zeros((0, num_dim)),
            y_obs: Array2::zeros((0, 0)),
            yvar: None,
            has_yvar: None,
            lengthscales: None,
            num_steps,
        }
    }

    fn call_fit(
        &mut self,
        x: &Array2<f64>,
        y: &Array2<f64>,
        yvar: Option<&Array2<f64>>,
    ) -> Result<(), ENNError> {
        let lengthscales = Python::attach(|py| -> PyResult<Option<Array1<f64>>> {
            let x_py = x.to_owned().into_pyarray(py);
            let y_py = y.to_owned().into_pyarray(py);
            let yvar_py = match yvar {
                Some(values) => values.to_owned().into_pyarray(py).into_any().unbind(),
                None => py.None(),
            };
            let kwargs = PyDict::new(py);
            kwargs.set_item("steps", self.num_steps)?;
            let result =
                self.provider
                    .bind(py)
                    .call_method("fit", (x_py, y_py, yvar_py), Some(&kwargs))?;
            Ok(result
                .getattr("lengthscales")?
                .extract::<Option<PyReadonlyArray1<'_, f64>>>()
                .map(|array| array.map(|values| values.as_array().to_owned()))?)
        })
        .map_err(surrogate_error)?;
        if let Some(values) = lengthscales.as_ref() {
            if values.len() != x.ncols()
                || values
                    .iter()
                    .any(|value| !value.is_finite() || *value <= 0.0)
            {
                return Err(surrogate_error(format!(
                    "lengthscales must contain {} finite positive values",
                    x.ncols()
                )));
            }
        }
        self.lengthscales = lengthscales;
        Ok(())
    }

    fn validate_batch(
        &self,
        x: &ArrayView2<'_, f64>,
        y: &ArrayView2<'_, f64>,
        yvar: Option<&ArrayView2<'_, f64>>,
    ) -> Result<(), ENNError> {
        if x.nrows() != y.nrows() || x.ncols() != self.x_obs.ncols() {
            return Err(ENNError::InvalidShape {
                expected: vec![x.nrows(), self.x_obs.ncols()],
                got: vec![y.nrows(), x.ncols()],
            });
        }
        if self.y_obs.ncols() != 0 && y.ncols() != self.y_obs.ncols() {
            return Err(ENNError::InvalidParameter(format!(
                "y has {} columns but the fitted surrogate has {}",
                y.ncols(),
                self.y_obs.ncols()
            )));
        }
        if let Some(variance) = yvar {
            if variance.shape() != y.shape() {
                return Err(ENNError::InvalidShape {
                    expected: y.shape().to_vec(),
                    got: variance.shape().to_vec(),
                });
            }
        }
        if let Some(has_yvar) = self.has_yvar {
            if has_yvar != yvar.is_some() {
                return Err(ENNError::InvalidParameter(format!(
                    "y_var must be {} on every tell()",
                    if has_yvar { "provided" } else { "omitted" }
                )));
            }
        }
        Ok(())
    }
}

impl Surrogate for PythonSurrogateAdapter {
    fn fit(
        &mut self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
        _rng: &mut dyn RngCore,
    ) -> Result<(), ENNError> {
        self.validate_batch(x, y, yvar)?;
        let x_new = x.to_owned();
        let y_new = y.to_owned();
        let yvar_new = yvar.map(|values| values.to_owned());
        self.call_fit(&x_new, &y_new, yvar_new.as_ref())?;
        self.x_obs = x_new;
        self.y_obs = y_new;
        self.yvar = yvar_new;
        self.has_yvar = Some(yvar.is_some());
        Ok(())
    }

    fn fit_append(
        &mut self,
        x_new: &ArrayView2<f64>,
        y_new: &ArrayView2<f64>,
        yvar_new: Option<&ArrayView2<f64>>,
        _rng: &mut dyn RngCore,
    ) -> Result<(), ENNError> {
        self.validate_batch(x_new, y_new, yvar_new)?;
        let next_x = append_rows(&self.x_obs, x_new)?;
        let next_y = append_rows(&self.y_obs, y_new)?;
        let next_yvar = match (&self.yvar, yvar_new) {
            (Some(current), Some(rows)) => Some(append_rows(current, rows)?),
            (None, Some(rows)) => Some(rows.to_owned()),
            (None, None) => None,
            (Some(_), None) => unreachable!("validate_batch rejects a mixed y_var stream"),
        };
        self.call_fit(&next_x, &next_y, next_yvar.as_ref())?;
        self.x_obs = next_x;
        self.y_obs = next_y;
        self.yvar = next_yvar;
        self.has_yvar = Some(yvar_new.is_some());
        Ok(())
    }

    fn predict(&self, x: &ArrayView2<f64>) -> Result<SurrogatePrediction, ENNError> {
        let pred = Python::attach(|py| -> PyResult<SurrogatePrediction> {
            let x_py = x.to_owned().into_pyarray(py);
            let result = self.provider.bind(py).call_method1("predict", (x_py,))?;
            let mu = result
                .getattr("mu")?
                .extract::<PyReadonlyArray2<'_, f64>>()?
                .as_array()
                .to_owned();
            let se = result
                .getattr("sigma")?
                .extract::<PyReadonlyArray2<'_, f64>>()?
                .as_array()
                .to_owned();
            Ok(SurrogatePrediction { mu, se })
        })
        .map_err(surrogate_error)?;
        let shape = [x.nrows(), self.y_obs.ncols()];
        if pred.mu.shape() != shape || pred.se.shape() != shape {
            return Err(surrogate_error(format!(
                "predict must return mu and sigma with shape {shape:?}"
            )));
        }
        if pred.mu.iter().any(|value| !value.is_finite())
            || pred
                .se
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(surrogate_error(
                "predict returned non-finite values or negative sigma",
            ));
        }
        Ok(pred)
    }

    fn sample(
        &self,
        x: &ArrayView2<f64>,
        num_samples: usize,
        rng: &mut dyn RngCore,
    ) -> Result<Array3<f64>, ENNError> {
        let seed = rng.next_u64() & ((1_u64 << 63) - 1);
        let samples = Python::attach(|py| -> PyResult<Array3<f64>> {
            let x_py = x.to_owned().into_pyarray(py);
            Ok(self
                .provider
                .bind(py)
                .call_method1("draw", (x_py, num_samples, seed))?
                .extract::<numpy::PyReadonlyArray3<'_, f64>>()
                .map(|samples| samples.as_array().to_owned())?)
        })
        .map_err(surrogate_error)?;
        let shape = [num_samples, x.nrows(), self.y_obs.ncols()];
        if samples.shape() != shape || samples.iter().any(|value| !value.is_finite()) {
            return Err(surrogate_error(format!(
                "draw must return finite samples with shape {shape:?}"
            )));
        }
        Ok(samples)
    }

    fn lengthscales(&self) -> Option<Array1<f64>> {
        self.lengthscales.clone()
    }

    fn fitted_num_metrics(&self) -> Option<usize> {
        (self.y_obs.ncols() > 0).then_some(self.y_obs.ncols())
    }

    fn observation_count(&self) -> Option<usize> {
        Some(self.x_obs.nrows())
    }

    fn observation_row_x(&self, idx: usize) -> Result<Array1<f64>, ENNError> {
        if idx >= self.x_obs.nrows() {
            return Err(ENNError::InvalidParameter(format!(
                "observation index {idx} out of bounds"
            )));
        }
        Ok(self.x_obs.row(idx).to_owned())
    }

    fn observation_row_y(&self, idx: usize) -> Result<Array1<f64>, ENNError> {
        if idx >= self.y_obs.nrows() {
            return Err(ENNError::InvalidParameter(format!(
                "observation index {idx} out of bounds"
            )));
        }
        Ok(self.y_obs.row(idx).to_owned())
    }

    fn observations_x(&self) -> Result<Option<Array2<f64>>, ENNError> {
        Ok((self.x_obs.nrows() > 0).then(|| self.x_obs.clone()))
    }

    fn observations_y(&self) -> Result<Option<Array2<f64>>, ENNError> {
        Ok((self.y_obs.nrows() > 0).then(|| self.y_obs.clone()))
    }
}
