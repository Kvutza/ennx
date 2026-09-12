//! Shared Python architecture validation for native FLAME evaluators.

#[cfg(target_os = "linux")]
use ennx::experimental::FlameConfig;
#[cfg(target_os = "macos")]
use ennx::experimental::MetalFlameConfig as FlameConfig;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict};

pub(crate) fn configuration(config: &Bound<'_, PyDict>) -> PyResult<FlameConfig> {
    if config.len() != 12 {
        return Err(PyValueError::new_err(
            "FLAME config must contain exactly the twelve architecture fields",
        ));
    }
    let field = |name: &str| -> PyResult<Bound<'_, PyAny>> {
        config
            .get_item(name)?
            .ok_or_else(|| PyValueError::new_err(format!("Missing FLAME config field {name}")))
    };
    let integer = |name: &str| -> PyResult<u32> {
        let value = field(name)?;
        if value.is_instance_of::<PyBool>() {
            return Err(PyTypeError::new_err(format!(
                "{name} must be an integer, not bool"
            )));
        }
        value.extract()
    };
    let scalar = |name: &str| -> PyResult<f32> {
        let value = field(name)?;
        if value.is_instance_of::<PyBool>() {
            return Err(PyTypeError::new_err(format!(
                "{name} must be a real scalar, not bool"
            )));
        }
        Ok(value.extract::<f64>()? as f32)
    };
    Ok(FlameConfig {
        layers: integer("layers")?,
        width: integer("width")?,
        heads: integer("heads")?,
        vocab: integer("vocab")?,
        dense_width: integer("dense_width")?,
        expert_width: integer("expert_width")?,
        shared_width: integer("shared_width")?,
        experts: integer("experts")?,
        top_k: integer("top_k")?,
        context: integer("context")?,
        epsilon: scalar("epsilon")?,
        rope_base: scalar("rope_base")?,
    })
}
