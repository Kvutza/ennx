//! Python bindings for ENN core algorithms using PyO3.

#![allow(
    clippy::pedantic,
    clippy::nursery,
    clippy::cargo,
    clippy::useless_conversion
)]

use pyo3::prelude::*;
use pyo3::wrap_pymodule;

pub mod xpybld {
    include!("xpybld_api.inc.rs");
    use super::link_rpath;
    ennx_api!(link_rpath);
}
pub mod adapter;
#[cfg_attr(
    not(all(target_os = "linux", target_arch = "x86_64", feature = "cuda")),
    allow(dead_code)
)]
mod dlpack;
pub mod link_rpath;
#[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
pub mod py_bf16;
pub mod py_experimental;
pub mod py_fit;
pub mod py_fitter;
pub mod py_hash;
pub mod py_hypervolume;
pub mod py_model;
pub mod py_optimizer;
pub mod py_parameter;
pub mod py_util;
pub mod py_weights;

mod pymodule_wrappers;

pub use pymodule_wrappers::{
    pymodule_experimental, pymodule_fit, pymodule_hash, pymodule_hypervolume, pymodule_model,
    pymodule_optimizer, pymodule_search, pymodule_util,
};

/// Main module, packaged by Bazel as `ennx.ennx_rust`.
#[pymodule]
#[pyo3(name = "ennx_rust")]
pub(crate) fn ennx_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<py_optimizer::PyMultiTrustRegion>()?;
    m.add_wrapped(wrap_pymodule!(pymodule_hypervolume))?;
    m.add_wrapped(wrap_pymodule!(pymodule_hash))?;
    m.add_wrapped(wrap_pymodule!(pymodule_util))?;
    m.add_wrapped(wrap_pymodule!(pymodule_model))?;
    m.add_wrapped(wrap_pymodule!(pymodule_fit))?;
    m.add_wrapped(wrap_pymodule!(pymodule_experimental))?;
    m.add_wrapped(wrap_pymodule!(pymodule_optimizer))?;
    m.add_wrapped(wrap_pymodule!(pymodule_search))?;
    Ok(())
}
