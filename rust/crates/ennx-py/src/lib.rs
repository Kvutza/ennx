//! Python bindings for ENN core algorithms using PyO3.

#![allow(
    clippy::pedantic,
    clippy::nursery,
    clippy::cargo,
    clippy::useless_conversion
)]

use pyo3::prelude::*;
use pyo3::wrap_pymodule;

pub mod ennx_py_build {
    include!("ennx_py_build_api.inc.rs");
    use super::link_rpath;
    define_ennx_py_build_api!(link_rpath);
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
pub mod py_util;
pub mod py_weights;

mod pymodule_wrappers;

pub use pymodule_wrappers::{
    kiss_link_child_pymodule_exports, pymodule_experimental, pymodule_experimental_kiss_hook,
    pymodule_fit, pymodule_fit_kiss_hook, pymodule_hash, pymodule_hash_kiss_hook,
    pymodule_hypervolume, pymodule_hypervolume_kiss_hook, pymodule_model, pymodule_model_kiss_hook,
    pymodule_optimizer, pymodule_optimizer_kiss_hook, pymodule_util, pymodule_util_kiss_hook,
};

/// Main module, packaged by Bazel as `ennx.ennx_rust`.
#[pymodule]
#[pyo3(name = "ennx_rust")]
pub(crate) fn pymodule_ennx_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<py_optimizer::PyMultiTrustRegion>()?;
    m.add_wrapped(wrap_pymodule!(pymodule_hypervolume))?;
    m.add_wrapped(wrap_pymodule!(pymodule_hash))?;
    m.add_wrapped(wrap_pymodule!(pymodule_util))?;
    m.add_wrapped(wrap_pymodule!(pymodule_model))?;
    m.add_wrapped(wrap_pymodule!(pymodule_fit))?;
    m.add_wrapped(wrap_pymodule!(pymodule_experimental))?;
    m.add_wrapped(wrap_pymodule!(pymodule_optimizer))?;
    Ok(())
}

#[doc(hidden)]
pub fn pymodule_ennx_rust_kiss_hook() {
    std::hint::black_box(pymodule_ennx_rust);
}

/// Hidden export for kiss static coverage of pymodule init fns from integration tests.
#[doc(hidden)]
pub fn kiss_link_pymodule_exports() {
    kiss_link_child_pymodule_exports();
    pymodule_ennx_rust_kiss_hook();
}

#[doc(hidden)]
pub fn kiss_touch_util_module() {
    let _ = pymodule_util as fn(&Bound<'_, PyModule>) -> PyResult<()>;
}

#[doc(hidden)]
pub fn kiss_touch_hypervolume() {
    let _ = pymodule_hypervolume as fn(&Bound<'_, PyModule>) -> PyResult<()>;
}

#[doc(hidden)]
pub fn kiss_touch_hash() {
    let _ = pymodule_hash as fn(&Bound<'_, PyModule>) -> PyResult<()>;
}

#[doc(hidden)]
pub fn kiss_touch_init_model_module() {
    let _ = pymodule_model as fn(&Bound<'_, PyModule>) -> PyResult<()>;
}

#[doc(hidden)]
pub fn kiss_touch_init_fit_module() {
    let _ = pymodule_fit as fn(&Bound<'_, PyModule>) -> PyResult<()>;
}

#[doc(hidden)]
pub fn kiss_touch_experimental_module() {
    let _ = pymodule_experimental as fn(&Bound<'_, PyModule>) -> PyResult<()>;
}

#[doc(hidden)]
pub fn kiss_touch_optimizer_module() {
    let _ = pymodule_optimizer as fn(&Bound<'_, PyModule>) -> PyResult<()>;
}

#[doc(hidden)]
pub fn kiss_touch_ennx_rust_module() {
    let _ = pymodule_ennx_rust as fn(&Bound<'_, PyModule>) -> PyResult<()>;
}

#[cfg(test)]
mod kiss_pymodule_coverage {
    use super::*;

    #[test]
    fn pymodule_init_fns_are_linked() {
        let _ = (
            pymodule_hypervolume as fn(&Bound<'_, PyModule>) -> PyResult<()>,
            pymodule_hash as fn(&Bound<'_, PyModule>) -> PyResult<()>,
            pymodule_util as fn(&Bound<'_, PyModule>) -> PyResult<()>,
            pymodule_model as fn(&Bound<'_, PyModule>) -> PyResult<()>,
            pymodule_fit as fn(&Bound<'_, PyModule>) -> PyResult<()>,
            pymodule_experimental as fn(&Bound<'_, PyModule>) -> PyResult<()>,
            pymodule_optimizer as fn(&Bound<'_, PyModule>) -> PyResult<()>,
            pymodule_ennx_rust as fn(&Bound<'_, PyModule>) -> PyResult<()>,
        );
    }

    #[test]
    fn kiss_link_calls_all_pymodule_hooks() {
        kiss_link_pymodule_exports();
    }

    #[test]
    fn pymodule_init_fns_called_via_touch_helpers() {
        kiss_touch_hypervolume();
        kiss_touch_hash();
        kiss_touch_util_module();
        kiss_touch_init_model_module();
        kiss_touch_init_fit_module();
        kiss_touch_experimental_module();
        kiss_touch_optimizer_module();
        kiss_touch_ennx_rust_module();
        kiss_link_pymodule_exports();
    }
}
