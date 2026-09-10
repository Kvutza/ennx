use pyo3::prelude::*;

/// Hypervolume calculation module
#[pymodule]
#[pyo3(name = "hypervolume")]
pub fn pymodule_hypervolume(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(crate::py_hypervolume::hypervolume_py, m)?)?;
    Ok(())
}

/// Hash-based RNG module
#[pymodule]
#[pyo3(name = "hash")]
pub fn pymodule_hash(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(crate::py_hash::normal_py, m)?)?;
    Ok(())
}

/// Utility functions module
#[pymodule]
#[pyo3(name = "util")]
pub fn pymodule_util(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(crate::py_util::y_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_util::pareto_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_util::sobol_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_util::bind_sobol, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_util::arms_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_util::q_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_util::bind_quantize, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_util::set_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_util::ensure_py, m)?)?;
    Ok(())
}

/// ENN model module
#[pymodule]
#[pyo3(name = "model")]
pub fn pymodule_model(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<crate::py_model::PyENN>()?;
    m.add_class::<crate::py_model::PyENNParams>()?;
    Ok(())
}

/// Parameter fitting module
#[pymodule]
#[pyo3(name = "fit")]
pub fn pymodule_fit(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<crate::py_fitter::PyENNStatefulFitter>()?;
    m.add_function(wrap_pyfunction!(crate::py_fit::subsample_py, m)?)?;
    Ok(())
}

/// Experimental native model-package API.
#[pymodule]
#[pyo3(name = "experimental")]
pub fn pymodule_experimental(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<crate::py_experimental::PyModelPackage>()?;
    m.add_class::<crate::py_experimental::PyResidentBoSession>()?;
    #[cfg(all(target_os = "macos", feature = "metal"))]
    m.add_class::<crate::py_experimental::PyNativeKdaModel>()?;
    Ok(())
}

/// Candidate evaluation coordination.
#[pymodule]
#[pyo3(name = "search")]
pub fn pymodule_search(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<crate::py_weights::PySearch>()?;
    m.add_class::<crate::py_weights::PySearchOptimizer>()?;
    m.add_class::<crate::py_weights::PyTrial>()?;
    m.add_class::<crate::py_parameter::PyParameter>()?;
    Ok(())
}

/// Optimizer module
#[pymodule]
#[pyo3(name = "optimizer")]
pub fn pymodule_optimizer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<crate::py_optimizer::PyOptimizer>()?;
    m.add_class::<crate::py_optimizer::PyMultiTrustRegion>()?;
    m.add_class::<crate::py_optimizer::PyTelemetry>()?;
    m.add_class::<crate::py_weights::PyDenseLinear>()?;
    #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
    m.add_class::<crate::py_weights::PyParamBuffer>()?;
    #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
    m.add_class::<crate::py_bf16::PyParamBlock>()?;
    #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
    m.add_class::<crate::py_bf16::PySearchState>()?;
    #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
    m.add_class::<crate::py_bf16::PyProposals>()?;
    m.add_class::<crate::py_weights::PyBpannHistory>()?;
    m.add_function(wrap_pyfunction!(crate::py_optimizer::create_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_optimizer::enn_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_optimizer::bind_regions, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_optimizer::bind_zero, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_optimizer::bind_lhd, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_weights::weight_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_weights::select_weight, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_weights::dense_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_weights::bind_dist, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_weights::bind_linear, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_weights::sparse_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_weights::bind_xor, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_weights::bind_missing, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_weights::bind_merge, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_weights::bind_take, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_weights::bind_apply, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_weights::bind_blocks, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_weights::bind_draw, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_weights::bind_select, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_weights::select_sparse, m)?)?;
    Ok(())
}
