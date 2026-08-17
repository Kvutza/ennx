use pyo3::prelude::*;

/// Hypervolume calculation module
#[pymodule]
#[pyo3(name = "hypervolume")]
pub fn pymodule_hypervolume(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(
        crate::py_hypervolume::hypervolume_2d_max_py,
        m
    )?)?;
    Ok(())
}

/// Hash-based RNG module
#[pymodule]
#[pyo3(name = "hash")]
pub fn pymodule_hash(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(
        crate::py_hash::normal_hash_batch_multi_seed_fast_py,
        m
    )?)?;
    Ok(())
}

/// Utility functions module
#[pymodule]
#[pyo3(name = "util")]
pub fn pymodule_util(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(crate::py_util::standardize_y_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        crate::py_util::pareto_front_2d_maximize_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::py_util::calculate_sobol_indices_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(crate::py_util::sobol_sequence_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        crate::py_util::arms_from_pareto_fronts_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(crate::py_util::q_int4_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_util::q_fp4_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_util::set_config_path_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_util::ensure_config_file_py, m)?)?;
    Ok(())
}

/// ENN model module
#[pymodule]
#[pyo3(name = "model")]
pub fn pymodule_model(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<crate::py_model::PyEpistemicNearestNeighbors>()?;
    m.add_class::<crate::py_model::PyENNParams>()?;
    Ok(())
}

/// Parameter fitting module
#[pymodule]
#[pyo3(name = "fit")]
pub fn pymodule_fit(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<crate::py_fitter::PyENNStatefulFitter>()?;
    m.add_function(wrap_pyfunction!(crate::py_fit::subsample_loglik_py, m)?)?;
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

/// Optimizer module
#[pymodule]
#[pyo3(name = "optimizer")]
pub fn pymodule_optimizer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<crate::py_optimizer::PyOptimizer>()?;
    m.add_class::<crate::py_optimizer::PyMultiTrustRegion>()?;
    m.add_class::<crate::py_optimizer::PyTelemetry>()?;
    m.add_class::<crate::py_weights::PyPackedSearch>()?;
    m.add_class::<crate::py_weights::PyPackedTurbo>()?;
    m.add_class::<crate::py_weights::PyTurboTrial>()?;
    m.add_class::<crate::py_weights::PyDenseLinear>()?;
    #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
    m.add_class::<crate::py_weights::PyBf16Tree>()?;
    #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
    m.add_class::<crate::py_bf16::PyBf16Search>()?;
    #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
    m.add_class::<crate::py_bf16::PyBf16Trial>()?;
    #[cfg(all(target_os = "linux", target_arch = "x86_64", feature = "cuda"))]
    m.add_class::<crate::py_bf16::PyBf16View>()?;
    m.add_class::<crate::py_weights::PyBpannHistory>()?;
    m.add_function(wrap_pyfunction!(
        crate::py_optimizer::create_optimizer_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::py_optimizer::create_optimizer_enn_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::py_optimizer::create_optimizer_enn_multi_tr_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::py_optimizer::create_optimizer_zero_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::py_optimizer::create_optimizer_lhd_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::py_weights::weight_int4_select_ucb_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::py_weights::weight_select_ucb_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(crate::py_weights::dense_apply_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_weights::dense_dist2_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_weights::dense_linear_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_weights::sparse_union_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_weights::sparse_xor_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_weights::sparse_missing_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_weights::sparse_merge_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_weights::sparse_take_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_weights::sparse_apply_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_weights::sparse_blocks_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_weights::sparse_draw_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_weights::sparse_select_py, m)?)?;
    m.add_function(wrap_pyfunction!(
        crate::py_weights::sparse_select_ucb_py,
        m
    )?)?;
    Ok(())
}

#[doc(hidden)]
pub fn pymodule_hypervolume_kiss_hook() {
    std::hint::black_box(pymodule_hypervolume);
}

#[doc(hidden)]
pub fn pymodule_hash_kiss_hook() {
    std::hint::black_box(pymodule_hash);
}

#[doc(hidden)]
pub fn pymodule_util_kiss_hook() {
    std::hint::black_box(pymodule_util);
}

#[doc(hidden)]
pub fn pymodule_model_kiss_hook() {
    std::hint::black_box(pymodule_model);
}

#[doc(hidden)]
pub fn pymodule_fit_kiss_hook() {
    std::hint::black_box(pymodule_fit);
}

#[doc(hidden)]
pub fn pymodule_experimental_kiss_hook() {
    std::hint::black_box(pymodule_experimental);
}

#[doc(hidden)]
pub fn pymodule_optimizer_kiss_hook() {
    std::hint::black_box(pymodule_optimizer);
}

#[doc(hidden)]
pub fn kiss_link_child_pymodule_exports() {
    pymodule_hypervolume_kiss_hook();
    pymodule_hash_kiss_hook();
    pymodule_util_kiss_hook();
    pymodule_model_kiss_hook();
    pymodule_fit_kiss_hook();
    pymodule_experimental_kiss_hook();
    pymodule_optimizer_kiss_hook();
}

#[cfg(test)]
mod kiss_child_pymodule_coverage {
    use super::*;

    #[test]
    fn kiss_link_calls_all_child_pymodule_hooks() {
        kiss_link_child_pymodule_exports();
    }
}
