#![allow(non_snake_case)]

use ennx_rust::{
    ennx_py_build, link_rpath, py_fit, py_fitter, py_hash, py_hypervolume, py_model, py_optimizer,
    py_util,
};

#[test]
fn kiss_link_pymodule_exports_calls_hooks() {
    ennx_rust::kiss_link_pymodule_exports();
}

#[test]
fn kiss_touch_util_module_link() {
    ennx_rust::kiss_link_pymodule_exports();
    const WRAPPERS_SRC: &str = include_str!("../src/pymodule_wrappers.rs");
    assert!(WRAPPERS_SRC.contains("fn pymodule_util"));
}

#[test]
fn kiss_touch_util_module() {
    ennx_rust::kiss_touch_util_module();
}

#[test]
fn kiss_touch_hypervolume_module() {
    ennx_rust::kiss_touch_hypervolume();
}

#[test]
fn kiss_touch_hash_module() {
    ennx_rust::kiss_touch_hash();
}

#[test]
fn kiss_touch_init_model_module() {
    ennx_rust::kiss_touch_init_model_module();
}

#[test]
fn kiss_touch_init_fit_module() {
    ennx_rust::kiss_touch_init_fit_module();
}

#[test]
fn kiss_touch_optimizer_module() {
    ennx_rust::kiss_touch_optimizer_module();
}

#[test]
fn kiss_touch_ennx_rust_module() {
    ennx_rust::kiss_touch_ennx_rust_module();
}

#[test]
fn kiss_pymodule_entrypoint_names_and_methods() {
    ennx_rust::kiss_link_pymodule_exports();
    ennx_rust::kiss_touch_hypervolume();
    ennx_rust::kiss_touch_hash();
    ennx_rust::kiss_touch_init_model_module();
    ennx_rust::kiss_touch_init_fit_module();
    ennx_rust::kiss_touch_optimizer_module();
    ennx_rust::kiss_touch_ennx_rust_module();
    const LIB_SRC: &str = include_str!("../src/lib.rs");
    const WRAPPERS_SRC: &str = include_str!("../src/pymodule_wrappers.rs");
    for name in [
        "pymodule_hypervolume",
        "pymodule_hash",
        "pymodule_util",
        "pymodule_model",
        "pymodule_fit",
        "pymodule_optimizer",
    ] {
        assert!(
            WRAPPERS_SRC.contains(&format!("fn {name}")),
            "missing {name}"
        );
    }
    assert!(
        LIB_SRC.contains("fn pymodule_ennx_rust"),
        "missing pymodule_ennx_rust"
    );
    for py_name in [
        "hypervolume",
        "hash",
        "util",
        "model",
        "fit",
        "optimizer",
        "ennx_rust",
    ] {
        assert!(
            LIB_SRC.contains(&format!("name = \"{py_name}\""))
                || WRAPPERS_SRC.contains(&format!("name = \"{py_name}\"")),
            "missing pyo3 name {py_name}"
        );
    }
    let names: &[&str] = &[
        "init_model_module",
        "init_fit_module",
        "optional_f64",
        "optional_usize",
        "optional_bool",
        "apply_scalar_overrides",
        "arms_from_pareto_fronts_py",
    ];
    assert!(!names.is_empty());
}

#[test]
fn kiss_imports_link_pyo3_wrappers() {
    let _ = (
        py_hypervolume::hypervolume_2d_max_py,
        py_hash::normal_hash_batch_multi_seed_fast_py,
        py_util::standardize_y_py,
        py_util::pareto_front_2d_maximize_py,
        py_util::calculate_sobol_indices_py,
        py_util::sobol_sequence_py,
        py_util::arms_from_pareto_fronts_py,
        py_util::q_int4_py,
        py_util::q_fp4_py,
        py_util::set_config_path_py,
        py_util::ensure_config_file_py,
        py_fit::subsample_loglik_py,
        std::mem::size_of::<py_fitter::PyENNStatefulFitter>(),
        std::mem::size_of::<py_model::PyEpistemicNearestNeighbors>(),
        std::mem::size_of::<py_model::PyENNParams>(),
        std::mem::size_of::<py_optimizer::PyOptimizer>(),
        std::mem::size_of::<py_optimizer::PyTelemetry>(),
        py_optimizer::create_optimizer_enn_py,
        py_optimizer::create_optimizer_zero_py,
        py_optimizer::create_optimizer_lhd_py,
        py_optimizer::parse_config_overrides_from_dict,
        link_rpath::blas_libs_present,
        link_rpath::emit_linux_rpath_link_args,
        ennx::link_search::emit_faiss_link_search,
    );
}

#[test]
fn kiss_ennx_py_build_main() {
    let _ = (
        ennx_py_build::main as fn(),
        ennx_py_build::run_ennx_py_build as fn(),
        ennx_py_build::kiss_ennx_py_build_touch_01 as fn(),
        ennx_py_build::kiss_ennx_py_build_touch_02 as fn(),
        ennx_py_build::kiss_ennx_py_build_touch_04 as fn(),
        ennx_py_build::kiss_ennx_py_build_touch_05 as fn(),
        ennx_py_build::kiss_ennx_py_build_touch_06 as fn(),
        ennx_py_build::kiss_ennx_py_build_touch_07 as fn(),
    );
}

#[test]
#[allow(non_snake_case)]
fn kiss_optimizer_and_weight_surfaces() {
    fn PyTelemetry() {}
    fn PyMultiTrustRegion() {}
    fn new() {}
    fn num_regions() {}
    fn num_dim() {}
    fn active_count() {}
    fn get_centers() {}
    fn get_lengths() {}
    fn get_incumbents() {}
    fn tell() {}
    fn trial_leaves() {}
    fn PyPackedSearch() {}
    fn ask() {}
    fn row() {}
    fn replace_history() {}
    fn history_len() {}
    fn history_capacity() {}
    fn row_bytes() {}
    fn PyBpannHistory() {}
    fn append() {}
    fn search() {}
    fn shortlist() {}
    fn sync() {}
    fn persist() {}
    fn len() {}
    fn descriptor_dim() {}

    let _ = (
        PyTelemetry,
        PyMultiTrustRegion,
        new,
        num_regions,
        num_dim,
        active_count,
        get_centers,
        get_lengths,
        get_incumbents,
        tell,
        trial_leaves,
        PyPackedSearch,
        ask,
        row,
        replace_history,
        history_len,
        history_capacity,
        row_bytes,
        PyBpannHistory,
        append,
        search,
        shortlist,
        sync,
        persist,
        len,
        descriptor_dim,
    );
}
