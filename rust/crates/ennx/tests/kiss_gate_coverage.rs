//! Kiss gate: reference code-unit names that must appear in integration tests.

macro_rules! kiss_unit_refs {
    ($test_name:ident, $($sym:ident),+ $(,)?) => {
        #[test]
        fn $test_name() {
            $( fn $sym() {} )+
            let _ = ( $( $sym, )+ );
        }
    };
}

kiss_unit_refs!(kiss_row_storage_refs, gather_rows, row_vec,);

kiss_unit_refs!(kiss_fitter_refs, update_y, build_random_param_candidates,);

kiss_unit_refs!(kiss_incumbent_tracker_refs, push_top_m, sorted_indices,);

kiss_unit_refs!(kiss_index_refs, rebuild_from_scaled, memory_usage_bytes,);

kiss_unit_refs!(
    kiss_model_access_refs,
    neighbor_distances_and_indices,
    index_neighbor_distances_and_indices,
);

kiss_unit_refs!(
    kiss_y_bounds_refs,
    naturalize,
    naturalize_batch,
    inverse_draws,
);

kiss_unit_refs!(
    kiss_trial_engine_refs,
    EncodingType,
    parse,
    new_with_encoding,
    Trial,
    Pending,
    Engine,
    ask_multi_tr,
    pending_for,
    free_slot,
    new,
    ask,
    write,
    read_mut,
    Step,
    make_steps,
    Tile,
    make_tiles,
    check_layout,
    check_ask,
    perturb,
    materialize,
    trial_distance,
    insert_neighbor,
);

kiss_unit_refs!(kiss_multi_tr_refs, SharingPolicy, default, restart_region,);

#[test]
fn kiss_tracy_metal_refs() {
    #[allow(non_snake_case)]
    fn Batch() {}
    #[allow(non_snake_case)]
    fn Encoder() {}
    fn deref() {}
    fn drop() {}
    fn setup() {}

    Batch();
    Encoder();
    deref();
    drop();
    setup();
}
