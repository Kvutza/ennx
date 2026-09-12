fn main() {
    if let Err(error) = run() {
        eprintln!("TuRBO-ENN worker: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let path = std::env::args_os()
        .nth(1)
        .ok_or("expected a resolved TuRBO-ENN TOML path")?;
    let artifact_dir = std::env::args_os()
        .nth(2)
        .map(std::path::PathBuf::from)
        .ok_or("expected an artifact directory")?;
    let (run, _) = ennx::config::load_turbo_enn_config(std::path::Path::new(&path))?;
    #[cfg(target_os = "macos")]
    {
        if run.study == Some(ennx::TurboEnnStudy::Pretrain) {
            let dataset = run.dataset()
                .ok_or("pretrain study was not resolved to an immutable dataset")?;
            let result = ennx::experimental::run_pretrain(&run, dataset)?;
            let enn = run.resident_enn(run.acquisition_seed())?;
            let ask = enn.ask;
            let length = run.length();
            let shape = run.trust_region_shape.unwrap_or(
                ennx::config::TrustRegionShape::TensorFamilyStatic,
            );
            let updates_path = artifact_dir.join("tensor-updates.jsonl");
            result.write_updates(&updates_path)?;
            eprintln!("[weights] per-tensor records: {}", updates_path.display());
            let controller_path = artifact_dir.join("controller.jsonl");
            result.write_controller(&controller_path)?;
            eprintln!("[controller] round records: {}", controller_path.display());
            let goal_met = result.max_wall_seconds <= f64::from(run.target_round_ms()) / 1000.0;
            std::fs::write(
                artifact_dir.join("result.toml"),
                format!(
                    "status = \"completed\"\nstage = \"candidate_applied_bo\"\nkernel = \"metal4_tensorops_m128_n64_plus_pisa1_simdgroup_matrix\"\nattention_shape = \"pisa1_q8_kv1_d64_c64_k8\"\nqkv_width = 640\nproposal = {:?}\ntrust_region_shape = {:?}\ncontroller = {:?}\nparameters = {}\nsearch_dimensions = {}\nlogical_history_capacity = 128\nresident_weight_rows = 2\nacquisition = {:?}\nk_neighbors = {}\nk_neighbors_role = \"configured_maximum\"\nfit_neighbors = {}\ndistance_scaling = {:?}\nlocal_scale_neighbors = {}\nepistemic_scale = {}\naleatoric_scale = {}\ny_scale = {}\nbeta = {}\nlength_init = {}\nlength_min = {}\nlength_max = {}\nproposal_seed = {}\nacquisition_seed = {}\ndistance = \"resident_exact_nonresident_approximate\"\ndata = \"causal_pretraining_paired_block128_ennxptn1\"\ndiagnostics = false\nloop_seconds = {:.9}\nactual_bo_gpu_median_ms = {:.6}\nactual_bo_wall_median_ms = {:.6}\nactual_bo_wall_min_ms = {:.6}\nactual_bo_wall_max_ms = {:.6}\nactual_bo_accepted = {}\ntarget_round_ms = {}\nfull_round_goal_met = {goal_met}\n",
                    format!("fp16_full_weight_{}", run.perturbation().name()),
                    shape.name(),
                    if run.reliability_controller().is_some() {
                        "reliability"
                    } else {
                        "turbo"
                    },
                    result.parameters,
                    result.parameters,
                    format!("{:?}", ask.acquisition).to_ascii_lowercase(),
                    ask.neighbors,
                    enn.fit_neighbors,
                    enn.distance_scaling.name(),
                    enn.local_scale_neighbors,
                    ask.epistemic_scale,
                    ask.aleatoric_scale,
                    ask.y_scale,
                    ask.beta,
                    length.length_init,
                    length.length_min,
                    length.length_max,
                    run.proposal_seed(),
                    run.acquisition_seed(),
                    result.loop_seconds,
                    result.median_gpu_seconds * 1000.0,
                    result.median_wall_seconds * 1000.0,
                    result.min_wall_seconds * 1000.0,
                    result.max_wall_seconds * 1000.0,
                    result.accepted,
                    run.target_round_ms(),
                ),
            ).map_err(|error| error.to_string())?;
            eprintln!(
                "TURBO_ENN_SUMMARY rounds={} median_seconds={:.9} max_seconds={:.9} accepted={} target_ms={} target_met={goal_met}",
                run.rounds(), result.median_wall_seconds, result.max_wall_seconds, result.accepted, run.target_round_ms(),
            );
            return Ok(());
        }
        if run.study == Some(ennx::TurboEnnStudy::MoeLayer) {
            let probe = ennx::experimental::run_grouped_moe_probe_with_dataset(
                run.rounds(),
                run.target_round_ms(),
                run.dataset(),
            )?;
            let updates_path = artifact_dir.join("tensor-updates.jsonl");
            probe.write_updates(&updates_path)?;
            eprintln!("[weights] per-tensor records: {}", updates_path.display());
            let data = if run.dataset().is_some() {
                "causal_pretraining_ennxptn1"
            } else {
                "deterministic_noncompressible_fp16"
            };
            std::fs::write(
                artifact_dir.join("result.toml"),
                format!(
                    "status = \"completed\"\nstage = \"candidate_applied_bo\"\nkernel = \"metal4_tensorops_m128_n64_plus_pisa1_simdgroup_matrix\"\nattention_shape = \"pisa1_q8_kv1_d64_c64_k8\"\nqkv_width = 640\nproposal = \"fp16_full_weight_independent_gaussian\"\nparameters = {}\nsearch_dimensions = {}\nhistory_capacity = 2\ndistance = \"full_realized_weights\"\ndata = {data:?}\nparity_reference = \"mps_full_output_and_cpu_samples\"\nrouting_gpu_ms = {:.6}\nactivation_gpu_ms = {:.6}\nresidual_gpu_ms = {:.6}\nmaterialize_gate_up_gpu_ms = {:.6}\nmaterialize_down_gpu_ms = {:.6}\nprojections_gpu_ms = {:.6}\nprojections_wall_ms = {:.6}\npisa1_pyramid_gpu_ms = {:.6}\npisa1_selection_gpu_ms = {:.6}\npisa1_attention_gpu_ms = {:.6}\npisa1_layer_gpu_ms = {:.6}\npisa1_layer_wall_ms = {:.6}\nlayer_gpu_ms = {:.6}\nlayer_wall_ms = {:.6}\nprojected_ffn_ms = {:.6}\nprojected_ffn_and_projections_ms = {:.6}\nprojected_measured_model_ms = {:.6}\nsustained_model_gpu_ms = {:.6}\nsustained_model_wall_ms = {:.6}\nsustained_materialization_gpu_ms = {:.6}\nsustained_projections_gpu_ms = {:.6}\nsustained_pisa1_gpu_ms = {:.6}\nsustained_ffn_gpu_ms = {:.6}\nsustained_mps_projections_gpu_ms = {:.6}\nsustained_mps_projections_wall_ms = {:.6}\nsustained_mps_ffn_gpu_ms = {:.6}\nsustained_mps_ffn_wall_ms = {:.6}\nsustained_scope = \"single_command_24_layers_2_passes_same_layer_weights\"\nobjective_flops = {}\nprojection_objective_flops = {}\npisa1_objective_flops = {}\neffective_tflops = {:.6}\nsustained_effective_tflops = {:.6}\ngate_up_max_abs_error = {:.9}\ndown_max_abs_error = {:.9}\nqkv_max_abs_error = {:.9}\noutput_projection_max_abs_error = {:.9}\npisa1_max_abs_error = {:.9}\nactual_bo_gpu_median_ms = {:.6}\nactual_bo_wall_median_ms = {:.6}\nactual_bo_wall_min_ms = {:.6}\nactual_bo_wall_max_ms = {:.6}\nactual_bo_accepted = {}\ntarget_round_ms = {}\nfull_round_goal_met = {}\n",
                    probe.parameters,
                    probe.parameters,
                    probe.routing_gpu_seconds * 1000.0,
                    probe.activation_gpu_seconds * 1000.0,
                    probe.residual_gpu_seconds * 1000.0,
                    probe.materialize_gate_up_gpu_seconds * 1000.0,
                    probe.materialize_down_gpu_seconds * 1000.0,
                    probe.projections_gpu_seconds * 1000.0,
                    probe.projections_wall_seconds * 1000.0,
                    probe.pisa1_pyramid_gpu_seconds * 1000.0,
                    probe.pisa1_selection_gpu_seconds * 1000.0,
                    probe.pisa1_attention_gpu_seconds * 1000.0,
                    probe.pisa1_layer_gpu_seconds * 1000.0,
                    probe.pisa1_layer_wall_seconds * 1000.0,
                    probe.layer_gpu_seconds * 1000.0,
                    probe.layer_wall_seconds * 1000.0,
                    probe.projected_ffn_seconds * 1000.0,
                    probe.projected_ffn_and_projections_seconds * 1000.0,
                    probe.projected_measured_model_seconds * 1000.0,
                    probe.sustained_model_gpu_seconds * 1000.0,
                    probe.sustained_model_wall_seconds * 1000.0,
                    probe.sustained_materialization_gpu_seconds * 1000.0,
                    probe.sustained_projections_gpu_seconds * 1000.0,
                    probe.sustained_pisa1_gpu_seconds * 1000.0,
                    probe.sustained_ffn_gpu_seconds * 1000.0,
                    probe.sustained_mps_projections_gpu_seconds * 1000.0,
                    probe.sustained_mps_projections_wall_seconds * 1000.0,
                    probe.sustained_mps_ffn_gpu_seconds * 1000.0,
                    probe.sustained_mps_ffn_wall_seconds * 1000.0,
                    probe.objective_flops,
                    probe.projection_objective_flops,
                    probe.pisa1_objective_flops,
                    probe.effective_tflops,
                    probe.objective_flops as f64 / probe.sustained_model_wall_seconds / 1e12,
                    probe.gate_up_max_abs_error,
                    probe.down_max_abs_error,
                    probe.qkv_max_abs_error,
                    probe.output_projection_max_abs_error,
                    probe.pisa1_max_abs_error,
                    probe.actual_bo_median_gpu_seconds * 1000.0,
                    probe.actual_bo_median_wall_seconds * 1000.0,
                    probe.actual_bo_min_wall_seconds * 1000.0,
                    probe.actual_bo_max_wall_seconds * 1000.0,
                    probe.actual_bo_accepted,
                    run.target_round_ms(),
                    probe.meets_target,
                ),
            )
            .map_err(|error| error.to_string())?;
            return Ok(());
        }
        let study = ennx::experimental::run_round_study(&run)?;
        let goal_met = study.meets_target(run.target_round_ms());
        let mut result = format!(
            "status = \"completed\"\nstage = \"round_study\"\nobservation_policy = \"one_observation_noisy_bo\"\ninitial_objective_ms = {:.6}\nmean_round_ms = {:.6}\nmax_round_ms = {:.6}\nmax_allocated_bytes = {}\ntarget_round_ms = {}\ngoal_met = {goal_met}\n",
            study.initial_objective_ms,
            study.mean_round_ms,
            study.max_round_ms(),
            study.max_allocated_bytes(),
            run.target_round_ms(),
        );
        for (index, round) in study.rounds.iter().enumerate() {
            result.push_str(&format!(
                "\n[[rounds]]\nindex = {}\nround_ms = {:.6}\nobjective_calls = 1\nsequence_scores = 2\ntransformer_passes = 4\nincumbent_nll = {:.9}\ncandidate_nll = {:.9}\nincumbent_variance = {:.9}\ncandidate_variance = {:.9}\nacceptance_threshold = {:.9}\nacceptance_margin = {:.9}\naccepted = {}\nradius = {:.9}\ntrust_length = {:.9}\nsuccess_counter = {}\nfailure_counter = {}\nrestarts = {}\nallocated_bytes = {}\n",
                index + 1,
                round.round_ms,
                round.incumbent_nll,
                round.candidate_nll,
                round.incumbent_variance,
                round.candidate_variance,
                round.acceptance_threshold,
                round.acceptance_margin,
                round.accepted,
                round.radius,
                round.trust_length,
                round.success_counter,
                round.failure_counter,
                round.restarts,
                round.allocated_bytes,
            ));
        }
        std::fs::write(artifact_dir.join("result.toml"), result)
            .map_err(|error| error.to_string())?;
        eprintln!(
            "TURBO_ENN_RESULT mean_round_ms={:.3} max_round_ms={:.3} target_round_ms={} goal_met={goal_met}",
            study.mean_round_ms,
            study.max_round_ms(),
            run.target_round_ms()
        );
        Ok(())
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (run, artifact_dir);
        Err("the TuRBO-ENN round study requires Apple silicon macOS".to_string())
    }
}
