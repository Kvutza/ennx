fn main() {
    if let Err(error) = run() {
        eprintln!("TuRBO-ENN worker: {error}");
        std::process::exit(1);
    }
}

#[cfg(target_os = "macos")]
struct PretrainRep {
    index: u32,
    proposal_seed: u64,
    acquisition_seed: u64,
    median_wall_ms: f64,
    median_gpu_ms: f64,
    max_wall_ms: f64,
    accepted: u32,
    goal_met: bool,
}

#[cfg(target_os = "macos")]
fn median(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut values = values.into_iter().collect::<Vec<_>>();
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    }
}

#[cfg(target_os = "macos")]
fn write_pretrain_rep(
    run: &ennx::config::ConfigOverrides,
    artifact_dir: &std::path::Path,
    rep: u32,
    result: &ennx::experimental::ActualBoResult,
) -> Result<PretrainRep, String> {
    let enn = run.resident_enn(run.acquisition_seed_for_rep(rep))?;
    let ask = enn.ask;
    let length = run.length();
    let shape = run
        .trust_region_shape
        .unwrap_or(ennx::config::TrustRegionShape::TensorFamilyStatic);
    let proposal_seed = run.proposal_seed_for_rep(rep);
    let acquisition_seed = run.acquisition_seed_for_rep(rep);
    let goal_met = result.max_wall_seconds <= f64::from(run.target_round_ms()) / 1000.0;
    let updates_path = artifact_dir.join("tensor-updates.jsonl");
    result.write_updates(&updates_path)?;
    eprintln!("[weights] per-tensor records: {}", updates_path.display());
    let controller_path = artifact_dir.join("controller.jsonl");
    result.write_controller(&controller_path)?;
    eprintln!("[controller] round records: {}", controller_path.display());
    std::fs::write(
        artifact_dir.join("result.toml"),
        format!(
            "status = \"completed\"\nstage = \"candidate_applied_bo\"\nrep = {}\nreps = {}\nkernel = \"metal4_tensorops_m128_n64_plus_pisa1_simdgroup_matrix\"\nattention_shape = \"pisa1_q8_kv1_d64_c64_k8\"\nqkv_width = 640\nproposal = {:?}\ntrust_region_shape = {:?}\ncontroller = {:?}\nparameters = {}\nsearch_dimensions = {}\nlogical_history_capacity = 128\nresident_weight_rows = 2\nacquisition = {:?}\nk_neighbors = {}\nk_neighbors_role = \"configured_maximum\"\nfit_neighbors = {}\ndistance_scaling = {:?}\nlocal_scale_neighbors = {}\nepistemic_scale = {}\naleatoric_scale = {}\ny_scale = {}\nbeta = {}\nlength_init = {}\nlength_min = {}\nlength_max = {}\nproposal_seed = {}\nacquisition_seed = {}\ndistance = \"resident_exact_nonresident_approximate\"\ndata = \"causal_pretraining_paired_block128_ennxptn1\"\ndiagnostics = false\nloop_seconds = {:.9}\nactual_bo_gpu_median_ms = {:.6}\nactual_bo_wall_median_ms = {:.6}\nactual_bo_wall_min_ms = {:.6}\nactual_bo_wall_max_ms = {:.6}\nactual_bo_accepted = {}\ntarget_round_ms = {}\nfull_round_goal_met = {goal_met}\n",
            rep + 1,
            run.reps(),
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
            proposal_seed,
            acquisition_seed,
            result.loop_seconds,
            result.median_gpu_seconds * 1000.0,
            result.median_wall_seconds * 1000.0,
            result.min_wall_seconds * 1000.0,
            result.max_wall_seconds * 1000.0,
            result.accepted,
            run.target_round_ms(),
        ),
    )
    .map_err(|error| error.to_string())?;
    Ok(PretrainRep {
        index: rep + 1,
        proposal_seed,
        acquisition_seed,
        median_wall_ms: result.median_wall_seconds * 1000.0,
        median_gpu_ms: result.median_gpu_seconds * 1000.0,
        max_wall_ms: result.max_wall_seconds * 1000.0,
        accepted: result.accepted,
        goal_met,
    })
}

#[cfg(target_os = "macos")]
fn run_pretrain_reps(
    run: &ennx::config::ConfigOverrides,
    artifact_dir: &std::path::Path,
) -> Result<(), String> {
    let dataset = run
        .dataset()
        .ok_or("pretrain study was not resolved to an immutable dataset")?;
    let mut repetitions = Vec::with_capacity(run.reps() as usize);
    for rep in 0..run.reps() {
        let output = if run.reps() == 1 {
            artifact_dir.to_path_buf()
        } else {
            let output = artifact_dir.join(format!("rep-{:03}", rep + 1));
            std::fs::create_dir(&output).map_err(|error| error.to_string())?;
            output
        };
        let proposal_seed = run.proposal_seed_for_rep(rep);
        let acquisition_seed = run.acquisition_seed_for_rep(rep);
        eprintln!(
            "TURBO_ENN_REP rep={} reps={} proposal_seed={proposal_seed} acquisition_seed={acquisition_seed}",
            rep + 1,
            run.reps()
        );
        let mut repetition = run.clone();
        repetition.reps = Some(1);
        repetition.proposal_seed = Some(proposal_seed);
        repetition.acquisition_seed = Some(acquisition_seed);
        let result = ennx::experimental::run_pretrain(&repetition, dataset)?;
        let summary = write_pretrain_rep(run, &output, rep, &result)?;
        eprintln!(
            "TURBO_ENN_SUMMARY rep={} reps={} rounds={} median_seconds={:.9} max_seconds={:.9} accepted={} target_ms={} target_met={}",
            summary.index,
            run.reps(),
            run.rounds(),
            summary.median_wall_ms / 1000.0,
            summary.max_wall_ms / 1000.0,
            summary.accepted,
            run.target_round_ms(),
            summary.goal_met,
        );
        repetitions.push(summary);
    }
    if repetitions.len() == 1 {
        return Ok(());
    }
    let median_wall_ms = median(repetitions.iter().map(|rep| rep.median_wall_ms));
    let median_gpu_ms = median(repetitions.iter().map(|rep| rep.median_gpu_ms));
    let max_wall_ms = repetitions
        .iter()
        .map(|rep| rep.max_wall_ms)
        .max_by(f64::total_cmp)
        .ok_or("pretraining produced no repetitions")?;
    let goal_met = repetitions.iter().all(|rep| rep.goal_met);
    let mut aggregate = format!(
        "status = \"completed\"\nstage = \"repeated_candidate_applied_bo\"\nreps = {}\nrounds_per_rep = {}\nmedian_rep_wall_ms = {median_wall_ms:.6}\nmedian_rep_gpu_ms = {median_gpu_ms:.6}\nmax_round_wall_ms = {max_wall_ms:.6}\ntarget_round_ms = {}\nall_reps_goal_met = {goal_met}\n",
        run.reps(),
        run.rounds(),
        run.target_round_ms(),
    );
    for rep in &repetitions {
        aggregate.push_str(&format!(
            "\n[[repetitions]]\nindex = {}\nproposal_seed = {}\nacquisition_seed = {}\nmedian_wall_ms = {:.6}\nmedian_gpu_ms = {:.6}\nmax_wall_ms = {:.6}\naccepted = {}\ngoal_met = {}\nartifact = {:?}\n",
            rep.index,
            rep.proposal_seed,
            rep.acquisition_seed,
            rep.median_wall_ms,
            rep.median_gpu_ms,
            rep.max_wall_ms,
            rep.accepted,
            rep.goal_met,
            format!("rep-{:03}", rep.index),
        ));
    }
    std::fs::write(artifact_dir.join("result.toml"), aggregate)
        .map_err(|error| error.to_string())?;
    Ok(())
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
            return run_pretrain_reps(&run, &artifact_dir);
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
