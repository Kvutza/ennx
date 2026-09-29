//! Bounded selection ablation on the production billion-weight scorer.
//! Held-out scores never enter the controller. Random selection still uses
//! the same fitted ENN acceptance policy, isolating acquisition selection.

use super::*;
use ennx_wire::json::json;
use metal::objc::rc::autoreleasepool;
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::io::Write;

fn load_config(path: &str) -> Result<crate::config::ConfigOverrides, String> {
    let (run, _) = crate::config::load_tune(std::path::Path::new(path))?;
    run.validate_experiment()?;
    let enn = run.resident_enn(run.acquisition_seed())?;
    if run.experiment != Some(crate::config::TurboEnnExperiment::Pretrain)
        || !matches!(enn.ask.acquisition, crate::weights::AcquisitionKind::Ucb)
        || enn.ask.neighbors != 10
    {
        return Err("This bounded protocol requires pretraining, UCB and ten neighbors".into());
    }
    Ok(run)
}

fn score(
    runtime: &Runtime,
    pipelines: &Pipelines,
    tensorops: &TensorOpsPipelines,
    pisa1: &Pisa1,
    buffers: &Buffers,
    weights: &CandidateWeights,
    row: &Buffer,
) -> Result<(f32, f32, [f32; 2]), String> {
    let command = runtime.queue.new_command_buffer();
    objective_fused(
        command,
        pipelines,
        tensorops,
        pisa1,
        buffers,
        weights.row(row)?,
    )?;
    complete(command)?;
    sequence_objective(buffers).map(|stats| (stats.reward, stats.variance, stats.sequence_nlls))
}

#[allow(clippy::too_many_arguments)]
fn scaling_search(
    runtime: &Runtime,
    pipelines: &Pipelines,
    tensorops: &TensorOpsPipelines,
    pisa1: &Pisa1,
    buffers: &Buffers,
    weights: &CandidateWeights,
    run: &crate::config::ConfigOverrides,
    train: &crate::pretrain_data::PretrainDataset,
    rep: u32,
    scaling: crate::config::DistanceScaling,
) -> Result<(SearchState, crate::config::ResidentEnnConfig), String> {
    let (mut search, _) = weights.search_shaped(
        run.perturbation(),
        run.length(),
        run.trust_region_shape
            .unwrap_or(crate::config::TrustRegionShape::TensorFamilyStatic),
    )?;
    let acquisition_seed = run.derived_seed(rep, "scaling-acquisition", 0);
    let mut enn = run.resident_enn(acquisition_seed)?;
    enn.distance_scaling = scaling;
    search.configure_enn(enn)?;
    pretrain_batch(buffers, train, 0)?;
    let (value, variance, _) = score(
        runtime,
        pipelines,
        tensorops,
        pisa1,
        buffers,
        weights,
        &search.base_buffer(),
    )?;
    search.observe_initial(value, variance)?;
    for step in 0..enn.ask.neighbors - 1 {
        pretrain_batch(buffers, train, step as u32 % train.batches())?;
        let proposal_seed = run.derived_seed(rep, "scaling-proposal", step as u64);
        let row = search.begin_initial(proposal_seed, step % 4)?;
        let proposal = search.finish_ask()?;
        let (value, variance, _) =
            score(runtime, pipelines, tensorops, pisa1, buffers, weights, &row)?;
        search.tell_initial(&proposal, value, variance)?;
        search.sync()?;
    }
    Ok((search, enn))
}

#[test]
#[ignore = "billion-weight GPU experiment; requires explicit config and output paths"]
fn selection_ablation() -> Result<(), String> {
    let config = std::env::var("ENNX_ABLATION_CONFIG").map_err(|e| e.to_string())?;
    let output = std::env::var("ENNX_ABLATION_OUTPUT").map_err(|e| e.to_string())?;
    let run = load_config(&config)?;
    let train_path = run
        .dataset()
        .ok_or("use a resolved pretraining experiment.toml")?;
    let validation_path = train_path.with_file_name("validation.ennxptn");
    let train = crate::pretrain_data::PretrainDataset::load(train_path)?;
    let validation = crate::pretrain_data::PretrainDataset::load(&validation_path)?;
    let heldout_batches = 8.min(validation.batches());
    let rounds = 32u32;
    let seeds: [u64; 3] =
        std::array::from_fn(|rep| run.derived_seed(rep as u32, "selection-run", 0));
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&output)
        .map_err(|e| e.to_string())?;
    let mut emit = |record: ennx_wire::json::Value| -> Result<(), String> {
        eprintln!(
            "ABLATION {}",
            ennx_wire::json::to_string(&record).map_err(|e| e.to_string())?
        );
        ennx_wire::json::write_line(&mut file, &record).map_err(|e| e.to_string())?;
        file.flush().map_err(|e| e.to_string())
    };
    emit(json!({"kind": "protocol", "parameters": FULL_PARAMETERS,
        "rounds": rounds, "proposal_seeds": seeds, "heldout_batches": heldout_batches,
        "train": train_path, "validation": validation_path,
        "config_text": std::fs::read_to_string(&config).map_err(|e| e.to_string())?,
        "comparison": "ucb_vs_uniform_selection_same_enn_acceptance",
        "initialization": "shared_nine_rounds_plus_initial_observation",
        "pool_matching": "common_innovations_and_batches; weights_and_radii_can_diverge",
        "geometry": "two_exact_resident_rows_plus_implicit_history",
        "model_initialization": "fixed_CandidateWeights_initialization_across_all_runs"}))?;
    autoreleasepool(|| -> Result<(), String> {
        let runtime = Runtime::shared()?;
        let pipelines = Pipelines::new(&runtime)?;
        let tensorops = TensorOpsPipelines::new(&runtime)?;
        let buffers = Buffers::new(&runtime);
        let pisa1 = Pisa1::new(&runtime)?;
        let weights = CandidateWeights::new(&runtime);
        for (rep, seed) in seeds.into_iter().enumerate() {
            // Alternate run order to reduce systematic timing/order bias.
            let policies = if rep % 2 == 0 {
                ["enn", "random"]
            } else {
                ["random", "enn"]
            };
            for policy in policies {
                autoreleasepool(|| -> Result<(), String> {
                    let (mut search, _) = weights.search_shaped(
                        run.perturbation(),
                        run.length(),
                        run.trust_region_shape
                            .unwrap_or(crate::config::TrustRegionShape::TensorFamilyStatic),
                    )?;
                    let acquisition_seed = run.derived_seed(rep as u32, "selection-acquisition", 0);
                    let enn = run.resident_enn(acquisition_seed)?;
                    search.configure_enn(enn)?;
                    let choice_seed = run.derived_seed(rep as u32, "selection-policy", 0);
                    let mut rng = StdRng::seed_from_u64(choice_seed);
                    let choices: Vec<usize> = (0..rounds).map(|_| rng.gen_range(0..4)).collect();
                    pretrain_batch(&buffers, &train, 0)?;
                    let (value, variance, _) = score(
                        &runtime,
                        &pipelines,
                        &tensorops,
                        &pisa1,
                        &buffers,
                        &weights,
                        &search.base_buffer(),
                    )?;
                    search.observe_initial(value, variance)?;
                    let mut evaluate = |phase: &str, row: &Buffer| -> Result<(), String> {
                        let mut nlls = Vec::new();
                        for batch in 0..heldout_batches {
                            pretrain_batch(&buffers, &validation, batch)?;
                            nlls.extend(
                                score(
                                    &runtime, &pipelines, &tensorops, &pisa1, &buffers, &weights,
                                    row,
                                )?
                                .2,
                            );
                        }
                        emit(json!({"kind": "heldout", "seed": seed, "policy": policy,
                            "phase": phase, "sequence_nlls": nlls,
                            "mean_nll": nlls.iter().map(|&x| f64::from(x)).sum::<f64>() / nlls.len() as f64}))
                    };
                    evaluate("initial", &search.base_buffer())?;
                    // End the closure's mutable borrow of the artifact writer.
                    drop(evaluate);
                    for step in 0..rounds {
                        pretrain_batch(&buffers, &train, step % train.batches())?;
                        search.compact_history()?;
                        let history = search.history_len()?;
                        let initializing = history < enn.ask.neighbors;
                        let mut ask = enn.ask;
                        ask.neighbors = ask.neighbors.min(history);
                        ask.seed = run.derived_seed(
                            rep as u32,
                            "selection-acquisition",
                            u64::from(step) + 1,
                        );
                        let proposal_seed =
                            run.derived_seed(rep as u32, "selection-proposal", u64::from(step));
                        let started = Instant::now();
                        let row = if initializing {
                            search.begin_initial(proposal_seed, step as usize % 4)?
                        } else if policy == "random" {
                            search.begin_forced(proposal_seed, ask, choices[step as usize])?
                        } else {
                            search.begin_ask(1, 4, proposal_seed, ask)?
                        };
                        let proposal = search.finish_ask()?;
                        let (value, variance, scores) = score(
                            &runtime, &pipelines, &tensorops, &pisa1, &buffers, &weights, &row,
                        )?;
                        let decision = if initializing {
                            search.tell_initial(&proposal, value, variance)?
                        } else {
                            search.tell_modeled(&proposal, value, variance)?
                        };
                        if search.sync()? != vec![decision.accepted] {
                            return Err("ablation tell/sync mismatch".into());
                        }
                        emit(json!({"kind": "round", "seed": seed, "policy": policy,
                            "round": step + 1, "initializing": initializing,
                            "candidate_seed": proposal.seed, "radius": proposal.length,
                            "reward": value, "variance": variance, "sequence_nlls": scores,
                            "accepted": decision.accepted, "threshold": decision.threshold,
                            "incumbent_mean": decision.incumbent_value,
                            "trust_outcome": format!("{:?}", decision.trust_outcome),
                            "trust_length": search.length()?, "seconds": started.elapsed().as_secs_f64()}))?;
                    }
                    let mut nlls = Vec::new();
                    for batch in 0..heldout_batches {
                        pretrain_batch(&buffers, &validation, batch)?;
                        nlls.extend(
                            score(
                                &runtime,
                                &pipelines,
                                &tensorops,
                                &pisa1,
                                &buffers,
                                &weights,
                                &search.base_buffer(),
                            )?
                            .2,
                        );
                    }
                    emit(json!({"kind": "heldout", "seed": seed, "policy": policy,
                        "phase": "final", "sequence_nlls": nlls,
                        "mean_nll": nlls.iter().map(|&x| f64::from(x)).sum::<f64>() / nlls.len() as f64}))?;
                    Ok(())
                })?;
            }
        }
        emit(json!({"kind": "completed"}))?;
        Ok(())
    })
}

/// Counterfactual comparison at a shared history: score all four choices on
/// validation, then compare UCB to the exact expectation of uniform selection.
/// This complements trajectories, whose incumbents can diverge after selection.
#[test]
#[ignore = "billion-weight same-pool diagnostic; held-out scores never train the model"]
fn pool_ablation() -> Result<(), String> {
    let config = std::env::var("ENNX_ABLATION_CONFIG").map_err(|e| e.to_string())?;
    let output = std::env::var("ENNX_POOL_OUTPUT").map_err(|e| e.to_string())?;
    let run = load_config(&config)?;
    let train_path = run
        .dataset()
        .ok_or("use a resolved pretraining experiment.toml")?;
    let validation_path = train_path.with_file_name("validation.ennxptn");
    let train = crate::pretrain_data::PretrainDataset::load(train_path)?;
    let validation = crate::pretrain_data::PretrainDataset::load(&validation_path)?;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .map_err(|e| e.to_string())?;
    autoreleasepool(|| -> Result<(), String> {
        let runtime = Runtime::shared()?;
        let pipelines = Pipelines::new(&runtime)?;
        let tensorops = TensorOpsPipelines::new(&runtime)?;
        let buffers = Buffers::new(&runtime);
        let pisa1 = Pisa1::new(&runtime)?;
        let weights = CandidateWeights::new(&runtime);
        for rep in 0..3u32 {
            autoreleasepool(|| -> Result<(), String> {
                let seed = run.derived_seed(rep, "pool-run", 0);
                let (mut search, _) = weights.search_shaped(
                    run.perturbation(),
                    run.length(),
                    run.trust_region_shape
                        .unwrap_or(crate::config::TrustRegionShape::TensorFamilyStatic),
                )?;
                let enn = run.resident_enn(run.derived_seed(rep, "pool-acquisition", 0))?;
                search.configure_enn(enn)?;
                pretrain_batch(&buffers, &train, 0)?;
                let (value, variance, _) = score(
                    &runtime,
                    &pipelines,
                    &tensorops,
                    &pisa1,
                    &buffers,
                    &weights,
                    &search.base_buffer(),
                )?;
                search.observe_initial(value, variance)?;
                for step in 0..enn.ask.neighbors - 1 {
                    pretrain_batch(&buffers, &train, step as u32 % train.batches())?;
                    let proposal_seed = run.derived_seed(rep, "pool-proposal", step as u64);
                    let row = search.begin_initial(proposal_seed, step % 4)?;
                    let proposal = search.finish_ask()?;
                    let (value, variance, _) = score(
                        &runtime, &pipelines, &tensorops, &pisa1, &buffers, &weights, &row,
                    )?;
                    search.tell_initial(&proposal, value, variance)?;
                    search.sync()?;
                }
                let root = run.derived_seed(rep, "pool-proposal", enn.ask.neighbors as u64);
                let mut ask = enn.ask;
                ask.seed = run.derived_seed(rep, "pool-acquisition", 1);
                let mut incumbent_nlls = Vec::new();
                for batch in 0..8.min(validation.batches()) {
                    pretrain_batch(&buffers, &validation, batch)?;
                    incumbent_nlls.extend(
                        score(
                            &runtime,
                            &pipelines,
                            &tensorops,
                            &pisa1,
                            &buffers,
                            &weights,
                            &search.base_buffer(),
                        )?
                        .2,
                    );
                }
                search.begin_ask(1, 4, root, ask)?;
                let proposal = search.finish_ask()?;
                let mut all_scores = Vec::new();
                for candidate in 0..4 {
                    let row = search.diagnostic_row(&proposal, root, ask, candidate)?;
                    let mut nlls = Vec::new();
                    for batch in 0..8.min(validation.batches()) {
                        pretrain_batch(&buffers, &validation, batch)?;
                        nlls.extend(
                            score(
                                &runtime, &pipelines, &tensorops, &pisa1, &buffers, &weights, &row,
                            )?
                            .2,
                        );
                    }
                    all_scores.push(nlls);
                    eprintln!("SAME_POOL seed={seed} candidate={candidate} complete");
                }
                // Restore the actual selected row even though this diagnostic
                // deliberately ends before tell, with no validation feedback.
                search.diagnostic_row(&proposal, root, ask, proposal.index)?;
                let means: Vec<_> = all_scores
                    .iter()
                    .map(|v| v.iter().map(|&x| f64::from(x)).sum::<f64>() / v.len() as f64)
                    .collect();
                let random_mean = means.iter().sum::<f64>() / 4.0;
                let same_radius_mean =
                    (means[proposal.index % 2] + means[proposal.index % 2 + 2]) / 2.0;
                let record = json!({"kind": "same_pool", "seed": seed,
                    "parameters": FULL_PARAMETERS, "history": search.history_len()?,
                    "selected": proposal.index, "candidate_mean_nlls": means,
                    "candidate_sequence_nlls": all_scores,
                    "incumbent_sequence_nlls": incumbent_nlls,
                    "incumbent_mean_nll": incumbent_nlls.iter().map(|&x| f64::from(x)).sum::<f64>() / incumbent_nlls.len() as f64,
                    "enn_minus_uniform_expected_nll": means[proposal.index] - random_mean,
                    "enn_minus_same_radius_expected_nll": means[proposal.index] - same_radius_mean,
                    "scope": "first_guided_pool; fixed_validation; no_validation_feedback",
                    "config": config, "train": train_path, "validation": validation_path});
                ennx_wire::json::write_line(&mut file, &record).map_err(|e| e.to_string())?;
                file.flush().map_err(|e| e.to_string())?;
                eprintln!(
                    "SAME_POOL {}",
                    ennx_wire::json::to_string(&record).map_err(|e| e.to_string())?
                );
                Ok(())
            })?;
        }
        Ok(())
    })
}

/// Compare global and self-tuning geometry on the exact same first guided
/// candidate pool. Held-out validation selects neither policy and never enters
/// either search state.
#[test]
#[ignore = "billion-weight same-pool distance-scaling eval; requires fixed corpus"]
fn scaling_ablation() -> Result<(), String> {
    let config = std::env::var("ENNX_ABLATION_CONFIG").map_err(|e| e.to_string())?;
    let output = std::env::var("ENNX_SCALING_OUTPUT").map_err(|e| e.to_string())?;
    let run = load_config(&config)?;
    let train_path = run
        .dataset()
        .ok_or("use a resolved pretraining experiment.toml")?;
    let validation_path = train_path.with_file_name("validation.ennxptn");
    let train = crate::pretrain_data::PretrainDataset::load(train_path)?;
    let validation = crate::pretrain_data::PretrainDataset::load(&validation_path)?;
    let heldout_batches = 8.min(validation.batches());
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .map_err(|e| e.to_string())?;
    autoreleasepool(|| -> Result<(), String> {
        let runtime = Runtime::shared()?;
        let pipelines = Pipelines::new(&runtime)?;
        let tensorops = TensorOpsPipelines::new(&runtime)?;
        let buffers = Buffers::new(&runtime);
        let pisa1 = Pisa1::new(&runtime)?;
        let weights = CandidateWeights::new(&runtime);
        for rep in 0..3u32 {
            autoreleasepool(|| -> Result<(), String> {
                let seed = run.derived_seed(rep, "distance-scaling-run", 0);
                let (mut global, global_enn) = scaling_search(
                    &runtime,
                    &pipelines,
                    &tensorops,
                    &pisa1,
                    &buffers,
                    &weights,
                    &run,
                    &train,
                    rep,
                    crate::config::DistanceScaling::Global,
                )?;
                let (mut local, local_enn) = scaling_search(
                    &runtime,
                    &pipelines,
                    &tensorops,
                    &pisa1,
                    &buffers,
                    &weights,
                    &run,
                    &train,
                    rep,
                    crate::config::DistanceScaling::SelfTuning,
                )?;
                if global.history_len()? != local.history_len()? {
                    return Err("paired scaling histories have different lengths".into());
                }
                let root =
                    run.derived_seed(rep, "scaling-proposal", global_enn.ask.neighbors as u64);
                let mut global_ask = global_enn.ask;
                global_ask.seed = run.derived_seed(rep, "scaling-acquisition", 1);
                let mut local_ask = local_enn.ask;
                local_ask.seed = global_ask.seed;
                global.begin_ask(1, 4, root, global_ask)?;
                let global_pool = global.finish_ask()?;
                local.begin_ask(1, 4, root, local_ask)?;
                let local_pool = local.finish_ask()?;
                if global_pool.pool_keys() != local_pool.pool_keys() {
                    return Err("distance-scaling policies did not receive one pool".into());
                }
                let mut candidate_scores = Vec::new();
                let mut candidate_sequence_nlls = Vec::new();
                for candidate in 0..4 {
                    let row = global.diagnostic_row(&global_pool, root, global_ask, candidate)?;
                    let mut nlls = Vec::new();
                    for batch in 0..heldout_batches {
                        pretrain_batch(&buffers, &validation, batch)?;
                        nlls.extend(
                            score(
                                &runtime, &pipelines, &tensorops, &pisa1, &buffers, &weights, &row,
                            )?
                            .2,
                        );
                    }
                    candidate_scores
                        .push(nlls.iter().map(|&x| f64::from(x)).sum::<f64>() / nlls.len() as f64);
                    candidate_sequence_nlls.push(nlls);
                }
                let uniform = candidate_scores.iter().sum::<f64>() / candidate_scores.len() as f64;
                let global_selected = global_pool.index;
                let local_selected = local_pool.index;
                let record = json!({
                    "kind": "distance_scaling_same_pool",
                    "schema": "ennx.production_optimizer_eval.v1",
                    "seed": seed,
                    "parameters": FULL_PARAMETERS,
                    "history": global.history_len()?,
                    "heldout_batches": heldout_batches,
                    "global_selected": global_selected,
                    "self_tuning_selected": local_selected,
                    "selection_changed": global_selected != local_selected,
                    "candidate_mean_nlls": candidate_scores,
                    "candidate_sequence_nlls": candidate_sequence_nlls,
                    "self_tuning_minus_global_nll": candidate_scores[local_selected] - candidate_scores[global_selected],
                    "global_minus_uniform_nll": candidate_scores[global_selected] - uniform,
                    "self_tuning_minus_uniform_nll": candidate_scores[local_selected] - uniform,
                    "local_scale_neighbors": local_enn.local_scale_neighbors,
                    "scope": "first_guided_pool; common_candidates; fixed_validation; no_validation_feedback",
                    "config": config,
                    "train": train_path,
                    "validation": validation_path,
                });
                ennx_wire::json::write_line(&mut file, &record).map_err(|e| e.to_string())?;
                file.flush().map_err(|e| e.to_string())?;
                eprintln!(
                    "DISTANCE_SCALING {}",
                    ennx_wire::json::to_string(&record).map_err(|e| e.to_string())?
                );
                Ok(())
            })?;
        }
        Ok(())
    })
}
