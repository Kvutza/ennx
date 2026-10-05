use super::generation::*;

pub(super) struct FinishReport {
    pub(super) experiment_start: Instant,
    pub(super) initial_validation: Vec<f32>,
    pub(super) final_validation: Vec<f32>,
    pub(super) reference_reward: Option<Vec<f32>>,
    pub(super) reward_audit: Option<ennx_wire::json::Value>,
}

pub(super) fn finish_run(
    run: &ConfigOverrides,
    config: &GenerationConfig,
    output: &Path,
    tokenizer: Option<&ByteDecoder>,
    state: &mut LoopState,
    report: FinishReport,
) -> Result<(), String> {
    let FinishReport {
        experiment_start,
        initial_validation,
        final_validation,
        reference_reward,
        reward_audit,
    } = report;
    let completion = state
        .incumbent_rollouts
        .first()
        .ok_or("generation produced no final completion")?;
    ennx_wire::json::pretty_writer(
        File::create(output.join("completion.tokens.json")).map_err(|e| e.to_string())?,
        &ennx_wire::json::json!({
            "schema": "ennx.generated_completion.v1",
            "tokens": completion.tokens,
            "token_count": completion.tokens.len(),
            "finish_reason": completion.finish_reason,
            "free_running_target_nll": completion.free_running_target_nll,
        }),
    )
    .map_err(|e| e.to_string())?;
    decoded_completion(tokenizer, completion, output)?;
    if let Some(path) = &config.save_checkpoint {
        let row = state.search.base_buffer();
        let architecture =
            ResidualArchitecture::from_model(run.model.ok_or("generation model is required")?);
        save_checkpoint(&row, path, architecture)?;
    }
    state.walls.sort_by(f64::total_cmp);
    let maximum = *state.walls.last().ok_or("no generation rounds")?;
    let middle = state.walls.len() / 2;
    let median = (state.walls[(state.walls.len() - 1) / 2] + state.walls[middle]) * 0.5;
    let latency_target_met = maximum * 1000.0 <= f64::from(run.target_ms());
    let effective_rounds = state.minimum_changed_weights > 0;
    let generation_target_met = state.minimum_generated_tokens == config.max_tokens as usize;
    let parameters =
        ResidualArchitecture::from_model(run.model.ok_or("generation model is required")?)
            .parameter_count();
    let stage = if config.draft.is_some() {
        "diffusion_draft_target_bo"
    } else if config.purpose == crate::config::GenerationPurpose::Pretrain {
        "free_running_pretraining_bo"
    } else {
        "generated_reward_bo"
    };
    let result = ennx_wire::json::json!({"status":"completed","stage":stage,"parameters":parameters,
        "rounds":run.rounds(),"accepted":state.accepted,"loop_seconds":state.loop_seconds,
        "elapsed_seconds":experiment_start.elapsed().as_secs_f64(),
        "generation_in_loop":true,
        "median_wall_ms":median*1000.0,"max_wall_ms":maximum*1000.0,
        "target_round_ms":run.target_ms(),"latency_target_met":latency_target_met,
        "minimum_changed_weights":state.minimum_changed_weights,"effective_rounds":effective_rounds,
        "generation_target_met":generation_target_met,
        "minimum_generated_tokens":state.minimum_generated_tokens,
        "target_met":latency_target_met && effective_rounds && generation_target_met,
        "generation_engine":if config.draft.is_some() {"diffusion-draft-causal-target"} else {"causal-target"},
        "scored_output":"target-completion",
        "timing_scope":"proposal-generation-scoring-tell-completion-write-controller-write",
        "timing_excludes":["setup","initial-incumbent","curve-write","console-display","heldout-validation","final-checkpoint"],
        "validation_initial":initial_validation,"validation_final":final_validation,
        "incumbent_temperature":state.temperature,"temperature_searched":config.temperature_bounds.is_some()});
    let mut result = result;
    let prompt = config
        .tasks
        .iter()
        .map(|task| task.prompt.len())
        .max()
        .ok_or("missing prompt")?;
    result["prompt_tokens"] = ennx_wire::json::json!(prompt);
    result["retained_context_tokens"] =
        ennx_wire::json::json!(prompt + state.minimum_generated_tokens);
    result["target_input_positions"] =
        ennx_wire::json::json!(prompt + config.max_tokens as usize - 1);
    result["draft_input_positions"] =
        ennx_wire::json::json!(config.draft.map(|_| prompt + config.max_tokens as usize));
    result["generated_tokens_per_candidate"] = ennx_wire::json::json!(config.max_tokens);
    result["generation_uses_reference_tokens"] = ennx_wire::json::json!(false);
    result["objective_uses_reference_tokens"] = ennx_wire::json::json!(matches!(
        config.reward,
        GenerationReward::DraftedCode { .. }
            | GenerationReward::CodeContrastive { .. }
            | GenerationReward::CodeReconstruction
            | GenerationReward::CodeObjectives { .. }
            | GenerationReward::ExactMatch
            | GenerationReward::TokenAccuracy
            | GenerationReward::FreeRunningCrossEntropy
    ));
    result["incumbent_control"] =
        ennx_wire::json::to_value(state.search.best()?).map_err(|e| e.to_string())?;
    result["purpose"] = ennx_wire::json::to_value(config.purpose).map_err(|e| e.to_string())?;
    result["initialization"] =
        ennx_wire::json::to_value(config.initialization).map_err(|e| e.to_string())?;
    result["corpus_reference_reward"] =
        ennx_wire::json::to_value(reference_reward).map_err(|e| e.to_string())?;
    result["reward_audit"] = reward_audit.unwrap_or(ennx_wire::json::Value::null());
    ennx_wire::json::pretty_writer(
        File::create(output.join("result.json")).map_err(|e| e.to_string())?,
        &result,
    )
    .map_err(|e| e.to_string())?;
    eprintln!(
        "ENNX_GENERATION_SUMMARY {}",
        ennx_wire::json::to_string(&result).map_err(|e| e.to_string())?
    );
    Ok(())
}
