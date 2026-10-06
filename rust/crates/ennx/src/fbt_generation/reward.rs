use super::generation::*;

pub(super) fn evaluate(
    config: &GenerationConfig,
    rollouts: &[decode::Rollout],
    diagnostics: &[ennx_wire::json::Value],
    path: &Path,
    evaluator: Option<&mut Evaluator>,
    tokenizer: Option<&ByteDecoder>,
) -> Result<RewardResult, String> {
    if matches!(config.reward, GenerationReward::DraftedCode { .. }) {
        return drafted(rollouts, config, path, evaluator, tokenizer);
    }
    if matches!(config.reward, GenerationReward::CodeExecution { .. }) {
        return evaluator.ok_or("missing execution evaluator")?.execute(
            tokenizer.ok_or("execution requires a tokenizer")?,
            rollouts,
            path,
        );
    }
    if matches!(config.reward, GenerationReward::CodeObjectives { .. }) {
        let likelihood = rollouts
            .iter()
            .map(|rollout| {
                rollout
                    .free_running_target_nll
                    .map(|loss| -loss)
                    .ok_or("block decoder did not produce free-running target NLL")
            })
            .collect::<Result<Vec<_>, _>>()?;
        let reconstruction = evaluator
            .ok_or("missing code reconstruction evaluator")?
            .score(tokenizer, &config.tasks, rollouts, path)?;
        let critical = rollouts
            .iter()
            .map(|rollout| {
                rollout
                    .target_quality
                    .map(|quality| -quality.worst_window_nll)
                    .ok_or("block decoder did not produce localized target loss")
            })
            .collect::<Result<Vec<_>, _>>()?;
        return protocol::native_vector(
            reconstruction.clone(),
            &[likelihood, critical, reconstruction],
            &[
                "mean_target_likelihood",
                "worst_window_target_likelihood",
                "byte_reconstruction",
            ],
        );
    }
    if let GenerationReward::CommandObjectives {
        program,
        args,
        timeout_ms,
    } = &config.reward
    {
        let response = command_response(
            config,
            rollouts,
            diagnostics,
            path,
            program,
            args,
            *timeout_ms,
            "ennx.generation_objectives.v2",
        )?;
        return protocol::parse_vector(&response, config.tasks.len());
    }
    let rewards = scalar_reward(config, rollouts, diagnostics, path, evaluator, tokenizer)?;
    if rewards.len() != config.tasks.len() || rewards.iter().any(|score| !score.is_finite()) {
        return Err("evaluator must return one finite reward per task; higher is better".into());
    }
    Ok(RewardResult {
        rewards,
        vector: None,
    })
}

fn drafted(
    rollouts: &[decode::Rollout],
    config: &GenerationConfig,
    path: &Path,
    evaluator: Option<&mut Evaluator>,
    tokenizer: Option<&ByteDecoder>,
) -> Result<RewardResult, String> {
    let quality = evaluator.ok_or("missing generated-code scorer")?.score(
        tokenizer,
        &config.tasks,
        rollouts,
        path,
    )?;
    let agreement = rollouts
        .iter()
        .map(|rollout| {
            let draft = rollout
                .draft
                .as_ref()
                .ok_or("missing learned draft evidence")?;
            Ok(draft.matching_tokens as f32 / rollout.tokens.len().max(1) as f32)
        })
        .collect::<Result<Vec<_>, String>>()?;
    let work = rollouts
        .iter()
        .map(|rollout| {
            -(rollout.evaluated_positions as f32 / rollout.committed_tokens.max(1) as f32)
        })
        .collect::<Vec<_>>();
    protocol::native_vector(
        quality.clone(),
        &[quality, agreement, work],
        &[
            "generated-code",
            "draft-agreement",
            "negative-position-amplification",
        ],
    )
}

fn free_running_cross_entropy_reward(
    rollouts: &[decode::Rollout],
    tokenizer: Option<&ByteDecoder>,
) -> Result<Vec<f32>, String> {
    rollouts
        .iter()
        .map(|rollout| {
            let nll = rollout
                .free_running_target_nll
                .map(|loss| -loss)
                .ok_or("block decoder did not produce free-running target NLL")?;
            let bonus = match tokenizer {
                Some(tok) => {
                    let sample_len = rollout.tokens.len().min(4096);
                    let bytes = tok.decode_bytes(&rollout.tokens[..sample_len]).unwrap_or_default();
                    let text = String::from_utf8_lossy(&bytes);
                    let report = crate::text::gemma4::evaluate_learnability(&text);
                    if report.learnable {
                        (report.repetition_ratio_4gram - 0.5) * 0.05
                    } else {
                        -0.1
                    }
                }
                None => 0.0,
            };
            Ok(nll + bonus)
        })
        .collect()
}

fn scalar_reward(
    config: &GenerationConfig,
    rollouts: &[decode::Rollout],
    diagnostics: &[ennx_wire::json::Value],
    path: &Path,
    evaluator: Option<&mut Evaluator>,
    tokenizer: Option<&ByteDecoder>,
) -> Result<Vec<f32>, String> {
    Ok(match &config.reward {
        GenerationReward::ExactMatch => config
            .tasks
            .iter()
            .zip(rollouts)
            .map(|(task, rollout)| {
                if task.expected == rollout.tokens {
                    1.0
                } else {
                    0.0
                }
            })
            .collect(),
        GenerationReward::TokenAccuracy => config
            .tasks
            .iter()
            .zip(rollouts)
            .map(|(task, rollout)| {
                let matches = task
                    .expected
                    .iter()
                    .zip(&rollout.tokens)
                    .filter(|(a, b)| a == b)
                    .count();
                matches as f32 / task.expected.len().max(rollout.tokens.len()) as f32
            })
            .collect(),
        GenerationReward::FreeRunningCrossEntropy => {
            free_running_cross_entropy_reward(rollouts, tokenizer)?
        }
        GenerationReward::FrozenQwen { .. }
        | GenerationReward::CodeReconstruction
        | GenerationReward::CodeContrastive { .. } => evaluator
            .ok_or("missing generation evaluator")?
            .score(tokenizer, &config.tasks, rollouts, path)?,
        GenerationReward::DraftedCode { .. } => {
            return Err("drafted-code requires vector scoring".into());
        }
        GenerationReward::Command {
            program,
            args,
            timeout_ms,
        } => command_reward(
            config,
            rollouts,
            diagnostics,
            path,
            program,
            args,
            *timeout_ms,
        )?,
        GenerationReward::CommandObjectives { .. } => unreachable!("vector command handled above"),
        GenerationReward::CodeObjectives { .. } => unreachable!("native vector handled above"),
        GenerationReward::CodeExecution { .. } => unreachable!("execution handled above"),
    })
}

fn command_reward(
    config: &GenerationConfig,
    rollouts: &[decode::Rollout],
    diagnostics: &[ennx_wire::json::Value],
    path: &Path,
    program: &Path,
    args: &[String],
    timeout_ms: u64,
) -> Result<Vec<f32>, String> {
    let response = command_response(
        config,
        rollouts,
        diagnostics,
        path,
        program,
        args,
        timeout_ms,
        "ennx.generation_reward.v1",
    )?;
    #[derive(Deserialize)]
    #[deser(deny_unknown_fields)]
    struct Response {
        rewards: Vec<f32>,
    }
    Ok(ennx_wire::json::from_reader::<Response, _>(
        File::open(response).map_err(|e| e.to_string())?,
    )
    .map_err(|e| format!("invalid reward response: {e}"))?
    .rewards)
}

fn command_response(
    config: &GenerationConfig,
    rollouts: &[decode::Rollout],
    diagnostics: &[ennx_wire::json::Value],
    path: &Path,
    program: &Path,
    args: &[String],
    timeout_ms: u64,
    schema: &str,
) -> Result<std::path::PathBuf, String> {
    let request = path.join("request.json");
    let response = path.join("response.json");
    let payload = ennx_wire::json::json!({
        "schema": schema,
        "tasks": config.tasks,
        "rollouts": rollouts,
        "diagnostics": diagnostics,
    });
    ennx_wire::json::to_writer(File::create(&request).map_err(|e| e.to_string())?, &payload)
        .map_err(|e| e.to_string())?;
    let log = File::create(path.join("evaluator.log")).map_err(|e| e.to_string())?;
    let mut child = Command::new(program)
        .args(args)
        .arg(&request)
        .arg(&response)
        .stdin(Stdio::null())
        .stdout(Stdio::from(log.try_clone().map_err(|e| e.to_string())?))
        .stderr(Stdio::from(log))
        .spawn()
        .map_err(|e| format!("reward evaluator: {e}"))?;
    let start = Instant::now();
    loop {
        if let Some(status) = child.try_wait().map_err(|e| e.to_string())? {
            if !status.success() {
                return Err(format!("reward evaluator exited {status}"));
            }
            break;
        }
        if start.elapsed() >= Duration::from_millis(timeout_ms) {
            let _ = child.kill();
            let _ = child.wait();
            return Err("reward evaluator timed out; candidate not accepted".into());
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    Ok(response)
}

use super::gen_protocol as protocol;
pub(super) use protocol::{RewardResult, VectorResult};
