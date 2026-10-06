use super::generation::*;

pub(super) struct Resources {
    pub(super) runtime: std::sync::Arc<Runtime>,
    pub(super) decoder: decode::Decoder,
    pub(super) verifier: Option<block_decode::BlockDecoder>,
    pub(super) evaluator: Option<Evaluator>,
    pub(super) context: u32,
    pub(super) validation_tasks: Vec<GenerationTask>,
}

pub(super) fn prepare_resources(
    config: &GenerationConfig,
    output: &Path,
    objective: &ennx_wire::json::Value,
    tokenizer: Option<&ByteDecoder>,
    architecture: ResidualArchitecture,
) -> Result<Resources, String> {
    let runtime = Runtime::shared()?;
    let patch = architecture.patch_size();
    let raw_positions = config
        .tasks
        .iter()
        .map(|task| {
            task.prompt.len() + config.max_tokens as usize - usize::from(config.draft.is_none())
        })
        .max()
        .ok_or("generation requires tasks")?;
    let core_positions = raw_positions.div_ceil(patch);
    let context = core_positions.next_power_of_two().max(CONTEXT as usize) as u32;
    let visits = config
        .draft
        .map(|diffusion| {
            crate::forward_program::RecurrentCore::selective_fbt()
                .layer_visits(MODEL_LAYERS as usize, diffusion.visits[1] as usize)
                .map(|steps| steps.len().max(architecture.layer_steps().len()))
        })
        .transpose()?
        .unwrap_or_else(|| architecture.layer_steps().len());
    let decoder = decode::Decoder::with_visits(
        &runtime,
        context,
        context > ROWS || config.draft.is_some(),
        visits,
    )?;
    let verify = config.draft.is_some()
        || match config.verify.mode {
            VerifyMode::Auto => config.max_tokens >= 32 && config.tasks.len() == 1,
            VerifyMode::Serial => false,
            VerifyMode::AcceptedPrefix => true,
        };
    if context > ROWS && !verify {
        return Err("contexts above 8192 require chunked accepted-prefix verification".into());
    }
    if verify && config.tasks.len() != 1 {
        return Err("accepted-prefix verification requires exactly one task".into());
    }
    let verifier = verify
        .then(|| block_decode::BlockDecoder::with_context(&runtime, context))
        .transpose()?;
    let mut evaluator = Evaluator::new(&config.reward, &config.tasks, tokenizer)?;
    let validation_tasks: Vec<GenerationTask> = objective
        .get("validation_tasks")
        .cloned()
        .map(ennx_wire::json::from_value)
        .transpose()
        .map_err(|e| e.to_string())?
        .unwrap_or_default();
    if let Some(evaluator) = evaluator.as_mut() {
        let mut audit_tasks = config.tasks.clone();
        audit_tasks.extend(validation_tasks.iter().cloned());
        if let Some(audit) = evaluator.audit(&audit_tasks, tokenizer)? {
            ennx_wire::json::pretty_writer(
                File::create(output.join("reward-controls.json")).map_err(|e| e.to_string())?,
                &audit,
            )
            .map_err(|e| e.to_string())?;
            if audit["passed"] != true {
                return Err("generation reward failed reference-versus-negative controls; reward-controls.json retained".into());
            }
            eprintln!(
                "ENNX_REWARD_CONTROLS passed=true tasks={} functional_correctness_established=false",
                audit_tasks.len()
            );
        }
    }
    Ok(Resources {
        runtime,
        decoder,
        verifier,
        evaluator,
        context,
        validation_tasks,
    })
}

pub(super) fn reference_controls(
    initial: &Evaluation,
    config: &GenerationConfig,
    output: &Path,
    evaluator: &mut Option<Evaluator>,
    tokenizer: Option<&ByteDecoder>,
) -> Result<(Option<Vec<f32>>, Option<ennx_wire::json::Value>), String> {
    let reference_reward = if let Some(evaluator) = evaluator.as_mut().and_then(Evaluator::frozen) {
        let reference_path = output.join("corpus-reference");
        std::fs::create_dir(&reference_path).map_err(|e| e.to_string())?;
        let references = initial
            .rollouts
            .iter()
            .zip(&config.tasks)
            .map(|(rollout, task)| {
                let mut reference = rollout.clone();
                reference.tokens = task.expected.clone();
                reference
            })
            .collect::<Vec<_>>();
        let rewards = evaluator.score(
            tokenizer.ok_or("missing corpus tokenizer")?,
            &config.tasks,
            &references,
            &reference_path,
        )?;
        eprintln!("ENNX_CORPUS_REFERENCE rewards={rewards:?} used_for_acceptance=false");
        Some(rewards)
    } else {
        None
    };
    let reward_audit = reference_reward
        .as_ref()
        .map(|reference| {
            let audit = reference_audit(&initial.rewards, reference, &initial.diagnostics)?;
            ennx_wire::json::pretty_writer(
                File::create(output.join("reward-audit.json")).map_err(|e| e.to_string())?,
                &audit,
            )
            .map_err(|e| e.to_string())?;
            eprintln!(
                "ENNX_REWARD_AUDIT status={} learning_quality_established=false",
                audit["status"].as_str().unwrap_or("unknown")
            );
            Ok::<_, String>(audit)
        })
        .transpose()?;
    Ok((reference_reward, reward_audit))
}
