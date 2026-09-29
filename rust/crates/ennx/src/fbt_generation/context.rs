use super::*;

/// Complete BO timing with the declared number of newly generated tokens.
/// This is a systems probe; concatenated sequences do not establish long-code quality.
pub fn context_loop(
    run: &ConfigOverrides,
    dataset: &Path,
    output: &Path,
    generated: u32,
    prompt: u32,
) -> Result<(), String> {
    run.validate_experiment()?;
    if generated == 0 || generated > crate::context::MAX_TOKENS || prompt == 0 {
        return Err(
            "generation workload requires 1..1048576 new tokens and a nonempty prompt".into(),
        );
    }
    let count = prompt
        .checked_add(generated)
        .ok_or("generation length overflow")?;
    let positions = count
        - u32::from(
            run.generation
                .as_ref()
                .is_none_or(|config| config.draft.is_none()),
        );
    let context = positions
        .checked_next_power_of_two()
        .ok_or("context capacity overflow")?
        .max(4096);
    crate::context::Layout::new(context, 4096)?;
    let dataset_tokens = crate::pretrain_data::PretrainDataset::load(dataset)?;
    let source = dataset_tokens.prefix(count as usize)?;
    let tokenizer = ByteDecoder::load(
        &dataset
            .parent()
            .ok_or("dataset has no parent")?
            .join("tokenizer.json"),
    )?;
    let mut run = run.clone();
    let config = run
        .generation
        .as_mut()
        .ok_or("context loop requires generation settings")?;
    config.max_tokens = generated;
    let prompt = prompt as usize;
    config.corpus_prompt.clear();
    config.corpus_prompt_tokens = None;
    config.episode_dataset = None;
    config.tasks = vec![GenerationTask {
        prompt: source[..prompt]
            .iter()
            .map(|&token| u32::from(token))
            .collect(),
        expected: source[prompt..]
            .iter()
            .map(|&token| u32::from(token))
            .collect(),
        decoys: Vec::new(),
    }];
    if let GenerationReward::CodeContrastive { decoys, .. }
    | GenerationReward::DraftedCode { decoys, .. } = config.reward
    {
        let length = (count as usize)
            .checked_add(
                decoys
                    .checked_mul(generated as usize)
                    .ok_or("decoy length overflow")?,
            )
            .ok_or("decoy length overflow")?;
        let negative = dataset_tokens.prefix(length)?;
        config.tasks[0].decoys = negative[count as usize..]
            .chunks_exact(generated as usize)
            .map(|tokens| tokens.iter().map(|&token| u32::from(token)).collect())
            .collect();
    }
    config.save_final_checkpoint = false;
    config.save_checkpoint = None;
    config.validate()?;
    run_config(
        &run,
        run.generation.as_ref().unwrap(),
        output,
        ennx_wire::json::json!({"kind":"packed-corpus-context-systems-probe", "dataset":dataset,
            "context_tokens":positions,"cache_capacity":context,"source_token_positions":count,"prompt_tokens":prompt,
            "generated_tokens":generated,"repeated_corpus":false,"long_context_quality_established":false}),
        Some(&tokenizer),
        None,
    )
}
