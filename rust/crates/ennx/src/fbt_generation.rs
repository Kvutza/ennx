//! Generate/evaluate/tell with the existing full-coordinate resident ENN.
pub(super) use super::causal::{CausalEvidence, CausalLog, CausalScorer};
pub(super) use super::generation_reward::{Evaluator, reference_audit};
pub(super) use super::*;

#[path = "fbt_generation/context.rs"]
mod context;
pub(super) use crate::config::{
    ConfigOverrides, GenerationConfig, GenerationReward, GenerationTask, VerifyMode,
};
pub(super) use crate::text::ByteDecoder;
use crate::traits::Oracle;
pub use context::context_loop;
pub(super) use deser::{Deserialize, Serialize};
pub(super) use std::collections::BTreeMap;
pub(super) use std::fs::{File, OpenOptions};
pub(super) use std::io::Write;
pub(super) use std::path::Path;
pub(super) use std::process::{Command, Stdio};
pub(super) use std::time::Duration;

pub(super) use super::gen_metrics::{RankAudit, rollout_diagnostics};
pub(super) use super::{
    gen_checkpoint as checkpoint, gen_finish as finish, gen_journal as journal,
    gen_oracle::{CausalObjective, FbtOracle, observe_base},
    gen_reward as reward, gen_rounds as rounds, gen_setup as setup,
};
pub(super) use checkpoint::*;
pub(super) use finish::*;
pub(super) use reward::evaluate;
pub(super) use rounds::*;
pub(super) use setup::*;

#[derive(Serialize)]
pub(super) struct Evaluation {
    pub(super) rollouts: Vec<decode::Rollout>,
    pub(super) rewards: Vec<f32>,
    pub(super) mean: f32,
    pub(super) variance: f32,
    #[deser(skip_serializing)]
    pub(super) observation: crate::objective_observation::ObjectiveObservation,
    #[deser(skip_serializing_if = Option::is_none)]
    pub(super) objectives: Option<reward::VectorResult>,
    pub(super) reward_seconds: f64,
    pub(super) diagnostics: Vec<ennx_wire::json::Value>,
    #[deser(skip_serializing_if = Option::is_none)]
    pub(super) causal: Option<CausalEvidence>,
}

pub(super) struct Validation {
    pub(super) rewards: Vec<f32>,
    pub(super) objectives: Vec<Vec<f32>>,
}

impl Validation {
    pub(super) fn objective_means(&self) -> Vec<f32> {
        let Some(width) = self.objectives.first().map(Vec::len) else {
            return Vec::new();
        };
        let mut means = vec![0.0; width];
        for row in &self.objectives {
            for (mean, value) in means.iter_mut().zip(row) {
                *mean += *value;
            }
        }
        for mean in &mut means {
            *mean /= self.objectives.len() as f32;
        }
        means
    }
}

impl Evaluation {
    pub(super) fn annotate(&self, record: &mut ennx_wire::json::Value) -> Result<(), String> {
        if let Some(objectives) = &self.objectives {
            record["objectives"] =
                ennx_wire::json::to_value(objectives).map_err(|error| error.to_string())?;
        }
        Ok(())
    }
}

pub(super) fn decoded_completion(
    tokenizer: Option<&ByteDecoder>,
    rollout: &decode::Rollout,
    path: &Path,
) -> Result<Option<String>, String> {
    let Some(tokenizer) = tokenizer else {
        return Ok(None);
    };
    let bytes = tokenizer.decode_bytes(&rollout.tokens)?;
    let text = String::from_utf8_lossy(&bytes).into_owned();
    std::fs::write(path.join("completion.bin"), &bytes).map_err(|error| error.to_string())?;
    std::fs::write(path.join("completion.txt"), &text).map_err(|error| error.to_string())?;
    Ok(Some(text))
}

pub(super) fn display_completion(header: &str, text: &str) {
    eprintln!("\n{header}");
    let max_preview = 4_000;
    let preview = if text.len() > max_preview {
        &text[..text.floor_char_boundary(max_preview)]
    } else {
        text
    };
    let mut chunk = String::with_capacity(4_000);
    for character in preview.chars() {
        chunk.push(character);
        if character == '\n' || chunk.len() >= 4_000 {
            eprint!("ENNX_GENERATED_TEXT_CHUNK {chunk}");
            if character != '\n' {
                eprintln!();
            }
            chunk.clear();
        }
    }
    if !chunk.is_empty() {
        eprintln!("ENNX_GENERATED_TEXT_CHUNK {chunk}");
    }
    if text.len() > max_preview {
        eprintln!(
            "ENNX_GENERATED_TEXT_CHUNK \n... [preview capped at {} chars; full completion preserved on disk] ...\n",
            max_preview
        );
    }
    eprintln!("ENNX_GENERATED_TEXT_END");
}

pub(super) fn score(
    runtime: &Runtime,
    decoder: &decode::Decoder,
    verifier: Option<(&block_decode::BlockDecoder, Option<&[decode::Rollout]>)>,
    row: CandidateRow<'_>,
    config: &GenerationConfig,
    seed: u64,
    path: &Path,
    evaluator: Option<&mut Evaluator>,
    tokenizer: Option<&ByteDecoder>,
) -> Result<Evaluation, String> {
    std::fs::create_dir(path).map_err(|e| e.to_string())?;
    let rollouts = if let Some((verifier, drafts)) = verifier {
        if let Some(drafts) = drafts {
            verifier.verify(runtime, decoder, row, &config.tasks, config, seed, drafts)?
        } else {
            verifier.generate(runtime, decoder, row, &config.tasks, config, seed)?
        }
    } else {
        config
            .tasks
            .iter()
            .enumerate()
            .map(|(index, task)| {
                decoder.generate(
                    runtime,
                    row,
                    task,
                    config,
                    crate::hash::splitmix64(seed ^ index as u64),
                )
            })
            .collect::<Result<Vec<_>, _>>()?
    };
    let diagnostics = rollouts.iter().map(rollout_diagnostics).collect::<Vec<_>>();
    let start = Instant::now();
    let reward = evaluate(config, &rollouts, &diagnostics, path, evaluator, tokenizer)?;
    let observation = reward.observation()?;
    let control = observation.control();
    let result = Evaluation {
        diagnostics,
        rollouts,
        rewards: reward.rewards,
        mean: control.mean,
        variance: control.variance,
        observation,
        objectives: reward.vector,
        reward_seconds: start.elapsed().as_secs_f64(),
        causal: None,
    };
    ennx_wire::json::pretty_writer(
        File::create(path.join("evaluation.json")).map_err(|e| e.to_string())?,
        &result,
    )
    .map_err(|e| e.to_string())?;
    Ok(result)
}

pub(super) fn attach_causal(
    evaluation: &mut Evaluation,
    evidence: CausalEvidence,
    vector_control: bool,
) -> Result<(), String> {
    if !vector_control || !evaluation.observation.is_vector() {
        evaluation.mean = evidence.optimizer_value;
        evaluation.variance = evidence.variance;
        evaluation.observation = crate::objective_observation::ObjectiveObservation::scalar(
            evidence.optimizer_value,
            evidence.variance,
        )?;
    }
    evaluation.causal = Some(evidence);
    Ok(())
}

fn rank_audit(
    config: &GenerationConfig,
    initial: &Evaluation,
) -> Result<Option<RankAudit>, String> {
    if !matches!(&config.reward, GenerationReward::CodeObjectives { .. }) {
        return Ok(None);
    }
    let nll = initial
        .rollouts
        .first()
        .and_then(|rollout| rollout.free_running_target_nll)
        .ok_or("code objectives require initial free-running target NLL")?;
    Ok(Some(RankAudit::new(-nll, initial.mean)))
}

pub fn run_generation(run: &ConfigOverrides, output: &Path) -> Result<(), String> {
    run.validate_experiment()?;
    let config = run
        .generation
        .as_ref()
        .ok_or("missing generation configuration")?;
    if let GenerationReward::CodeExecution { environment, .. } = &config.reward {
        let spec = super::code_environment::Environment::load(environment)?;
        let tokenizer = ByteDecoder::load(&spec.tokenizer)?;
        let mut config = config.clone();
        let task = spec.task(&tokenizer)?;
        if !config.tasks.is_empty() && config.tasks != [task.clone()] {
            return Err("code_execution tasks must match the environment prompt exactly".into());
        }
        config.tasks = vec![task];
        if config.save_final_checkpoint && config.save_checkpoint.is_none() {
            config.save_checkpoint = Some(output.join("checkpoint.safetensors"));
        }
        config.validate()?;
        return run_config(
            run,
            &config,
            output,
            spec.provenance(),
            Some(&tokenizer),
            None,
        );
    }
    if config.tasks.is_empty() {
        return Err("standalone generation requires explicit tasks".into());
    }
    run_config(
        run,
        config,
        output,
        ennx_wire::json::json!({"kind":"explicit_generation_tasks"}),
        None,
        None,
    )
}

pub fn run_generated(
    run: &ConfigOverrides,
    dataset_path: &Path,
    output: &Path,
) -> Result<(), String> {
    run.validate_experiment()?;
    let mut config = run
        .generation
        .clone()
        .ok_or("missing generated-pretraining configuration")?;
    let tokenizer_path = dataset_path
        .parent()
        .ok_or("generated-pretraining dataset has no parent directory")?
        .join("tokenizer.json");
    let tokenizer = ByteDecoder::load(&tokenizer_path)?;
    let task_seed = run.derived_seed(0, "generated-pretrain-task", 0);
    let coding = matches!(
        config.reward,
        GenerationReward::CodeReconstruction
            | GenerationReward::CodeContrastive { .. }
            | GenerationReward::DraftedCode { .. }
            | GenerationReward::CodeObjectives { .. }
    );
    let decoy_count = match config.reward {
        GenerationReward::CodeContrastive { decoys, .. }
        | GenerationReward::DraftedCode { decoys, .. } => decoys,
        _ => 0,
    };
    let mut decoys = Vec::new();
    let mut validation_tasks = Vec::<GenerationTask>::new();
    let (prompt, expected, sequence_index, episode) = if let Some(path) = &config.episode_dataset {
        let document: ennx_wire::json::Value =
            ennx_wire::json::from_reader(File::open(path).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
        if document["schema"] != "ennx.document_episodes.v2" {
            return Err("unsupported generation episode schema".into());
        }
        let episodes = episode_rows(&document, "train", coding)?;
        let index = (task_seed % episodes.len() as u64) as usize;
        let episode = episodes[index].clone();
        let heldout = episode_rows(&document, "validation", coding)?;
        let first =
            (run.derived_seed(0, "generated-validation-task", 0) % heldout.len() as u64) as usize;
        for offset in 0..heldout.len().min(2) {
            validation_tasks.push(episode_task(
                &heldout,
                (first + offset) % heldout.len(),
                decoy_count,
            )?);
        }
        let task = episode_task(&episodes, index, decoy_count)?;
        decoys = task.decoys;
        (task.prompt, task.expected, index as u32, Some(episode))
    } else {
        let (p, e, idx) = corpus_continuation_task(&mut config, dataset_path, task_seed, coding)?;
        (p, e, idx, None)
    };
    if expected.len() != config.max_tokens as usize {
        return Err(format!(
            "generated pretraining requires max_tokens={}, matching the corpus sequence; found {}",
            expected.len(),
            config.max_tokens
        ));
    }
    config.corpus_prompt.clear();
    config.corpus_prompt_tokens = None;
    config.episode_dataset = None;
    config.tasks = vec![GenerationTask {
        prompt,
        expected,
        decoys,
    }];
    if config.save_final_checkpoint && config.save_checkpoint.is_none() {
        config.save_checkpoint = Some(output.join("checkpoint.safetensors"));
    }
    config.validate()?;
    run_config(
        run,
        &config,
        output,
        ennx_wire::json::json!({
            "kind":"free_running_corpus_continuation",
            "dataset":dataset_path,
            "sequence_index":sequence_index,
            "task_seed":task_seed,
            "episode":episode,
            "validation_tasks":validation_tasks,
        }),
        Some(&tokenizer),
        Some(dataset_path),
    )
}

fn corpus_continuation_task(
    config: &mut GenerationConfig,
    dataset_path: &Path,
    task_seed: u64,
    coding: bool,
) -> Result<(Vec<u32>, Vec<u32>, u32), String> {
    if coding {
        return Err("code_reconstruction requires document-aligned corpus_prompt_tokens; packed cross-file targets are not coding tasks".into());
    }
    let dataset = crate::pretrain_data::PretrainDataset::load(dataset_path)?;
    let prompt_len = if let Some(tokens) = config.corpus_prompt_tokens {
        tokens as usize
    } else if config.corpus_prompt.is_empty() {
        config.max_tokens as usize
    } else {
        config.corpus_prompt.len()
    };
    if config.max_tokens > crate::pretrain_data::CONTEXT
        || prompt_len > crate::pretrain_data::CONTEXT as usize
    {
        let total = prompt_len + config.max_tokens as usize;
        let source = dataset.prefix(total)?;
        let p = if !config.corpus_prompt.is_empty() {
            std::mem::take(&mut config.corpus_prompt)
        } else {
            source[..prompt_len].iter().map(|&t| u32::from(t)).collect()
        };
        let e = source[prompt_len..total].iter().map(|&t| u32::from(t)).collect();
        Ok((p, e, 0))
    } else {
        let index = (task_seed % u64::from(dataset.sequences())) as u32;
        let expected = dataset
            .sequence(index)?
            .iter()
            .map(|&token| u32::from(token))
            .collect::<Vec<_>>();
        let prompt = if let Some(tokens) = config.corpus_prompt_tokens {
            expected[..tokens as usize].to_vec()
        } else {
            std::mem::take(&mut config.corpus_prompt)
        };
        Ok((prompt, expected, index))
    }
}

pub(super) fn episode_task(
    rows: &[&ennx_wire::json::Value],
    index: usize,
    count: usize,
) -> Result<GenerationTask, String> {
    let target = rows[index];
    let mut decoys = Vec::new();
    let mut documents = std::collections::HashSet::new();
    for offset in 1..rows.len() {
        let row = rows[(index + offset) % rows.len()];
        let key = (
            row["pool"].as_str().unwrap_or_default(),
            row["document_offset"].as_u64().unwrap_or_default(),
        );
        if row["expected"] != target["expected"]
            && (row["pool"] != target["pool"]
                || row["document_offset"] != target["document_offset"])
            && documents.insert(key)
            && decoys.len() < count
        {
            decoys.push(row["expected"].clone());
        }
    }
    if decoys.len() != count {
        return Err(format!(
            "episode has only {} distinct-document decoys; requires {count}",
            decoys.len()
        ));
    }
    ennx_wire::json::from_value(ennx_wire::json::json!({"prompt":target["prompt"],"expected":target["expected"],"decoys":decoys})).map_err(|e| e.to_string())
}

pub(super) fn episode_rows<'a>(
    document: &'a ennx_wire::json::Value,
    split: &str,
    coding: bool,
) -> Result<Vec<&'a ennx_wire::json::Value>, String> {
    let rows = ennx_wire::json::pointer(document, &format!("/splits/{split}/episodes"))
        .and_then(|value| value.as_seq())
        .ok_or_else(|| format!("missing {split} generation episodes"))?;
    let rows = rows
        .iter()
        .filter(|row| !coding || matches!(row["bucket"].as_str(), Some("implementation" | "tests")))
        .collect::<Vec<_>>();
    if rows.is_empty() {
        return Err(format!("no eligible {split} generation episodes"));
    }
    Ok(rows)
}

pub(super) fn write_config(
    run: &ConfigOverrides,
    config: &GenerationConfig,
    output: &Path,
    objective: &ennx_wire::json::Value,
    runtime: &Runtime,
    context: u32,
    model_seed: u64,
    proposal_seed: u64,
    acquisition_seed: u64,
    sampling_seed: u64,
) -> Result<(), String> {
    let parameters =
        ResidualArchitecture::from_model(run.model.ok_or("generation model is required")?)
            .parameter_count();
    let resolved = ennx_wire::json::json!({"model_seed":model_seed,"proposal_seed":proposal_seed,
        "acquisition_seed":acquisition_seed,"sampling_seed":sampling_seed,"config":config,
        "objective":objective,
        "selection":run.selection.unwrap_or_default(),"random_is_selection_ablation":true,
        "history_geometry":run.history_geometry.unwrap_or_default().name(),
        "generation_in_loop":true,
        "parameters":parameters,"cache_policy":"rebuild_per_candidate_per_task",
        "device":runtime.info().name,
        "low_power_mode":crate::apple_gpu::power_mode().ok(),
        "student_cache_context":context,
        "position_encoding":{"kind":"rope","base":ROPE_BASE,"dimensions":HEAD_DIM},
        "sampling_policy":"fixed_common_random_numbers_across_candidates",
        "backend":"native_metal_mlx_derived_gemvt_pisa",
        "timing":"proposal+prefill+decode+reward+tell+artifact_writes; initial incumbent excluded"});
    ennx_wire::json::pretty_writer(
        File::create(output.join("generation.json")).map_err(|e| e.to_string())?,
        &resolved,
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

pub(super) fn display_initial(
    tokenizer: Option<&ByteDecoder>,
    initial: &Evaluation,
    output: &Path,
) -> Result<(), String> {
    if let Some(text) =
        decoded_completion(tokenizer, &initial.rollouts[0], &output.join("initial"))?
    {
        display_completion(
            &format!(
                "ENNX_GENERATED_TEXT phase=initial tokens={} reward={}",
                initial.rollouts[0].tokens.len(),
                initial.mean,
            ),
            &text,
        );
    }
    Ok(())
}

pub(super) fn validate_initial(
    runtime: &Runtime,
    decoder: &decode::Decoder,
    verifier: Option<&block_decode::BlockDecoder>,
    weights: &CandidateWeights,
    buffer: &metal::Buffer,
    config: &GenerationConfig,
    sampling_seed: u64,
    output: &Path,
    evaluator: Option<&mut Evaluator>,
    tokenizer: Option<&ByteDecoder>,
    validation_tasks: &[GenerationTask],
    journal: &mut journal::Journal,
) -> Result<Vec<f32>, String> {
    let validation_start = Instant::now();
    let initial_validation = validate_generation(
        runtime,
        decoder,
        verifier,
        weights,
        buffer,
        config,
        sampling_seed,
        &output.join("validation-initial"),
        evaluator,
        tokenizer,
        validation_tasks,
    )?;
    journal.validation(
        0,
        &initial_validation,
        validation_start.elapsed().as_secs_f64(),
    )?;
    Ok(initial_validation.rewards)
}

#[derive(Clone, Copy)]
struct GenerationSeeds {
    sampling: u64,
    proposal: u64,
    acquisition: u64,
    model: u64,
}

impl GenerationSeeds {
    fn new(run: &ConfigOverrides, config: &GenerationConfig) -> Self {
        Self {
            sampling: config
                .seed
                .unwrap_or_else(|| run.derived_seed(0, "sampling", 0)),
            proposal: run.proposal_seed(),
            acquisition: run.acquisition_seed(),
            model: run.model_seed(),
        }
    }
}

fn initial_causal(
    scorer: Option<&CausalScorer>,
    runtime: &Runtime,
    weights: &CandidateWeights,
    base: &metal::Buffer,
    initial: &mut Evaluation,
    vector_control: bool,
) -> Result<(), String> {
    let Some(scorer) = scorer else {
        return Ok(());
    };
    let evidence = scorer.initial(runtime, weights.row(base)?)?;
    attach_causal(initial, evidence, vector_control)
}

fn initialize_search(
    run: &ConfigOverrides,
    config: &GenerationConfig,
    runtime: &Runtime,
    seeds: GenerationSeeds,
) -> Result<
    (
        CandidateWeights,
        SearchState,
        UpdateLog,
        crate::config::ResidentEnnConfig,
    ),
    String,
> {
    let weights = CandidateWeights::initialized(
        runtime,
        seeds.model,
        config.initialization,
        ResidualArchitecture::from_model(run.model.ok_or("generation model is required")?),
    );
    if let Some(path) = &config.checkpoint {
        load_checkpoint(&weights, path)?;
    } else {
        eprintln!(
            "[generation] purpose={:?} initialization={:?}: no input checkpoint loaded",
            config.purpose, config.initialization
        );
    }
    let (mut search, updates) = weights.search_shaped(
        run.perturbation(),
        run.length(),
        run.trust_region_shape
            .unwrap_or(crate::config::TrustRegionShape::TensorFamilyStatic),
    )?;
    configure_temperature(&mut search, config)?;
    let policy = run.resident_enn(seeds.acquisition)?;
    search.configure_enn(policy)?;
    if let Some(controller) = run.reliability_controller() {
        search.configure_controller(controller)?;
    }
    Ok((weights, search, updates, policy))
}

struct InitialReport {
    observation: Evaluation,
    journal: journal::Journal,
    validation: Vec<f32>,
    reference_reward: Option<Vec<f32>>,
    reward_audit: Option<ennx_wire::json::Value>,
}

struct InitialContext<'a> {
    runtime: &'a Runtime,
    decoder: &'a decode::Decoder,
    verifier: Option<&'a block_decode::BlockDecoder>,
    weights: &'a CandidateWeights,
    base: &'a metal::Buffer,
    config: &'a GenerationConfig,
    sampling_seed: u64,
    output: &'a Path,
    evaluator: &'a mut Option<Evaluator>,
    tokenizer: Option<&'a ByteDecoder>,
    validation_tasks: &'a [GenerationTask],
    causal: Option<&'a CausalScorer>,
    causal_log: Option<&'a mut CausalLog>,
    vector_control: bool,
    experiment_start: Instant,
}

impl InitialContext<'_> {
    fn run(self) -> Result<InitialReport, String> {
        let Self {
            runtime,
            decoder,
            verifier,
            weights,
            base,
            config,
            sampling_seed,
            output,
            evaluator,
            tokenizer,
            validation_tasks,
            causal,
            causal_log,
            vector_control,
            experiment_start,
        } = self;
        let mut observation = observe_base(
            runtime,
            decoder,
            verifier,
            weights,
            base,
            config,
            sampling_seed,
            output,
            evaluator.as_mut(),
            tokenizer,
        )?;
        initial_causal(
            causal,
            runtime,
            weights,
            base,
            &mut observation,
            vector_control,
        )?;
        if let (Some(scorer), Some(log)) = (causal, causal_log) {
            log.observe(scorer, runtime, weights.row(base)?, 0)?;
        }
        display_initial(tokenizer, &observation, output)?;
        let (reference_reward, reward_audit) =
            reference_controls(&observation, config, output, evaluator, tokenizer)?;
        let mut journal = journal::Journal::new(output, experiment_start, &observation)?;
        let validation = validate_initial(
            runtime,
            decoder,
            verifier,
            weights,
            base,
            config,
            sampling_seed,
            output,
            evaluator.as_mut(),
            tokenizer,
            validation_tasks,
            &mut journal,
        )?;
        Ok(InitialReport {
            observation,
            journal,
            validation,
            reference_reward,
            reward_audit,
        })
    }
}

pub(super) fn run_config(
    run: &ConfigOverrides,
    config: &GenerationConfig,
    output: &Path,
    objective: ennx_wire::json::Value,
    tokenizer: Option<&ByteDecoder>,
    _dataset: Option<&Path>,
) -> Result<(), String> {
    let experiment_start = Instant::now();
    verify_policy(config)?;
    let Resources {
        runtime,
        decoder,
        verifier,
        mut evaluator,
        context,
        validation_tasks,
    } = prepare_resources(
        config,
        output,
        &objective,
        tokenizer,
        ResidualArchitecture::from_model(run.model.ok_or("generation model is required")?),
    )?;
    // Free-running generation objectives drive generated pretraining. The
    // corpus target is used by those objectives, never as a teacher-forced
    // optimization path.
    let mut causal: Option<CausalScorer> = None;
    let mut causal_log = causal
        .as_ref()
        .map(|_| CausalLog::new(output))
        .transpose()?;
    let seeds = GenerationSeeds::new(run, config);
    let (weights, mut search, updates, policy) = initialize_search(run, config, &runtime, seeds)?;
    write_config(
        run,
        config,
        output,
        &objective,
        &runtime,
        context,
        seeds.model,
        seeds.proposal,
        seeds.acquisition,
        seeds.sampling,
    )?;
    let base = search.base_buffer();
    let initial = InitialContext {
        runtime: &runtime,
        decoder: &decoder,
        verifier: verifier.as_ref(),
        weights: &weights,
        base: &base,
        config,
        sampling_seed: seeds.sampling,
        output,
        evaluator: &mut evaluator,
        tokenizer,
        validation_tasks: &validation_tasks,
        causal: causal.as_ref(),
        causal_log: causal_log.as_mut(),
        vector_control: run.objective_acquisition.is_some(),
        experiment_start,
    }
    .run()?;
    search.observe_objectives(initial.observation.observation.clone())?;
    search.configure_objectives(run.resident_objectives(0)?)?;
    let mut state = LoopState {
        updates,
        journal: initial.journal,
        records: OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(output.join("controller.jsonl"))
            .map_err(|e| e.to_string())?,
        walls: Vec::new(),
        minimum_changed_weights: u64::MAX,
        minimum_generated_tokens: usize::MAX,
        accepted: 0,
        incumbent_version: crate::hash::splitmix64(seeds.model ^ 0x656e_6e78_2d62_6173),
        incumbent_rollouts: initial.observation.rollouts.clone(),
        regions: RegionState::initial(
            &search,
            seeds.model,
            &initial.observation.rollouts,
            config.temperature,
        ),
        search,
        temperature: config.temperature,
        loop_seconds: 0.0,
        rank_audit: rank_audit(config, &initial.observation)?,
        causal_value: initial
            .observation
            .causal
            .map(|evidence| evidence.optimizer_value),
    };
    LoopContext {
        run,
        config,
        output,
        runtime: &runtime,
        decoder: &decoder,
        verifier: &verifier,
        weights: &weights,
        sampling_seed: seeds.sampling,
        tokenizer,
        evaluator: &mut evaluator,
        validation_tasks: &validation_tasks,
        policy,
        proposal_seed: seeds.proposal,
        acquisition_seed: seeds.acquisition,
        causal: &mut causal,
        causal_log: &mut causal_log,
    }
    .run(&mut state)?;
    state.finalize_state(config.record_tensor_updates, output)?;
    if let (Some(scorer), Some(log)) = (causal.as_ref(), causal_log.as_mut()) {
        let base = state.search.base_buffer();
        log.observe(scorer, &runtime, weights.row(&base)?, run.rounds())?;
    }
    let validation_start = Instant::now();
    let final_validation = validate_final(
        &runtime,
        &decoder,
        verifier.as_ref(),
        &weights,
        &state.search.base_buffer(),
        config,
        state.temperature,
        seeds.sampling,
        &output.join("validation-final"),
        evaluator.as_mut(),
        tokenizer,
        &validation_tasks,
    )?;
    state.journal.validation(
        run.rounds(),
        &final_validation,
        validation_start.elapsed().as_secs_f64(),
    )?;
    finish_run(
        run,
        config,
        output,
        tokenizer,
        &mut state,
        FinishReport {
            experiment_start,
            initial_validation: initial.validation,
            final_validation: final_validation.rewards,
            reference_reward: initial.reference_reward,
            reward_audit: initial.reward_audit,
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn validate_final(
    runtime: &Runtime,
    decoder: &decode::Decoder,
    verifier: Option<&block_decode::BlockDecoder>,
    weights: &CandidateWeights,
    buffer: &metal::Buffer,
    config: &GenerationConfig,
    temperature: f32,
    seed: u64,
    path: &Path,
    evaluator: Option<&mut Evaluator>,
    tokenizer: Option<&ByteDecoder>,
    tasks: &[GenerationTask],
) -> Result<Validation, String> {
    let mut config = config.clone();
    config.temperature = temperature;
    validate_generation(
        runtime, decoder, verifier, weights, buffer, &config, seed, path, evaluator, tokenizer,
        tasks,
    )
}

fn configure_temperature(
    search: &mut SearchState,
    config: &GenerationConfig,
) -> Result<(), String> {
    match (config.temperature_bounds, config.temperature_step) {
        (Some(bounds), Some(step)) => search.configure_axis(config.temperature, bounds, step),
        (None, None) => Ok(()),
        _ => Err("temperature search configuration was not validated".into()),
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn validate_generation(
    runtime: &Runtime,
    decoder: &decode::Decoder,
    verifier: Option<&block_decode::BlockDecoder>,
    weights: &CandidateWeights,
    buffer: &metal::Buffer,
    config: &GenerationConfig,
    seed: u64,
    path: &Path,
    mut evaluator: Option<&mut Evaluator>,
    tokenizer: Option<&ByteDecoder>,
    tasks: &[GenerationTask],
) -> Result<Validation, String> {
    if tasks.is_empty() {
        return Ok(Validation {
            rewards: Vec::new(),
            objectives: Vec::new(),
        });
    }
    std::fs::create_dir(path).map_err(|e| e.to_string())?;
    let mut rewards = Vec::new();
    let mut objectives = Vec::new();
    for (index, task) in tasks.iter().enumerate() {
        let mut evaluation_config = config.clone();
        evaluation_config.tasks = vec![task.clone()];
        let task_path = path.join(format!("task-{:04}", index + 1));
        let candidate =
            crate::search::DeviceView::metal(buffer.clone(), 0, buffer.length() as usize);
        let mut oracle = FbtOracle {
            runtime,
            decoder,
            weights,
            verifier: verifier.map(|verifier| (verifier, None)),
            config: &evaluation_config,
            seed,
            path: &task_path,
            evaluator: evaluator.as_deref_mut(),
            tokenizer,
            audit: false,
            causal: None,
            vector_control: false,
        };
        let (_, result) = oracle.observe(candidate)?;
        decoded_completion(tokenizer, &result.rollouts[0], &task_path)?;
        rewards.push(result.mean);
        if result.observation.estimates().len() > 1 {
            let row = result
                .observation
                .estimates()
                .iter()
                .map(|estimate| estimate.mean)
                .collect::<Vec<_>>();
            if objectives
                .first()
                .is_some_and(|previous: &Vec<f32>| previous.len() != row.len())
            {
                return Err("held-out objective width changed between tasks".into());
            }
            objectives.push(row);
        }
    }
    eprintln!(
        "ENNX_VALIDATION phase={} rewards={rewards:?} used_for_acceptance=false",
        path.file_name().unwrap_or_default().to_string_lossy()
    );
    if !objectives.is_empty() && objectives.len() != rewards.len() {
        return Err("held-out tasks disagree on scalar versus vector objectives".into());
    }
    Ok(Validation {
        rewards,
        objectives,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validation_means() {
        let validation = Validation {
            rewards: vec![0.2, 0.4],
            objectives: vec![vec![-8.0, -6.0, 0.25], vec![-10.0, -8.0, 0.75]],
        };
        assert_eq!(validation.objective_means(), [-9.0, -7.0, 0.5]);
    }

    struct TestOracle(Evaluation);

    impl Oracle for TestOracle {
        type Evidence = ();

        fn observe(
            &mut self,
            _candidate: crate::search::DeviceView<'_>,
        ) -> Result<
            (
                crate::objective_observation::ObjectiveObservation,
                Self::Evidence,
            ),
            String,
        > {
            Ok((self.0.observation, ()))
        }
    }

    #[test]
    fn checkpoint_shapes() {
        let layout = checkpoint_layout(ResidualArchitecture::Legacy);
        assert_eq!(layout.len(), 11);
        assert_eq!(
            layout
                .iter()
                .map(|(_, shape)| shape.iter().product::<usize>())
                .sum::<usize>(),
            FULL_PARAMETERS
        );
        let mut names = layout.iter().map(|(name, _)| *name).collect::<Vec<_>>();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), layout.len());

        let mhc = checkpoint_layout(ResidualArchitecture::Mhc4);
        assert_eq!(mhc.len(), 15);
        assert_eq!(
            mhc.iter()
                .map(|(_, shape)| shape.iter().product::<usize>())
                .sum::<usize>(),
            MHC_FULLPARAMS
        );
    }

    fn evaluated(control: f32, objective: f32) -> Evaluation {
        use crate::objective_observation::{ObjectiveEstimate, ObjectiveObservation};
        let estimate = |mean| ObjectiveEstimate {
            mean,
            variance: 0.0,
        };
        Evaluation {
            rollouts: Vec::new(),
            rewards: vec![900.0],
            mean: control,
            variance: 0.0,
            observation: ObjectiveObservation::new(
                &[estimate(objective), estimate(objective)],
                estimate(control),
            )
            .unwrap(),
            objectives: None,
            reward_seconds: 0.0,
            diagnostics: Vec::new(),
            causal: None,
        }
    }

    #[test]
    fn vector_routing() {
        metal::objc::rc::autoreleasepool(|| {
            let base = [0x3e80; 17];
            let block = crate::bf16_metal::ParamBlock::new(71, 0, base.len(), 0.25, 1.0).unwrap();
            let mut search = SearchState::new_implicit(
                &base,
                vec![block],
                3,
                crate::TRLengthConfig::new(0.1, 0.001, 0.4),
                crate::Perturbation::Gaussian,
            )
            .unwrap();
            let mut policy = ConfigOverrides::default().resident_enn(19).unwrap();
            policy.ask.neighbors = 2;
            search.configure_enn(policy).unwrap();
            search
                .observe_objectives(evaluated(0.0, 0.0).observation)
                .unwrap();
            search
                .configure_objectives(Some(crate::bf16_metal::ObjectivePolicy {
                    acquisition: crate::bf16_metal::ObjectiveAcquisition::Pareto,
                    scales: vec![1.0, 1.0],
                }))
                .unwrap();
            let (row, proposal, initializing) = search.propose(29, policy.ask, 0, None).unwrap();
            assert!(initializing);
            let candidate = crate::search::DeviceView::metal(row.clone(), 0, row.length() as usize);
            let mut accepted = TestOracle(evaluated(-1.0, 1000.0));
            let (decision, (), _) = search
                .evaluate(&proposal, candidate, true, &mut accepted)
                .unwrap();
            assert!(decision.accepted);
            let row = search.objective_observations().unwrap().last().unwrap();
            assert_eq!(row.observation, accepted.0.observation);
            let mut ask = policy.ask;
            ask.neighbors = 2;
            let (row, proposal, initializing) = search.propose(31, ask, 0, None).unwrap();
            assert!(!initializing);
            let control = proposal.incumbent_mean + 4.0 * proposal.incumbent_standard_error + 1.0;
            let candidate = crate::search::DeviceView::metal(row.clone(), 0, row.length() as usize);
            let mut rejected = TestOracle(evaluated(control, -1000.0));
            let (decision, (), _) = search
                .evaluate(&proposal, candidate, false, &mut rejected)
                .unwrap();
            assert!(!decision.accepted);
            assert_eq!(
                search.incumbent_objectives().unwrap().unwrap().observation,
                accepted.0.observation
            );
        });
    }

    #[test]
    fn causal_vector() {
        let mut evaluation = evaluated(2.0, 3.0);
        let observation = evaluation.observation;
        attach_causal(
            &mut evaluation,
            CausalEvidence {
                batch: 0,
                candidate_nll: 9.0,
                incumbent_nll: 9.1,
                improvement: 0.1,
                optimizer_value: -9.0,
                variance: 0.25,
                gpu_seconds: 0.0,
            },
            true,
        )
        .unwrap();
        assert_eq!(evaluation.observation, observation);
        assert_eq!(evaluation.mean, 2.0);
        assert_eq!(evaluation.causal.unwrap().optimizer_value, -9.0);
    }
}
