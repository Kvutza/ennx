//! Generated-token objectives. Prompts and rewards are explicit experiment data.
use super::feedback::FeedbackTransition;
use super::initialization::ModelInitialization;
use deser::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[deser(deny_unknown_fields, rename_all = "kebab-case")]
pub struct GenerationConfig {
    pub purpose: GenerationPurpose,
    #[deser(default)]
    pub initialization: ModelInitialization,
    #[deser(default)]
    pub feedback_transition: FeedbackTransition,
    pub max_tokens: u32,
    pub temperature: f32,
    /// Optional closed BO search interval. The scalar `temperature` remains the
    /// initial incumbent and fixed-temperature configurations remain unchanged.
    pub temperature_bounds: Option<[f32; 2]>,
    /// Initial trust-region step in temperature units when bounds are enabled.
    pub temperature_step: Option<f32>,
    pub eos_token: Option<u32>,
    pub seed: Option<u64>,
    pub checkpoint: Option<PathBuf>,
    pub qualification_manifest: Option<PathBuf>,
    pub save_checkpoint: Option<PathBuf>,
    #[deser(default = enabled())]
    pub save_final_checkpoint: bool,
    #[deser(default = enabled())]
    pub record_tensor_updates: bool,
    pub signal_gate: Option<SignalGate>,
    #[deser(default)]
    pub verify: VerifyConfig,
    pub draft: Option<super::DiffusionConfig>,
    pub reward: GenerationReward,
    #[deser(default)]
    pub corpus_prompt: Vec<u32>,
    pub corpus_prompt_tokens: Option<u32>,
    pub episode_dataset: Option<PathBuf>,
    #[deser(default)]
    pub tasks: Vec<GenerationTask>,
}

/// Execution policy for exact greedy generation. Accepted-prefix verification
/// commits only tokens proven equal to serial greedy decoding.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[deser(deny_unknown_fields, rename_all = "kebab-case")]
pub struct VerifyConfig {
    #[deser(default)]
    pub mode: VerifyMode,
    #[deser(default = verify_window())]
    pub window: u32,
    #[deser(default = verify_window())]
    pub max_window: u32,
    #[deser(default = verify_passes())]
    pub passes: u32,
    #[deser(default = verify_unroll())]
    pub unroll: u32,
    /// Run one independent serial decode for each evaluated BO candidate and
    /// require token-for-token equality with accepted-prefix verification.
    #[deser(default)]
    pub audit_candidate: bool,
}

impl Default for VerifyConfig {
    fn default() -> Self {
        Self {
            mode: VerifyMode::Auto,
            window: verify_window(),
            max_window: verify_window(),
            passes: verify_passes(),
            unroll: verify_unroll(),
            audit_candidate: false,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[deser(rename_all = "kebab-case")]
pub enum VerifyMode {
    /// Preserve the historical choice for configurations that omit `verify`.
    #[default]
    Auto,
    Serial,
    AcceptedPrefix,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[deser(rename_all = "kebab-case")]
pub enum GenerationPurpose {
    SystemsProbe,
    /// Learn from an untrained neural initialization, without a qualified policy.
    Pretrain,
    CodingOptimization,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[deser(deny_unknown_fields, rename_all = "kebab-case")]
pub struct GenerationTask {
    pub prompt: Vec<u32>,
    #[deser(default)]
    pub expected: Vec<u32>,
    #[deser(default)]
    pub decoys: Vec<Vec<u32>>,
}

fn enabled() -> bool {
    true
}

fn verify_window() -> u32 {
    128
}

fn verify_passes() -> u32 {
    2
}

fn verify_unroll() -> u32 {
    1
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[deser(deny_unknown_fields, rename_all = "kebab-case")]
pub struct SignalGate {
    pub rounds: u32,
    pub min_distinct_rewards: usize,
    pub min_reward_span: f32,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
#[deser(rename_all = "kebab-case")]
pub enum FrozenReadout {
    #[default]
    Reference,
    MpsFp32,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
#[deser(rename_all = "kebab-case")]
pub enum FrozenAttention {
    #[default]
    Reference,
    Tiled16,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[deser(rename_all = "kebab-case")]
pub enum FrozenBackend {
    Reference,
    Fp32,
    Fp16,
}

impl FrozenBackend {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Reference => "reference",
            Self::Fp32 => "fp32",
            Self::Fp16 => "fp16",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[deser(
    tag = "kind",
    rename_all = "kebab-case",
    rename_all_fields = "kebab-case",
    deny_unknown_fields
)]
pub enum GenerationReward {
    ExactMatch,
    TokenAccuracy,
    /// Exact normalized byte edit similarity of a freely generated code completion.
    CodeReconstruction,
    /// Clipped non-whitespace byte overlap with the target minus the strongest decoy.
    CodeContrastive {
        max_ngram: usize,
        decoys: usize,
    },
    /// Generated code overlap, draft/target agreement, and negative position amplification.
    DraftedCode {
        max_ngram: usize,
        decoys: usize,
    },
    /// Mean target NLL under the model's generated prefix, negated so higher is better.
    FreeRunningCrossEntropy,
    /// Target-reconstruction falsification probe. These channels are not a
    /// pretraining or coding-quality objective.
    CodeObjectives {
        critical_window: u32,
    },
    /// Execute the entire decoded completion against fixed input/output cases.
    CodeExecution {
        environment: PathBuf,
        interpreter: PathBuf,
        timeout_ms: u64,
    },
    /// Score the generated text under one immutable external Qwen checkpoint.
    FrozenQwen {
        checkpoint: PathBuf,
        tokenizer_program: PathBuf,
        #[deser(default)]
        tokenizer_args: Vec<String>,
        backend: FrozenBackend,
        #[deser(default)]
        readout: FrozenReadout,
        #[deser(default)]
        attention: FrozenAttention,
        max_tokens: u32,
        timeout_ms: u64,
    },
    /// Executable receives request and response JSON file paths as final arguments.
    Command {
        program: PathBuf,
        args: Vec<String>,
        timeout_ms: u64,
    },
    /// Versioned independent objective observations plus explicit scalar control.
    CommandObjectives {
        program: PathBuf,
        args: Vec<String>,
        timeout_ms: u64,
    },
}

impl GenerationConfig {
    pub(crate) fn causal_pretrain(&self) -> bool {
        self.purpose == GenerationPurpose::Pretrain
            && (self.corpus_prompt_tokens.is_some() || !self.corpus_prompt.is_empty())
    }

    pub fn validate(&self) -> Result<(), String> {
        if let Some(diffusion) = self.draft {
            diffusion.validate()?;
            if self.eos_token.is_some() || self.verify.audit_candidate {
                return Err("diffusion drafting requires full blocks; EOS and runtime serial auditing are unsupported".into());
            }
            if self.max_tokens % diffusion.block != 0
                || self
                    .tasks
                    .iter()
                    .any(|task| task.prompt.len() % diffusion.block as usize != 0)
                || self
                    .corpus_prompt_tokens
                    .is_some_and(|tokens| tokens % diffusion.block != 0)
                || !self
                    .corpus_prompt
                    .len()
                    .is_multiple_of(diffusion.block as usize)
            {
                return Err(
                    "diffusion prompt and generation lengths must align to the configured block"
                        .into(),
                );
            }
            if matches!(
                self.reward,
                GenerationReward::FreeRunningCrossEntropy | GenerationReward::CodeObjectives { .. }
            ) {
                return Err(
                    "diffusion drafting requires a generated-output reward, not target likelihood"
                        .into(),
                );
            }
        }
        self.validate_policy()?;
        self.validate_purpose()?;
        self.validate_source()
    }

    fn validate_purpose(&self) -> Result<(), String> {
        match self.purpose {
            GenerationPurpose::Pretrain => {
                if self.initialization == ModelInitialization::Patterned
                    || self.qualification_manifest.is_some()
                {
                    return Err(
                        "pretrain requires a neural initialization and no policy qualification"
                            .into(),
                    );
                }
            }
            GenerationPurpose::SystemsProbe => {
                if self.qualification_manifest.is_some() {
                    return Err("systems_probe must not claim a base-policy qualification".into());
                }
            }
            GenerationPurpose::CodingOptimization => {
                if self.checkpoint.is_none() || self.qualification_manifest.is_none() {
                    return Err(
                        "coding_optimization requires checkpoint and qualification_manifest".into(),
                    );
                }
            }
        }
        Ok(())
    }

    fn validate_source(&self) -> Result<(), String> {
        let explicit_tasks = !self.tasks.is_empty();
        let execution = matches!(self.reward, GenerationReward::CodeExecution { .. });
        let corpus_task = !self.corpus_prompt.is_empty() || self.corpus_prompt_tokens.is_some();
        if !self.corpus_prompt.is_empty() && self.corpus_prompt_tokens.is_some() {
            return Err("choose corpus_prompt or corpus_prompt_tokens".into());
        }
        if self
            .corpus_prompt_tokens
            .is_some_and(|count| count == 0 || count > crate::context::MAX_TOKENS)
        {
            return Err("corpus_prompt_tokens must be in 1..1048576".into());
        }
        if self.max_tokens == 0
            || self.max_tokens > crate::context::MAX_TOKENS
            || if execution {
                corpus_task || self.tasks.len() > 1
            } else {
                explicit_tasks == corpus_task
            }
            || !self.temperature.is_finite()
            || self.temperature < 0.0
            || self.eos_token.is_some_and(|token| token >= 8192)
            || self.seed.is_some_and(|seed| seed > i64::MAX as u64)
        {
            return Err("generation requires exactly one of explicit tasks or corpus_prompt, 1..1048576 tokens, finite nonnegative temperature, valid EOS and seed".into());
        }
        match (self.temperature_bounds, self.temperature_step) {
            (Some([lower, upper]), Some(step))
                if lower.is_finite()
                    && upper.is_finite()
                    && lower >= 0.0
                    && lower < upper
                    && (lower..=upper).contains(&self.temperature)
                    && step.is_finite()
                    && step > 0.0
                    && step <= upper - lower => {}
            (None, None) => {}
            _ => {
                return Err("generation temperature search requires bounds containing temperature and a positive step no larger than their span".into());
            }
        }
        if corpus_task
            && (self.corpus_prompt.len() + self.max_tokens as usize
                > crate::context::MAX_CONTEXT as usize + 1
                || self.corpus_prompt.iter().any(|&token| token >= 8192))
        {
            return Err(
                "generation corpus_prompt must fit the context and use vocabulary IDs 0..8191"
                    .into(),
            );
        }
        self.validate_tasks()?;
        if matches!(
            self.reward,
            GenerationReward::FreeRunningCrossEntropy | GenerationReward::CodeObjectives { .. }
        ) && (self.max_tokens < 32
            || self.eos_token.is_some()
            || self.verify.mode == VerifyMode::Serial
            || (explicit_tasks && self.tasks.len() != 1)
            || self
                .tasks
                .first()
                .is_some_and(|task| task.expected.len() != self.max_tokens as usize))
        {
            return Err("free_running_cross_entropy requires one task with max_tokens expected tokens, no early EOS, and accepted-prefix verification".into());
        }
        self.validate_command()
    }

    fn validate_policy(&self) -> Result<(), String> {
        let verify = self.verify;
        if verify.window < 128
            || verify.max_window < verify.window
            || verify.max_window > 4096
            || !verify.window.is_power_of_two()
            || !verify.max_window.is_power_of_two()
            || verify.passes > 8
            || !(1..=8).contains(&verify.unroll)
        {
            return Err("generation verify requires power-of-two window bounds in 128..4096, window <= max_window, and 1..8 broad and repair passes".into());
        }
        if verify.mode == VerifyMode::AcceptedPrefix && self.tasks.len() > 1 {
            return Err("accepted_prefix verification requires exactly one generation task".into());
        }
        if verify.audit_candidate && verify.mode != VerifyMode::AcceptedPrefix {
            return Err("candidate decode auditing requires accepted_prefix verification".into());
        }
        if let Some(gate) = &self.signal_gate {
            if gate.rounds == 0
                || gate.min_distinct_rewards < 2
                || gate.min_distinct_rewards > gate.rounds as usize + 1
                || !gate.min_reward_span.is_finite()
                || gate.min_reward_span <= 0.0
            {
                return Err(
                    "signal_gate requires positive rounds/span and at least two distinct rewards"
                        .into(),
                );
            }
        }
        if !self.save_final_checkpoint && self.save_checkpoint.is_some() {
            return Err("save_checkpoint conflicts with save_final_checkpoint=false".into());
        }
        if let GenerationReward::CodeContrastive { max_ngram, decoys }
        | GenerationReward::DraftedCode { max_ngram, decoys } = self.reward
        {
            if !(1..=4).contains(&max_ngram) || decoys == 0 {
                return Err(
                    "code_contrastive requires max_ngram in 1..4 and positive decoys".into(),
                );
            }
        }
        if matches!(self.reward, GenerationReward::DraftedCode { .. }) && self.draft.is_none() {
            return Err("drafted-code requires generation.draft".into());
        }
        Ok(())
    }

    fn validate_tasks(&self) -> Result<(), String> {
        for task in &self.tasks {
            if task.prompt.is_empty()
                || task.prompt.len() + self.max_tokens as usize
                    > crate::context::MAX_CONTEXT as usize + 1
                || task
                    .prompt
                    .iter()
                    .chain(&task.expected)
                    .chain(task.decoys.iter().flatten())
                    .any(|&token| token >= 8192)
            {
                return Err("generation prompt must be nonempty, fit context with output, and use vocabulary IDs 0..8191".into());
            }
            if matches!(
                self.reward,
                GenerationReward::ExactMatch
                    | GenerationReward::TokenAccuracy
                    | GenerationReward::CodeReconstruction
                    | GenerationReward::CodeContrastive { .. }
                    | GenerationReward::DraftedCode { .. }
                    | GenerationReward::FreeRunningCrossEntropy
                    | GenerationReward::CodeObjectives { .. }
            ) && (task.expected.is_empty() || task.expected.len() > self.max_tokens as usize)
            {
                return Err(
                    "token rewards require a nonempty expected completion within max_tokens".into(),
                );
            }
            if let GenerationReward::CodeContrastive { decoys, .. }
            | GenerationReward::DraftedCode { decoys, .. } = self.reward
            {
                if task.decoys.len() != decoys || task.decoys.iter().any(Vec::is_empty) {
                    return Err(
                        "code_contrastive task requires the configured nonempty decoys".into(),
                    );
                }
            }
        }
        if let GenerationReward::CodeObjectives { critical_window } = &self.reward
            && (*critical_window == 0 || *critical_window > self.max_tokens)
        {
            return Err("code_objectives critical_window must be in 1..=max_tokens".into());
        }
        Ok(())
    }

    fn validate_command(&self) -> Result<(), String> {
        if let GenerationReward::CodeExecution {
            environment,
            interpreter,
            timeout_ms,
        } = &self.reward
            && (environment.as_os_str().is_empty()
                || interpreter.as_os_str().is_empty()
                || *timeout_ms == 0
                || *timeout_ms > 10_000)
        {
            return Err(
                "code_execution requires environment, interpreter and timeout_ms in 1..=10000"
                    .into(),
            );
        }
        if let GenerationReward::Command {
            program,
            timeout_ms,
            ..
        }
        | GenerationReward::CommandObjectives {
            program,
            timeout_ms,
            ..
        } = &self.reward
        {
            if program.as_os_str().is_empty() || *timeout_ms == 0 {
                return Err("reward command requires a program and positive timeout_ms".into());
            }
        }
        if let GenerationReward::FrozenQwen {
            checkpoint,
            tokenizer_program,
            max_tokens,
            timeout_ms,
            ..
        } = &self.reward
        {
            if checkpoint.as_os_str().is_empty()
                || tokenizer_program.as_os_str().is_empty()
                || *max_tokens < 2
                || *max_tokens > 32768
                || *timeout_ms == 0
            {
                return Err("frozen_qwen requires checkpoint, tokenizer_program, max_tokens in 2..32768, and positive timeout_ms".into());
            }
        }
        Ok(())
    }

    pub fn resolve(&mut self, parent: &std::path::Path) -> Result<(), String> {
        if let GenerationReward::CodeExecution {
            environment,
            interpreter,
            ..
        } = &mut self.reward
        {
            for path in [environment, interpreter] {
                *path = parent
                    .join(&*path)
                    .canonicalize()
                    .map_err(|error| error.to_string())?;
            }
        }
        if let Some(path) = &mut self.episode_dataset {
            *path = parent
                .join(&*path)
                .canonicalize()
                .map_err(|error| error.to_string())?;
        }
        if let Some(path) = &mut self.save_checkpoint {
            *path = parent.join(&*path);
            if path.exists() {
                return Err("save_checkpoint already exists; refusing to overwrite".into());
            }
        }
        if let Some(path) = &mut self.checkpoint {
            *path = parent
                .join(&*path)
                .canonicalize()
                .map_err(|error| error.to_string())?;
        }
        if let Some(path) = &mut self.qualification_manifest {
            *path = parent
                .join(&*path)
                .canonicalize()
                .map_err(|error| error.to_string())?;
        }
        if let GenerationReward::Command { program, .. }
        | GenerationReward::CommandObjectives { program, .. } = &mut self.reward
        {
            *program = parent
                .join(&*program)
                .canonicalize()
                .map_err(|error| error.to_string())?;
        }
        if let GenerationReward::FrozenQwen {
            checkpoint,
            tokenizer_program,
            ..
        } = &mut self.reward
        {
            *checkpoint = parent
                .join(&*checkpoint)
                .canonicalize()
                .map_err(|error| error.to_string())?;
            *tokenizer_program = parent
                .join(&*tokenizer_program)
                .canonicalize()
                .map_err(|error| error.to_string())?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vector_config(policy: &str) -> crate::config::ConfigOverrides {
        ennx_wire::toml::from_str(&format!(
            r#"
[generation]
purpose = "systems-probe"
max-tokens = 16
temperature = 0.0
[[generation.tasks]]
prompt = [1]
[generation.reward]
kind = "command-objectives"
program = "score"
args = []
timeout-ms = 10
{policy}
"#
        ))
        .unwrap()
    }

    #[test]
    fn vector_policy() {
        let config = vector_config("[objective_acquisition]\nmode='pareto'\nscales=[1.0,2.0]");
        assert!(config.generation.as_ref().unwrap().validate().is_ok());
        assert!(config.validate_objectives().is_ok());
        assert!(vector_config("").validate_objectives().is_err());
        let mut scalar = config.clone();
        scalar.generation.as_mut().unwrap().reward = GenerationReward::TokenAccuracy;
        assert!(scalar.validate_objectives().is_err());
    }

    #[test]
    fn vector_command() {
        let mut config = vector_config("[objective_acquisition]\nmode='pareto'\nscales=[1.0,2.0]");
        let generation = config.generation.as_mut().unwrap();
        if let GenerationReward::CommandObjectives { timeout_ms, .. } = &mut generation.reward {
            *timeout_ms = 0;
        }
        assert!(generation.validate().is_err());
        assert!(
            ennx_wire::toml::from_str::<GenerationReward>(
                "kind='command-objectives'\nprogram='score'\nargs=[]\ntimeout-ms=10\nweights=[1.0]"
            )
            .is_err()
        );
    }

    #[test]
    fn temperature_bounds() {
        let mut config = vector_config("");
        let generation = config.generation.as_mut().unwrap();
        generation.temperature = 0.6457;
        generation.temperature_bounds = Some([0.0001, 0.9997]);
        generation.temperature_step = Some(0.2);
        assert!(generation.validate().is_ok());
        generation.temperature = 1.0;
        assert!(generation.validate().is_err());
    }
}
