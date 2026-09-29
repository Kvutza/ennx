//! Persistent native rewards and optional frozen evaluators across BO candidates.

use super::decode::Rollout;
use crate::config::{FrozenAttention, FrozenReadout, GenerationReward, GenerationTask};
use crate::qwen_metal::QwenEvaluator;
use crate::text::ByteDecoder;
use deser::Deserialize;
use ennx_wire::json::{Value, json};
use metal::Buffer;
use std::io::{BufRead, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{Receiver, channel};
use std::time::{Duration, Instant};

pub(super) enum Evaluator {
    Frozen(Box<FrozenEvaluator>),
    Reconstruction(super::reconstruction::Reconstruction),
    Contrast(super::code_contrast::Scorer),
    Execution(super::code_environment::Executor),
}

impl Evaluator {
    pub(super) fn new(
        reward: &GenerationReward,
        tasks: &[GenerationTask],
        tokenizer: Option<&ByteDecoder>,
    ) -> Result<Option<Self>, String> {
        if let GenerationReward::CodeExecution {
            environment,
            interpreter,
            timeout_ms,
        } = reward
        {
            return Ok(Some(Self::Execution(
                super::code_environment::Executor::new(environment, interpreter, *timeout_ms)?,
            )));
        }
        if let GenerationReward::CodeContrastive { max_ngram, .. }
        | GenerationReward::DraftedCode { max_ngram, .. } = reward
        {
            return Ok(Some(Self::Contrast(super::code_contrast::Scorer::new(
                *max_ngram,
            ))));
        }
        if matches!(
            reward,
            GenerationReward::CodeReconstruction | GenerationReward::CodeObjectives { .. }
        ) {
            return Ok(Some(Self::Reconstruction(
                super::reconstruction::Reconstruction::new(
                    tasks,
                    tokenizer.ok_or("code_reconstruction requires the corpus tokenizer")?,
                )?,
            )));
        }
        Ok(FrozenEvaluator::new(reward)?.map(|evaluator| Self::Frozen(Box::new(evaluator))))
    }

    pub(super) fn frozen(&mut self) -> Option<&mut FrozenEvaluator> {
        match self {
            Self::Frozen(evaluator) => Some(evaluator),
            Self::Reconstruction(_) | Self::Contrast(_) | Self::Execution(_) => None,
        }
    }

    pub(super) fn score(
        &mut self,
        tokenizer: Option<&ByteDecoder>,
        tasks: &[GenerationTask],
        rollouts: &[Rollout],
        path: &Path,
    ) -> Result<Vec<f32>, String> {
        match self {
            Self::Execution(_) => {
                Err("execution requires independent objective observations".into())
            }
            Self::Contrast(evaluator) => evaluator.score(
                tokenizer.ok_or("code contrast requires the corpus tokenizer")?,
                tasks,
                rollouts,
                path,
            ),
            Self::Frozen(evaluator) => evaluator.score(
                tokenizer.ok_or("frozen evaluator requires the corpus tokenizer")?,
                tasks,
                rollouts,
                path,
            ),
            Self::Reconstruction(evaluator) => evaluator.score(
                tokenizer.ok_or("code_reconstruction requires the corpus tokenizer")?,
                tasks,
                rollouts,
                path,
            ),
        }
    }

    pub(super) fn audit(
        &mut self,
        tasks: &[GenerationTask],
        tokenizer: Option<&ByteDecoder>,
    ) -> Result<Option<Value>, String> {
        match self {
            Self::Execution(executor) => executor.audit().map(Some),
            Self::Contrast(scorer) => scorer
                .audit(
                    tasks,
                    tokenizer.ok_or("code contrast requires a tokenizer")?,
                )
                .map(Some),
            _ => Ok(None),
        }
    }

    pub(super) fn execute(
        &mut self,
        tokenizer: &ByteDecoder,
        rollouts: &[Rollout],
        path: &Path,
    ) -> Result<super::gen_protocol::RewardResult, String> {
        let Self::Execution(executor) = self else {
            return Err("missing execution evaluator".into());
        };
        executor.score(tokenizer, rollouts, path)
    }
}

/// One negative control, not a certificate that the reward measures learning.
pub(super) fn reference_audit(
    generated: &[f32],
    reference: &[f32],
    diagnostics: &[Value],
) -> Result<Value, String> {
    if generated.is_empty()
        || generated.len() != reference.len()
        || generated.len() != diagnostics.len()
        || generated
            .iter()
            .chain(reference)
            .any(|score| !score.is_finite())
    {
        return Err("reward audit requires aligned finite scores and diagnostics".into());
    }
    let comparisons = generated
        .iter()
        .zip(reference)
        .zip(diagnostics)
        .map(|((generated, reference), diagnostics)| {
            json!({
                "generated_reward":generated,
                "corpus_reward":reference,
                "generated_preferred":generated > reference,
                "generated_minus_corpus":f64::from(*generated) - f64::from(*reference),
                "generated_diagnostics":diagnostics,
            })
        })
        .collect::<Vec<_>>();
    let prefers_generated = generated.iter().zip(reference).any(|(g, r)| g > r);
    Ok(json!({
        "schema":"ennx.reward_audit.v1",
        "status":if prefers_generated {"generated_preferred_to_corpus"} else {"corpus_not_outscored"},
        "higher_is_better":true,
        "learning_quality_established":false,
        "interpretation":"A generated completion is not automatically a negative control. Inspect its diagnostics and text; this comparison alone cannot certify the objective.",
        "used_for_acceptance":false,
        "comparisons":comparisons,
        "unmeasured_controls":["shuffled_continuation","unrelated_continuation","controlled_repetition"],
    }))
}

pub(super) struct FrozenEvaluator {
    evaluator: QwenEvaluator,
    weights: Buffer,
    child: Child,
    input: ChildStdin,
    output: Receiver<Result<Value, String>>,
    timeout: Duration,
}

#[derive(Deserialize)]
#[deser(deny_unknown_fields)]
struct Row {
    tokens: Vec<i32>,
    mask: Vec<bool>,
    completion_tokens: usize,
    completion_bytes: usize,
}

fn relay_responses(
    reader: std::process::ChildStdout,
    sender: std::sync::mpsc::Sender<Result<Value, String>>,
) {
    for line in std::io::BufReader::new(reader).lines() {
        let result = line.map_err(|error| error.to_string()).and_then(|line| {
            ennx_wire::json::from_str(&line)
                .map_err(|error| format!("invalid tokenizer response: {error}"))
        });
        if sender.send(result).is_err() {
            break;
        }
    }
}

impl FrozenEvaluator {
    pub(super) fn new(reward: &GenerationReward) -> Result<Option<Self>, String> {
        let GenerationReward::FrozenQwen {
            checkpoint,
            tokenizer_program,
            tokenizer_args,
            max_tokens,
            timeout_ms,
            backend,
            readout,
            attention,
        } = reward
        else {
            return Ok(None);
        };
        // This native evaluator implements the pinned dense-control architecture.
        // Validate provenance before allocating/loading its immutable weights.
        let manifest: Value = ennx_wire::json::from_reader(
            std::fs::File::open(checkpoint.join("manifest.json")).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        if manifest["model_id"] != "Qwen/Qwen2.5-Coder-1.5B"
            || manifest["revision"] != "dba20987fcbfb46dcca5a257e10b5ab39c9ec7ce"
            || manifest["weights_downloaded"] != true
        {
            return Err(
                "frozen evaluator requires the pinned Qwen2.5-Coder-1.5B checkpoint".into(),
            );
        }
        let mut evaluator = QwenEvaluator::new_backend(
            *max_tokens,
            backend.as_str(),
            matches!(attention, FrozenAttention::Tiled16),
        )?;
        let weights = evaluator.load_weights(&checkpoint.join("model.safetensors"))?;
        if matches!(readout, FrozenReadout::MpsFp32) {
            evaluator.prepare_readout(&weights)?;
        }
        let mut child = Command::new(tokenizer_program)
            .args(tokenizer_args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| e.to_string())?;
        let input = child.stdin.take().ok_or("missing tokenizer input")?;
        let reader = child.stdout.take().ok_or("missing tokenizer output")?;
        let (sender, output) = channel();
        std::thread::spawn(move || relay_responses(reader, sender));
        let mut result = Self {
            evaluator,
            weights,
            child,
            input,
            output,
            timeout: Duration::from_millis(*timeout_ms),
        };
        let ready = result.request(&json!({"checkpoint":checkpoint,"max_tokens":max_tokens}))?;
        if ready["ready"] != true {
            return Err("frozen evaluator tokenizer did not become ready".into());
        }
        Ok(Some(result))
    }

    fn request(&mut self, request: &Value) -> Result<Value, String> {
        ennx_wire::json::to_writer(&mut self.input, request).map_err(|e| e.to_string())?;
        self.input
            .write_all(b"\n")
            .and_then(|_| self.input.flush())
            .map_err(|e| e.to_string())?;
        self.output
            .recv_timeout(self.timeout)
            .map_err(|e| format!("tokenizer response: {e}"))?
    }

    pub(super) fn score(
        &mut self,
        tokenizer: &ByteDecoder,
        tasks: &[GenerationTask],
        rollouts: &[Rollout],
        path: &Path,
    ) -> Result<Vec<f32>, String> {
        let start = Instant::now();
        let mut normalization = Vec::with_capacity(tasks.len());
        let mut text = Vec::with_capacity(tasks.len());
        for (task, rollout) in tasks.iter().zip(rollouts) {
            let prompt = tokenizer.decode_bytes(&task.prompt)?;
            let completion = tokenizer.decode_bytes(&rollout.tokens)?;
            normalization.push(json!({
                "prompt_valid_utf8":std::str::from_utf8(&prompt).is_ok(),
                "completion_valid_utf8":std::str::from_utf8(&completion).is_ok(),
                "raw_completion_bytes":completion.len(),
            }));
            text.push(json!({
                "prompt":String::from_utf8_lossy(&prompt),
                "completion":String::from_utf8_lossy(&completion),
            }));
        }
        let response = self.request(&json!({"rows":text}))?;
        let tokenization_ms = start.elapsed().as_secs_f64() * 1000.0;
        if let Some(error) = response.get("error") {
            return Err(format!(
                "frozen evaluator tokenization: {}",
                ennx_wire::json::to_string(error).map_err(|e| e.to_string())?
            ));
        }
        let rows: Vec<Row> =
            ennx_wire::json::from_value(response["rows"].clone()).map_err(|e| e.to_string())?;
        if rows.len() != tasks.len()
            || rows.iter().any(|row| {
                row.completion_tokens == 0
                    || row.completion_bytes == 0
                    || row.mask.iter().filter(|&&scored| scored).count() != row.completion_tokens
            })
        {
            return Err("frozen evaluator returned invalid completion coverage".into());
        }
        let tokens = rows
            .iter()
            .map(|row| row.tokens.clone())
            .collect::<Vec<_>>();
        let masks = rows.iter().map(|row| row.mask.clone()).collect::<Vec<_>>();
        let losses = self.evaluator.losses(&self.weights, &tokens, &masks)?;
        let profile = self.evaluator.loss_profile().map(|profile| {
            json!({
                "tokens":profile.tokens,
                "scored_tokens":profile.scored_tokens,
                "cached_tokens":profile.cached_tokens,
                "tiled_attention":profile.tile_attn,
                "kv_cache_bytes":profile.kv_cache_bytes,
                "write_ms":profile.write_ms,
                "forward_ms":profile.forward_ms,
                "output_ms":profile.output_ms,
                "total_ms":profile.total_ms,
            })
        });
        let components = rows.iter().zip(&losses).zip(normalization).map(|((row, nll), normalization)| json!({"mean_token_nll":nll,"completion_tokens":row.completion_tokens,"completion_bytes":row.completion_bytes,"nats_per_byte":f64::from(*nll) * row.completion_tokens as f64 / row.completion_bytes as f64,"normalization":normalization})).collect::<Vec<_>>();
        ennx_wire::json::pretty_writer(std::fs::File::create(path.join("reward.json")).map_err(|e| e.to_string())?, &json!({"schema":"ennx.frozen_qwen_reward.v2","objective":"negative_mean_teacher_token_nll","text_policy":"utf8_lossy_explicit_segmentation","text_preparation_and_tokenization_ms":tokenization_ms,"evaluator_profile":profile,"components":components})).map_err(|e| e.to_string())?;
        Ok(losses.into_iter().map(|loss| -loss).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audit_direction() {
        let diagnostics = [json!({"generated_tokens":4096})];
        let audit = reference_audit(&[-0.428], &[-1.614], &diagnostics).unwrap();
        assert_eq!(audit["status"], "generated_preferred_to_corpus");
        assert_eq!(audit["comparisons"][0]["generated_preferred"], true);
        assert_eq!(audit["learning_quality_established"], false);
        let audit = reference_audit(&[-2.0], &[-1.0], &diagnostics).unwrap();
        assert_eq!(audit["status"], "corpus_not_outscored");
        assert_eq!(audit["learning_quality_established"], false);
    }

    #[test]
    fn audit_invalid() {
        assert!(reference_audit(&[], &[], &[]).is_err());
        assert!(reference_audit(&[0.0], &[], &[Value::null()]).is_err());
        assert!(reference_audit(&[f32::NAN], &[0.0], &[Value::null()]).is_err());
    }
}

impl Drop for FrozenEvaluator {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
