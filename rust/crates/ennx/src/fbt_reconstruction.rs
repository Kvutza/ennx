//! Resident reference masks; score every actual completion without another forward.

use super::decode::Rollout;
use crate::config::GenerationTask;
use crate::reconstruction::Reference;
use crate::text::ByteDecoder;
use ennx_wire::json::json;
use std::path::Path;

pub(super) struct Reconstruction {
    references: Vec<(Vec<u32>, Reference<u8>)>,
}

impl Reconstruction {
    pub(super) fn new(tasks: &[GenerationTask], tokenizer: &ByteDecoder) -> Result<Self, String> {
        Ok(Self {
            references: tasks
                .iter()
                .map(|task| {
                    Ok((
                        task.expected.clone(),
                        Reference::new(&tokenizer.decode_bytes(&task.expected)?),
                    ))
                })
                .collect::<Result<_, String>>()?,
        })
    }

    fn reference(
        &mut self,
        tokens: &[u32],
        tokenizer: &ByteDecoder,
    ) -> Result<&Reference<u8>, String> {
        let index = self.references.iter().position(|(key, _)| key == tokens);
        let index = match index {
            Some(index) => index,
            None => {
                let index = self.references.len();
                self.references.push((
                    tokens.to_vec(),
                    Reference::new(&tokenizer.decode_bytes(tokens)?),
                ));
                index
            }
        };
        Ok(&self.references[index].1)
    }

    pub(super) fn score(
        &mut self,
        tokenizer: &ByteDecoder,
        tasks: &[GenerationTask],
        rollouts: &[Rollout],
        path: &Path,
    ) -> Result<Vec<f32>, String> {
        if tasks.len() != rollouts.len() || tasks.iter().any(|task| task.expected.is_empty()) {
            return Err("reconstruction requires aligned nonempty reference completions".into());
        }
        let start = std::time::Instant::now();
        let mut rewards = Vec::with_capacity(tasks.len());
        let mut components = Vec::with_capacity(tasks.len());
        for (task, rollout) in tasks.iter().zip(rollouts) {
            let completion = tokenizer.decode_bytes(&rollout.tokens)?;
            let reference = self.reference(&task.expected, tokenizer)?;
            let (distance, reward) = reference.reward(&completion);
            rewards.push(reward);
            components.push(json!({
                "edit_distance":distance,
                "reference_tokens":task.expected.len(),
                "reference_bytes":reference.len(),
                "generated_tokens":rollout.tokens.len(),
                "generated_bytes":completion.len(),
                "generated_valid_utf8":std::str::from_utf8(&completion).is_ok(),
                "reward":reward,
            }));
        }
        let profile = json!({
            "schema":"ennx.code_reconstruction_reward.v2",
            "stage":"corpus_reconstruction_bootstrap",
            "objective":"one_minus_exact_byte_edit_distance_over_max_length",
            "text_policy":"exact_decoded_bytes_without_utf8_replacement",
            "algorithm":"word_parallel_unit_cost_levenshtein",
            "teacher_forcing":false,
            "extra_model_forwards":0,
            "functional_correctness_established":false,
            "scoring_ms":start.elapsed().as_secs_f64()*1000.0,
            "components":components,
        });
        ennx_wire::json::pretty_writer(
            std::fs::File::create(path.join("reward.json")).map_err(|error| error.to_string())?,
            &profile,
        )
        .map_err(|error| error.to_string())?;
        Ok(rewards)
    }
}
