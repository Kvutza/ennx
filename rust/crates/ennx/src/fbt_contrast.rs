//! Cached code-overlap profiles and explicit falsification controls.

use super::decode::Rollout;
use crate::code_overlap::Contrast;
use crate::config::GenerationTask;
use crate::text::ByteDecoder;
use ennx_wire::json::{Value, json};
use std::path::Path;

pub(super) struct Scorer {
    order: usize,
    references: Vec<(GenerationTask, Contrast, Value)>,
}

impl Scorer {
    pub(super) fn new(order: usize) -> Self {
        Self {
            order,
            references: Vec::new(),
        }
    }

    fn reference(
        &mut self,
        task: &GenerationTask,
        tokenizer: &ByteDecoder,
    ) -> Result<usize, String> {
        if let Some(index) = self.references.iter().position(|(key, _, _)| key == task) {
            return Ok(index);
        }
        let target = tokenizer.decode_bytes(&task.expected)?;
        let decoys = task
            .decoys
            .iter()
            .map(|tokens| tokenizer.decode_bytes(tokens))
            .collect::<Result<Vec<_>, _>>()?;
        let contrast = Contrast::new(&target, &decoys, self.order)?;
        let audit = controls(&contrast, &target, &decoys);
        let index = self.references.len();
        self.references.push((task.clone(), contrast, audit));
        Ok(index)
    }

    pub(super) fn audit(
        &mut self,
        tasks: &[GenerationTask],
        tokenizer: &ByteDecoder,
    ) -> Result<Value, String> {
        let mut rows = Vec::new();
        for task in tasks {
            let index = self.reference(task, tokenizer)?;
            rows.push(self.references[index].2.clone());
        }
        let passed = rows.iter().all(|row| row["passed"] == true);
        Ok(
            json!({"schema":"ennx.code_contrast_controls.v1", "passed":passed,
            "functional_correctness_established":false,"tasks":rows}),
        )
    }

    pub(super) fn score(
        &mut self,
        tokenizer: &ByteDecoder,
        tasks: &[GenerationTask],
        rollouts: &[Rollout],
        path: &Path,
    ) -> Result<Vec<f32>, String> {
        if tasks.len() != rollouts.len() {
            return Err("code contrast requires aligned tasks and completions".into());
        }
        let start = std::time::Instant::now();
        let mut rewards = Vec::with_capacity(tasks.len());
        let mut components = Vec::with_capacity(tasks.len());
        for (task, rollout) in tasks.iter().zip(rollouts) {
            let index = self.reference(task, tokenizer)?;
            let output = tokenizer.decode_bytes(&rollout.tokens)?;
            let (target, decoy) = self.references[index].1.components(&output);
            rewards.push((target - decoy) as f32);
            components.push(json!({"target_overlap":target,"max_decoy_overlap":decoy,
                "reward":target-decoy,"generated_bytes":output.len(),"generated_tokens":rollout.tokens.len()}));
        }
        let profile = json!({"schema":"ennx.code_contrast_reward.v1",
            "objective":"mean_clipped_byte_ngram_dice_target_minus_max_decoy",
            "max_ngram":self.order,"whitespace_policy":"ignore_ascii_whitespace",
            "extra_model_forwards":0,"functional_correctness_established":false,
            "scoring_ms":start.elapsed().as_secs_f64()*1000.0,"components":components});
        ennx_wire::json::pretty_writer(
            std::fs::File::create(path.join("reward.json")).map_err(|e| e.to_string())?,
            &profile,
        )
        .map_err(|e| e.to_string())?;
        Ok(rewards)
    }
}

fn controls(contrast: &Contrast, target: &[u8], decoys: &[Vec<u8>]) -> Value {
    let reference = contrast.reward(target);
    let mut shuffled = target.to_vec();
    let mut state = target.iter().fold(0u64, |state, byte| {
        crate::hash::splitmix64(state ^ u64::from(*byte))
    });
    for end in (1..shuffled.len()).rev() {
        state = crate::hash::splitmix64(state);
        shuffled.swap(end, state as usize % (end + 1));
    }
    let repetition = (0u8..=255)
        .map(|byte| contrast.reward(&vec![byte; target.len()]))
        .fold(f64::NEG_INFINITY, f64::max);
    let unrelated = decoys
        .iter()
        .map(|bytes| contrast.reward(bytes))
        .fold(f64::NEG_INFINITY, f64::max);
    let whitespace = contrast.reward(&vec![b' '; target.len()]);
    let shuffled = contrast.reward(&shuffled);
    let passed = [whitespace, repetition, shuffled, unrelated]
        .iter()
        .all(|&score| score < reference);
    json!({"reference":reference,"whitespace":whitespace,"max_single_byte_repetition":repetition,
        "shuffled":shuffled,"max_unrelated":unrelated,"passed":passed})
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn control_failures() {
        let target = b"aaaa";
        let decoys = vec![b"bbbb".to_vec()];
        let contrast = Contrast::new(target, &decoys, 4).unwrap();
        let report = controls(&contrast, target, &decoys);
        assert_eq!(report["passed"], false);
        assert_eq!(report["reference"], report["max_single_byte_repetition"]);
    }
}
