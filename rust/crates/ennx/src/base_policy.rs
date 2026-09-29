//! Immutable qualification records for coding-policy checkpoints.

use deser::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::Read;
use std::path::Path;

pub const POLICY_SCHEMA: &str = "ennx.base_policy.v1";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[deser(deny_unknown_fields)]
pub struct BasePolicyManifest {
    pub schema: String,
    pub model: String,
    pub checkpoint_sha256: String,
    pub checkpoint_bytes: u64,
    pub tokenizer_sha256: String,
    pub training_corpus_manifest_sha256: String,
    pub training_tokens: u64,
    pub objective: String,
    pub qualification: PolicyQualification,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[deser(deny_unknown_fields)]
pub struct PolicyQualification {
    pub evaluator: String,
    pub split: String,
    pub split_manifest_sha256: String,
    pub episodes: u32,
    pub generated_tokens_per_episode: u32,
    pub full_length_episodes: u32,
    pub noncollapsed_episodes: u32,
    pub syntax_checks: u32,
    pub syntax_checks_passed: u32,
    pub test_split_used_for_selection: bool,
}

impl BasePolicyManifest {
    pub fn load_verified(manifest: &Path, checkpoint: &Path) -> Result<Self, String> {
        let record: Self = ennx_wire::json::from_reader(
            File::open(manifest).map_err(|error| format!("open base-policy manifest: {error}"))?,
        )
        .map_err(|error| format!("parse base-policy manifest: {error}"))?;
        record.validate()?;
        let metadata = checkpoint
            .metadata()
            .map_err(|error| format!("read checkpoint metadata: {error}"))?;
        if metadata.len() != record.checkpoint_bytes {
            return Err("base-policy checkpoint byte length does not match its manifest".into());
        }
        let digest = file_sha256(checkpoint)?;
        if digest != record.checkpoint_sha256 {
            return Err("base-policy checkpoint SHA256 does not match its manifest".into());
        }
        Ok(record)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != POLICY_SCHEMA
            || self.model != "fbt-pisa1-legacy-v1"
            || self.objective != "causal_next_token"
        {
            return Err("base-policy schema, model, or training objective is unsupported".into());
        }
        for (name, digest) in [
            ("checkpoint", self.checkpoint_sha256.as_str()),
            ("tokenizer", self.tokenizer_sha256.as_str()),
            (
                "training corpus manifest",
                self.training_corpus_manifest_sha256.as_str(),
            ),
            (
                "qualification split manifest",
                self.qualification.split_manifest_sha256.as_str(),
            ),
        ] {
            if !is_sha256(digest) {
                return Err(format!("base-policy {name} SHA256 is invalid"));
            }
        }
        let qualification = &self.qualification;
        if self.checkpoint_bytes == 0
            || self.training_tokens == 0
            || qualification.evaluator.is_empty()
            || qualification.split != "validation"
            || qualification.episodes < 3
            || qualification.generated_tokens_per_episode != 4096
            || qualification.full_length_episodes != qualification.episodes
            || qualification.noncollapsed_episodes != qualification.episodes
            || qualification.syntax_checks < qualification.episodes
            || qualification.syntax_checks_passed != qualification.syntax_checks
            || qualification.test_split_used_for_selection
        {
            return Err(
                "base policy has not passed the held-out 4096-token qualification gate".into(),
            );
        }
        Ok(())
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn file_sha256(path: &Path) -> Result<String, String> {
    let mut file = File::open(path).map_err(|error| format!("open checkpoint: {error}"))?;
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|error| format!("hash checkpoint: {error}"))?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest(checkpoint: &[u8]) -> BasePolicyManifest {
        BasePolicyManifest {
            schema: POLICY_SCHEMA.into(),
            model: "fbt-pisa1-legacy-v1".into(),
            checkpoint_sha256: format!("{:x}", Sha256::digest(checkpoint)),
            checkpoint_bytes: checkpoint.len() as u64,
            tokenizer_sha256: "1".repeat(64),
            training_corpus_manifest_sha256: "2".repeat(64),
            training_tokens: 1_000_000,
            objective: "causal_next_token".into(),
            qualification: PolicyQualification {
                evaluator: "ennx-policy-qualification-v1".into(),
                split: "validation".into(),
                split_manifest_sha256: "3".repeat(64),
                episodes: 3,
                generated_tokens_per_episode: 4096,
                full_length_episodes: 3,
                noncollapsed_episodes: 3,
                syntax_checks: 3,
                syntax_checks_passed: 3,
                test_split_used_for_selection: false,
            },
        }
    }

    #[test]
    fn qualified_identity() {
        let directory = tempfile::tempdir().unwrap();
        let checkpoint = directory.path().join("checkpoint.safetensors");
        std::fs::write(&checkpoint, b"checkpoint").unwrap();
        let record = manifest(b"checkpoint");
        let path = directory.path().join("base-policy.json");
        ennx_wire::json::to_writer(File::create(&path).unwrap(), &record).unwrap();
        BasePolicyManifest::load_verified(&path, &checkpoint).unwrap();
        std::fs::write(&checkpoint, b"checkpoinu").unwrap();
        assert!(
            BasePolicyManifest::load_verified(&path, &checkpoint)
                .unwrap_err()
                .contains("SHA256")
        );
    }

    #[test]
    fn invalid_qualification() {
        let mut record = manifest(b"checkpoint");
        record.qualification.test_split_used_for_selection = true;
        assert!(record.validate().is_err());
        record.qualification.test_split_used_for_selection = false;
        record.qualification.noncollapsed_episodes = 2;
        assert!(record.validate().is_err());
    }
}
