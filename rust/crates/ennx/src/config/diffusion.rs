use deser::{Deserialize, Serialize};

/// Learned block-causal draft proposals. The causal target verifies their tokens.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[deser(default, deny_unknown_fields, rename_all = "kebab-case")]
pub struct DiffusionConfig {
    pub block: u32,
    pub steps: u32,
    pub visits: [u32; 2],
    /// Confidence-weighted interpolation between the sampled token and mask.
    pub soft: bool,
    pub index: IndexMode,
    /// Relative RMS drift budget for index reuse. Zero disables reuse.
    pub reuse: f32,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[deser(rename_all = "kebab-case")]
pub enum IndexMode {
    #[default]
    Independent,
    Shared,
    Refined,
}

impl Default for DiffusionConfig {
    fn default() -> Self {
        Self {
            block: 128,
            steps: 4,
            visits: [1, 4],
            soft: true,
            index: IndexMode::Refined,
            reuse: 0.0,
        }
    }
}

impl DiffusionConfig {
    pub fn validate(self) -> Result<(), String> {
        if !(128..=4096).contains(&self.block)
            || !self.block.is_power_of_two()
            || !(1..=64).contains(&self.steps)
            || self.visits[0] == 0
            || self.visits[0] > self.visits[1]
            || self.visits[1] > 4
            || !self.reuse.is_finite()
            || !(0.0..=1.0).contains(&self.reuse)
        {
            return Err("diffusion requires block=128..4096 (power of two), steps=1..64, visits within 1..4, and finite reuse fraction in 0..1".into());
        }
        Ok(())
    }
}
