//! Backend-neutral block-denoising state and reproducible corruption.
use crate::config::DiffusionConfig;
use crate::hash::splitmix64;
use deser::Serialize;

pub const MASK: u32 = u32::MAX;

#[derive(Debug, Clone, Default, Serialize)]
#[deser(rename_all = "kebab-case")]
pub struct DiffusionMetrics {
    pub blocks: usize,
    pub steps: usize,
    pub visits: u32,
    pub changed_tokens: usize,
    pub generated_tokens: usize,
    pub evaluated_positions: usize,
    pub accepted_prefix: usize,
    pub matching_tokens: usize,
    pub wall_seconds: f64,
    pub index_reused: usize,
    pub index_refreshed: usize,
    pub generation_positions: usize,
    pub cache_positions: usize,
    pub generation_gpu_seconds: f64,
    pub cache_gpu_seconds: f64,
}

/// Per-block heterogeneous corruption. Clean labels never enter generated
/// trajectories; callers use this only for the separate supervised objective.
pub fn corrupt(clean: &[u32], block: usize, seed: u64) -> (Vec<u32>, Vec<bool>) {
    assert!(block != 0);
    let mut tokens = clean.to_vec();
    let mut scored = vec![false; clean.len()];
    for (index, (token, score)) in tokens.iter_mut().zip(&mut scored).enumerate() {
        let group = index / block;
        let rate = uniform(seed ^ group as u64 ^ 0x6e6f_6973_65).max(1.0 / block as f32);
        if uniform(seed ^ index as u64 ^ 0x6d61_736b) < rate || index % block == 0 {
            *token = MASK;
            *score = true;
        }
    }
    (tokens, scored)
}

pub fn uniform(seed: u64) -> f32 {
    ((splitmix64(seed) >> 40) as f32 + 0.5) / 16_777_216.0
}

pub fn visits(config: DiffusionConfig, seed: u64) -> u32 {
    config.visits[0]
        + (splitmix64(seed ^ 0x6c6f_6f70) % u64::from(config.visits[1] - config.visits[0] + 1))
            as u32
}

/// Linear mask-to-prediction interpolation with embedding-RMS normalization handled
/// by the input kernel. Final decoding emits discrete vocabulary tokens.
pub fn confidence(probability: f32, step: u32, steps: u32, soft: bool) -> f32 {
    if !soft || step + 1 == steps {
        1.0
    } else {
        probability.clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corruptions() {
        let clean = vec![17; 1024];
        let (tokens, score) = corrupt(&clean, 128, 41);
        assert_eq!(corrupt(&clean, 128, 41), (tokens.clone(), score.clone()));
        assert!(score.iter().any(|v| *v));
        assert!(score.iter().any(|v| !*v));
        for (index, token) in tokens.iter().enumerate() {
            assert_eq!(*token == MASK, score[index]);
            if !score[index] {
                assert_eq!(*token, clean[index]);
            }
        }
        assert!(tokens.chunks(128).all(|group| group.contains(&MASK)));
        let count = score
            .chunks(128)
            .map(|group| group.iter().filter(|v| **v).count())
            .collect::<Vec<_>>();
        assert!(count.windows(2).any(|pair| pair[0] != pair[1]));
    }
}
