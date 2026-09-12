//! Full-bandwidth transformer primitives. This is not yet a complete model.
//!
//! Normalization is explicit: the paper does not fully specify its learned
//! input scales. No Qwen/Nanochat defaults are imported here.

#[cfg(all(target_os = "macos", feature = "metal"))]
pub use crate::fbt_attention::{Attention, RmsNorm};
#[cfg(all(target_os = "macos", feature = "metal"))]
pub use crate::fbt_metal::{FeedForward, Feedback, Linear};
#[cfg(all(target_os = "macos", feature = "metal"))]
pub use crate::fbt_model::{
    BatchScore, ChunkTiming, Model, ModelConfig, ModelMemory, Parameter, Score, ScoreMode,
};

/// Bias-free projection activation, selected explicitly by the graph owner.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProjectionActivation {
    Identity,
    Sigmoid,
}

/// A cache is valid for exactly one candidate, document and prefill pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CacheKey {
    pub candidate: u64,
    pub sequence: u64,
    pub pass: u32,
}

/// Explicit primitive choices, not implicit defaults for the paper's backbone.
/// Q/K use optional unit RMSNorm BEFORE split-half RoPE. Head gates are supplied
/// by the graph owner and multiply the attention output before its projection.
#[derive(Clone, Copy, Debug)]
pub struct AttentionConfig {
    pub heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub capacity: u32,
    /// Number of visible positions INCLUDING the current token; None is full.
    pub window: Option<u32>,
    pub qk_norm: InputNorm,
    pub rope_base: f32,
    pub score_scale: f32,
}

impl AttentionConfig {
    pub fn validate(self) -> Result<(), String> {
        if self.heads == 0
            || self.kv_heads == 0
            || self.heads % self.kv_heads != 0
            || self.head_dim == 0
            || self.head_dim > 256
            || self.head_dim % 2 != 0
            || self.capacity == 0
            || self.window == Some(0)
            || !self.rope_base.is_finite()
            || self.rope_base <= 1.0
            || !self.score_scale.is_finite()
            || self.score_scale <= 0.0
        {
            return Err("Invalid FBT attention dimensions, window, RoPE or score scale".into());
        }
        self.qk_norm.epsilon()?;
        for heads in [self.heads, self.kv_heads] {
            let _ = self
                .capacity
                .checked_mul(heads)
                .and_then(|n| n.checked_mul(self.head_dim))
                .ok_or("FBT attention index overflow")?;
        }
        Ok(())
    }
}

/// SiLU-GLU matrix layout: gate and up are [intermediate, width]; down is
/// [width, intermediate]. Biases and surrounding normalization are separate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FeedForwardLayout {
    pub width: u32,
    pub intermediate: u32,
    pub gate_offset: usize,
    pub up_offset: usize,
    pub down_offset: usize,
    pub elements: usize,
}

impl FeedForwardLayout {
    pub fn new(width: u32, intermediate: u32) -> Result<Self, String> {
        if width == 0 || intermediate == 0 {
            return Err("FBT feed-forward dimensions must be nonzero".into());
        }
        let matrix = (width as usize)
            .checked_mul(intermediate as usize)
            .ok_or("FBT feed-forward matrix size overflow")?;
        let elements = matrix
            .checked_mul(3)
            .ok_or("FBT feed-forward size overflow")?;
        elements
            .checked_mul(2)
            .ok_or("FBT feed-forward byte size overflow")?;
        Ok(Self {
            width,
            intermediate,
            gate_offset: 0,
            up_offset: matrix,
            down_offset: 2 * matrix,
            elements,
        })
    }
}

/// No implicit depth scaling: the caller supplies the branch multiplier.
#[derive(Clone, Copy, Debug)]
pub struct FeedForwardConfig {
    pub width: u32,
    pub intermediate: u32,
    pub residual_scale: f32,
}

impl FeedForwardConfig {
    pub fn validate(self) -> Result<FeedForwardLayout, String> {
        if !self.residual_scale.is_finite() {
            return Err("FBT residual scale must be finite".into());
        }
        FeedForwardLayout::new(self.width, self.intermediate)
    }
}

/// Unit-scale RMS normalization; learned scales use the separate RmsNorm primitive.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum InputNorm {
    None,
    UnitRms { epsilon: f32 },
}

impl InputNorm {
    pub(crate) fn epsilon(self) -> Result<f32, String> {
        match self {
            Self::None => Ok(0.0),
            Self::UnitRms { epsilon } if epsilon.is_finite() && epsilon > 0.0 => Ok(epsilon),
            Self::UnitRms { .. } => Err("FBT RMS epsilon must be finite and positive".into()),
        }
    }
}

/// Feedback matrices are consecutive row-major [out, in] BF16 arrays.
/// This describes only the feedback parameters, not the full BO model layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FeedbackLayout {
    pub width: u32,
    pub state_offset: usize,
    pub gate_offset: usize,
    pub elements: usize,
}

impl FeedbackLayout {
    pub fn new(width: u32) -> Result<Self, String> {
        if width == 0 {
            return Err("FBT feedback width must be nonzero".into());
        }
        let matrix = (width as usize)
            .checked_mul(width as usize)
            .ok_or("FBT matrix size overflow")?;
        let elements = matrix.checked_mul(2).ok_or("FBT parameter size overflow")?;
        elements
            .checked_mul(2)
            .ok_or("FBT BF16 byte size overflow")?;
        Ok(Self {
            width,
            state_offset: 0,
            gate_offset: matrix,
            elements,
        })
    }
}

/// Explicit primitive configuration, not a claimed resolution of paper ambiguities.
#[derive(Clone, Copy, Debug)]
pub struct FeedbackConfig {
    pub width: u32,
    pub token_norm: InputNorm,
    pub fused_norm: InputNorm,
}

impl FeedbackConfig {
    pub fn validate(self) -> Result<FeedbackLayout, String> {
        self.token_norm.epsilon()?;
        self.fused_norm.epsilon()?;
        FeedbackLayout::new(self.width)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_validation() {
        let layout = FeedbackLayout::new(1536).unwrap();
        assert_eq!(layout.state_offset, 0);
        assert_eq!(layout.gate_offset, 2_359_296);
        assert_eq!(layout.elements, 4_718_592);
        assert!(FeedbackLayout::new(0).is_err());
        assert!(FeedbackLayout::new(u32::MAX).is_err());
        for epsilon in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert!(
                FeedbackConfig {
                    width: 96,
                    token_norm: InputNorm::UnitRms { epsilon },
                    fused_norm: InputNorm::None,
                }
                .validate()
                .is_err()
            );
        }
    }

    #[test]
    fn feed_forwardlayout() {
        let layout = FeedForwardLayout::new(1536, 6656).unwrap();
        assert_eq!(layout.up_offset, 10_223_616);
        assert_eq!(layout.down_offset, 20_447_232);
        assert_eq!(layout.elements, 30_670_848);
        for (width, intermediate) in [(0, 1), (1, 0), (u32::MAX, u32::MAX)] {
            assert!(FeedForwardLayout::new(width, intermediate).is_err());
        }
        assert!(
            FeedForwardConfig {
                width: 1,
                intermediate: 1,
                residual_scale: f32::NAN
            }
            .validate()
            .is_err()
        );
    }
}
