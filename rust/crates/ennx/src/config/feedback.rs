use deser::{Deserialize, Serialize};

/// Transition between repeated passes through the shared Transformer stack.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[deser(rename_all = "kebab-case")]
pub enum FeedbackTransition {
    /// Preserve the completed pass as the next pass's residual stream.
    Identity,
    /// Project the state and multiply it by a separately projected sigmoid gate.
    #[default]
    ProjectedSigmoid,
}
