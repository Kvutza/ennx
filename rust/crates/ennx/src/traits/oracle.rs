use crate::objective_observation::ObjectiveObservation;
use crate::search::DeviceView;

/// A black-box function evaluated at a resident candidate.
///
/// The objective vector is the optimizer-facing result. Evidence contains
/// domain-specific material, such as generated tokens, timings, or artifacts,
/// and never participates in acquisition unless converted into the objective
/// observation explicitly.
pub trait Oracle {
    type Evidence;

    fn observe(
        &mut self,
        candidate: DeviceView<'_>,
    ) -> Result<(ObjectiveObservation, Self::Evidence), String>;
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl crate::bf16_metal::SearchState {
    /// Evaluate one selected candidate and publish its observation atomically
    /// through the resident controller.
    ///
    /// Oracle failures leave the proposal pending. A successful oracle call is
    /// followed by exactly one tell and one device-state synchronization.
    pub fn evaluate<O: Oracle>(
        &mut self,
        proposal: &crate::bf16_metal::Proposals,
        candidate: DeviceView<'_>,
        initial: bool,
        oracle: &mut O,
    ) -> Result<
        (
            crate::bf16_metal::NoisyDecision,
            O::Evidence,
            std::time::Duration,
        ),
        String,
    > {
        let (observation, evidence) = oracle.observe(candidate)?;
        let start = std::time::Instant::now();
        let decision = if initial {
            self.initial_objectives(proposal, observation)?
        } else {
            self.modeled_objectives(proposal, observation)?
        };
        if self.sync()? != vec![decision.accepted] {
            return Err("resident search acceptance mismatch".into());
        }
        Ok((decision, evidence, start.elapsed()))
    }
}
