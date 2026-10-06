use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ResidualArchitecture {
    Legacy,
    Residual1,
    ProjectedBoundary,
    Hc4,
    Mhc4,
    LoopedMhc4,
    DiffusionMhc4,
    HnetMhc4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct LayerStep {
    pub layer: u32,
    pub pass: u32,
    pub execution: u32,
}

const FULL_STACK: [LayerStep; 10] = [
    LayerStep {
        layer: 0,
        pass: 0,
        execution: 0,
    },
    LayerStep {
        layer: 1,
        pass: 0,
        execution: 1,
    },
    LayerStep {
        layer: 2,
        pass: 0,
        execution: 2,
    },
    LayerStep {
        layer: 3,
        pass: 0,
        execution: 3,
    },
    LayerStep {
        layer: 4,
        pass: 0,
        execution: 4,
    },
    LayerStep {
        layer: 0,
        pass: 1,
        execution: 5,
    },
    LayerStep {
        layer: 1,
        pass: 1,
        execution: 6,
    },
    LayerStep {
        layer: 2,
        pass: 1,
        execution: 7,
    },
    LayerStep {
        layer: 3,
        pass: 1,
        execution: 8,
    },
    LayerStep {
        layer: 4,
        pass: 1,
        execution: 9,
    },
];

static SELECTIVE_CORE: std::sync::OnceLock<Vec<LayerStep>> = std::sync::OnceLock::new();

impl ResidualArchitecture {
    pub(super) fn from_model(model: crate::config::PretrainModel) -> Self {
        match model {
            crate::config::PretrainModel::FbtPisa1MoeV1 => Self::Legacy,
            crate::config::PretrainModel::FbtPisa1Residual1V1 => Self::Residual1,
            crate::config::PretrainModel::FbtPisa1ProjectedBoundaryV1 => Self::ProjectedBoundary,
            crate::config::PretrainModel::FbtPisa1Hc4V1 => Self::Hc4,
            crate::config::PretrainModel::FbtPisa1Mhc4V1 => Self::Mhc4,
            crate::config::PretrainModel::FbtPisa1LoopedMhc4V1 => Self::LoopedMhc4,
            crate::config::PretrainModel::FbtPisa1DiffusionMhc4V1 => Self::DiffusionMhc4,
            crate::config::PretrainModel::FbtPisa1HnetMhc4V1 => Self::HnetMhc4,
        }
    }

    pub(super) const fn is_multistream(self) -> bool {
        matches!(
            self,
            Self::Hc4 | Self::Mhc4 | Self::LoopedMhc4 | Self::DiffusionMhc4 | Self::HnetMhc4
        )
    }

    pub(super) const fn parameter_count(self) -> usize {
        if matches!(self, Self::DiffusionMhc4) {
            MHC_FULLPARAMS + WIDTH as usize + 64 * 64
        } else if self.is_multistream() {
            MHC_FULLPARAMS
        } else {
            FULL_PARAMETERS
        }
    }

    pub(super) const fn kernel_code(self) -> u32 {
        match self {
            Self::Legacy | Self::Residual1 | Self::ProjectedBoundary => 0,
            Self::Hc4 => 1,
            Self::Mhc4 | Self::LoopedMhc4 | Self::DiffusionMhc4 | Self::HnetMhc4 => 2,
        }
    }

    pub(super) const fn feedback_transition(
        self,
        configured: crate::config::FeedbackTransition,
    ) -> crate::config::FeedbackTransition {
        match self {
            Self::Legacy => configured,
            Self::Residual1 => crate::config::FeedbackTransition::Identity,
            Self::ProjectedBoundary => crate::config::FeedbackTransition::ProjectedSigmoid,
            Self::Hc4 | Self::Mhc4 | Self::LoopedMhc4 | Self::DiffusionMhc4 | Self::HnetMhc4 => {
                crate::config::FeedbackTransition::Identity
            }
        }
    }

    pub(super) const fn patch_size(self) -> usize {
        match self {
            Self::HnetMhc4 => 64,
            _ => 1,
        }
    }

    pub(super) fn layer_steps(self) -> &'static [LayerStep] {
        match self {
            Self::LoopedMhc4 | Self::DiffusionMhc4 | Self::HnetMhc4 => {
                SELECTIVE_CORE.get_or_init(|| {
                    crate::forward_program::RecurrentCore::selective_fbt()
                        .layer_visits(MODEL_LAYERS as usize, FEEDBACK_PASSES as usize)
                        .expect("built-in FBT recurrence must fit its physical layer stack")
                        .into_iter()
                        .map(|step| LayerStep {
                            layer: step.layer as u32,
                            pass: step.visit as u32,
                            execution: step.execution as u32,
                        })
                        .collect()
                })
            }
            _ => &FULL_STACK,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_sequences() {
        let full = ResidualArchitecture::Mhc4.layer_steps();
        assert_eq!(full.len(), 10);
        assert_eq!(
            full.iter().map(|step| step.layer).collect::<Vec<_>>(),
            [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
        );

        let looped = ResidualArchitecture::LoopedMhc4.layer_steps();
        assert_eq!(looped.len(), 7);
        assert_eq!(
            looped.iter().map(|step| step.layer).collect::<Vec<_>>(),
            [0, 1, 2, 1, 2, 3, 4]
        );
        assert_eq!(
            looped.iter().map(|step| step.execution).collect::<Vec<_>>(),
            [0, 1, 2, 3, 4, 5, 6]
        );
    }
}
