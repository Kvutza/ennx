use deser::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[deser(rename_all = "kebab-case")]
pub enum PretrainSelection {
    #[default]
    Enn,
    Random,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[deser(rename_all = "kebab-case")]
pub enum DistanceScaling {
    #[default]
    Global,
    SelfTuning,
}

impl DistanceScaling {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Global => "global",
            Self::SelfTuning => "self-tuning",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[deser(rename_all = "kebab-case")]
pub enum HistoryGeometry {
    #[default]
    Realized,
    Latent,
}

impl HistoryGeometry {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Realized => "realized",
            Self::Latent => "latent",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[deser(rename_all = "kebab-case")]
pub enum TrustRegionShape {
    Scalar,
    TensorFamilyStatic,
    TensorFamilyLearned,
}

impl TrustRegionShape {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::TensorFamilyStatic => "tensor-family-static",
            Self::TensorFamilyLearned => "tensor-family-learned",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[deser(rename_all = "kebab-case")]
pub enum ObjectiveReference {
    #[default]
    MovingIncumbent,
    FrozenInitial,
}

impl ObjectiveReference {
    pub const fn name(self) -> &'static str {
        match self {
            Self::MovingIncumbent => "moving-incumbent",
            Self::FrozenInitial => "frozen-initial",
        }
    }
}
