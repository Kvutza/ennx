use super::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[deser(default, deny_unknown_fields, rename_all = "kebab-case")]
pub struct ProposalSpec {
    pub method: crate::procedural_pool::ProposalMethod,
    pub distribution: Option<crate::Perturbation>,
    /// Total proposals, distinct from ENN fitting trials.
    pub candidates: u32,
    pub arms: u32,
}

impl Default for ProposalSpec {
    fn default() -> Self {
        let pool = crate::procedural_pool::ProceduralPool::legacy();
        Self {
            method: crate::procedural_pool::ProposalMethod::default(),
            distribution: None,
            candidates: pool.count(),
            arms: pool.arms(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[deser(default, deny_unknown_fields, rename_all = "kebab-case")]
pub struct EnnSpec {
    pub neighbors: i32,
    pub epistemic_scale: f64,
    pub aleatoric_scale: f64,
    pub y_scale: f64,
    pub geometry: HistoryGeometry,
    pub scaling: DistanceScaling,
    pub local_neighbors: Option<usize>,
    pub fit: FitSpec,
}

impl Default for EnnSpec {
    fn default() -> Self {
        let ask = crate::trials::Ask::default();
        Self {
            neighbors: ask.neighbors as i32,
            epistemic_scale: f64::from(ask.epistemic_scale),
            aleatoric_scale: f64::from(ask.aleatoric_scale),
            y_scale: f64::from(ask.y_scale),
            geometry: HistoryGeometry::default(),
            scaling: DistanceScaling::default(),
            local_neighbors: None,
            fit: FitSpec::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[deser(default, deny_unknown_fields, rename_all = "kebab-case")]
pub struct FitSpec {
    pub neighbors: bool,
    pub candidates: usize,
    pub samples: usize,
}

impl Default for FitSpec {
    fn default() -> Self {
        let SurrogateConfig::ENN(enn) = turbo_enn().surrogate else {
            unreachable!()
        };
        Self {
            neighbors: false,
            candidates: enn.num_candidates,
            samples: enn.num_samples,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[deser(
    tag = "method",
    rename_all = "kebab-case",
    rename_all_fields = "kebab-case",
    deny_unknown_fields
)]
pub enum AcquisitionSpec {
    Ucb {
        #[deser(default = default_beta())]
        beta: f64,
    },
    Thompson,
    Pareto {
        scales: Vec<f32>,
    },
    AugmentedChebyshev {
        scales: Vec<f32>,
        preferences: Vec<f32>,
        alpha: f32,
        seed_domain: String,
    },
}

fn default_regions() -> usize {
    4
}

fn default_beta() -> f64 {
    let AcquisitionConfig::UCB { beta } = AcquisitionConfig::default() else {
        unreachable!()
    };
    beta
}

impl Default for AcquisitionSpec {
    fn default() -> Self {
        Self::Ucb {
            beta: default_beta(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[deser(default, deny_unknown_fields, rename_all = "kebab-case")]
pub struct TrustRegionBounds {
    pub initial: f64,
    pub min: f64,
    pub max: f64,
    pub shape: Option<TrustRegionShape>,
}

impl Default for TrustRegionBounds {
    fn default() -> Self {
        let length = TRLengthConfig::default();
        Self {
            initial: length.length_init,
            min: length.length_min,
            max: length.length_max,
            shape: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[deser(
    tag = "method",
    rename_all = "kebab-case",
    rename_all_fields = "kebab-case",
    deny_unknown_fields
)]
pub enum TrustRegionSpec {
    Turbo {
        #[deser(flatten)]
        bounds: TrustRegionBounds,
    },
    Morbo {
        #[deser(flatten)]
        bounds: TrustRegionBounds,
        #[deser(default = default_regions())]
        regions: usize,
        rescalarize: Rescalarize,
        clip: bool,
    },
    Reliability {
        #[deser(flatten)]
        bounds: TrustRegionBounds,
        #[deser(flatten)]
        policy: crate::ReliabilityPolicy,
    },
}

impl Default for TrustRegionSpec {
    fn default() -> Self {
        Self::Turbo {
            bounds: TrustRegionBounds::default(),
        }
    }
}

impl TrustRegionSpec {
    pub fn bounds(&self) -> &TrustRegionBounds {
        match self {
            Self::Turbo { bounds }
            | Self::Morbo { bounds, .. }
            | Self::Reliability { bounds, .. } => bounds,
        }
    }

    pub fn bounds_mut(&mut self) -> &mut TrustRegionBounds {
        match self {
            Self::Turbo { bounds }
            | Self::Morbo { bounds, .. }
            | Self::Reliability { bounds, .. } => bounds,
        }
    }
}
