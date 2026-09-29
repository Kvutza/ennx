use super::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[deser(default, deny_unknown_fields, rename_all = "kebab-case")]
pub struct RunSpec {
    pub rounds: u32,
    pub reps: u32,
    pub target_ms: u32,
    pub selection: Option<PretrainSelection>,
    pub validation_interval: Option<u32>,
}

impl Default for RunSpec {
    fn default() -> Self {
        let config = ConfigOverrides::default();
        Self {
            rounds: config.rounds(),
            reps: config.reps(),
            target_ms: config.target_ms(),
            selection: None,
            validation_interval: None,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[deser(default, deny_unknown_fields, rename_all = "kebab-case")]
pub struct DataSpec {
    pub train: Option<PathBuf>,
    pub validation: Option<PathBuf>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[deser(default, deny_unknown_fields, rename_all = "kebab-case")]
pub struct SeedSpec {
    pub model: Option<u64>,
    pub reference: Option<u64>,
    pub proposal: Option<u64>,
    pub acquisition: Option<u64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[deser(default, deny_unknown_fields, rename_all = "kebab-case")]
pub struct DiagnosticSpec {
    pub trace: bool,
    pub stage_samples: Option<u32>,
    pub kernels: Option<KernelTrial>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[deser(default, deny_unknown_fields, rename_all = "kebab-case")]
pub struct ObjectiveSpec {
    pub reference: Option<ObjectiveReference>,
}
