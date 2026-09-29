//! Version 2 experiment files deserialize directly into closed, typed sections.
use super::*;

use super::{spec_execution::*, spec_optimizer::*};

/// Closed experiment syntax shared by authored inputs and frozen run records.
/// Authored inputs may omit paths and defaults; resolution materializes them in
/// the run record without rewriting the source file.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[deser(deny_unknown_fields, rename_all = "kebab-case")]
pub struct TuneSpec {
    pub version: u32,
    pub experiment: TurboEnnExperiment,
    pub model: Option<PretrainModel>,
    pub corpus: Option<PretrainCorpus>,
    pub output: Option<PathBuf>,
    #[deser(default)]
    pub run: RunSpec,
    #[deser(default)]
    pub data: DataSpec,
    #[deser(default)]
    pub proposal: ProposalSpec,
    #[deser(default)]
    pub enn: EnnSpec,
    #[deser(default)]
    pub acquisition: AcquisitionSpec,
    #[deser(default)]
    pub trust_region: TrustRegionSpec,
    #[deser(default)]
    pub objective: ObjectiveSpec,
    #[deser(default)]
    pub seeds: SeedSpec,
    #[deser(default)]
    pub diagnostics: DiagnosticSpec,
    pub generation: Option<GenerationConfig>,
}

impl TuneSpec {
    pub fn parse(text: &str) -> Result<Self, String> {
        let spec: Self = ennx_wire::toml::from_str(text)
            .map_err(|error| format!("invalid experiment: {error}"))?;
        if spec.version != 2 {
            return Err("typed experiment version must be 2".into());
        }
        spec.overrides()?;
        Ok(spec)
    }

    pub fn to_toml(&self) -> Result<String, String> {
        self.overrides()?;
        ennx_wire::toml::pretty_string(self).map_err(|error| error.to_string())
    }

    /// The corpus preparer receives a concrete request validated by Rust. Its
    /// historical wire format is independent of the public experiment syntax.
    pub fn corpus_request(&self, parent: &std::path::Path) -> Result<String, String> {
        #[derive(Serialize)]
        struct Request<'a> {
            version: u32,
            #[deser(flatten)]
            config: &'a ConfigOverrides,
        }
        let mut config = self.overrides()?;
        if let Some(generation) = &mut config.generation {
            generation.resolve(parent)?;
        }
        ennx_wire::toml::pretty_string(&Request {
            version: 1,
            config: &config,
        })
        .map_err(|error| error.to_string())
    }
}
