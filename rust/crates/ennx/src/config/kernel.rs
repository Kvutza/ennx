//! Explicit, opt-in shader inputs for full-loop kernel experiments.

use deser::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// A trial changes shader implementations, not the model or optimizer workload.
/// An empty trial measures production with the same validation instrumentation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[deser(deny_unknown_fields)]
pub struct KernelTrial {
    pub pisa: Option<PathBuf>,
    pub moe: Option<PathBuf>,
}

impl KernelTrial {
    pub(crate) fn resolve(&mut self, parent: &Path) -> Result<(), String> {
        for path in [&mut self.pisa, &mut self.moe].into_iter().flatten() {
            *path = parent
                .join(&*path)
                .canonicalize()
                .map_err(|error| format!("kernel source {}: {error}", path.display()))?;
        }
        Ok(())
    }

    /// Embedded sources are the defaults actually compiled into this executable.
    pub fn source(operator: &str) -> Result<&'static str, String> {
        match operator {
            "pisa" => Ok(include_str!("../fbt_pisa1.metal")),
            "moe" => Ok(include_str!("../fbt_moe_routing_tensorops.metal")),
            _ => Err(format!(
                "unsupported kernel operator {operator:?}; use pisa or moe"
            )),
        }
    }
}
