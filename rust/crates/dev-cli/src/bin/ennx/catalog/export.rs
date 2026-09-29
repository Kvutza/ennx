//! Relocate inputs without opening them; give each export independent outputs.
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use ennx::config::{GenerationReward, TuneSpec};

use super::Catalog;

fn absolute(path: &mut Option<PathBuf>, parent: &Path) {
    if let Some(path) = path {
        *path = parent.join(&*path);
    }
}

pub(super) fn rebase(spec: &mut TuneSpec, parent: &Path) {
    absolute(&mut spec.output, parent);
    absolute(&mut spec.data.train, parent);
    absolute(&mut spec.data.validation, parent);
    if let Some(kernels) = &mut spec.diagnostics.kernels {
        absolute(&mut kernels.pisa, parent);
        absolute(&mut kernels.moe, parent);
    }
    if let Some(generation) = &mut spec.generation {
        absolute(&mut generation.checkpoint, parent);
        absolute(&mut generation.qualification_manifest, parent);
        absolute(&mut generation.episode_dataset, parent);
        absolute(&mut generation.save_checkpoint, parent);
        match &mut generation.reward {
            GenerationReward::CodeExecution {
                environment,
                interpreter,
                ..
            } => {
                *environment = parent.join(&*environment);
                *interpreter = parent.join(&*interpreter);
            }
            GenerationReward::FrozenQwen {
                checkpoint,
                tokenizer_program,
                ..
            } => {
                *checkpoint = parent.join(&*checkpoint);
                *tokenizer_program = parent.join(&*tokenizer_program);
            }
            GenerationReward::Command { program, .. }
            | GenerationReward::CommandObjectives { program, .. } => {
                *program = parent.join(&*program)
            }
            _ => {}
        }
    }
}

pub(super) fn write(
    out: &Path,
    catalog: &Catalog,
    manifest: &ennx_wire::json::Value,
    baseline: &TuneSpec,
) -> Result<(), String> {
    // Fail before writing anything if the directory already exists.
    fs::create_dir(out).map_err(|e| format!("create new catalog {}: {e}", out.display()))?;
    let root = out.canonicalize().map_err(|e| e.to_string())?;
    save(&root.join("baseline.toml"), &baseline.to_toml()?)?;
    for entry in &catalog.entries {
        let mut spec = entry.spec.clone();
        spec.output = Some(root.join("runs").join(&entry.id));
        if let Some(generation) = &mut spec.generation {
            if generation.save_checkpoint.is_some() {
                generation.save_checkpoint =
                    Some(root.join("runs").join(&entry.id).join("checkpoint.fp16"));
            }
        }
        save(&root.join(format!("{}.toml", entry.id)), &spec.to_toml()?)?;
    }
    // Manifest last is the completion marker. Partial exports are never campaigns.
    save(
        &root.join("manifest.json"),
        &ennx_wire::json::pretty_string(manifest).map_err(|e| e.to_string())?,
    )
}

fn save(path: &Path, text: &str) -> Result<(), String> {
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| e.to_string())?;
    file.write_all(text.as_bytes()).map_err(|e| e.to_string())
}
