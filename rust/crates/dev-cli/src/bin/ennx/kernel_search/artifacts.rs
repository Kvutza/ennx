use super::Candidate;
use deser::Serialize;
use ennx::config::{ConfigOverrides, KernelTrial};
use ennx_wire::json::{Value, json};
use std::fs::{self, File, OpenOptions};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

pub(super) struct Archive {
    pub path: PathBuf,
    history_root: PathBuf,
    _lock: File,
}

pub(super) fn write_json(path: &Path, value: &impl Serialize) -> Result<(), String> {
    let bytes = ennx_wire::json::pretty_vec(value).map_err(|error| error.to_string())?;
    let temporary = path.with_extension("json.tmp");
    fs::write(&temporary, bytes).map_err(|error| error.to_string())?;
    fs::rename(temporary, path).map_err(|error| error.to_string())
}

impl Archive {
    pub fn create(root: &Path, output: &Path) -> Result<Self, String> {
        let cache = root.join(".cache/ennx");
        fs::create_dir_all(&cache).map_err(|error| error.to_string())?;
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(cache.join("kernel-search.lock"))
            .map_err(|error| error.to_string())?;
        lock.try_lock()
            .map_err(|_| "another kernel search holds the workspace GPU lock")?;
        fs::create_dir_all(output).map_err(|error| error.to_string())?;
        let history_root = output.canonicalize().map_err(|error| error.to_string())?;
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| error.to_string())?
            .as_nanos();
        let path = history_root.join(format!("search-{stamp}-{}", std::process::id()));
        fs::create_dir(&path).map_err(|error| error.to_string())?;
        fs::create_dir(path.join("inputs")).map_err(|error| error.to_string())?;
        Ok(Self {
            path,
            history_root,
            _lock: lock,
        })
    }

    pub fn freeze_dataset(&self, baseline: &mut ConfigOverrides) -> Result<(), String> {
        let source = baseline
            .dataset
            .as_ref()
            .ok_or("baseline dataset was not resolved")?;
        let target = self.path.join("inputs/dataset.ennxptn");
        fs::copy(source, &target).map_err(|error| format!("snapshot dataset: {error}"))?;
        read_only(&target)?;
        baseline.dataset = Some(target);
        Ok(())
    }

    pub fn freeze_candidates(
        &self,
        parent: &Path,
        candidates: &[Candidate],
    ) -> Result<Vec<KernelTrial>, String> {
        for operator in ["pisa", "moe", "decode", "readout", "perturb", "mhc"] {
            fs::write(
                self.path
                    .join(format!("inputs/production-{operator}.metal")),
                KernelTrial::source(operator)?,
            )
            .map_err(|error| error.to_string())?;
        }
        candidates
            .iter()
            .map(|candidate| {
                let body = match &candidate.source {
                    Some(path) => fs::read_to_string(parent.join(path))
                        .map_err(|error| format!("read {}: {error}", path.display()))?,
                    None => KernelTrial::source(&candidate.operator)?.to_owned(),
                };
                let mut source = candidate
                    .defines
                    .iter()
                    .map(|name| format!("#define {name}\n"))
                    .collect::<String>();
                source.push_str(&body);
                let path = self
                    .path
                    .join(format!("inputs/candidate-{}.metal", candidate.name));
                fs::write(&path, source).map_err(|error| error.to_string())?;
                read_only(&path)?;
                let mut trial = KernelTrial::default();
                match candidate.operator.as_str() {
                    "pisa" => trial.pisa = Some(path),
                    "moe" => trial.moe = Some(path),
                    "decode" => trial.decode = Some(path),
                    "readout" => trial.readout = Some(path),
                    "perturb" => trial.perturb = Some(path),
                    "mhc" => trial.mhc = Some(path),
                    _ => return Err("unsupported operator".into()),
                }
                Ok(trial)
            })
            .collect()
    }

    pub fn history(&self) -> Result<Vec<Value>, String> {
        let mut paths = fs::read_dir(&self.history_root)
            .map_err(|error| error.to_string())?
            .map(|entry| entry.map(|entry| entry.path()))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| error.to_string())?;
        paths.sort();
        let mut history = Vec::new();
        for path in paths.into_iter().rev().filter(|path| path != &self.path) {
            if !path.join("feedback.json").is_file() && !path.join("failure.json").is_file() {
                continue;
            }
            let read = |name| -> Result<Option<Value>, String> {
                let file = path.join(name);
                if !file.exists() {
                    return Ok(None);
                }
                let bytes = fs::read(&file).map_err(|error| error.to_string())?;
                ennx_wire::json::from_slice(&bytes)
                    .map(Some)
                    .map_err(|error| format!("invalid history {}: {error}", file.display()))
            };
            history.push(json!({"artifact": path, "feedback": read("feedback.json")?, "failure": read("failure.json")?}));
            if history.len() == 32 {
                break;
            }
        }
        Ok(history)
    }

    pub fn write_experiment(&self, path: &Path, config: &ConfigOverrides) -> Result<(), String> {
        let text = ennx::config::TuneSpec::from_overrides(config)?.to_toml()?;
        fs::write(path, text).map_err(|error| error.to_string())
    }

    pub fn build_worker(&self, root: &Path) -> Result<PathBuf, String> {
        // Reject stale embedded defaults before comparing them with a fresh worker.
        for (operator, file) in [
            ("pisa", "fbt_pisa1.metal"),
            ("moe", "fbt_moe_routing_tensorops.metal"),
            ("decode", "fbt_decode.metal"),
            ("readout", "fbt_moe.metal"),
            ("perturb", "fbt_denoise.metal"),
            ("mhc", "fbt_moe.metal"),
        ] {
            let path = root.join("rust/crates/ennx/src").join(file);
            if fs::read_to_string(path).map_err(|error| error.to_string())?
                != KernelTrial::source(operator)?
            {
                return Err("shader sources changed since the CLI build; rerun ./ennx tune".into());
            }
        }
        println!("Building the full-loop worker once; candidate shaders compile at runtime.");
        let build_system = crate::build_system::active()?;
        let artifact = build_system
            .build_artifact(root, "//rust/crates/ennx:turbo-enn-worker")
            .map_err(|error| format!("{error}; see build.log"))?;
        fs::write(self.path.join("build.log"), artifact.log).map_err(|error| error.to_string())?;
        let binary = artifact.path;
        let frozen = self.path.join("worker");
        fs::copy(&binary, &frozen).map_err(|error| format!("snapshot worker: {error}"))?;
        read_only(&frozen)?;
        let sources = self.path.join("sources");
        snapshot(&root.join("rust/crates/ennx/src"), &sources)?;
        write_json(
            &self.path.join("worker.json"),
            &json!({
                "original": binary, "snapshot": frozen,
                "build_log": "build.log", "hardware": hardware(),
                "build_system": build_system.name(),
                "source_snapshot": sources,
            }),
        )?;
        Ok(frozen)
    }
}

impl Drop for Archive {
    fn drop(&mut self) {
        let _ = self._lock.unlock();
    }
}

fn read_only(path: &Path) -> Result<(), String> {
    let mut permissions = fs::metadata(path)
        .map_err(|error| error.to_string())?
        .permissions();
    permissions.set_readonly(true);
    fs::set_permissions(path, permissions).map_err(|error| error.to_string())
}

fn snapshot(source: &Path, target: &Path) -> Result<(), String> {
    fs::create_dir(target).map_err(|error| error.to_string())?;
    for entry in fs::read_dir(source).map_err(|error| error.to_string())? {
        let entry = entry.map_err(|error| error.to_string())?;
        let kind = entry.file_type().map_err(|error| error.to_string())?;
        let destination = target.join(entry.file_name());
        if kind.is_dir() {
            snapshot(&entry.path(), &destination)?;
        } else if kind.is_file() {
            fs::copy(entry.path(), destination).map_err(|error| error.to_string())?;
        }
    }
    Ok(())
}

pub(super) fn hardware() -> Value {
    let read = |program: &str, args: &[&str]| {
        Command::new(program)
            .args(args)
            .output()
            .ok()
            .filter(|result| result.status.success())
            .map(|result| String::from_utf8_lossy(&result.stdout).trim().to_string())
    };
    json!({
        "os": read("sw_vers", &[]), "chip": read("sysctl", &["-n", "machdep.cpu.brand_string"]),
        "power_source": read("pmset", &["-g", "batt"]), "power_policy": read("pmset", &["-g", "custom"]),
        "xcode": read("xcodebuild", &["-version"]),
    })
}
