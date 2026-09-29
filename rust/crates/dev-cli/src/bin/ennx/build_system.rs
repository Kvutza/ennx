use std::env;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BuildSystem {
    Bazel,
    Buck2,
}

pub(crate) struct BuiltArtifact {
    pub path: PathBuf,
    pub log: Vec<u8>,
}

pub(crate) fn active() -> Result<BuildSystem, String> {
    match env::var("ENNX_BUILD_SYSTEM").as_deref() {
        Ok("bazel") => Ok(BuildSystem::Bazel),
        Ok("buck2") | Err(env::VarError::NotPresent) => Ok(BuildSystem::Buck2),
        Ok(value) => Err(format!(
            "ENNX_BUILD_SYSTEM must be 'buck2' or 'bazel', got {value:?}"
        )),
        Err(error) => Err(format!("read ENNX_BUILD_SYSTEM: {error}")),
    }
}

impl BuildSystem {
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Bazel => "bazel",
            Self::Buck2 => "buck2",
        }
    }

    pub(crate) fn run(self, root: &Path, target: &str, args: &[String]) -> Result<(), String> {
        let status = self
            .run_command(root, target, args)
            .status()
            .map_err(|error| format!("start {}: {error}", self.name()))?;
        if status.success() {
            Ok(())
        } else {
            Err(format!("{} exited with {status}", self.name()))
        }
    }

    pub(crate) fn run_output(
        self,
        root: &Path,
        target: &str,
        args: &[String],
    ) -> Result<String, String> {
        let output = self
            .run_command(root, target, args)
            .output()
            .map_err(|error| format!("start {}: {error}", self.name()))?;
        checked_stdout(self.name(), output)
    }

    pub(crate) fn build_artifact(self, root: &Path, target: &str) -> Result<BuiltArtifact, String> {
        match self {
            Self::Buck2 => buck_artifact(root, target),
            Self::Bazel => bazel_artifact(root, target),
        }
    }

    pub(crate) fn run_command(self, root: &Path, target: &str, args: &[String]) -> Command {
        let mut command = match self {
            Self::Buck2 => {
                let mut command = Command::new("./buck2w");
                command.args(["--isolation-dir", "dev", "run", target, "--"]);
                command
            }
            Self::Bazel => {
                let mut command = Command::new("bazel");
                command.args([
                    "run",
                    "--config=release",
                    "--config=constrained",
                    target,
                    "--",
                ]);
                command
            }
        };
        command.current_dir(root).args(args);
        command
    }
}

fn buck_artifact(root: &Path, target: &str) -> Result<BuiltArtifact, String> {
    let output = Command::new("./buck2w")
        .current_dir(root)
        .args(["build", target, "--show-full-output"])
        .output()
        .map_err(|error| format!("start buck2: {error}"))?;
    let log = combined_log(&output);
    if !output.status.success() {
        return Err("Buck2 worker build failed".into());
    }
    let stdout = String::from_utf8(output.stdout)
        .map_err(|error| format!("Buck2 output was not UTF-8: {error}"))?;
    let path = stdout
        .lines()
        .find_map(|line| {
            line.strip_prefix(&format!("root{target} "))
                .or_else(|| line.strip_prefix(&format!("{target} ")))
        })
        .map(str::trim)
        .map(PathBuf::from)
        .ok_or("Buck2 did not return the worker executable path")?;
    Ok(BuiltArtifact {
        path: absolute(root, path),
        log,
    })
}

fn bazel_artifact(root: &Path, target: &str) -> Result<BuiltArtifact, String> {
    let output = Command::new("bazel")
        .current_dir(root)
        .args(["build", "--config=release", "--config=constrained", target])
        .output()
        .map_err(|error| format!("start bazel: {error}"))?;
    let mut log = combined_log(&output);
    if !output.status.success() {
        return Err("Bazel worker build failed".into());
    }
    let query = Command::new("bazel")
        .current_dir(root)
        .args([
            "cquery",
            "--config=release",
            "--config=constrained",
            target,
            "--output=files",
        ])
        .output()
        .map_err(|error| format!("query Bazel artifact: {error}"))?;
    log.extend_from_slice(&combined_log(&query));
    let relative = checked_stdout("bazel cquery", query)?
        .lines()
        .find(|line| !line.trim().is_empty())
        .map(str::trim)
        .map(PathBuf::from)
        .ok_or("Bazel did not return the worker executable path")?;
    let execution_root = checked_stdout(
        "bazel info",
        Command::new("bazel")
            .current_dir(root)
            .args(["info", "execution_root"])
            .output()
            .map_err(|error| format!("query Bazel execution root: {error}"))?,
    )?;
    Ok(BuiltArtifact {
        path: absolute(Path::new(execution_root.trim()), relative),
        log,
    })
}

fn checked_stdout(program: &str, output: Output) -> Result<String, String> {
    if !output.status.success() {
        return Err(format!(
            "{program} exited with {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    String::from_utf8(output.stdout)
        .map_err(|error| format!("{program} output was not UTF-8: {error}"))
}

fn combined_log(output: &Output) -> Vec<u8> {
    let mut log = output.stdout.clone();
    log.extend_from_slice(&output.stderr);
    log
}

fn absolute(root: &Path, path: PathBuf) -> PathBuf {
    if path.is_absolute() {
        path
    } else {
        root.join(path)
    }
}
