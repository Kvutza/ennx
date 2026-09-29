//! Build metadata is a deterministic projection of Cargo, shared by both engines.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

use clap::Parser;
use deser::Deserialize;

#[derive(Parser)]
struct Options {
    #[arg(long)]
    check: bool,
    #[arg(long)]
    parity: bool,
}

#[derive(Deserialize)]
struct Metadata {
    workspace_root: PathBuf,
    packages: Vec<Package>,
}

#[derive(Deserialize)]
struct Package {
    name: String,
    version: String,
    manifest_path: PathBuf,
    edition: String,
    dependencies: Vec<Dependency>,
    targets: Vec<Target>,
}

#[derive(Deserialize)]
struct Dependency {
    name: String,
    kind: Option<String>,
    optional: bool,
    path: Option<PathBuf>,
    target: Option<String>,
}

#[derive(Deserialize)]
struct Target {
    name: String,
    kind: Vec<String>,
    src_path: PathBuf,
    #[deser(default, rename = "required-features")]
    required_features: Vec<String>,
}

fn main() -> ExitCode {
    match run(Options::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("build graph: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run(options: Options) -> Result<(), String> {
    if let Some(root) = std::env::var_os("BUILD_WORKSPACE_DIRECTORY") {
        std::env::set_current_dir(root).map_err(|error| error.to_string())?;
    }
    let output = capture(
        "cargo",
        &["metadata", "--no-deps", "--offline", "--format-version=1"],
    )?;
    let mut metadata: Metadata = ennx_wire::json::from_str(&output).map_err(|e| e.to_string())?;
    metadata.packages.sort_by(|a, b| a.name.cmp(&b.name));
    let generated = render(&metadata)?;
    let path = metadata.workspace_root.join("build/targets.bzl");
    if options.check {
        if fs::read_to_string(&path).map_err(|e| e.to_string())? != generated {
            return Err("Cargo projection is stale; run tools/build-parity --sync".into());
        }
    } else {
        fs::write(path, generated).map_err(|e| e.to_string())?;
    }
    if options.parity {
        parity(&metadata)?;
    }
    println!("Cargo build graph is current");
    Ok(())
}

fn external(package: &Package, development: bool) -> Vec<String> {
    package
        .dependencies
        .iter()
        .filter(|dep| {
            (dep.kind.is_none() || development && dep.kind.as_deref() == Some("dev"))
                && !dep.optional
                && dep.path.is_none()
                && dep.target.is_none()
        })
        .map(|dep| dep.name.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn packages(metadata: &Metadata) -> impl Iterator<Item = &Package> {
    let directory = metadata.workspace_root.join("rust/crates");
    metadata
        .packages
        .iter()
        .filter(move |package| package.manifest_path.starts_with(&directory))
}

fn render(metadata: &Metadata) -> Result<String, String> {
    let mut output = String::from(
        "\"\"\"Generated from Cargo metadata by tools/build-parity --sync. Do not edit.\"\"\"\n\n",
    );
    for package in packages(metadata) {
        let directory = package
            .manifest_path
            .parent()
            .ok_or("manifest has no parent")?;
        let stem = directory
            .file_name()
            .ok_or("package has no directory")?
            .to_string_lossy()
            .replace('-', "_")
            .to_uppercase();
        output.push_str(&format!(
            "{stem}_CRATES = {:?}\n\n{stem}_TEST_CRATES = {:?}\n\n",
            external(package, false),
            external(package, true)
        ));
        output.push_str(&format!("{stem}_PROGRAMS = [\n"));
        for target in &package.targets {
            if target
                .kind
                .iter()
                .any(|kind| kind == "bin" || kind == "example")
            {
                program(&mut output, package, target, directory)?;
            }
        }
        output.push_str("]\n\n");
    }
    let edition = packages(metadata)
        .next()
        .ok_or("workspace has no Rust packages")?
        .edition
        .clone();
    if packages(metadata).any(|package| package.edition != edition) {
        return Err("workspace packages use different Rust editions".into());
    }
    output.push_str(&format!("RUST_EDITION = {edition:?}\n"));
    Ok(output)
}

fn program(
    output: &mut String,
    package: &Package,
    target: &Target,
    directory: &Path,
) -> Result<(), String> {
    output.push_str(&format!(
        "    {{\n        \"version\": {:?},\n",
        package.version
    ));
    let root = target
        .src_path
        .strip_prefix(directory)
        .map_err(|e| e.to_string())?
        .to_string_lossy();
    let local = package
        .dependencies
        .iter()
        .filter(|dep| dep.kind.is_none() && !dep.optional)
        .filter_map(|dep| dep.path.as_ref()?.file_name()?.to_str().map(str::to_owned))
        .chain(
            package
                .targets
                .iter()
                .filter(|target| target.kind.iter().any(|kind| kind == "lib"))
                .map(|_| {
                    directory
                        .file_name()
                        .unwrap()
                        .to_string_lossy()
                        .into_owned()
                }),
        )
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    output.push_str(&format!("        \"name\": {:?},\n        \"crate\": {:?},\n        \"root\": {:?},\n        \"crate_deps\": {:?},\n        \"local_deps\": {:?},\n        \"required_features\": {:?},\n    }},\n", target.name, target.name.replace('-', "_"), root, external(package, target.kind.iter().any(|kind| kind == "example")), local, target.required_features));
    Ok(())
}

fn capture(program: &str, args: &[&str]) -> Result<String, String> {
    let output = Command::new(program)
        .args(args)
        .output()
        .map_err(|e| format!("start {program}: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "{program} {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    String::from_utf8(output.stdout).map_err(|e| e.to_string())
}

fn parity(metadata: &Metadata) -> Result<(), String> {
    let buck = capture("./buck2w", &["targets", "//rust/crates/..."])?;
    let bazel = capture("bazel", &["query", "//rust/crates/...", "--output=label"])?;
    let labels = |text: &str| {
        text.lines()
            .map(|line| line.trim_start_matches("root").to_owned())
            .collect::<BTreeSet<_>>()
    };
    let buck = labels(&buck);
    let bazel = labels(&bazel);
    for package in packages(metadata) {
        let directory = package
            .manifest_path
            .parent()
            .ok_or("manifest has no parent")?
            .strip_prefix(&metadata.workspace_root)
            .map_err(|e| e.to_string())?;
        if !metadata
            .workspace_root
            .join(directory)
            .join("BUILD.bazel")
            .is_file()
        {
            continue;
        }
        for target in &package.targets {
            if !target
                .kind
                .iter()
                .any(|kind| kind == "bin" || kind == "example")
                || target
                    .required_features
                    .iter()
                    .any(|feature| feature == "cuda")
            {
                continue;
            }
            let label = format!("//{}:{}", directory.display(), target.name);
            if !buck.contains(&label) || !bazel.contains(&label) {
                return Err(format!(
                    "Cargo target {label} is missing from Buck2 or Bazel"
                ));
            }
        }
    }
    println!("Buck2/Bazel Cargo targets agree");
    Ok(())
}
