use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Output};

mod tune;

const VERSION: &str = "0.1.1";

const REPAIR_HOOKS: &[&str] = &[
    "trailing-whitespace",
    "end-of-file-fixer",
    "repair-rust",
    "repair-python",
    "repair-python-format",
];

const CHECK_HOOKS: &[&str] = &[
    "trailing-whitespace",
    "end-of-file-fixer",
    "check-yaml",
    "check-json",
    "check-added-large-files",
    "check-rust",
    "check-python",
    "check-python-format",
    "kiss-check",
];

const FULL_BUILD_TARGETS: &[&str] = &[
    "//rust/crates/bpann:bpann",
    "//rust/crates/bpann:bpann-unit",
    "//rust/crates/dev-cli:ennx",
    "//rust/crates/dev-cli:ennx-test",
    "//rust/crates/ennx-py:ennx-py",
    "//rust/crates/ennx:ennx",
    "//rust/crates/ennx:ennx-unit",
];

#[derive(Debug, Clone, PartialEq, Eq)]
enum Mode {
    Dev { full: bool },
    Ci,
    Wheel,
    Tune(PathBuf),
    Help,
    Version,
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => ExitCode::from(code),
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<u8, String> {
    let root = repo_root()?;
    match parse_mode(&env::args().skip(1).collect::<Vec<_>>())? {
        Mode::Help => {
            print_help();
            Ok(0)
        }
        Mode::Version => {
            println!("ennx {VERSION}");
            Ok(0)
        }
        Mode::Tune(path) => tune::run(&root, &[path.to_string_lossy().into_owned()]),
        Mode::Wheel => run_wheel(&root),
        Mode::Ci => run_ci(&root),
        Mode::Dev { full } => run_dev(&root, full),
    }
}

fn parse_mode(args: &[String]) -> Result<Mode, String> {
    let Some(command) = args.first().map(String::as_str) else {
        return Ok(Mode::Help);
    };
    match command {
        "help" | "--help" | "-h" => Ok(Mode::Help),
        "version" | "--version" | "-V" => Ok(Mode::Version),
        "dev" => parse_dev(args),
        "ci" if args.len() == 1 => Ok(Mode::Ci),
        "wheel" if args.len() == 1 => Ok(Mode::Wheel),
        "tune" => parse_tune(args),
        legacy => Err(format!(
            "unknown ennx command {legacy:?}\n\n{}",
            help_text()
        )),
    }
}

fn parse_dev(args: &[String]) -> Result<Mode, String> {
    let mut full = false;
    for arg in &args[1..] {
        match arg.as_str() {
            "--full" => {
                if full {
                    return Err("ennx dev accepts --full at most once".to_string());
                }
                full = true;
            }
            other => {
                return Err(format!(
                    "unknown ennx dev flag {other:?}\n\n{}",
                    help_text()
                ));
            }
        }
    }
    Ok(Mode::Dev { full })
}

fn parse_tune(args: &[String]) -> Result<Mode, String> {
    if args.len() != 2 {
        return Err("usage: ennx tune CONFIG.toml".to_string());
    }
    Ok(Mode::Tune(PathBuf::from(&args[1])))
}

fn run_dev(root: &Path, full: bool) -> Result<u8, String> {
    let _initial_diff = capture_jj_diff(root)?;
    let mut changed_paths = discover_changed_paths(root)?;
    if !changed_paths.is_empty() {
        run_prek(root, false, &changed_paths, REPAIR_HOOKS)?;
        changed_paths = discover_changed_paths(root)?;
    }

    if full {
        run_prek(root, true, &[], CHECK_HOOKS)?;
        run_buck_graph(root, &buck_graph_for_full_mode())?;
        run_python_tests(root)?;
    } else {
        run_prek(root, false, &changed_paths, CHECK_HOOKS)?;
        let graph = affected_graph(root, &changed_paths)?;
        run_buck_graph(root, &graph)?;
        if needs_python_tests(&changed_paths) {
            run_python_tests(root)?;
        }
    }

    let final_diff = capture_jj_diff(root)?;
    stamp_successful_diff(root, final_diff)?;
    Ok(0)
}

fn run_ci(root: &Path) -> Result<u8, String> {
    let _initial_diff = capture_jj_diff(root)?;
    run_prek(root, true, &[], CHECK_HOOKS)?;
    run_buck_graph(root, &buck_graph_for_full_mode())?;
    run_python_tests(root)?;
    let final_diff = capture_jj_diff(root)?;
    stamp_successful_diff(root, final_diff)?;
    Ok(0)
}

fn run_wheel(root: &Path) -> Result<u8, String> {
    command_status(root, "tools/buck2-wheel-verify", &[])
}

fn run_python_tests(root: &Path) -> Result<u8, String> {
    command_status(root, "tools/buck2-wheel-verify", &["--tests"])
}

fn needs_python_tests(paths: &[PathBuf]) -> bool {
    paths.iter().any(|path| {
        let path = path.to_string_lossy();
        path == "pixi.toml"
            || path == "pixi.lock"
            || path == "pytest.ini"
            || path == "BUCK"
            || path == "buck2/wheel/pack.rs"
            || path == "tools/buck2-wheel-verify"
            || path.starts_with("src/")
            || path.starts_with("tests/")
            || path.starts_with("rust/crates/ennx/")
            || path.starts_with("rust/crates/ennx-py/")
    })
}

fn run_buck_graph(root: &Path, graph: &[String]) -> Result<u8, String> {
    if graph.is_empty() {
        return Ok(0);
    }
    let build_targets = graph
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    command_targets(
        root,
        "./buck2w",
        &[
            "--isolation-dir",
            "dev",
            "build",
            "--local-only",
            "--num-threads",
            "4",
            "-M",
            "none",
        ],
        &build_targets,
    )?;
    let tests = build_targets
        .iter()
        .copied()
        .filter(|target| is_test_target(target))
        .collect::<Vec<_>>();
    if !tests.is_empty() {
        command_targets(
            root,
            "./buck2w",
            &[
                "--isolation-dir",
                "dev",
                "test",
                "--local-only",
                "--num-threads",
                "4",
            ],
            &tests,
        )?;
    }
    Ok(0)
}

fn buck_graph_for_full_mode() -> Vec<String> {
    let mut targets = FULL_BUILD_TARGETS
        .iter()
        .map(|target| (*target).to_string())
        .collect::<Vec<_>>();
    if cfg!(target_os = "macos") {
        targets.push("//rust/crates/ennx:knn_metal_test".to_string());
    }
    targets
}

fn affected_graph(root: &Path, changed_paths: &[PathBuf]) -> Result<Vec<String>, String> {
    let mut targets = BTreeSet::new();
    for path in changed_paths {
        if let Some(fallback) = fallback_targets(path) {
            for target in fallback {
                targets.insert((*target).to_string());
            }
            continue;
        }
        if root.join(path).exists() {
            for target in owner_targets(root, path)? {
                targets.insert(target);
            }
        }
    }
    let mut expanded = targets.into_iter().collect::<BTreeSet<_>>();
    let mut queue = expanded.iter().cloned().collect::<Vec<_>>();
    while let Some(target) = queue.pop() {
        for downstream in downstream_targets(&target) {
            if expanded.insert(downstream.clone()) {
                queue.push(downstream);
            }
        }
    }
    Ok(expanded.into_iter().collect())
}

fn fallback_targets(path: &Path) -> Option<&'static [&'static str]> {
    let path = path.to_string_lossy();
    match path.as_ref() {
        "BUCK"
        | "BUILD.bazel"
        | "Cargo.Bazel.lock"
        | "Cargo.lock"
        | "Cargo.toml"
        | "Reindeer.toml"
        | "prek.toml" => {
            Some(FULL_BUILD_TARGETS)
        }
        "rust/BUCK" | "rust/BUILD.bazel" | "pixi.lock" | "pixi.toml" => {
            Some(FULL_BUILD_TARGETS)
        }
        _ if path.starts_with("rust/crates/") && path.ends_with("Cargo.toml") => {
            Some(FULL_BUILD_TARGETS)
        }
        _ if path.starts_with("rust/crates/") && path.ends_with("BUCK") => Some(FULL_BUILD_TARGETS),
        _ if path.starts_with("rust/crates/") && path.ends_with("BUILD.bazel") => {
            Some(FULL_BUILD_TARGETS)
        }
        _ if path.starts_with("rust/crates/bpann/") => Some(&["//rust/crates/bpann:bpann"]),
        _ if path.starts_with("rust/crates/dev-cli/") => {
            Some(&["//rust/crates/dev-cli:ennx", "//rust/crates/dev-cli:ennx-test"])
        }
        _ if path.starts_with("rust/crates/ennx-py/") => {
            Some(&["//rust/crates/ennx-py:ennx-py"])
        }
        _ if path.starts_with("rust/crates/ennx/") => Some(&["//rust/crates/ennx:ennx"]),
        _ if path.starts_with("cuda/") => Some(FULL_BUILD_TARGETS),
        _ if path.starts_with("buck2/") => Some(FULL_BUILD_TARGETS),
        _ => None,
    }
}

fn downstream_targets(target: &str) -> Vec<String> {
    let target = normalize_target(target);
    match target.as_str() {
        "//rust/crates/bpann:bpann-source" => vec!["//rust/crates/bpann:bpann".to_string()],
        "//rust/crates/bpann:bpann" => vec![
            "//rust/crates/bpann:bpann-unit".to_string(),
            "//rust/crates/dev-cli:ennx".to_string(),
            "//rust/crates/ennx-py:ennx-py".to_string(),
            "//rust/crates/ennx:ennx".to_string(),
            "//rust/crates/ennx:ennx-unit".to_string(),
        ],
        "//rust/crates/bpann:bpann-unit" => vec![],
        "//rust/crates/dev-cli:ennx" => vec![],
        "//rust/crates/ennx-py:python-source" => {
            vec!["//rust/crates/ennx-py:ennx-py".to_string()]
        }
        "//rust/crates/ennx-py:ennx-py" => vec![],
        "//rust/crates/ennx:ennx-source" => vec!["//rust/crates/ennx:ennx".to_string()],
        "//rust/crates/ennx:ennx" => {
            let mut targets = vec![
                "//rust/crates/dev-cli:ennx".to_string(),
                "//rust/crates/ennx-py:ennx-py".to_string(),
            ];
            targets.push("//rust/crates/ennx:ennx-unit".to_string());
            if cfg!(target_os = "macos") {
                targets.push("//rust/crates/ennx:knn_metal_test".to_string());
            }
            targets
        }
        "//rust/crates/ennx:ennx-unit" => vec![],
        "//rust/crates/ennx:knn_metal_test" => vec![],
        other if other.starts_with("//") => vec![],
        _ => vec![],
    }
}

fn is_test_target(target: &str) -> bool {
    matches!(
        normalize_target(target).as_str(),
        "//rust/crates/bpann:bpann-unit"
            | "//rust/crates/dev-cli:ennx-test"
            | "//rust/crates/ennx:ennx-unit"
            | "//rust/crates/ennx:knn_metal_test"
    )
}

fn owner_targets(root: &Path, path: &Path) -> Result<Vec<String>, String> {
    let query = format!("owner({})", path.to_string_lossy());
    let output = buck2_output(root, "uquery", &[query.as_str()])?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        if stderr.contains("No owner was found") {
            return Ok(vec![]);
        }
        return Err(format!(
            "buck2 uquery owner({}) failed:\n{}",
            path.display(),
            stderr.trim()
        ));
    }
    let targets = String::from_utf8(output.stdout).map_err(|error| error.to_string())?;
    Ok(targets
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(normalize_target)
        .filter(|target| is_dev_target(target))
        .collect())
}

fn is_dev_target(target: &str) -> bool {
    target.starts_with("//rust/") || target.starts_with("//cuda")
}

fn discover_changed_paths(root: &Path) -> Result<Vec<PathBuf>, String> {
    let output = command_output(root, "jj", &["diff", "--name-only"])?;
    if !output.status.success() {
        return Err(format!(
            "jj diff --name-only failed:\n{}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let text = String::from_utf8(output.stdout).map_err(|error| error.to_string())?;
    Ok(text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(PathBuf::from)
        .collect())
}

fn capture_jj_diff(root: &Path) -> Result<String, String> {
    let output = command_output(root, "jj", &["diff", "--git", "--color=never"])?;
    if !output.status.success() {
        return Err(format!(
            "jj diff --git failed:\n{}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    String::from_utf8(output.stdout).map_err(|error| error.to_string())
}

fn stamp_successful_diff(root: &Path, diff: String) -> Result<u8, String> {
    let stamp = root.join(".cache/ennx/dev-ok.diff");
    let parent = stamp.parent().ok_or("invalid dev stamp path")?;
    fs::create_dir_all(parent)
        .map_err(|error| format!("create {}: {error}", parent.display()))?;
    fs::write(&stamp, diff).map_err(|error| format!("write {}: {error}", stamp.display()))?;
    println!("verified jj diff stamped at {}", stamp.display());
    Ok(0)
}

fn run_prek(root: &Path, all_files: bool, files: &[PathBuf], hooks: &[&str]) -> Result<u8, String> {
    let mut args = vec!["run"];
    args.extend(hooks.iter().copied());
    args.extend([
        "--config",
        "prek.toml",
        "--fail-fast",
        "--show-diff-on-failure",
    ]);
    let existing_files = files
        .iter()
        .filter(|file| root.join(file).exists())
        .map(|file| file.to_str().ok_or("changed path is not valid UTF-8"))
        .collect::<Result<Vec<_>, _>>()?;
    if all_files || existing_files.is_empty() {
        args.push("--all-files");
    } else {
        args.push("--files");
        args.extend(existing_files);
    }
    command_status(root, "tools/prek", &args)
}

fn command_targets(
    root: &Path,
    program: &str,
    args: &[&str],
    targets: &[&str],
) -> Result<u8, String> {
    let mut full_args = args.to_vec();
    full_args.extend(targets.iter().copied());
    command_status(root, program, &full_args)
}

fn command_status(root: &Path, program: &str, args: &[&str]) -> Result<u8, String> {
    let status = Command::new(program)
        .current_dir(root)
        .args(args)
        .status()
        .map_err(|error| format!("start {program}: {error}"))?;
    if status.success() {
        Ok(0)
    } else {
        Err(format!("{program} exited with {status}"))
    }
}

fn command_output(root: &Path, program: &str, args: &[&str]) -> Result<Output, String> {
    Command::new(program)
        .current_dir(root)
        .args(args)
        .output()
        .map_err(|error| format!("start {program}: {error}"))
}

fn buck2_output(root: &Path, command: &str, args: &[&str]) -> Result<Output, String> {
    let mut full_args = vec!["--isolation-dir", "dev", command];
    full_args.extend(args.iter().copied());
    command_output(root, "./buck2w", &full_args)
}

fn repo_root() -> Result<PathBuf, String> {
    env::current_dir().map_err(|error| error.to_string())
}

fn normalize_target(target: &str) -> String {
    target
        .strip_prefix("root")
        .unwrap_or(target)
        .to_string()
}

fn help_text() -> &'static str {
    "Usage: ennx <COMMAND>\n\nCommands:\n  dev [--full]   Repair and verify the current jj diff, including affected Buck2 and Python tests.\n  ci             Run full source, Buck2, wheel, and Python verification without repair.\n  wheel          Build the current platform wheel and verify the installed artifact.\n  tune CONFIG.toml\n                 Run the tuning workflow described by CONFIG.toml.\n  help           Show this help text.\n  version        Print the CLI version."
}

fn print_help() {
    println!("{}", help_text());
}

#[cfg(test)]
#[path = "tests_main.rs"]
mod tests;
