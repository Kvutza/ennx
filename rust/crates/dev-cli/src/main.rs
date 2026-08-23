use std::env;
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

mod tune;

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
    let args = env::args().skip(1).collect::<Vec<_>>();
    if args.is_empty() || args[0] == "help" || args[0] == "--help" || args[0] == "-h" {
        print_help();
        return Ok(0);
    }

    match args[0].as_str() {
        "build" | "test" | "wheel" | "verify" | "ci" => project(&root, &args),
        "check" => check(&root, &args[1..]),
        "buck" => buck(&root, &args[1..]),
        "bazel" => bazel(&root, &args[1..]),
        "rust" => rust(&root, &args[1..]),
        "python" => python(&root, &args[1..]),
        "cuda" => cuda(&root, &args[1..]),
        "release" => release(&root, &args[1..]),
        "tune" => tune::run(&root, &args[1..]),
        "version" | "--version" | "-V" => {
            println!("ennx dev-cli 0.1.0");
            Ok(0)
        }
        command => Err(format!(
            "unknown ennx command {command:?}\n\n{}",
            help_text()
        )),
    }
}

fn project(root: &Path, args: &[String]) -> Result<u8, String> {
    match args[0].as_str() {
        "build" => buck_build(root),
        "test" => buck_test(root),
        "wheel" => buck_wheel(root),
        "verify" => verify(root),
        "ci" => ci(root),
        _ => unreachable!("project only receives known commands"),
    }
}

fn buck(root: &Path, args: &[String]) -> Result<u8, String> {
    match args.first().map(String::as_str) {
        Some("build") => buck_build(root),
        Some("test") => buck_test(root),
        Some("wheel") => buck_wheel(root),
        Some("ci") => ci(root),
        Some(command) => Err(format!("unknown ennx buck command {command:?}")),
        None => Err("missing ennx buck command".to_string()),
    }
}

fn bazel(root: &Path, args: &[String]) -> Result<u8, String> {
    match args.first().map(String::as_str) {
        Some("test") => command(
            root,
            "bazel",
            [
                "test",
                "//rust/crates/ennx:ennx_test",
                "--config=constrained",
            ],
        ),
        Some("build") => command(
            root,
            "bazel",
            ["build", "//rust/crates/ennx:ennx", "--config=constrained"],
        ),
        Some(command) => Err(format!("unknown ennx bazel command {command:?}")),
        None => Err("missing ennx bazel command".to_string()),
    }
}

fn rust(root: &Path, args: &[String]) -> Result<u8, String> {
    match args.first().map(String::as_str) {
        Some("fast") => rust_fast(root),
        Some("full") => rust_full(root),
        Some(command) => Err(format!("unknown ennx rust command {command:?}")),
        None => Err("missing ennx rust command".to_string()),
    }
}

fn python(root: &Path, args: &[String]) -> Result<u8, String> {
    match args.first().map(String::as_str) {
        Some("fast") => python_fast(root),
        Some("verify") | Some("wheel") => verify(root),
        Some(command) => Err(format!("unknown ennx python command {command:?}")),
        None => Err("missing ennx python command".to_string()),
    }
}

fn cuda(root: &Path, args: &[String]) -> Result<u8, String> {
    match args.first().map(String::as_str) {
        Some("wheel") => cuda_wheel(root),
        Some("parity") => cuda_parity(root),
        Some(command) => Err(format!("unknown ennx cuda command {command:?}")),
        None => Err("missing ennx cuda command".to_string()),
    }
}

fn release(root: &Path, args: &[String]) -> Result<u8, String> {
    match args.first().map(String::as_str) {
        Some("upload") => release_upload(root, &args[1..]),
        Some(command) => Err(format!("unknown ennx release command {command:?}")),
        None => Err("missing ennx release command".to_string()),
    }
}

fn release_upload(root: &Path, args: &[String]) -> Result<u8, String> {
    let (tag, wheels) = args
        .split_first()
        .ok_or_else(|| "usage: ennx release upload vX.Y.Z WHEEL...".to_string())?;
    if !tag.starts_with('v') {
        return Err("release tag must start with v".to_string());
    }
    if wheels.is_empty() {
        return Err("provide at least one wheel".to_string());
    }

    command(
        root,
        "gh",
        std::iter::once("release")
            .chain(std::iter::once("upload"))
            .chain(std::iter::once(tag.as_str()))
            .chain(wheels.iter().map(String::as_str))
            .chain(std::iter::once("--clobber")),
    )
}

fn ci(root: &Path) -> Result<u8, String> {
    for step in [buck_build, buck_test, buck_wheel, verify] {
        let code = step(root)?;
        if code != 0 {
            return Ok(code);
        }
    }
    Ok(0)
}

fn rust_fast(root: &Path) -> Result<u8, String> {
    for step in [
        cargo_bpann_tests,
        cargo_enn_no_default_tests,
        cargo_enn_no_default_examples,
    ] {
        let code = step(root)?;
        if code != 0 {
            return Ok(code);
        }
    }
    Ok(0)
}

fn rust_full(root: &Path) -> Result<u8, String> {
    for step in [
        cargo_bpann_tests,
        cargo_enn_default_tests,
        cargo_enn_default_examples,
    ] {
        let code = step(root)?;
        if code != 0 {
            return Ok(code);
        }
    }
    Ok(0)
}

fn buck_build(root: &Path) -> Result<u8, String> {
    buck2(
        root,
        [
            "--isolation-dir",
            "dev",
            "build",
            "//rust/crates/bpann:bpann",
            "//rust/crates/ennx:ennx",
            "//rust/crates/ennx-py:ennx-py",
            "//rust/crates/dev-cli:ennx",
            "--local-only",
            "--num-threads",
            "4",
            "-M",
            "none",
        ],
    )
}

fn buck_test(root: &Path) -> Result<u8, String> {
    buck2(
        root,
        [
            "--isolation-dir",
            "dev",
            "test",
            "//buck2/tests:all",
            "--local-only",
            "--num-threads",
            "4",
        ],
    )
}

fn buck_wheel(root: &Path) -> Result<u8, String> {
    buck2(
        root,
        [
            "--isolation-dir",
            "release",
            "build",
            "//:wheel",
            "--config",
            "ennx.profile=release",
            "--local-only",
            "--num-threads",
            "4",
            "--show-output",
        ],
    )
}

fn cuda_wheel(root: &Path) -> Result<u8, String> {
    buck2(
        root,
        [
            "--isolation-dir",
            "cuda",
            "build",
            "//:cuda-wheel",
            "--target-platforms",
            "//:linux-x86_64-platform",
            "--local-only",
            "--num-threads",
            "4",
            "--show-output",
        ],
    )
}

fn cuda_parity(root: &Path) -> Result<u8, String> {
    buck2(
        root,
        [
            "--isolation-dir",
            "cuda",
            "build",
            "//:cuda-parity",
            "--target-platforms",
            "//:linux-x86_64-platform",
            "--local-only",
            "--num-threads",
            "4",
            "--show-output",
        ],
    )
}

fn verify(root: &Path) -> Result<u8, String> {
    command(root, "tools/buck2-wheel-verify", std::iter::empty::<&str>())
}

fn cargo_bpann_tests(root: &Path) -> Result<u8, String> {
    command(
        root,
        "cargo",
        ["test", "--manifest-path", "Cargo.toml", "-p", "ennx-bpann"],
    )
}

fn cargo_enn_no_default_tests(root: &Path) -> Result<u8, String> {
    command(
        root,
        "cargo",
        [
            "test",
            "--manifest-path",
            "Cargo.toml",
            "-p",
            "ennx",
            "--no-default-features",
            "--lib",
            "--tests",
            "--",
            "--test-threads=1",
        ],
    )
}

fn cargo_enn_no_default_examples(root: &Path) -> Result<u8, String> {
    command(
        root,
        "cargo",
        [
            "test",
            "--manifest-path",
            "Cargo.toml",
            "-p",
            "ennx",
            "--no-default-features",
            "--examples",
            "--no-run",
        ],
    )
}

fn cargo_enn_default_tests(root: &Path) -> Result<u8, String> {
    command(
        root,
        "cargo",
        [
            "test",
            "--manifest-path",
            "Cargo.toml",
            "-p",
            "ennx",
            "--lib",
            "--tests",
            "--",
            "--test-threads=1",
        ],
    )
}

fn cargo_enn_default_examples(root: &Path) -> Result<u8, String> {
    command(
        root,
        "cargo",
        [
            "test",
            "--manifest-path",
            "Cargo.toml",
            "-p",
            "ennx",
            "--examples",
            "--no-run",
        ],
    )
}

fn python_fast(root: &Path) -> Result<u8, String> {
    command(
        root,
        "python",
        [
            "-m",
            "pytest",
            "-q",
            "tests/test_turbo_config.py",
            "tests/test_candidate_gen_direct.py",
            "tests/test_encode.py",
            "tests/test_quantization.py",
        ],
    )
}

fn check(root: &Path, args: &[String]) -> Result<u8, String> {
    let all = check_all(args)?;
    let files = if all {
        Vec::new()
    } else {
        changed_files(root)?
    };
    if !all && files.is_empty() {
        println!("No files changed in the JJ working copy.");
        return Ok(0);
    }

    let mut prek_args = vec!["run".to_string()];
    if all {
        prek_args.push("--all-files".to_string());
    } else {
        prek_args.push("--files".to_string());
        prek_args.extend(files);
    }
    run_prek(root, &prek_args)
}

fn check_all(args: &[String]) -> Result<bool, String> {
    match args {
        [] => Ok(false),
        [all] if all == "--all" => Ok(true),
        _ => Err("usage: ennx check [--all]".to_string()),
    }
}

fn changed_files(root: &Path) -> Result<Vec<String>, String> {
    let output = Command::new("jj")
        .args(["diff", "--name-only"])
        .current_dir(root)
        .output()
        .map_err(|error| format!("failed to run jj: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "jj diff --name-only failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter(|path| {
            let candidate = root.join(path);
            candidate.exists() || candidate.is_symlink()
        })
        .map(str::to_owned)
        .collect())
}

fn run_prek(root: &Path, args: &[String]) -> Result<u8, String> {
    if command(root, "tools/bootstrap-dotslash", std::iter::empty::<&str>())? != 0 {
        return Err("failed to bootstrap Dotslash for prek".to_string());
    }
    let bin_dir = root.join(".buck2-tools/bin");
    let dotslash = [bin_dir.join("dotslash"), bin_dir.join("dotslash.exe")]
        .into_iter()
        .find(|path| path.is_file())
        .ok_or("Dotslash bootstrap did not produce an executable")?;
    let status = Command::new(dotslash)
        .arg(root.join("tools/prek"))
        .args(args)
        .current_dir(root)
        .status()
        .map_err(|error| format!("failed to run prek: {error}"))?;
    Ok(status
        .code()
        .map_or(1, |code| u8::try_from(code).unwrap_or(1)))
}

fn buck2<const N: usize>(root: &Path, args: [&str; N]) -> Result<u8, String> {
    command(root, "./buck2w", args)
}

fn command<I, S>(root: &Path, program: &str, args: I) -> Result<u8, String>
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    let status = Command::new(program)
        .args(args)
        .current_dir(root)
        .status()
        .map_err(|error| format!("failed to run {program}: {error}"))?;
    Ok(status
        .code()
        .map_or(1, |code| u8::try_from(code).unwrap_or(1)))
}

fn repo_root() -> Result<PathBuf, String> {
    let dir = env::current_dir().map_err(|error| format!("cannot read cwd: {error}"))?;
    repo_root_from(dir)
}

fn repo_root_from(mut dir: PathBuf) -> Result<PathBuf, String> {
    loop {
        if dir.join(".buckconfig").is_file() {
            return Ok(dir);
        }
        if !dir.pop() {
            return Err("ennx must be run from an ENNX source checkout".to_string());
        }
    }
}

fn print_help() {
    println!("{}", help_text());
}

fn help_text() -> &'static str {
    "usage:
  ennx build
  ennx test
  ennx check [--all]
  ennx wheel
  ennx verify
  ennx ci
  ennx rust fast|full
  ennx python fast|verify|wheel
  ennx cuda wheel|parity
  ennx release upload vX.Y.Z WHEEL...
  ennx tune CONFIG.toml
  ennx buck build|test|wheel|ci
  ennx bazel build|test"
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::{check_all, help_text, repo_root_from};

    #[test]
    fn help_lists_language_gates() {
        let help = help_text();
        assert!(help.contains("ennx rust fast|full"));
        assert!(help.contains("ennx check [--all]"));
        assert!(help.contains("ennx python fast|verify|wheel"));
        assert!(help.contains("ennx cuda wheel|parity"));
        assert!(help.contains("ennx release upload vX.Y.Z WHEEL..."));
    }

    #[test]
    fn repo_root_uses_buckconfig() {
        let root = std::env::temp_dir().join(format!("ennx-dev-cli-{}", std::process::id()));
        let nested = root.join("nested");
        fs::create_dir_all(&nested).unwrap();
        fs::write(root.join(".buckconfig"), "").unwrap();

        assert_eq!(repo_root_from(nested).unwrap(), root);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn check_accepts_only_the_all_flag() {
        assert!(!check_all(&[]).unwrap());
        assert!(check_all(&["--all".to_string()]).unwrap());
        assert!(check_all(&["--all-files".to_string()]).is_err());
    }
}
