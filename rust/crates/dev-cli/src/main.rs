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
        "build" => buck_build(&root),
        "test" => buck_test(&root),
        "wheel" => buck_wheel(&root),
        "verify" => verify(&root),
        "ci" => ci(&root),
        "buck" => buck(&root, &args[1..]),
        "bazel" => bazel(&root, &args[1..]),
        "tune" => tune::run(&root, &args[1..]),
        "version" | "--version" | "-V" => {
            println!("ennx dev-cli 0.0.0");
            Ok(0)
        }
        command => Err(format!(
            "unknown ennx command {command:?}\n\n{}",
            help_text()
        )),
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

fn ci(root: &Path) -> Result<u8, String> {
    for step in [buck_build, buck_test, buck_wheel, verify] {
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

fn verify(root: &Path) -> Result<u8, String> {
    command(root, "tools/buck2-wheel-verify", std::iter::empty::<&str>())
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
    let mut dir = env::current_dir().map_err(|error| format!("cannot read cwd: {error}"))?;
    loop {
        if dir.join(".buckroot").is_file() {
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
  ennx wheel
  ennx verify
  ennx ci
  ennx tune CONFIG.toml
  ennx buck build|test|wheel|ci
  ennx bazel build|test"
}
