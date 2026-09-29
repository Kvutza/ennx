use std::io::{self, IsTerminal, Read, Write};
use std::process::Command;

pub struct CommandEntry {
    pub name: &'static str,
    pub group: &'static str,
    pub summary: &'static str,
    pub details: &'static str,
    pub options: &'static [&'static str],
    pub example: &'static str,
}

pub const COMMANDS: &[CommandEntry] = &[
    CommandEntry {
        name: "target",
        group: "Accelerators & Hardware Targets",
        summary: "Hardware backends, synthesized kernels, and device execution gates",
        details: "Unified hardware accelerator command consolidating NVIDIA GPU tracks (cuda, ptx),\n\
                  Apple Neural Engine (ane), and mobile device testing (ios / iphone).",
        options: &[
            "cuda          Run pinned CUDA-Oxide track (parity, prefill, generate-t4, modal)",
            "ptx           Emit synthesized PTX or execute hardware gate on Modal",
            "ane           Measure Apple Neural Engine execution through Core ML",
            "ios           Build, deploy, and benchmark the iPhone Metal worker",
        ],
        example: "./ennx target cuda generate-t4 --context 4096 --prompt 128 --visits 2",
    },
    CommandEntry {
        name: "opt",
        group: "Bayesian Optimization & Protocols",
        summary: "Bayesian optimization, surrogate tuning, evaluation suites, and protocols",
        details: "Unified optimization domain combining single-run tuning, black-box benchmarking,\n\
                  paired protocol experiment repetition plans, and categorical algorithm catalogs.",
        options: &[
            "run           Execute a configured Bayesian optimization experiment",
            "protocol      Plan, run, and inspect a registered experiment protocol",
            "eval          Evaluate optimizers on paired black-box benchmarking tasks",
            "catalog       Inspect algorithm families or enumerate tune configurations",
        ],
        example: "./ennx opt run results/metal/bo-4k-guided-20261005/experiment.toml",
    },
    CommandEntry {
        name: "data",
        group: "Data & Model Artifacts",
        summary: "Corpora generation, token datasets, and tensor container inspection",
        details: "Unified data and artifact command managing deterministic SFST corpus synthesis\n\
                  and zero-copy Safetensors header validation and layout inspection.",
        options: &[
            "corpus        Sample a normalized stochastic FST into an immutable text corpus",
            "inspect       Validate and describe Safetensors container without loading payload",
        ],
        example: "./ennx data inspect model.safetensors",
    },
    CommandEntry {
        name: "model",
        group: "Optimization & Model Graph",
        summary: "Inspect semantic model graphs, PISA attention caches, and compute scaling",
        details: "Provides tooling to analyze layer visits, recurrent state transitions, compiler lowering\n\
                  coverage, multi-resolution PISA context caches up to 1M tokens, and arithmetic work\n\
                  requirements across flat broad-pass and 2-stage hierarchical context architectures.",
        options: &[
            "scale         Calculate model work and throughput bounds (supports --hierarchical)",
            "context       Measure cached PISA attention over up to 1,048,576 context tokens",
            "inspect       Show operations assigned to a backend and operations still missing",
            "context-loop  Measure complete BO rounds with newly generated token workloads",
            "decode        Decode generated vocabulary IDs with corpus byte-level BPE tokenizer",
            "tirx-probe    Probe for optional Apache TVM / TIRX compiler installation",
        ],
        example: "./ennx model scale --generated 1024 --prompt 1047552 --round-ms 52042 --hierarchical",
    },
    CommandEntry {
        name: "test",
        group: "Toolchain & Developer Environment",
        summary: "Run Rust unit tests, kernel tests, and parity suites",
        details: "Executes all workspace unit tests and integration tests via Buck2,\n\
                  guaranteeing zero test failures and numerical parity across devices.",
        options: &["--python      Also run installed-wheel Python integration tests"],
        example: "./ennx test",
    },
    CommandEntry {
        name: "build",
        group: "Toolchain & Developer Environment",
        summary: "Build the library, CLI, tests, and Python wheels",
        details: "Compiles Rust crates with Buck2, links native accelerator binaries,\n\
                  and builds Python extension wheels with abi3 compatibility.",
        options: &[
            "--tests       Run full Python verification suite instead of smoke tests",
            "--out         Target directory where built wheel artifacts should be written",
        ],
        example: "./ennx build --out dist",
    },
    CommandEntry {
        name: "fmt",
        group: "Toolchain & Developer Environment",
        summary: "Format the repository or check formatting without edits",
        details: "Enforces formatting rules across Rust source files and build configurations.",
        options: &[
            "--check       Check formatting compliance without writing modifications to files",
        ],
        example: "./ennx fmt --check",
    },
    CommandEntry {
        name: "dev",
        group: "Toolchain & Developer Environment",
        summary: "Run the complete format, build, and test verification cycle",
        details: "Convenience target executing format checks, artifact compilation,\n\
                  and comprehensive unit test suites in a single pass.",
        options: &[],
        example: "./ennx dev",
    },
    CommandEntry {
        name: "util",
        group: "Toolchain & Developer Environment",
        summary: "Infrequently used utility commands such as shell completions",
        details: "Provides ancillary developer utilities matching the Jujutsu (`jj util`) design.\n\
                  Currently hosts shell completion script generation for Zsh, Bash, and Fish.",
        options: &["completion    Print a command-line-completion script (zsh, bash, fish)"],
        example: "./ennx util completion zsh",
    },
    CommandEntry {
        name: "menu",
        group: "Toolchain & Developer Environment",
        summary: "Open interactive command navigator (navigate with Up/Down arrow keys)",
        details: "An interactive terminal user interface enabling immediate exploration\n\
                  of all ENNX domains, subcommands, options, and live execution examples.",
        options: &[],
        example: "./ennx menu",
    },
];

struct RawTerminalGuard {
    saved_state: Option<String>,
}

impl RawTerminalGuard {
    fn new() -> Self {
        let output = Command::new("sh")
            .args(["-c", "stty -g < /dev/tty 2>/dev/null"])
            .output();
        let saved_state = match output {
            Ok(out) if out.status.success() => {
                let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
                if !s.is_empty() {
                    // Use cbreak -echo: allows single-character input while preserving
                    // opost/onlcr output translation so newlines don't staircase
                    let _ = Command::new("sh")
                        .args(["-c", "stty cbreak -echo < /dev/tty 2>/dev/null"])
                        .status();
                    // Enter alternate screen buffer and hide cursor
                    let mut stdout = io::stdout().lock();
                    let _ = write!(stdout, "\x1b[?1049h\x1b[?25l");
                    let _ = stdout.flush();
                    Some(s)
                } else {
                    None
                }
            }
            _ => None,
        };
        Self { saved_state }
    }
}

impl Drop for RawTerminalGuard {
    fn drop(&mut self) {
        // Show cursor and leave alternate screen buffer
        let mut stdout = io::stdout().lock();
        let _ = write!(stdout, "\x1b[?25h\x1b[?1049l");
        let _ = stdout.flush();

        if let Some(ref saved) = self.saved_state {
            let cmd = format!("stty {saved} < /dev/tty 2>/dev/null");
            let _ = Command::new("sh").args(["-c", &cmd]).status();
        } else {
            let _ = Command::new("sh")
                .args(["-c", "stty sane < /dev/tty 2>/dev/null"])
                .status();
        }
    }
}

pub fn run() -> Result<(), String> {
    if !is_interactive() {
        print_guide();
        return Ok(());
    }

    let mut tty_file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open("/dev/tty")
        .map_err(|e| format!("cannot open /dev/tty: {e}"))?;

    let mut selected: usize = 0;
    let _guard = RawTerminalGuard::new();

    loop {
        render(selected)?;
        match read_key(&mut tty_file)? {
            Key::Up => {
                if selected == 0 {
                    selected = COMMANDS.len() - 1;
                } else {
                    selected -= 1;
                }
            }
            Key::Down => {
                selected = (selected + 1) % COMMANDS.len();
            }
            Key::Enter => {
                drop(_guard);
                let cmd = COMMANDS[selected].name;
                println!("\nExecuting: ./ennx {cmd} --help\n");
                let _ = Command::new("./ennx").args([cmd, "--help"]).status();
                return Ok(());
            }
            Key::Quit => {
                drop(_guard);
                return Ok(());
            }
            Key::Other => {}
        }
    }
}

pub fn is_interactive() -> bool {
    io::stdin().is_terminal() || std::path::Path::new("/dev/tty").exists()
}

enum Key {
    Up,
    Down,
    Enter,
    Quit,
    Other,
}

fn read_key(tty: &mut std::fs::File) -> Result<Key, String> {
    let mut buf = [0u8; 1];
    tty.read_exact(&mut buf).map_err(|e| e.to_string())?;

    if buf[0] == b'q' || buf[0] == 3 {
        return Ok(Key::Quit);
    }
    if buf[0] == b'\r' || buf[0] == b'\n' {
        return Ok(Key::Enter);
    }
    if buf[0] == b'k' {
        return Ok(Key::Up);
    }
    if buf[0] == b'j' {
        return Ok(Key::Down);
    }
    if buf[0] == 27 {
        // Escape sequence (e.g., [A, [B, OA, OB)
        let mut seq = [0u8; 2];
        if tty.read_exact(&mut seq).is_ok() {
            if seq[0] == b'[' || seq[0] == b'O' {
                if seq[1] == b'A' {
                    return Ok(Key::Up);
                } else if seq[1] == b'B' {
                    return Ok(Key::Down);
                }
            }
        }
    }
    Ok(Key::Other)
}

fn render(selected: usize) -> Result<(), String> {
    let mut stdout = io::stdout().lock();
    // Move cursor to top-left and clear display
    write!(stdout, "\x1b[H\x1b[2J").map_err(|e| e.to_string())?;

    write!(
        stdout,
        "\x1b[1mENNX Command Navigator\x1b[0m \x1b[2m(Use \x1b[0m\x1b[36m↑/↓\x1b[0m\x1b[2m or \x1b[0m\x1b[36mj/k\x1b[0m\x1b[2m to navigate, \x1b[0m\x1b[32mEnter\x1b[0m\x1b[2m to view help, \x1b[0m\x1b[31mq\x1b[0m\x1b[2m to quit)\x1b[0m\r\n\r\n"
    )
    .map_err(|e| e.to_string())?;

    write!(stdout, "\x1b[1mCommands:\x1b[0m\r\n").map_err(|e| e.to_string())?;
    for (idx, cmd) in COMMANDS.iter().enumerate() {
        if idx == selected {
            write!(
                stdout,
                "  \x1b[1;36m>\x1b[0m \x1b[1;36m{:<10}\x1b[0m  {}\r\n",
                cmd.name, cmd.summary
            )
            .map_err(|e| e.to_string())?;
        } else {
            write!(
                stdout,
                "    \x1b[1m{:<10}\x1b[0m  \x1b[2m{}\x1b[0m\r\n",
                cmd.name, cmd.summary
            )
            .map_err(|e| e.to_string())?;
        }
    }

    let active = &COMMANDS[selected];
    write!(stdout, "\r\n\x1b[2m─────────────────────────────────────────────────────────────────────────────\x1b[0m\r\n").map_err(|e| e.to_string())?;
    write!(
        stdout,
        "\x1b[1;36m./ennx {}\x1b[0m \x1b[2m- {}\x1b[0m\r\n",
        active.name, active.summary
    )
    .map_err(|e| e.to_string())?;
    for line in active.details.lines() {
        write!(stdout, "{line}\r\n").map_err(|e| e.to_string())?;
    }

    if !active.options.is_empty() {
        write!(stdout, "\r\n\x1b[1mSubcommands & Options:\x1b[0m\r\n")
            .map_err(|e| e.to_string())?;
        for opt in active.options {
            write!(stdout, "  {opt}\r\n").map_err(|e| e.to_string())?;
        }
    }

    write!(stdout, "\r\n\x1b[1mUsage Example:\x1b[0m\r\n").map_err(|e| e.to_string())?;
    write!(stdout, "  \x1b[32m{}\x1b[0m\r\n", active.example).map_err(|e| e.to_string())?;
    write!(stdout, "\x1b[2m─────────────────────────────────────────────────────────────────────────────\x1b[0m\r\n").map_err(|e| e.to_string())?;

    stdout.flush().map_err(|e| e.to_string())?;
    Ok(())
}

fn print_guide() {
    println!("ENNX Command Catalog:\n");
    for cmd in COMMANDS {
        println!("  {:<12} {}", cmd.name, cmd.summary);
    }
    println!("\nRun './ennx <command> --help' for full parameter specifications.");
}
