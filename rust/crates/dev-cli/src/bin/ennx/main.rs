use std::env;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

use clap::{Parser, Subcommand};

mod ane;
mod build_system;
mod catalog;
mod completion;
mod corpus;
mod cuda_track;
mod eval;
mod experiment;
mod iphone;
mod kernel_search;
mod menu;
mod model_program;
mod protocol;
mod protocol_analysis;
mod ptx;
mod radar;
#[path = "eval/reports.rs"]
mod reports;
mod tune;

#[derive(Debug, Parser)]
#[command(
    name = "ennx",
    bin_name = "./ennx",
    version = env!("CARGO_PKG_VERSION"),
    about = "Build, verify, tune, and evaluate hardware-resident ENNX models",
    long_about = "ENNX: Systems, kernels, and Bayesian optimization platform for extreme long-context language modeling.\n\n\
                 Supports resident recurrent architectures (PISA, mHC, MoE) across Apple Metal (unified memory) \
                 and NVIDIA CUDA (Turing/Ampere via CUDA-Oxide and Modal).",
    arg_required_else_help = true
)]
struct Cli {
    #[command(subcommand)]
    action: Action,
}

#[derive(Debug, PartialEq, Eq, Subcommand)]
enum Action {
    /// Build the library, CLI, tests, and Python wheels.
    ///
    /// Compiles Rust crates with Buck2, links native accelerator binaries,
    /// and builds Python extension wheels with abi3 compatibility.
    Build {
        /// Run full Python verification instead of smoke tests.
        #[arg(long, help_heading = "Build Options")]
        tests: bool,
        /// Write wheels into this directory.
        #[arg(long, default_value = "dist", help_heading = "Output Options")]
        out: PathBuf,
    },
    /// Run Rust unit tests, kernel tests, and parity suites.
    ///
    /// Executes all workspace unit tests and integration tests via Buck2,
    /// guaranteeing zero test failures and numerical parity across devices.
    Test {
        /// Run installed-wheel Python tests; requires ENNX_WHEEL_PATH.
        #[arg(long, help_heading = "Test Options")]
        python: bool,
        /// Only run test targets reaching working copy changes.
        #[arg(long, help_heading = "Test Options")]
        affected: bool,
    },
    /// Format the repository or check formatting without edits.
    ///
    /// Enforces formatting rules across Rust source files and build configurations.
    Fmt {
        /// Check formatting compliance without writing modifications to files.
        #[arg(long, help_heading = "Format Options")]
        check: bool,
    },
    /// Inspect the semantic model graph and a backend lowering plan.
    ///
    /// Provides tooling to analyze layer visits, recurrent state transitions,
    /// compiler lowering coverage, multi-resolution PISA context caches up to 1M tokens,
    /// and arithmetic work requirements across flat and hierarchical architectures.
    Model {
        #[command(subcommand)]
        command: model_program::Action,
    },
    /// Accelerator backends, synthesized kernels, and device execution gates.
    ///
    /// Consolidates hardware execution across NVIDIA GPUs (CUDA-Oxide / PTX),
    /// Apple Neural Engine (ANE / Core ML), and mobile targets (iPhone Metal / WebGPU).
    Target {
        #[command(subcommand)]
        command: TargetAction,
    },
    /// Bayesian optimization, surrogate tuning, evaluation suites, and experiment protocols.
    ///
    /// Provides tools to execute optimization loops, benchmark candidate acquisition policies,
    /// and evaluate matched protocol repetitions.
    Opt {
        #[command(subcommand)]
        command: OptAction,
    },
    /// Corpora generation, token datasets, and tensor container inspection.
    ///
    /// Manages data artifacts including OpenGrm stochastic grammar sampling
    /// and zero-copy Safetensors inspection.
    Data {
        #[command(subcommand)]
        command: DataAction,
    },
    /// Run the complete format, build, and test verification cycle.
    ///
    /// Convenience target executing format checks, artifact compilation,
    /// and comprehensive unit test suites in a single pass.
    Dev,
    /// Codebase radar, semantic call graphs, blast radius, and quality gates.
    ///
    /// Surfaces AST-scoped symbol search, transitive impact sets,
    /// quality deltas against HEAD, and token-compressed agent contexts.
    Radar {
        #[command(subcommand)]
        command: radar::RadarAction,
    },
    /// Open the interactive command navigator.
    ///
    /// Launches an interactive terminal user interface enabling immediate exploration
    /// of all ENNX subcommands, options, and workflows using Up/Down arrow keys.
    Menu,
    /// Infrequently used utility commands such as shell completion scripts.
    ///
    /// Matches the design pattern of Jujutsu (`jj util`).
    Util {
        #[command(subcommand)]
        command: UtilAction,
    },
    /// Check or synchronize the managed Rust toolchain.
    #[command(hide = true)]
    Toolchain {
        #[command(subcommand)]
        command: ToolchainAction,
    },
    /// Measure Apple Neural Engine execution through Core ML [legacy alias: target ane].
    #[command(hide = true)]
    Ane {
        #[command(subcommand)]
        command: ane::Action,
    },
    /// Evaluate optimizers on paired black-box tasks [legacy alias: opt eval].
    #[command(hide = true)]
    Eval {
        /// Path to the evaluation suite configuration TOML file.
        config: PathBuf,
        /// Override the artifact path declared in the configuration.
        #[arg(long, help_heading = "Output Options")]
        output: Option<PathBuf>,
    },
    /// Run a configured Bayesian optimization experiment [legacy alias: opt run].
    #[command(hide = true)]
    Tune {
        /// Path to the experiment configuration TOML file.
        config: PathBuf,
        /// Prepare supported experiment inputs without starting optimization.
        #[arg(long, help_heading = "Execution Options")]
        prepare: bool,
    },
    /// Inspect algorithm families or enumerate tune configurations [legacy alias: opt catalog].
    #[command(hide = true)]
    Catalog {
        /// Version 2 baseline configuration; omit to inspect the global library inventory.
        config: Option<PathBuf>,
        /// Export enumerated configurations to this target directory.
        #[arg(long, help_heading = "Output Options")]
        out: Option<PathBuf>,
        /// Emit the catalog inventory or enumeration manifest as JSON.
        #[arg(long, help_heading = "Reporting Options")]
        json: bool,
    },
    /// Plan, run, and inspect a registered experiment protocol [legacy alias: opt protocol].
    #[command(hide = true)]
    Experiment {
        /// Registered protocol name; omit to list available protocols.
        protocol: Option<String>,
        /// Version 2 baseline configuration used to construct matched arms.
        baseline: Option<PathBuf>,
        /// Directory where experiment plans and run artifacts will be stored.
        #[arg(long, help_heading = "Output Options")]
        out: Option<PathBuf>,
        /// Complete optimization rounds per paired repetition.
        #[arg(long, default_value_t = 512, help_heading = "Workload Options")]
        rounds: u32,
        /// Paired repetitions executed per experimental arm.
        #[arg(long, default_value_t = 3, help_heading = "Workload Options")]
        reps: u32,
        /// Execute every arm after writing the protocol execution plan.
        #[arg(long, help_heading = "Execution Options")]
        run: bool,
        /// Emit status and plan manifests as machine-readable JSON.
        #[arg(long, help_heading = "Reporting Options")]
        json: bool,
    },
    /// Inspect framework-neutral tensor containers [legacy alias: data inspect].
    #[command(hide = true)]
    Tensor {
        #[command(subcommand)]
        command: TensorAction,
    },
    /// Generate reproducible corpora from stochastic grammars [legacy alias: data corpus].
    #[command(hide = true)]
    Corpus {
        #[command(subcommand)]
        command: corpus::Action,
    },
    /// Inspect or run the pinned CUDA-Oxide track [legacy alias: target cuda].
    #[command(hide = true)]
    Cuda {
        #[command(subcommand)]
        command: cuda_track::Action,
    },
    /// Emit synthesized PTX or execute hardware gate [legacy alias: target ptx].
    #[command(hide = true)]
    Ptx {
        #[command(subcommand)]
        command: ptx::Action,
    },
    /// Build, deploy, and measure the iPhone Metal worker [legacy alias: target ios].
    #[command(hide = true)]
    Iphone {
        #[command(subcommand)]
        command: iphone::Action,
    },
    /// Print a shell completion script for zsh, bash, or fish [legacy alias: util completion].
    #[command(hide = true)]
    Completion {
        /// Target shell for completion script generation.
        #[arg(value_enum)]
        shell: completion::Shell,
    },
}

#[derive(Debug, PartialEq, Eq, Subcommand)]
enum TargetAction {
    /// Inspect or run the pinned CUDA-Oxide track without invoking Cargo directly.
    Cuda {
        #[command(subcommand)]
        command: cuda_track::Action,
    },
    /// Emit synthesized PTX or execute its hardware gate on Modal.
    Ptx {
        #[command(subcommand)]
        command: ptx::Action,
    },
    /// Measure Apple Neural Engine execution through Core ML.
    Ane {
        #[command(subcommand)]
        command: ane::Action,
    },
    /// Build, deploy, and measure the iPhone Metal worker.
    Ios {
        #[command(subcommand)]
        command: iphone::Action,
    },
}

#[derive(Debug, PartialEq, Eq, Subcommand)]
enum OptAction {
    /// Run a configured Bayesian optimization experiment.
    Run {
        /// Path to the experiment configuration TOML file.
        config: PathBuf,
        /// Prepare supported experiment inputs without starting optimization.
        #[arg(long, help_heading = "Execution Options")]
        prepare: bool,
    },
    /// Plan, run, and inspect a registered experiment protocol.
    Protocol {
        /// Registered protocol name; omit to list available protocols.
        protocol: Option<String>,
        /// Version 2 baseline configuration used to construct matched arms.
        baseline: Option<PathBuf>,
        /// Directory where experiment plans and run artifacts will be stored.
        #[arg(long, help_heading = "Output Options")]
        out: Option<PathBuf>,
        /// Complete optimization rounds per paired repetition.
        #[arg(long, default_value_t = 512, help_heading = "Workload Options")]
        rounds: u32,
        /// Paired repetitions executed per experimental arm.
        #[arg(long, default_value_t = 3, help_heading = "Workload Options")]
        reps: u32,
        /// Execute every arm after writing the protocol execution plan.
        #[arg(long, help_heading = "Execution Options")]
        run: bool,
        /// Emit status and plan manifests as machine-readable JSON.
        #[arg(long, help_heading = "Reporting Options")]
        json: bool,
    },
    /// Evaluate optimizers on paired black-box tasks.
    Eval {
        /// Path to the evaluation suite configuration TOML file.
        config: PathBuf,
        /// Override the artifact path declared in the configuration.
        #[arg(long, help_heading = "Output Options")]
        output: Option<PathBuf>,
    },
    /// Inspect algorithm families or enumerate categorical tune configurations.
    Catalog {
        /// Version 2 baseline configuration; omit to inspect the global library inventory.
        config: Option<PathBuf>,
        /// Export enumerated configurations to this target directory.
        #[arg(long, help_heading = "Output Options")]
        out: Option<PathBuf>,
        /// Emit the catalog inventory or enumeration manifest as JSON.
        #[arg(long, help_heading = "Reporting Options")]
        json: bool,
    },
}

#[derive(Debug, PartialEq, Eq, Subcommand)]
enum DataAction {
    /// Generate reproducible corpora from stochastic grammars.
    Corpus {
        #[command(subcommand)]
        command: corpus::Action,
    },
    /// Inspect framework-neutral tensor containers.
    Inspect {
        /// Path to the Safetensors container file to validate.
        file: PathBuf,
        /// Emit tensor metadata, byte offsets, and shapes as structured JSON.
        #[arg(long, help_heading = "Reporting Options")]
        json: bool,
    },
}

#[derive(Debug, PartialEq, Eq, Subcommand)]
enum UtilAction {
    /// Print a command-line completion script (compatible with Jujutsu `jj util completion`).
    Completion {
        /// Target shell for completion script generation.
        #[arg(value_enum)]
        shell: completion::Shell,
    },
}

#[derive(Debug, PartialEq, Eq, Subcommand)]
enum ToolchainAction {
    /// Check every managed toolchain declaration across Buck2, Cargo, and CI.
    Check,
    /// Synchronize declarations across build systems, optionally upgrading Rust nightly.
    Sync {
        /// Optional nightly toolchain date string (e.g. "2026-09-28").
        nightly: Option<String>,
    },
}

#[derive(Debug, PartialEq, Eq, Subcommand)]
enum TensorAction {
    /// Validate and describe a Safetensors container without loading its payload into host RAM.
    Inspect {
        /// Path to the Safetensors container file to validate.
        file: PathBuf,
        /// Emit tensor metadata, byte offsets, and shapes as structured JSON.
        #[arg(long, help_heading = "Reporting Options")]
        json: bool,
    },
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("ennx: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let cli = Cli::parse();
    let root = env::var_os("BUILD_WORKSPACE_DIRECTORY")
        .map(PathBuf::from)
        .map_or_else(|| env::current_dir().map_err(|error| error.to_string()), Ok)?;
    env::set_current_dir(&root).map_err(|error| error.to_string())?;
    dispatch(&root, cli.action)
}

fn dispatch(root: &Path, action: Action) -> Result<(), String> {
    match action {
        Action::Build { tests, out } => build(root, tests, out),
        Action::Test { python, affected } => test(root, python, affected),
        Action::Fmt { check } => format(root, check),
        Action::Radar { command } => radar::run(root, command),
        Action::Toolchain { command } => toolchain(root, command),
        Action::Dev => execute(developer_command(root, "dev")),
        action => workflow(root, action),
    }
}

fn workflow(root: &Path, action: Action) -> Result<(), String> {
    match action {
        Action::Target { command } => dispatch_target(root, command),
        Action::Opt { command } => dispatch_opt(root, command),
        Action::Data { command } => dispatch_data(root, command),
        Action::Model { command } => model_program::run(command),
        Action::Menu => menu::run(),
        Action::Util { command } => match command {
            UtilAction::Completion { shell } => completion::run(shell),
        },
        action => dispatch_legacy(root, action),
    }
}

fn dispatch_target(root: &Path, command: TargetAction) -> Result<(), String> {
    match command {
        TargetAction::Cuda { command } => cuda_track::run(root, command),
        TargetAction::Ptx { command } => ptx::run(root, command),
        TargetAction::Ane { command } => ane::run(root, command),
        TargetAction::Ios { command } => iphone::run(root, command),
    }
}

fn dispatch_opt(root: &Path, command: OptAction) -> Result<(), String> {
    match command {
        OptAction::Run { config, prepare } => experiment::tune(root, &config, prepare),
        OptAction::Protocol {
            protocol,
            baseline,
            out,
            rounds,
            reps,
            run,
            json,
        } => protocol::run(
            root,
            protocol::Request {
                protocol: protocol.as_deref(),
                baseline: baseline.as_deref(),
                out: out.as_deref(),
                rounds,
                reps,
                execute: run,
                json,
            },
        ),
        OptAction::Eval { config, output } => eval::run(root, &config, output.as_deref()),
        OptAction::Catalog { config, out, json } => {
            catalog::run(root, config.as_deref(), out.as_deref(), json)
        }
    }
}

fn dispatch_data(root: &Path, command: DataAction) -> Result<(), String> {
    match command {
        DataAction::Corpus { command } => corpus::run(root, command),
        DataAction::Inspect { file, json } => {
            experiment::tensor(TensorAction::Inspect { file, json })
        }
    }
}

fn dispatch_legacy(root: &Path, action: Action) -> Result<(), String> {
    match action {
        Action::Ane { command } => ane::run(root, command),
        Action::Eval { config, output } => eval::run(root, &config, output.as_deref()),
        Action::Tune { config, prepare } => experiment::tune(root, &config, prepare),
        Action::Catalog { config, out, json } => {
            catalog::run(root, config.as_deref(), out.as_deref(), json)
        }
        Action::Experiment {
            protocol,
            baseline,
            out,
            rounds,
            reps,
            run,
            json,
        } => protocol::run(
            root,
            protocol::Request {
                protocol: protocol.as_deref(),
                baseline: baseline.as_deref(),
                out: out.as_deref(),
                rounds,
                reps,
                execute: run,
                json,
            },
        ),
        Action::Tensor { command } => experiment::tensor(command),
        Action::Corpus { command } => corpus::run(root, command),
        Action::Cuda { command } => cuda_track::run(root, command),
        Action::Ptx { command } => ptx::run(root, command),
        Action::Iphone { command } => iphone::run(root, command),
        Action::Completion { shell } => completion::run(shell),
        _ => unreachable!("unhandled action in dispatch_legacy"),
    }
}

pub(crate) fn dev_exec(root: &Path, action: &str, args: &[&str]) -> Result<(), String> {
    let mut command = developer_command(root, action);
    command.args(args);
    execute(command)
}

fn build(root: &Path, tests: bool, out: PathBuf) -> Result<(), String> {
    let mode = if tests { "full" } else { "smoke" };
    let out_str = out.to_string_lossy();
    dev_exec(root, "build", &[mode, &out_str])
}

fn test(root: &Path, python: bool, affected: bool) -> Result<(), String> {
    let target = if python {
        "python"
    } else if affected {
        "affected"
    } else {
        "rust"
    };
    dev_exec(root, "test", &[target])
}

fn format(root: &Path, check: bool) -> Result<(), String> {
    let mode = if check { "check" } else { "write" };
    dev_exec(root, "fmt", &[mode])
}

fn toolchain(root: &Path, action: ToolchainAction) -> Result<(), String> {
    let mut command = developer_command(root, "toolchain");
    match action {
        ToolchainAction::Check => command.arg("check"),
        ToolchainAction::Sync { nightly: None } => command.arg("sync"),
        ToolchainAction::Sync {
            nightly: Some(nightly),
        } => command.args(["sync", &nightly]),
    };
    execute(command)
}

fn developer_command(root: &Path, action: &str) -> Command {
    let mut command = Command::new("sh");
    command
        .current_dir(root)
        .args(["tools/dev-actions", action]);
    command
}

pub(crate) fn execute(mut command: Command) -> Result<(), String> {
    let status = command.status().map_err(|error| error.to_string())?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("command exited with {status}"))
    }
}
