use clap::Subcommand;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, PartialEq, Eq, Subcommand)]
pub(crate) enum Action {
    /// Inspect CUDA-Oxide kernel lowering coverage and symbol tables without GPU hardware.
    ///
    /// Evaluates compiled PTX entry points, target compute capability constraints (SM75+),
    /// and driver symbol bindings without initializing the CUDA runtime or allocating device buffers.
    Inspect,
    /// Compile CUDA-Oxide device modules and synthesized PTX kernels for the pinned T4 target.
    ///
    /// Translates type-driven kernel definitions into PTX and invokes nvcc/nvptx64 for SM75
    /// (NVIDIA Tesla T4), targeting hardware FP16 tensor-core MMA intrinsics with zero bank conflicts.
    Build,
    /// Run mathematical and bitwise parity checks against the host CPU reference implementation.
    ///
    /// Validates numerical equivalence across Rademacher perturbations, dense matrix multiplications,
    /// Softmax attention reductions, and mHC residual updates within strict floating-point epsilon bounds.
    Parity,
    /// Run end-to-end resident search parity checks on the CUDA device.
    ///
    /// Executes candidate proposal, speculative verification, and Thompson sampling scoring
    /// entirely within GPU VRAM without host synchronization stalls, confirming trajectory equivalence.
    Resident,
    /// Measure device latency and memory bandwidth consumption during the resident prefill slice.
    ///
    /// Isolates causal attention prefill wall time across variable prompt lengths, evaluating
    /// QKV projection throughput and logarithmic PISA attention pyramid construction latency.
    Prefill,
    /// Verify numerical stability and contractive convergence of the 5-layer mHC model.
    ///
    /// Evaluates one, two, and four recurrent layer visits over resident state representations,
    /// verifying that multi-head manifold contractions prevent gradient vanishing or explosion.
    ModelCheck,
    /// Benchmark bounded gather queries over a persistent million-token PISA cache.
    ///
    /// Allocates and queries a 1,048,576-token logarithmic attention index in device memory,
    /// measuring query gather latency and verifying sub-millisecond retrieval guarantees.
    Context {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Execute looped mHC model checkpoints on input token sequences and persist outputs.
    ///
    /// Runs forward inference over tokenized prompt sequences using pinned 5-layer recurrent weights,
    /// persisting logits, hidden state checkpoints, and generation metadata to disk.
    Model {
        #[arg(required = true, trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Generate tokens using device-side speculative accepted-prefix verification and repair.
    ///
    /// Executes parallel speculative drafting followed by immediate device-side prefix verification,
    /// performing single-cycle suffix repair upon verification divergence without host roundtrips.
    Generate {
        #[arg(required = true, trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Measure real accepted-prefix generation on a Modal T4.
    ///
    /// Launches a cloud sandbox on Modal, executes causal prefill and speculative repair
    /// passes on the target GPU, and records exact token progress and wall/device timings.
    GenerateT4 {
        /// Total context sequence capacity in tokens (powers of two up to 1,048,576).
        #[arg(long, default_value_t = 4_096, help_heading = "Sequence Dimensions")]
        context: usize,
        /// Prompt prefix length in tokens.
        #[arg(long, default_value_t = 128, help_heading = "Sequence Dimensions")]
        prompt: usize,
        /// Number of recurrent layer visits executed per wave.
        #[arg(long, default_value_t = 2, help_heading = "Recurrence & Execution")]
        visits: usize,
        /// Softmax sampling temperature for token generation.
        #[arg(long, default_value = "0.8", help_heading = "Sampling & Randomness")]
        temperature: String,
        /// Pseudo-random number generator seed for reproducible sampling.
        #[arg(long, default_value_t = 17, help_heading = "Sampling & Randomness")]
        seed: u64,
        /// Verifier wave unroll factor.
        #[arg(long, default_value_t = 2, help_heading = "Recurrence & Execution")]
        unroll: usize,
        /// Stream bounded activation chunks while retaining causal KV; zero uses packed execution.
        #[arg(long, default_value_t = 0, help_heading = "Memory & Streaming")]
        chunk: usize,
        /// Output path where generation timing, token IDs, and repair traces are serialized.
        #[arg(
            long,
            default_value = "results/cuda/generation-t4",
            help_heading = "Output Options"
        )]
        out: PathBuf,
    },
    /// Generate tokens using non-autoregressive block diffusion checkpoint sampling.
    ///
    /// Performs iterative mask denoising across contiguous token blocks without sequential
    /// autoregressive dependencies, generating candidate proposals in parallel steps.
    Diffusion {
        #[arg(required = true, trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Verify learned mask inputs, corruption schedules, and denoising sampling on the CUDA fixture.
    ///
    /// Runs synthetic test fixtures through the block diffusion pipeline to guarantee correct
    /// coordinate masking, noise variance scheduling, and cross-entropy loss computation.
    DiffusionCheck,
    /// Check native CUDA actions under compute-sanitizer for memory leaks and race conditions.
    ///
    /// Invokes NVIDIA compute-sanitizer (memcheck, racecheck, initcheck, synccheck) over GPU
    /// kernels to verify absence of out-of-bounds accesses, shared memory bank conflicts, and uninitialized reads.
    Sanitize {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Measure packed Rademacher perturbation bandwidth and memory bus throughput.
    ///
    /// Profiles memory bus efficiency for bit-packed pseudo-random coordinate generation
    /// and in-place weight perturbation, reporting effective GB/s against theoretical device peak.
    Bench,
    /// Measure the real resident BF16 BO curve on a Modal T4.
    ///
    /// Evaluates sequential candidate acquisition, device-side perturbation, and surrogate updates
    /// directly in GPU VRAM across multiple optimization iterations.
    Modal {
        /// Number of candidate points acquired per optimization batch.
        #[arg(long, default_value_t = 4, help_heading = "Optimization Workload")]
        candidates: usize,
        /// Historical observation capacity retained in the surrogate Gaussian process.
        #[arg(long, default_value_t = 32, help_heading = "Optimization Workload")]
        history: usize,
        /// Number of sequential Bayesian optimization iterations to execute.
        #[arg(long, default_value_t = 50, help_heading = "Optimization Workload")]
        iterations: usize,
        /// File path where convergence curve and regret records should be saved.
        #[arg(
            long,
            default_value = "results/cuda/bf16-bo-curve-t4.json",
            help_heading = "Output Options"
        )]
        out: PathBuf,
    },
}

impl Action {
    fn command(&self) -> (&'static str, &[String]) {
        match self {
            Action::Context { args } => ("context", args.as_slice()),
            Action::Generate { args } => ("generate", args.as_slice()),
            Action::Diffusion { args } => ("diffusion", args.as_slice()),
            Action::Model { args } => ("model", args.as_slice()),
            Action::Sanitize { args } => ("sanitize", args.as_slice()),
            _ => (self.diagnostic(), &[][..]),
        }
    }

    fn diagnostic(&self) -> &'static str {
        match self {
            Action::Build => "build",
            Action::Parity => "parity",
            Action::Resident => "resident",
            Action::Prefill => "prefill",
            Action::ModelCheck => "model-check",
            Action::DiffusionCheck => "diffusion-check",
            Action::Bench => "bench",
            _ => unreachable!("action requires arguments or has no native command"),
        }
    }
}

fn run_modal(
    root: &Path,
    candidates: usize,
    history: usize,
    iterations: usize,
    out: &Path,
) -> Result<(), String> {
    if candidates == 0 || history == 0 || iterations == 0 {
        return Err("CUDA Modal BO dimensions must be positive".into());
    }
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let status = Command::new(root.join("buck2w"))
        .current_dir(root)
        .args(["run", "//rust/crates/modal-runner:ennx-modal", "--", "bo"])
        .arg("--candidates")
        .arg(candidates.to_string())
        .arg("--history")
        .arg(history.to_string())
        .arg("--iterations")
        .arg(iterations.to_string())
        .arg("--output")
        .arg(out)
        .status()
        .map_err(|error| error.to_string())?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("Modal CUDA BO gate exited with {status}"))
    }
}

#[allow(clippy::too_many_arguments)]
fn generate_t4(
    root: &Path,
    context: usize,
    prompt: usize,
    visits: usize,
    temperature: &str,
    seed: u64,
    unroll: usize,
    chunk: usize,
    out: &Path,
) -> Result<(), String> {
    let temperature_value = temperature
        .parse::<f32>()
        .map_err(|_| "CUDA generation temperature must be a number")?;
    if context == 0
        || prompt == 0
        || prompt >= context
        || visits == 0
        || !temperature_value.is_finite()
        || temperature_value < 0.0
        || unroll == 0
        || unroll > 32
        || (chunk != 0
            && (!(64..=4096).contains(&chunk) || !chunk.is_power_of_two() || context % chunk != 0))
    {
        return Err("invalid CUDA generation dimensions or sampling parameters".into());
    }
    let status = Command::new(root.join("buck2w"))
        .current_dir(root)
        .args([
            "run",
            "//rust/crates/modal-runner:ennx-modal",
            "--",
            "generate",
        ])
        .arg("--context")
        .arg(context.to_string())
        .arg("--prompt")
        .arg(prompt.to_string())
        .arg("--visits")
        .arg(visits.to_string())
        .arg("--temperature")
        .arg(temperature.to_string())
        .arg("--seed")
        .arg(seed.to_string())
        .arg("--unroll")
        .arg(unroll.to_string())
        .arg("--chunk")
        .arg(chunk.to_string())
        .arg("--output")
        .arg(out)
        .status()
        .map_err(|error| error.to_string())?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("Modal CUDA generation exited with {status}"))
    }
}

pub(crate) fn run(root: &Path, action: Action) -> Result<(), String> {
    match action {
        Action::Inspect => super::model_program::cudaprefill(),
        Action::Modal {
            candidates,
            history,
            iterations,
            out,
        } => run_modal(root, candidates, history, iterations, &out),
        Action::GenerateT4 {
            context,
            prompt,
            visits,
            temperature,
            seed,
            unroll,
            chunk,
            out,
        } => generate_t4(
            root,
            context,
            prompt,
            visits,
            &temperature,
            seed,
            unroll,
            chunk,
            &out,
        ),
        other => {
            if !cfg!(target_os = "linux") {
                return Err("CUDA execution requires an NVIDIA Linux host; use the same ./ennx cuda command on the T4 session".into());
            }
            let (name, args) = other.command();
            let status = Command::new(root.join("tools/cuda-run"))
                .arg(name)
                .args(args)
                .current_dir(root)
                .status()
                .map_err(|error| format!("start CUDA execution: {error}"))?;
            if status.success() {
                Ok(())
            } else {
                Err(format!("CUDA execution exited with {status}"))
            }
        }
    }
}
