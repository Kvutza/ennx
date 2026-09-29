use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use clap::{Subcommand, ValueEnum};
use ptx_synth::{
    TuringGemmConfig, synthesize_turing_fp16_gemm, synthesize_turing_fused_rademacher,
    synthesize_turing_vector_scale,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum Kernel {
    Gemm,
    Rademacher,
    Vector,
}

#[derive(Debug, PartialEq, Eq, Subcommand)]
pub enum Action {
    /// Write one synthesized PTX module to disk or stdout.
    ///
    /// Generates symbolic PTX source for NVIDIA Turing GPUs (SM75), targeting
    /// tensor-core MMA operations, fused perturbation distributions, or vector scaling.
    Emit {
        /// Kernel architecture recipe to synthesize.
        #[arg(value_enum, help_heading = "Kernel Selection")]
        kernel: Kernel,
        /// Target file path to write the emitted PTX code; writes to stdout when omitted.
        #[arg(long, help_heading = "Output Options")]
        out: Option<PathBuf>,
    },
    /// Execute synthesized kernels and their parity gates on a Modal T4.
    ///
    /// Submits the synthesized PTX module to a cloud-managed NVIDIA Tesla T4 worker,
    /// verifies execution under varying problem sizes, checks numerical parity against CPU,
    /// and logs benchmark timing records.
    Modal {
        /// Number of tensor/array elements evaluated in the benchmark problem.
        #[arg(long, default_value_t = 1_048_576, help_heading = "Benchmark Workload")]
        elements: u32,
        /// Number of repetition iterations executed to establish stable median latency.
        #[arg(long, default_value_t = 100, help_heading = "Benchmark Workload")]
        iterations: u32,
        /// File path where benchmark timing and parity verification JSON should be saved.
        #[arg(
            long,
            default_value = "results/ptx/t4-kernels.json",
            help_heading = "Output Options"
        )]
        out: PathBuf,
    },
}

pub fn run(root: &Path, action: Action) -> Result<(), String> {
    match action {
        Action::Emit { kernel, out } => emit(kernel, out.as_deref()),
        Action::Modal {
            elements,
            iterations,
            out,
        } => modal(root, elements, iterations, &out),
    }
}

fn emit(kernel: Kernel, output: Option<&Path>) -> Result<(), String> {
    let ptx = match kernel {
        Kernel::Gemm => synthesize_turing_fp16_gemm(&TuringGemmConfig::default()),
        Kernel::Rademacher => synthesize_turing_fused_rademacher("ennx_turing_rademacher"),
        Kernel::Vector => synthesize_turing_vector_scale("ennx_turing_vector_scale"),
    };
    if let Some(path) = output {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        }
        fs::write(path, ptx).map_err(|error| error.to_string())
    } else {
        print!("{ptx}");
        Ok(())
    }
}

fn modal(root: &Path, elements: u32, iterations: u32, output: &Path) -> Result<(), String> {
    if elements == 0 || elements % 4 != 0 || iterations == 0 {
        return Err(
            "PTX Modal gate requires positive iterations and an element count divisible by four"
                .into(),
        );
    }
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let status = Command::new(root.join("buck2w"))
        .current_dir(root)
        .args([
            "run",
            "//rust/crates/modal-runner:ennx-modal",
            "--",
            "ptx",
            "--elements",
            &elements.to_string(),
            "--iterations",
            &iterations.to_string(),
            "--output",
        ])
        .arg(output)
        .status()
        .map_err(|error| error.to_string())?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("Modal PTX gate exited with {status}"))
    }
}
