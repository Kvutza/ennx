use clap::{Subcommand, ValueEnum};
use std::fs;
use std::path::Path;
use std::process::Command;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(crate) enum Units {
    Cpu,
    Gpu,
    Ane,
    All,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(crate) enum Weights {
    Fixed,
    Input,
}

impl Weights {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Fixed => "fixed",
            Self::Input => "input",
        }
    }
}

impl Units {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Gpu => "gpu",
            Self::Ane => "ane",
            Self::All => "all",
        }
    }
}

#[derive(Debug, PartialEq, Eq, Subcommand)]
pub(crate) enum Action {
    /// Measure a shape-faithful FP16 readout and report Core ML placement.
    ///
    /// Compiles a Core ML model instance with shape [rows, width] -> [rows, outputs],
    /// measures device wall time, and verifies whether Apple Neural Engine (ANE)
    /// compute units were selected by the runtime without falling back to CPU.
    Probe {
        /// Number of input sequence rows / batch tokens evaluated.
        #[arg(
            long,
            default_value_t = 128,
            help_heading = "Matrix Geometry & Dimensions"
        )]
        rows: usize,
        /// Hidden feature dimension / channel width of the input representation.
        #[arg(
            long,
            default_value_t = 512,
            help_heading = "Matrix Geometry & Dimensions"
        )]
        width: usize,
        /// Output projection dimensionality (e.g. vocabulary size or projection targets).
        #[arg(
            long,
            default_value_t = 8_192,
            help_heading = "Matrix Geometry & Dimensions"
        )]
        outputs: usize,
        /// Number of measured execution repetitions to record.
        #[arg(
            long,
            default_value_t = 7,
            help_heading = "Benchmarking & Repetition Protocol"
        )]
        repeats: usize,
        /// Number of unmeasured warmup rounds to prime Core ML compilation caches.
        #[arg(
            long,
            default_value_t = 3,
            help_heading = "Benchmarking & Repetition Protocol"
        )]
        warmups: usize,
        /// Target compute units permitted by MLModelConfiguration (ane, gpu, cpu, or all).
        #[arg(long, value_enum, default_value_t = Units::Ane, help_heading = "Hardware Execution & Placement")]
        units: Units,
        /// Weight parameter residency mode: runtime input tensor or precompiled model constant.
        #[arg(long, value_enum, default_value_t = Weights::Input, help_heading = "Hardware Execution & Placement")]
        weights: Weights,
    },
}

pub(crate) fn run(root: &Path, action: Action) -> Result<(), String> {
    match action {
        Action::Probe {
            rows,
            width,
            outputs,
            repeats,
            warmups,
            units,
            weights,
        } => probe(root, rows, width, outputs, repeats, warmups, units, weights),
    }
}

fn probe(
    root: &Path,
    rows: usize,
    width: usize,
    outputs: usize,
    repeats: usize,
    warmups: usize,
    units: Units,
    weights: Weights,
) -> Result<(), String> {
    if rows == 0 || width == 0 || outputs == 0 || repeats == 0 {
        return Err("rows, width, outputs, and repeats must be positive".into());
    }
    let source = root.join("apple/ane/EnnxANE.swift");
    let shared = root.join("apple/ane/ReadoutProbe.swift");
    let cache = root.join(".cache/ennx/ane");
    let executable = cache.join("ennx-ane");
    fs::create_dir_all(&cache).map_err(|error| format!("create ANE cache: {error}"))?;
    let rebuild = executable
        .metadata()
        .and_then(|binary| {
            let binary_time = binary.modified().ok();
            let source_time = source.metadata()?.modified().ok();
            let shared_time = shared.metadata()?.modified().ok();
            Ok(source_time > binary_time || shared_time > binary_time)
        })
        .unwrap_or(true);
    if rebuild {
        let status = Command::new("xcrun")
            .current_dir(root)
            .args(["swiftc", "-O", "-parse-as-library"])
            .arg(&shared)
            .arg(&source)
            .args(["-framework", "CoreML", "-o"])
            .arg(&executable)
            .status()
            .map_err(|error| format!("compile ANE probe: {error}"))?;
        if !status.success() {
            return Err(format!("compile ANE probe exited with {status}"));
        }
    }
    let status = Command::new(&executable)
        .current_dir(root)
        .args(["--rows", &rows.to_string()])
        .args(["--width", &width.to_string()])
        .args(["--outputs", &outputs.to_string()])
        .args(["--repeats", &repeats.to_string()])
        .args(["--warmups", &warmups.to_string()])
        .args(["--units", units.as_str()])
        .args(["--weights", weights.as_str()])
        .status()
        .map_err(|error| format!("run ANE probe: {error}"))?;
    status
        .success()
        .then_some(())
        .ok_or_else(|| format!("ANE probe exited with {status}"))
}
