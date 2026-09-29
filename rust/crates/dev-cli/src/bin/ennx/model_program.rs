use clap::{Subcommand, ValueEnum};
use ennx::experimental::{
    Backend, BackendSchedule, Compiler, ModelProgram, ProgramKind, SemanticOp,
};
use ennx_wire::json::{json, pretty_string};
use std::env;
use std::path::PathBuf;

#[derive(Debug, PartialEq, Eq, Subcommand)]
pub(crate) enum Action {
    /// Show the operations assigned to a backend and the operations still missing.
    ///
    /// Inspects the semantic execution graph against compiler lowering rules
    /// for Metal, CUDA-Oxide, or Apache TVM targets, reporting kernel coverage,
    /// tiling geometry, and unimplemented semantic boundaries.
    Inspect {
        /// Target compute backend to evaluate lowering coverage for.
        #[arg(long, value_enum, default_value_t = BackendName::Metal, help_heading = "Backend Selection")]
        backend: BackendName,
        /// Program function entry point to analyze.
        #[arg(long, value_enum, default_value_t = FunctionName::ExperimentRound, help_heading = "Backend Selection")]
        function: FunctionName,
        /// Exit with non-zero status if any operation in the graph is not assigned to a kernel.
        #[arg(long, help_heading = "Validation Rules")]
        strict: bool,
        /// Emit results as structured machine-readable JSON.
        #[arg(long, help_heading = "Reporting Options")]
        json: bool,
    },
    /// Decode generated vocabulary IDs with the corpus byte-level BPE tokenizer.
    ///
    /// Reconstructs UTF-8 string output from raw token integer arrays emitted
    /// during verification and generation rounds.
    Decode {
        /// Path to the tokenizer.json file.
        tokenizer: PathBuf,
        /// Path to the binary or JSON token sequence file.
        tokens: PathBuf,
        /// Path where the decoded text should be written.
        output: PathBuf,
    },
    /// Probe for an optional Apache TVM compiler/runtime installation.
    ///
    /// Checks for the presence of TVM / TIRX shared libraries, compiler passes,
    /// and target runtime bindings.
    TirxProbe {
        /// Emit probe findings as machine-readable JSON.
        #[arg(long, help_heading = "Reporting Options")]
        json: bool,
    },
    /// Measure cached PISA attention over up to one million context tokens.
    ///
    /// Constructs a persistent logarithmic PISA pyramid index over the declared token count,
    /// benchmarks hierarchical tree build latency, and executes bounded gather queries
    /// across unified memory to verify sub-millisecond retrieval bounds.
    Context {
        /// Total context sequence length in tokens (powers of two up to 1,048,576).
        #[arg(long, default_value_t = 1_048_576, help_heading = "Context Dimensions")]
        tokens: u32,
        /// Number of query positions evaluated in parallel per benchmark iteration.
        #[arg(long, default_value_t = 128, help_heading = "Context Dimensions")]
        queries: u32,
        /// Number of repetition passes for statistical timing stability.
        #[arg(long, default_value_t = 5, help_heading = "Benchmark Options")]
        repeats: u32,
    },
    /// Measure complete BO rounds with the stated number of newly generated tokens.
    ///
    /// Loads a full experiment specification and token dataset, executes complete rounds
    /// including candidate proposal, perturbation, generation, prefix verification,
    /// scoring, and Gaussian process update across extreme context budgets.
    ContextLoop {
        /// Path to the experiment configuration TOML file.
        config: PathBuf,
        /// Path to the token dataset file.
        dataset: PathBuf,
        /// Number of newly generated tokens per round.
        #[arg(long, default_value_t = 1_048_576, help_heading = "Workload Options")]
        generated: u32,
        /// Number of prompt tokens preceding generation.
        #[arg(long, default_value_t = 128, help_heading = "Workload Options")]
        prompt: u32,
        /// Number of complete Bayesian optimization rounds to execute sequentially.
        #[arg(long, default_value_t = 1, help_heading = "Workload Options")]
        rounds: u32,
    },
    /// Calculate the model work and throughput required by a generation loop.
    ///
    /// Evaluates arithmetic FLOP counts, parameter coordinate densities, memory bandwidth
    /// traffic, and required sustained TFLOP/s to reach the target execution latency.
    /// Supports comparing flat baseline execution against 2-stage hierarchical context compression.
    Scale {
        /// Number of newly generated suffix tokens to emit during the repair loop.
        #[arg(
            long,
            default_value_t = 1_048_576,
            help_heading = "Workload Configuration"
        )]
        generated: u64,
        /// Number of prompt/context tokens processed in the first prefill wave.
        #[arg(long, default_value_t = 128, help_heading = "Workload Configuration")]
        prompt: u64,
        /// Total model positions actually evaluated, including verifier retries and repair waves.
        #[arg(long, help_heading = "Workload Configuration")]
        evaluated: Option<u64>,
        /// Measured complete-round wall time in milliseconds from a benchmark execution.
        #[arg(long, help_heading = "Execution & Latency Bounds")]
        round_ms: Option<u64>,
        /// Target wall-clock duration in milliseconds (default: 1000 ms for subsecond milestone).
        #[arg(
            long,
            default_value_t = 1000,
            help_heading = "Execution & Latency Bounds"
        )]
        target_ms: u64,
        /// Evaluate scale under asymmetric 2-stage hierarchical context compression.
        #[arg(long, help_heading = "Architecture & Compression Strategy")]
        hierarchical: bool,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum BackendName {
    Metal,
    CudaOxide,
    TvmMetal,
    TvmCuda,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum FunctionName {
    Prefill,
    Decode,
    Draft,
    Verify,
    Score,
    Objective,
    ExperimentRound,
}

pub(crate) fn run(action: Action) -> Result<(), String> {
    match action {
        Action::Inspect {
            backend,
            function,
            strict,
            json,
        } => inspect(backend, function, strict, json),
        Action::TirxProbe { json } => tirx_probe(json),
        Action::Decode {
            tokenizer,
            tokens,
            output,
        } => decode(&tokenizer, &tokens, &output),
        Action::Context {
            tokens,
            queries,
            repeats,
        } => context(tokens, queries, repeats),
        Action::ContextLoop {
            config,
            dataset,
            generated,
            prompt,
            rounds,
        } => context_loop(&config, &dataset, generated, prompt, rounds),
        Action::Scale {
            generated,
            prompt,
            evaluated,
            round_ms,
            target_ms,
            hierarchical,
        } => scale(
            generated,
            prompt,
            evaluated,
            round_ms,
            target_ms,
            hierarchical,
        ),
    }
}

fn scale(
    generated: u64,
    prompt: u64,
    evaluated: Option<u64>,
    round_ms: Option<u64>,
    target_ms: u64,
    hierarchical: bool,
) -> Result<(), String> {
    const PARAMETERS: u64 = 1_047_699_736;
    const VISITS: u64 = 7;
    const COEFFICIENTS: u64 = 19_852_800;
    const FLOPS: u64 = 2 * COEFFICIENTS;
    const KV_BYTES: u64 = 2 * 64 * 2;
    const TREE_BYTES: u64 = 2 * 64 / 64 * 2;

    let evaluated = evaluated.unwrap_or(generated);
    if generated == 0
        || prompt == 0
        || evaluated < generated
        || target_ms == 0
        || round_ms == Some(0)
    {
        return Err(
            "scale requires positive lengths and times, with evaluated >= generated".into(),
        );
    }
    let positions = prompt
        .checked_add(generated)
        .and_then(|count| count.checked_sub(1))
        .ok_or("context length overflow")?;
    let broad = positions.div_ceil(128) * 128;
    let capacity = positions
        .checked_next_power_of_two()
        .ok_or("context capacity overflow")?
        .max(4096);
    let cache_per_visit = capacity
        .checked_mul(KV_BYTES + TREE_BYTES)
        .ok_or("cache size overflow")?;
    let cache_bytes = cache_per_visit
        .checked_mul(VISITS)
        .ok_or("cache size overflow")?;
    let flops = FLOPS.checked_mul(evaluated).ok_or("work count overflow")?;
    let broad_flops = FLOPS.checked_mul(broad).ok_or("work count overflow")?;
    let repair = evaluated.saturating_sub(broad);
    let target_seconds = target_ms as f64 / 1000.0;
    let measured_tflops = round_ms.map(|ms| flops as f64 / (ms as f64 / 1000.0) / 1.0e12);
    let speedup = round_ms.map(|ms| ms as f64 / target_ms as f64);

    if hierarchical {
        const SUBSURFACE_FLOPS: u64 = 1_500;
        const POOLING_FACTOR: u64 = 32;
        let super_tokens = prompt.div_ceil(POOLING_FACTOR);
        let subsurface_flops = prompt
            .checked_mul(SUBSURFACE_FLOPS)
            .ok_or("work count overflow")?;
        let core_positions = super_tokens
            .checked_add(generated)
            .ok_or("position overflow")?;
        let core_flops = FLOPS
            .checked_mul(core_positions)
            .ok_or("work count overflow")?;
        let total_flops = subsurface_flops
            .checked_add(core_flops)
            .ok_or("work count overflow")?;
        let hierarchical_target_tflops = total_flops as f64 / target_seconds / 1.0e12;
        let arithmetic_reduction = broad_flops as f64 / total_flops as f64;
        let realistic_t4_seconds = (total_flops as f64 / 20.0e12) + 0.0038;
        let realistic_metal_seconds = (total_flops as f64 / 14.0e12) + 0.0038;

        println!(
            "{}",
            pretty_string(&json!({
                "model": "fbt-pisa1-hierarchical-v1",
                "parameters": PARAMETERS,
                "physical-layers": 5,
                "executed-layer-visits": VISITS,
                "prompt-tokens": prompt,
                "generated-tokens": generated,
                "logical-context-positions": positions,
                "subsurface-compression-ratio": POOLING_FACTOR,
                "compressed-super-tokens": super_tokens,
                "subsurface-flops-per-token": SUBSURFACE_FLOPS,
                "subsurface-prefill-flops": subsurface_flops,
                "core-evaluated-positions": core_positions,
                "core-contraction-flops": core_flops,
                "total-hierarchical-flops": total_flops,
                "flat-baseline-broad-pass-flops": broad_flops,
                "arithmetic-reduction-ratio": arithmetic_reduction,
                "target-ms": target_ms,
                "target-generated-tokens-per-second": generated as f64 / target_seconds,
                "hierarchical-target-tflops": hierarchical_target_tflops,
                "measured-flat-round-ms": round_ms,
                "required-wall-speedup-from-flat": speedup,
                "realistic-projected-t4-ms": realistic_t4_seconds * 1000.0,
                "realistic-projected-metal-ms": realistic_metal_seconds * 1000.0,
                "architecture-verdict": if hierarchical_target_tflops <= 20.0 { "subsecond-achievable" } else { "exceeds-sustained-tflops" },
                "counting": "hierarchical: 1-pass sub-surface context encoding + 32x pooled super-tokens + suffix evaluated through 7 recurrent MoE visits"
            }))
            .map_err(|error| error.to_string())?
        );
        return Ok(());
    }
    println!(
        "{}",
        pretty_string(&json!({
            "model": "fbt-pisa1-looped-mhc4-v1",
            "parameters": PARAMETERS,
            "physical-layers": 5,
            "executed-layer-visits": VISITS,
            "prompt-tokens": prompt,
            "generated-tokens": generated,
            "logical-context-positions": positions,
            "cache-capacity": capacity,
            "cache-bytes-all-visits": cache_bytes,
            "broad-pass-positions": broad,
            "evaluated-model-positions": evaluated,
            "repair-positions": repair,
            "repair-fraction-of-evaluated-work": repair as f64 / evaluated as f64,
            "verifier-amplification": evaluated as f64 / generated as f64,
            "parameter-coordinates-per-generated-token": PARAMETERS as f64 / generated as f64,
            "contraction-coefficients-per-model-position": COEFFICIENTS,
            "contraction-coefficient-fraction-of-arena": COEFFICIENTS as f64 / PARAMETERS as f64,
            "contraction-flops-per-model-position": FLOPS,
            "contraction-flops-per-generated-token": flops as f64 / generated as f64,
            "one-broad-pass-contraction-flops": broad_flops,
            "total-contraction-flops": flops,
            "target-ms": target_ms,
            "target-generated-tokens-per-second": generated as f64 / target_seconds,
            "one-broad-pass-target-tflops": broad_flops as f64 / target_seconds / 1.0e12,
            "target-effective-tflops": flops as f64 / target_seconds / 1.0e12,
            "measured-round-ms": round_ms,
            "measured-effective-tflops": measured_tflops,
            "required-wall-speedup": speedup,
            "counting": "dense contraction multiply and add counted separately; excludes attention selection, mHC, normalization, routing logistics, sampling, proposal, and scoring",
        }))
        .map_err(|error| error.to_string())?
    );
    Ok(())
}

fn context_loop(
    config: &std::path::Path,
    dataset: &std::path::Path,
    generated: u32,
    prompt: u32,
    rounds: u32,
) -> Result<(), String> {
    #[cfg(target_os = "macos")]
    {
        if rounds == 0 {
            return Err("context loop requires positive rounds".into());
        }
        let (mut settings, _) = ennx::config::load_tune(config)?;
        settings.rounds = Some(rounds);
        settings.reps = Some(1);
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| e.to_string())?
            .as_millis();
        let output = PathBuf::from(".cache/ennx/runs/context-loop")
            .join(format!("run-{stamp}-{}", std::process::id()));
        std::fs::create_dir_all(output.parent().unwrap()).map_err(|e| e.to_string())?;
        std::fs::create_dir(&output).map_err(|e| e.to_string())?;
        std::fs::copy(config, output.join("experiment.toml")).map_err(|e| e.to_string())?;
        eprintln!(
            "ENNX_CONTEXT_LOOP generated={generated} prompt={prompt} rounds={rounds} artifact={}",
            output.display()
        );
        ennx::experimental::context_loop(&settings, dataset, &output, generated, prompt)?;
        println!("{}", output.display());
        Ok(())
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (config, dataset, generated, prompt, rounds);
        Err("complete BO context loop currently requires Metal".into())
    }
}

fn context(tokens: u32, queries: u32, repeats: u32) -> Result<(), String> {
    #[cfg(target_os = "macos")]
    {
        let layout = ennx::context::Layout::new(tokens, queries)?;
        let mut cache = ennx::experimental::ContextCache::new(layout)?;
        let report = ennx::context::probe::run(&mut cache, layout, repeats)?;
        println!("{}", pretty_string(&json!({
            "stage": "metal-cached-pisa-attention", "context": tokens, "queries": queries,
            "kv-bytes-per-visit": layout.kv_bytes(), "tree-bytes-per-visit": layout.tree_bytes(),
            "query-work-bytes": layout.work_bytes(), "selected-token-budget": 512,
            "tree-build-ms": report.build_ms, "repair-ms": report.repair_ms,
            "median-query-device-ms": report.device_ms, "median-query-wall-ms": report.wall_ms,
            "max-abs-error": report.error, "checked-ranges": report.ranges,
            "full-model": false, "learned-capability": false
        })).map_err(|e| e.to_string())?);
        Ok(())
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (tokens, queries, repeats);
        Err("use ./ennx cuda context on an NVIDIA Linux host".into())
    }
}

fn decode(
    tokenizer: &std::path::Path,
    tokens: &std::path::Path,
    output: &std::path::Path,
) -> Result<(), String> {
    use std::io::Write;
    let decoder = ennx::text::ByteDecoder::load(tokenizer)?;
    let tokens: Vec<u32> =
        ennx_wire::json::from_reader(std::fs::File::open(tokens).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    let bytes = decoder.decode_bytes(&tokens)?;
    std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(output)
        .map_err(|e| e.to_string())?
        .write_all(&bytes)
        .map_err(|e| e.to_string())?;
    println!(
        "decoded {} tokens into {} bytes: {}",
        tokens.len(),
        bytes.len(),
        output.display()
    );
    Ok(())
}

fn tirx_probe(as_json: bool) -> Result<(), String> {
    let paths = tvm_paths();
    let compiler = find_library(
        &paths,
        &[
            "libtvm_compiler.dylib",
            "libtvm_compiler.so",
            "libtvm_compiler.so.0",
            "tvm_compiler.dll",
        ],
    );
    let runtime = find_library(
        &paths,
        &[
            "libtvm_runtime.dylib",
            "libtvm_runtime.so",
            "libtvm_runtime.so.0",
            "tvm_runtime.dll",
        ],
    );
    let paired = compiler.is_some() && runtime.is_some();
    let status = if paired {
        "libraries-present"
    } else {
        "unavailable"
    };
    let note = if paired {
        "compiler and runtime libraries were found; TIRx API registration and target support are not verified"
    } else {
        "set TVM_LIBRARY_PATH to a TVM build directory containing libtvm_compiler and libtvm_runtime"
    };
    if as_json {
        println!(
            "{}",
            pretty_string(&json!({
                "integration": "apache-tvm-tirx",
                "status": status,
                "compiler-runtime-present": paired,
                "compiler-library": compiler.as_ref().map(|path| path.display().to_string()),
                "runtime-library": runtime.as_ref().map(|path| path.display().to_string()),
                "note": note,
            }))
            .map_err(|error| error.to_string())?
        );
    } else {
        println!("Apache TVM TIRx: {status}");
        if let Some(path) = compiler {
            println!("compiler  {}", path.display());
        }
        if let Some(path) = runtime {
            println!("runtime   {}", path.display());
        }
        println!("{note}");
    }
    Ok(())
}

fn tvm_paths() -> Vec<PathBuf> {
    let mut paths = env::var_os("TVM_LIBRARY_PATH")
        .into_iter()
        .flat_map(|value| env::split_paths(&value).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    if let Some(home) = env::var_os("TVM_HOME") {
        paths.push(PathBuf::from(home).join("build"));
    }
    paths
}

fn find_library(paths: &[PathBuf], names: &[&str]) -> Option<PathBuf> {
    paths
        .iter()
        .flat_map(|directory| names.iter().map(|name| directory.join(name)))
        .find(|path| path.is_file())
}

pub(crate) fn cudaprefill() -> Result<(), String> {
    inspect(BackendName::CudaOxide, FunctionName::Prefill, false, false)
}

fn inspect(
    backend: BackendName,
    function: FunctionName,
    strict: bool,
    json_output: bool,
) -> Result<(), String> {
    let program = ModelProgram::fbt_pisa();
    let schedule = schedule(backend)?;
    schedule.validate(&program)?;
    let kind = function.kind();
    let graph = program
        .function(kind)
        .ok_or("selected model function is missing")?;
    let missing = schedule.coverage(&program, kind)?;
    let recurrent = program
        .recurrent_core
        .ok_or("selected model has no recurrent core")?;
    let visits = program.layer_visits(recurrent.training_visits)?;
    let kernels = schedule
        .kernels
        .iter()
        .filter(|kernel| kernel.nodes.iter().any(|node| graph.nodes.contains(node)))
        .collect::<Vec<_>>();

    if json_output {
        let kernels = kernels
            .iter()
            .map(|kernel| {
                json!({
                    "kernel": kernel.kernel,
                    "nodes": kernel.nodes,
                    "tile": kernel.tile,
                    "threads": kernel.threads,
                })
            })
            .collect::<Vec<_>>();
        let missing = missing.iter().map(|op| op_name(*op)).collect::<Vec<_>>();
        println!(
            "{}",
            pretty_string(&json!({
                "model": program.name,
                "version": program.version,
                "function": graph.name,
                "backend": backend_name(schedule.backend),
                "compiler": compiler_name(schedule.compiler),
                "nodes": graph.nodes,
                "kernels": kernels,
                "missing": missing,
                "schedule-complete": missing.is_empty(),
                "coverage-kind": "lowering-plan",
                "physical-layers": program.physical_layers,
                "recurrent-core": {
                    "first-layer": recurrent.first_layer,
                    "layer-count": recurrent.layer_count,
                    "training-visits": recurrent.training_visits,
                    "min-inference-visits": recurrent.min_inference_visits,
                    "max-inference-visits": recurrent.max_inference_visits,
                },
                "layer-visits": visits.iter().map(|visit| json!({
                    "layer": visit.layer,
                    "visit": visit.visit,
                    "execution": visit.execution,
                    "recurrent": visit.recurrent,
                })).collect::<Vec<_>>(),
            }))
            .map_err(|error| error.to_string())?
        );
    } else {
        println!(
            "{} v{} {} | {} via {}",
            program.name,
            program.version,
            graph.name,
            backend_name(schedule.backend),
            compiler_name(schedule.compiler)
        );
        println!(
            "layers={} recurrent={}..{} training-visits={} inference-visits={}..={} executions={}",
            program.physical_layers,
            recurrent.first_layer,
            recurrent.first_layer + recurrent.layer_count,
            recurrent.training_visits,
            recurrent.min_inference_visits,
            recurrent.max_inference_visits,
            visits.len(),
        );
        println!(
            "execution: {}",
            visits
                .iter()
                .map(|visit| format!("L{}:V{}", visit.layer, visit.visit))
                .collect::<Vec<_>>()
                .join(" -> ")
        );
        for kernel in kernels {
            println!("{} <- {}", kernel.kernel, kernel.nodes.join(", "));
        }
        if missing.is_empty() {
            println!(
                "schedule coverage complete; execution evidence is reported by backend checks"
            );
        } else {
            println!(
                "missing: {}",
                missing
                    .iter()
                    .map(|op| op_name(*op))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
    }
    if strict && !missing.is_empty() {
        return Err("backend lowering plan is incomplete".into());
    }
    Ok(())
}

fn schedule(backend: BackendName) -> Result<BackendSchedule, String> {
    match backend {
        BackendName::Metal => Ok(BackendSchedule::existing_metal()),
        BackendName::CudaOxide => Ok(BackendSchedule::cuda_oxide()),
        BackendName::TvmMetal => Ok(BackendSchedule::tvm_metal()),
        BackendName::TvmCuda => Ok(BackendSchedule::tvm_cuda()),
    }
}

impl FunctionName {
    fn kind(self) -> ProgramKind {
        match self {
            Self::Prefill => ProgramKind::Prefill,
            Self::Decode => ProgramKind::Decode,
            Self::Draft => ProgramKind::Draft,
            Self::Verify => ProgramKind::Verify,
            Self::Score => ProgramKind::Score,
            Self::Objective => ProgramKind::Objective,
            Self::ExperimentRound => ProgramKind::ExperimentRound,
        }
    }
}

fn backend_name(backend: Backend) -> &'static str {
    match backend {
        Backend::Metal => "metal",
        Backend::CudaOxide => "cuda-oxide",
    }
}

fn compiler_name(compiler: Compiler) -> &'static str {
    match compiler {
        Compiler::Native => "native",
        Compiler::CudaOxide => "cuda-oxide",
        Compiler::Tvm => "tvm",
    }
}

fn op_name(op: SemanticOp) -> &'static str {
    match op {
        SemanticOp::Embed => "embed",
        SemanticOp::Normalize => "normalize",
        SemanticOp::ProjectQkv => "project-qkv",
        SemanticOp::PisaSummaries => "pisa-summaries",
        SemanticOp::PisaSelect => "pisa-select",
        SemanticOp::PisaAttention => "pisa-attention",
        SemanticOp::ProjectAttention => "project-attention",
        other => moe_name(other),
    }
}

fn moe_name(op: SemanticOp) -> &'static str {
    match op {
        SemanticOp::RouteTopK => "route-top-k",
        SemanticOp::PackRoutes => "pack-routes",
        SemanticOp::RoutedGateUp => "routed-gate-up",
        SemanticOp::RoutedDown => "routed-down",
        SemanticOp::CombineRoutes => "combine-routes",
        other => tail_name(other),
    }
}

fn tail_name(op: SemanticOp) -> &'static str {
    match op {
        SemanticOp::Residual => "residual",
        SemanticOp::Feedback => "feedback",
        SemanticOp::Readout => "readout",
        SemanticOp::Sample => "sample",
        SemanticOp::Draft => "draft",
        SemanticOp::Verify => "verify",
        SemanticOp::ScoreGeneration => "score-generation",
        SemanticOp::Objective => "objective",
        _ => unreachable!("operation is named by op_name"),
    }
}
