use clap::ValueEnum;
use std::io::{self, Write};

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum Shell {
    Zsh,
    Bash,
    Fish,
}

pub fn run(shell: Shell) -> Result<(), String> {
    let mut stdout = io::stdout().lock();
    match shell {
        Shell::Zsh => {
            stdout
                .write_all(ZSH_COMPLETION.as_bytes())
                .map_err(|e| e.to_string())?;
        }
        Shell::Bash => {
            stdout
                .write_all(BASH_COMPLETION.as_bytes())
                .map_err(|e| e.to_string())?;
        }
        Shell::Fish => {
            stdout
                .write_all(FISH_COMPLETION.as_bytes())
                .map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

const ZSH_COMPLETION: &str = r#"#compdef ennx ./ennx

autoload -U is-at-least

_ennx() {
    typeset -A opt_args
    typeset -a _arguments_options
    local ret=1

    if is-at-least 5.2; then
        _arguments_options=(-s -S -C)
    else
        _arguments_options=(-s -C)
    fi

    local context curcontext="$curcontext" state line
    _arguments "${_arguments_options[@]}" : \
        '-h[Print help (see a summary with -h)]' \
        '--help[Print help (see a summary with -h)]' \
        '-V[Print version]' \
        '--version[Print version]' \
        ":: :_ennx_commands" \
        "*::: :->subcmd" \
        && ret=0

    case $state in
    (subcmd)
        case $words[1] in
        (target)
            _arguments "${_arguments_options[@]}" : \
                '-h[Print help]' '--help[Print help]' \
                ":: :_ennx_target_commands" \
                "*::: :->target_sub" && ret=0
            case $state in
            (target_sub)
                case $words[1] in
                (cuda)
                    _arguments "${_arguments_options[@]}" : \
                        '-h[Print help]' '--help[Print help]' \
                        ":: :_ennx_cuda_commands" \
                        "*::: :->cuda_sub" && ret=0
                    case $state in
                    (cuda_sub)
                        case $words[1] in
                        (generate-t4)
                            _arguments "${_arguments_options[@]}" : \
                                '--context=[Context sequence capacity in tokens]:CONTEXT:_default' \
                                '--prompt=[Prompt prefix length in tokens]:PROMPT:_default' \
                                '--visits=[Recurrent layer visits per wave]:VISITS:_default' \
                                '--temperature=[Softmax sampling temperature]:TEMPERATURE:_default' \
                                '--seed=[PRNG seed for reproducible sampling]:SEED:_default' \
                                '--unroll=[Verifier wave unroll factor]:UNROLL:_default' \
                                '--chunk=[Activation chunk size (0=packed)]:CHUNK:_default' \
                                '--out=[Output path for serialized records]:OUT:_files' \
                                '-h[Print help]' '--help[Print help]' && ret=0
                            ;;
                        (modal)
                            _arguments "${_arguments_options[@]}" : \
                                '--candidates=[Candidate points acquired per batch]:CANDIDATES:_default' \
                                '--history=[Historical observation capacity in GP]:HISTORY:_default' \
                                '--iterations=[Bayesian optimization iterations]:ITERATIONS:_default' \
                                '--out=[Output path for convergence curve JSON]:OUT:_files' \
                                '-h[Print help]' '--help[Print help]' && ret=0
                            ;;
                        esac
                        ;;
                    esac
                    ;;
                (ptx)
                    _arguments "${_arguments_options[@]}" : \
                        '-h[Print help]' '--help[Print help]' \
                        ":: :_ennx_ptx_commands" \
                        "*::: :->ptx_sub" && ret=0
                    case $state in
                    (ptx_sub)
                        case $words[1] in
                        (emit)
                            _arguments "${_arguments_options[@]}" : \
                                '--out=[Target file path for emitted PTX]:OUT:_files' \
                                '-h[Print help]' '--help[Print help]' \
                                ':kernel:(gemm rademacher vector)' && ret=0
                            ;;
                        (modal)
                            _arguments "${_arguments_options[@]}" : \
                                '--elements=[Tensor element count]:ELEMENTS:_default' \
                                '--iterations=[Benchmark repetition count]:ITERATIONS:_default' \
                                '--out=[Output path for benchmark JSON]:OUT:_files' \
                                '-h[Print help]' '--help[Print help]' && ret=0
                            ;;
                        esac
                        ;;
                    esac
                    ;;
                (ane)
                    _arguments "${_arguments_options[@]}" : \
                        '-h[Print help]' '--help[Print help]' \
                        ":: :_ennx_ane_commands" \
                        "*::: :->ane_sub" && ret=0
                    case $state in
                    (ane_sub)
                        case $words[1] in
                        (probe)
                            _arguments "${_arguments_options[@]}" : \
                                '--rows=[Input sequence rows / batch tokens]:ROWS:_default' \
                                '--width=[Hidden feature dimension]:WIDTH:_default' \
                                '--outputs=[Output projection dimensionality]:OUTPUTS:_default' \
                                '--repeats=[Measured execution repetitions]:REPEATS:_default' \
                                '--warmups=[Unmeasured warmup rounds]:WARMUPS:_default' \
                                '--units=[Target compute units]:UNITS:(cpu gpu ane all)' \
                                '--weights=[Weight residency mode]:WEIGHTS:(fixed input)' \
                                '-h[Print help]' '--help[Print help]' && ret=0
                            ;;
                        esac
                        ;;
                    esac
                    ;;
                (ios)
                    _arguments "${_arguments_options[@]}" : \
                        '-h[Print help]' '--help[Print help]' \
                        ":: :_ennx_iphone_commands" \
                        "*::: :->iphone_sub" && ret=0
                    ;;
                esac
                ;;
            esac
            ;;
        (opt)
            _arguments "${_arguments_options[@]}" : \
                '-h[Print help]' '--help[Print help]' \
                ":: :_ennx_opt_commands" \
                "*::: :->opt_sub" && ret=0
            case $state in
            (opt_sub)
                case $words[1] in
                (run)
                    _arguments "${_arguments_options[@]}" : \
                        '--prepare[Prepare supported experiment inputs without starting optimization]' \
                        '-h[Print help]' '--help[Print help]' \
                        ':config:_files' && ret=0
                    ;;
                (protocol)
                    _arguments "${_arguments_options[@]}" : \
                        '--out=[Directory where experiment plans and run artifacts are stored]:OUT:_files -/' \
                        '--rounds=[Complete optimization rounds per paired repetition]:ROUNDS:_default' \
                        '--reps=[Paired repetitions executed per experimental arm]:REPS:_default' \
                        '--run[Execute every arm after writing protocol execution plan]' \
                        '--json[Emit status and plan manifests as machine-readable JSON]' \
                        '-h[Print help]' '--help[Print help]' \
                        '::protocol:_default' '::baseline:_files' && ret=0
                    ;;
                (eval)
                    _arguments "${_arguments_options[@]}" : \
                        '--output=[Override artifact path declared in configuration]:OUTPUT:_files' \
                        '-h[Print help]' '--help[Print help]' \
                        ':config:_files' && ret=0
                    ;;
                (catalog)
                    _arguments "${_arguments_options[@]}" : \
                        '--out=[Export enumerated configurations to directory]:OUT:_files -/' \
                        '--json[Emit catalog inventory or manifest as JSON]' \
                        '-h[Print help]' '--help[Print help]' \
                        '::config:_files' && ret=0
                    ;;
                esac
                ;;
            esac
            ;;
        (data)
            _arguments "${_arguments_options[@]}" : \
                '-h[Print help]' '--help[Print help]' \
                ":: :_ennx_data_commands" \
                "*::: :->data_sub" && ret=0
            case $state in
            (data_sub)
                case $words[1] in
                (corpus)
                    _arguments "${_arguments_options[@]}" : \
                        '-h[Print help]' '--help[Print help]' \
                        ":: :_ennx_corpus_commands" \
                        "*::: :->corpus_sub" && ret=0
                    ;;
                (inspect)
                    _arguments "${_arguments_options[@]}" : \
                        '--json[Emit metadata as machine-readable JSON]' \
                        '-h[Print help]' '--help[Print help]' \
                        ':file:_files' && ret=0
                    ;;
                esac
                ;;
            esac
            ;;
        (model)
            _arguments "${_arguments_options[@]}" : \
                '-h[Print help]' \
                '--help[Print help]' \
                ":: :_ennx_model_commands" \
                "*::: :->model_sub" && ret=0
            case $state in
            (model_sub)
                case $words[1] in
                (scale)
                    _arguments "${_arguments_options[@]}" : \
                        '--generated=[Newly generated suffix tokens]:GENERATED:_default' \
                        '--prompt=[Prompt context tokens in first wave]:PROMPT:_default' \
                        '--evaluated=[Total model positions evaluated]:EVALUATED:_default' \
                        '--round-ms=[Measured complete-round wall time in ms]:ROUND_MS:_default' \
                        '--target-ms=[Target wall-clock duration in ms]:TARGET_MS:_default' \
                        '--hierarchical[Evaluate scale under 2-stage hierarchical context compression]' \
                        '-h[Print help]' '--help[Print help]' && ret=0
                    ;;
                (context)
                    _arguments "${_arguments_options[@]}" : \
                        '--tokens=[Total context sequence length in tokens]:TOKENS:_default' \
                        '--queries=[Query positions evaluated in parallel]:QUERIES:_default' \
                        '--repeats=[Repetition passes for timing stability]:REPEATS:_default' \
                        '-h[Print help]' '--help[Print help]' && ret=0
                    ;;
                (inspect)
                    _arguments "${_arguments_options[@]}" : \
                        '--backend=[Target compute backend]:BACKEND:(metal cuda-oxide tvm-metal tvm-cuda)' \
                        '--function=[Program function entry point]:FUNCTION:(prefill decode draft verify score objective experiment-round)' \
                        '--strict[Exit non-zero if any operation is unassigned]' \
                        '--json[Emit results as machine-readable JSON]' \
                        '-h[Print help]' '--help[Print help]' && ret=0
                    ;;
                (context-loop)
                    _arguments "${_arguments_options[@]}" : \
                        '--generated=[Newly generated tokens per round]:GENERATED:_default' \
                        '--prompt=[Prompt tokens preceding generation]:PROMPT:_default' \
                        '--rounds=[Complete Bayesian optimization rounds]:ROUNDS:_default' \
                        '-h[Print help]' '--help[Print help]' \
                        ':config:_files' ':dataset:_files' && ret=0
                    ;;
                (decode)
                    _arguments "${_arguments_options[@]}" : \
                        '-h[Print help]' '--help[Print help]' \
                        ':tokenizer:_files' ':tokens:_files' ':output:_files' && ret=0
                    ;;
                (tirx-probe)
                    _arguments "${_arguments_options[@]}" : \
                        '--json[Emit probe findings as machine-readable JSON]' \
                        '-h[Print help]' '--help[Print help]' && ret=0
                    ;;
                esac
                ;;
            esac
            ;;
        (ptx)
            _arguments "${_arguments_options[@]}" : \
                '-h[Print help]' \
                '--help[Print help]' \
                ":: :_ennx_ptx_commands" \
                "*::: :->ptx_sub" && ret=0
            case $state in
            (ptx_sub)
                case $words[1] in
                (emit)
                    _arguments "${_arguments_options[@]}" : \
                        '--out=[Target file path for emitted PTX]:OUT:_files' \
                        '-h[Print help]' '--help[Print help]' \
                        ':kernel:(gemm rademacher vector)' && ret=0
                    ;;
                (modal)
                    _arguments "${_arguments_options[@]}" : \
                        '--elements=[Tensor element count]:ELEMENTS:_default' \
                        '--iterations=[Benchmark repetition count]:ITERATIONS:_default' \
                        '--out=[Output path for benchmark JSON]:OUT:_files' \
                        '-h[Print help]' '--help[Print help]' && ret=0
                    ;;
                esac
                ;;
            esac
            ;;
        (cuda)
            _arguments "${_arguments_options[@]}" : \
                '-h[Print help]' \
                '--help[Print help]' \
                ":: :_ennx_cuda_commands" \
                "*::: :->cuda_sub" && ret=0
            case $state in
            (cuda_sub)
                case $words[1] in
                (generate-t4)
                    _arguments "${_arguments_options[@]}" : \
                        '--context=[Context sequence capacity in tokens]:CONTEXT:_default' \
                        '--prompt=[Prompt prefix length in tokens]:PROMPT:_default' \
                        '--visits=[Recurrent layer visits per wave]:VISITS:_default' \
                        '--temperature=[Softmax sampling temperature]:TEMPERATURE:_default' \
                        '--seed=[PRNG seed for reproducible sampling]:SEED:_default' \
                        '--unroll=[Verifier wave unroll factor]:UNROLL:_default' \
                        '--chunk=[Activation chunk size (0=packed)]:CHUNK:_default' \
                        '--out=[Output path for serialized records]:OUT:_files' \
                        '-h[Print help]' '--help[Print help]' && ret=0
                    ;;
                (modal)
                    _arguments "${_arguments_options[@]}" : \
                        '--candidates=[Candidate points acquired per batch]:CANDIDATES:_default' \
                        '--history=[Historical observation capacity in GP]:HISTORY:_default' \
                        '--iterations=[Bayesian optimization iterations]:ITERATIONS:_default' \
                        '--out=[Output path for convergence curve JSON]:OUT:_files' \
                        '-h[Print help]' '--help[Print help]' && ret=0
                    ;;
                (context)
                    _arguments "${_arguments_options[@]}" : \
                        '-h[Print help]' '--help[Print help]' \
                        '*:args:_default' && ret=0
                    ;;
                (model|generate|diffusion|sanitize)
                    _arguments "${_arguments_options[@]}" : \
                        '-h[Print help]' '--help[Print help]' \
                        '*:args:_default' && ret=0
                    ;;
                esac
                ;;
            esac
            ;;
        (ane)
            _arguments "${_arguments_options[@]}" : \
                '-h[Print help]' \
                '--help[Print help]' \
                ":: :_ennx_ane_commands" \
                "*::: :->ane_sub" && ret=0
            case $state in
            (ane_sub)
                case $words[1] in
                (probe)
                    _arguments "${_arguments_options[@]}" : \
                        '--rows=[Input sequence rows / batch tokens]:ROWS:_default' \
                        '--width=[Hidden feature dimension]:WIDTH:_default' \
                        '--outputs=[Output projection dimensionality]:OUTPUTS:_default' \
                        '--repeats=[Measured execution repetitions]:REPEATS:_default' \
                        '--warmups=[Unmeasured warmup rounds]:WARMUPS:_default' \
                        '--units=[Target compute units]:UNITS:(cpu gpu ane all)' \
                        '--weights=[Weight residency mode]:WEIGHTS:(fixed input)' \
                        '-h[Print help]' '--help[Print help]' && ret=0
                    ;;
                esac
                ;;
            esac
            ;;
        (iphone)
            _arguments "${_arguments_options[@]}" : \
                '-h[Print help]' \
                '--help[Print help]' \
                ":: :_ennx_iphone_commands" \
                "*::: :->iphone_sub" && ret=0
            case $state in
            (iphone_sub)
                case $words[1] in
                (build)
                    _arguments "${_arguments_options[@]}" : \
                        '--team=[Apple development team ID]:TEAM:_default' \
                        '-h[Print help]' '--help[Print help]' && ret=0
                    ;;
                (deploy)
                    _arguments "${_arguments_options[@]}" : \
                        '--team=[Apple development team ID]:TEAM:_default' \
                        '-h[Print help]' '--help[Print help]' \
                        ':device:_default' && ret=0
                    ;;
                (web)
                    _arguments "${_arguments_options[@]}" : \
                        '--run=[Completed ENNX run to evaluate]:RUN:_files' \
                        '--stage=[Candidate stage inside run]:STAGE:_default' \
                        '--port=[Local HTTP server port]:PORT:_default' \
                        '--coordinates=[Coordinate elements sampled]:COORDINATES:_default' \
                        '--repeats=[Measured execution repetitions]:REPEATS:_default' \
                        '--workload=[Evaluation workload kind]:WORKLOAD:(proposal readout)' \
                        '-h[Print help]' '--help[Print help]' && ret=0
                    ;;
                (launch)
                    _arguments "${_arguments_options[@]}" : \
                        '-h[Print help]' '--help[Print help]' \
                        ':device:_default' && ret=0
                    ;;
                (probe)
                    _arguments "${_arguments_options[@]}" : \
                        '--coordinates=[Coordinate elements sampled]:COORDINATES:_default' \
                        '--repeats=[Measured execution repetitions]:REPEATS:_default' \
                        '-h[Print help]' '--help[Print help]' \
                        ':address:_default' && ret=0
                    ;;
                (ane)
                    _arguments "${_arguments_options[@]}" : \
                        '--rows=[Input sequence rows / batch tokens]:ROWS:_default' \
                        '--width=[Hidden feature dimension]:WIDTH:_default' \
                        '--outputs=[Output projection dimensionality]:OUTPUTS:_default' \
                        '--repeats=[Measured execution repetitions]:REPEATS:_default' \
                        '--warmups=[Unmeasured warmup rounds]:WARMUPS:_default' \
                        '--units=[Target compute units]:UNITS:(cpu gpu ane all)' \
                        '--weights=[Weight residency mode]:WEIGHTS:(fixed input)' \
                        '-h[Print help]' '--help[Print help]' \
                        ':address:_default' && ret=0
                    ;;
                esac
                ;;
            esac
            ;;
        (corpus)
            _arguments "${_arguments_options[@]}" : \
                '-h[Print help]' \
                '--help[Print help]' \
                ":: :_ennx_corpus_commands" \
                "*::: :->corpus_sub" && ret=0
            case $state in
            (corpus_sub)
                case $words[1] in
                (grammar)
                    _arguments "${_arguments_options[@]}" : \
                        '--out=[Output directory for corpus]:OUT:_files -/' \
                        '--seed=[Deterministic 64-bit PRNG seed]:SEED:_default' \
                        '--sequences=[Sampled paths to retain]:SEQUENCES:_default' \
                        '--max-tokens=[Maximum state transitions]:MAX_TOKENS:_default' \
                        '--sampler=[Path to prebuilt sfstrandgen]:SAMPLER:_files' \
                        '--printer=[Path to prebuilt fstprint]:PRINTER:_files' \
                        '-h[Print help]' '--help[Print help]' \
                        ':model:_files' && ret=0
                    ;;
                esac
                ;;
            esac
            ;;
        (tensor)
            _arguments "${_arguments_options[@]}" : \
                '-h[Print help]' \
                '--help[Print help]' \
                ":: :_ennx_tensor_commands" \
                "*::: :->tensor_sub" && ret=0
            case $state in
            (tensor_sub)
                case $words[1] in
                (inspect)
                    _arguments "${_arguments_options[@]}" : \
                        '--json[Emit metadata as machine-readable JSON]' \
                        '-h[Print help]' '--help[Print help]' \
                        ':file:_files' && ret=0
                    ;;
                esac
                ;;
            esac
            ;;
        (toolchain)
            _arguments "${_arguments_options[@]}" : \
                '-h[Print help]' \
                '--help[Print help]' \
                ":: :_ennx_toolchain_commands" \
                "*::: :->toolchain_sub" && ret=0
            case $state in
            (toolchain_sub)
                case $words[1] in
                (sync)
                    _arguments "${_arguments_options[@]}" : \
                        '-h[Print help]' '--help[Print help]' \
                        '::nightly:_default' && ret=0
                    ;;
                esac
                ;;
            esac
            ;;
        (util)
            _arguments "${_arguments_options[@]}" : \
                '-h[Print help]' \
                '--help[Print help]' \
                ":: :_ennx_util_commands" \
                "*::: :->util_sub" && ret=0
            case $state in
            (util_sub)
                case $words[1] in
                (completion)
                    _arguments "${_arguments_options[@]}" : \
                        '-h[Print help]' '--help[Print help]' \
                        ':shell:(zsh bash fish)' && ret=0
                    ;;
                esac
                ;;
            esac
            ;;
        (completion)
            _arguments "${_arguments_options[@]}" : \
                '-h[Print help]' '--help[Print help]' \
                ':shell:(zsh bash fish)' && ret=0
            ;;
        (tune)
            _arguments "${_arguments_options[@]}" : \
                '--prepare[Prepare supported experiment inputs without starting optimization]' \
                '-h[Print help]' '--help[Print help]' \
                ':config:_files' && ret=0
            ;;
        (eval)
            _arguments "${_arguments_options[@]}" : \
                '--output=[Override artifact path declared in configuration]:OUTPUT:_files' \
                '-h[Print help]' '--help[Print help]' \
                ':config:_files' && ret=0
            ;;
        (catalog)
            _arguments "${_arguments_options[@]}" : \
                '--out=[Export enumerated configurations to directory]:OUT:_files -/' \
                '--json[Emit catalog inventory or manifest as JSON]' \
                '-h[Print help]' '--help[Print help]' \
                '::config:_files' && ret=0
            ;;
        (experiment)
            _arguments "${_arguments_options[@]}" : \
                '--out=[Directory where experiment plans and run artifacts are stored]:OUT:_files -/' \
                '--rounds=[Complete optimization rounds per paired repetition]:ROUNDS:_default' \
                '--reps=[Paired repetitions executed per experimental arm]:REPS:_default' \
                '--run[Execute every arm after writing protocol execution plan]' \
                '--json[Emit status and plan manifests as machine-readable JSON]' \
                '-h[Print help]' '--help[Print help]' \
                '::protocol:_default' '::baseline:_files' && ret=0
            ;;
        (build)
            _arguments "${_arguments_options[@]}" : \
                '--tests[Run full Python verification instead of smoke tests]' \
                '--out=[Write wheels into this directory]:OUT:_files -/' \
                '-h[Print help]' '--help[Print help]' && ret=0
            ;;
        (test)
            _arguments "${_arguments_options[@]}" : \
                '--python[Run installed-wheel Python tests (requires ENNX_WHEEL_PATH)]' \
                '-h[Print help]' '--help[Print help]' && ret=0
            ;;
        (fmt)
            _arguments "${_arguments_options[@]}" : \
                '--check[Check formatting compliance without modifying files]' \
                '-h[Print help]' '--help[Print help]' && ret=0
            ;;
        (menu)
            _arguments "${_arguments_options[@]}" : \
                '-h[Print help]' '--help[Print help]' && ret=0
            ;;
        (dev)
            _arguments "${_arguments_options[@]}" : \
                '-h[Print help]' '--help[Print help]' && ret=0
            ;;
        esac
        ;;
    esac
    return ret
}

_ennx_commands() {
    local -a commands; commands=(
        'target:Hardware backends, synthesized kernels, and device execution gates'
        'opt:Bayesian optimization, surrogate tuning, evaluation suites, and protocols'
        'data:Corpora generation, token datasets, and tensor container inspection'
        'model:Inspect semantic model graphs, PISA attention caches, and compute scaling'
        'test:Run Rust unit tests, kernel tests, and parity suites'
        'build:Build the library, CLI, tests, and Python wheels'
        'fmt:Format the repository or check formatting without edits'
        'dev:Run the complete format, build, and test verification cycle'
        'menu:Open the interactive command navigator (navigate with Up/Down arrow keys)'
        'completion:Print a command-line completion script for zsh, bash, or fish'
        'util:Infrequently used utility commands such as shell completion scripts'
        'tune:Run a configured Bayesian optimization experiment [alias: opt run]'
        'eval:Evaluate optimizers on paired black-box tasks [alias: opt eval]'
        'experiment:Plan, run, and inspect a registered experiment protocol [alias: opt protocol]'
        'catalog:Inspect algorithm families or enumerate tune configurations [alias: opt catalog]'
        'cuda:Inspect or run the pinned CUDA-Oxide track [alias: target cuda]'
        'ptx:Emit synthesized PTX or execute its hardware gate on Modal [alias: target ptx]'
        'ane:Measure Apple Neural Engine execution through Core ML [alias: target ane]'
        'iphone:Build, deploy, and measure the iPhone Metal worker [alias: target ios]'
        'corpus:Generate reproducible corpora from stochastic grammars [alias: data corpus]'
        'tensor:Inspect framework-neutral tensor containers [alias: data inspect]'
        'toolchain:Check or synchronize the managed Rust toolchain'
        'help:Print this message or the help of the given subcommand(s)'
    )
    _describe -t commands 'commands' commands "$@"
}

_ennx_target_commands() {
    local -a commands; commands=(
        'cuda:Run pinned CUDA-Oxide track (parity, prefill, generate-t4, modal)'
        'ptx:Emit synthesized PTX or execute hardware gate on Modal'
        'ane:Measure Apple Neural Engine execution through Core ML'
        'ios:Build, deploy, and benchmark the iPhone Metal worker'
    )
    _describe -t commands 'target subcommands' commands "$@"
}

_ennx_opt_commands() {
    local -a commands; commands=(
        'run:Execute a configured Bayesian optimization experiment'
        'protocol:Plan, run, and inspect a registered experiment protocol'
        'eval:Evaluate optimizers on paired black-box benchmarking tasks'
        'catalog:Inspect algorithm families or enumerate tune configurations'
    )
    _describe -t commands 'opt subcommands' commands "$@"
}

_ennx_data_commands() {
    local -a commands; commands=(
        'corpus:Sample a normalized stochastic FST into an immutable text corpus'
        'inspect:Validate and describe Safetensors container without loading payload'
    )
    _describe -t commands 'data subcommands' commands "$@"
}

_ennx_model_commands() {
    local -a commands; commands=(
        'scale:Calculate the model work and throughput required by a generation loop'
        'context:Measure cached PISA attention over up to one million context tokens'
        'inspect:Show the operations assigned to a backend and the operations still missing'
        'context-loop:Measure complete BO rounds with the stated number of newly generated tokens'
        'decode:Decode generated vocabulary IDs with the corpus byte-level BPE tokenizer'
        'tirx-probe:Probe for an optional Apache TVM compiler/runtime installation'
    )
    _describe -t commands 'model subcommands' commands "$@"
}

_ennx_ptx_commands() {
    local -a commands; commands=(
        'emit:Write one synthesized PTX module to disk or stdout'
        'modal:Execute synthesized kernels and their parity gates on a Modal T4'
    )
    _describe -t commands 'ptx subcommands' commands "$@"
}

_ennx_cuda_commands() {
    local -a commands; commands=(
        'inspect:Inspect CUDA-Oxide kernel lowering coverage and symbol tables without GPU'
        'build:Compile CUDA-Oxide device modules and synthesized PTX for pinned T4'
        'parity:Run mathematical and bitwise parity checks against host CPU reference'
        'resident:Run end-to-end resident search parity checks on the CUDA device'
        'prefill:Measure device latency and bandwidth during resident prefill slice'
        'model-check:Verify numerical stability and contractive convergence of 5-layer mHC'
        'context:Benchmark bounded gather queries over persistent million-token PISA cache'
        'model:Execute looped mHC model checkpoints on input tokens and persist outputs'
        'generate:Generate tokens using device-side speculative accepted-prefix repair'
        'generate-t4:Measure real accepted-prefix generation on a Modal T4'
        'diffusion:Generate tokens using non-autoregressive block diffusion sampling'
        'diffusion-check:Verify learned mask inputs, corruption schedules, and denoising'
        'sanitize:Check native CUDA actions under compute-sanitizer for memory/race bugs'
        'bench:Measure packed Rademacher perturbation bandwidth and memory bus throughput'
        'modal:Measure the real resident BF16 BO curve on a Modal T4'
    )
    _describe -t commands 'cuda subcommands' commands "$@"
}

_ennx_ane_commands() {
    local -a commands; commands=(
        'probe:Measure a shape-faithful FP16 readout and report Core ML placement'
    )
    _describe -t commands 'ane subcommands' commands "$@"
}

_ennx_iphone_commands() {
    local -a commands; commands=(
        'devices:Query and list physical Apple iOS devices recognized by Xcode'
        'build:Compile the native iOS Metal compute worker bundle using Xcode'
        'deploy:Build, sign, install, and execute the worker app on a physical iPhone'
        'web:Run a zero-installation WebGPU compute probe in Mobile Safari'
        'launch:Launch an already-installed ENNX compute worker on the specified device'
        'probe:Run and validate a resident Rademacher proposal on the iPhone GPU'
        'ane:Measure an FP16 readout projection on the iPhone Neural Engine via Core ML'
    )
    _describe -t commands 'iphone subcommands' commands "$@"
}

_ennx_corpus_commands() {
    local -a commands; commands=(
        'grammar:Sample a normalized stochastic FST into an immutable text corpus'
    )
    _describe -t commands 'corpus subcommands' commands "$@"
}

_ennx_tensor_commands() {
    local -a commands; commands=(
        'inspect:Validate and describe Safetensors container without loading payload'
    )
    _describe -t commands 'tensor subcommands' commands "$@"
}

_ennx_toolchain_commands() {
    local -a commands; commands=(
        'check:Check every managed toolchain declaration across Buck2, Cargo, and CI'
        'sync:Synchronize toolchain declarations, optionally upgrading Rust nightly'
    )
    _describe -t commands 'toolchain subcommands' commands "$@"
}

_ennx_util_commands() {
    local -a commands; commands=(
        'completion:Print a command-line-completion script (compatible with jj util completion)'
    )
    _describe -t commands 'util subcommands' commands "$@"
}
"#;

const BASH_COMPLETION: &str = r#"_ennx_completions() {
    local cur prev words cword
    _init_completion || return

    local top_commands="target opt data model tune eval experiment cuda ptx ane iphone corpus tensor catalog test build fmt dev toolchain menu completion util help"
    if [[ $cword -eq 1 ]]; then
        COMPREPLY=($(compgen -W "$top_commands" -- "$cur"))
        return
    fi

    case "${words[1]}" in
        target)
            if [[ $cword -eq 2 ]]; then
                COMPREPLY=($(compgen -W "cuda ptx ane ios" -- "$cur"))
            fi
            ;;
        opt)
            if [[ $cword -eq 2 ]]; then
                COMPREPLY=($(compgen -W "run protocol eval catalog" -- "$cur"))
            fi
            ;;
        data)
            if [[ $cword -eq 2 ]]; then
                COMPREPLY=($(compgen -W "corpus inspect" -- "$cur"))
            fi
            ;;
        model)
            if [[ $cword -eq 2 ]]; then
                COMPREPLY=($(compgen -W "scale context inspect context-loop decode tirx-probe" -- "$cur"))
            else
                case "${words[2]}" in
                    scale)
                        COMPREPLY=($(compgen -W "--generated --prompt --evaluated --round-ms --target-ms --hierarchical -h --help" -- "$cur"))
                        ;;
                    context)
                        COMPREPLY=($(compgen -W "--tokens --queries --repeats -h --help" -- "$cur"))
                        ;;
                    inspect)
                        COMPREPLY=($(compgen -W "--backend --function --strict --json -h --help" -- "$cur"))
                        ;;
                    context-loop)
                        COMPREPLY=($(compgen -W "--generated --prompt --rounds -h --help" -- "$cur"))
                        ;;
                esac
            fi
            ;;
        cuda)
            if [[ $cword -eq 2 ]]; then
                COMPREPLY=($(compgen -W "inspect build parity resident prefill model-check context model generate generate-t4 diffusion diffusion-check sanitize bench modal" -- "$cur"))
            else
                case "${words[2]}" in
                    generate-t4)
                        COMPREPLY=($(compgen -W "--context --prompt --visits --temperature --seed --unroll --chunk --out -h --help" -- "$cur"))
                        ;;
                    modal)
                        COMPREPLY=($(compgen -W "--candidates --history --iterations --out -h --help" -- "$cur"))
                        ;;
                esac
            fi
            ;;
        ptx)
            if [[ $cword -eq 2 ]]; then
                COMPREPLY=($(compgen -W "emit modal" -- "$cur"))
            else
                case "${words[2]}" in
                    emit)
                        COMPREPLY=($(compgen -W "gemm rademacher vector --out -h --help" -- "$cur"))
                        ;;
                    modal)
                        COMPREPLY=($(compgen -W "--elements --iterations --out -h --help" -- "$cur"))
                        ;;
                esac
            fi
            ;;
        ane)
            if [[ $cword -eq 2 ]]; then
                COMPREPLY=($(compgen -W "probe" -- "$cur"))
            else
                COMPREPLY=($(compgen -W "--rows --width --outputs --repeats --warmups --units --weights -h --help" -- "$cur"))
            fi
            ;;
        iphone)
            if [[ $cword -eq 2 ]]; then
                COMPREPLY=($(compgen -W "devices build deploy web launch probe ane" -- "$cur"))
            fi
            ;;
        corpus)
            if [[ $cword -eq 2 ]]; then
                COMPREPLY=($(compgen -W "grammar" -- "$cur"))
            else
                COMPREPLY=($(compgen -W "--out --seed --sequences --max-tokens --sampler --printer -h --help" -- "$cur"))
            fi
            ;;
        tensor)
            if [[ $cword -eq 2 ]]; then
                COMPREPLY=($(compgen -W "inspect" -- "$cur"))
            else
                COMPREPLY=($(compgen -W "--json -h --help" -- "$cur"))
            fi
            ;;
        toolchain)
            if [[ $cword -eq 2 ]]; then
                COMPREPLY=($(compgen -W "check sync" -- "$cur"))
            fi
            ;;
        util)
            if [[ $cword -eq 2 ]]; then
                COMPREPLY=($(compgen -W "completion" -- "$cur"))
            else
                COMPREPLY=($(compgen -W "zsh bash fish -h --help" -- "$cur"))
            fi
            ;;
        completion)
            COMPREPLY=($(compgen -W "zsh bash fish -h --help" -- "$cur"))
            ;;
        tune)
            COMPREPLY=($(compgen -W "--prepare -h --help" -- "$cur"))
            ;;
        eval)
            COMPREPLY=($(compgen -W "--output -h --help" -- "$cur"))
            ;;
        catalog)
            COMPREPLY=($(compgen -W "--out --json -h --help" -- "$cur"))
            ;;
        experiment)
            COMPREPLY=($(compgen -W "--out --rounds --reps --run --json -h --help" -- "$cur"))
            ;;
        build)
            COMPREPLY=($(compgen -W "--tests --out -h --help" -- "$cur"))
            ;;
        test)
            COMPREPLY=($(compgen -W "--python -h --help" -- "$cur"))
            ;;
        fmt)
            COMPREPLY=($(compgen -W "--check -h --help" -- "$cur"))
            ;;
    esac
}
complete -F _ennx_completions ennx ./ennx
"#;

const FISH_COMPLETION: &str = r#"complete -c ennx -n "__fish_use_subcommand" -a "target" -d "Hardware backends, synthesized kernels, and device execution gates"
complete -c ennx -n "__fish_use_subcommand" -a "opt" -d "Bayesian optimization, surrogate tuning, evaluation suites, and protocols"
complete -c ennx -n "__fish_use_subcommand" -a "data" -d "Corpora generation, token datasets, and tensor container inspection"
complete -c ennx -n "__fish_use_subcommand" -a "model" -d "Inspect semantic model graphs, PISA attention caches, and compute scaling"
complete -c ennx -n "__fish_use_subcommand" -a "tune" -d "Run a configured Bayesian optimization experiment"
complete -c ennx -n "__fish_use_subcommand" -a "eval" -d "Evaluate optimizers on paired black-box tasks"
complete -c ennx -n "__fish_use_subcommand" -a "experiment" -d "Plan, run, and inspect a registered experiment protocol"
complete -c ennx -n "__fish_use_subcommand" -a "cuda" -d "Inspect or run the pinned CUDA-Oxide track without invoking Cargo directly"
complete -c ennx -n "__fish_use_subcommand" -a "ptx" -d "Emit synthesized PTX or execute its hardware gate on Modal"
complete -c ennx -n "__fish_use_subcommand" -a "ane" -d "Measure Apple Neural Engine execution through Core ML"
complete -c ennx -n "__fish_use_subcommand" -a "iphone" -d "Build, deploy, and measure the iPhone Metal worker"
complete -c ennx -n "__fish_use_subcommand" -a "corpus" -d "Generate reproducible corpora from stochastic grammars"
complete -c ennx -n "__fish_use_subcommand" -a "tensor" -d "Inspect framework-neutral tensor containers"
complete -c ennx -n "__fish_use_subcommand" -a "catalog" -d "Inspect algorithm families or enumerate categorical tune configurations"
complete -c ennx -n "__fish_use_subcommand" -a "test" -d "Run Rust unit tests, kernel tests, and parity suites"
complete -c ennx -n "__fish_use_subcommand" -a "build" -d "Build the library, CLI, tests, and Python wheels"
complete -c ennx -n "__fish_use_subcommand" -a "fmt" -d "Format the repository or check formatting without edits"
complete -c ennx -n "__fish_use_subcommand" -a "dev" -d "Run the complete format, build, and test verification cycle"
complete -c ennx -n "__fish_use_subcommand" -a "toolchain" -d "Check or synchronize the managed Rust toolchain"
complete -c ennx -n "__fish_use_subcommand" -a "menu" -d "Open the interactive command navigator (navigate with Up/Down arrow keys)"
complete -c ennx -n "__fish_use_subcommand" -a "completion" -d "Print a shell completion script for zsh, bash, or fish"
complete -c ennx -n "__fish_use_subcommand" -a "util" -d "Infrequently used utility commands such as shell completion scripts"
"#;
