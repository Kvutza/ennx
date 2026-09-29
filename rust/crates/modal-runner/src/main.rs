use modal_rs::{
    AppOptions, Image, ModalClient, SandboxExecOptions, SandboxOptions, VolumeFromNameOptions,
};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use tempfile::NamedTempFile;

const CUDA_IMAGE: &str = "im-tZRy6QPZXIJyPrv4zZqPOM";
const CUDA_REV: &str = "6abfaa091e29a6275c1943895bfbc97efa306e98";
const RUST_NIGHTLY: &str = "nightly-2026-08-28";

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

enum Action {
    Bo {
        output: PathBuf,
        candidates: u32,
        history: u32,
        iterations: u32,
    },
    Image,
    Generate {
        output: PathBuf,
        context: u32,
        prompt: u32,
        visits: u32,
        temperature: f32,
        seed: u64,
        unroll: u32,
        chunk: u32,
    },
    Ptx {
        output: PathBuf,
        elements: u32,
        iterations: u32,
    },
    Wheel {
        output: PathBuf,
        mjx: bool,
    },
}

fn parse_bo<I: Iterator<Item = std::ffi::OsString>>(mut args: I) -> Result<Action> {
    let mut output = PathBuf::from("results/cuda/bf16-bo-curve-t4.json");
    let mut candidates = 4_u32;
    let mut history = 32_u32;
    let mut iterations = 50_u32;
    while let Some(flag) = args.next() {
        let value = args.next().ok_or_else(|| {
            io::Error::other(format!("missing value for {}", flag.to_string_lossy()))
        })?;
        match flag.to_str() {
            Some("--output") => output = value.into(),
            Some("--candidates") => candidates = value.to_string_lossy().parse()?,
            Some("--history") => history = value.to_string_lossy().parse()?,
            Some("--iterations") => iterations = value.to_string_lossy().parse()?,
            _ => {
                return Err(io::Error::other(
                    "usage: ennx-modal bo --output PATH --candidates N --history N --iterations N",
                )
                .into());
            }
        }
    }
    if candidates == 0 || history == 0 || iterations == 0 {
        return Err(io::Error::other("BO probe dimensions must be positive").into());
    }
    Ok(Action::Bo {
        output,
        candidates,
        history,
        iterations,
    })
}

fn parse_generate<I: Iterator<Item = std::ffi::OsString>>(mut args: I) -> Result<Action> {
    let mut output = PathBuf::from("results/cuda/generation-t4");
    let mut context = 4_096_u32;
    let mut prompt = 128_u32;
    let mut visits = 2_u32;
    let mut temperature = 0.8_f32;
    let mut seed = 17_u64;
    let mut unroll = 2_u32;
    let mut chunk = 0_u32;
    while let Some(flag) = args.next() {
        let value = args.next().ok_or_else(|| {
            io::Error::other(format!("missing value for {}", flag.to_string_lossy()))
        })?;
        match flag.to_str() {
            Some("--output") => output = value.into(),
            Some("--context") => context = value.to_string_lossy().parse()?,
            Some("--prompt") => prompt = value.to_string_lossy().parse()?,
            Some("--visits") => visits = value.to_string_lossy().parse()?,
            Some("--temperature") => temperature = value.to_string_lossy().parse()?,
            Some("--seed") => seed = value.to_string_lossy().parse()?,
            Some("--unroll") => unroll = value.to_string_lossy().parse()?,
            Some("--chunk") => chunk = value.to_string_lossy().parse()?,
            _ => return Err(io::Error::other("usage: ennx-modal generate --output DIR --context N --prompt N --visits N --temperature F --seed N --unroll N").into()),
        }
    }
    if context == 0
        || prompt == 0
        || prompt >= context
        || visits == 0
        || !temperature.is_finite()
        || temperature < 0.0
        || unroll == 0
        || unroll > 32
        || (chunk != 0
            && (!(64..=4096).contains(&chunk) || !chunk.is_power_of_two() || context % chunk != 0))
    {
        return Err(
            io::Error::other("invalid generation dimensions or sampling parameters").into(),
        );
    }
    Ok(Action::Generate {
        output,
        context,
        prompt,
        visits,
        temperature,
        seed,
        unroll,
        chunk,
    })
}

fn parse_wheel<I: Iterator<Item = std::ffi::OsString>>(args: I) -> Result<Action> {
    let mut output = PathBuf::from("/tmp/ennx-cuda-wheel.whl");
    let mut output_set = false;
    let mut mjx = false;
    for arg in args {
        if arg == "--mjx" {
            mjx = true;
        } else if !output_set {
            output = arg.into();
            output_set = true;
        } else {
            return Err(io::Error::other("usage: ennx-modal wheel [OUTPUT] [--mjx]").into());
        }
    }
    Ok(Action::Wheel { output, mjx })
}

fn parse_ptx<I: Iterator<Item = std::ffi::OsString>>(mut args: I) -> Result<Action> {
    let mut output = PathBuf::from("results/ptx/t4-kernels.json");
    let mut elements = 1_048_576_u32;
    let mut iterations = 100_u32;
    while let Some(flag) = args.next() {
        let value = args.next().ok_or_else(|| {
            io::Error::other(format!("missing value for {}", flag.to_string_lossy()))
        })?;
        match flag.to_str() {
            Some("--output") => output = value.into(),
            Some("--elements") => elements = value.to_string_lossy().parse()?,
            Some("--iterations") => iterations = value.to_string_lossy().parse()?,
            _ => {
                return Err(io::Error::other(
                    "usage: ennx-modal ptx [--output PATH] [--elements N] [--iterations N]",
                )
                .into());
            }
        }
    }
    if elements == 0 || elements % 4 != 0 || iterations == 0 {
        return Err(io::Error::other(
            "PTX probe requires positive iterations and elements divisible by four",
        )
        .into());
    }
    Ok(Action::Ptx {
        output,
        elements,
        iterations,
    })
}

fn parse_args() -> Result<Action> {
    let mut args = std::env::args_os().skip(1);
    let action = args.next().ok_or_else(|| {
        io::Error::other("usage: ennx-modal <bo | generate | image | ptx | wheel>")
    })?;
    match action.to_str() {
        Some("bo") => parse_bo(args),
        Some("image") if args.next().is_none() => Ok(Action::Image),
        Some("generate") => parse_generate(args),
        Some("wheel") => parse_wheel(args),
        Some("ptx") => parse_ptx(args),
        _ => {
            Err(io::Error::other("usage: ennx-modal <bo | generate | image | ptx | wheel>").into())
        }
    }
}

fn tool_image() -> Result<Image> {
    let llvm = "/usr/lib/llvm-21";
    let cuda = "/opt/cuda-oxide";
    let path = format!(
        "/root/.cargo/bin:{llvm}/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    );
    let image = Image::from_registry("nvidia/cuda:13.0.1-devel-ubuntu24.04")
        .run_commands([
            "DEBIAN_FRONTEND=noninteractive apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ca-certificates curl g++ gcc gnupg libc6-dev make pkg-config python3 xz-utils && rm -rf /var/lib/apt/lists/*".to_string(),
            "curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key | gpg --dearmor -o /usr/share/keyrings/apt.llvm.org.gpg && echo 'deb [signed-by=/usr/share/keyrings/apt.llvm.org.gpg] https://apt.llvm.org/noble/ llvm-toolchain-noble-21 main' > /etc/apt/sources.list.d/llvm-21.list && apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends clang-21 libclang-common-21-dev libclang-21-dev lld-21 llvm-21 llvm-21-dev && rm -rf /var/lib/apt/lists/*".to_string(),
            format!(
                "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain {RUST_NIGHTLY} && /root/.cargo/bin/rustup component add --toolchain {RUST_NIGHTLY} rust-src rustc-dev llvm-tools rust-analyzer clippy rustfmt"
            ),
            format!(
                "mkdir -p {cuda} && curl -fsSL https://github.com/NVlabs/cuda-oxide/archive/{CUDA_REV}.tar.gz | tar -xz --strip-components=1 -C {cuda} && PATH={path} LLVM_CONFIG_PATH=/usr/bin/llvm-config-21 LIBCLANG_PATH={llvm}/lib /root/.cargo/bin/cargo +{RUST_NIGHTLY} install --path {cuda}/crates/cargo-oxide --locked --force && cd {cuda} && PATH={path} LLVM_CONFIG_PATH=/usr/bin/llvm-config-21 LIBCLANG_PATH={llvm}/lib CUDA_OXIDE_LLC=/usr/bin/llc-21 CUDA_HOME=/usr/local/cuda RUSTUP_TOOLCHAIN={RUST_NIGHTLY} /root/.cargo/bin/cargo oxide setup"
            ),
        ])?
        .env([
            ("CUDA_HOME", "/usr/local/cuda"),
            ("CUDA_PATH", "/usr/local/cuda"),
            ("CUDA_TOOLKIT_PATH", "/usr/local/cuda"),
            ("CUDA_OXIDE_LLC", "/usr/bin/llc-21"),
            (
                "CUDA_OXIDE_BACKEND",
                "/opt/cuda-oxide/crates/rustc-codegen-cuda/target/x86_64-unknown-linux-gnu/debug/librustc_codegen_cuda.so",
            ),
            ("ENNX_FAISS_UNAVAILABLE", "1"),
            ("LIBCLANG_PATH", "/usr/lib/llvm-21/lib"),
            ("LLVM_CONFIG_PATH", "/usr/bin/llvm-config-21"),
            ("PATH", path.as_str()),
            ("PYTHONPATH", "/opt/ennx/ops"),
            ("RUSTUP_TOOLCHAIN", RUST_NIGHTLY),
        ])?;
    Ok(image)
}

async fn stream_exec(
    sandbox: &modal_rs::Sandbox,
    client: &mut ModalClient,
    options: SandboxExecOptions,
) -> Result<modal_rs::SandboxExecExitStatus> {
    let mut stream = sandbox.exec_stream(client, options).await?;
    let mut stdout = stream.take_stdout().ok_or("stdout stream is missing")?;
    let mut stderr = stream.take_stderr().ok_or("stderr stream is missing")?;
    let wait = stream.take_wait().ok_or("exec wait handle is missing")?;
    let out_task = tokio::spawn(async move {
        while let Some(chunk) = stdout.recv().await {
            print!("{}", String::from_utf8_lossy(&chunk?));
        }
        Ok::<(), modal_rs::Error>(())
    });
    let err_task = tokio::spawn(async move {
        while let Some(chunk) = stderr.recv().await {
            eprint!("{}", String::from_utf8_lossy(&chunk?));
        }
        Ok::<(), modal_rs::Error>(())
    });
    let status = wait.await??;
    let (out, err) = tokio::join!(out_task, err_task);
    out??;
    err??;
    Ok(status)
}

fn repo_root() -> Result<PathBuf> {
    std::env::var_os("BUILD_WORKSPACE_DIRECTORY")
        .map(PathBuf::from)
        .map_or_else(
            || std::env::current_dir().map_err(Into::into),
            |root| Ok(root),
        )
}

fn source_tar() -> Result<NamedTempFile> {
    let root = repo_root()?;
    let files = Command::new("jj")
        .arg("-R")
        .arg(&root)
        .args(["file", "list", "-r", "@", "-T", "path ++ \"\\x00\""])
        .output()?;
    if !files.status.success() {
        return Err(io::Error::other(format!(
            "jj file list failed: {}",
            String::from_utf8_lossy(&files.stderr).trim()
        ))
        .into());
    }
    let present = files
        .stdout
        .split(|byte| *byte == 0)
        .filter(|path| {
            !path.is_empty() && root.join(String::from_utf8_lossy(path).as_ref()).exists()
        })
        .flat_map(|path| path.iter().copied().chain(std::iter::once(0)))
        .collect::<Vec<_>>();
    let target = NamedTempFile::new()?;
    let mut child = Command::new("tar")
        .args(["--null", "-T", "-", "-czf"])
        .arg(target.path())
        .current_dir(root)
        .env("COPYFILE_DISABLE", "1")
        .stdin(Stdio::piped())
        .spawn()?;
    child
        .stdin
        .take()
        .ok_or_else(|| io::Error::other("tar stdin is unavailable"))?
        .write_all(&present)?;
    let status = child.wait()?;
    if !status.success() {
        return Err(io::Error::other(format!("source archive failed: {status}")).into());
    }
    Ok(target)
}

fn wheel_cmd(mjx: bool) -> String {
    let wheel = wheel_path();
    let mut command = format!(
        "set -euo pipefail; cd /opt/ennx; \
         rm -rf /tmp/ennx-wheel /tmp/ennx-wheel-env; \
         mkdir -p /tmp/ennx-wheel; \
         python -m venv /tmp/ennx-wheel-env; \
         /tmp/ennx-wheel-env/bin/python -m pip install --quiet 'jax[cuda12]'; \
         export PATH=/tmp/ennx-wheel-env/bin:$PATH; \
         export XLA_PYTHON_CLIENT_PREALLOCATE=false; \
         unset LD_LIBRARY_PATH; \
         PARITY=$(./buck2w --isolation-dir cuda build //:cuda-parity \
         --target-platforms //:linux-x86_64-platform --local-only --num-threads 4 \
         --show-full-simple-output); \
         cat \"$PARITY\"; \
         ./buck2w --isolation-dir cuda build //:cuda-wheel \
         --target-platforms //:linux-x86_64-platform --local-only --num-threads 4 \
         --out {wheel}; \
         /tmp/ennx-wheel-env/bin/python -m pip install --quiet {wheel}; \
         /tmp/ennx-wheel-env/bin/python ops/cusprbench.py; \
         /tmp/ennx-wheel-env/bin/python ops/bf16_bench.py",
    );
    if mjx {
        command.push_str(
            "; /tmp/ennx-wheel-env/bin/python -m pip install --quiet \
             mujoco==3.6.0 mujoco-mjx==3.6.0; \
             /tmp/ennx-wheel-env/bin/python ops/mjx_batch.py",
        );
    }
    command
}

fn wheel_path() -> String {
    format!(
        "/tmp/ennx-wheel/ennx-{}+cuda75-cp312-cp312-manylinux_2_28_x86_64.whl",
        env!("CARGO_PKG_VERSION")
    )
}

fn ptx_cmd(elements: u32, iterations: u32) -> String {
    format!(
        "set -euo pipefail; cd /opt/ennx; \
         CARGO_HOME=/cache/cargo CARGO_TARGET_DIR=/cache/target ./tools/cuda-run build; \
         /cache/target/release/ennx-cuda ptx-probe {elements} {iterations} > /tmp/ptx-synth-result.json; \
         cat /tmp/ptx-synth-result.json"
    )
}

fn bo_cmd(candidates: u32, history: u32, iterations: u32) -> String {
    format!(
        "set -euo pipefail; cd /opt/ennx; \
         CARGO_HOME=/cache/cargo CARGO_TARGET_DIR=/cache/target ./tools/cuda-run build; \
         /cache/target/release/ennx-cuda parity; \
         /cache/target/release/ennx-cuda bf16-curve {candidates} {history} {iterations} \
         > /tmp/ennx-bo-result.json; \
         cat /tmp/ennx-bo-result.json"
    )
}

fn generation_cmd(
    context: u32,
    prompt: u32,
    visits: u32,
    temperature: f32,
    seed: u64,
    unroll: u32,
    chunk: u32,
    synth: Option<u32>,
    prompt_file: bool,
) -> String {
    let mut env = synth.map_or(String::new(), |k| format!("export ENNX_SYNTH_GEMM={k}; "));
    if std::env::var_os("ENNX_CUDA_PROFILE").is_some() {
        env.push_str("export ENNX_CUDA_PROFILE=1; ");
    }
    if let Ok(rows) = std::env::var("ENNX_CUDA_REPAIR_CHUNK") {
        // Parse before constructing the remote shell command.
        if let Ok(rows) = rows.parse::<u32>() {
            env.push_str(&format!("export ENNX_CUDA_REPAIR_CHUNK={rows}; "));
        }
    }
    let prompt_arg = if prompt_file {
        " /tmp/ennx-prompt.json"
    } else {
        ""
    };
    format!(
        "set -euo pipefail; cd /opt/ennx; \
         {env}\
         CARGO_HOME=/cache/cargo CARGO_TARGET_DIR=/cache/target ./tools/cuda-run build; \
         test -f /ckpt/looped-571e8175/checkpoint.safetensors; \
         /cache/target/release/ennx-cuda model-check; \
         /cache/target/release/ennx-cuda diffusion-check; \
         /cache/target/release/ennx-cuda context 4096 128 1; \
         rm -rf /tmp/ennx-generation; \
         /cache/target/release/ennx-cuda generation-bench \
         /ckpt/looped-571e8175/checkpoint.safetensors /tmp/ennx-generation \
         {context} {prompt} {visits} {temperature} {seed} {unroll} {chunk}{prompt_arg}"
    )
}

async fn upload_source(
    sandbox: &modal_rs::Sandbox,
    client: &mut ModalClient,
    archive: &NamedTempFile,
) -> Result<()> {
    let source = std::fs::read(archive.path())?;
    let upload = sandbox
        .exec(
            client,
            SandboxExecOptions::new(vec![
                "bash",
                "-lc",
                "rm -rf /opt/ennx && mkdir -p /opt/ennx && tar -xzf - -C /opt/ennx",
            ])
            .with_stdin(source)
            .with_timeout(120),
        )
        .await?;
    if upload.exit_status.is_success() {
        Ok(())
    } else {
        Err(io::Error::other(format!("source upload failed: {:?}", upload.exit_status)).into())
    }
}

async fn ptx_run(
    client: &mut ModalClient,
    app: &modal_rs::App,
    output: &Path,
    elements: u32,
    iterations: u32,
) -> Result<()> {
    let archive = source_tar()?;
    let cache = client
        .volumes()
        .from_name(
            "ennx-cuda-build",
            "main",
            VolumeFromNameOptions::create_if_missing(),
        )
        .await?;
    let mut image = match std::env::var("ENNX_MODAL_IMAGE") {
        Ok(image_id) => Image::from_id(image_id),
        Err(_) => {
            let mut image = tool_image()?;
            image.build(client, app).await?;
            image
        }
    };
    let sandbox = client
        .sandboxes()
        .create(
            app,
            &image,
            SandboxOptions::default()
                .with_gpu_type("T4")
                .with_milli_cpu(4_000)
                .with_memory_mb(8_192)
                .with_timeout(1_200)
                .with_volume(&cache, "/cache"),
        )
        .await?;
    let run: Result<(Vec<u8>, Vec<u8>, bool)> = async {
        upload_source(&sandbox, client, &archive).await?;
        let status = stream_exec(
            &sandbox,
            client,
            SandboxExecOptions::new(vec!["bash", "-lc", &ptx_cmd(elements, iterations)])
                .with_timeout(1_100),
        )
        .await?;
        let gate_passed = status.is_success();
        let artifact = sandbox
            .exec(
                client,
                SandboxExecOptions::new(vec!["cat", "/tmp/ptx-synth-result.json"]).with_timeout(30),
            )
            .await?;
        if !artifact.exit_status.is_success() {
            return Err(io::Error::other("PTX artifact download failed").into());
        }
        let result = artifact
            .stdout
            .ok_or_else(|| io::Error::other("PTX artifact was empty"))?;
        let sass = sandbox
            .exec(
                client,
                SandboxExecOptions::new(vec!["cat", "/tmp/ptx-synth.sass"]).with_timeout(30),
            )
            .await?;
        if !sass.exit_status.is_success() {
            return Err(io::Error::other("PTX SASS download failed").into());
        }
        let sass = sass
            .stdout
            .ok_or_else(|| io::Error::other("PTX SASS artifact was empty"))?;
        Ok((result, sass, gate_passed))
    }
    .await;
    let stopped = sandbox.terminate(client).await;
    let (artifact, sass, gate_passed) = run?;
    stopped?;
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(output, artifact)?;
    let sass_output = output.with_extension("sass");
    std::fs::write(&sass_output, sass)?;
    println!(
        "MODAL_PTX ok={} output={} sass={}",
        gate_passed,
        output.display(),
        sass_output.display()
    );
    if gate_passed {
        Ok(())
    } else {
        Err(io::Error::other("PTX hardware gate rejected the measured artifact").into())
    }
}

async fn bo_run(
    client: &mut ModalClient,
    app: &modal_rs::App,
    output: &Path,
    candidates: u32,
    history: u32,
    iterations: u32,
) -> Result<()> {
    let archive = source_tar()?;
    let cache = client
        .volumes()
        .from_name(
            "ennx-cuda-build",
            "main",
            VolumeFromNameOptions::create_if_missing(),
        )
        .await?;
    let mut image = match std::env::var("ENNX_MODAL_IMAGE") {
        Ok(image_id) => Image::from_id(image_id),
        Err(_) => {
            let mut image = tool_image()?;
            image.build(client, app).await?;
            image
        }
    };
    let sandbox = client
        .sandboxes()
        .create(
            app,
            &image,
            SandboxOptions::default()
                .with_gpu_type("T4")
                .with_milli_cpu(4_000)
                .with_memory_mb(8_192)
                .with_timeout(1_200)
                .with_volume(&cache, "/cache"),
        )
        .await?;
    let run: Result<Vec<u8>> = async {
        upload_source(&sandbox, client, &archive).await?;
        let status = stream_exec(
            &sandbox,
            client,
            SandboxExecOptions::new(vec![
                "bash",
                "-lc",
                &bo_cmd(candidates, history, iterations),
            ])
            .with_timeout(1_100),
        )
        .await?;
        if !status.is_success() {
            return Err(io::Error::other(format!("CUDA BO benchmark failed: {status:?}")).into());
        }
        let artifact = sandbox
            .exec(
                client,
                SandboxExecOptions::new(vec!["cat", "/tmp/ennx-bo-result.json"]).with_timeout(30),
            )
            .await?;
        if !artifact.exit_status.is_success() {
            return Err(io::Error::other("CUDA BO artifact download failed").into());
        }
        artifact
            .stdout
            .ok_or_else(|| io::Error::other("CUDA BO artifact was empty").into())
    }
    .await;
    let stopped = sandbox.terminate(client).await;
    let artifact = run?;
    stopped?;
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(output, artifact)?;
    println!("MODAL_BO ok=true output={}", output.display());
    Ok(())
}

fn env_config(chunk: u32, output: &Path) -> Result<(Option<Vec<u8>>, Option<u32>)> {
    let prompt_data = std::env::var_os("ENNX_CUDA_PROMPT")
        .map(std::fs::read)
        .transpose()?;
    if let Ok(value) = std::env::var("ENNX_CUDA_REPAIR_CHUNK") {
        let rows = value.parse::<u32>()?;
        if !(64..=4096).contains(&rows) || !rows.is_power_of_two() || chunk == 0 {
            return Err("ENNX_CUDA_REPAIR_CHUNK requires streamed execution and a power of two from 64 through 4096".into());
        }
    }
    let synth = std::env::var("ENNX_SYNTH_GEMM")
        .ok()
        .map(|s| s.parse::<u32>())
        .transpose()?;
    if synth.is_some_and(|k| ![8, 16, 32, 64].contains(&k)) {
        return Err("ENNX_SYNTH_GEMM must be 8, 16, 32, or 64".into());
    }
    if output.exists() {
        return Err("generation output already exists; choose a fresh result directory".into());
    }
    Ok((prompt_data, synth))
}

async fn make_sandbox(client: &mut ModalClient, app: &modal_rs::App) -> Result<modal_rs::Sandbox> {
    let volume = client
        .volumes()
        .from_name("ennx-ckpt", "main", VolumeFromNameOptions::default())
        .await?
        .read_only();
    let cache = client
        .volumes()
        .from_name(
            "ennx-cuda-build",
            "main",
            VolumeFromNameOptions::create_if_missing(),
        )
        .await?;
    let mut image = match std::env::var("ENNX_MODAL_IMAGE") {
        Ok(image_id) => Image::from_id(image_id),
        Err(_) => {
            let mut image = tool_image()?;
            image.build(client, app).await?;
            image
        }
    };
    let sandbox = client
        .sandboxes()
        .create(
            app,
            &image,
            SandboxOptions::default()
                .with_gpu_type("T4")
                .with_milli_cpu(4_000)
                .with_memory_mb(16_384)
                .with_timeout(3_600)
                .with_volume(&volume, "/ckpt")
                .with_volume(&cache, "/cache"),
        )
        .await?;
    Ok(sandbox)
}

async fn pull_artifacts(
    sandbox: &modal_rs::Sandbox,
    client: &mut ModalClient,
    output: &Path,
    synth: Option<u32>,
) -> Result<(Vec<u8>, Vec<u8>)> {
    let result = sandbox
        .exec(
            client,
            SandboxExecOptions::new(vec!["cat", "/tmp/ennx-generation/result.json"])
                .with_timeout(30),
        )
        .await?;
    let tokens = sandbox
        .exec(
            client,
            SandboxExecOptions::new(vec!["cat", "/tmp/ennx-generation/tokens.json"])
                .with_timeout(30),
        )
        .await?;
    if !result.exit_status.is_success() || !tokens.exit_status.is_success() {
        return Err(io::Error::other("CUDA generation artifact download failed").into());
    }
    if synth.is_some() {
        let gemm = sandbox
            .exec(
                client,
                SandboxExecOptions::new(vec!["cat", "/tmp/ennx-generation/gemm.json"])
                    .with_timeout(30),
            )
            .await?;
        if !gemm.exit_status.is_success() {
            return Err("GEMM measurement download failed".into());
        }
        std::fs::create_dir_all(output)?;
        std::fs::write(
            output.join("gemm.json"),
            gemm.stdout.ok_or("GEMM measurement was empty")?,
        )?;
    }
    Ok((
        result
            .stdout
            .ok_or_else(|| io::Error::other("generation result was empty"))?,
        tokens
            .stdout
            .ok_or_else(|| io::Error::other("generation tokens were empty"))?,
    ))
}

#[allow(clippy::too_many_arguments)]
async fn generation_run(
    client: &mut ModalClient,
    app: &modal_rs::App,
    output: &Path,
    context: u32,
    prompt: u32,
    visits: u32,
    temperature: f32,
    seed: u64,
    unroll: u32,
    chunk: u32,
) -> Result<()> {
    let (prompt_data, synth) = env_config(chunk, output)?;
    let archive = source_tar()?;
    let sandbox = make_sandbox(client, app).await?;
    let run: Result<(Vec<u8>, Vec<u8>)> = async {
        upload_source(&sandbox, client, &archive).await?;
        if let Some(data) = &prompt_data {
            let upload = sandbox
                .exec(
                    client,
                    SandboxExecOptions::new(vec!["bash", "-lc", "cat > /tmp/ennx-prompt.json"])
                        .with_stdin(data.clone())
                        .with_timeout(30),
                )
                .await?;
            if !upload.exit_status.is_success() {
                return Err("prompt upload failed".into());
            }
        }
        let status = stream_exec(
            &sandbox,
            client,
            SandboxExecOptions::new(vec![
                "bash",
                "-lc",
                &generation_cmd(
                    context,
                    prompt,
                    visits,
                    temperature,
                    seed,
                    unroll,
                    chunk,
                    synth,
                    prompt_data.is_some(),
                ),
            ])
            .with_timeout(3_500),
        )
        .await?;
        if !status.is_success() {
            return Err(io::Error::other(format!("CUDA generation failed: {status:?}")).into());
        }
        pull_artifacts(&sandbox, client, output, synth).await
    }
    .await;
    let stopped = sandbox.terminate(client).await;
    let (result, tokens) = run?;
    stopped?;
    std::fs::create_dir_all(output)?;
    std::fs::write(output.join("result.json"), &result)?;
    std::fs::write(output.join("tokens.json"), tokens)?;
    print!("{}", String::from_utf8_lossy(&result));
    println!("GENERATION_T4 ok=true output={}", output.display());
    Ok(())
}

async fn wheel_run(
    client: &mut ModalClient,
    app: &modal_rs::App,
    output: &Path,
    mjx: bool,
) -> Result<()> {
    let archive = source_tar()?;
    let image_id = std::env::var("ENNX_MODAL_IMAGE").unwrap_or_else(|_| CUDA_IMAGE.to_string());
    let image = Image::from_id(image_id);
    let sandbox = client
        .sandboxes()
        .create(
            app,
            &image,
            SandboxOptions::default()
                .with_gpu_type("T4")
                .with_milli_cpu(8_000)
                .with_memory_mb(16_384)
                .with_timeout(3_600),
        )
        .await?;
    let run: Result<()> = async {
        let source = std::fs::read(archive.path())?;
        let upload = sandbox
            .exec(
                client,
                SandboxExecOptions::new(vec![
                    "bash",
                    "-lc",
                    "rm -rf /opt/ennx && mkdir -p /opt/ennx && tar -xzf - -C /opt/ennx",
                ])
                .with_stdin(source)
                .with_timeout(120),
            )
            .await?;
        if !upload.exit_status.is_success() {
            return Err(io::Error::other(format!(
                "source upload failed: {:?}",
                upload.exit_status
            ))
            .into());
        }
        let status = stream_exec(
            &sandbox,
            client,
            SandboxExecOptions::new(vec!["bash", "-lc", &wheel_cmd(mjx)]).with_timeout(3_600),
        )
        .await?;
        if !status.is_success() {
            return Err(io::Error::other(format!("wheel gate failed: {status:?}")).into());
        }
        let artifact = sandbox
            .exec(
                client,
                SandboxExecOptions::new(vec!["cat", &wheel_path()]).with_timeout(120),
            )
            .await?;
        if !artifact.exit_status.is_success() {
            return Err(io::Error::other(format!(
                "wheel download failed: {:?}",
                artifact.exit_status
            ))
            .into());
        }
        let wheel = artifact
            .stdout
            .ok_or_else(|| io::Error::other("wheel download returned no bytes"))?;
        std::fs::write(output, wheel)?;
        Ok(())
    }
    .await;
    let stopped = sandbox.terminate(client).await;
    run?;
    stopped?;
    println!("MODAL_WHEEL ok=true output={}", output.display());
    Ok(())
}

async fn image_run(client: &mut ModalClient, app: &modal_rs::App) -> Result<()> {
    let mut image = tool_image()?;
    image.build(client, app).await?;
    println!("MODAL_IMAGE ok=true image={}", image.id()?);
    Ok(())
}

#[tokio::main]
async fn main() {
    if let Err(error) = run().await {
        // Debug formatting of RPC errors can include authentication metadata.
        eprintln!("ennx-modal: {error}");
        std::process::exit(1);
    }
}

async fn run() -> Result<()> {
    let _output = modal_rs::enable_output();
    let action = parse_args()?;
    let mut client = ModalClient::connect().await?;
    let app = client
        .get_or_create_app("ennx-rust-modal", "main", AppOptions::create_if_missing())
        .await?;
    match action {
        Action::Bo {
            output,
            candidates,
            history,
            iterations,
        } => bo_run(&mut client, &app, &output, candidates, history, iterations).await,
        Action::Image => image_run(&mut client, &app).await,
        Action::Generate {
            output,
            context,
            prompt,
            visits,
            temperature,
            seed,
            unroll,
            chunk,
        } => {
            generation_run(
                &mut client,
                &app,
                &output,
                context,
                prompt,
                visits,
                temperature,
                seed,
                unroll,
                chunk,
            )
            .await
        }
        Action::Ptx {
            output,
            elements,
            iterations,
        } => ptx_run(&mut client, &app, &output, elements, iterations).await,
        Action::Wheel { output, mjx } => wheel_run(&mut client, &app, &output, mjx).await,
    }
}
