use modal_rs::{AppOptions, Image, ModalClient, SandboxExecOptions, SandboxOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use tempfile::NamedTempFile;

const CUDA_IMAGE: &str = "im-tZRy6QPZXIJyPrv4zZqPOM";
const CUDA_REV: &str = "1f4d813719012d384f2db12b88efc9314c8bf50c";
const RUST_NIGHTLY: &str = "nightly-2026-04-03";

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

enum Action {
    Image,
    Wheel { output: PathBuf, mjx: bool },
}

fn parse_args() -> Result<Action> {
    let mut args = std::env::args_os().skip(1);
    let action = args
        .next()
        .ok_or_else(|| io::Error::other("usage: ennx-modal <image | wheel [OUTPUT] [--mjx]>"))?;
    match action.to_str() {
        Some("image") if args.next().is_none() => Ok(Action::Image),
        Some("wheel") => {
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
        _ => Err(io::Error::other("usage: ennx-modal <image | wheel [OUTPUT] [--mjx]>").into()),
    }
}

fn tool_image() -> Result<Image> {
    let llvm = "/opt/tool/.pixi/envs/default";
    let cuda = "/opt/cuda-oxide";
    let path = format!(
        "{llvm}/bin:/root/.cargo/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    );
    let image = Image::from_registry("nvidia/cuda:12.8.1-devel-rockylinux8")
        .run_commands([
            "dnf install -y ca-certificates curl gcc gcc-c++ libffi-devel make patch pkgconf-pkg-config xz && dnf clean all".to_string(),
            "curl -fsSL https://github.com/jj-vcs/jj/releases/download/v0.41.0/jj-v0.41.0-x86_64-unknown-linux-musl.tar.gz | tar -xz -C /usr/local/bin jj".to_string(),
            format!(
                "curl -fsSL https://pixi.sh/install.sh | bash && /root/.pixi/bin/pixi init /opt/tool --channel conda-forge && /root/.pixi/bin/pixi add --manifest-path /opt/tool/pixi.toml 'llvmdev=21.*' 'clang=21.*' 'libclang=21.*' 'lld=21.*' 'python=3.12.*' pip && ln -sf {llvm}/bin/python /usr/local/bin/python && ln -sf {llvm}/bin/python /usr/local/bin/python3"
            ),
            format!(
                "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain {RUST_NIGHTLY} && /root/.cargo/bin/rustup component add --toolchain {RUST_NIGHTLY} rust-src rustc-dev llvm-tools rust-analyzer clippy rustfmt"
            ),
            format!(
                "jj git clone https://github.com/NVlabs/cuda-oxide.git {cuda} && jj -R {cuda} edit {CUDA_REV} && PATH={path} LLVM_CONFIG_PATH={llvm}/bin/llvm-config LIBCLANG_PATH={llvm}/lib /root/.cargo/bin/cargo +{RUST_NIGHTLY} install --path {cuda}/crates/cargo-oxide --locked && cd {cuda} && PATH={path} LLVM_CONFIG_PATH={llvm}/bin/llvm-config LIBCLANG_PATH={llvm}/lib CUDA_OXIDE_LLC={llvm}/bin/llc RUSTUP_TOOLCHAIN={RUST_NIGHTLY} /root/.cargo/bin/cargo oxide setup"
            ),
        ])?
        .env([
            ("CUDA_HOME", "/usr/local/cuda"),
            ("CUDA_PATH", "/usr/local/cuda"),
            ("CUDA_TOOLKIT_PATH", "/usr/local/cuda"),
            ("CUDA_OXIDE_LLC", "/opt/tool/.pixi/envs/default/bin/llc"),
            (
                "CUDA_OXIDE_BACKEND",
                "/opt/cuda-oxide/crates/rustc-codegen-cuda/target/x86_64-unknown-linux-gnu/debug/librustc_codegen_cuda.so",
            ),
            ("ENNX_FAISS_UNAVAILABLE", "1"),
            ("LIBCLANG_PATH", "/opt/tool/.pixi/envs/default/lib"),
            (
                "LLVM_CONFIG_PATH",
                "/opt/tool/.pixi/envs/default/bin/llvm-config",
            ),
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
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .map(Path::to_path_buf)
        .ok_or_else(|| io::Error::other("modal runner must remain under rust/crates").into())
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
         /tmp/ennx-wheel-env/bin/python ops/cuda_sparse_bench.py; \
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
async fn main() -> Result<()> {
    let _output = modal_rs::enable_output();
    let action = parse_args()?;
    let mut client = ModalClient::connect().await?;
    let app = client
        .get_or_create_app("ennx-rust-modal", "main", AppOptions::create_if_missing())
        .await?;
    match action {
        Action::Image => image_run(&mut client, &app).await,
        Action::Wheel { output, mjx } => wheel_run(&mut client, &app, &output, mjx).await,
    }
}
