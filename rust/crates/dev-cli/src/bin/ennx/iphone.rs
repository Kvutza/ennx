use clap::{Subcommand, ValueEnum};
use ennx_wire::json::{Value, json, pretty_string};
use std::env;
use std::fs;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

const BUNDLE: &str = "dev.ennx.compute";
const PORT: u16 = 47_123;

#[derive(Debug, PartialEq, Eq, Subcommand)]
pub(crate) enum Action {
    /// Query and list physical Apple iOS devices recognized by CoreDevice and devicectl.
    ///
    /// Interrogates Xcode device managers to enumerate connected iOS targets, their unique
    /// device identifiers (UDID), hardware architecture (Apple Silicon A-series / M-series),
    /// paired connection status, and CoreDevice network hostnames.
    Devices,
    /// Compile the native iOS Metal compute worker bundle using Xcode toolchains.
    ///
    /// Synthesizes Metal shading language kernels, packages the application container,
    /// and configures memory allocation allowances for resident long-context evaluation.
    /// Builds without codesigning by default unless an Apple Team ID is provided.
    Build {
        /// Apple development team ID used for cryptographic codesigning and provisioning profiles.
        #[arg(long, help_heading = "Signing & Credentials")]
        team: Option<String>,
    },
    /// Build, cryptographically sign, install, and execute the worker app on a physical iPhone.
    ///
    /// Automates the full deployment pipeline: compiles the application bundle, signs with the
    /// designated team certificate, installs over USB or Wi-Fi via CoreDevice, and launches the resident daemon.
    Deploy {
        /// Target device name, hardware identifier, UDID, or CoreDevice hostname.
        device: String,
        /// Apple development team ID used for automatic codesigning and provisioning profiles.
        #[arg(long, help_heading = "Signing & Credentials")]
        team: Option<String>,
    },
    /// Run a zero-installation WebGPU compute probe in Mobile Safari over Tailscale.
    ///
    /// Launches a local HTTP daemon that serves a WebGPU compute shader harness, streaming
    /// candidate weight proposals or dense readout projections directly to mobile browser shaders.
    Web {
        /// Completed ENNX run whose generated candidate should be evaluated.
        #[arg(long, help_heading = "Workload & Artifact Parameters")]
        run: Option<PathBuf>,
        /// Candidate stage inside --run.
        #[arg(
            long,
            default_value = "round-0001",
            help_heading = "Workload & Artifact Parameters"
        )]
        stage: String,
        /// Local port used by the temporary HTTP server serving the WebGPU probe.
        #[arg(
            long,
            default_value_t = 47_124,
            help_heading = "Network & Execution Parameters"
        )]
        port: u16,
        /// Total coordinate elements sampled during the WebGPU proposal kernel.
        #[arg(
            long,
            default_value_t = 16_777_216,
            help_heading = "Network & Execution Parameters"
        )]
        coordinates: u64,
        /// Number of measured execution repetitions to record.
        #[arg(
            long,
            default_value_t = 7,
            help_heading = "Network & Execution Parameters"
        )]
        repeats: u32,
        /// Evaluation workload kind: candidate proposal or dense attention readout.
        #[arg(long, value_enum, default_value_t = WebWorkload::Proposal, help_heading = "Workload & Artifact Parameters")]
        workload: WebWorkload,
    },
    /// Launch an already-installed ENNX compute worker on the specified physical device.
    ///
    /// Spawns the background compute process via CoreDevice and establishes TCP telemetry
    /// connectivity over port 47123 without reinstalling the binary bundle.
    Launch {
        /// Target device name, hardware identifier, UDID, or CoreDevice hostname.
        device: String,
    },
    /// Run and validate a resident Rademacher proposal on the iPhone GPU.
    ///
    /// Connects to the on-device worker daemon, dispatches bit-packed Rademacher perturbation
    /// kernels across the phone GPU, and verifies coordinate validity and execution bandwidth.
    Probe {
        /// Phone hostname or IP address; port 47123 is implied when omitted.
        address: String,
        /// Total coordinate elements sampled in the device proposal kernel.
        #[arg(
            long,
            default_value_t = 16_777_216,
            help_heading = "Workload Dimensions"
        )]
        coordinates: u64,
        /// Number of measured execution repetitions to record.
        #[arg(long, default_value_t = 7, help_heading = "Benchmarking Protocol")]
        repeats: u32,
    },
    /// Measure an FP16 readout projection on the iPhone Neural Engine via Core ML.
    ///
    /// Evaluates dense projection latency on the on-device Apple Neural Engine (ANE),
    /// measuring wall execution time and confirming whether hardware placement remained on ANE
    /// or experienced unassigned kernel fallbacks to the mobile CPU.
    Ane {
        /// Phone hostname or IP; port 47123 is implied when omitted.
        address: String,
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
        #[arg(long, value_enum, default_value_t = super::ane::Units::Ane, help_heading = "Hardware Execution & Placement")]
        units: super::ane::Units,
        /// Weight parameter residency mode: runtime input tensor or precompiled model constant.
        #[arg(long, value_enum, default_value_t = super::ane::Weights::Input, help_heading = "Hardware Execution & Placement")]
        weights: super::ane::Weights,
    },
}

pub(crate) fn run(root: &Path, action: Action) -> Result<(), String> {
    match action {
        Action::Devices => devices(root),
        Action::Build { team } => {
            let team = team.or_else(|| env::var("ENNX_APPLE_TEAM").ok());
            let app = build(root, team.as_deref())?;
            println!("{}", app.display());
            Ok(())
        }
        Action::Deploy { device, team } => {
            let team = team
                .or_else(|| env::var("ENNX_APPLE_TEAM").ok())
                .ok_or("deploy requires --team or ENNX_APPLE_TEAM")?;
            deploy(root, &device, &team)
        }
        Action::Web {
            run,
            stage,
            port,
            coordinates,
            repeats,
            workload,
        } => web(
            root,
            run.as_deref(),
            &stage,
            port,
            coordinates,
            repeats,
            workload,
        ),
        Action::Launch { device } => launch(root, &device),
        Action::Probe {
            address,
            coordinates,
            repeats,
        } => probe(&address, coordinates, repeats),
        Action::Ane {
            address,
            rows,
            width,
            outputs,
            repeats,
            warmups,
            units,
            weights,
        } => ane(
            &address, rows, width, outputs, repeats, warmups, units, weights,
        ),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(crate) enum WebWorkload {
    Proposal,
    Readout,
}

fn web(
    root: &Path,
    run: Option<&Path>,
    stage: &str,
    port: u16,
    coordinates: u64,
    repeats: u32,
    workload: WebWorkload,
) -> Result<(), String> {
    if coordinates == 0 || repeats == 0 || coordinates > u32::MAX.into() {
        return Err("coordinates and repeats must be positive; coordinates must fit u32".into());
    }
    let listener = TcpListener::bind(("127.0.0.1", port))
        .map_err(|error| format!("listen for Safari worker on port {port}: {error}"))?;
    let page = fs::read(root.join("apple/iphone/Web/index.html"))
        .map_err(|error| format!("read Safari worker page: {error}"))?;
    let url = tail_url(root, port)?;
    let (job, expected) = match run {
        Some(run) => objective_job(run, stage)?,
        None => (web_job(workload, coordinates, repeats)?, None),
    };
    println!("open {url} in iPhone Safari and keep it in the foreground");
    let mut sent = None;

    for stream in listener.incoming() {
        let mut stream = stream.map_err(|error| format!("accept Safari worker: {error}"))?;
        let request = request(&mut stream)?;
        match (request.method.as_str(), request.path.as_str()) {
            ("GET", "/") => reply(&mut stream, 200, "text/html; charset=utf-8", &page)?,
            ("GET", "/job") => {
                reply(&mut stream, 200, "application/json", &job)?;
                sent = Some(Instant::now());
                eprintln!("ENNX_IPHONE_JOB bytes={}", job.len());
            }
            ("GET", "/favicon.ico") => reply(&mut stream, 204, "image/x-icon", &[])?,
            ("POST", "/result") => {
                let value: Value = ennx_wire::json::from_slice(&request.body)
                    .map_err(|error| format!("invalid Safari result: {error}"))?;
                reply(&mut stream, 204, "application/json", &[])?;
                if let Some(sent) = sent {
                    eprintln!(
                        "ENNX_IPHONE_RETURN elapsed_ms={:.3}",
                        sent.elapsed().as_secs_f64() * 1e3
                    );
                }
                if let Some(expected) = expected {
                    let observed = value["objective"]
                        .as_f64()
                        .ok_or("Safari candidate result has no objective")?;
                    if (observed - expected).abs() > 1e-12 {
                        return Err(format!(
                            "Safari candidate objective {observed} differs from Rust {expected}"
                        ));
                    }
                }
                println!(
                    "{}",
                    pretty_string(&value).map_err(|error| error.to_string())?
                );
                return value["ok"]
                    .as_bool()
                    .unwrap_or(false)
                    .then_some(())
                    .ok_or_else(|| {
                        value["error"]
                            .as_str()
                            .unwrap_or("Safari worker failed")
                            .to_owned()
                    });
            }
            _ => reply(&mut stream, 404, "text/plain", b"not found")?,
        }
    }
    Err("Safari worker listener stopped".into())
}

fn web_job(workload: WebWorkload, coordinates: u64, repeats: u32) -> Result<Vec<u8>, String> {
    let value = match workload {
        WebWorkload::Proposal => json!({
            "protocol": "ennx.iphone.web.v2",
            "command": "proposal-probe",
            "coordinates": coordinates,
            "repeats": repeats,
        }),
        WebWorkload::Readout => json!({
            "protocol": "ennx.iphone.web.v2",
            "command": "readout-probe",
            "rows": 128,
            "input-width": 512,
            "output-width": 8192,
            "repeats": repeats,
        }),
    };
    ennx_wire::json::to_vec(&value).map_err(|error| error.to_string())
}

fn objective_job(run: &Path, stage: &str) -> Result<(Vec<u8>, Option<f64>), String> {
    let generation: Value = ennx_wire::json::from_reader(
        fs::File::open(run.join("generation.json"))
            .map_err(|error| format!("open generation record: {error}"))?,
    )
    .map_err(|error| format!("read generation record: {error}"))?;
    let evaluation: Value = ennx_wire::json::from_reader(
        fs::File::open(run.join(stage).join("evaluation.json"))
            .map_err(|error| format!("open candidate evaluation: {error}"))?,
    )
    .map_err(|error| format!("read candidate evaluation: {error}"))?;
    let candidate = token_array(&evaluation["rollouts"][0]["tokens"], "candidate tokens")?;
    let reference = token_array(
        &generation["config"]["tasks"][0]["expected"],
        "reference tokens",
    )?;
    let matches = candidate
        .iter()
        .zip(&reference)
        .filter(|(left, right)| left == right)
        .count();
    let objective = matches as f64 / candidate.len().max(reference.len()) as f64;
    let job = ennx_wire::json::to_vec(&json!({
        "protocol": "ennx.iphone.web.v2",
        "command": "token-accuracy",
        "candidate": candidate,
        "reference": reference,
        "expected-objective": objective,
    }))
    .map_err(|error| error.to_string())?;
    Ok((job, Some(objective)))
}

fn token_array(value: &Value, label: &str) -> Result<Vec<u32>, String> {
    value
        .as_seq()
        .ok_or_else(|| format!("{label} are missing"))?
        .iter()
        .map(|token| {
            token
                .as_u64()
                .and_then(|token| u32::try_from(token).ok())
                .ok_or_else(|| format!("{label} contain a non-u32 value"))
        })
        .collect()
}

struct Request {
    method: String,
    path: String,
    body: Vec<u8>,
}

fn request(stream: &mut TcpStream) -> Result<Request, String> {
    stream
        .set_read_timeout(Some(Duration::from_secs(10)))
        .map_err(|error| format!("set Safari request timeout: {error}"))?;
    let mut bytes = Vec::new();
    let mut chunk = [0_u8; 4096];
    let header_end = loop {
        let read = stream
            .read(&mut chunk)
            .map_err(|error| format!("read Safari request: {error}"))?;
        if read == 0 {
            return Err("Safari closed an incomplete request".into());
        }
        bytes.extend_from_slice(&chunk[..read]);
        if bytes.len() > 1_048_576 {
            return Err("Safari request exceeded 1 MiB".into());
        }
        if let Some(index) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
            break index + 4;
        }
    };
    let header = std::str::from_utf8(&bytes[..header_end])
        .map_err(|error| format!("Safari request header is not UTF-8: {error}"))?;
    let mut lines = header.lines();
    let mut start = lines
        .next()
        .ok_or("Safari request has no request line")?
        .split_whitespace();
    let method = start
        .next()
        .ok_or("Safari request has no method")?
        .to_owned();
    let path = start
        .next()
        .ok_or("Safari request has no path")?
        .split('?')
        .next()
        .unwrap_or("/")
        .to_owned();
    let length = lines
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().ok())
                .flatten()
        })
        .unwrap_or(0);
    while bytes.len() < header_end + length {
        let read = stream
            .read(&mut chunk)
            .map_err(|error| format!("read Safari request body: {error}"))?;
        if read == 0 {
            return Err("Safari closed an incomplete request body".into());
        }
        bytes.extend_from_slice(&chunk[..read]);
    }
    Ok(Request {
        method,
        path,
        body: bytes[header_end..header_end + length].to_vec(),
    })
}

fn reply(stream: &mut TcpStream, status: u16, kind: &str, body: &[u8]) -> Result<(), String> {
    let phrase = match status {
        200 => "OK",
        204 => "No Content",
        _ => "Not Found",
    };
    let header = format!(
        "HTTP/1.1 {status} {phrase}\r\nContent-Type: {kind}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream
        .write_all(header.as_bytes())
        .and_then(|()| stream.write_all(body))
        .map_err(|error| format!("reply to Safari worker: {error}"))
}

fn tail_url(root: &Path, port: u16) -> Result<String, String> {
    checked(
        Command::new("tailscale").current_dir(root).args([
            "serve",
            "--bg",
            "--yes",
            &format!("--https={port}"),
            &format!("http://127.0.0.1:{port}"),
        ]),
        "publish Safari worker inside the tailnet",
    )?;
    let output = Command::new("tailscale")
        .current_dir(root)
        .args(["status", "--json"])
        .output()
        .map_err(|error| format!("read Tailscale status: {error}"))?;
    let value: Value = ennx_wire::json::from_slice(&output.stdout)
        .map_err(|error| format!("invalid Tailscale status: {error}"))?;
    let name = value["Self"]["DNSName"]
        .as_str()
        .ok_or("Tailscale status has no DNS name")?
        .trim_end_matches('.');
    Ok(format!("https://{name}:{port}/"))
}

fn devices(root: &Path) -> Result<(), String> {
    checked(
        Command::new("xcrun")
            .current_dir(root)
            .args(["devicectl", "list", "devices"]),
        "list iPhone devices",
    )
}

fn build(root: &Path, team: Option<&str>) -> Result<PathBuf, String> {
    let cache = root.join(".cache/ennx/iphone");
    let project = root.join("apple/iphone/EnnxCompute.xcodeproj");
    let mut command = Command::new("xcodebuild");
    command.current_dir(root).args([
        "-project",
        project.to_str().ok_or("iPhone project path is not UTF-8")?,
        "-target",
        "EnnxCompute",
        "-configuration",
        "Release",
        "-sdk",
        "iphoneos",
    ]);
    command
        .arg(format!(
            "SYMROOT={}",
            cache.join("Build/Products").display()
        ))
        .arg(format!(
            "OBJROOT={}",
            cache.join("Build/Intermediates.noindex").display()
        ));
    if let Some(team) = team {
        command
            .arg("-allowProvisioningUpdates")
            .arg(format!("DEVELOPMENT_TEAM={team}"));
    } else {
        command.arg("CODE_SIGNING_ALLOWED=NO");
    }
    command.arg("build");
    checked(&mut command, "build iPhone worker")?;
    let app = cache.join("Build/Products/Release-iphoneos/EnnxCompute.app");
    app.is_dir()
        .then_some(app)
        .ok_or_else(|| "Xcode completed without producing EnnxCompute.app".into())
}

fn deploy(root: &Path, device: &str, team: &str) -> Result<(), String> {
    let app = build(root, Some(team))?;
    checked(
        Command::new("xcrun")
            .current_dir(root)
            .args(["devicectl", "device", "install", "app", "--device", device])
            .arg(&app),
        "install iPhone worker",
    )?;
    launch(root, device)?;
    println!("worker listens on TCP {PORT}; use the Hostname from `./ennx iphone devices`");
    Ok(())
}

fn launch(root: &Path, device: &str) -> Result<(), String> {
    checked(
        Command::new("xcrun").current_dir(root).args([
            "devicectl",
            "device",
            "process",
            "launch",
            "--device",
            device,
            "--terminate-existing",
            BUNDLE,
        ]),
        "launch iPhone worker",
    )
}

fn probe(address: &str, coordinates: u64, repeats: u32) -> Result<(), String> {
    if coordinates == 0 || repeats == 0 {
        return Err("coordinates and repeats must both be positive".into());
    }
    native(
        address,
        json!({
            "protocol": "ennx.iphone.v1",
            "command": "proposal-probe",
            "coordinates": coordinates,
            "repeats": repeats,
        }),
    )
}

fn ane(
    address: &str,
    rows: usize,
    width: usize,
    outputs: usize,
    repeats: usize,
    warmups: usize,
    units: super::ane::Units,
    weights: super::ane::Weights,
) -> Result<(), String> {
    if rows == 0 || width == 0 || outputs == 0 || repeats == 0 {
        return Err("rows, width, outputs, and repeats must be positive".into());
    }
    native(
        address,
        json!({
            "protocol": "ennx.iphone.v1",
            "command": "ane-readout",
            "rows": rows,
            "width": width,
            "outputs": outputs,
            "repeats": repeats,
            "warmups": warmups,
            "units": units.as_str(),
            "weights": weights.as_str(),
        }),
    )
}

fn native(address: &str, request: Value) -> Result<(), String> {
    let endpoint = if address.contains(':') {
        address.to_owned()
    } else {
        format!("{address}:{PORT}")
    };
    let mut stream = TcpStream::connect(&endpoint)
        .map_err(|error| format!("connect to iPhone worker at {endpoint}: {error}"))?;
    stream
        .set_read_timeout(Some(Duration::from_secs(120)))
        .map_err(|error| format!("set iPhone read timeout: {error}"))?;
    stream
        .set_write_timeout(Some(Duration::from_secs(5)))
        .map_err(|error| format!("set iPhone write timeout: {error}"))?;
    let request = ennx_wire::json::to_vec(&request).map_err(|error| error.to_string())?;
    stream
        .write_all(&request)
        .and_then(|()| stream.write_all(b"\n"))
        .map_err(|error| format!("send iPhone request: {error}"))?;
    let mut response = Vec::new();
    stream
        .read_to_end(&mut response)
        .map_err(|error| format!("read iPhone response: {error}"))?;
    let value: Value = ennx_wire::json::from_slice(&response)
        .map_err(|error| format!("invalid iPhone response: {error}"))?;
    if value["ok"].as_bool() != Some(true) {
        return Err(value["error"]
            .as_str()
            .unwrap_or("iPhone worker rejected the request")
            .into());
    }
    println!(
        "{}",
        pretty_string(&value).map_err(|error| error.to_string())?
    );
    Ok(())
}

fn checked(command: &mut Command, operation: &str) -> Result<(), String> {
    let status = command
        .status()
        .map_err(|error| format!("{operation}: {error}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("{operation} exited with {status}"))
    }
}
