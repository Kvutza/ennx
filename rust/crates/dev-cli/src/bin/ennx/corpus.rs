use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

use clap::Subcommand;
use ennx_wire::json::{Value, json};
use sha2::{Digest, Sha256};

use crate::build_system::BuildSystem;

const OPENGRM_REVISION: &str = "ce0d58aa1d6ae84151ca1531c695e4e9debed77f";
const OPENFST_REVISION: &str = "4a2f04b8ec3b445d153c653d49aa5d055811e90d";
const SAMPLER_TARGET: &str = "@opengrm//opengrm/sfst:sfstrandgen";
const PRINTER_TARGET: &str = "@com_google_openfst//openfst/bin:fstprint";

#[derive(Debug, PartialEq, Eq, Subcommand)]
pub(crate) enum Action {
    /// Sample a normalized stochastic FST into an immutable text corpus.
    ///
    /// Evaluates probabilistic language grammars with OpenGrm and OpenFst, generating
    /// bitwise-reproducible sequence tokens with exact provenance manifests and SHA-256 digests.
    Grammar {
        /// Canonical, normalized SFST model produced by OpenGrm.
        model: PathBuf,
        /// New output directory where generated sample files and the manifest will be stored.
        #[arg(long, help_heading = "Output Options")]
        out: PathBuf,
        /// Deterministic nonzero 64-bit PRNG seed for OpenGrm random path generation.
        #[arg(long, help_heading = "Sampling Parameters")]
        seed: u64,
        /// Total count of sampled paths / sequences to generate.
        #[arg(long, help_heading = "Sampling Parameters")]
        sequences: u32,
        /// Maximum state transitions permitted in a single sampled path before forced termination.
        #[arg(long, default_value_t = 4096, help_heading = "Sampling Parameters")]
        max_tokens: u32,
        /// Path to prebuilt sfstrandgen executable (defaults to managed hermetic toolchain).
        #[arg(
            long,
            requires = "printer",
            help_heading = "Hermetic Toolchain Binaries"
        )]
        sampler: Option<PathBuf>,
        /// Path to prebuilt fstprint executable (defaults to managed hermetic toolchain).
        #[arg(
            long,
            requires = "sampler",
            help_heading = "Hermetic Toolchain Binaries"
        )]
        printer: Option<PathBuf>,
    },
}

pub(crate) fn run(root: &Path, action: Action) -> Result<(), String> {
    match action {
        Action::Grammar {
            model,
            out,
            seed,
            sequences,
            max_tokens,
            sampler,
            printer,
        } => grammar(
            root,
            &model,
            &out,
            Request {
                seed,
                sequences,
                max_tokens,
            },
            sampler.zip(printer),
        ),
    }
}

#[derive(Clone, Copy)]
struct Request {
    seed: u64,
    sequences: u32,
    max_tokens: u32,
}

fn grammar(
    root: &Path,
    model: &Path,
    out: &Path,
    request: Request,
    tools: Option<(PathBuf, PathBuf)>,
) -> Result<(), String> {
    validate(model, out, request)?;
    let model = absolute(root, model);
    let out = absolute(root, out);
    let (sampler, printer, build_log) = match tools {
        Some((sampler, printer)) => (
            absolute(root, &sampler),
            absolute(root, &printer),
            Vec::new(),
        ),
        None => build_tools(root)?,
    };
    fs::create_dir(&out).map_err(|error| format!("create {}: {error}", out.display()))?;
    let result = generate(&model, &out, request, &sampler, &printer, &build_log);
    if result.is_err() {
        let _ = fs::remove_dir_all(&out);
    }
    result?;
    println!(
        "Generated {} grammar paths in {}",
        request.sequences,
        out.display()
    );
    Ok(())
}

fn validate(model: &Path, out: &Path, request: Request) -> Result<(), String> {
    if !model.is_file() {
        return Err(format!("grammar model does not exist: {}", model.display()));
    }
    if out.exists() {
        return Err(format!("grammar output already exists: {}", out.display()));
    }
    if request.seed == 0 || request.sequences == 0 || request.max_tokens == 0 {
        return Err("grammar seed, sequences, and max-tokens must be positive".into());
    }
    Ok(())
}

fn build_tools(root: &Path) -> Result<(PathBuf, PathBuf, Vec<u8>), String> {
    let sampler = BuildSystem::Bazel.build_artifact(root, SAMPLER_TARGET)?;
    let printer = BuildSystem::Bazel.build_artifact(root, PRINTER_TARGET)?;
    let mut log = sampler.log;
    log.extend_from_slice(&printer.log);
    Ok((sampler.path, printer.path, log))
}

fn generate(
    model: &Path,
    out: &Path,
    request: Request,
    sampler: &Path,
    printer: &Path,
    build_log: &[u8],
) -> Result<(), String> {
    fs::write(out.join("build.log"), build_log).map_err(|error| error.to_string())?;
    let model_sha256 = digest(model)?;
    let samples_path = out.join("samples.txt");
    let samples = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&samples_path)
        .map_err(|error| error.to_string())?;
    let mut samples = BufWriter::new(samples);
    let scratch = out.join("work");
    fs::create_dir(&scratch).map_err(|error| error.to_string())?;
    let mut retained = 0u32;
    let mut batch = 0u64;
    while retained < request.sequences {
        let wanted = (request.sequences - retained).min(4096);
        let sample_seed = splitmix64(request.seed.wrapping_add(batch));
        let fst = scratch.join(format!("{batch}.far"));
        let paths = sample_batch(
            model,
            sampler,
            printer,
            request.max_tokens,
            wanted,
            sample_seed,
            &fst,
        )?;
        let before = retained;
        for path in paths {
            writeln!(samples, "{path}").map_err(|error| error.to_string())?;
            retained += 1;
            if retained == request.sequences {
                break;
            }
        }
        if retained == before {
            return Err("OpenGrm produced no successful paths".into());
        }
        batch += 1;
        if batch > u64::from(request.sequences).saturating_add(16) {
            return Err("OpenGrm did not produce the requested path count".into());
        }
    }
    samples.flush().map_err(|error| error.to_string())?;
    fs::remove_dir_all(&scratch).map_err(|error| error.to_string())?;
    let samples_sha256 = digest(&samples_path)?;
    let manifest = manifest(request, &model_sha256, &samples_sha256, batch);
    let encoded = ennx_wire::json::pretty_string(&manifest).map_err(|error| error.to_string())?;
    fs::write(out.join("manifest.json"), encoded).map_err(|error| error.to_string())
}

fn sample_batch(
    model: &Path,
    sampler: &Path,
    printer: &Path,
    max_tokens: u32,
    wanted: u32,
    seed: u64,
    fst: &Path,
) -> Result<Vec<String>, String> {
    checked(
        Command::new(sampler).args([
            format!("--seed={seed}"),
            format!("--npath={wanted}"),
            format!("--max_length={max_tokens}"),
            model.display().to_string(),
            fst.display().to_string(),
        ]),
        "sfstrandgen",
    )?;
    let output = Command::new(printer)
        .arg(fst)
        .output()
        .map_err(|error| format!("start fstprint: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "fstprint exited with {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let graph = String::from_utf8(output.stdout).map_err(|error| error.to_string())?;
    paths(&graph, wanted as usize, max_tokens as usize)
}

#[derive(Debug)]
struct Arc {
    next: u64,
    output: String,
}

fn paths(graph: &str, wanted: usize, max_tokens: usize) -> Result<Vec<String>, String> {
    let mut arcs: BTreeMap<u64, Vec<Arc>> = BTreeMap::new();
    let mut finals = BTreeSet::new();
    for (index, line) in graph.lines().enumerate() {
        let fields: Vec<_> = line.split_whitespace().collect();
        if fields.is_empty() {
            continue;
        }
        let state = parse_state(fields[0], index)?;
        if fields.len() <= 2 {
            finals.insert(state);
            continue;
        }
        if fields.len() < 4 {
            return Err(format!("invalid fstprint arc on line {}", index + 1));
        }
        arcs.entry(state).or_default().push(Arc {
            next: parse_state(fields[1], index)?,
            output: fields[3].to_owned(),
        });
    }
    let mut found = Vec::with_capacity(wanted);
    let mut pending = vec![(0u64, Vec::new())];
    while let Some((state, tokens)) = pending.pop() {
        if finals.contains(&state) && !tokens.is_empty() {
            found.push(tokens.join(" "));
            if found.len() == wanted {
                break;
            }
        }
        if tokens.len() == max_tokens {
            continue;
        }
        if let Some(next) = arcs.get(&state) {
            for arc in next.iter().rev() {
                let mut path = tokens.clone();
                if !matches!(arc.output.as_str(), "0" | "<eps>" | "<epsilon>") {
                    path.push(arc.output.clone());
                }
                pending.push((arc.next, path));
            }
        }
    }
    Ok(found)
}

fn parse_state(field: &str, line: usize) -> Result<u64, String> {
    field
        .parse()
        .map_err(|_| format!("invalid fstprint state on line {}", line + 1))
}

fn manifest(request: Request, model: &str, samples: &str, batches: u64) -> Value {
    json!({
        "format":"ennx.grammar-corpus.v1",
        "generator":{"library":"OpenGrm SFst","binary":"sfstrandgen","revision":OPENGRM_REVISION,
            "openfst_revision":OPENFST_REVISION,"selection":"normalized_path_probability"},
        "model":{"sha256":model},
        "sampling":{"seed":request.seed,"sequences":request.sequences,
            "max_tokens":request.max_tokens,"batches":batches},
        "samples":{"path":"samples.txt","sha256":samples}
    })
}

fn checked(command: &mut Command, name: &str) -> Result<(), String> {
    let output = command
        .output()
        .map_err(|error| format!("start {name}: {error}"))?;
    if output.status.success() {
        Ok(())
    } else {
        Err(format!(
            "{name} exited with {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        ))
    }
}

fn digest(path: &Path) -> Result<String, String> {
    let mut file = File::open(path).map_err(|error| error.to_string())?;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let count = file.read(&mut buffer).map_err(|error| error.to_string())?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn absolute(root: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_owned()
    } else {
        root.join(path)
    }
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provenance() {
        let value = manifest(
            Request {
                seed: 7,
                sequences: 11,
                max_tokens: 13,
            },
            "model",
            "samples",
            2,
        );
        assert_eq!(value["generator"]["binary"].as_str(), Some("sfstrandgen"));
        assert_eq!(value["sampling"]["sequences"].as_u64(), Some(11));
        assert_eq!(value["samples"]["sha256"].as_str(), Some("samples"));
    }

    #[test]
    fn seeds() {
        assert_eq!(splitmix64(42), splitmix64(42));
        assert_ne!(splitmix64(42), splitmix64(43));
    }

    #[test]
    fn branching_paths() {
        let graph = "0\t1\talpha\talpha\n0\t2\tbeta\tbeta\n1\n2\n";
        assert_eq!(paths(graph, 2, 4).unwrap(), ["alpha", "beta"]);
    }

    #[test]
    fn epsilon_output() {
        let graph = "0 1 0 0\n1 2 word word\n2\n";
        assert_eq!(paths(graph, 1, 4).unwrap(), ["word"]);
    }
}
