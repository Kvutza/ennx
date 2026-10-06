use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use ennx::Perturbation;
use ennx::config::{
    AcquisitionSpec, HistoryGeometry, PretrainCorpus, TrustRegionSpec, TuneSpec, TurboEnnExperiment,
};
use ennx::procedural_pool::ProposalMethod;
use ennx_wire::json::{Value, json};
use sha2::{Digest, Sha256};

use crate::experiment;

const THRESHOLD: &str = "threshold-proposal-v1";
const QUESTION: &str =
    "Does posterior-adaptive spectral sampling improve Pareto progress per objective call?";

struct Arm {
    id: &'static str,
    role: &'static str,
    method: ProposalMethod,
    spec: TuneSpec,
}

struct Counts {
    complete: usize,
    failed: usize,
    running: usize,
}

pub(crate) struct Request<'a> {
    pub protocol: Option<&'a str>,
    pub baseline: Option<&'a Path>,
    pub out: Option<&'a Path>,
    pub rounds: u32,
    pub reps: u32,
    pub execute: bool,
    pub json: bool,
}

pub(crate) fn run(root: &Path, request: Request<'_>) -> Result<(), String> {
    let Some(protocol) = request.protocol else {
        return list(request.json);
    };
    if protocol != THRESHOLD {
        return Err(format!("unknown experiment protocol {protocol:?}"));
    }
    let baseline = request
        .baseline
        .ok_or("threshold-proposal-v1 requires a baseline TOML")?;
    let text = fs::read_to_string(baseline).map_err(|error| error.to_string())?;
    let mut base = TuneSpec::parse(&text)?;
    check(&base, request.rounds, request.reps, false)?;
    base.output = None;
    base.run.rounds = request.rounds;
    base.run.reps = request.reps;
    base.run.validation_interval = Some(64);
    base.proposal.distribution = Some(Perturbation::Rademacher);
    base.enn.geometry = HistoryGeometry::Latent;
    let fingerprint = fingerprint(&base.to_toml()?);
    let out = request.out.map(PathBuf::from).unwrap_or_else(|| {
        root.join(".cache/ennx/experiments")
            .join(THRESHOLD)
            .join(&fingerprint)
    });
    let arms = arms(&base);
    write_plan(&out, &fingerprint, &arms, request.rounds, request.reps)?;
    if request.execute {
        execute(root, &out, &arms)?;
    }
    display(&out, &fingerprint, &arms, request.json)
}

fn list(as_json: bool) -> Result<(), String> {
    if as_json {
        println!(
            "{}",
            ennx_wire::json::pretty_string(&json!({
                "protocols": [{
                    "id": THRESHOLD,
                    "kind": "proposal-ablation",
                    "question": QUESTION,
                    "arms": ["independent-rademacher", "spectral-basis", "threshold-posterior"],
                }]
            }))
            .map_err(|error| error.to_string())?
        );
    } else {
        println!("{THRESHOLD} | proposal-ablation | {QUESTION}");
    }
    Ok(())
}

fn check(spec: &TuneSpec, rounds: u32, reps: u32, execute: bool) -> Result<(), String> {
    if spec.experiment != TurboEnnExperiment::Pretrain {
        return Err("threshold-proposal-v1 requires a pretrain baseline".into());
    }
    if !matches!(spec.acquisition, AcquisitionSpec::AugmentedChebyshev { .. })
        || !matches!(spec.trust_region, TrustRegionSpec::Morbo { .. })
    {
        return Err("threshold-proposal-v1 requires MORBO acquisition and control".into());
    }
    if rounds <= ennx::threshold::FEATURE_COUNT as u32 {
        return Err(format!(
            "threshold-proposal-v1 requires more than {} rounds so the posterior is evaluated",
            ennx::threshold::FEATURE_COUNT
        ));
    }
    if reps < 2 {
        return Err("threshold-proposal-v1 requires at least two paired repetitions".into());
    }
    if [
        spec.seeds.model,
        spec.seeds.reference,
        spec.seeds.proposal,
        spec.seeds.acquisition,
    ]
    .iter()
    .any(Option::is_some)
    {
        return Err("paired experiment protocols derive their seeds; remove explicit seeds".into());
    }
    if execute
        && (spec.data.train.is_none()
            || (spec.generation.is_none() && spec.data.validation.is_none()))
    {
        return Err("experiment execution requires resolved corpus splits".into());
    }
    Ok(())
}

fn arms(base: &TuneSpec) -> Vec<Arm> {
    [
        (
            "independent-rademacher",
            "baseline",
            ProposalMethod::Independent,
        ),
        ("spectral-basis", "control", ProposalMethod::SpectralBasis),
        (
            "threshold-posterior",
            "candidate",
            ProposalMethod::PolynomialThreshold,
        ),
    ]
    .into_iter()
    .map(|(id, role, method)| {
        let mut spec = base.clone();
        spec.proposal.method = method;
        Arm {
            id,
            role,
            method,
            spec,
        }
    })
    .collect()
}

fn write_plan(
    out: &Path,
    fingerprint: &str,
    arms: &[Arm],
    rounds: u32,
    reps: u32,
) -> Result<(), String> {
    fs::create_dir_all(out).map_err(|error| error.to_string())?;
    let arm_dir = out.join("arms");
    fs::create_dir_all(&arm_dir).map_err(|error| error.to_string())?;
    for arm in arms {
        write_same(
            &arm_dir.join(format!("{}.toml", arm.id)),
            &arm.spec.to_toml()?,
        )?;
    }
    let entries = arms
        .iter()
        .map(|arm| {
            json!({
                "id": arm.id,
                "role": arm.role,
                "proposal": method(arm.method),
            })
        })
        .collect::<Vec<_>>();
    let manifest = json!({
        "schema": "ennx.experiment.v1",
        "protocol": THRESHOLD,
        "kind": "proposal-ablation",
        "question": QUESTION,
        "fingerprint": fingerprint,
        "rounds": rounds,
        "repetitions": reps,
        "posterior-burn-in": ennx::threshold::FEATURE_COUNT,
        "primary-metric": "normalized-hypervolume-gain-auc-per-objective-call",
        "arms": entries,
    });
    let manifest = ennx_wire::json::pretty_string(&manifest).map_err(|error| error.to_string())?;
    write_same(&out.join("experiment.json"), &manifest)
}

fn write_same(path: &Path, text: &str) -> Result<(), String> {
    match fs::read_to_string(path) {
        Ok(existing) if existing == text => Ok(()),
        Ok(_) => Err(format!("refusing to replace differing {}", path.display())),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            fs::write(path, text).map_err(|error| error.to_string())
        }
        Err(error) => Err(error.to_string()),
    }
}

fn execute(root: &Path, out: &Path, arms: &[Arm]) -> Result<(), String> {
    let request_dir = out.join(".requests");
    fs::create_dir_all(&request_dir).map_err(|error| error.to_string())?;
    let mut errors = Vec::new();
    for arm in arms {
        let mut spec = arm.spec.clone();
        spec.output = Some(out.join("runs").join(arm.id));
        if spec.data.train.is_none() {
            resolve_corpus(root, &mut spec)?;
        }
        check(&spec, spec.run.rounds, spec.run.reps, true)?;
        let path = request_dir.join(format!("{}.toml", arm.id));
        fs::write(&path, spec.to_toml()?).map_err(|error| error.to_string())?;
        println!("Running {} ({})", arm.id, arm.role);
        if let Err(error) = experiment::tune(root, &path, false) {
            errors.push(format!("{}: {error}", arm.id));
        }
        let _ = fs::remove_file(path);
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(format!("experiment arms failed: {}", errors.join("; ")))
    }
}

fn display(out: &Path, fingerprint: &str, arms: &[Arm], as_json: bool) -> Result<(), String> {
    let rows = arms
        .iter()
        .map(|arm| {
            let counts = counts(&out.join("runs").join(arm.id))?;
            Ok((arm, counts))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let failed = rows.iter().map(|(_, count)| count.failed).sum::<usize>();
    let complete = rows.iter().all(|(_, count)| count.complete > 0);
    let any = rows
        .iter()
        .any(|(_, count)| count.complete + count.failed + count.running > 0);
    let execution = if failed > 0 {
        "failed"
    } else if complete {
        "complete"
    } else if any {
        "running"
    } else {
        "planned"
    };
    let evidence = if failed > 0 {
        "invalid"
    } else if complete {
        "ready"
    } else {
        "collecting"
    };
    let analysis = if complete {
        let inputs = arms
            .iter()
            .map(|arm| {
                let scales = match &arm.spec.acquisition {
                    AcquisitionSpec::AugmentedChebyshev { scales, .. } => scales.as_slice(),
                    _ => unreachable!("checked protocol acquisition"),
                };
                crate::protocol_analysis::Arm {
                    id: arm.id,
                    role: arm.role,
                    scales,
                }
            })
            .collect::<Vec<_>>();
        crate::protocol_analysis::write(out, &inputs)?
    } else {
        None
    };
    let decision = analysis
        .as_ref()
        .and_then(|report| report["winner"].as_str())
        .unwrap_or("pending-analysis");
    if as_json {
        let rows = rows
            .iter()
            .map(|(arm, count)| {
                json!({
                    "id": arm.id,
                    "role": arm.role,
                    "complete": count.complete,
                    "failed": count.failed,
                    "running": count.running,
                })
            })
            .collect::<Vec<Value>>();
        println!(
            "{}",
            ennx_wire::json::pretty_string(&json!({
                "protocol": THRESHOLD,
                "fingerprint": fingerprint,
                "execution": execution,
                "evidence": evidence,
                "decision": decision,
                "arms": rows,
                "analysis": analysis,
                "artifact": out,
            }))
            .map_err(|error| error.to_string())?
        );
        return Ok(());
    }
    println!("{THRESHOLD} | proposal-ablation");
    println!("question   {QUESTION}");
    println!("execution  {execution}");
    println!("evidence   {evidence}");
    println!("decision   {decision}");
    for (arm, count) in rows {
        println!(
            "{:<20} {:<9} complete={} failed={} running={}",
            arm.id, arm.role, count.complete, count.failed, count.running
        );
    }
    println!("artifacts  {}", out.display());
    Ok(())
}

fn counts(root: &Path) -> Result<Counts, String> {
    let mut count = Counts {
        complete: 0,
        failed: 0,
        running: 0,
    };
    if !root.exists() {
        return Ok(count);
    }
    for entry in fs::read_dir(root).map_err(|error| error.to_string())? {
        let path = entry.map_err(|error| error.to_string())?.path();
        if !path.is_dir() {
            continue;
        }
        let result = path.join("result.json");
        if !result.exists() {
            count.running += 1;
            continue;
        }
        let result: Value = ennx_wire::json::from_reader(
            fs::File::open(result).map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        if result.get("status").and_then(|value| value.kind().as_str()) == Some("completed") {
            count.complete += 1;
        } else {
            count.failed += 1;
        }
    }
    Ok(count)
}

pub(crate) fn resolve_corpus(root: &Path, spec: &mut TuneSpec) -> Result<(), String> {
    let corpus = spec.corpus.ok_or("path-free pretraining requires corpus")?;
    let generation = spec.generation.as_ref();
    let is_episodes = generation.is_some_and(|g| {
        !matches!(
            g.reward,
            ennx::config::GenerationReward::FreeRunningCrossEntropy
        )
    });
    let recipe = corpus_recipe(
        corpus,
        if is_episodes {
            generation.and_then(|generation| generation.corpus_prompt_tokens)
        } else {
            None
        },
        if is_episodes {
            generation.map(|generation| generation.max_tokens)
        } else {
            None
        },
    )?;
    let id = corpus_id(&recipe)?;
    let directory = root.join(".cache/ennx/corpora").join(&id);
    let manifest = corpus_manifest(&directory, &id, &recipe)?;
    let paths = corpus_splits(&directory, &id, &recipe, &manifest)?;
    if is_episodes {
        corpus_episodes(&directory, &id, &recipe, &manifest, spec)?;
    }
    spec.data.train = Some(paths[0].clone());
    if spec.generation.is_none() {
        spec.data.validation = Some(paths[1].clone());
    }
    Ok(())
}

fn corpus_manifest(directory: &Path, id: &str, recipe: &Value) -> Result<Value, String> {
    let manifest_path = directory.join("manifest.json");
    let manifest: Value = ennx_wire::json::from_reader(
        fs::File::open(&manifest_path)
            .map_err(|error| format!("corpus {id} is not cached: {error}"))?,
    )
    .map_err(|error| {
        format!(
            "invalid corpus manifest {}: {error}",
            manifest_path.display()
        )
    })?;
    if manifest
        .get("format")
        .and_then(|value| value.kind().as_str())
        != Some("ennx.pretraining.v1")
        || manifest
            .get("recipe_id")
            .and_then(|value| value.kind().as_str())
            != Some(id)
        || manifest.get("source") != recipe.get("source")
        || manifest.get("split_policy") != recipe.get("split_policy")
    {
        return Err(format!(
            "cached corpus {id} has a mismatched recipe identity or provenance"
        ));
    }
    if let Some(stored) = manifest.get("recipe") {
        if canonical_json(stored)? != canonical_json(&recipe)? {
            return Err(format!(
                "cached corpus {id} recipe does not match the requested recipe"
            ));
        }
    }
    Ok(manifest)
}

fn corpus_splits(
    directory: &Path,
    id: &str,
    recipe: &Value,
    manifest: &Value,
) -> Result<[PathBuf; 3], String> {
    let sequences = recipe
        .get("sequences")
        .ok_or("corpus recipe has no split sizes")?;
    let splits = manifest
        .get("splits")
        .ok_or("corpus manifest has no splits")?;
    let mut paths = Vec::new();
    for split in ["train", "validation", "test"] {
        let expected = sequences
            .get(split)
            .and_then(|value| value.kind().as_u64())
            .ok_or("corpus recipe split size is invalid")?;
        let record = splits
            .get(split)
            .ok_or_else(|| format!("cached corpus {id} is missing {split} metadata"))?;
        if record
            .get("sequences")
            .and_then(|value| value.kind().as_u64())
            != Some(expected)
        {
            return Err(format!(
                "cached corpus {id} has the wrong {split} sequence count"
            ));
        }
        let path = directory.join(format!("{split}.ennxptn"));
        verify_file(
            &path,
            record.get("sha256").and_then(|value| value.kind().as_str()),
        )?;
        paths.push(path);
    }
    let tokenizer = manifest
        .get("tokenizer")
        .ok_or("corpus manifest has no tokenizer record")?;
    verify_file(
        &directory.join("tokenizer.json"),
        tokenizer
            .get("sha256")
            .and_then(|value| value.kind().as_str()),
    )?;
    paths
        .try_into()
        .map_err(|_| "corpus recipe did not resolve all required splits".into())
}

fn corpus_episodes(
    directory: &Path,
    id: &str,
    recipe: &Value,
    manifest: &Value,
    spec: &mut TuneSpec,
) -> Result<(), String> {
    if let Some(episodes) = recipe.get("episodes") {
        let record = manifest
            .get("episodes")
            .ok_or("cached corpus is missing requested episode metadata")?;
        for key in [
            "schema",
            "prompt_tokens",
            "generated_tokens",
            "counts",
            "sources",
            "buckets",
        ] {
            if record.get(key) != episodes.get(key) {
                return Err(format!(
                    "cached corpus {id} has mismatched generation episodes"
                ));
            }
        }
        let episode_path = directory.join("episodes.json");
        verify_file(
            &episode_path,
            record.get("sha256").and_then(|value| value.kind().as_str()),
        )?;
        if let Some(generation) = &mut spec.generation {
            generation.episode_dataset = Some(episode_path);
        }
    } else if manifest.get("episodes").is_some() {
        return Err(format!(
            "cached corpus {id} unexpectedly contains generation episodes"
        ));
    }
    Ok(())
}

fn corpus_recipe(
    corpus: PretrainCorpus,
    prompt: Option<u32>,
    generated: Option<u32>,
) -> Result<Value, String> {
    let (preset, source, split_policy, sequences) = match corpus {
        PretrainCorpus::StackV3PythonPilotV1 => (
            "stack_v3_python_pilot_v1",
            "HuggingFaceCode/stack-v3-train",
            "repository_sha256_v2",
            json!({"train":20,"validation":16,"test":16}),
        ),
        PretrainCorpus::StackV3Python800kV1 => (
            "stack_v3_python_800k_v1",
            "HuggingFaceCode/stack-v3-train",
            "repository_sha256_v2",
            json!({"train":200,"validation":16,"test":16}),
        ),
        PretrainCorpus::Fineweb10btPilotV1 => (
            "fineweb_10bt_pilot_v1",
            "HuggingFaceFW/fineweb",
            "sha256_lowercase_url_host_mod20_v1",
            json!({"train":20,"validation":16,"test":16}),
        ),
    };
    if corpus == PretrainCorpus::Fineweb10btPilotV1 && prompt.is_some() {
        return Err("FineWeb corpus does not support generation episodes".into());
    }
    let mut recipe = if corpus == PretrainCorpus::Fineweb10btPilotV1 {
        json!({
            "preset":preset,"format":"ennx.pretraining.v1",
            "source":{"id":source,"revision":"9bb295ddab0e05d785b879661af7260fed5140fc","subset":"sample-10BT","directory":"sample/10BT"},
            "split_policy":split_policy,"deduplication":"exact_utf8_sha256_across_splits",
            "filter":{"min_characters":64,"max_characters":262144},
            "tokenizer":{"kind":"byte_level_bpe","vocabulary":8192,"package":"tokenizers==0.21.4"},
            "datasets_package":"datasets==4.1.1","reader":"parquet_file_projected_single_thread_v1",
            "context":4096,"sequences":sequences,
            "required_characters":{"train":8000000,"validation":393216,"test":393216},
            "seed_policy":"sha256_corpus_recipe"
        })
    } else {
        json!({
            "preset":preset,"format":"ennx.pretraining.v1",
            "source":{"id":source,"revision":"1f61b735bc0a5698345ce2196730f24bfa467f33"},
            "collector_version":2,"split_policy":split_policy,"shuffle_repositories":1,"parquet_batch_repositories":1,
            "tokenizer":{"kind":"byte_level_bpe","vocabulary":8192,"package":"tokenizers==0.21.4"},
            "datasets_package":"datasets==4.1.1","context":4096,"seed":1162759768,
            "sequences":sequences,
            "mixture_tokens_per_sequence":{"implementation":2867,"tests":615,"documentation":410,"configuration":204}
        })
    };
    if let Some(prompt_tokens) = prompt {
        let generated_tokens = generated.ok_or("generation corpus prompt requires max-tokens")?;
        recipe.as_map_mut().ok_or("corpus recipe must be an object")?.insert("episodes", json!({
            "schema":"ennx.document_episodes.v2","prompt_tokens":prompt_tokens,"generated_tokens":generated_tokens,
            "counts":{"train":16,"validation":3,"test":3},
            "sources":{"train":["train"],"validation":["validation"],"test":["test"]},
            "buckets":["implementation","tests"]
        }));
    }
    Ok(recipe)
}

fn corpus_id(recipe: &Value) -> Result<String, String> {
    let mut hash = Sha256::new();
    hash.update(b"ennx-corpus-v1\0");
    hash.update(canonical_json(recipe)?.as_bytes());
    Ok(format!("{:x}", hash.finalize())[..20].to_owned())
}

fn canonical_json(value: &Value) -> Result<String, String> {
    if let Some(map) = value.as_map() {
        let mut entries = map
            .iter()
            .map(|(key, value)| {
                Ok((
                    key.kind()
                        .as_str()
                        .ok_or("corpus recipe key is not a string")?,
                    value,
                ))
            })
            .collect::<Result<Vec<_>, String>>()?;
        entries.sort_by(|left, right| left.0.cmp(right.0));
        let fields = entries
            .into_iter()
            .map(|(key, value)| {
                Ok(format!(
                    "{}:{}",
                    ennx_wire::json::to_string(&key).map_err(|error| error.to_string())?,
                    canonical_json(value)?
                ))
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(format!("{{{}}}", fields.join(",")))
    } else if let Some(sequence) = value.as_seq() {
        let items = sequence
            .iter()
            .map(canonical_json)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(format!("[{}]", items.join(",")))
    } else {
        ennx_wire::json::to_string(value).map_err(|error| error.to_string())
    }
}

fn verify_file(path: &Path, expected: Option<&str>) -> Result<(), String> {
    let expected =
        expected.ok_or_else(|| format!("manifest has no digest for {}", path.display()))?;
    if expected.len() != 64 || !expected.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!(
            "manifest has invalid SHA256 for {}",
            path.display()
        ));
    }
    let mut file = fs::File::open(path)
        .map_err(|error| format!("required corpus file {}: {error}", path.display()))?;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let count = file.read(&mut buffer).map_err(|error| error.to_string())?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    if format!("{:x}", hash.finalize()) != expected {
        return Err(format!("cached corpus file is corrupt: {}", path.display()));
    }
    Ok(())
}

fn method(value: ProposalMethod) -> &'static str {
    value.name()
}

fn fingerprint(text: &str) -> String {
    let hash = text.bytes().fold(0xcbf2_9ce4_8422_2325u64, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
    });
    format!("{hash:016x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASE: &str = r#"
version = 2
experiment = "pretrain"
model = "fbt-pisa1-legacy-v1"
corpus = "stack-v3-python-pilot-v1"

[proposal]
method = "polynomial-threshold"
distribution = "rademacher"

[enn]
geometry = "latent"

[acquisition]
method = "augmented-chebyshev"
scales = [1.0, 1.0]
preferences = [1.0, 1.0]
alpha = 0.0
seed-domain = "threshold-protocol-test"
[trust-region]
method = "morbo"
regions = 2
rescalarize = "on-restart"
clip = true

[generation]
purpose = "systems-probe"
initialization = "patterned"
max-tokens = 32
temperature = 0.0
corpus-prompt-tokens = 1

[generation.reward]
kind = "code-objectives"
critical-window = 1
"#;

    #[test]
    fn threshold_arms() {
        let mut base = TuneSpec::parse(BASE).unwrap();
        base.run.rounds = 512;
        base.run.reps = 3;
        check(&base, 512, 3, false).unwrap();
        let arms = arms(&base);
        assert_eq!(arms.len(), 3);
        assert_eq!(arms[0].spec.proposal.method, ProposalMethod::Independent);
        assert_eq!(arms[1].spec.proposal.method, ProposalMethod::SpectralBasis);
        assert_eq!(
            arms[2].spec.proposal.method,
            ProposalMethod::PolynomialThreshold
        );
        for arm in arms {
            arm.spec.overrides().unwrap().validate_experiment().unwrap();
        }
    }

    #[test]
    fn posterior_budget() {
        let base = TuneSpec::parse(BASE).unwrap();
        assert!(check(&base, ennx::threshold::FEATURE_COUNT as u32, 3, false).is_err());
        assert!(check(&base, 512, 1, false).is_err());
    }

    #[test]
    fn corpus_identity() {
        let recipe =
            corpus_recipe(PretrainCorpus::StackV3PythonPilotV1, Some(128), Some(4096)).unwrap();
        assert_eq!(corpus_id(&recipe).unwrap(), "95bda3b17e0602eb02ac");
    }

    #[test]
    fn corpus_files() {
        let directory =
            std::env::temp_dir().join(format!("ennx-corpus-test-{}", std::process::id()));
        fs::create_dir_all(&directory).unwrap();
        let path = directory.join("train.ennxptn");
        fs::write(&path, b"train").unwrap();
        let digest = format!("{:x}", Sha256::digest(b"train"));
        verify_file(&path, Some(&digest)).unwrap();
        assert!(verify_file(&path, Some(&"0".repeat(64))).is_err());
        let _ = fs::remove_dir_all(directory);
    }
}
