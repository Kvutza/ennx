use serde::Deserialize;
use std::collections::HashSet;

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct KnnTuneConfig {
    pub output: String,
    pub rounds: usize,
    pub points: Vec<String>,
}

#[derive(Debug, PartialEq)]
pub(crate) struct ProposalTuneConfig {
    pub output: String,
    pub elements: usize,
    pub history: usize,
    pub candidates: usize,
    pub rounds: usize,
    pub warmup: usize,
    pub device: String,
    pub encoding: String,
    pub acquisition: String,
    pub neighbors: usize,
    pub edited_parameters: usize,
    pub seed: u64,
    pub length: f32,
    pub beta: f32,
    pub memory_budget_mib: Option<usize>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct KnnExperiment {
    version: u32,
    knn: Knn,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Knn {
    output: String,
    rounds: usize,
    #[serde(default)]
    defaults: Dimensions,
    points: Vec<Point>,
}

#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct Dimensions {
    rows: Option<usize>,
    queries: Option<usize>,
    dims: Option<usize>,
    k: Option<usize>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Point {
    name: String,
    rows: Option<usize>,
    queries: Option<usize>,
    dims: Option<usize>,
    k: Option<usize>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProposalExperiment {
    version: u32,
    proposal: Proposal,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Proposal {
    output: String,
    elements: usize,
    history: usize,
    candidates: usize,
    rounds: usize,
    #[serde(default)]
    warmup: Option<usize>,
    #[serde(default)]
    device: Option<String>,
    #[serde(default)]
    encoding: Option<String>,
    #[serde(default)]
    acquisition: Option<String>,
    #[serde(default)]
    neighbors: Option<usize>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    length: Option<f32>,
    #[serde(default)]
    beta: Option<f32>,
    #[serde(default)]
    memory_budget_mib: Option<usize>,
    #[serde(default)]
    edited_parameters: Option<usize>,
}

pub(crate) fn parse_knn(text: &str) -> Result<KnnTuneConfig, String> {
    let experiment: KnnExperiment = toml::from_str(text).map_err(|error| error.to_string())?;
    if experiment.version != 1 {
        return Err("version must be 1".into());
    }
    let config = experiment.knn;
    if config.output.trim().is_empty() || config.rounds == 0 || config.points.is_empty() {
        return Err("knn requires non-empty output and points, and positive rounds".into());
    }
    let mut names = HashSet::new();
    let mut points = Vec::new();
    for point in config.points {
        if point.name.is_empty()
            || !point
                .name
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || b"_-.".contains(&c))
            || !names.insert(point.name.clone())
        {
            return Err(format!(
                "point name {:?} must be unique and contain only letters, digits, _, -, or .",
                point.name
            ));
        }
        let dimension = |name, value: Option<usize>, default| {
            value.or(default).filter(|v| *v > 0).ok_or_else(|| {
                format!("point {}: {name} must be supplied and positive", point.name)
            })
        };
        let rows = dimension("rows", point.rows, config.defaults.rows)?;
        let queries = dimension("queries", point.queries, config.defaults.queries)?;
        let dims = dimension("dims", point.dims, config.defaults.dims)?;
        let k = dimension("k", point.k, config.defaults.k)?;
        if k > rows || k > 2048 {
            return Err(format!(
                "point {}: k must not exceed rows or 2048",
                point.name
            ));
        }
        let distance_bytes = rows.checked_mul(queries).and_then(|n| n.checked_mul(4));
        if !matches!(distance_bytes, Some(n) if n <= 256 * 1024 * 1024) {
            return Err(format!(
                "point {}: distance matrix exceeds 256 MiB",
                point.name
            ));
        }
        for count in [rows, queries] {
            count
                .checked_mul(dims)
                .and_then(|n| n.checked_mul(8))
                .ok_or_else(|| format!("point {}: input allocation size overflows", point.name))?;
        }
        // Encode positional arguments only at the legacy benchmark boundary.
        points.push(format!("{}:{rows}:{queries}:{dims}:{k}", point.name));
    }
    Ok(KnnTuneConfig {
        output: config.output,
        rounds: config.rounds,
        points,
    })
}

pub(crate) fn parse_config(text: &str) -> Result<ProposalTuneConfig, String> {
    let experiment: ProposalExperiment = toml::from_str(text).map_err(|error| error.to_string())?;
    if experiment.version != 1 {
        return Err("version must be 1".into());
    }
    let config = experiment.proposal;
    if config.output.trim().is_empty() {
        return Err("proposal requires a non-empty output".into());
    }
    if config.elements == 0 || config.history == 0 || config.candidates == 0 || config.rounds == 0 {
        return Err("proposal requires positive elements, history, candidates, and rounds".into());
    }
    let warmup = config
        .warmup
        .unwrap_or_else(|| config.history.saturating_sub(1));
    let device = normalize_choice(
        config.device.as_deref(),
        "auto",
        "device",
        &["auto", "cpu", "metal", "opencl", "cuda"],
    )?;
    let encoding = normalize_choice(
        config.encoding.as_deref(),
        "int4",
        "encoding",
        &[
            "int4", "int8", "fp4", "fp4_e2m1", "e2m1", "fp8", "fp8_e4m3", "e4m3", "fp8_e5m2",
            "e5m2",
        ],
    )?;
    let acquisition = normalize_choice(
        config.acquisition.as_deref(),
        "ucb",
        "acquisition",
        &["ucb", "thompson", "pareto"],
    )?;
    let neighbors = config.neighbors.unwrap_or(1);
    if neighbors == 0 || neighbors > config.history {
        return Err("proposal neighbors must be between 1 and history".into());
    }
    let length = config.length.unwrap_or(0.8);
    if !length.is_finite() || length <= 0.0 {
        return Err("proposal length must be finite and positive".into());
    }
    let beta = config.beta.unwrap_or(1.0);
    if !beta.is_finite() || beta < 0.0 {
        return Err("proposal beta must be finite and nonnegative".into());
    }
    if let Some(mib) = config.memory_budget_mib {
        if mib == 0 {
            return Err("proposal memory_budget_mib must be positive".into());
        }
    }
    let edited_parameters = config.edited_parameters.unwrap_or(0);
    if edited_parameters > config.elements {
        return Err("proposal edited_parameters must not exceed elements".into());
    }
    Ok(ProposalTuneConfig {
        output: config.output,
        elements: config.elements,
        history: config.history,
        candidates: config.candidates,
        rounds: config.rounds,
        warmup,
        device,
        encoding,
        acquisition,
        neighbors,
        edited_parameters,
        seed: config.seed.unwrap_or(0),
        length,
        beta,
        memory_budget_mib: config.memory_budget_mib,
    })
}

fn normalize_choice(
    value: Option<&str>,
    default: &str,
    name: &str,
    allowed: &[&str],
) -> Result<String, String> {
    let normalized = value.unwrap_or(default).trim().to_ascii_lowercase();
    if normalized.is_empty() {
        return Err(format!("{name} must be non-empty"));
    }
    if allowed.contains(&normalized.as_str()) {
        Ok(normalized)
    } else {
        Err(format!(
            "proposal {name} {normalized:?} is unsupported; expected one of {}",
            allowed.join(", ")
        ))
    }
}
