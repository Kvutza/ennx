use deser::Deserialize;
use std::collections::HashSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExperimentKind {
    Knn,
    Proposal,
    TurboEnn,
    KernelSearch,
}

pub(crate) fn is_pretrain(text: &str) -> Result<bool, String> {
    Ok(ennx::config::parse_tune(text)?.experiment
        == Some(ennx::config::TurboEnnExperiment::Pretrain))
}

pub(crate) fn experiment_kind(text: &str) -> Result<ExperimentKind, String> {
    let value: ennx_wire::toml::Value =
        ennx_wire::toml::from_str(text).map_err(|error| error.to_string())?;
    let table = value
        .as_map()
        .ok_or("experiment config must be a TOML table")?;
    let mut kinds = Vec::new();
    if table.contains_key("kernel-search") {
        kinds.push(ExperimentKind::KernelSearch);
        if table.contains_key("pretrain") || table.contains_key("experiment") {
            return Err("kernel-search must reference a separate baseline configuration".into());
        }
    }
    if table.contains_key("knn") {
        kinds.push(ExperimentKind::Knn);
    }
    if table
        .get("proposal")
        .and_then(|value| value.as_map())
        .is_some_and(|proposal| {
            ["output", "elements", "history", "candidates", "rounds"]
                .iter()
                .all(|key| proposal.contains_key(*key))
        })
    {
        kinds.push(ExperimentKind::Proposal);
    }
    let selected = kinds
        .into_iter()
        .map(|kind| ("experiment", kind))
        .collect::<Vec<_>>();
    match selected.as_slice() {
        [(_, kind)] => Ok(*kind),
        [] => Ok(ExperimentKind::TurboEnn),
        _ => Err("experiment config contains multiple experiment tables".into()),
    }
}

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
#[deser(deny_unknown_fields)]
struct KnnExperiment {
    version: u32,
    knn: Knn,
}

#[derive(Deserialize)]
#[deser(deny_unknown_fields)]
struct Knn {
    output: String,
    rounds: usize,
    #[deser(default)]
    defaults: Dimensions,
    points: Vec<Point>,
}

#[derive(Default, Deserialize)]
#[deser(deny_unknown_fields)]
struct Dimensions {
    rows: Option<usize>,
    queries: Option<usize>,
    dims: Option<usize>,
    k: Option<usize>,
}

#[derive(Deserialize)]
#[deser(deny_unknown_fields)]
struct Point {
    name: String,
    rows: Option<usize>,
    queries: Option<usize>,
    dims: Option<usize>,
    k: Option<usize>,
}

#[derive(Deserialize)]
#[deser(deny_unknown_fields)]
struct ProposalExperiment {
    version: u32,
    proposal: Proposal,
}

#[derive(Deserialize)]
#[deser(deny_unknown_fields)]
struct Proposal {
    output: String,
    elements: usize,
    history: usize,
    candidates: usize,
    rounds: usize,
    #[deser(default)]
    warmup: Option<usize>,
    #[deser(default)]
    device: Option<String>,
    #[deser(default)]
    encoding: Option<String>,
    #[deser(default)]
    acquisition: Option<String>,
    #[deser(default)]
    neighbors: Option<usize>,
    #[deser(default)]
    seed: Option<u64>,
    #[deser(default)]
    length: Option<f32>,
    #[deser(default)]
    beta: Option<f32>,
    #[deser(default)]
    memory_budget_mib: Option<usize>,
    #[deser(default)]
    edited_parameters: Option<usize>,
}

pub(crate) fn parse_knn(text: &str) -> Result<KnnTuneConfig, String> {
    let experiment: KnnExperiment =
        ennx_wire::toml::from_str(text).map_err(|error| error.to_string())?;
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
    let experiment: ProposalExperiment =
        ennx_wire::toml::from_str(text).map_err(|error| error.to_string())?;
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
