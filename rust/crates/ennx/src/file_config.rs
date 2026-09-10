//! File-backed ennx configuration (`~/.ennx/config.toml`).

use std::fs;
use std::io::{ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::sync::RwLock;

use serde::{Deserialize, Serialize};

static CONFIG_OVERRIDE: RwLock<Option<PathBuf>> = RwLock::new(None);

/// Override the config file path (tests / ops tools). Pass `None` to restore the default.
pub fn set_path(path: Option<PathBuf>) -> Result<(), String> {
    let mut active = CONFIG_OVERRIDE
        .write()
        .map_err(|_| "config path lock poisoned")?;
    let config = Config::with_path(path.clone().unwrap_or_else(config_path));
    let tuning = config.load()?.bpann.to_tuning();
    // Validate before replacing the active path or tuning.
    bpann::set_provider(Box::new(move || tuning));
    *active = path;
    Ok(())
}

/// Default path: `~/.ennx/config.toml`.
pub fn config_path() -> PathBuf {
    home_dir().join(".ennx").join("config.toml")
}

fn home_dir() -> PathBuf {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}

fn active_path() -> PathBuf {
    CONFIG_OVERRIDE
        .read()
        .expect("config path lock")
        .clone()
        .unwrap_or_else(config_path)
}

/// Tunable BPANN parameters persisted under `[bpann]` in the config file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BpannConfig {
    pub index_fragment: usize,
    pub index_max: usize,
    pub search_fragment: usize,
    pub fragment_rows: usize,
    pub search_limit: usize,
    /// When set, used as the k-means build seed; otherwise the batch start row is used.
    pub build_seed: Option<u64>,
    /// Rows of pending observations before an index flush is scheduled.
    pub soft_threshold: usize,
    /// Hard pending cap (soft-sync on caller). `None` means the key was absent in TOML;
    /// resolved to `max(PENDING_HARD, soft)` on load.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hard_threshold: Option<usize>,
    /// Batch size at or below which builds use a single row-ID leaf (no k-means tree).
    pub structured_limit: usize,
    /// Beam width used during approximate tree traversal.
    pub search_width: usize,
    /// Max indexed rows using exhaustive leaf search; build stores no skip edges at or below.
    pub exhaustive_limit: usize,
    /// Max indexed rows using skip-refinement search; build stores skip edges in the middle band.
    pub skip_limit: usize,
}

impl Default for BpannConfig {
    /// Mirror the compiled-in BPANN defaults so the values live in exactly one place.
    fn default() -> Self {
        bpann::BpannTuning::default().into()
    }
}

impl From<bpann::BpannTuning> for BpannConfig {
    fn from(t: bpann::BpannTuning) -> Self {
        Self {
            index_fragment: t.index_fragment,
            index_max: t.index_max,
            search_fragment: t.search_fragment,
            fragment_rows: t.fragment_rows,
            search_limit: t.search_limit,
            build_seed: t.build_seed,
            soft_threshold: t.soft_threshold,
            hard_threshold: Some(t.hard_threshold),
            structured_limit: t.structured_limit,
            search_width: t.search_width,
            exhaustive_limit: t.exhaustive_limit,
            skip_limit: t.skip_limit,
        }
    }
}

impl BpannConfig {
    /// Validate all tunable fields. Returns an error describing the first violation.
    pub fn validate(&self) -> Result<(), String> {
        self.to_tuning().validate()
    }

    /// Resolve the hard pending cap: absent key → `max(DEFAULT_HARD, soft)`.
    pub fn resolved_threshold(&self) -> usize {
        self.hard_threshold
            .unwrap_or_else(|| std::cmp::max(bpann::PENDING_HARD, self.soft_threshold))
    }

    fn to_tuning(&self) -> bpann::BpannTuning {
        bpann::BpannTuning {
            index_fragment: self.index_fragment,
            index_max: self.index_max,
            search_fragment: self.search_fragment,
            fragment_rows: self.fragment_rows,
            search_limit: self.search_limit,
            build_seed: self.build_seed,
            soft_threshold: self.soft_threshold,
            hard_threshold: self.resolved_threshold(),
            structured_limit: self.structured_limit,
            search_width: self.search_width,
            exhaustive_limit: self.exhaustive_limit,
            skip_limit: self.skip_limit,
        }
    }
}

/// Root document for `~/.ennx/config.toml`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ConfigFile {
    pub bpann: BpannConfig,
}

/// File-backed configuration object.
///
/// Each parameter accessor re-reads `~/.ennx/config.toml` (or the path from
/// [`set_path`]). If the file is missing, it is created with defaults.
/// Invalid files return an error with the path; only missing files receive defaults.
#[derive(Debug, Clone)]
pub struct Config {
    path: PathBuf,
}

impl Default for Config {
    fn default() -> Self {
        Self::new()
    }
}

impl Config {
    /// Config for the active path (`set_path` override or `~/.ennx/config.toml`).
    pub fn new() -> Self {
        Self {
            path: active_path(),
        }
    }

    /// Config for an explicit path (does not change the process-wide override).
    pub fn with_path(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Ensure the config file exists, then load and validate it.
    ///
    /// Report unreadable files, malformed TOML, and invalid parameter values.
    pub fn load(&self) -> Result<ConfigFile, String> {
        self.ensure_exists()?;
        let text = fs::read_to_string(&self.path)
            .map_err(|error| format!("read config {}: {error}", self.path.display()))?;
        let file: ConfigFile = toml::from_str(&text)
            .map_err(|error| format!("parse config {}: {error}", self.path.display()))?;
        file.bpann
            .validate()
            .map_err(|error| format!("invalid config {}: {error}", self.path.display()))?;
        Ok(file)
    }

    /// Write `file` to this config path, creating parent directories as needed.
    ///
    /// Rejects invalid BPANN parameter values.
    pub fn save(&self, file: &ConfigFile) -> Result<(), String> {
        file.bpann.validate()?;
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        let text = toml::to_string_pretty(file).map_err(|e| e.to_string())?;
        fs::write(&self.path, text).map_err(|e| e.to_string())
    }

    fn ensure_exists(&self) -> Result<(), String> {
        match fs::metadata(&self.path) {
            Ok(_) => return Ok(()),
            Err(error) if error.kind() == ErrorKind::NotFound => {}
            Err(error) => return Err(format!("read config {}: {error}", self.path.display())),
        }
        let create = || -> Result<(), String> {
            if let Some(parent) = self
                .path
                .parent()
                .filter(|parent| !parent.as_os_str().is_empty())
            {
                fs::create_dir_all(parent).map_err(|error| error.to_string())?;
            }
            let text = toml::to_string_pretty(&ConfigFile::default())
                .map_err(|error| error.to_string())?;
            match fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&self.path)
            {
                Ok(mut file) => file
                    .write_all(text.as_bytes())
                    .map_err(|error| error.to_string()),
                Err(error) if error.kind() == ErrorKind::AlreadyExists => Ok(()),
                Err(error) => Err(error.to_string()),
            }
        };
        create().map_err(|error| format!("create config {}: {error}", self.path.display()))
    }

    pub fn index_fragment(&self) -> Result<usize, String> {
        Ok(self.load()?.bpann.index_fragment)
    }

    pub fn index_max(&self) -> Result<usize, String> {
        Ok(self.load()?.bpann.index_max)
    }

    pub fn search_fragment(&self) -> Result<usize, String> {
        Ok(self.load()?.bpann.search_fragment)
    }

    pub fn fragment_rows(&self) -> Result<usize, String> {
        Ok(self.load()?.bpann.fragment_rows)
    }

    pub fn search_limit(&self) -> Result<usize, String> {
        Ok(self.load()?.bpann.search_limit)
    }

    pub fn build_seed(&self) -> Result<Option<u64>, String> {
        Ok(self.load()?.bpann.build_seed)
    }

    pub fn soft_threshold(&self) -> Result<usize, String> {
        Ok(self.load()?.bpann.soft_threshold)
    }

    pub fn hard_threshold(&self) -> Result<usize, String> {
        Ok(self.load()?.bpann.resolved_threshold())
    }

    pub fn structured_limit(&self) -> Result<usize, String> {
        Ok(self.load()?.bpann.structured_limit)
    }

    pub fn search_width(&self) -> Result<usize, String> {
        Ok(self.load()?.bpann.search_width)
    }

    pub fn exhaustive_limit(&self) -> Result<usize, String> {
        Ok(self.load()?.bpann.exhaustive_limit)
    }

    pub fn skip_limit(&self) -> Result<usize, String> {
        Ok(self.load()?.bpann.skip_limit)
    }
}

/// Install BPANN tuning so the `bpann` crate reads values via [`Config`].
///
/// Snapshot once at install time. Reloading the TOML inside the provider (old
/// behavior) put disk I/O on every `current_tuning()` — including per-query
/// search mode selection — and dominated TuRBO ask time.
pub fn bpann_config() -> Result<(), String> {
    let cached = Config::new().load()?.bpann.to_tuning();
    bpann::set_provider(Box::new(move || cached));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn creates_missing() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("config.toml");
        assert!(!path.exists());
        let cfg = Config::with_path(&path);
        assert_eq!(cfg.index_fragment().unwrap(), 10_000);
        assert_eq!(cfg.soft_threshold().unwrap(), 250);
        assert_eq!(cfg.hard_threshold().unwrap(), 3000);
        assert_eq!(cfg.structured_limit().unwrap(), 1024);
        assert_eq!(cfg.search_width().unwrap(), 1);
        assert_eq!(cfg.exhaustive_limit().unwrap(), 2500);
        assert_eq!(cfg.skip_limit().unwrap(), 150_000);
        assert!(path.exists());
        let text = fs::read_to_string(&path).unwrap();
        assert!(text.contains("index_fragment"));
        assert!(text.contains("soft_threshold"));
        assert!(text.contains("hard_threshold"));
        assert!(text.contains("structured_limit"));
        assert!(text.contains("search_width"));
        assert!(text.contains("exhaustive_limit"));
        assert!(text.contains("skip_limit"));
        assert!(text.contains("search_limit"));
        assert!(text.contains("10000"));
        assert!(text.contains("soft_threshold = 250"));
        assert!(text.contains("hard_threshold = 3000"));
        assert!(text.contains("exhaustive_limit = 2500"));
        assert!(text.contains("skip_limit = 150000"));
        assert!(text.contains("search_limit = 1"));
    }

    #[test]
    fn accessors_file() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("config.toml");
        let cfg = Config::with_path(&path);
        assert_eq!(cfg.search_limit().unwrap(), 1);
        let mut file = ConfigFile::default();
        file.bpann.search_limit = 7;
        file.bpann.search_width = 4;
        file.bpann.soft_threshold = 2_000;
        file.bpann.hard_threshold = Some(4_000);
        file.bpann.structured_limit = 2_048;
        file.bpann.exhaustive_limit = 5_000;
        file.bpann.skip_limit = 200_000;
        cfg.save(&file).unwrap();
        assert_eq!(cfg.search_limit().unwrap(), 7);
        assert_eq!(cfg.search_width().unwrap(), 4);
        assert_eq!(cfg.soft_threshold().unwrap(), 2_000);
        assert_eq!(cfg.hard_threshold().unwrap(), 4_000);
        assert_eq!(cfg.structured_limit().unwrap(), 2_048);
        assert_eq!(cfg.exhaustive_limit().unwrap(), 5_000);
        assert_eq!(cfg.skip_limit().unwrap(), 200_000);
    }

    #[test]
    fn missing_soft() {
        // Q5: absent hard must not full-default-fallback when soft is elevated.
        // Resolve policy: hard = max(DEFAULT_HARD, soft). With DEFAULT_HARD=3000
        // and soft=2000, hard becomes 3000; soft stays 2000.
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("config.toml");
        fs::write(&path, "[bpann]\nsoft_threshold = 2000\nsearch_width = 1\n").unwrap();
        let cfg = Config::with_path(&path);
        assert_eq!(cfg.soft_threshold().unwrap(), 2_000);
        assert_eq!(cfg.hard_threshold().unwrap(), 3_000);
    }

    #[test]
    fn missing_soft2() {
        // When soft exceeds DEFAULT_HARD, absent hard resolves to soft.
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("config.toml");
        fs::write(&path, "[bpann]\nsoft_threshold = 5000\nsearch_width = 1\n").unwrap();
        let cfg = Config::with_path(&path);
        assert_eq!(cfg.soft_threshold().unwrap(), 5_000);
        assert_eq!(cfg.hard_threshold().unwrap(), 5_000);
    }

    #[test]
    fn missing_hard() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("config.toml");
        fs::write(&path, "[bpann]\nsoft_threshold = 250\nsearch_width = 1\n").unwrap();
        let cfg = Config::with_path(&path);
        assert_eq!(cfg.soft_threshold().unwrap(), 250);
        assert_eq!(cfg.hard_threshold().unwrap(), 3000);
    }

    #[test]
    fn rejects_other() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("config.toml");
        fs::write(
            &path,
            "[bpann]\nsoft_threshold = 2000\nhard_threshold = 1000\nsearch_width = 1\n",
        )
        .unwrap();
        let cfg = Config::with_path(&path);
        let error = cfg.load().unwrap_err();
        assert!(error.contains("hard_threshold"));
        assert!(error.contains(path.to_str().unwrap()));
    }

    #[test]
    fn save_soft() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("config.toml");
        let cfg = Config::with_path(&path);
        let mut file = ConfigFile::default();
        file.bpann.soft_threshold = 500;
        file.bpann.hard_threshold = Some(100);
        let err = cfg.save(&file).unwrap_err();
        assert!(err.contains("hard_threshold"));
    }

    #[test]
    fn save_parameters() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("config.toml");
        let cfg = Config::with_path(&path);
        let mut file = ConfigFile::default();
        file.bpann.search_width = 0;
        let err = cfg.save(&file).unwrap_err();
        assert!(err.contains("search_width"));

        let mut file = ConfigFile::default();
        file.bpann.exhaustive_limit = 0;
        let err = cfg.save(&file).unwrap_err();
        assert!(err.contains("exhaustive_limit"));

        let mut file = ConfigFile::default();
        file.bpann.exhaustive_limit = 100;
        file.bpann.skip_limit = 50;
        let err = cfg.save(&file).unwrap_err();
        assert!(err.contains("skip_limit"));
    }

    #[test]
    fn rejects_file() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("config.toml");
        fs::write(&path, "[bpann]\nindex_fragment = 0\nsearch_width = 1\n").unwrap();
        let cfg = Config::with_path(&path);
        let error = cfg.load().unwrap_err();
        assert!(error.contains("index_fragment"));
        assert!(error.contains(path.to_str().unwrap()));
        assert!(fs::read_to_string(&path).unwrap().contains("= 0"));
    }

    #[test]
    fn set_overrides() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("override.toml");
        set_path(Some(path.clone())).unwrap();
        let cfg = Config::new();
        assert_eq!(cfg.path(), path.as_path());
        assert_eq!(cfg.index_max().unwrap(), 32);
        set_path(None).unwrap();
    }

    #[test]
    fn bpann_config2() {
        assert!(BpannConfig::default().validate().is_ok());
    }

    #[test]
    fn ennx_home() {
        let expected = home_dir().join(".ennx").join("config.toml");
        assert_eq!(config_path(), expected);
        assert!(config_path().ends_with("config.toml"));
    }

    #[test]
    fn active_override() {
        assert_eq!(active_path(), config_path());
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("override.toml");
        set_path(Some(path.clone())).unwrap();
        assert_eq!(active_path(), path);
        set_path(None).unwrap();
        assert_eq!(active_path(), config_path());
    }

    #[test]
    fn accessors_seed() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("config.toml");
        let cfg = Config::with_path(&path);
        assert_eq!(cfg.search_fragment().unwrap(), 80_000);
        assert_eq!(cfg.fragment_rows().unwrap(), 15_000);
        assert_eq!(cfg.build_seed().unwrap(), None);

        let mut file = ConfigFile::default();
        file.bpann.search_fragment = 50_000;
        file.bpann.fragment_rows = 9_000;
        file.bpann.build_seed = Some(42);
        cfg.save(&file).unwrap();

        assert_eq!(cfg.search_fragment().unwrap(), 50_000);
        assert_eq!(cfg.fragment_rows().unwrap(), 9_000);
        assert_eq!(cfg.build_seed().unwrap(), Some(42));
    }

    #[test]
    fn install_overrides() {
        // Avoid set_path here: it is process-global and races other tests.
        // Mirror bpann_config by loading via with_path → to_tuning.
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("config.toml");
        let cfg = Config::with_path(&path);
        let mut file = ConfigFile::default();
        file.bpann.exhaustive_limit = 7_000;
        file.bpann.skip_limit = 90_000;
        cfg.save(&file).unwrap();

        let t = cfg.load().unwrap().bpann.to_tuning();
        assert_eq!(t.exhaustive_limit, 7_000);
        assert_eq!(t.skip_limit, 90_000);
        assert!(!t.rows_edges(5_000));
        assert!(t.rows_edges(8_000));
        assert!(!t.rows_edges(100_000));
    }
}
