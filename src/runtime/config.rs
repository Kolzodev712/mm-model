//! Versioned experiment configuration loaded from JSON.
//!
//! `AppConfig::load` reads a file. `AppConfig::parse` does the same work from
//! text so tests do not touch process environment. `validate` checks values
//! after deserialization. Unknown JSON fields are rejected.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::Deserialize;

use crate::model::fair_value::FairValueMethod;
use crate::model::market::MarketOutcomeDefinition;

/// Schema version this build accepts. Older and newer files are rejected.
pub const CONFIG_VERSION: u32 = 1;

/// Path used when `MM_CONFIG` is unset.
pub const DEFAULT_CONFIG_PATH: &str = "configs/default.json";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunMode {
    Collect,
    Train,
    LiveInference,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Exchange {
    BinanceSpot,
}

impl Exchange {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::BinanceSpot => "binance",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Symbol(String);

impl Symbol {
    pub fn new(value: impl Into<String>) -> Result<Self, ConfigError> {
        let value = value.into();
        if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_alphanumeric()) {
            return Err(ConfigError::Symbol);
        }
        Ok(Self(value))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MarketConfig {
    pub exchange: Exchange,
    pub symbol: Symbol,
    pub depth: u16,
    pub update_ms: u64,
    pub fair_value: FairValueMethod,
}

/// Which probability model a training run fits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TrainingTarget {
    /// `P(fill within horizon_ns | market state, quote context)`.
    QuoteFill { horizon_ns: i64 },
    /// A public-book outcome. No quote history is required.
    MarketOutcome(MarketOutcomeDefinition),
}

impl TrainingTarget {
    #[must_use]
    pub const fn horizon_ns(self) -> i64 {
        match self {
            Self::QuoteFill { horizon_ns } => horizon_ns,
            Self::MarketOutcome(definition) => definition.horizon_ns(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidencePolicy {
    /// Percent of the latest decision rows held out.
    /// Earlier rows are training candidates. A candidate whose horizon snapshot
    /// reaches the first holdout decision is removed from fitting.
    pub holdout_percent: u8,
    pub min_train_examples: usize,
    pub min_train_positives: usize,
    pub min_train_negatives: usize,
    pub min_holdout_examples: usize,
    pub min_holdout_positives: usize,
    pub min_holdout_negatives: usize,
}

impl Default for EvidencePolicy {
    fn default() -> Self {
        Self {
            holdout_percent: 20,
            min_train_examples: 8,
            min_train_positives: 3,
            min_train_negatives: 3,
            min_holdout_examples: 4,
            min_holdout_positives: 2,
            min_holdout_negatives: 2,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct QuoteProbe {
    pub distance_bps: f64,
    pub quantity: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LiveModel {
    pub artifact_path: PathBuf,
    pub probes: Vec<QuoteProbe>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CollectTask {
    pub run_for: Option<Duration>,
    pub flush_every: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrainTask {
    pub target: TrainingTarget,
    pub window_from_ns: i64,
    pub window_to_ns: i64,
    pub quotes_path: Option<PathBuf>,
    pub artifact_path: PathBuf,
    pub evidence: EvidencePolicy,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LiveInferenceTask {
    pub run_for: Option<Duration>,
    pub report_every: Duration,
    pub models: Vec<LiveModel>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TaskConfig {
    Collect(CollectTask),
    Train(TrainTask),
    LiveInference(LiveInferenceTask),
}

#[derive(Debug, Clone, PartialEq)]
pub struct AppConfig {
    pub version: u32,
    pub data_root: PathBuf,
    pub market: MarketConfig,
    pub task: TaskConfig,
}

impl AppConfig {
    /// Reads JSON from `path`. `data_root_override` replaces the file's data root
    /// before relative paths are joined. Absolute paths are left unchanged.
    pub fn load(path: &Path, data_root_override: Option<&Path>) -> Result<Self, ConfigError> {
        let text = fs::read_to_string(path).map_err(|err| ConfigError::Io(err.to_string()))?;
        Self::parse(&text, data_root_override)
    }

    /// Deserializes JSON text and resolves data paths. Does not read the environment.
    pub fn parse(text: &str, data_root_override: Option<&Path>) -> Result<Self, ConfigError> {
        let file: FileConfig =
            serde_json::from_str(text).map_err(|err| ConfigError::Json(err.to_string()))?;
        if file.version != CONFIG_VERSION {
            return Err(ConfigError::Version);
        }
        let data_root = data_root_override
            .map(Path::to_path_buf)
            .unwrap_or(file.data_root);
        let symbol = Symbol::new(file.market.symbol)?;
        let task = match file.task {
            TaskFile::Collect {
                run_for_seconds,
                flush_every_records,
            } => TaskConfig::Collect(CollectTask {
                run_for: seconds(run_for_seconds)?,
                flush_every: flush_every_records,
            }),
            TaskFile::Train {
                target,
                history,
                quote_history_path,
                artifact_path,
                validation,
            } => {
                let target = match target {
                    TargetFile::QuoteFill { horizon_ms } => TrainingTarget::QuoteFill {
                        horizon_ns: ms_to_ns(horizon_ms)?,
                    },
                    TargetFile::MarketUpMove {
                        horizon_ms,
                        threshold_bps,
                    } => TrainingTarget::MarketOutcome(
                        MarketOutcomeDefinition::new_up_move(ms_to_ns(horizon_ms)?, threshold_bps)
                            .map_err(|_| ConfigError::Threshold)?,
                    ),
                };
                if let Some(path) = quote_history_path.as_ref() {
                    if path.as_os_str().is_empty() {
                        return Err(ConfigError::Path);
                    }
                }
                if artifact_path.as_os_str().is_empty() {
                    return Err(ConfigError::Path);
                }
                TaskConfig::Train(TrainTask {
                    target,
                    window_from_ns: history.from_unix_ns,
                    window_to_ns: history.to_unix_ns,
                    quotes_path: quote_history_path.map(|path| resolve_under(&data_root, &path)),
                    artifact_path: resolve_under(&data_root, &artifact_path),
                    evidence: validation,
                })
            }
            TaskFile::LiveInference {
                run_for_seconds,
                report_every_seconds,
                models,
            } => {
                let mut resolved = Vec::with_capacity(models.len());
                for model in models {
                    if model.artifact.as_os_str().is_empty() {
                        return Err(ConfigError::Path);
                    }
                    resolved.push(LiveModel {
                        artifact_path: resolve_under(&data_root, &model.artifact),
                        probes: model
                            .quote_probes
                            .into_iter()
                            .map(|probe| QuoteProbe {
                                distance_bps: probe.distance_bps,
                                quantity: probe.quantity,
                            })
                            .collect(),
                    });
                }
                TaskConfig::LiveInference(LiveInferenceTask {
                    run_for: seconds(run_for_seconds)?,
                    report_every: Duration::from_secs(report_every_seconds),
                    models: resolved,
                })
            }
        };
        Ok(Self {
            version: file.version,
            data_root,
            market: MarketConfig {
                exchange: file.market.exchange,
                symbol,
                depth: file.market.depth,
                update_ms: file.market.update_ms,
                fair_value: file.market.fair_value,
            },
            task,
        })
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.version != CONFIG_VERSION {
            return Err(ConfigError::Version);
        }
        if !matches!(self.market.depth, 5 | 10 | 20) || !matches!(self.market.update_ms, 100 | 1000)
        {
            return Err(ConfigError::Market);
        }
        match &self.task {
            TaskConfig::Collect(task) => {
                if task.flush_every == 0 {
                    return Err(ConfigError::Flush);
                }
                if task.run_for.is_some_and(|limit| limit.is_zero()) {
                    return Err(ConfigError::Duration);
                }
            }
            TaskConfig::Train(task) => {
                if task.window_from_ns >= task.window_to_ns {
                    return Err(ConfigError::Window);
                }
                if task.target.horizon_ns() <= 0 {
                    return Err(ConfigError::Horizon);
                }
                if let TrainingTarget::MarketOutcome(definition) = task.target {
                    if !definition.threshold_bps().is_finite() || definition.threshold_bps() <= 0.0
                    {
                        return Err(ConfigError::Threshold);
                    }
                    if task.quotes_path.is_some() {
                        return Err(ConfigError::QuoteHistory);
                    }
                }
                if matches!(task.target, TrainingTarget::QuoteFill { .. })
                    && task.quotes_path.is_none()
                {
                    return Err(ConfigError::QuoteHistory);
                }
                if task.artifact_path.as_os_str().is_empty() {
                    return Err(ConfigError::Path);
                }
                validate_evidence(task.evidence)?;
            }
            TaskConfig::LiveInference(task) => {
                if task.report_every.is_zero() || task.run_for.is_some_and(|limit| limit.is_zero())
                {
                    return Err(ConfigError::Duration);
                }
                if task.models.is_empty() {
                    return Err(ConfigError::Inference);
                }
                for model in &task.models {
                    if model.artifact_path.as_os_str().is_empty() {
                        return Err(ConfigError::Path);
                    }
                    if model.probes.len() > crate::model::probe::MAX_PROBES {
                        return Err(ConfigError::Grid);
                    }
                    for probe in &model.probes {
                        if !probe.distance_bps.is_finite() || probe.distance_bps < 0.0 {
                            return Err(ConfigError::Grid);
                        }
                        if !(probe.quantity.is_finite() && probe.quantity > 0.0) {
                            return Err(ConfigError::Quantity);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[must_use]
    pub fn mode(&self) -> RunMode {
        match self.task {
            TaskConfig::Collect(_) => RunMode::Collect,
            TaskConfig::Train(_) => RunMode::Train,
            TaskConfig::LiveInference(_) => RunMode::LiveInference,
        }
    }
}

fn validate_evidence(evidence: EvidencePolicy) -> Result<(), ConfigError> {
    if !(1..50).contains(&evidence.holdout_percent)
        || evidence.min_train_examples == 0
        || evidence.min_train_positives == 0
        || evidence.min_train_negatives == 0
        || evidence.min_holdout_examples == 0
        || evidence.min_holdout_positives == 0
        || evidence.min_holdout_negatives == 0
    {
        return Err(ConfigError::Evidence);
    }
    Ok(())
}

fn seconds(value: Option<u64>) -> Result<Option<Duration>, ConfigError> {
    match value {
        None => Ok(None),
        Some(0) => Err(ConfigError::Duration),
        Some(seconds) => Ok(Some(Duration::from_secs(seconds))),
    }
}

fn ms_to_ns(ms: u64) -> Result<i64, ConfigError> {
    if ms == 0 {
        return Err(ConfigError::Horizon);
    }
    let ns = ms.checked_mul(1_000_000).ok_or(ConfigError::Horizon)?;
    i64::try_from(ns).map_err(|_| ConfigError::Horizon)
}

/// Joins `path` onto `root` when `path` is relative.
#[must_use]
pub fn resolve_under(root: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileConfig {
    version: u32,
    data_root: PathBuf,
    market: MarketFile,
    task: TaskFile,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct MarketFile {
    exchange: Exchange,
    symbol: String,
    depth: u16,
    update_ms: u64,
    fair_value: FairValueMethod,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
enum TaskFile {
    Collect {
        #[serde(default)]
        run_for_seconds: Option<u64>,
        flush_every_records: u32,
    },
    Train {
        target: TargetFile,
        history: HistoryFile,
        #[serde(default)]
        quote_history_path: Option<PathBuf>,
        artifact_path: PathBuf,
        validation: EvidencePolicy,
    },
    LiveInference {
        #[serde(default)]
        run_for_seconds: Option<u64>,
        report_every_seconds: u64,
        models: Vec<LiveModelFile>,
    },
}

#[derive(Debug, Deserialize)]
#[serde(tag = "target", rename_all = "snake_case", deny_unknown_fields)]
enum TargetFile {
    QuoteFill { horizon_ms: u64 },
    MarketUpMove { horizon_ms: u64, threshold_bps: f64 },
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct HistoryFile {
    from_unix_ns: i64,
    to_unix_ns: i64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LiveModelFile {
    artifact: PathBuf,
    #[serde(default)]
    quote_probes: Vec<ProbeFile>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProbeFile {
    distance_bps: f64,
    quantity: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigError {
    Io(String),
    Json(String),
    Version,
    Symbol,
    Window,
    Horizon,
    Threshold,
    Flush,
    Duration,
    Market,
    Quantity,
    Grid,
    Evidence,
    Inference,
    Path,
    QuoteHistory,
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(err) => write!(f, "configuration file could not be read: {err}"),
            Self::Json(err) => write!(f, "configuration JSON was rejected: {err}"),
            Self::Version => write!(f, "configuration version must be {CONFIG_VERSION}"),
            Self::Symbol => f.write_str("symbol must be non-empty ASCII letters or digits"),
            Self::Window => f.write_str("training window must run forward"),
            Self::Horizon => f.write_str("evaluation horizon must be positive"),
            Self::Threshold => {
                f.write_str("up-move threshold must be finite and strictly positive")
            }
            Self::Flush => f.write_str("flush interval must be positive"),
            Self::Duration => f.write_str("a supplied duration must be positive; omit it or use null to run until a signal"),
            Self::Market => {
                f.write_str("depth must be 5, 10, or 20 and the update interval 100 or 1000 ms")
            }
            Self::Quantity => f.write_str("probe quantity must be finite and positive"),
            Self::Grid => f.write_str("probe distance must be a finite, non-negative number of basis points"),
            Self::Evidence => f.write_str(
                "evidence policy thresholds must be positive and holdout under 50 percent",
            ),
            Self::Inference => f.write_str("live inference needs at least one model artifact"),
            Self::Path => f.write_str("a configured path must be non-empty"),
            Self::QuoteHistory => f.write_str(
                "quote-fill training requires quote_history_path; market-outcome training must not set it",
            ),
        }
    }
}

impl std::error::Error for ConfigError {}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLES: &[&str] = &[
        include_str!("../../configs/default.json"),
        include_str!("../../configs/collect-btcusdt-2h.json"),
        include_str!("../../configs/collect-btcusdt-continuous.json"),
        include_str!("../../configs/train-market-up-1bp-1s.json"),
        include_str!("../../configs/train-market-up-2bp-5s.json"),
        include_str!("../../configs/train-quote-fill-1s.json"),
        include_str!("../../configs/live-inference-market.json"),
        include_str!("../../configs/live-inference-quote-fill.json"),
        include_str!("../../configs/live-inference-both.json"),
    ];

    #[test]
    fn every_checked_in_sample_parses_and_validates() {
        for sample in SAMPLES {
            AppConfig::parse(sample, None).unwrap().validate().unwrap();
        }
    }

    #[test]
    fn unknown_fields_and_old_versions_are_rejected() {
        let unknown = r#"{
            "version": 1,
            "data_root": "./data",
            "horizn_ms": 1000,
            "market": {
                "exchange": "binance_spot",
                "symbol": "BTCUSDT",
                "depth": 10,
                "update_ms": 100,
                "fair_value": "microprice"
            },
            "task": {"mode": "collect", "flush_every_records": 10}
        }"#;
        assert!(matches!(
            AppConfig::parse(unknown, None),
            Err(ConfigError::Json(_))
        ));
        let version = r#"{
            "version": 2,
            "data_root": "./data",
            "market": {
                "exchange": "binance_spot",
                "symbol": "BTCUSDT",
                "depth": 10,
                "update_ms": 100,
                "fair_value": "microprice"
            },
            "task": {"mode": "collect", "flush_every_records": 10}
        }"#;
        assert_eq!(
            AppConfig::parse(version, None).unwrap_err(),
            ConfigError::Version
        );
    }

    #[test]
    fn quote_fill_requires_quote_history_and_market_training_does_not() {
        let market = include_str!("../../configs/train-market-up-1bp-1s.json");
        let parsed = AppConfig::parse(market, None).unwrap();
        parsed.validate().unwrap();
        match parsed.task {
            TaskConfig::Train(task) => assert!(task.quotes_path.is_none()),
            _ => panic!("expected a training task"),
        }
        let missing = r#"{
            "version": 1,
            "data_root": "./data",
            "market": {
                "exchange": "binance_spot",
                "symbol": "BTCUSDT",
                "depth": 10,
                "update_ms": 100,
                "fair_value": "microprice"
            },
            "task": {
                "mode": "train",
                "target": {"target": "quote_fill", "horizon_ms": 1000},
                "history": {"from_unix_ns": 0, "to_unix_ns": 10},
                "artifact_path": "models/quote.json",
                "validation": {
                    "holdout_percent": 20,
                    "min_train_examples": 8,
                    "min_train_positives": 3,
                    "min_train_negatives": 3,
                    "min_holdout_examples": 4,
                    "min_holdout_positives": 2,
                    "min_holdout_negatives": 2
                }
            }
        }"#;
        let parsed = AppConfig::parse(missing, None).unwrap();
        assert_eq!(parsed.validate(), Err(ConfigError::QuoteHistory));
    }

    #[test]
    fn live_inference_without_models_and_zero_duration_fail() {
        let empty = r#"{
            "version": 1,
            "data_root": "./data",
            "market": {
                "exchange": "binance_spot",
                "symbol": "BTCUSDT",
                "depth": 10,
                "update_ms": 100,
                "fair_value": "microprice"
            },
            "task": {
                "mode": "live_inference",
                "run_for_seconds": 30,
                "report_every_seconds": 5,
                "models": []
            }
        }"#;
        let parsed = AppConfig::parse(empty, None).unwrap();
        assert_eq!(parsed.validate(), Err(ConfigError::Inference));
        let zero = r#"{
            "version": 1,
            "data_root": "./data",
            "market": {
                "exchange": "binance_spot",
                "symbol": "BTCUSDT",
                "depth": 10,
                "update_ms": 100,
                "fair_value": "microprice"
            },
            "task": {"mode": "collect", "run_for_seconds": 0, "flush_every_records": 10}
        }"#;
        assert_eq!(
            AppConfig::parse(zero, None).unwrap_err(),
            ConfigError::Duration
        );
    }

    #[test]
    fn relative_paths_join_the_data_root_and_absolute_paths_do_not() {
        let text = r#"{
            "version": 1,
            "data_root": "./data",
            "market": {
                "exchange": "binance_spot",
                "symbol": "BTCUSDT",
                "depth": 10,
                "update_ms": 100,
                "fair_value": "microprice"
            },
            "task": {
                "mode": "train",
                "target": {"target": "quote_fill", "horizon_ms": 1000},
                "history": {"from_unix_ns": 0, "to_unix_ns": 10},
                "quote_history_path": "quotes/binance/BTCUSDT.parquet",
                "artifact_path": "/tmp/quote.json",
                "validation": {
                    "holdout_percent": 20,
                    "min_train_examples": 8,
                    "min_train_positives": 3,
                    "min_train_negatives": 3,
                    "min_holdout_examples": 4,
                    "min_holdout_positives": 2,
                    "min_holdout_negatives": 2
                }
            }
        }"#;
        let overridden = AppConfig::parse(text, Some(Path::new("/data"))).unwrap();
        assert_eq!(overridden.data_root, PathBuf::from("/data"));
        match overridden.task {
            TaskConfig::Train(task) => {
                assert_eq!(
                    task.quotes_path.unwrap(),
                    PathBuf::from("/data/quotes/binance/BTCUSDT.parquet")
                );
                assert_eq!(task.artifact_path, PathBuf::from("/tmp/quote.json"));
                assert!(matches!(
                    task.target,
                    TrainingTarget::QuoteFill {
                        horizon_ns: 1_000_000_000
                    }
                ));
            }
            _ => panic!("expected a training task"),
        }
        let local = AppConfig::parse(text, None).unwrap();
        match local.task {
            TaskConfig::Train(task) => {
                assert_eq!(
                    task.quotes_path.unwrap(),
                    PathBuf::from("./data/quotes/binance/BTCUSDT.parquet")
                );
            }
            _ => panic!("expected a training task"),
        }
    }
}
