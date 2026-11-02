//! Typed experiment configuration. The binary fills this in one place.

use std::path::PathBuf;
use std::time::Duration;

use crate::model::fair_value::FairValueMethod;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunMode {
    Collect,
    Train,
    CollectAndTrain,
    Shadow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StorageConfig {
    pub root: PathBuf,
    pub flush_every: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainingTrigger {
    /// Fit after this many newly published partitions.
    EveryPublishedPartitions(u32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EvidencePolicy {
    /// Percent of the latest decision rows held out.
    /// Earlier rows are training candidates. A candidate whose horizon snapshot
    /// reaches the first holdout decision is removed from fitting.
    pub holdout_percent: u8,
    pub min_train_examples: usize,
    pub min_train_fills: usize,
    pub min_train_misses: usize,
    pub min_holdout_examples: usize,
    pub min_holdout_fills: usize,
    pub min_holdout_misses: usize,
}

impl Default for EvidencePolicy {
    fn default() -> Self {
        Self {
            holdout_percent: 20,
            min_train_examples: 8,
            min_train_fills: 3,
            min_train_misses: 3,
            min_holdout_examples: 4,
            min_holdout_fills: 2,
            min_holdout_misses: 2,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrainingConfig {
    pub horizon_ns: i64,
    pub window_from_ns: i64,
    pub window_to_ns: i64,
    pub trigger: TrainingTrigger,
    pub artifact_path: PathBuf,
    pub quotes_path: PathBuf,
    pub evidence: EvidencePolicy,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ShadowConfig {
    pub quantity: f64,
    pub offsets_bps: String,
    pub demo_model: bool,
    pub model_path: PathBuf,
    pub report_every: Duration,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AppConfig {
    pub mode: RunMode,
    pub market: MarketConfig,
    pub storage: StorageConfig,
    pub training: TrainingConfig,
    pub shadow: ShadowConfig,
    /// Bound for collect and shadow loops. `None` runs until the source closes.
    pub run_for: Option<Duration>,
}

impl AppConfig {
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.training.horizon_ns <= 0 {
            return Err(ConfigError::Horizon);
        }
        if self.training.window_from_ns >= self.training.window_to_ns {
            return Err(ConfigError::Window);
        }
        if matches!(
            self.training.trigger,
            TrainingTrigger::EveryPublishedPartitions(0)
        ) {
            return Err(ConfigError::Trigger);
        }
        if self.storage.flush_every == 0 {
            return Err(ConfigError::Flush);
        }
        if self.run_for.is_some_and(|limit| limit.is_zero()) {
            return Err(ConfigError::Duration);
        }
        if !matches!(self.market.depth, 5 | 10 | 20) || !matches!(self.market.update_ms, 100 | 1000)
        {
            return Err(ConfigError::Market);
        }
        if !(self.shadow.quantity.is_finite() && self.shadow.quantity > 0.0) {
            return Err(ConfigError::Quantity);
        }
        if self.shadow.report_every.is_zero() {
            return Err(ConfigError::Duration);
        }
        crate::model::candidate::BpsGrid::parse(&self.shadow.offsets_bps, self.shadow.quantity)
            .map_err(|_| ConfigError::Grid)?;
        let evidence = self.training.evidence;
        if !(1..50).contains(&evidence.holdout_percent)
            || evidence.min_train_examples == 0
            || evidence.min_train_fills == 0
            || evidence.min_train_misses == 0
            || evidence.min_holdout_examples == 0
            || evidence.min_holdout_fills == 0
            || evidence.min_holdout_misses == 0
        {
            return Err(ConfigError::Evidence);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigError {
    Symbol,
    Window,
    Horizon,
    Trigger,
    Flush,
    Duration,
    Market,
    Quantity,
    Grid,
    Evidence,
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::Symbol => "symbol must be non-empty ASCII letters or digits",
            Self::Window => "training window must run forward",
            Self::Horizon => "evaluation horizon must be positive",
            Self::Trigger => "training trigger count must be positive",
            Self::Flush => "flush interval must be positive",
            Self::Duration => "durations must be positive",
            Self::Market => "depth must be 5, 10, or 20 and the update interval 100 or 1000 ms",
            Self::Quantity => "shadow quantity must be positive",
            Self::Grid => "candidate grid must be a list of positive basis-point offsets",
            Self::Evidence => {
                "evidence policy thresholds must be positive and holdout under 50 percent"
            }
        };
        f.write_str(message)
    }
}

impl std::error::Error for ConfigError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> AppConfig {
        AppConfig {
            mode: RunMode::Collect,
            market: MarketConfig {
                exchange: Exchange::BinanceSpot,
                symbol: Symbol::new("BTCUSDT").unwrap(),
                depth: 10,
                update_ms: 100,
                fair_value: FairValueMethod::Microprice,
            },
            storage: StorageConfig {
                root: PathBuf::from("data"),
                flush_every: 10,
            },
            training: TrainingConfig {
                horizon_ns: 1_000,
                window_from_ns: 0,
                window_to_ns: 10,
                trigger: TrainingTrigger::EveryPublishedPartitions(1),
                artifact_path: PathBuf::from("model.json"),
                quotes_path: PathBuf::from("quotes.parquet"),
                evidence: EvidencePolicy::default(),
            },
            shadow: ShadowConfig {
                quantity: 1.0,
                offsets_bps: "1,5".into(),
                demo_model: true,
                model_path: PathBuf::from("model.json"),
                report_every: Duration::from_secs(1),
            },
            run_for: None,
        }
    }

    #[test]
    fn invalid_trigger_flush_horizon_and_window_fail() {
        let mut config = sample();
        config.validate().unwrap();
        config.training.trigger = TrainingTrigger::EveryPublishedPartitions(0);
        assert_eq!(config.validate(), Err(ConfigError::Trigger));
        config = sample();
        config.storage.flush_every = 0;
        assert_eq!(config.validate(), Err(ConfigError::Flush));
        config = sample();
        config.training.horizon_ns = 0;
        assert_eq!(config.validate(), Err(ConfigError::Horizon));
        config = sample();
        config.training.window_from_ns = 10;
        config.training.window_to_ns = 10;
        assert_eq!(config.validate(), Err(ConfigError::Window));
    }
}
