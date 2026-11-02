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
    /// Raw partition length. This build accepts one UTC hour only.
    pub partition_duration: Duration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainingTrigger {
    /// Fit after this many newly published partitions.
    EveryPublishedPartitions(u32),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrainingConfig {
    pub horizon_ns: i64,
    pub window_from_ns: i64,
    pub window_to_ns: i64,
    pub trigger: TrainingTrigger,
    pub artifact_path: PathBuf,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ShadowConfig {
    pub quantity: f64,
    pub offsets_bps: String,
    pub demo_model: bool,
    pub model_path: PathBuf,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigError {
    Symbol,
    Window,
    Horizon,
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::Symbol => "symbol must be non-empty ASCII letters or digits",
            Self::Window => "training window must run forward",
            Self::Horizon => "evaluation horizon must be positive",
        };
        f.write_str(message)
    }
}

impl std::error::Error for ConfigError {}
