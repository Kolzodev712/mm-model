//! Composition root. The experiment is a JSON file, not source edits.
//!
//! This process reads public market data and writes probabilities. It does
//! not submit orders or choose a quote.

#![deny(unsafe_code)]

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use mm_model::data::market::binance::client::BinanceBookClient;
use mm_model::data::market::binance::parser::{BinanceBookConfig, BINANCE_SPOT_WS};
use mm_model::model::artifact::{ModelArtifact, ModelFamily};
use mm_model::model::feature::market::MARKET_FEATURE_COUNT;
use mm_model::model::feature::v1::FEATURE_COUNT;
use mm_model::model::fill::LogisticRegression;
use mm_model::model::market::MarketOutcomeDefinition;
use mm_model::runtime::clock::SystemClock;
use mm_model::runtime::config::{AppConfig, RunMode, TaskConfig, DEFAULT_CONFIG_PATH};
use mm_model::runtime::inference::run_live;
use mm_model::runtime::orchestrator::run_source;
use mm_model::runtime::reconnect::ReconnectPolicy;

fn main() -> ExitCode {
    match execute() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("mm_model: {err}");
            ExitCode::FAILURE
        }
    }
}

fn execute() -> Result<(), Box<dyn std::error::Error>> {
    let config_path = std::env::var_os("MM_CONFIG")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_CONFIG_PATH));
    let data_root = std::env::var_os("MM_DATA_ROOT").map(PathBuf::from);
    let config = AppConfig::load(&config_path, data_root.as_deref())?;
    config.validate()?;
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .with_target(false)
        .init();
    let stop = Arc::new(AtomicBool::new(false));
    signal_hook::flag::register(signal_hook::consts::SIGINT, Arc::clone(&stop))?;
    signal_hook::flag::register(signal_hook::consts::SIGTERM, Arc::clone(&stop))?;
    let should_stop = || stop.load(Ordering::Relaxed);
    let market = BinanceBookConfig::new(
        BINANCE_SPOT_WS,
        config.market.symbol.as_str(),
        config.market.depth,
        config.market.update_ms,
    )?;
    let policy = ReconnectPolicy::new(8, Duration::from_millis(250), Duration::from_secs(5))
        .map_err(|err| err.to_string())?;
    match config.mode() {
        RunMode::Train => {
            let mut idle = mm_model::data::market::source::SliceSource::new(Vec::new());
            let outcomes = run_source(&config, &mut idle, should_stop)?;
            tracing::info!(results = outcomes.len(), "train mode finished");
        }
        RunMode::LiveInference => {
            let loaded = load_live_models(&config)?;
            let mut client = BinanceBookClient::connect(&market, SystemClock, policy)?;
            let stats = run_live(
                &config,
                &mut client,
                loaded.quote.as_ref(),
                loaded.market.as_ref(),
                loaded.definition,
                loaded.quote_horizon_ns,
                &should_stop,
            )?;
            tracing::info!(
                books = stats.books_processed,
                "live inference finished; logged numbers are probabilities"
            );
        }
        RunMode::Collect => {
            let mut client = BinanceBookClient::connect(&market, SystemClock, policy)?;
            run_source(&config, &mut client, should_stop)?;
            tracing::info!(reconnects = client.reconnects(), "collection finished");
        }
    }
    Ok(())
}

struct LoadedModels {
    quote: Option<LogisticRegression<FEATURE_COUNT>>,
    market: Option<LogisticRegression<MARKET_FEATURE_COUNT>>,
    definition: Option<MarketOutcomeDefinition>,
    quote_horizon_ns: Option<i64>,
}

fn load_live_models(config: &AppConfig) -> Result<LoadedModels, Box<dyn std::error::Error>> {
    let TaskConfig::LiveInference(task) = &config.task else {
        return Err("live inference requires a live_inference task".into());
    };
    let mut loaded = LoadedModels {
        quote: None,
        market: None,
        definition: None,
        quote_horizon_ns: None,
    };
    for model in &task.models {
        let artifact = ModelArtifact::read_json(&model.artifact_path)?;
        match artifact.family {
            ModelFamily::QuoteFill => {
                if loaded.quote.is_some() {
                    return Err("live inference accepts one quote-fill artifact".into());
                }
                if model.probes.is_empty() {
                    return Err("a quote-fill artifact needs at least one quote probe".into());
                }
                artifact.validate_for_inference(
                    ModelFamily::QuoteFill,
                    config.market.fair_value,
                    artifact.training_horizon_ns,
                    None,
                )?;
                loaded.quote_horizon_ns = Some(artifact.training_horizon_ns);
                loaded.quote = Some(artifact.quote_model()?);
            }
            ModelFamily::MarketUpMove => {
                if loaded.market.is_some() {
                    return Err("live inference accepts one market-outcome artifact".into());
                }
                if !model.probes.is_empty() {
                    return Err(
                        "quote probes are only defined for a quote-fill artifact; a market-outcome artifact does not take them"
                            .into(),
                    );
                }
                let threshold = artifact
                    .threshold_bps
                    .ok_or("market artifact is missing its up-move threshold")?;
                artifact.validate_for_inference(
                    ModelFamily::MarketUpMove,
                    config.market.fair_value,
                    artifact.training_horizon_ns,
                    Some(threshold),
                )?;
                loaded.definition = Some(MarketOutcomeDefinition::new_up_move(
                    artifact.training_horizon_ns,
                    threshold,
                )?);
                loaded.market = Some(artifact.market_model()?);
            }
        }
    }
    Ok(loaded)
}
