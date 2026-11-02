//! Composition root. Change the experiment by editing [`configuration`].
//!
//! This process does not send orders.

#![deny(unsafe_code)]

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use mm_model::data::market::binance::client::BinanceBookClient;
use mm_model::data::market::binance::parser::{BinanceBookConfig, BINANCE_SPOT_WS};
use mm_model::model::fair_value::FairValueMethod;
use mm_model::runtime::clock::SystemClock;
use mm_model::runtime::config::{
    AppConfig, EvidencePolicy, Exchange, MarketConfig, RunMode, ShadowConfig, StorageConfig,
    Symbol, TrainingConfig, TrainingTrigger,
};
use mm_model::runtime::orchestrator::run_source;
use mm_model::runtime::reconnect::ReconnectPolicy;
use mm_model::runtime::shadow::run_live;
use mm_model::storage::quotes::quote_file;

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
    let config = configuration()?;
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
    match config.mode {
        RunMode::Train => {
            let mut idle = mm_model::data::market::source::SliceSource::new(Vec::new());
            let outcomes = run_source(&config, &mut idle, should_stop)?;
            tracing::info!(results = outcomes.len(), "train mode finished");
        }
        RunMode::Shadow => {
            if config.shadow.demo_model {
                eprintln!("DEMO MODEL — predictions are not trading-valid");
            }
            let (fill, pnl) = if config.shadow.demo_model {
                mm_model::data::synthetic::train_demo_models()?
            } else {
                let artifact =
                    mm_model::model::artifact::ModelArtifact::read_json(&config.shadow.model_path)?;
                artifact
                    .validate_for_shadow(config.market.fair_value, config.training.horizon_ns)?;
                artifact.models()?
            };
            let mut client = BinanceBookClient::connect(&market, SystemClock, policy)?;
            let stats = run_live(&config, &mut client, &fill, &pnl, &should_stop)?;
            tracing::info!(
                books = stats.books_processed,
                "shadow finished; predictions are not realized fills or P&L"
            );
        }
        RunMode::Collect | RunMode::CollectAndTrain => {
            let mut client = BinanceBookClient::connect(&market, SystemClock, policy)?;
            let outcomes = run_source(&config, &mut client, should_stop)?;
            tracing::info!(
                reconnects = client.reconnects(),
                training_results = outcomes.len(),
                "collection finished"
            );
        }
    }
    Ok(())
}

fn configuration() -> Result<AppConfig, mm_model::runtime::config::ConfigError> {
    let root = std::env::var_os("MM_DATA_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("./data"));
    let symbol = Symbol::new("BTCUSDT")?;
    Ok(AppConfig {
        // Edit this mode to change the experiment. No CLI parser is required.
        mode: RunMode::Shadow,
        market: MarketConfig {
            exchange: Exchange::BinanceSpot,
            symbol,
            depth: 10,
            update_ms: 100,
            fair_value: FairValueMethod::Microprice,
        },
        storage: StorageConfig {
            root: root.clone(),
            flush_every: 10,
        },
        training: TrainingConfig {
            horizon_ns: 60_000_000_000,
            window_from_ns: 0,
            window_to_ns: i64::MAX / 4,
            trigger: TrainingTrigger::EveryPublishedPartitions(1),
            artifact_path: root.join("models/current.json"),
            quotes_path: quote_file(&root, "binance", "BTCUSDT"),
            evidence: EvidencePolicy::default(),
        },
        shadow: ShadowConfig {
            quantity: 0.001,
            offsets_bps: "0.5,1,2,5".into(),
            demo_model: true,
            model_path: root.join("models/current.json"),
            report_every: Duration::from_secs(5),
        },
        run_for: Some(Duration::from_secs(30)),
    })
}
