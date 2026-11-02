//! Lifecycle only: rotate raw partitions, publish Parquet, maybe train.
//!
//! This reference orchestrator is sequential. Conversion of a closed hour
//! finishes before the next book is read, and training runs on that same
//! thread. The collector loop therefore pauses during those steps. The steps
//! stay separate functions so a later worker split does not move the math.

use crate::data::market::source::BookSource;
use crate::data::training::dataset::QuoteEvent;
use crate::domain::book::BookRecord;
use crate::model::fair_value::FairValueMethod;
use crate::runtime::collector::{CollectError, Collector};
use crate::runtime::config::{AppConfig, RunMode, TrainingTrigger};
use crate::runtime::converter::{publish_partition, ConvertError};
use crate::runtime::trainer::{train_history, TrainFailure, TrainOutcome, TrainingJob};
use crate::storage::manifest::{Catalog, ManifestError};
use crate::storage::parquet::{ParquetConfig, ParquetHistory};
use crate::storage::partition::PartitionId;

macro_rules! runtime_info {
    ($($arg:tt)*) => {{
        #[cfg(feature = "observe")]
        tracing::info!($($arg)*);
    }};
}

/// Lifecycle signals between collection, conversion, and training.
///
/// There is no event bus. The orchestrator records these as it runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeEvent {
    PartitionClosed(PartitionId),
    PartitionPublished(PartitionId),
    PartitionFailed(PartitionId),
    TrainingCompleted(String),
    TrainingRejected(String),
    TrainingSkipped,
    TrainingFailed,
}

#[derive(Debug)]
pub enum OrchestratorError {
    Collect(CollectError),
    Convert(ConvertError),
    Train(TrainFailure),
    Catalog(ManifestError),
    Source(String),
    Config,
}

impl std::fmt::Display for OrchestratorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Collect(err) => write!(f, "{err}"),
            Self::Convert(err) => write!(f, "{err}"),
            Self::Train(err) => write!(f, "{err}"),
            Self::Catalog(err) => write!(f, "{err}"),
            Self::Source(err) => write!(f, "{err}"),
            Self::Config => f.write_str("this build only supports one-hour UTC partitions"),
        }
    }
}

impl std::error::Error for OrchestratorError {}

impl From<CollectError> for OrchestratorError {
    fn from(value: CollectError) -> Self {
        Self::Collect(value)
    }
}

impl From<ConvertError> for OrchestratorError {
    fn from(value: ConvertError) -> Self {
        Self::Convert(value)
    }
}

impl From<TrainFailure> for OrchestratorError {
    fn from(value: TrainFailure) -> Self {
        Self::Train(value)
    }
}

impl From<ManifestError> for OrchestratorError {
    fn from(value: ManifestError) -> Self {
        Self::Catalog(value)
    }
}

/// Drives collection from an already-open source. `Train` ignores `source`.
pub fn run_source<S>(
    config: &AppConfig,
    source: &mut S,
    quotes: &[QuoteEvent],
    mut should_stop: impl FnMut() -> bool,
) -> Result<Vec<TrainOutcome>, OrchestratorError>
where
    S: BookSource,
    S::Error: std::fmt::Display,
{
    if config.storage.partition_duration != std::time::Duration::from_secs(60 * 60) {
        return Err(OrchestratorError::Config);
    }
    match config.mode {
        RunMode::Train => {
            let catalog = Catalog::open(&config.storage.root)?;
            Ok(vec![train_from(config, &catalog, quotes)?])
        }
        RunMode::Shadow => Ok(Vec::new()),
        RunMode::Collect | RunMode::CollectAndTrain => {
            run_collect(config, source, quotes, &mut should_stop)
        }
    }
}

fn run_collect<S>(
    config: &AppConfig,
    source: &mut S,
    quotes: &[QuoteEvent],
    should_stop: &mut impl FnMut() -> bool,
) -> Result<Vec<TrainOutcome>, OrchestratorError>
where
    S: BookSource,
    S::Error: std::fmt::Display,
{
    let started = std::time::Instant::now();
    let catalog = Catalog::open(&config.storage.root)?;
    let mut collector = Collector::new(
        catalog,
        config.market.exchange.as_str(),
        config.market.symbol.as_str(),
        config.storage.flush_every,
    );
    let mut since_train = 0u32;
    let mut trained = Vec::new();
    let mut events = Vec::new();
    loop {
        if should_stop()
            || config
                .run_for
                .is_some_and(|limit| started.elapsed() >= limit)
        {
            break;
        }
        let book = source
            .next_book()
            .map_err(|err| OrchestratorError::Source(err.to_string()))?;
        let Some(book) = book else { break };
        if let Some(closed) = collector.push(&book)? {
            since_train = after_close(
                config,
                collector.catalog_mut(),
                &closed,
                since_train,
                quotes,
                &mut trained,
                &mut events,
            )?;
        }
    }
    if let Some(closed) = collector.finish()? {
        let _ = after_close(
            config,
            collector.catalog_mut(),
            &closed,
            since_train,
            quotes,
            &mut trained,
            &mut events,
        )?;
    }
    runtime_info!(lifecycle_events = events.len(), "collection finished");
    Ok(trained)
}

fn after_close(
    config: &AppConfig,
    catalog: &mut Catalog,
    closed: &PartitionId,
    mut since_train: u32,
    quotes: &[QuoteEvent],
    trained: &mut Vec<TrainOutcome>,
    events: &mut Vec<RuntimeEvent>,
) -> Result<u32, OrchestratorError> {
    events.push(RuntimeEvent::PartitionClosed(closed.clone()));
    match publish_partition(catalog, closed, ParquetConfig::default()) {
        Ok(()) => {
            events.push(RuntimeEvent::PartitionPublished(closed.clone()));
            since_train += 1;
        }
        Err(
            err @ (ConvertError::StillOpen
            | ConvertError::Ambiguous
            | ConvertError::Missing
            | ConvertError::Manifest(_)),
        ) => return Err(err.into()),
        Err(err) => {
            events.push(RuntimeEvent::PartitionFailed(closed.clone()));
            let _ = err;
            return Ok(since_train);
        }
    }
    if config.mode == RunMode::Collect {
        return Ok(since_train);
    }
    let due = match config.training.trigger {
        TrainingTrigger::EveryPublishedPartitions(n) => since_train >= n && n > 0,
    };
    if due {
        match train_from(config, catalog, quotes) {
            Ok(TrainOutcome::Published { dataset_id, state }) => {
                events.push(RuntimeEvent::TrainingCompleted(dataset_id.clone()));
                trained.push(TrainOutcome::Published { dataset_id, state });
                since_train = 0;
            }
            Ok(TrainOutcome::Rejected { reason, state }) => {
                events.push(RuntimeEvent::TrainingRejected(reason.clone()));
                trained.push(TrainOutcome::Rejected { reason, state });
                since_train = 0;
            }
            Ok(TrainOutcome::SkippedNoLabels) => {
                events.push(RuntimeEvent::TrainingSkipped);
                trained.push(TrainOutcome::SkippedNoLabels);
                since_train = 0;
            }
            Err(_) => {
                events.push(RuntimeEvent::TrainingFailed);
            }
        }
    }
    Ok(since_train)
}

fn train_from(
    config: &AppConfig,
    catalog: &Catalog,
    quotes: &[QuoteEvent],
) -> Result<TrainOutcome, TrainFailure> {
    let history = ParquetHistory::new(catalog);
    train_history(
        &history,
        &TrainingJob {
            exchange: config.market.exchange.as_str(),
            symbol: config.market.symbol.as_str(),
            quotes,
            method: config.market.fair_value,
            horizon_ns: config.training.horizon_ns,
            window: crate::runtime::trainer::TrainingWindow {
                from_ns: config.training.window_from_ns,
                to_ns: config.training.window_to_ns,
            },
            artifact_path: &config.training.artifact_path,
        },
    )
}

/// Books already in memory. Used by tests and by a replay that is not live.
pub fn run_books(
    config: &AppConfig,
    books: &[BookRecord],
    quotes: &[QuoteEvent],
) -> Result<Vec<TrainOutcome>, OrchestratorError> {
    let mut source = crate::data::market::source::SliceSource::new(books.to_vec());
    run_source(config, &mut source, quotes, || false)
}

/// Shadow prediction does not write partitions or change weights.
pub fn shadow_books(
    method: FairValueMethod,
    books: &[BookRecord],
    grid: &crate::model::candidate::BpsGrid,
    fill: &crate::model::fill::LogisticRegression<{ crate::model::feature::v1::FEATURE_COUNT }>,
    pnl: &crate::model::pnl::LinearRegression<{ crate::model::feature::v1::FEATURE_COUNT }>,
) -> Result<usize, crate::runtime::shadow::EvalError> {
    let mut candidates = [crate::model::candidate::CandidateSlot {
        id: crate::domain::units::CandidateId::new(0),
        quote: crate::domain::market::CandidateQuote::new(1.0, 1.0).expect("placeholder"),
    }; crate::model::candidate::MAX_CANDIDATES];
    let mut predictions = [crate::model::prediction::QuotePrediction {
        fill_probability: crate::domain::units::Probability::new(0.0).expect("zero"),
        expected_pnl_if_filled: crate::domain::units::Pnl::new(0.0).expect("zero"),
        expected_value: 0.0,
    }; crate::model::candidate::MAX_CANDIDATES];
    let mut scored = 0usize;
    for book in books {
        crate::runtime::shadow::evaluate_book(
            book,
            method,
            grid,
            fill,
            pnl,
            &mut candidates,
            &mut predictions,
        )?;
        scored += 1;
    }
    Ok(scored)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::training::dataset::QuoteEvent;
    use crate::data::training::example::{FillFacts, LabelProvenance};
    use crate::domain::book::{BookRecord, BookStamp, Instrument, Level, BOOK_DEPTH};
    use crate::domain::market::CandidateQuote;
    use crate::model::fair_value::FairValueMethod;
    use crate::runtime::config::{
        AppConfig, Exchange, MarketConfig, RunMode, ShadowConfig, StorageConfig, Symbol,
        TrainingConfig, TrainingTrigger,
    };
    use crate::runtime::trainer::TrainOutcome;
    use crate::storage::manifest::Catalog;
    use crate::storage::partition::PartitionState;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn book(ns: i64, bid: f64, ask: f64) -> BookRecord {
        let mut bids = [Level::EMPTY; BOOK_DEPTH];
        let mut asks = [Level::EMPTY; BOOK_DEPTH];
        bids[0] = Level {
            price: bid,
            quantity: 2.0,
        };
        asks[0] = Level {
            price: ask,
            quantity: 1.0,
        };
        BookRecord::try_new(
            Instrument::new("binance", "BTCUSDT").unwrap(),
            BookStamp {
                local_receive_ns: ns,
                exchange_event_ns: None,
                update_id: Some(1),
            },
            bids,
            1,
            asks,
            1,
        )
        .unwrap()
    }

    fn config(root: &std::path::Path, mode: RunMode) -> AppConfig {
        AppConfig {
            mode,
            market: MarketConfig {
                exchange: Exchange::BinanceSpot,
                symbol: Symbol::new("BTCUSDT").unwrap(),
                depth: 10,
                update_ms: 100,
                fair_value: FairValueMethod::Microprice,
            },
            storage: StorageConfig {
                root: root.to_path_buf(),
                flush_every: 1,
                partition_duration: std::time::Duration::from_secs(60 * 60),
            },
            training: TrainingConfig {
                horizon_ns: 500,
                window_from_ns: 0,
                window_to_ns: i64::MAX / 4,
                trigger: TrainingTrigger::EveryPublishedPartitions(1),
                artifact_path: root.join("models/current.json"),
            },
            shadow: ShadowConfig {
                quantity: 1.0,
                offsets_bps: "1,5".into(),
                demo_model: true,
                model_path: root.join("models/current.json"),
            },
            run_for: None,
        }
    }

    fn temp() -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("mm-model-{nanos}"));
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn closed_partition_publishes_and_open_partition_is_not_trained() {
        let root = temp();
        let hour = 1_793_624_400_000_000_000i64;
        let early = book(hour + 1_000, 99.9, 100.1);
        let later = book(hour + 2_000, 99.9, 100.1);
        let next_hour = book(hour + 3_600_000_000_000, 100.0, 100.2);
        let quote = QuoteEvent::new(
            hour + 1_000,
            CandidateQuote::new(99.8, 1.0).unwrap(),
            LabelProvenance::Simulated,
            FillFacts::Filled {
                execution_price: 99.8,
                fees: 0.0,
            },
        );
        let outcomes = run_books(
            &config(&root, RunMode::CollectAndTrain),
            &[early, later, next_hour],
            &[quote],
        )
        .unwrap();
        assert!(matches!(
            outcomes.first(),
            Some(TrainOutcome::Published { .. })
        ));
        let catalog = Catalog::open(&root).unwrap();
        let published = catalog.published_overlapping("binance", "BTCUSDT", 0, i64::MAX / 4);
        assert_eq!(published.len(), 2);
        assert!(published
            .iter()
            .all(|row| row.state == PartitionState::Published));
        assert!(published
            .iter()
            .all(|row| { row.parquet_path.as_deref().unwrap().ends_with(".parquet") }));
        crate::runtime::converter::publish_partition(
            &mut Catalog::open(&root).unwrap(),
            &published[0].id,
            crate::storage::parquet::ParquetConfig::default(),
        )
        .unwrap();
        let again = Catalog::open(&root).unwrap();
        assert_eq!(
            again
                .published_overlapping("binance", "BTCUSDT", 0, i64::MAX / 4)
                .len(),
            2
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn open_partition_is_not_published_or_trained() {
        let root = temp();
        let hour = 1_793_624_400_000_000_000i64;
        let mut collector = crate::runtime::collector::Collector::new(
            Catalog::open(&root).unwrap(),
            "binance",
            "BTCUSDT",
            1,
        );
        assert!(collector.push(&book(hour, 99.9, 100.1)).unwrap().is_none());
        let id = crate::storage::partition::partition_for(
            "binance",
            "BTCUSDT",
            &book(hour, 99.9, 100.1),
        )
        .unwrap();
        let err = crate::runtime::converter::publish_partition(
            collector.catalog_mut(),
            &id,
            crate::storage::parquet::ParquetConfig::default(),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            crate::runtime::converter::ConvertError::StillOpen
        ));
        assert!(collector
            .catalog_mut()
            .published_overlapping("binance", "BTCUSDT", 0, i64::MAX / 4)
            .is_empty());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn train_mode_does_not_read_the_book_source() {
        struct ClosedSource;
        impl crate::data::market::source::BookSource for ClosedSource {
            type Error = &'static str;
            fn next_book(&mut self) -> Result<Option<BookRecord>, Self::Error> {
                panic!("train mode read a book source");
            }
        }
        let root = temp();
        let outcome = run_source(
            &config(&root, RunMode::Train),
            &mut ClosedSource,
            &[],
            || false,
        )
        .unwrap();
        assert_eq!(outcome, vec![TrainOutcome::SkippedNoLabels]);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn conversion_failure_keeps_the_partition_unpublished() {
        let root = temp();
        let hour = 1_793_624_400_000_000_000i64;
        let mut collector = crate::runtime::collector::Collector::new(
            Catalog::open(&root).unwrap(),
            "binance",
            "BTCUSDT",
            1,
        );
        collector.push(&book(hour, 99.9, 100.1)).unwrap();
        let id = collector.finish().unwrap().unwrap();
        let raw = id.raw_path(&root);
        std::fs::remove_file(&raw).unwrap();
        let err = crate::runtime::converter::publish_partition(
            collector.catalog_mut(),
            &id,
            crate::storage::parquet::ParquetConfig::default(),
        )
        .unwrap_err();
        assert!(matches!(err, crate::runtime::converter::ConvertError::Raw));
        assert_eq!(
            collector.catalog_mut().get(&id).unwrap().state,
            PartitionState::Failed
        );
        assert!(collector
            .catalog_mut()
            .published_overlapping("binance", "BTCUSDT", 0, i64::MAX / 4)
            .is_empty());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn training_failure_does_not_stop_collection() {
        let root = temp();
        let mut cfg = config(&root, RunMode::CollectAndTrain);
        let blocker = root.join("not-a-directory");
        std::fs::write(&blocker, b"file").unwrap();
        cfg.training.artifact_path = blocker.join("current.json");
        let hour = 1_793_624_400_000_000_000i64;
        let quote = QuoteEvent::new(
            hour + 1_000,
            CandidateQuote::new(99.8, 1.0).unwrap(),
            LabelProvenance::Simulated,
            FillFacts::Filled {
                execution_price: 99.8,
                fees: 0.0,
            },
        );
        let outcomes = run_books(
            &cfg,
            &[
                book(hour + 1_000, 99.9, 100.1),
                book(hour + 2_000, 99.9, 100.1),
                book(hour + 3_600_000_000_000, 100.0, 100.2),
            ],
            &[quote],
        )
        .unwrap();
        assert!(outcomes.is_empty());
        let catalog = Catalog::open(&root).unwrap();
        assert_eq!(
            catalog
                .published_overlapping("binance", "BTCUSDT", 0, i64::MAX / 4)
                .len(),
            2
        );
        assert_eq!(std::fs::read(&blocker).unwrap(), b"file");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[cfg(feature = "live")]
    #[test]
    #[ignore = "opens a public Binance websocket"]
    fn short_collect_smoke() {
        let root = temp();
        let mut cfg = config(&root, RunMode::Collect);
        cfg.run_for = Some(std::time::Duration::from_secs(5));
        let market = crate::data::market::binance::parser::BinanceBookConfig::new(
            crate::data::market::binance::parser::BINANCE_SPOT_WS,
            "BTCUSDT",
            10,
            100,
        )
        .unwrap();
        let policy = crate::runtime::reconnect::ReconnectPolicy::new(
            2,
            std::time::Duration::from_millis(200),
            std::time::Duration::from_secs(1),
        )
        .unwrap();
        let mut client = crate::data::market::binance::client::BinanceBookClient::connect(
            &market,
            crate::runtime::clock::SystemClock,
            policy,
        )
        .unwrap();
        let outcomes = run_source(&cfg, &mut client, &[], || false).unwrap();
        assert!(outcomes.is_empty());
        let catalog = Catalog::open(&root).unwrap();
        assert_eq!(
            catalog
                .published_overlapping("binance", "BTCUSDT", 0, i64::MAX / 4)
                .len(),
            1
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn collect_mode_does_not_train() {
        let root = temp();
        let hour = 1_793_624_400_000_000_000i64;
        let outcomes = run_books(
            &config(&root, RunMode::Collect),
            &[
                book(hour, 99.9, 100.1),
                book(hour + 3_600_000_000_000, 100.0, 100.2),
            ],
            &[],
        )
        .unwrap();
        assert!(outcomes.is_empty());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn shadow_mode_does_not_publish_partitions() {
        let root = temp();
        let outcomes = run_books(
            &config(&root, RunMode::Shadow),
            &[book(10, 99.9, 100.1)],
            &[],
        )
        .unwrap();
        assert!(outcomes.is_empty());
        assert!(!root.join("manifest.json").exists());
        let _ = std::fs::remove_dir_all(&root);
    }
}
