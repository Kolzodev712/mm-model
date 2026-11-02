//! Collection lifecycle: rotate raw partitions and publish Parquet.
//!
//! The collector thread only appends MMBK. A closed hour wakes one worker,
//! which converts. Training is a separate mode and does not run on this loop.
//! Shutdown flushes the open hour and leaves it open.

use crate::data::market::source::BookSource;
use crate::domain::book::BookRecord;
use crate::runtime::clock::Clock;
use crate::runtime::collector::{CollectError, Collector};
use crate::runtime::config::{AppConfig, TaskConfig, TrainingTarget};
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

macro_rules! runtime_error {
    ($($arg:tt)*) => {{
        #[cfg(feature = "observe")]
        tracing::error!($($arg)*);
    }};
}

#[derive(Debug)]
pub enum OrchestratorError {
    Collect(CollectError),
    Convert(ConvertError),
    Train(TrainFailure),
    Catalog(ManifestError),
    Source(String),
    /// Conversion is more than two hours behind. Books already written are kept.
    Worker,
    /// Train mode was asked to run without a quote-history file.
    QuoteHistory,
}

impl std::fmt::Display for OrchestratorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Collect(err) => write!(f, "{err}"),
            Self::Convert(err) => write!(f, "{err}"),
            Self::Train(err) => write!(f, "{err}"),
            Self::Catalog(err) => write!(f, "{err}"),
            Self::Source(err) => write!(f, "{err}"),
            Self::Worker => f.write_str("conversion worker stopped unexpectedly"),
            Self::QuoteHistory => f.write_str("train mode requires a quote-history file"),
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
    mut should_stop: impl FnMut() -> bool,
) -> Result<Vec<TrainOutcome>, OrchestratorError>
where
    S: BookSource,
    S::Error: std::fmt::Display,
{
    match &config.task {
        TaskConfig::Train(task) => {
            if matches!(task.target, TrainingTarget::QuoteFill { .. })
                && !task.quotes_path.as_ref().is_some_and(|path| path.is_file())
            {
                return Err(OrchestratorError::QuoteHistory);
            }
            let catalog = Catalog::open(&config.data_root)?;
            Ok(vec![train_from(config, &catalog)?])
        }
        TaskConfig::LiveInference(_) => Ok(Vec::new()),
        TaskConfig::Collect(task) => run_collect(config, task, source, &mut should_stop),
    }
}

fn run_collect<S>(
    config: &AppConfig,
    task: &crate::runtime::config::CollectTask,
    source: &mut S,
    should_stop: &mut impl FnMut() -> bool,
) -> Result<Vec<TrainOutcome>, OrchestratorError>
where
    S: BookSource,
    S::Error: std::fmt::Display,
{
    let started = std::time::Instant::now();
    let catalog = Catalog::open(&config.data_root)?;
    let mut collector = Collector::new(
        catalog,
        config.market.exchange.as_str(),
        config.market.symbol.as_str(),
        task.flush_every,
    );
    let (tx, rx) = std::sync::mpsc::sync_channel::<()>(1);
    let worker_catalog = collector.shared_catalog();
    let worker = std::thread::spawn(move || worker_loop(worker_catalog, rx));
    if let Ok(now_ns) = crate::runtime::clock::SystemClock.unix_nanos() {
        let _ = seal_stale_partitions(collector.catalog_mutex(), now_ns);
    }
    let _ = tx.try_send(());
    let mut failed = None;
    loop {
        if should_stop() || task.run_for.is_some_and(|limit| started.elapsed() >= limit) {
            break;
        }
        let book = match source.next_book() {
            Ok(book) => book,
            Err(err) if source.is_idle(&err) => continue,
            Err(err) => {
                failed = Some(OrchestratorError::Source(err.to_string()));
                break;
            }
        };
        let Some(book) = book else { break };
        match collector.push(&book) {
            Ok(Some(_)) => {
                if tx.try_send(()).is_err() {
                    let pending = collector
                        .catalog_mutex()
                        .lock()
                        .unwrap_or_else(|err| err.into_inner())
                        .pending_conversion()
                        .len();
                    runtime_info!(pending, "conversion wakeup already pending");
                }
            }
            Ok(None) => {}
            Err(err) => {
                failed = Some(err.into());
                break;
            }
        }
    }
    if let Err(err) = collector.finish() {
        failed.get_or_insert(err.into());
    }
    drop(tx);
    match worker.join() {
        Ok(result) => result?,
        Err(_) => return Err(OrchestratorError::Worker),
    };
    runtime_info!("collection finished");
    match failed {
        Some(err) => Err(err),
        None => Ok(Vec::new()),
    }
}

fn seal_stale_partitions(catalog: &std::sync::Mutex<Catalog>, now_ns: i64) -> Vec<PartitionId> {
    let mut guard = catalog.lock().unwrap_or_else(|err| err.into_inner());
    let mut sealed = Vec::new();
    let snapshot: Vec<_> = guard.records().to_vec();
    for row in snapshot {
        let retry = matches!(
            row.state,
            crate::storage::partition::PartitionState::Closed
                | crate::storage::partition::PartitionState::Validated
                | crate::storage::partition::PartitionState::Converted
        );
        let stale_open = row.state == crate::storage::partition::PartitionState::Open
            && crate::storage::partition::PartitionId::from_receive_ns(
                &row.id.exchange,
                &row.id.symbol,
                now_ns,
            )
            .ok()
            .is_some_and(|current| {
                (row.id.day.as_str(), row.id.hour) < (current.day.as_str(), current.hour)
            });
        if retry {
            sealed.push(row.id);
            continue;
        }
        if !stale_open {
            continue;
        }
        let path = row.id.raw_path(guard.root());
        let count = std::fs::File::open(&path)
            .ok()
            .and_then(|mut file| crate::storage::raw::read_book_log(&mut file).ok())
            .map(|books| books.len() as u64);
        let Some(count) = count else {
            continue;
        };
        if let Some(existing) = guard.get(&row.id).cloned() {
            let mut existing = existing;
            existing.state = crate::storage::partition::PartitionState::Closed;
            existing.raw_count = count;
            let id = existing.id.clone();
            guard.upsert(existing);
            if guard.save().is_ok() {
                sealed.push(id);
            }
        }
    }
    sealed
}

fn worker_loop(
    catalog: std::sync::Arc<std::sync::Mutex<Catalog>>,
    rx: std::sync::mpsc::Receiver<()>,
) -> Result<(), OrchestratorError> {
    while rx.recv().is_ok() {
        while rx.try_recv().is_ok() {}
        let pending = {
            let guard = catalog.lock().unwrap_or_else(|err| err.into_inner());
            let pending = guard.pending_conversion();
            runtime_info!(backlog = pending.len(), "conversion worker woke");
            pending
        };
        for id in pending {
            match publish_partition(&catalog, &id, ParquetConfig::default()) {
                Ok(()) => {}
                Err(ConvertError::StillOpen | ConvertError::Ambiguous | ConvertError::Missing) => {
                    return Err(ConvertError::Ambiguous.into());
                }
                Err(ConvertError::Manifest(err)) => return Err(err.into()),
                Err(err) => {
                    runtime_error!(error = %err, "partition conversion failed");
                }
            }
        }
    }
    Ok(())
}

fn train_from(config: &AppConfig, catalog: &Catalog) -> Result<TrainOutcome, TrainFailure> {
    let history = ParquetHistory::new(catalog);
    let TaskConfig::Train(task) = &config.task else {
        return Err(TrainFailure::Dataset);
    };
    let quotes = crate::storage::quotes::ParquetQuoteHistory {
        path: task.quotes_path.clone().unwrap_or_default(),
        exchange: config.market.exchange.as_str().to_string(),
        symbol: config.market.symbol.as_str().to_string(),
    };
    train_history(
        &history,
        &quotes,
        &TrainingJob {
            exchange: config.market.exchange.as_str(),
            symbol: config.market.symbol.as_str(),
            method: config.market.fair_value,
            window: crate::runtime::trainer::TrainingWindow {
                from_ns: task.window_from_ns,
                to_ns: task.window_to_ns,
            },
            evidence: task.evidence,
            artifact_path: &task.artifact_path,
            target: task.target,
        },
    )
    .map_err(|err| match err {
        TrainFailure::Quotes => TrainFailure::Quotes,
        other => other,
    })
}

/// Books already in memory. Used by tests and by a replay that is not live.
pub fn run_books(
    config: &AppConfig,
    books: &[BookRecord],
) -> Result<Vec<TrainOutcome>, OrchestratorError> {
    let mut source = crate::data::market::source::SliceSource::new(books.to_vec());
    run_source(config, &mut source, || false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::book::{BookRecord, BookStamp, Instrument, Level, BOOK_DEPTH};
    use crate::model::fair_value::FairValueMethod;
    use crate::runtime::config::{
        AppConfig, CollectTask, Exchange, LiveInferenceTask, MarketConfig, RunMode, Symbol,
        TaskConfig, TrainTask, TrainingTarget,
    };
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
        let task = match mode {
            RunMode::Collect => TaskConfig::Collect(CollectTask {
                run_for: None,
                flush_every: 1,
            }),
            RunMode::Train => TaskConfig::Train(TrainTask {
                target: TrainingTarget::QuoteFill { horizon_ns: 500 },
                window_from_ns: 0,
                window_to_ns: i64::MAX / 4,
                quotes_path: Some(root.join("quotes/binance/BTCUSDT.parquet")),
                artifact_path: root.join("models/current.json"),
                evidence: crate::runtime::config::EvidencePolicy::default(),
            }),
            RunMode::LiveInference => TaskConfig::LiveInference(LiveInferenceTask {
                run_for: None,
                report_every: std::time::Duration::from_secs(5),
                models: Vec::new(),
            }),
        };
        AppConfig {
            version: 1,
            data_root: root.to_path_buf(),
            market: MarketConfig {
                exchange: Exchange::BinanceSpot,
                symbol: Symbol::new("BTCUSDT").unwrap(),
                depth: 10,
                update_ms: 100,
                fair_value: FairValueMethod::Microprice,
            },
            task,
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
        let outcomes =
            run_books(&config(&root, RunMode::Collect), &[early, later, next_hour]).unwrap();
        assert!(outcomes.is_empty());
        let catalog = Catalog::open(&root).unwrap();
        let published = catalog.published_overlapping("binance", "BTCUSDT", 0, i64::MAX / 4);
        assert_eq!(published.len(), 1);
        assert!(published
            .iter()
            .all(|row| row.state == PartitionState::Published));
        assert!(published
            .iter()
            .all(|row| row.id.parquet_path(&root).is_file()));
        let again_catalog = std::sync::Mutex::new(Catalog::open(&root).unwrap());
        crate::runtime::converter::publish_partition(
            &again_catalog,
            &published[0].id,
            crate::storage::parquet::ParquetConfig::default(),
        )
        .unwrap();
        let again = Catalog::open(&root).unwrap();
        assert_eq!(
            again
                .published_overlapping("binance", "BTCUSDT", 0, i64::MAX / 4)
                .len(),
            1
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
            collector.catalog_mutex(),
            &id,
            crate::storage::parquet::ParquetConfig::default(),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            crate::runtime::converter::ConvertError::StillOpen
        ));
        assert!(collector
            .catalog_mutex()
            .lock()
            .unwrap()
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
        let err =
            run_source(&config(&root, RunMode::Train), &mut ClosedSource, || false).unwrap_err();
        assert!(matches!(err, OrchestratorError::QuoteHistory));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn more_than_two_stale_partitions_are_converted() {
        let root = temp();
        let start = 1_700_000_000_000_000_000i64;
        let mut catalog = Catalog::open(&root).unwrap();
        for step in 0..3 {
            let ns = start + step * 3_600_000_000_000;
            let sample = book(ns, 99.9, 100.1);
            let id =
                crate::storage::partition::partition_for("binance", "BTCUSDT", &sample).unwrap();
            let path = id.raw_path(&root);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            let mut writer = crate::storage::raw::BookLogWriter::open(&path, 1).unwrap();
            writer.write(&sample).unwrap();
            writer.close().unwrap();
            let mut row = crate::storage::partition::PartitionRecord::open(id, ns);
            row.state = PartitionState::Closed;
            row.raw_count = 1;
            row.end_ns = ns;
            catalog.upsert(row);
        }
        catalog.save().unwrap();
        run_books(&config(&root, RunMode::Collect), &[]).unwrap();
        let catalog = Catalog::open(&root).unwrap();
        assert_eq!(
            catalog
                .published_overlapping("binance", "BTCUSDT", 0, i64::MAX / 4)
                .len(),
            3
        );
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
        let id = collector.seal().unwrap().unwrap();
        let raw = id.raw_path(&root);
        std::fs::remove_file(&raw).unwrap();
        let err = crate::runtime::converter::publish_partition(
            collector.catalog_mutex(),
            &id,
            crate::storage::parquet::ParquetConfig::default(),
        )
        .unwrap_err();
        assert!(matches!(err, crate::runtime::converter::ConvertError::Raw));
        assert_eq!(
            collector
                .catalog_mutex()
                .lock()
                .unwrap()
                .get(&id)
                .unwrap()
                .state,
            PartitionState::Failed
        );
        assert!(collector
            .catalog_mutex()
            .lock()
            .unwrap()
            .published_overlapping("binance", "BTCUSDT", 0, i64::MAX / 4)
            .is_empty());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[cfg(feature = "live")]
    #[test]
    #[ignore = "opens a public Binance websocket"]
    fn short_collect_smoke() {
        let root = temp();
        let mut cfg = config(&root, RunMode::Collect);
        if let TaskConfig::Collect(task) = &mut cfg.task {
            task.run_for = Some(std::time::Duration::from_secs(5));
        }
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
        let outcomes = run_source(&cfg, &mut client, || false).unwrap();
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
        )
        .unwrap();
        assert!(outcomes.is_empty());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn live_inference_mode_does_not_publish_partitions() {
        let root = temp();
        let outcomes = run_books(
            &config(&root, RunMode::LiveInference),
            &[book(10, 99.9, 100.1)],
        )
        .unwrap();
        assert!(outcomes.is_empty());
        assert!(!root.join("manifest.json").exists());
        let _ = std::fs::remove_dir_all(&root);
    }
}
