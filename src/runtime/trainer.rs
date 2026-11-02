//! Offline training over published Parquet books.
//!
//! Reference logistic and linear fits still materialize one sample slice.
//! The catalog scan itself is one published partition at a time. An open raw
//! partition is never included. A failed fit does not replace an existing artifact.

use std::path::Path;

use crate::data::training::dataset::{build_training_examples, DatasetConfig, QuoteEvent};
use crate::data::training::example::{LabelProvenance, ObservedOutcome};
use crate::data::training::history::{BookRange, HistoricalBooks};
use crate::domain::book::BookRecord;
use crate::model::artifact::ModelArtifact;
use crate::model::fair_value::FairValueMethod;
use crate::model::feature::v1::FEATURE_COUNT;
use crate::model::fill::LogisticRegression;
use crate::model::pnl::LinearRegression;
use crate::model::prediction::TrainConfig;

macro_rules! runtime_info {
    ($($arg:tt)*) => {{
        #[cfg(feature = "observe")]
        tracing::info!($($arg)*);
    }};
}

/// Availability of an artifact for later shadow runs. This is not a trading approval.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelState {
    Trained,
    Validated,
    Rejected,
    Published,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrainOutcome {
    Published {
        dataset_id: String,
        state: ModelState,
    },
    Rejected {
        reason: String,
        state: ModelState,
    },
    SkippedNoLabels,
}

pub struct TrainingWindow {
    pub from_ns: i64,
    pub to_ns: i64,
}

/// One offline fit. The historical source is separate so storage stays outside the fit.
pub struct TrainingJob<'a> {
    pub exchange: &'a str,
    pub symbol: &'a str,
    pub quotes: &'a [QuoteEvent],
    pub method: FairValueMethod,
    pub horizon_ns: i64,
    pub window: TrainingWindow,
    pub artifact_path: &'a Path,
}

/// Fits the two reference models on books returned by `history` inside the job window.
///
/// The scan visits whatever the source yields. This function then materializes
/// that window into one `Vec` because the current full-batch fit needs a slice.
/// It is not a streaming trainer. Callers must pass a source that already
/// excludes open, failed, and unpublished partitions.
pub fn train_history<H>(history: &H, job: &TrainingJob<'_>) -> Result<TrainOutcome, TrainFailure>
where
    H: HistoricalBooks,
{
    if job.quotes.is_empty() {
        return Ok(TrainOutcome::SkippedNoLabels);
    }
    let range = BookRange {
        exchange: job.exchange.to_string(),
        symbol: job.symbol.to_string(),
        from_ns: job.window.from_ns,
        to_ns: job.window.to_ns,
    };
    let mut books = Vec::new();
    history
        .for_each(&range, |book| books.push(*book))
        .map_err(|_| TrainFailure::History)?;
    books.sort_by_key(|book| book.stamp().local_receive_ns);
    runtime_info!(
        partitions_note = "books were scanned through the historical source",
        books = books.len(),
        "training window loaded"
    );
    fit_books(
        &books,
        job.quotes,
        job.method,
        job.horizon_ns,
        &job.window,
        job.artifact_path,
    )
}

fn fit_books(
    books: &[BookRecord],
    quotes: &[QuoteEvent],
    method: FairValueMethod,
    horizon_ns: i64,
    window: &TrainingWindow,
    artifact_path: &Path,
) -> Result<TrainOutcome, TrainFailure> {
    if books.is_empty() {
        return Ok(TrainOutcome::Rejected {
            reason: "no published books in the training window".into(),
            state: ModelState::Rejected,
        });
    }
    let config = DatasetConfig::new(method, horizon_ns).map_err(|_| TrainFailure::Dataset)?;
    let set = match build_training_examples(books, quotes, config) {
        Ok(set) => set,
        Err(err) => {
            return Ok(TrainOutcome::Rejected {
                reason: err.to_string(),
                state: ModelState::Rejected,
            })
        }
    };
    runtime_info!(examples = set.examples().len(), "training examples built");
    let fills = set
        .examples()
        .iter()
        .filter(|row| matches!(row.outcome(), ObservedOutcome::Filled { .. }))
        .count();
    runtime_info!(fills, "filled examples");
    if fills == 0 {
        return Ok(TrainOutcome::Rejected {
            reason: "training needs at least one filled example".into(),
            state: ModelState::Rejected,
        });
    }
    let provenance = set.examples()[0].provenance();
    let fill_rows =
        crate::data::training::example::logistic_samples(set.examples(), provenance, method)
            .map_err(|_| TrainFailure::Dataset)?;
    let pnl_rows = crate::data::training::example::pnl_samples(set.examples(), provenance, method)
        .map_err(|_| TrainFailure::Dataset)?;
    let train = TrainConfig {
        learning_rate: 0.05,
        epochs: 40,
    };
    let mut fill = LogisticRegression::<FEATURE_COUNT>::zeros();
    let before_fill = fill.mean_loss(&fill_rows).map_err(|_| TrainFailure::Fit)?;
    let after_fill = fill
        .train(&fill_rows, train)
        .map_err(|_| TrainFailure::Fit)?;
    let mut pnl = LinearRegression::<FEATURE_COUNT>::zeros();
    let before_pnl = pnl.mean_loss(&pnl_rows).map_err(|_| TrainFailure::Fit)?;
    let after_pnl = pnl.train(&pnl_rows, train).map_err(|_| TrainFailure::Fit)?;
    if !after_fill.is_finite()
        || !after_pnl.is_finite()
        || after_fill > before_fill
        || after_pnl > before_pnl
    {
        runtime_info!("model rejected");
        return Ok(TrainOutcome::Rejected {
            reason: "reference training did not reduce loss".into(),
            state: ModelState::Rejected,
        });
    }
    let dataset_id = format!(
        "{}-{}-{}-{}-{}",
        method.name(),
        window.from_ns,
        window.to_ns,
        set.examples().len(),
        fills
    );
    let mut artifact = ModelArtifact::from_models(method, horizon_ns, &fill, &pnl);
    artifact.example_count = set.examples().len() as u64;
    artifact.fill_count = fills as u64;
    artifact.label_provenance = match provenance {
        LabelProvenance::Observed => "observed".into(),
        LabelProvenance::Simulated => "simulated".into(),
    };
    artifact.window_from_ns = window.from_ns;
    artifact.window_to_ns = window.to_ns;
    artifact.dataset_id = dataset_id.clone();
    if let Some(parent) = artifact_path.parent() {
        std::fs::create_dir_all(parent).map_err(|_| TrainFailure::Artifact)?;
    }
    artifact
        .write_json(artifact_path)
        .map_err(|_| TrainFailure::Artifact)?;
    runtime_info!(
        before_fill,
        after_fill,
        before_pnl,
        after_pnl,
        dataset_id = %dataset_id,
        "model artifact published"
    );
    Ok(TrainOutcome::Published {
        dataset_id,
        state: ModelState::Published,
    })
}

#[derive(Debug)]
pub enum TrainFailure {
    Dataset,
    Fit,
    Artifact,
    History,
}

impl std::fmt::Display for TrainFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dataset => f.write_str("training rows could not be built"),
            Self::Fit => f.write_str("reference model training failed"),
            Self::Artifact => f.write_str("model artifact could not be written"),
            Self::History => f.write_str("historical book scan failed"),
        }
    }
}

impl std::error::Error for TrainFailure {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::training::dataset::QuoteEvent;
    use crate::data::training::example::{FillFacts, LabelProvenance};
    use crate::data::training::history::MemoryBooks;
    use crate::domain::book::{BookRecord, BookStamp, Instrument, Level, BOOK_DEPTH};
    use crate::domain::market::CandidateQuote;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn book(ns: i64) -> BookRecord {
        let mut bids = [Level::EMPTY; BOOK_DEPTH];
        let mut asks = [Level::EMPTY; BOOK_DEPTH];
        bids[0] = Level {
            price: 99.9,
            quantity: 2.0,
        };
        asks[0] = Level {
            price: 100.1,
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

    #[test]
    fn trainer_uses_the_historical_source_boundary() {
        let books = [book(1_000), book(2_000)];
        let history = MemoryBooks {
            exchange: "binance",
            symbol: "BTCUSDT",
            books: &books,
        };
        let quote = QuoteEvent::new(
            1_000,
            CandidateQuote::new(99.8, 1.0).unwrap(),
            LabelProvenance::Simulated,
            FillFacts::Filled {
                execution_price: 99.8,
                fees: 0.0,
            },
        );
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("mm-train-{nanos}.json"));
        let outcome = train_history(
            &history,
            &TrainingJob {
                exchange: "binance",
                symbol: "BTCUSDT",
                quotes: &[quote],
                method: FairValueMethod::Microprice,
                horizon_ns: 500,
                window: TrainingWindow {
                    from_ns: 0,
                    to_ns: 5_000,
                },
                artifact_path: &path,
            },
        )
        .unwrap();
        assert!(matches!(outcome, TrainOutcome::Published { .. }));
        assert!(path.is_file());
        let _ = std::fs::remove_file(&path);
    }
}
