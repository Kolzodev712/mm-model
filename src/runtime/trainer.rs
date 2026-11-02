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

/// Experimental status of a fit. None of these is a trading approval.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelState {
    /// The sample is too small, or the holdout cannot support a comparison.
    InsufficientData,
    /// The fit ran and failed the holdout comparison with the constant baseline.
    Rejected,
    /// Holdout loss is no worse than the constant baseline. Economic usefulness
    /// was not established, so the file is not released for shadow.
    StatisticallyValidated,
    /// Observed labels, enough evidence, and a holdout that matched the baseline.
    /// Shadow may load this. It is still not a claim of profitability.
    PublishedForShadow,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrainOutcome {
    PublishedForShadow { dataset_id: String },
    InsufficientData { reason: String },
    Rejected { reason: String },
    SkippedNoLabels,
}

const MIN_EXAMPLES: usize = 20;
const MIN_FILLS: usize = 5;

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
        return Ok(TrainOutcome::InsufficientData {
            reason: "no published books in the training window".into(),
        });
    }
    let config = DatasetConfig::new(method, horizon_ns).map_err(|_| TrainFailure::Dataset)?;
    let set = match build_training_examples(books, quotes, config) {
        Ok(set) => set,
        Err(err) => {
            return Ok(TrainOutcome::Rejected {
                reason: err.to_string(),
            })
        }
    };
    let mut rows = set.examples().to_vec();
    rows.sort_by_key(|row| row.timing().decision_ns);
    let fills = rows
        .iter()
        .filter(|row| matches!(row.outcome(), ObservedOutcome::Filled { .. }))
        .count();
    runtime_info!(examples = rows.len(), fills, "training examples built");
    if rows.len() < MIN_EXAMPLES || fills < MIN_FILLS {
        let _ = ModelState::InsufficientData;
        return Ok(TrainOutcome::InsufficientData {
            reason: format!(
                "need at least {MIN_EXAMPLES} examples and {MIN_FILLS} fills; got {} examples and {fills} fills",
                rows.len()
            ),
        });
    }
    let n = rows.len();
    let train_end = n * 6 / 10;
    let val_end = n * 8 / 10;
    if train_end == 0 || val_end <= train_end || val_end >= n {
        return Ok(TrainOutcome::InsufficientData {
            reason: "chronological split produced an empty train, validation, or test slice".into(),
        });
    }
    let train_rows = &rows[..train_end];
    let val_rows = &rows[train_end..val_end];
    let test_rows = &rows[val_end..];
    let provenance = train_rows[0].provenance();
    if rows.iter().any(|row| row.provenance() != provenance) {
        return Ok(TrainOutcome::Rejected {
            reason: "mixed label provenance".into(),
        });
    }
    let train_fills = train_rows
        .iter()
        .filter(|row| matches!(row.outcome(), ObservedOutcome::Filled { .. }))
        .count();
    let train_unfilled = train_rows.len() - train_fills;
    let val_fills = val_rows
        .iter()
        .filter(|row| matches!(row.outcome(), ObservedOutcome::Filled { .. }))
        .count();
    if train_fills < MIN_FILLS || train_unfilled == 0 || val_fills < MIN_FILLS {
        return Ok(TrainOutcome::InsufficientData {
            reason: "the training slice and the validation slice each need at least 5 fills".into(),
        });
    }
    let fill_rows =
        crate::data::training::example::logistic_samples(train_rows, provenance, method)
            .map_err(|_| TrainFailure::Dataset)?;
    let pnl_rows = crate::data::training::example::pnl_samples(train_rows, provenance, method)
        .map_err(|_| TrainFailure::Dataset)?;
    let val_fill = crate::data::training::example::logistic_samples(val_rows, provenance, method)
        .map_err(|_| TrainFailure::Dataset)?;
    let val_pnl = crate::data::training::example::pnl_samples(val_rows, provenance, method)
        .map_err(|_| TrainFailure::Dataset)?;
    let train = TrainConfig {
        learning_rate: 0.2,
        epochs: 200,
    };
    let mut fill = LogisticRegression::<FEATURE_COUNT>::zeros();
    let after_fill = fill
        .train(&fill_rows, train)
        .map_err(|_| TrainFailure::Fit)?;
    let mut pnl = LinearRegression::<FEATURE_COUNT>::zeros();
    let after_pnl = pnl.train(&pnl_rows, train).map_err(|_| TrainFailure::Fit)?;
    if !after_fill.is_finite() || !after_pnl.is_finite() {
        return Ok(TrainOutcome::Rejected {
            reason: "reference training produced a non-finite loss".into(),
        });
    }
    let rate = train_fills as f64 / train_rows.len() as f64;
    let logit = (rate / (1.0 - rate)).ln();
    let fill_baseline =
        LogisticRegression::<FEATURE_COUNT>::from_weights(logit, [0.0; FEATURE_COUNT])
            .map_err(|_| TrainFailure::Fit)?;
    let mean_pnl = val_mean_pnl(&pnl_rows);
    let pnl_baseline =
        LinearRegression::<FEATURE_COUNT>::from_weights(mean_pnl, [0.0; FEATURE_COUNT])
            .map_err(|_| TrainFailure::Fit)?;
    let learned_fill = fill.mean_loss(&val_fill).map_err(|_| TrainFailure::Fit)?;
    let learned_pnl = pnl.mean_loss(&val_pnl).map_err(|_| TrainFailure::Fit)?;
    let base_fill = fill_baseline
        .mean_loss(&val_fill)
        .map_err(|_| TrainFailure::Fit)?;
    let base_pnl = pnl_baseline
        .mean_loss(&val_pnl)
        .map_err(|_| TrainFailure::Fit)?;
    if !learned_fill.is_finite()
        || !learned_pnl.is_finite()
        || learned_fill > base_fill
        || learned_pnl > base_pnl
    {
        let _ = ModelState::Rejected;
        return Ok(TrainOutcome::Rejected {
            reason: "validation loss is worse than the constant baseline".into(),
        });
    }
    let _ = (test_rows.len(), ModelState::StatisticallyValidated);
    if provenance != LabelProvenance::Observed || val_fills < MIN_FILLS {
        return Ok(TrainOutcome::InsufficientData {
            reason: "holdout matched the baseline, but the labels cannot support a shadow release"
                .into(),
        });
    }
    let dataset_id = format!(
        "{}-{}-{}-{}-{}",
        method.name(),
        window.from_ns,
        window.to_ns,
        rows.len(),
        fills
    );
    let mut artifact = ModelArtifact::from_models(method, horizon_ns, &fill, &pnl);
    artifact.example_count = rows.len() as u64;
    artifact.fill_count = fills as u64;
    artifact.train_fill_count = train_fills as u64;
    artifact.label_provenance = "observed".into();
    artifact.window_from_ns = window.from_ns;
    artifact.window_to_ns = window.to_ns;
    artifact.dataset_id = dataset_id.clone();
    artifact.model_state = "published_for_shadow".into();
    artifact.validation_note = format!(
        "chronological 60/20/20; validation fill loss {learned_fill:.6} vs baseline {base_fill:.6}; validation pnl loss {learned_pnl:.6} vs baseline {base_pnl:.6}; test rows {}",
        test_rows.len()
    );
    artifact
        .write_json(artifact_path)
        .map_err(|_| TrainFailure::Artifact)?;
    runtime_info!(dataset_id = %dataset_id, "model artifact released for shadow");
    let _ = ModelState::PublishedForShadow;
    Ok(TrainOutcome::PublishedForShadow { dataset_id })
}

fn val_mean_pnl(rows: &[crate::model::pnl::PnlSample<FEATURE_COUNT>]) -> f64 {
    if rows.is_empty() {
        return 0.0;
    }
    rows.iter().map(|row| row.pnl).sum::<f64>() / rows.len() as f64
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
        assert!(matches!(outcome, TrainOutcome::InsufficientData { .. }));
        assert!(!path.is_file());
    }
}
