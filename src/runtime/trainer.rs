//! Offline training over published Parquet books.
//!
//! Reference logistic and linear fits still materialize one sample slice.
//! The catalog scan itself is one published partition at a time. An open raw
//! partition is never included. A failed fit does not replace an existing artifact.

use std::path::Path;

use crate::data::training::dataset::{build_training_examples, DatasetConfig, QuoteEvent};
use crate::data::training::example::{LabelProvenance, ObservedOutcome};
use crate::data::training::history::{BookRange, HistoricalBooks};
use crate::data::training::quotes::{QuoteHistory, QuoteRange};
use crate::domain::book::BookRecord;
use crate::model::artifact::ModelArtifact;
use crate::model::artifact::ValidationReport;
use crate::model::fair_value::FairValueMethod;
use crate::model::feature::v1::FEATURE_COUNT;
use crate::model::fill::LogisticRegression;
use crate::model::pnl::LinearRegression;
use crate::model::prediction::predict_quote;
use crate::model::prediction::TrainConfig;
use crate::runtime::config::EvidencePolicy;

macro_rules! runtime_info {
    ($($arg:tt)*) => {{
        #[cfg(feature = "observe")]
        tracing::info!($($arg)*);
    }};
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrainOutcome {
    PublishedForShadow { dataset_id: String },
    InsufficientData { reason: String },
    Rejected { reason: String },
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
    pub method: FairValueMethod,
    pub horizon_ns: i64,
    pub window: TrainingWindow,
    pub evidence: EvidencePolicy,
    pub artifact_path: &'a Path,
}

/// Fits the two reference models on books and quotes inside the job window.
///
/// The scan visits whatever each source yields. This function then materializes
/// that window into one `Vec` because the current full-batch fit needs a slice.
/// It is not a streaming trainer. Callers must pass a book source that already
/// excludes open, failed, and unpublished partitions.
pub fn train_history<H, Q>(
    history: &H,
    quotes: &Q,
    job: &TrainingJob<'_>,
) -> Result<TrainOutcome, TrainFailure>
where
    H: HistoricalBooks,
    Q: QuoteHistory,
{
    let range = QuoteRange {
        exchange: job.exchange.to_string(),
        symbol: job.symbol.to_string(),
        from_ns: job.window.from_ns,
        to_ns: job.window.to_ns,
    };
    let mut loaded = Vec::new();
    quotes
        .for_each(&range, |quote| loaded.push(*quote))
        .map_err(|_| TrainFailure::Quotes)?;
    if loaded.is_empty() {
        return Ok(TrainOutcome::SkippedNoLabels);
    }
    let books_range = BookRange {
        exchange: job.exchange.to_string(),
        symbol: job.symbol.to_string(),
        from_ns: job.window.from_ns,
        to_ns: job.window.to_ns,
    };
    let mut books = Vec::new();
    history
        .for_each(&books_range, |book| books.push(*book))
        .map_err(|_| TrainFailure::History)?;
    books.sort_by_key(|book| book.stamp().local_receive_ns);
    runtime_info!(
        books = books.len(),
        quotes = loaded.len(),
        "training window loaded"
    );
    fit_books(&books, &loaded, job)
}

fn fit_books(
    books: &[BookRecord],
    quotes: &[QuoteEvent],
    job: &TrainingJob<'_>,
) -> Result<TrainOutcome, TrainFailure> {
    let method = job.method;
    let horizon_ns = job.horizon_ns;
    let window = &job.window;
    let artifact_path = job.artifact_path;
    let policy = job.evidence;
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
    let n = rows.len();
    let train_end = n * (100 - u32::from(policy.holdout_percent)) as usize / 100;
    if train_end == 0 || train_end >= n {
        return Ok(TrainOutcome::InsufficientData {
            reason: "chronological split produced an empty train or holdout slice".into(),
        });
    }
    let train_rows = &rows[..train_end];
    let holdout_rows = &rows[train_end..];
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
    let holdout_fills = holdout_rows
        .iter()
        .filter(|row| matches!(row.outcome(), ObservedOutcome::Filled { .. }))
        .count();
    let holdout_misses = holdout_rows.len() - holdout_fills;
    if train_rows.len() < policy.min_train_examples
        || train_fills < policy.min_train_fills
        || train_unfilled < policy.min_train_misses
        || holdout_rows.len() < policy.min_holdout_examples
        || holdout_fills < policy.min_holdout_fills
        || holdout_misses < policy.min_holdout_misses
    {
        return Ok(TrainOutcome::InsufficientData {
            reason: "train or holdout does not meet the evidence policy".into(),
        });
    }
    let fill_rows =
        crate::data::training::example::logistic_samples(train_rows, provenance, method)
            .map_err(|_| TrainFailure::Dataset)?;
    let pnl_rows = crate::data::training::example::pnl_samples(train_rows, provenance, method)
        .map_err(|_| TrainFailure::Dataset)?;
    let holdout_fill =
        crate::data::training::example::logistic_samples(holdout_rows, provenance, method)
            .map_err(|_| TrainFailure::Dataset)?;
    let holdout_pnl = crate::data::training::example::pnl_samples(holdout_rows, provenance, method)
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
    let learned_fill = fill
        .mean_loss(&holdout_fill)
        .map_err(|_| TrainFailure::Fit)?;
    let learned_pnl = pnl.mean_loss(&holdout_pnl).map_err(|_| TrainFailure::Fit)?;
    let base_fill = fill_baseline
        .mean_loss(&holdout_fill)
        .map_err(|_| TrainFailure::Fit)?;
    let base_pnl = pnl_baseline
        .mean_loss(&holdout_pnl)
        .map_err(|_| TrainFailure::Fit)?;
    let (ev_model_error, ev_baseline_error) = ev_mae(&fill, &pnl, holdout_rows, rate * mean_pnl)?;
    if !learned_fill.is_finite()
        || !learned_pnl.is_finite()
        || !ev_model_error.is_finite()
        || learned_fill > base_fill
        || learned_pnl > base_pnl
        || ev_model_error > ev_baseline_error
    {
        return Ok(TrainOutcome::Rejected {
            reason: "holdout error is worse than the constant training baseline".into(),
        });
    }
    if provenance != LabelProvenance::Observed {
        return Ok(TrainOutcome::InsufficientData {
            reason: "simulated labels are not released for shadow".into(),
        });
    }
    let report = ValidationReport {
        train_examples: train_rows.len() as u64,
        holdout_examples: holdout_rows.len() as u64,
        train_fills: train_fills as u64,
        train_misses: train_unfilled as u64,
        holdout_fills: holdout_fills as u64,
        holdout_misses: holdout_misses as u64,
        fill_model_loss: learned_fill,
        fill_baseline_loss: base_fill,
        pnl_model_loss: learned_pnl,
        pnl_baseline_loss: base_pnl,
        ev_model_error,
        ev_baseline_error,
    };
    let dataset_id = format!(
        "{}-{}-{}-{}-{}",
        method.name(),
        window.from_ns,
        window.to_ns,
        rows.len(),
        fills
    );
    let artifact = ModelArtifact::published_for_shadow(
        method,
        horizon_ns,
        &fill,
        &pnl,
        crate::model::artifact::ArtifactMetadata {
            provenance,
            window_from_ns: window.from_ns,
            window_to_ns: window.to_ns,
            dataset_id: dataset_id.clone(),
        },
        report,
    )
    .map_err(|_| TrainFailure::Artifact)?;
    artifact
        .write_json(artifact_path)
        .map_err(|_| TrainFailure::Artifact)?;
    runtime_info!(dataset_id = %dataset_id, "model artifact released for shadow");
    Ok(TrainOutcome::PublishedForShadow { dataset_id })
}

fn ev_mae(
    fill: &LogisticRegression<FEATURE_COUNT>,
    pnl: &LinearRegression<FEATURE_COUNT>,
    holdout: &[crate::data::training::example::TrainingExample],
    baseline_ev: f64,
) -> Result<(f64, f64), TrainFailure> {
    if holdout.is_empty() || !baseline_ev.is_finite() {
        return Err(TrainFailure::Fit);
    }
    let mut model = 0.0;
    let mut baseline = 0.0;
    for row in holdout {
        let inputs = row.inputs();
        let market = inputs.market();
        let quote = inputs.quote();
        let prediction =
            predict_quote(&market, &quote, fill, pnl).map_err(|_| TrainFailure::Fit)?;
        let observed = match row.outcome() {
            ObservedOutcome::Unfilled => 0.0,
            ObservedOutcome::Filled { pnl, .. } => pnl,
        };
        model += (prediction.expected_value - observed).abs();
        baseline += (baseline_ev - observed).abs();
    }
    let count = holdout.len() as f64;
    Ok((model / count, baseline / count))
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
    Quotes,
}

impl std::fmt::Display for TrainFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dataset => f.write_str("training rows could not be built"),
            Self::Fit => f.write_str("reference model training failed"),
            Self::Artifact => f.write_str("model artifact could not be written"),
            Self::History => f.write_str("historical book scan failed"),
            Self::Quotes => f.write_str("quote history could not be read"),
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
        let quotes = crate::data::training::quotes::MemoryQuoteHistory {
            exchange: "binance",
            symbol: "BTCUSDT",
            quotes: &[quote],
        };
        let outcome = train_history(
            &history,
            &quotes,
            &TrainingJob {
                exchange: "binance",
                symbol: "BTCUSDT",
                method: FairValueMethod::Microprice,
                horizon_ns: 500,
                window: TrainingWindow {
                    from_ns: 0,
                    to_ns: 5_000,
                },
                evidence: EvidencePolicy::default(),
                artifact_path: &path,
            },
        )
        .unwrap();
        assert!(matches!(outcome, TrainOutcome::InsufficientData { .. }));
        assert!(!path.is_file());
    }

    #[test]
    fn a_later_quote_history_is_read_on_the_next_cycle() {
        let books = [book(1_000), book(2_000)];
        let history = MemoryBooks {
            exchange: "binance",
            symbol: "BTCUSDT",
            books: &books,
        };
        let path = std::env::temp_dir().join("mm-train-later.json");
        let job = TrainingJob {
            exchange: "binance",
            symbol: "BTCUSDT",
            method: FairValueMethod::Microprice,
            horizon_ns: 500,
            window: TrainingWindow {
                from_ns: 0,
                to_ns: 5_000,
            },
            evidence: EvidencePolicy::default(),
            artifact_path: &path,
        };
        let empty = crate::data::training::quotes::MemoryQuoteHistory {
            exchange: "binance",
            symbol: "BTCUSDT",
            quotes: &[],
        };
        assert!(matches!(
            train_history(&history, &empty, &job).unwrap(),
            TrainOutcome::SkippedNoLabels
        ));
        let quote = QuoteEvent::new(
            1_000,
            CandidateQuote::new(99.8, 1.0).unwrap(),
            LabelProvenance::Observed,
            FillFacts::Unfilled,
        );
        let later = crate::data::training::quotes::MemoryQuoteHistory {
            exchange: "binance",
            symbol: "BTCUSDT",
            quotes: &[quote],
        };
        assert!(matches!(
            train_history(&history, &later, &job).unwrap(),
            TrainOutcome::InsufficientData { .. }
        ));
    }
}
