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
    let split = match checked_split(&rows, policy) {
        Ok(split) => split,
        Err(outcome) => return Ok(outcome),
    };
    let train_rows = split.train;
    let holdout_rows = split.holdout;
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
    let fill_rows =
        crate::data::training::example::logistic_samples(&train_rows, provenance, method)
            .map_err(|_| TrainFailure::Dataset)?;
    let pnl_rows = crate::data::training::example::pnl_samples(&train_rows, provenance, method)
        .map_err(|_| TrainFailure::Dataset)?;
    let holdout_fill =
        crate::data::training::example::logistic_samples(&holdout_rows, provenance, method)
            .map_err(|_| TrainFailure::Dataset)?;
    let holdout_pnl =
        crate::data::training::example::pnl_samples(&holdout_rows, provenance, method)
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
    let (ev_model_error, ev_baseline_error) = ev_mae(&fill, &pnl, &holdout_rows, rate * mean_pnl)?;
    if !learned_fill.is_finite()
        || !learned_pnl.is_finite()
        || !ev_model_error.is_finite()
        || !ev_baseline_error.is_finite()
        || !crate::model::artifact::losses_accept(
            learned_fill,
            base_fill,
            learned_pnl,
            base_pnl,
            ev_model_error,
            ev_baseline_error,
        )
    {
        return Ok(TrainOutcome::Rejected {
            reason: "holdout did not beat the constant training baselines; combined EV must be strictly better".into(),
        });
    }
    if provenance != LabelProvenance::Observed {
        return Ok(TrainOutcome::InsufficientData {
            reason: "simulated labels are not released for shadow".into(),
        });
    }
    let report = ValidationReport {
        rows_before_purge: split.rows_before_purge as u64,
        purged_training_rows: split.purged as u64,
        train_examples: train_rows.len() as u64,
        holdout_examples: holdout_rows.len() as u64,
        holdout_start_ns: split.holdout_start_ns,
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

struct PurgedSplit {
    holdout_start_ns: i64,
    rows_before_purge: usize,
    purged: usize,
    train: Vec<crate::data::training::example::TrainingExample>,
    holdout: Vec<crate::data::training::example::TrainingExample>,
}

/// Sort is the caller's job. Holdout is the last `holdout_percent` of rows by decision time.
/// A training candidate is dropped when its horizon snapshot is at or after the first holdout decision.
fn checked_split(
    rows: &[crate::data::training::example::TrainingExample],
    policy: EvidencePolicy,
) -> Result<PurgedSplit, TrainOutcome> {
    let Some(split) = purge_holdout(rows, policy.holdout_percent) else {
        return Err(TrainOutcome::InsufficientData {
            reason: "chronological split produced an empty train or holdout slice".into(),
        });
    };
    let train_fills = split
        .train
        .iter()
        .filter(|row| matches!(row.outcome(), ObservedOutcome::Filled { .. }))
        .count();
    let train_misses = split.train.len() - train_fills;
    let holdout_fills = split
        .holdout
        .iter()
        .filter(|row| matches!(row.outcome(), ObservedOutcome::Filled { .. }))
        .count();
    let holdout_misses = split.holdout.len() - holdout_fills;
    if split.train.is_empty()
        || split.train.len() < policy.min_train_examples
        || train_fills < policy.min_train_fills
        || train_misses < policy.min_train_misses
        || split.holdout.len() < policy.min_holdout_examples
        || holdout_fills < policy.min_holdout_fills
        || holdout_misses < policy.min_holdout_misses
    {
        return Err(TrainOutcome::InsufficientData {
            reason: "train or holdout does not meet the evidence policy after purging".into(),
        });
    }
    Ok(split)
}

fn purge_holdout(
    rows: &[crate::data::training::example::TrainingExample],
    holdout_percent: u8,
) -> Option<PurgedSplit> {
    let n = rows.len();
    let train_end = n * (100 - u32::from(holdout_percent)) as usize / 100;
    if train_end == 0 || train_end >= n {
        return None;
    }
    let holdout_start_ns = rows[train_end].timing().decision_ns;
    let mut train = Vec::new();
    let mut purged = 0usize;
    for row in &rows[..train_end] {
        if row.timing().horizon_snapshot_ns >= holdout_start_ns {
            purged += 1;
        } else {
            train.push(*row);
        }
    }
    Some(PurgedSplit {
        holdout_start_ns,
        rows_before_purge: train_end,
        purged,
        train,
        holdout: rows[train_end..].to_vec(),
    })
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

    fn labeled_row(
        decision: i64,
        horizon_snapshot: i64,
        filled: bool,
        pnl: f64,
    ) -> crate::data::training::example::TrainingExample {
        use crate::data::training::example::{
            DecisionTimeInputs, LabelTiming, ObservedOutcome, TrainingExample,
        };
        let market = FairValueMethod::Microprice
            .derive_top(99.9, 1.0, 100.1, 1.0)
            .unwrap();
        TrainingExample::new(
            DecisionTimeInputs::new(market, CandidateQuote::new(99.8, 1.0).unwrap()),
            if filled {
                ObservedOutcome::Filled {
                    pnl,
                    horizon_fair_value: 100.0,
                }
            } else {
                ObservedOutcome::Unfilled
            },
            LabelProvenance::Observed,
            FairValueMethod::Microprice,
            LabelTiming {
                decision_ns: decision,
                decision_snapshot_ns: decision,
                requested_horizon_ns: 1_000,
                horizon_snapshot_ns: horizon_snapshot,
            },
        )
        .unwrap()
    }

    /// Ten decisions at 0,100,...,900. A 20% holdout starts at decision 800.
    fn boundary_rows(crossing_pnl: f64) -> Vec<crate::data::training::example::TrainingExample> {
        (0..10)
            .map(|index| {
                let decision = index * 100;
                let horizon = if index == 7 { 800 } else { decision + 50 };
                let filled = index == 7 || index % 2 == 0;
                let pnl = if index == 7 { crossing_pnl } else { 0.25 };
                labeled_row(decision, horizon, filled, pnl)
            })
            .collect()
    }

    #[test]
    fn a_label_that_finishes_before_the_holdout_stays_in_training() {
        let rows = boundary_rows(1.0);
        let split = purge_holdout(&rows, 20).unwrap();
        assert_eq!(split.holdout_start_ns, 800);
        assert!(split
            .train
            .iter()
            .any(|row| row.timing().decision_ns == 600 && row.timing().horizon_snapshot_ns == 650));
    }

    #[test]
    fn a_label_that_reaches_the_holdout_is_purged_and_not_moved() {
        let rows = boundary_rows(9.0);
        let split = purge_holdout(&rows, 20).unwrap();
        assert_eq!(split.purged, 1);
        assert_eq!(split.rows_before_purge, 8);
        assert_eq!(split.train.len(), 7);
        assert!(split
            .train
            .iter()
            .all(|row| row.timing().decision_ns != 700));
        assert!(split
            .holdout
            .iter()
            .all(|row| row.timing().decision_ns >= 800));
        assert_eq!(split.holdout.len(), 2);
    }

    #[test]
    fn purging_below_the_evidence_policy_is_insufficient_data() {
        let mut rows = boundary_rows(1.0);
        for row in rows.iter_mut().take(8) {
            if row.timing().decision_ns != 0 {
                *row = labeled_row(row.timing().decision_ns, 800, false, 0.0);
            }
        }
        let policy = EvidencePolicy {
            min_train_examples: 3,
            min_train_fills: 1,
            min_train_misses: 1,
            min_holdout_examples: 2,
            min_holdout_fills: 1,
            min_holdout_misses: 1,
            ..EvidencePolicy::default()
        };
        let before = purge_holdout(&rows, policy.holdout_percent).unwrap();
        assert!(before.rows_before_purge >= policy.min_train_examples);
        assert!(before.train.len() < policy.min_train_examples);
        assert!(matches!(
            checked_split(&rows, policy),
            Err(TrainOutcome::InsufficientData { .. })
        ));
    }

    #[test]
    fn changing_a_purged_rows_holdout_label_does_not_change_the_fitted_slice() {
        let low = boundary_rows(0.1);
        let high = boundary_rows(50.0);
        let low_split = purge_holdout(&low, 20).unwrap();
        let high_split = purge_holdout(&high, 20).unwrap();
        assert_eq!(low_split.train, high_split.train);
        assert!(low_split
            .train
            .iter()
            .all(|row| row.timing().horizon_snapshot_ns < low_split.holdout_start_ns));
        let fitted_pnls: Vec<f64> = low_split
            .train
            .iter()
            .filter_map(|row| match row.outcome() {
                crate::data::training::example::ObservedOutcome::Filled { pnl, .. } => Some(pnl),
                crate::data::training::example::ObservedOutcome::Unfilled => None,
            })
            .collect();
        assert!(!fitted_pnls.contains(&50.0));
        assert!(!fitted_pnls.contains(&0.1));
    }

    #[test]
    fn holdout_labels_do_not_enter_the_training_baseline_slice() {
        let mut rows = boundary_rows(1.0);
        rows[8] = labeled_row(800, 850, true, 40.0);
        rows[9] = labeled_row(900, 950, false, 0.0);
        let split = purge_holdout(&rows, 20).unwrap();
        let fills: Vec<f64> = split
            .train
            .iter()
            .filter_map(|row| match row.outcome() {
                crate::data::training::example::ObservedOutcome::Filled { pnl, .. } => Some(pnl),
                crate::data::training::example::ObservedOutcome::Unfilled => None,
            })
            .collect();
        assert!(!fills.contains(&40.0));
        let mean = fills.iter().sum::<f64>() / fills.len() as f64;
        assert!((mean - 0.25).abs() < 1e-12);
    }
}
