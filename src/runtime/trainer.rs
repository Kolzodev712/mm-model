//! Offline training over published book history.
//!
//! Quote-fill training also reads quote outcomes. Market-outcome training does
//! not. Both fits materialize one window, then use a purged chronological
//! holdout. A failed fit does not replace an existing artifact.

use std::path::Path;

use crate::data::training::dataset::{build_quote_fill_examples, DatasetConfig, QuoteEvent};
use crate::data::training::example::LabelProvenance;
use crate::data::training::history::{BookRange, HistoricalBooks};
use crate::data::training::market::{build_market_examples, MarketDatasetConfig, MarketExample};
use crate::data::training::quotes::{QuoteHistory, QuoteRange};
use crate::domain::book::BookRecord;
use crate::model::artifact::{
    log_loss_accepts, ArtifactMetadata, LabelSource, ModelArtifact, ModelFamily, ValidationReport,
};
use crate::model::calibration::score_probabilities;
use crate::model::fair_value::FairValueMethod;
use crate::model::feature::market::MARKET_FEATURE_COUNT;
use crate::model::fill::{FillSample, LogisticRegression};
use crate::model::prediction::{FillProbabilityModel, TrainConfig};
use crate::runtime::config::{EvidencePolicy, TrainingTarget};

macro_rules! runtime_info {
    ($($arg:tt)*) => {{
        #[cfg(feature = "observe")]
        tracing::info!($($arg)*);
    }};
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrainOutcome {
    Validated { dataset_id: String },
    InsufficientData { reason: String },
    Rejected { reason: String },
    SkippedNoLabels,
}

pub struct TrainingWindow {
    pub from_ns: i64,
    pub to_ns: i64,
}

/// One offline fit. The book source is separate so storage stays outside the fit.
pub struct TrainingJob<'a> {
    pub exchange: &'a str,
    pub symbol: &'a str,
    pub method: FairValueMethod,
    pub window: TrainingWindow,
    pub evidence: EvidencePolicy,
    pub artifact_path: &'a Path,
    pub target: TrainingTarget,
}

/// Fits the selected probability model.
///
/// `quotes` is read only for [`TrainingTarget::QuoteFill`]. Market-outcome
/// training ignores it, including a missing quote file.
pub fn train_history<H, Q>(
    history: &H,
    quotes: &Q,
    job: &TrainingJob<'_>,
) -> Result<TrainOutcome, TrainFailure>
where
    H: HistoricalBooks,
    Q: QuoteHistory,
{
    let books = load_books(history, job)?;
    match job.target {
        TrainingTarget::QuoteFill { horizon_ns } => {
            let loaded = load_quotes(quotes, job)?;
            if loaded.is_empty() {
                return Ok(TrainOutcome::SkippedNoLabels);
            }
            runtime_info!(
                books = books.len(),
                quotes = loaded.len(),
                "quote-fill window loaded"
            );
            fit_quote(&books, &loaded, job, horizon_ns)
        }
        TrainingTarget::MarketOutcome(definition) => {
            runtime_info!(books = books.len(), "market-outcome window loaded");
            fit_market(&books, job, definition)
        }
    }
}

fn load_books<H: HistoricalBooks>(
    history: &H,
    job: &TrainingJob<'_>,
) -> Result<Vec<BookRecord>, TrainFailure> {
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
    Ok(books)
}

fn load_quotes<Q: QuoteHistory>(
    quotes: &Q,
    job: &TrainingJob<'_>,
) -> Result<Vec<QuoteEvent>, TrainFailure> {
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
    Ok(loaded)
}

fn fit_quote(
    books: &[BookRecord],
    quotes: &[QuoteEvent],
    job: &TrainingJob<'_>,
    horizon_ns: i64,
) -> Result<TrainOutcome, TrainFailure> {
    if books.is_empty() {
        return Ok(TrainOutcome::InsufficientData {
            reason: "no published books in the training window".into(),
        });
    }
    let config = DatasetConfig::new(job.method, horizon_ns).map_err(|_| TrainFailure::Dataset)?;
    let set = match build_quote_fill_examples(books, quotes, config) {
        Ok(set) => set,
        Err(err) => {
            return Ok(TrainOutcome::Rejected {
                reason: err.to_string(),
            })
        }
    };
    runtime_info!(
        input_rows = set.input_rows,
        positive_rows = set.positive_rows,
        negative_rows = set.negative_rows,
        censored_rows = set.censored_rows,
        "quote-fill label summary"
    );
    if set.examples.is_empty() {
        return Ok(TrainOutcome::InsufficientData {
            reason: "no usable fill-within-horizon rows; censored observations were excluded"
                .into(),
        });
    }
    let mut rows = set.examples;
    rows.sort_by_key(|row| row.decision_ns);
    let decisions: Vec<i64> = rows.iter().map(|row| row.decision_ns).collect();
    let resolved: Vec<i64> = rows.iter().map(|row| row.label_resolved_ns).collect();
    let split = match checked_split(&decisions, &resolved, job.evidence, |index| {
        rows[index].filled_within_horizon
    }) {
        Ok(split) => split,
        Err(outcome) => return Ok(outcome),
    };
    let provenance = rows[split.train_index[0]].provenance;
    if rows.iter().any(|row| row.provenance != provenance) {
        return Ok(TrainOutcome::Rejected {
            reason: "mixed label provenance".into(),
        });
    }
    let train: Vec<_> = split
        .train_index
        .iter()
        .map(|index| FillSample {
            features: rows[*index].features,
            positive: rows[*index].filled_within_horizon,
        })
        .collect();
    let holdout: Vec<_> = split
        .holdout_index
        .iter()
        .map(|index| FillSample {
            features: rows[*index].features,
            positive: rows[*index].filled_within_horizon,
        })
        .collect();
    let label_source = match provenance {
        LabelProvenance::Observed => LabelSource::ObservedQuote,
        LabelProvenance::Simulated => LabelSource::SimulatedQuote,
    };
    finish_fit(FitRequest {
        job,
        family: ModelFamily::QuoteFill,
        horizon_ns,
        threshold_bps: None,
        label_source,
        split: &split,
        train: &train,
        holdout: &holdout,
        raw_rows: set.input_rows,
    })
}

fn fit_market(
    books: &[BookRecord],
    job: &TrainingJob<'_>,
    definition: crate::model::market::MarketOutcomeDefinition,
) -> Result<TrainOutcome, TrainFailure> {
    if books.is_empty() {
        return Ok(TrainOutcome::InsufficientData {
            reason: "no published books in the training window".into(),
        });
    }
    let config =
        MarketDatasetConfig::new(job.method, definition).map_err(|_| TrainFailure::Dataset)?;
    let mut rows = match build_market_examples(books, config) {
        Ok(rows) => rows,
        Err(err) => {
            return Ok(TrainOutcome::Rejected {
                reason: err.to_string(),
            })
        }
    };
    rows.sort_by_key(|row| row.decision_ns);
    let decisions: Vec<i64> = rows.iter().map(|row| row.decision_ns).collect();
    let horizons: Vec<i64> = rows.iter().map(|row| row.horizon_snapshot_ns).collect();
    let split = match checked_split(&decisions, &horizons, job.evidence, |index| {
        rows[index].up_move
    }) {
        Ok(split) => split,
        Err(outcome) => return Ok(outcome),
    };
    let train = samples_from(&rows, &split.train_index);
    let holdout = samples_from(&rows, &split.holdout_index);
    finish_fit(FitRequest {
        job,
        family: ModelFamily::MarketUpMove,
        horizon_ns: definition.horizon_ns(),
        threshold_bps: Some(definition.threshold_bps()),
        label_source: LabelSource::PublicBook,
        split: &split,
        train: &train,
        holdout: &holdout,
        raw_rows: rows.len(),
    })
}

fn samples_from(
    rows: &[MarketExample],
    indexes: &[usize],
) -> Vec<FillSample<MARKET_FEATURE_COUNT>> {
    indexes
        .iter()
        .map(|index| FillSample {
            features: rows[*index].features,
            positive: rows[*index].up_move,
        })
        .collect()
}

struct FitRequest<'a, const N: usize> {
    job: &'a TrainingJob<'a>,
    family: ModelFamily,
    horizon_ns: i64,
    threshold_bps: Option<f64>,
    label_source: LabelSource,
    split: &'a ChronoSplit,
    train: &'a [FillSample<N>],
    holdout: &'a [FillSample<N>],
    raw_rows: usize,
}

fn finish_fit<const N: usize>(request: FitRequest<'_, N>) -> Result<TrainOutcome, TrainFailure> {
    let FitRequest {
        job,
        family,
        horizon_ns,
        threshold_bps,
        label_source,
        split,
        train,
        holdout,
        raw_rows,
    } = request;
    let train_config = TrainConfig {
        learning_rate: 0.2,
        epochs: 200,
    };
    let mut model = LogisticRegression::<N>::zeros();
    let after = model
        .train(train, train_config)
        .map_err(|_| TrainFailure::Fit)?;
    if !after.is_finite() {
        return Ok(TrainOutcome::Rejected {
            reason: "reference training produced a non-finite loss".into(),
        });
    }
    let positives = train.iter().filter(|row| row.positive).count();
    let rate = positives as f64 / train.len() as f64;
    let logit = (rate / (1.0 - rate)).ln();
    let baseline =
        LogisticRegression::<N>::from_weights(logit, [0.0; N]).map_err(|_| TrainFailure::Fit)?;
    let learned_loss = model.mean_loss(holdout).map_err(|_| TrainFailure::Fit)?;
    let baseline_loss = baseline.mean_loss(holdout).map_err(|_| TrainFailure::Fit)?;
    let learned_scores = probability_pairs(&model, holdout)?;
    let baseline_scores = probability_pairs(&baseline, holdout)?;
    let learned = score_probabilities(&learned_scores).map_err(|_| TrainFailure::Fit)?;
    let baseline_metrics = score_probabilities(&baseline_scores).map_err(|_| TrainFailure::Fit)?;
    if !learned_loss.is_finite()
        || !baseline_loss.is_finite()
        || !log_loss_accepts(learned_loss, baseline_loss)
    {
        return Ok(TrainOutcome::Rejected {
            reason:
                "holdout log loss is not strictly better than the constant training-rate baseline"
                    .into(),
        });
    }
    let (train_positives, train_negatives) = counts(train);
    let (holdout_positives, holdout_negatives) = counts(holdout);
    let report = ValidationReport {
        rows_before_purge: split.rows_before_purge as u64,
        purged_training_rows: split.purged as u64,
        train_examples: train.len() as u64,
        holdout_examples: holdout.len() as u64,
        holdout_start_ns: split.holdout_start_ns,
        train_positives,
        train_negatives,
        holdout_positives,
        holdout_negatives,
        log_loss: learned.log_loss,
        baseline_log_loss: baseline_metrics.log_loss,
        brier: learned.brier,
        baseline_brier: baseline_metrics.brier,
        calibration: learned.bins,
    };
    let dataset_id = format!(
        "{}-{}-{}-{}-{}",
        family_name(family),
        job.method.name(),
        job.window.from_ns,
        job.window.to_ns,
        raw_rows
    );
    let artifact = ModelArtifact::validated(
        family,
        job.method,
        horizon_ns,
        threshold_bps,
        &model,
        ArtifactMetadata {
            label_source,
            window_from_ns: job.window.from_ns,
            window_to_ns: job.window.to_ns,
            dataset_id: dataset_id.clone(),
        },
        report,
    )
    .map_err(|_| TrainFailure::Artifact)?;
    artifact
        .write_json(job.artifact_path)
        .map_err(|_| TrainFailure::Artifact)?;
    runtime_info!(dataset_id = %dataset_id, "validated model written");
    Ok(TrainOutcome::Validated { dataset_id })
}

fn family_name(family: ModelFamily) -> &'static str {
    match family {
        ModelFamily::QuoteFill => "quote-fill",
        ModelFamily::MarketUpMove => "market-up-move",
    }
}

fn counts<const N: usize>(rows: &[FillSample<N>]) -> (u64, u64) {
    let positives = rows.iter().filter(|row| row.positive).count() as u64;
    (positives, rows.len() as u64 - positives)
}

fn probability_pairs<const N: usize>(
    model: &LogisticRegression<N>,
    rows: &[FillSample<N>],
) -> Result<Vec<(f64, bool)>, TrainFailure> {
    let mut pairs = Vec::with_capacity(rows.len());
    for row in rows {
        let probability = model
            .predict_fill_probability(&row.features)
            .map_err(|_| TrainFailure::Fit)?
            .get();
        pairs.push((probability, row.positive));
    }
    Ok(pairs)
}

struct ChronoSplit {
    holdout_start_ns: i64,
    rows_before_purge: usize,
    purged: usize,
    train_index: Vec<usize>,
    holdout_index: Vec<usize>,
}

fn checked_split(
    decision_ns: &[i64],
    horizon_ns: &[i64],
    policy: EvidencePolicy,
    mut is_positive: impl FnMut(usize) -> bool,
) -> Result<ChronoSplit, TrainOutcome> {
    let Some(split) = chronological_split(decision_ns, horizon_ns, policy.holdout_percent) else {
        return Err(TrainOutcome::InsufficientData {
            reason: "chronological split produced an empty train or holdout slice".into(),
        });
    };
    let train_positives = split
        .train_index
        .iter()
        .filter(|index| is_positive(**index))
        .count();
    let train_negatives = split.train_index.len() - train_positives;
    let holdout_positives = split
        .holdout_index
        .iter()
        .filter(|index| is_positive(**index))
        .count();
    let holdout_negatives = split.holdout_index.len() - holdout_positives;
    if split.train_index.is_empty()
        || split.train_index.len() < policy.min_train_examples
        || train_positives < policy.min_train_positives
        || train_negatives < policy.min_train_negatives
        || split.holdout_index.len() < policy.min_holdout_examples
        || holdout_positives < policy.min_holdout_positives
        || holdout_negatives < policy.min_holdout_negatives
    {
        return Err(TrainOutcome::InsufficientData {
            reason: "train or holdout does not meet the evidence policy after purging".into(),
        });
    }
    Ok(split)
}

/// Rows must already be sorted by decision time.
///
/// A training candidate is dropped when its horizon snapshot is at or after
/// the first holdout decision. It is not moved into the holdout.
fn chronological_split(
    decision_ns: &[i64],
    horizon_ns: &[i64],
    holdout_percent: u8,
) -> Option<ChronoSplit> {
    let n = decision_ns.len();
    if n != horizon_ns.len() {
        return None;
    }
    let train_end = n * (100 - u32::from(holdout_percent)) as usize / 100;
    if train_end == 0 || train_end >= n {
        return None;
    }
    let holdout_start_ns = decision_ns[train_end];
    let mut train_index = Vec::new();
    let mut purged = 0usize;
    for (index, horizon) in horizon_ns.iter().enumerate().take(train_end) {
        if *horizon >= holdout_start_ns {
            purged += 1;
        } else {
            train_index.push(index);
        }
    }
    Some(ChronoSplit {
        holdout_start_ns,
        rows_before_purge: train_end,
        purged,
        train_index,
        holdout_index: (train_end..n).collect(),
    })
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
    use crate::data::training::example::{
        DecisionTimeInputs, LabelProvenance, LabelTiming, ObservedOutcome, QuoteOutcome,
        TrainingExample,
    };
    use crate::data::training::history::MemoryBooks;
    use crate::domain::book::{BookRecord, BookStamp, Instrument, Level, BOOK_DEPTH};
    use crate::domain::market::QuoteContext;
    use crate::model::market::MarketOutcomeDefinition;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn book(ns: i64) -> BookRecord {
        let mut bids = [Level::EMPTY; BOOK_DEPTH];
        let mut asks = [Level::EMPTY; BOOK_DEPTH];
        bids[0] = Level {
            price: 99.9,
            quantity: 1.0,
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

    fn quote_job(path: &Path) -> TrainingJob<'_> {
        TrainingJob {
            exchange: "binance",
            symbol: "BTCUSDT",
            method: FairValueMethod::Microprice,
            window: TrainingWindow {
                from_ns: 0,
                to_ns: 5_000,
            },
            evidence: EvidencePolicy::default(),
            artifact_path: path,
            target: TrainingTarget::QuoteFill { horizon_ns: 500 },
        }
    }

    #[test]
    fn trainer_uses_the_historical_source_boundary() {
        let books = [book(1_000), book(2_000)];
        let history = MemoryBooks {
            exchange: "binance",
            symbol: "BTCUSDT",
            books: &books,
        };
        let quote = QuoteEvent::try_new(
            1_000,
            QuoteContext::new(99.8, 1.0).unwrap(),
            LabelProvenance::Simulated,
            QuoteOutcome::Filled {
                fill_ns: 1_200,
                execution_price: 99.8,
                fees: 0.0,
            },
        )
        .unwrap();
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
        let outcome = train_history(&history, &quotes, &quote_job(&path)).unwrap();
        assert!(matches!(outcome, TrainOutcome::InsufficientData { .. }));
        assert!(!path.is_file());
    }

    #[test]
    fn market_training_does_not_read_quote_history() {
        let books = [book(1_000), book(2_000)];
        let history = MemoryBooks {
            exchange: "binance",
            symbol: "BTCUSDT",
            books: &books,
        };
        let path = std::env::temp_dir().join(format!(
            "mm-market-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let quotes = crate::data::training::quotes::MemoryQuoteHistory {
            exchange: "binance",
            symbol: "BTCUSDT",
            quotes: &[],
        };
        let job = TrainingJob {
            target: TrainingTarget::MarketOutcome(
                MarketOutcomeDefinition::new_up_move(500, 1.0).unwrap(),
            ),
            ..quote_job(&path)
        };
        let outcome = train_history(&history, &quotes, &job).unwrap();
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
        let job = quote_job(&path);
        let empty = crate::data::training::quotes::MemoryQuoteHistory {
            exchange: "binance",
            symbol: "BTCUSDT",
            quotes: &[],
        };
        assert!(matches!(
            train_history(&history, &empty, &job).unwrap(),
            TrainOutcome::SkippedNoLabels
        ));
        let quote = QuoteEvent::try_new(
            1_000,
            QuoteContext::new(99.8, 1.0).unwrap(),
            LabelProvenance::Observed,
            QuoteOutcome::Unfilled {
                observed_until_ns: 2_000,
            },
        )
        .unwrap();
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
    ) -> TrainingExample {
        let market = FairValueMethod::Microprice
            .derive_top(99.9, 1.0, 100.1, 1.0)
            .unwrap();
        TrainingExample::new(
            DecisionTimeInputs::new(market, QuoteContext::new(99.8, 1.0).unwrap()),
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

    fn boundary_rows(crossing_pnl: f64) -> Vec<TrainingExample> {
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

    fn times(rows: &[TrainingExample]) -> (Vec<i64>, Vec<i64>) {
        (
            rows.iter().map(|row| row.timing().decision_ns).collect(),
            rows.iter()
                .map(|row| row.timing().horizon_snapshot_ns)
                .collect(),
        )
    }

    #[test]
    fn a_label_that_finishes_before_the_holdout_stays_in_training() {
        let rows = boundary_rows(1.0);
        let (decision, horizon) = times(&rows);
        let split = chronological_split(&decision, &horizon, 20).unwrap();
        assert_eq!(split.holdout_start_ns, 800);
        assert!(split.train_index.iter().any(|index| {
            rows[*index].timing().decision_ns == 600
                && rows[*index].timing().horizon_snapshot_ns == 650
        }));
    }

    #[test]
    fn a_label_that_reaches_the_holdout_is_purged_and_not_moved() {
        let rows = boundary_rows(9.0);
        let (decision, horizon) = times(&rows);
        let split = chronological_split(&decision, &horizon, 20).unwrap();
        assert_eq!(split.purged, 1);
        assert_eq!(split.rows_before_purge, 8);
        assert_eq!(split.train_index.len(), 7);
        assert!(split
            .train_index
            .iter()
            .all(|index| rows[*index].timing().decision_ns != 700));
        assert!(split
            .holdout_index
            .iter()
            .all(|index| rows[*index].timing().decision_ns >= 800));
        assert_eq!(split.holdout_index.len(), 2);
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
            min_train_positives: 1,
            min_train_negatives: 1,
            min_holdout_examples: 2,
            min_holdout_positives: 1,
            min_holdout_negatives: 1,
            ..EvidencePolicy::default()
        };
        let (decision, horizon) = times(&rows);
        let before = chronological_split(&decision, &horizon, policy.holdout_percent).unwrap();
        assert!(before.rows_before_purge >= policy.min_train_examples);
        assert!(before.train_index.len() < policy.min_train_examples);
        assert!(matches!(
            checked_split(&decision, &horizon, policy, |index| matches!(
                rows[index].outcome(),
                ObservedOutcome::Filled { .. }
            )),
            Err(TrainOutcome::InsufficientData { .. })
        ));
    }

    #[test]
    fn market_outcome_rows_use_the_same_purge_boundary() {
        let decisions = [0, 100, 200, 700, 800, 900];
        let horizons = [50, 150, 250, 800, 850, 950];
        let split = chronological_split(&decisions, &horizons, 20).unwrap();
        assert_eq!(split.holdout_start_ns, 800);
        assert!(!split.train_index.contains(&3));
        assert_eq!(split.holdout_index, vec![4, 5]);
    }

    #[test]
    fn changing_a_purged_rows_holdout_label_does_not_change_the_fitted_slice() {
        let low = boundary_rows(0.1);
        let high = boundary_rows(50.0);
        let (low_d, low_h) = times(&low);
        let (high_d, high_h) = times(&high);
        let low_split = chronological_split(&low_d, &low_h, 20).unwrap();
        let high_split = chronological_split(&high_d, &high_h, 20).unwrap();
        let low_train: Vec<_> = low_split
            .train_index
            .iter()
            .map(|index| low[*index])
            .collect();
        let high_train: Vec<_> = high_split
            .train_index
            .iter()
            .map(|index| high[*index])
            .collect();
        assert_eq!(low_train, high_train);
        assert!(low_split.train_index.iter().all(|index| {
            low[*index].timing().horizon_snapshot_ns < low_split.holdout_start_ns
        }));
    }

    #[test]
    fn an_early_fill_stays_trainable_when_only_its_horizon_crosses_the_holdout() {
        let decisions = [0, 8_000, 9_000, 10_000, 10_500];
        let resolved = [100, 10_500, 10_500, 10_200, 11_000];
        let split = chronological_split(&decisions, &resolved, 20).unwrap();
        assert_eq!(split.holdout_start_ns, 10_500);
        assert!(split.train_index.contains(&3));
        assert!(!split.train_index.contains(&1));
        assert!(!split.train_index.contains(&2));
        assert_eq!(split.holdout_index, vec![4]);
    }
}
