//! Offline join from recorded books and quote outcomes to training rows.
//!
//! ```text
//! book at t        -> FairValueMethod -> FV_t
//! book at t + H    -> same method     -> FV_(t+H)
//! actual execution + FV_(t+H)         -> mark-to-reference P&L
//! ```
//!
//! No external fair-value series is read. Quote events carry the execution
//! price and fees, not a horizon price.
//!
//! # Alignment
//!
//! Timestamps are local receive times, unix nanoseconds.
//!
//! - Decision book: the latest snapshot with time `<= t` that the configured
//!   estimator accepts. Snapshots after `t` are not used. If several valid
//!   snapshots share that latest time, the last one in the sorted slice is used.
//! - Horizon book: the first snapshot with time `>= t + H` that the estimator
//!   accepts. Snapshots before `t + H` are not used, including a snapshot that
//!   is closer to the target but early. If several snapshots share the first
//!   valid time, the earliest in the slice is used.
//!
//! `H` must be strictly positive. The recorded offset is
//! `horizon_snapshot_time - (t + H)` and is therefore zero or positive.
//!
//! A book print still does not say whether a quote that was never sent would
//! have filled. Empty quote input returns [`BuildError::NoFillLabels`].

use std::fmt;

use crate::data::training::example::{
    DecisionTimeInputs, FillFacts, LabelProvenance, LabelTiming, ObservedOutcome, TrainingExample,
};
use crate::domain::book::BookRecord;
use crate::domain::market::{CandidateQuote, MarketState};
use crate::domain::outcome::mark_to_fair_pnl;
use crate::model::fair_value::{FairValueError, FairValueMethod};

/// One submitted or explicitly simulated quote.
///
/// The fill facts do not include a horizon reference price.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuoteEvent {
    decision_ns: i64,
    quote: CandidateQuote,
    provenance: LabelProvenance,
    facts: FillFacts,
}

impl QuoteEvent {
    /// Stores a quote outcome. The outcome is not inferred from the book.
    #[must_use]
    pub fn new(
        decision_ns: i64,
        quote: CandidateQuote,
        provenance: LabelProvenance,
        facts: FillFacts,
    ) -> Self {
        Self {
            decision_ns,
            quote,
            provenance,
            facts,
        }
    }

    #[must_use]
    pub fn decision_ns(self) -> i64 {
        self.decision_ns
    }

    #[must_use]
    pub fn quote(self) -> CandidateQuote {
        self.quote
    }

    #[must_use]
    pub fn provenance(self) -> LabelProvenance {
        self.provenance
    }

    #[must_use]
    pub fn facts(self) -> FillFacts {
        self.facts
    }
}

/// Fair-value definition and horizon used for one build.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DatasetConfig {
    method: FairValueMethod,
    horizon_ns: i64,
    max_decision_staleness_ns: i64,
    max_horizon_slippage_ns: i64,
}

const DEFAULT_DECISION_STALENESS_NS: i64 = 5_000_000_000;
const DEFAULT_HORIZON_SLIPPAGE_NS: i64 = 5_000_000_000;

impl DatasetConfig {
    /// `horizon_ns` is `H` and must be strictly positive.
    pub fn new(method: FairValueMethod, horizon_ns: i64) -> Result<Self, BuildError> {
        Self::with_limits(
            method,
            horizon_ns,
            DEFAULT_DECISION_STALENESS_NS,
            DEFAULT_HORIZON_SLIPPAGE_NS,
        )
    }

    /// Same as [`Self::new`] with explicit staleness and slippage caps.
    ///
    /// A decision book older than `max_decision_staleness_ns`, or a horizon
    /// book later than `t + H` by more than `max_horizon_slippage_ns`, is rejected.
    pub fn with_limits(
        method: FairValueMethod,
        horizon_ns: i64,
        max_decision_staleness_ns: i64,
        max_horizon_slippage_ns: i64,
    ) -> Result<Self, BuildError> {
        if horizon_ns <= 0 {
            return Err(BuildError::NonPositiveHorizon);
        }
        if max_decision_staleness_ns < 0 || max_horizon_slippage_ns < 0 {
            return Err(BuildError::InvalidTimingLimit);
        }
        Ok(Self {
            method,
            horizon_ns,
            max_decision_staleness_ns,
            max_horizon_slippage_ns,
        })
    }

    /// Microprice, the default experimental estimator, with horizon `H`.
    pub fn microprice(horizon_ns: i64) -> Result<Self, BuildError> {
        Self::new(FairValueMethod::Microprice, horizon_ns)
    }

    #[must_use]
    pub fn method(self) -> FairValueMethod {
        self.method
    }

    #[must_use]
    pub fn horizon_ns(self) -> i64 {
        self.horizon_ns
    }
}

/// Rows produced by one estimator and one horizon.
///
/// [`Self::method`] is the definition that produced every reference price in
/// `examples`. Do not score these rows with a model trained under another method.
#[derive(Debug, Clone, PartialEq)]
pub struct TrainingSet {
    method: FairValueMethod,
    requested_horizon_ns: i64,
    examples: Vec<TrainingExample>,
}

impl TrainingSet {
    #[must_use]
    pub fn method(&self) -> FairValueMethod {
        self.method
    }

    #[must_use]
    pub fn requested_horizon_ns(&self) -> i64 {
        self.requested_horizon_ns
    }

    #[must_use]
    pub fn examples(&self) -> &[TrainingExample] {
        &self.examples
    }
}

/// Joins sorted book prints with quote outcomes.
///
/// `books` must be sorted by local receive time, ascending. The same
/// [`DatasetConfig::method`] is applied to the decision snapshot and the
/// horizon snapshot.
pub fn build_training_examples(
    books: &[BookRecord],
    quotes: &[QuoteEvent],
    config: DatasetConfig,
) -> Result<TrainingSet, BuildError> {
    if quotes.is_empty() {
        return Err(BuildError::NoFillLabels);
    }
    let provenance = quotes[0].provenance();
    if quotes.iter().any(|quote| quote.provenance() != provenance) {
        return Err(BuildError::MixedProvenance);
    }
    if !is_time_sorted(books, |book| book.stamp().local_receive_ns) {
        return Err(BuildError::UnsortedBooks);
    }

    let mut examples = Vec::with_capacity(quotes.len());
    for quote in quotes {
        let target_ns = quote
            .decision_ns()
            .checked_add(config.horizon_ns)
            .ok_or(BuildError::HorizonOverflow)?;
        let (decision, decision_fair_value) =
            decision_book(books, quote.decision_ns(), config.method)?;
        let (horizon, horizon_fair_value) = horizon_book(books, target_ns, config.method)?;
        let bid = decision
            .best_bid()
            .ok_or(BuildError::InvalidBook(FairValueError::NoBid))?;
        let ask = decision
            .best_ask()
            .ok_or(BuildError::InvalidBook(FairValueError::NoAsk))?;
        let market = MarketState::from_estimate(
            config.method,
            decision_fair_value,
            bid.price,
            bid.quantity,
            ask.price,
            ask.quantity,
        );
        let decision_age = quote
            .decision_ns()
            .checked_sub(decision.stamp().local_receive_ns)
            .filter(|age| *age >= 0)
            .ok_or(BuildError::InvalidTiming)?;
        if decision_age > config.max_decision_staleness_ns {
            return Err(BuildError::StaleDecision {
                decision_ns: quote.decision_ns(),
                age_ns: decision_age,
            });
        }
        let slippage = horizon
            .stamp()
            .local_receive_ns
            .checked_sub(target_ns)
            .filter(|slip| *slip >= 0)
            .ok_or(BuildError::InvalidTiming)?;
        if slippage > config.max_horizon_slippage_ns {
            return Err(BuildError::LateHorizon {
                target_ns,
                slippage_ns: slippage,
            });
        }
        if let FillFacts::Filled {
            execution_price, ..
        } = quote.facts()
        {
            if execution_price > quote.quote().price() {
                return Err(BuildError::InvalidExample);
            }
        }
        let outcome =
            outcome_from_facts(quote.facts(), quote.quote().quantity(), horizon_fair_value)?;
        let timing = LabelTiming {
            decision_ns: quote.decision_ns(),
            decision_snapshot_ns: decision.stamp().local_receive_ns,
            requested_horizon_ns: config.horizon_ns,
            horizon_snapshot_ns: horizon.stamp().local_receive_ns,
        };
        examples.push(
            TrainingExample::new(
                DecisionTimeInputs::new(market, quote.quote()),
                outcome,
                quote.provenance(),
                config.method,
                timing,
            )
            .map_err(|_| BuildError::InvalidExample)?,
        );
    }
    Ok(TrainingSet {
        method: config.method,
        requested_horizon_ns: config.horizon_ns,
        examples,
    })
}

/// Why a training set could not be built.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuildError {
    /// No submitted or simulated quote outcomes were provided.
    ///
    /// Order-book data alone cannot label an unsubmitted quote.
    NoFillLabels,
    /// Observed and simulated quotes were in the same slice.
    MixedProvenance,
    /// Book prints were not sorted by local receive time.
    UnsortedBooks,
    /// `H` was not strictly positive.
    NonPositiveHorizon,
    /// `t + H` overflowed the timestamp type.
    HorizonOverflow,
    /// No snapshot at or before the decision time could be estimated.
    MissingBook { decision_ns: i64 },
    /// Snapshots at or before the decision existed, but none were a valid book.
    InvalidDecisionBook { decision_ns: i64 },
    /// No valid snapshot existed at or after `t + H`.
    MissingHorizon { target_ns: i64 },
    /// A selected snapshot was rejected by the estimator.
    InvalidBook(FairValueError),
    /// The execution price or fees could not form a label.
    InvalidExample,
    /// A timing limit was negative.
    InvalidTimingLimit,
    /// Timestamp arithmetic overflowed or went backwards.
    InvalidTiming,
    /// The decision book was older than the configured staleness limit.
    StaleDecision { decision_ns: i64, age_ns: i64 },
    /// The horizon book was later than the configured slippage limit.
    LateHorizon { target_ns: i64, slippage_ns: i64 },
}

impl fmt::Display for BuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoFillLabels => f.write_str(
                "fill-model training requires actual quote outcomes; a book print does not say whether an unsubmitted quote would have filled",
            ),
            Self::MixedProvenance => {
                f.write_str("observed and simulated quote outcomes must be built separately")
            }
            Self::UnsortedBooks => f.write_str("book prints must be sorted by local receive time"),
            Self::NonPositiveHorizon => f.write_str("evaluation horizon must be strictly positive"),
            Self::HorizonOverflow => f.write_str("decision time plus horizon overflowed"),
            Self::MissingBook { decision_ns } => {
                write!(f, "no book print at or before {decision_ns}")
            }
            Self::InvalidDecisionBook { decision_ns } => {
                write!(f, "no valid book print at or before {decision_ns}")
            }
            Self::MissingHorizon { target_ns } => {
                write!(f, "no valid book print at or after {target_ns}")
            }
            Self::InvalidBook(err) => write!(f, "book snapshot was rejected: {err}"),
            Self::InvalidExample => f.write_str("quote outcome is not a valid training row"),
            Self::InvalidTimingLimit => f.write_str("timing limits must be non-negative"),
            Self::InvalidTiming => f.write_str("book timestamps are out of order or overflowed"),
            Self::StaleDecision { decision_ns, age_ns } => {
                write!(f, "decision book for {decision_ns} is stale by {age_ns} ns")
            }
            Self::LateHorizon {
                target_ns,
                slippage_ns,
            } => write!(
                f,
                "horizon book for {target_ns} slipped by {slippage_ns} ns"
            ),
        }
    }
}

impl std::error::Error for BuildError {}

fn outcome_from_facts(
    facts: FillFacts,
    quantity: f64,
    horizon_fair_value: f64,
) -> Result<ObservedOutcome, BuildError> {
    match facts {
        FillFacts::Unfilled => Ok(ObservedOutcome::Unfilled),
        FillFacts::Filled {
            execution_price,
            fees,
        } => {
            if !execution_price.is_finite()
                || execution_price <= 0.0
                || !fees.is_finite()
                || fees < 0.0
            {
                return Err(BuildError::InvalidExample);
            }
            Ok(ObservedOutcome::Filled {
                pnl: mark_to_fair_pnl(horizon_fair_value, execution_price, quantity, fees),
                horizon_fair_value,
            })
        }
    }
}

fn decision_book(
    books: &[BookRecord],
    decision_ns: i64,
    method: FairValueMethod,
) -> Result<(&BookRecord, f64), BuildError> {
    let end = books.partition_point(|book| book.stamp().local_receive_ns <= decision_ns);
    if end == 0 {
        return Err(BuildError::MissingBook { decision_ns });
    }
    for book in books[..end].iter().rev() {
        if let Ok(fair_value) = method.estimate_book(book) {
            return Ok((book, fair_value));
        }
    }
    Err(BuildError::InvalidDecisionBook { decision_ns })
}

fn horizon_book(
    books: &[BookRecord],
    target_ns: i64,
    method: FairValueMethod,
) -> Result<(&BookRecord, f64), BuildError> {
    let start = books.partition_point(|book| book.stamp().local_receive_ns < target_ns);
    for book in &books[start..] {
        if let Ok(fair_value) = method.estimate_book(book) {
            return Ok((book, fair_value));
        }
    }
    Err(BuildError::MissingHorizon { target_ns })
}

fn is_time_sorted<T>(rows: &[T], time_of: impl Fn(&T) -> i64) -> bool {
    rows.windows(2)
        .all(|pair| time_of(&pair[0]) <= time_of(&pair[1]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::training::example::{LabelProvenance, ObservedOutcome};
    use crate::domain::book::{BookRecord, BookStamp, Instrument, Level, BOOK_DEPTH};
    use crate::domain::market::CandidateQuote;
    use crate::domain::outcome::mark_to_fair_pnl;
    use crate::model::fair_value::FairValueMethod;

    fn book(local_ns: i64, bid: f64, bid_qty: f64, ask: f64, ask_qty: f64) -> BookRecord {
        let mut bids = [Level::EMPTY; BOOK_DEPTH];
        let mut asks = [Level::EMPTY; BOOK_DEPTH];
        bids[0] = Level {
            price: bid,
            quantity: bid_qty,
        };
        asks[0] = Level {
            price: ask,
            quantity: ask_qty,
        };
        BookRecord::try_new(
            Instrument::new("binance", "BTCUSDT").unwrap(),
            BookStamp {
                local_receive_ns: local_ns,
                exchange_event_ns: None,
                update_id: Some(local_ns as u64),
            },
            bids,
            1,
            asks,
            1,
        )
        .unwrap()
    }

    fn timeline() -> Vec<BookRecord> {
        vec![
            book(900, 99.0, 1.0, 99.2, 1.0),
            book(1_000, 99.90, 2.0, 100.10, 1.0),
            book(1_200, 110.0, 1.0, 111.0, 1.0),
            book(1_500, 100.2, 1.0, 100.0, 1.0),
            book(1_600, 101.0, 1.0, 101.4, 3.0),
            book(2_000, 102.0, 1.0, 103.0, 1.0),
        ]
    }

    fn filled_quote() -> QuoteEvent {
        QuoteEvent::new(
            1_000,
            CandidateQuote::new(99.8, 2.0).unwrap(),
            LabelProvenance::Observed,
            FillFacts::Filled {
                execution_price: 99.8,
                fees: 0.5,
            },
        )
    }

    #[test]
    fn book_history_without_quotes_does_not_create_labels() {
        let err =
            build_training_examples(&timeline(), &[], DatasetConfig::microprice(500).unwrap())
                .unwrap_err();
        assert_eq!(err, BuildError::NoFillLabels);
        assert!(err.to_string().contains("unsubmitted"));
    }

    #[test]
    fn horizon_fair_value_comes_from_the_later_book_only() {
        let config = DatasetConfig::microprice(500).unwrap();
        let set = build_training_examples(&timeline(), &[filled_quote()], config).unwrap();
        assert_eq!(set.method(), FairValueMethod::Microprice);
        assert_eq!(set.method().name(), "microprice");
        assert_eq!(set.requested_horizon_ns(), 500);
        assert_eq!(set.examples().len(), 1);

        let row = &set.examples()[0];
        let decision_fv = FairValueMethod::Microprice
            .estimate(99.90, 2.0, 100.10, 1.0)
            .unwrap();
        let horizon_fv = FairValueMethod::Microprice
            .estimate(101.0, 1.0, 101.4, 3.0)
            .unwrap();
        assert_eq!(row.inputs().market().fair_value(), decision_fv);
        assert_eq!(row.fair_value_method(), FairValueMethod::Microprice);
        assert_eq!(row.timing().decision_ns, 1_000);
        assert_eq!(row.timing().decision_snapshot_ns, 1_000);
        assert_eq!(row.timing().requested_horizon_ns, 500);
        assert_eq!(row.timing().horizon_snapshot_ns, 1_600);
        assert_eq!(row.timing().horizon_offset_ns().unwrap(), 100);
        match row.outcome() {
            ObservedOutcome::Filled {
                pnl,
                horizon_fair_value,
            } => {
                assert_eq!(horizon_fair_value, horizon_fv);
                assert_eq!(pnl, mark_to_fair_pnl(horizon_fv, 99.8, 2.0, 0.5));
                assert_ne!(horizon_fair_value, decision_fv);
            }
            ObservedOutcome::Unfilled => panic!("filled quote should keep its pnl"),
        }
    }

    #[test]
    fn changing_the_estimator_changes_both_reference_prices() {
        let micro = build_training_examples(
            &timeline(),
            &[filled_quote()],
            DatasetConfig::microprice(500).unwrap(),
        )
        .unwrap();
        let mid = build_training_examples(
            &timeline(),
            &[filled_quote()],
            DatasetConfig::new(FairValueMethod::Midpoint, 500).unwrap(),
        )
        .unwrap();
        assert_eq!(mid.method(), FairValueMethod::Midpoint);
        let micro_fv = micro.examples()[0].inputs().market().fair_value();
        let mid_fv = mid.examples()[0].inputs().market().fair_value();
        assert!((mid_fv - ((99.90 + 100.10) / 2.0)).abs() < 1e-12);
        assert_ne!(micro_fv, mid_fv);
        let micro_horizon = match micro.examples()[0].outcome() {
            ObservedOutcome::Filled {
                horizon_fair_value, ..
            } => horizon_fair_value,
            ObservedOutcome::Unfilled => unreachable!(),
        };
        let mid_horizon = match mid.examples()[0].outcome() {
            ObservedOutcome::Filled {
                horizon_fair_value, ..
            } => horizon_fair_value,
            ObservedOutcome::Unfilled => unreachable!(),
        };
        assert!((mid_horizon - ((101.0 + 101.4) / 2.0)).abs() < 1e-12);
        assert_ne!(micro_horizon, mid_horizon);
    }

    #[test]
    fn early_snapshot_is_not_used_as_the_horizon() {
        let books = [
            book(1_000, 99.9, 1.0, 100.1, 1.0),
            book(1_200, 110.0, 1.0, 111.0, 1.0),
        ];
        let err = build_training_examples(
            &books,
            &[filled_quote()],
            DatasetConfig::microprice(500).unwrap(),
        )
        .unwrap_err();
        assert_eq!(err, BuildError::MissingHorizon { target_ns: 1_500 });
    }

    #[test]
    fn mixed_quote_provenance_is_rejected() {
        let quote = CandidateQuote::new(99.8, 1.0).unwrap();
        let quotes = [
            QuoteEvent::new(1_000, quote, LabelProvenance::Observed, FillFacts::Unfilled),
            QuoteEvent::new(
                1_000,
                quote,
                LabelProvenance::Simulated,
                FillFacts::Unfilled,
            ),
        ];
        assert_eq!(
            build_training_examples(
                &timeline(),
                &quotes,
                DatasetConfig::microprice(500).unwrap()
            ),
            Err(BuildError::MixedProvenance)
        );
    }

    #[test]
    fn stale_decision_book_is_rejected() {
        let books = [
            book(0, 99.9, 1.0, 100.1, 1.0),
            book(20_000, 99.9, 1.0, 100.1, 1.0),
        ];
        let quote = QuoteEvent::new(
            10_000,
            CandidateQuote::new(99.8, 1.0).unwrap(),
            LabelProvenance::Observed,
            FillFacts::Unfilled,
        );
        let config =
            DatasetConfig::with_limits(FairValueMethod::Microprice, 500, 1_000, 1_000).unwrap();
        assert_eq!(
            build_training_examples(&books, &[quote], config),
            Err(BuildError::StaleDecision {
                decision_ns: 10_000,
                age_ns: 10_000,
            })
        );
    }

    #[test]
    fn late_horizon_book_is_rejected() {
        let books = [
            book(1_000, 99.9, 1.0, 100.1, 1.0),
            book(5_000, 101.0, 1.0, 101.2, 1.0),
        ];
        let config =
            DatasetConfig::with_limits(FairValueMethod::Microprice, 500, 5_000, 100).unwrap();
        assert_eq!(
            build_training_examples(&books, &[filled_quote()], config),
            Err(BuildError::LateHorizon {
                target_ns: 1_500,
                slippage_ns: 3_500,
            })
        );
    }
}
