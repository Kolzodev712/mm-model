//! Model A rows: decision-time features and a fill-within-horizon label.
//!
//! The label comes from quote lifecycle timing. A later book is not read, and
//! mark-to-fair P&L is not computed. Public books do not create fill labels.

use std::fmt;

use crate::data::training::example::{LabelProvenance, QuoteOutcome};
use crate::domain::book::BookRecord;
use crate::domain::market::{MarketState, QuoteContext};
use crate::domain::units::{Fee, Price};
use crate::model::fair_value::{FairValueError, FairValueMethod};
use crate::model::feature::v1::{extract_features, FeatureVector, FEATURE_COUNT};

/// One submitted or explicitly simulated quote, with lifecycle timing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuoteEvent {
    decision_ns: i64,
    quote: QuoteContext,
    provenance: LabelProvenance,
    outcome: QuoteOutcome,
}

impl QuoteEvent {
    /// Stores a quote outcome. Timestamps that run backwards are rejected.
    pub fn try_new(
        decision_ns: i64,
        quote: QuoteContext,
        provenance: LabelProvenance,
        outcome: QuoteOutcome,
    ) -> Result<Self, crate::data::training::example::ExampleError> {
        use crate::data::training::example::ExampleError;
        if decision_ns < 0 {
            return Err(ExampleError::InvalidTiming);
        }
        match outcome {
            QuoteOutcome::Filled {
                fill_ns,
                execution_price,
                fees,
            } => {
                if fill_ns < decision_ns {
                    return Err(ExampleError::InvalidTiming);
                }
                Price::new(execution_price).map_err(|_| ExampleError::InvalidFill)?;
                Fee::new(fees).map_err(|_| ExampleError::InvalidFill)?;
            }
            QuoteOutcome::Unfilled { observed_until_ns } => {
                if observed_until_ns < decision_ns {
                    return Err(ExampleError::InvalidTiming);
                }
            }
        }
        Ok(Self {
            decision_ns,
            quote,
            provenance,
            outcome,
        })
    }

    #[must_use]
    pub fn decision_ns(self) -> i64 {
        self.decision_ns
    }

    #[must_use]
    pub fn quote(self) -> QuoteContext {
        self.quote
    }

    #[must_use]
    pub fn provenance(self) -> LabelProvenance {
        self.provenance
    }

    #[must_use]
    pub fn outcome(self) -> QuoteOutcome {
        self.outcome
    }
}

/// A resolved fill-within-`H` label, or a censored observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HorizonClass {
    /// The label is known. `fill_ns <= t + H` is positive, inclusive of the deadline.
    Resolved {
        positive: bool,
        label_resolved_ns: i64,
    },
    /// Observation ended before `t + H`. This is not a negative label.
    Censored,
}

/// Classifies one lifecycle against `deadline = decision_ns + horizon_ns`.
///
/// A fill at the deadline is positive. A later fill is a negative whose
/// resolution time is the deadline, not the fill time. An unfilled record
/// that stops before the deadline is censored.
pub fn label_within_horizon(
    decision_ns: i64,
    horizon_ns: i64,
    outcome: QuoteOutcome,
) -> Result<HorizonClass, BuildError> {
    if horizon_ns <= 0 {
        return Err(BuildError::NonPositiveHorizon);
    }
    let deadline = decision_ns
        .checked_add(horizon_ns)
        .ok_or(BuildError::HorizonOverflow)?;
    Ok(match outcome {
        QuoteOutcome::Filled { fill_ns, .. } => {
            if fill_ns <= deadline {
                HorizonClass::Resolved {
                    positive: true,
                    label_resolved_ns: fill_ns,
                }
            } else {
                HorizonClass::Resolved {
                    positive: false,
                    label_resolved_ns: deadline,
                }
            }
        }
        QuoteOutcome::Unfilled { observed_until_ns } => {
            if observed_until_ns >= deadline {
                HorizonClass::Resolved {
                    positive: false,
                    label_resolved_ns: deadline,
                }
            } else {
                HorizonClass::Censored
            }
        }
    })
}

/// Fair-value definition, horizon, and decision-book staleness for one build.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DatasetConfig {
    method: FairValueMethod,
    horizon_ns: i64,
    max_decision_staleness_ns: i64,
}

const DEFAULT_DECISION_STALENESS_NS: i64 = 5_000_000_000;

impl DatasetConfig {
    /// `horizon_ns` is `H` and must be strictly positive.
    pub fn new(method: FairValueMethod, horizon_ns: i64) -> Result<Self, BuildError> {
        Self::with_staleness(method, horizon_ns, DEFAULT_DECISION_STALENESS_NS)
    }

    /// Same as [`Self::new`] with an explicit decision-book staleness cap.
    pub fn with_staleness(
        method: FairValueMethod,
        horizon_ns: i64,
        max_decision_staleness_ns: i64,
    ) -> Result<Self, BuildError> {
        if horizon_ns <= 0 {
            return Err(BuildError::NonPositiveHorizon);
        }
        if max_decision_staleness_ns < 0 {
            return Err(BuildError::InvalidTimingLimit);
        }
        Ok(Self {
            method,
            horizon_ns,
            max_decision_staleness_ns,
        })
    }

    /// Microprice with horizon `H`.
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

/// One Model A row. Features use only the decision-time book and quote context.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuoteFillExample {
    pub decision_ns: i64,
    pub label_resolved_ns: i64,
    pub features: FeatureVector<FEATURE_COUNT>,
    pub filled_within_horizon: bool,
    pub provenance: LabelProvenance,
    pub method: FairValueMethod,
}

/// Usable rows plus a count of censored quotes that were excluded.
#[derive(Debug, Clone, PartialEq)]
pub struct QuoteFillDataset {
    pub method: FairValueMethod,
    pub horizon_ns: i64,
    pub input_rows: usize,
    pub positive_rows: usize,
    pub negative_rows: usize,
    pub censored_rows: usize,
    pub examples: Vec<QuoteFillExample>,
}

/// Joins sorted books with quote lifecycles.
///
/// `books` must be sorted by local receive time. Only the latest valid book at
/// or before the decision is read. No book at `t + H` is required.
pub fn build_quote_fill_examples(
    books: &[BookRecord],
    quotes: &[QuoteEvent],
    config: DatasetConfig,
) -> Result<QuoteFillDataset, BuildError> {
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

    let mut examples = Vec::new();
    let mut positive_rows = 0usize;
    let mut negative_rows = 0usize;
    let mut censored_rows = 0usize;
    for quote in quotes {
        match label_within_horizon(quote.decision_ns, config.horizon_ns, quote.outcome)? {
            HorizonClass::Censored => censored_rows += 1,
            HorizonClass::Resolved {
                positive,
                label_resolved_ns,
            } => {
                if let QuoteOutcome::Filled {
                    execution_price, ..
                } = quote.outcome
                {
                    if execution_price > quote.quote.price() {
                        return Err(BuildError::InvalidExample);
                    }
                }
                let (decision, fair_value) =
                    decision_book(books, quote.decision_ns, config.method)?;
                let decision_age = quote
                    .decision_ns
                    .checked_sub(decision.stamp().local_receive_ns)
                    .filter(|age| *age >= 0)
                    .ok_or(BuildError::InvalidTiming)?;
                if decision_age > config.max_decision_staleness_ns {
                    return Err(BuildError::StaleDecision {
                        decision_ns: quote.decision_ns,
                        age_ns: decision_age,
                    });
                }
                let bid = decision
                    .best_bid()
                    .ok_or(BuildError::InvalidBook(FairValueError::NoBid))?;
                let ask = decision
                    .best_ask()
                    .ok_or(BuildError::InvalidBook(FairValueError::NoAsk))?;
                let market = MarketState::from_estimate(
                    config.method,
                    fair_value,
                    bid.price,
                    bid.quantity,
                    ask.price,
                    ask.quantity,
                );
                let features = extract_features(&market, &quote.quote)
                    .map_err(|_| BuildError::InvalidExample)?;
                if positive {
                    positive_rows += 1;
                } else {
                    negative_rows += 1;
                }
                examples.push(QuoteFillExample {
                    decision_ns: quote.decision_ns,
                    label_resolved_ns,
                    features,
                    filled_within_horizon: positive,
                    provenance,
                    method: config.method,
                });
            }
        }
    }
    Ok(QuoteFillDataset {
        method: config.method,
        horizon_ns: config.horizon_ns,
        input_rows: quotes.len(),
        positive_rows,
        negative_rows,
        censored_rows,
        examples,
    })
}

/// Why a Model A dataset could not be built.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuildError {
    /// No submitted or simulated quote outcomes were provided.
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
    /// A selected snapshot was rejected by the estimator.
    InvalidBook(FairValueError),
    /// The execution price is above the quote, or features could not be built.
    InvalidExample,
    /// A timing limit was negative.
    InvalidTimingLimit,
    /// Timestamp arithmetic overflowed or went backwards.
    InvalidTiming,
    /// The decision book was older than the configured staleness limit.
    StaleDecision { decision_ns: i64, age_ns: i64 },
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
            Self::InvalidBook(err) => write!(f, "book snapshot was rejected: {err}"),
            Self::InvalidExample => f.write_str("quote outcome is not a valid training row"),
            Self::InvalidTimingLimit => f.write_str("timing limits must be non-negative"),
            Self::InvalidTiming => f.write_str("book timestamps are out of order or overflowed"),
            Self::StaleDecision { decision_ns, age_ns } => {
                write!(f, "decision book for {decision_ns} is stale by {age_ns} ns")
            }
        }
    }
}

impl std::error::Error for BuildError {}

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

fn is_time_sorted<T>(rows: &[T], time_of: impl Fn(&T) -> i64) -> bool {
    rows.windows(2)
        .all(|pair| time_of(&pair[0]) <= time_of(&pair[1]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::book::{BookRecord, BookStamp, Instrument, Level, BOOK_DEPTH};

    fn book(local_ns: i64) -> BookRecord {
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
                local_receive_ns: local_ns,
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

    fn event(decision: i64, outcome: QuoteOutcome) -> QuoteEvent {
        QuoteEvent::try_new(
            decision,
            QuoteContext::new(99.8, 1.0).unwrap(),
            LabelProvenance::Observed,
            outcome,
        )
        .unwrap()
    }

    fn filled(fill_ns: i64) -> QuoteOutcome {
        QuoteOutcome::Filled {
            fill_ns,
            execution_price: 99.8,
            fees: 0.0,
        }
    }

    #[test]
    fn a_fill_before_or_at_the_horizon_is_positive() {
        let early = label_within_horizon(0, 1_000, filled(500)).unwrap();
        assert_eq!(
            early,
            HorizonClass::Resolved {
                positive: true,
                label_resolved_ns: 500,
            }
        );
        let exact = label_within_horizon(0, 1_000, filled(1_000)).unwrap();
        assert_eq!(
            exact,
            HorizonClass::Resolved {
                positive: true,
                label_resolved_ns: 1_000,
            }
        );
    }

    #[test]
    fn a_fill_after_the_horizon_is_negative_at_the_deadline() {
        let label = label_within_horizon(0, 1_000, filled(1_500)).unwrap();
        assert_eq!(
            label,
            HorizonClass::Resolved {
                positive: false,
                label_resolved_ns: 1_000,
            }
        );
    }

    #[test]
    fn an_unfilled_quote_observed_through_the_horizon_is_negative() {
        let label = label_within_horizon(
            0,
            1_000,
            QuoteOutcome::Unfilled {
                observed_until_ns: 1_500,
            },
        )
        .unwrap();
        assert_eq!(
            label,
            HorizonClass::Resolved {
                positive: false,
                label_resolved_ns: 1_000,
            }
        );
    }

    #[test]
    fn an_observation_that_ends_before_the_horizon_is_censored() {
        let books = [book(0)];
        let quote = event(
            0,
            QuoteOutcome::Unfilled {
                observed_until_ns: 500,
            },
        );
        let set =
            build_quote_fill_examples(&books, &[quote], DatasetConfig::microprice(1_000).unwrap())
                .unwrap();
        assert!(set.examples.is_empty());
        assert_eq!(set.input_rows, 1);
        assert_eq!(set.censored_rows, 1);
        assert_eq!(set.positive_rows, 0);
        assert_eq!(set.negative_rows, 0);
    }

    #[test]
    fn timestamps_before_the_decision_are_rejected() {
        let quote = QuoteContext::new(99.8, 1.0).unwrap();
        assert!(QuoteEvent::try_new(0, quote, LabelProvenance::Observed, filled(-1)).is_err());
        assert!(QuoteEvent::try_new(
            10,
            quote,
            LabelProvenance::Observed,
            QuoteOutcome::Unfilled {
                observed_until_ns: 9,
            },
        )
        .is_err());
        assert!(QuoteEvent::try_new(-1, quote, LabelProvenance::Observed, filled(0)).is_err());
    }

    #[test]
    fn model_a_does_not_need_a_book_at_the_horizon() {
        let books = [book(0)];
        let set = build_quote_fill_examples(
            &books,
            &[event(0, filled(200))],
            DatasetConfig::microprice(1_000).unwrap(),
        )
        .unwrap();
        assert_eq!(set.examples.len(), 1);
        assert!(set.examples[0].filled_within_horizon);
        assert_eq!(set.examples[0].label_resolved_ns, 200);
        assert_eq!(set.positive_rows, 1);
    }

    #[test]
    fn book_history_without_quotes_does_not_create_labels() {
        let err = build_quote_fill_examples(&[], &[], DatasetConfig::microprice(500).unwrap())
            .unwrap_err();
        assert_eq!(err, BuildError::NoFillLabels);
        assert!(err.to_string().contains("unsubmitted"));
    }

    #[test]
    fn mixed_quote_provenance_is_rejected() {
        let quote = QuoteContext::new(99.8, 1.0).unwrap();
        let observed = QuoteEvent::try_new(
            0,
            quote,
            LabelProvenance::Observed,
            QuoteOutcome::Unfilled {
                observed_until_ns: 1_000,
            },
        )
        .unwrap();
        let simulated = QuoteEvent::try_new(
            0,
            quote,
            LabelProvenance::Simulated,
            QuoteOutcome::Unfilled {
                observed_until_ns: 1_000,
            },
        )
        .unwrap();
        assert_eq!(
            build_quote_fill_examples(
                &[book(0)],
                &[observed, simulated],
                DatasetConfig::microprice(500).unwrap()
            ),
            Err(BuildError::MixedProvenance)
        );
    }

    #[test]
    fn stale_decision_book_is_rejected() {
        let quote = event(
            10_000,
            QuoteOutcome::Unfilled {
                observed_until_ns: 20_000,
            },
        );
        let config =
            DatasetConfig::with_staleness(FairValueMethod::Microprice, 500, 1_000).unwrap();
        assert_eq!(
            build_quote_fill_examples(&[book(0)], &[quote], config),
            Err(BuildError::StaleDecision {
                decision_ns: 10_000,
                age_ns: 10_000,
            })
        );
    }
}
