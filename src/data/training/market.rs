//! Model B rows. Labels come from a later public book. Features do not.

use crate::domain::book::BookRecord;
use crate::domain::market::MarketState;
use crate::model::fair_value::{FairValueError, FairValueMethod};
use crate::model::feature::market::{extract_market_features, MARKET_FEATURE_COUNT};
use crate::model::feature::v1::FeatureVector;
use crate::model::market::{MarketOutcomeDefinition, OutcomeError};

/// One market-outcome row.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MarketExample {
    pub decision_ns: i64,
    pub horizon_snapshot_ns: i64,
    pub features: FeatureVector<MARKET_FEATURE_COUNT>,
    pub up_move: bool,
    pub return_bps: f64,
}

/// Timing and target used while scanning a book history.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MarketDatasetConfig {
    pub method: FairValueMethod,
    pub definition: MarketOutcomeDefinition,
    pub max_horizon_slippage_ns: i64,
}

impl MarketDatasetConfig {
    pub const DEFAULT_SLIPPAGE_NS: i64 = 5_000_000_000;

    pub fn new(
        method: FairValueMethod,
        definition: MarketOutcomeDefinition,
    ) -> Result<Self, MarketBuildError> {
        Self::with_slippage(method, definition, Self::DEFAULT_SLIPPAGE_NS)
    }

    pub fn with_slippage(
        method: FairValueMethod,
        definition: MarketOutcomeDefinition,
        max_horizon_slippage_ns: i64,
    ) -> Result<Self, MarketBuildError> {
        if max_horizon_slippage_ns < 0 {
            return Err(MarketBuildError::Slippage);
        }
        Ok(Self {
            method,
            definition,
            max_horizon_slippage_ns,
        })
    }
}

/// Builds rows from books sorted by local receive time.
///
/// A book without a later snapshot inside the slippage limit is skipped.
/// Unsorted input is an error. No quote history is read.
pub fn build_market_examples(
    books: &[BookRecord],
    config: MarketDatasetConfig,
) -> Result<Vec<MarketExample>, MarketBuildError> {
    if books
        .windows(2)
        .any(|pair| pair[0].stamp().local_receive_ns > pair[1].stamp().local_receive_ns)
    {
        return Err(MarketBuildError::Unsorted);
    }
    let mut rows = Vec::new();
    for (index, book) in books.iter().enumerate() {
        let Ok(now) = config.method.estimate_book(book) else {
            continue;
        };
        let decision_ns = book.stamp().local_receive_ns;
        let Some(target_ns) = decision_ns.checked_add(config.definition.horizon_ns()) else {
            return Err(MarketBuildError::Horizon);
        };
        let Some((horizon_ns, future)) = future_reference(books, index, target_ns, config.method)
        else {
            continue;
        };
        let slippage = horizon_ns - target_ns;
        if slippage > config.max_horizon_slippage_ns {
            continue;
        }
        let market = market_state(config.method, book, now)?;
        let features = extract_market_features(&market).map_err(|_| MarketBuildError::Features)?;
        let return_bps = MarketOutcomeDefinition::return_bps(now, future)?;
        let up_move = config.definition.label(now, future)?;
        rows.push(MarketExample {
            decision_ns,
            horizon_snapshot_ns: horizon_ns,
            features,
            up_move,
            return_bps,
        });
    }
    Ok(rows)
}

fn market_state(
    method: FairValueMethod,
    book: &BookRecord,
    fair_value: f64,
) -> Result<MarketState, MarketBuildError> {
    let bid = book
        .best_bid()
        .ok_or(MarketBuildError::Book(FairValueError::NoBid))?;
    let ask = book
        .best_ask()
        .ok_or(MarketBuildError::Book(FairValueError::NoAsk))?;
    Ok(MarketState::from_estimate(
        method,
        fair_value,
        bid.price,
        bid.quantity,
        ask.price,
        ask.quantity,
    ))
}

fn future_reference(
    books: &[BookRecord],
    decision_index: usize,
    target_ns: i64,
    method: FairValueMethod,
) -> Option<(i64, f64)> {
    let start = books.partition_point(|book| book.stamp().local_receive_ns < target_ns);
    for book in books.iter().skip(start.max(decision_index + 1)) {
        if let Ok(fair_value) = method.estimate_book(book) {
            return Some((book.stamp().local_receive_ns, fair_value));
        }
    }
    None
}

/// Why market-outcome rows could not be built.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarketBuildError {
    Unsorted,
    Horizon,
    Slippage,
    Features,
    Book(FairValueError),
    Outcome(OutcomeError),
}

impl From<OutcomeError> for MarketBuildError {
    fn from(value: OutcomeError) -> Self {
        Self::Outcome(value)
    }
}

impl std::fmt::Display for MarketBuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsorted => f.write_str("book prints must be sorted by local receive time"),
            Self::Horizon => f.write_str("decision time plus horizon overflowed"),
            Self::Slippage => f.write_str("horizon slippage limit must be non-negative"),
            Self::Features => f.write_str("market features could not be extracted"),
            Self::Book(err) => write!(f, "book snapshot was rejected: {err}"),
            Self::Outcome(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for MarketBuildError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::book::{BookStamp, Instrument, Level, BOOK_DEPTH};

    fn book(ns: i64, bid: f64, ask: f64) -> BookRecord {
        let mut bids = [Level::EMPTY; BOOK_DEPTH];
        let mut asks = [Level::EMPTY; BOOK_DEPTH];
        bids[0] = Level {
            price: bid,
            quantity: 1.0,
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

    fn config(slippage: i64) -> MarketDatasetConfig {
        MarketDatasetConfig::with_slippage(
            FairValueMethod::Midpoint,
            MarketOutcomeDefinition::new_up_move(1_000, 1.0).unwrap(),
            slippage,
        )
        .unwrap()
    }

    #[test]
    fn the_label_uses_the_later_book_and_features_do_not() {
        let books = [
            book(0, 100.0, 100.2),
            book(500, 90.0, 90.2),
            book(1_000, 100.2, 100.4),
        ];
        let rows = build_market_examples(&books, config(0)).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].decision_ns, 0);
        assert_eq!(rows[0].horizon_snapshot_ns, 1_000);
        assert!(rows[0].up_move);
        let changed = [
            book(0, 100.0, 100.2),
            book(500, 90.0, 90.2),
            book(1_000, 99.0, 99.2),
        ];
        let other = build_market_examples(&changed, config(0)).unwrap();
        assert_eq!(rows[0].features, other[0].features);
        assert!(!other[0].up_move);
    }

    #[test]
    fn a_horizon_book_past_the_slippage_limit_is_skipped() {
        let books = [book(0, 100.0, 100.2), book(5_000, 101.0, 101.2)];
        assert!(build_market_examples(&books, config(1_000))
            .unwrap()
            .is_empty());
        assert_eq!(
            build_market_examples(&books, config(4_000)).unwrap().len(),
            1
        );
    }
}
