//! Fixed-size decision-time features for the learned baselines.
//!
//! Extraction reads [`MarketState`] and [`CandidateQuote`] only. It does not
//! see fills, execution prices, or the fair value at a later horizon.
//!
//! The reference price in these features is the book-derived estimate stored
//! on [`MarketState`], not a separately supplied value. Microprice already
//! uses the best bid and ask quantities, so it is correlated with
//! [`TOP_QUANTITY_IMBALANCE`]. Both are kept so that overlap can be measured
//! later rather than removed by assumption.
//!
//! The first set is four relative quantities. Candidate size is not a feature:
//! a conditional-P&L label is the total mark-to-reference P&L of that fill, so
//! a model fit on mixed sizes does not see size explicitly. That is a
//! limitation of this feature set, not a hidden scaling inside expected value.
//!
//! Inference writes one stack array. It does not allocate.

use std::fmt;

use crate::decision::{CandidateQuote, MarketState};

/// Number of features in the first learned baseline.
pub const FEATURE_COUNT: usize = 4;

/// `(fair_value - quote_price) / fair_value`. Dimensionless.
///
/// Positive when a buy is bid below the book-derived reference price.
/// Source: decision-time reference price and candidate price.
pub const QUOTE_DISTANCE_FROM_FAIR: usize = 0;

/// `(best_ask_price - best_bid_price) / fair_value`. Dimensionless.
///
/// The exchange spread scaled by the reference price.
/// Source: decision-time best bid, best ask, and the reference price.
/// A crossed book never reaches this function: the estimator rejects it.
pub const EXCHANGE_SPREAD: usize = 1;

/// `(quote_price - best_bid_price) / fair_value`. Dimensionless.
///
/// Buy-side distance from the current best bid. Zero joins the touch.
/// Distance to the ask is omitted because it is determined by this value and
/// the spread. Source: candidate price, best bid, and fair value.
pub const QUOTE_DISTANCE_FROM_BEST_BID: usize = 2;

/// `(bid_qty - ask_qty) / (bid_qty + ask_qty)`. Dimensionless, in `[-1, 1]`
/// when the denominator is positive.
///
/// The estimator rejects non-positive best quantities, so the denominator is
/// positive for every [`MarketState`]. The zero branch remains so extraction
/// itself does not divide by zero. Source: decision-time best bid and ask
/// quantities. Microprice uses those same quantities.
pub const TOP_QUANTITY_IMBALANCE: usize = 3;

/// Stable names for logs and dataset columns. Not read during inference.
pub const FEATURE_NAMES: [&str; FEATURE_COUNT] = [
    "quote_distance_from_fair",
    "exchange_spread",
    "quote_distance_from_best_bid",
    "top_quantity_imbalance",
];

/// Stack-allocated feature row of length `N`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FeatureVector<const N: usize> {
    values: [f64; N],
}

impl<const N: usize> FeatureVector<N> {
    /// Copies a finite row onto the stack.
    pub fn new(values: [f64; N]) -> Result<Self, FeatureError> {
        if values.iter().any(|value| !value.is_finite()) {
            return Err(FeatureError::NonFinite);
        }
        Ok(Self { values })
    }

    /// Row as a fixed array.
    #[must_use]
    pub fn as_array(&self) -> &[f64; N] {
        &self.values
    }

    /// Dot product with a weight array of the same length.
    #[must_use]
    pub fn dot(&self, weights: &[f64; N]) -> f64 {
        let mut sum = 0.0;
        for (value, weight) in self.values.iter().zip(weights.iter()) {
            sum += value * weight;
        }
        sum
    }
}

/// Builds the first feature row from a decision-time market and buy quote.
///
/// No heap allocation. The reference price was estimated once on the market
/// state and is only read here.
#[must_use]
pub fn extract_features(
    market: &MarketState,
    quote: &CandidateQuote,
) -> FeatureVector<FEATURE_COUNT> {
    let fair = market.fair_value();
    debug_assert!(fair > 0.0 && fair.is_finite());

    let bid_qty = market.best_bid_quantity();
    let ask_qty = market.best_ask_quantity();
    let depth = bid_qty + ask_qty;
    let imbalance = if depth > 0.0 {
        (bid_qty - ask_qty) / depth
    } else {
        0.0
    };

    // Inputs are finite and fair is positive, so these quotients are finite.
    FeatureVector {
        values: [
            (fair - quote.price()) / fair,
            (market.best_ask_price() - market.best_bid_price()) / fair,
            (quote.price() - market.best_bid_price()) / fair,
            imbalance,
        ],
    }
}

/// A feature row contained a non-finite value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureError {
    /// At least one coordinate was NaN or infinite.
    NonFinite,
}

impl fmt::Display for FeatureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("feature values must be finite")
    }
}

impl std::error::Error for FeatureError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fair_value::{FairValueError, FairValueMethod};

    fn market(bid_qty: f64, ask_qty: f64) -> MarketState {
        FairValueMethod::Microprice
            .derive_top(99.9, bid_qty, 100.1, ask_qty)
            .unwrap()
    }

    #[test]
    fn known_market_matches_definitions() {
        let state = market(2.0, 1.0);
        let quote = CandidateQuote::new(99.8, 1.0).unwrap();
        let fair = state.fair_value();
        let features = extract_features(&state, &quote);
        let values = features.as_array();
        assert_eq!(values.len(), FEATURE_COUNT);
        assert_eq!(FEATURE_NAMES.len(), FEATURE_COUNT);
        assert!((values[QUOTE_DISTANCE_FROM_FAIR] - ((fair - 99.8) / fair)).abs() < 1e-12);
        assert!((values[EXCHANGE_SPREAD] - ((100.1 - 99.9) / fair)).abs() < 1e-12);
        assert!((values[QUOTE_DISTANCE_FROM_BEST_BID] - ((99.8 - 99.9) / fair)).abs() < 1e-12);
        assert!((values[TOP_QUANTITY_IMBALANCE] - (1.0 / 3.0)).abs() < 1e-12);
        assert!(values.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn non_positive_touch_quantity_is_rejected_before_feature_scaling() {
        let err = FairValueMethod::Microprice.derive_top(99.9, 0.0, 100.1, 0.0);
        assert_eq!(err, Err(FairValueError::NonPositiveQuantity));
    }
}
