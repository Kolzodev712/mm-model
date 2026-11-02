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
//! Schema version 1 has five relative features. The conditional-P&L label is
//! total P&L, which scales with size, so candidate quantity is included as a
//! fraction of displayed top-of-book size.
//!
//! Inference writes one stack array. It does not allocate.

use std::fmt;

use crate::domain::market::{CandidateQuote, MarketState};

/// Schema version stored in model artifacts.
///
/// Version 2 keeps the five ratios and expresses the three price ratios in
/// basis points (`ratio * 10_000`). Version 1 weights are not compatible:
/// those ratios sit near `1e-4`, and the reference gradient step cannot move
/// them. A model file from version 1 is rejected.
pub const FEATURE_SCHEMA_V1: u32 = 2;

/// Multiplier applied to the three price ratios so a basis point is about 1.
pub const PRICE_RATIO_SCALE: f64 = 10_000.0;

/// Number of features in [`FEATURE_SCHEMA_V1`].
pub const FEATURE_COUNT: usize = 5;

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

/// `(quote_price - best_bid_price) / fair_value`, then scaled to basis points.
///
/// Buy-side distance from the current best bid. Zero joins the touch.
/// Under midpoint this is an exact linear combination of distance-from-fair
/// and spread. Under microprice it is determined by those two features and
/// the quantity imbalance. The weight on this coordinate is not separately
/// identified. It stays in the row so older comparisons remain explicit.
/// Source: candidate price, best bid, and fair value.
pub const QUOTE_DISTANCE_FROM_BEST_BID: usize = 2;

/// `(bid_qty - ask_qty) / (bid_qty + ask_qty)`. Dimensionless, in `[-1, 1]`
/// when the denominator is positive.
///
/// The estimator rejects non-positive best quantities, so the denominator is
/// positive for every [`MarketState`]. The zero branch remains so extraction
/// itself does not divide by zero. Source: decision-time best bid and ask
/// quantities. Microprice uses those same quantities.
pub const TOP_QUANTITY_IMBALANCE: usize = 3;

/// `candidate_quantity / (best_bid_quantity + best_ask_quantity)`. Dimensionless.
///
/// Decision-time only. Present because the conditional-P&L label is total P&L,
/// which changes with size. The denominator is the same displayed size used by
/// microprice and imbalance, so this feature is correlated with both.
pub const RELATIVE_CANDIDATE_QUANTITY: usize = 4;

/// Stable names for logs and dataset columns. Not read during inference.
pub const FEATURE_NAMES: [&str; FEATURE_COUNT] = [
    "quote_distance_from_fair",
    "exchange_spread",
    "quote_distance_from_best_bid",
    "top_quantity_imbalance",
    "relative_candidate_quantity",
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
pub fn extract_features(
    market: &MarketState,
    quote: &CandidateQuote,
) -> Result<FeatureVector<FEATURE_COUNT>, FeatureError> {
    let fair = market.fair_value();
    let bid_qty = market.best_bid_quantity();
    let ask_qty = market.best_ask_quantity();
    let depth = bid_qty + ask_qty;
    if !(fair.is_finite() && fair > 0.0 && depth.is_finite() && depth > 0.0) {
        return Err(FeatureError::NonPositiveDepth);
    }
    let imbalance = (bid_qty - ask_qty) / depth;
    let relative_quantity = quote.quantity() / depth;
    FeatureVector::new([
        (fair - quote.price()) / fair * PRICE_RATIO_SCALE,
        (market.best_ask_price() - market.best_bid_price()) / fair * PRICE_RATIO_SCALE,
        (quote.price() - market.best_bid_price()) / fair * PRICE_RATIO_SCALE,
        imbalance,
        relative_quantity,
    ])
}

/// A feature row contained a non-finite value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureError {
    /// At least one coordinate was NaN or infinite.
    NonFinite,
    /// Top-of-book size was not a positive finite number.
    NonPositiveDepth,
}

impl fmt::Display for FeatureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::NonFinite => "feature values must be finite",
            Self::NonPositiveDepth => "top-of-book quantity must be positive to scale features",
        };
        f.write_str(message)
    }
}

impl std::error::Error for FeatureError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::fair_value::{FairValueError, FairValueMethod};

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
        let features = extract_features(&state, &quote).unwrap();
        let values = features.as_array();
        assert_eq!(values.len(), FEATURE_COUNT);
        assert_eq!(FEATURE_NAMES.len(), FEATURE_COUNT);
        assert!(
            (values[QUOTE_DISTANCE_FROM_FAIR] - ((fair - 99.8) / fair * PRICE_RATIO_SCALE)).abs()
                < 1e-9
        );
        assert!(
            (values[EXCHANGE_SPREAD] - ((100.1 - 99.9) / fair * PRICE_RATIO_SCALE)).abs() < 1e-9
        );
        assert!(
            (values[QUOTE_DISTANCE_FROM_BEST_BID] - ((99.8 - 99.9) / fair * PRICE_RATIO_SCALE))
                .abs()
                < 1e-9
        );
        assert!((values[TOP_QUANTITY_IMBALANCE] - (1.0 / 3.0)).abs() < 1e-12);
        assert!((values[RELATIVE_CANDIDATE_QUANTITY] - (1.0 / 3.0)).abs() < 1e-12);
        assert_eq!(FEATURE_SCHEMA_V1, 2);
        assert!(values.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn non_positive_touch_quantity_is_rejected_before_feature_scaling() {
        let err = FairValueMethod::Microprice.derive_top(99.9, 0.0, 100.1, 0.0);
        assert_eq!(err, Err(FairValueError::NonPositiveQuantity));
    }
}
