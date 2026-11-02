//! Decision-time features for the market-outcome model.
//!
//! Every coordinate is a function of the current touch. The future reference
//! price is not an input.

use crate::domain::market::MarketState;
use crate::model::feature::v1::{FeatureError, FeatureVector, PRICE_RATIO_SCALE};

/// Schema version stored on a market-outcome artifact.
pub const MARKET_FEATURE_SCHEMA_VERSION: u32 = 1;

/// Number of market-outcome features.
pub const MARKET_FEATURE_COUNT: usize = 3;

/// Stable names for logs. Not read during inference.
pub const MARKET_FEATURE_NAMES: [&str; MARKET_FEATURE_COUNT] = [
    "exchange_spread_bps",
    "top_quantity_imbalance",
    "microprice_minus_midpoint_bps",
];

/// Features from one already-estimated market state.
///
/// No heap allocation. `fair` is the configured reference price on `market`.
pub fn extract_market_features(
    market: &MarketState,
) -> Result<FeatureVector<MARKET_FEATURE_COUNT>, FeatureError> {
    let fair = market.fair_value();
    let bid = market.best_bid_price();
    let ask = market.best_ask_price();
    let bid_qty = market.best_bid_quantity();
    let ask_qty = market.best_ask_quantity();
    let depth = bid_qty + ask_qty;
    if !(fair.is_finite() && fair > 0.0 && depth.is_finite() && depth > 0.0) {
        return Err(FeatureError::NonPositiveDepth);
    }
    let midpoint = (bid + ask) / 2.0;
    let microprice = (ask * bid_qty + bid * ask_qty) / depth;
    FeatureVector::new([
        (ask - bid) / fair * PRICE_RATIO_SCALE,
        (bid_qty - ask_qty) / depth,
        (microprice - midpoint) / fair * PRICE_RATIO_SCALE,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::fair_value::FairValueMethod;

    #[test]
    fn features_use_only_the_current_touch() {
        let market = FairValueMethod::Microprice
            .derive_top(99.0, 3.0, 101.0, 1.0)
            .unwrap();
        let values = extract_market_features(&market).unwrap();
        let fair = market.fair_value();
        let spread = (101.0 - 99.0) / fair * PRICE_RATIO_SCALE;
        assert!((values.as_array()[0] - spread).abs() < 1e-9);
        assert!((values.as_array()[1] - 0.5).abs() < 1e-12);
        let microprice = (101.0 * 3.0 + 99.0 * 1.0) / 4.0;
        let midpoint = 100.0;
        let gap = (microprice - midpoint) / fair * PRICE_RATIO_SCALE;
        assert!((values.as_array()[2] - gap).abs() < 1e-9);
    }
}
