//! Order-book reference price used by the learned pipeline.
//!
//! This is a Binance-order-book-derived fair-value estimate, not an
//! objectively correct or independent asset value. Decision-time features and
//! this estimate are computed from the same book, so the estimator adds no
//! outside information. A later multi-venue composite, a predictive model, or
//! a production engine can replace [`FairValueMethod`] without changing the
//! logistic or linear models.
//!
//! Prices and quantities are `f64` in this educational build. NaN and
//! infinities are rejected. Estimates are not used as map keys. A later
//! fixed-point book can feed the same method enum.
//!
//! The estimator has no clock. A caller that already knows a snapshot is stale
//! must skip it. Nothing here invents a replacement price for an invalid book.
//!
//! # Midpoint
//!
//! `FV_mid = (P_bid + P_ask) / 2`.
//!
//! Uses the best bid and ask prices only. Displayed size does not move the
//! result. Size is still required to be strictly positive so a one-sided or
//! empty touch is not treated as a market.
//!
//! # Microprice
//!
//! `FV_micro = (P_ask * Q_bid + P_bid * Q_ask) / (Q_bid + Q_ask)`.
//!
//! `P_bid` and `Q_bid` are the best bid price and displayed quantity.
//! `P_ask` and `Q_ask` are the best ask price and displayed quantity.
//! The larger displayed quantity pulls the reference toward the opposite
//! price: more size on the bid moves the estimate toward the ask, and more
//! size on the ask moves it toward the bid.
//!
//! Estimation reads the best levels through references. It does not allocate.

use std::fmt;

use crate::domain::book::{BookRecord, Level};
use crate::domain::market::MarketState;

pub use crate::domain::market::FairValueMethod;

impl FairValueMethod {
    /// Estimates a reference price from the best bid and ask.
    ///
    /// O(1). No allocation. Identical inputs return the same `f64` bits.
    pub fn estimate(
        self,
        bid_price: f64,
        bid_qty: f64,
        ask_price: f64,
        ask_qty: f64,
    ) -> Result<f64, FairValueError> {
        if !bid_price.is_finite() || !ask_price.is_finite() {
            return Err(FairValueError::NonFinitePrice);
        }
        if !bid_qty.is_finite() || !ask_qty.is_finite() {
            return Err(FairValueError::NonFiniteQuantity);
        }
        if bid_price <= 0.0 || ask_price <= 0.0 {
            return Err(FairValueError::NonPositivePrice);
        }
        if bid_qty <= 0.0 || ask_qty <= 0.0 {
            return Err(FairValueError::NonPositiveQuantity);
        }
        if bid_price >= ask_price {
            return Err(FairValueError::LockedOrCrossed);
        }
        let depth = bid_qty + ask_qty;
        if depth <= 0.0 {
            return Err(FairValueError::NonPositiveQuantity);
        }
        let fair = match self {
            Self::Midpoint => (bid_price + ask_price) / 2.0,
            Self::Microprice => (ask_price * bid_qty + bid_price * ask_qty) / depth,
        };
        if !fair.is_finite() || fair <= 0.0 {
            return Err(FairValueError::NonFiniteEstimate);
        }
        Ok(fair)
    }

    /// Estimates from a fixed-depth snapshot without copying the book.
    ///
    /// Index 0 is the best level. `N` is the array length, not the number of
    /// live levels. Empty sides and counts past `N` are errors.
    pub fn estimate_snapshot<const N: usize>(
        self,
        bids: &[Level; N],
        bid_count: u8,
        asks: &[Level; N],
        ask_count: u8,
    ) -> Result<f64, FairValueError> {
        if bid_count == 0 || bid_count as usize > N {
            return Err(FairValueError::NoBid);
        }
        if ask_count == 0 || ask_count as usize > N {
            return Err(FairValueError::NoAsk);
        }
        let bid = bids[0];
        let ask = asks[0];
        self.estimate(bid.price, bid.quantity, ask.price, ask.quantity)
    }

    /// Estimates from a recorded book print.
    pub fn estimate_book(self, book: &BookRecord) -> Result<f64, FairValueError> {
        self.estimate_snapshot(book.bids(), book.bid_count(), book.asks(), book.ask_count())
    }

    /// Builds a decision-time state, estimating the reference price once.
    pub fn derive_top(
        self,
        bid_price: f64,
        bid_qty: f64,
        ask_price: f64,
        ask_qty: f64,
    ) -> Result<MarketState, FairValueError> {
        let fair_value = self.estimate(bid_price, bid_qty, ask_price, ask_qty)?;
        Ok(MarketState::from_estimate(
            self, fair_value, bid_price, bid_qty, ask_price, ask_qty,
        ))
    }

    /// Builds a decision-time state from a recorded book.
    ///
    /// The book is borrowed. Only the best levels are copied into the state.
    pub fn derive_book(self, book: &BookRecord) -> Result<MarketState, FairValueError> {
        let bid = book.best_bid().ok_or(FairValueError::NoBid)?;
        let ask = book.best_ask().ok_or(FairValueError::NoAsk)?;
        self.derive_top(bid.price, bid.quantity, ask.price, ask.quantity)
    }
}

/// The book cannot produce a reference price. No fallback value is returned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FairValueError {
    /// The snapshot has no bid level.
    NoBid,
    /// The snapshot has no ask level.
    NoAsk,
    /// A best price was NaN or infinite.
    NonFinitePrice,
    /// A best quantity was NaN or infinite.
    NonFiniteQuantity,
    /// A best price was not strictly positive.
    NonPositivePrice,
    /// A best quantity was not strictly positive, or the two quantities summed to a non-positive value.
    NonPositiveQuantity,
    /// Best bid was greater than or equal to best ask.
    LockedOrCrossed,
    /// The arithmetic result was non-finite or not strictly positive.
    NonFiniteEstimate,
}

impl fmt::Display for FairValueError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::NoBid => "book has no bid level",
            Self::NoAsk => "book has no ask level",
            Self::NonFinitePrice => "best bid or ask price is not finite",
            Self::NonFiniteQuantity => "best bid or ask quantity is not finite",
            Self::NonPositivePrice => "best bid or ask price must be positive",
            Self::NonPositiveQuantity => "best bid and ask quantities must be positive",
            Self::LockedOrCrossed => "best bid must be strictly below best ask",
            Self::NonFiniteEstimate => "fair-value estimate is not a positive finite price",
        };
        f.write_str(message)
    }
}

impl std::error::Error for FairValueError {}

#[cfg(test)]
mod tests {
    use super::*;

    const BID: f64 = 99.90;
    const ASK: f64 = 100.10;

    fn approx(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= 1e-12,
            "actual {actual} != expected {expected}"
        );
    }

    #[test]
    fn midpoint_is_the_average_of_the_touch() {
        let fair = FairValueMethod::Midpoint
            .estimate(BID, 2.0, ASK, 7.0)
            .unwrap();
        approx(fair, (BID + ASK) / 2.0);
        approx(fair, 100.0);
    }

    #[test]
    fn microprice_matches_the_size_weighted_formula() {
        let bid_qty = 2.0;
        let ask_qty = 1.0;
        let fair = FairValueMethod::Microprice
            .estimate(BID, bid_qty, ASK, ask_qty)
            .unwrap();
        approx(fair, (ASK * bid_qty + BID * ask_qty) / (bid_qty + ask_qty));
    }

    #[test]
    fn equal_quantities_make_microprice_match_midpoint() {
        let micro = FairValueMethod::Microprice
            .estimate(BID, 4.0, ASK, 4.0)
            .unwrap();
        let mid = FairValueMethod::Midpoint
            .estimate(BID, 4.0, ASK, 4.0)
            .unwrap();
        approx(micro, mid);
    }

    #[test]
    fn larger_bid_quantity_moves_microprice_toward_the_ask() {
        let mid = FairValueMethod::Midpoint
            .estimate(BID, 3.0, ASK, 1.0)
            .unwrap();
        let micro = FairValueMethod::Microprice
            .estimate(BID, 3.0, ASK, 1.0)
            .unwrap();
        assert!(micro > mid);
        assert!(micro < ASK);
        assert!(micro > BID);
        approx(micro, (ASK * 3.0 + BID * 1.0) / 4.0);
    }

    #[test]
    fn larger_ask_quantity_moves_microprice_toward_the_bid() {
        let mid = FairValueMethod::Midpoint
            .estimate(BID, 1.0, ASK, 3.0)
            .unwrap();
        let micro = FairValueMethod::Microprice
            .estimate(BID, 1.0, ASK, 3.0)
            .unwrap();
        assert!(micro < mid);
        assert!(micro > BID);
        assert!(micro < ASK);
        approx(micro, (ASK * 1.0 + BID * 3.0) / 4.0);
    }

    #[test]
    fn valid_microprice_stays_between_bid_and_ask() {
        for (bid_qty, ask_qty) in [(1.0, 1.0), (5.0, 0.5), (0.25, 8.0)] {
            let fair = FairValueMethod::Microprice
                .estimate(BID, bid_qty, ASK, ask_qty)
                .unwrap();
            assert!(fair > BID && fair < ASK);
        }
    }

    #[test]
    fn repeated_estimates_are_identical() {
        let method = FairValueMethod::Microprice;
        let first = method.estimate(BID, 2.5, ASK, 1.5).unwrap();
        let second = method.estimate(BID, 2.5, ASK, 1.5).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn zero_quantity_is_rejected_for_both_methods() {
        for method in [FairValueMethod::Midpoint, FairValueMethod::Microprice] {
            assert_eq!(
                method.estimate(BID, 0.0, ASK, 1.0),
                Err(FairValueError::NonPositiveQuantity)
            );
            assert_eq!(
                method.estimate(BID, 1.0, ASK, 0.0),
                Err(FairValueError::NonPositiveQuantity)
            );
        }
    }

    #[test]
    fn non_finite_inputs_and_crossed_books_are_rejected() {
        let method = FairValueMethod::Microprice;
        assert_eq!(
            method.estimate(f64::NAN, 1.0, ASK, 1.0),
            Err(FairValueError::NonFinitePrice)
        );
        assert_eq!(
            method.estimate(BID, f64::INFINITY, ASK, 1.0),
            Err(FairValueError::NonFiniteQuantity)
        );
        assert_eq!(
            method.estimate(ASK, 1.0, BID, 1.0),
            Err(FairValueError::LockedOrCrossed)
        );
        assert_eq!(
            method.estimate(100.0, 1.0, 100.0, 1.0),
            Err(FairValueError::LockedOrCrossed)
        );
        assert_eq!(FairValueMethod::default(), FairValueMethod::Microprice);
    }

    #[test]
    fn empty_snapshot_sides_are_rejected_without_a_fallback() {
        let bids = [Level {
            price: BID,
            quantity: 1.0,
        }];
        let asks = [Level::EMPTY];
        assert_eq!(
            FairValueMethod::Microprice.estimate_snapshot(&bids, 0, &asks, 0),
            Err(FairValueError::NoBid)
        );
        assert_eq!(
            FairValueMethod::Midpoint.estimate_snapshot(&bids, 1, &asks, 0),
            Err(FairValueError::NoAsk)
        );
    }
}
