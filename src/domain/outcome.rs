//! Recorded quoting attempts and their mark-to-fair-value P&L.
//!
//! This is the legacy historical baseline. [`HistoricalObservation`] stores a
//! horizon reference price that the caller already has. The learned dataset
//! builder does not use that field: it estimates both the decision-time and
//! horizon prices from recorded books.
//!
//! A filled buy is marked against the fair value at a fixed horizon:
//!
//! ```text
//! pnl = (fair_value_at_horizon - execution_price) * quantity - fees
//! ```
//!
//! An unfilled attempt has P&L of zero. Fees on an unfilled attempt are not
//! applied.

use std::fmt;

/// Label on a historical quote record. Grouping key for descriptive counts.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StrategyId(String);

impl StrategyId {
    /// Creates a non-empty strategy identifier.
    pub fn new(id: impl Into<String>) -> Result<Self, EmptyStrategyId> {
        let id = id.into();
        if id.is_empty() {
            return Err(EmptyStrategyId);
        }
        Ok(Self(id))
    }

    /// Returns the identifier text.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for StrategyId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl AsRef<str> for StrategyId {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

/// Rejected empty [`StrategyId`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmptyStrategyId;

impl fmt::Display for EmptyStrategyId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("strategy id must be non-empty")
    }
}

impl std::error::Error for EmptyStrategyId {}

/// Whether a submitted buy quote traded.
///
/// Unfilled attempts do not carry an execution price.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FillOutcome {
    /// The quote did not trade.
    Unfilled,
    /// The quote traded in full at `execution_price`.
    Filled { execution_price: f64 },
}

impl FillOutcome {
    /// Returns the execution price when this attempt filled.
    pub fn execution_price(self) -> Option<f64> {
        match self {
            Self::Filled { execution_price } => Some(execution_price),
            Self::Unfilled => None,
        }
    }

    /// Reports whether the quote traded.
    pub fn is_filled(self) -> bool {
        matches!(self, Self::Filled { .. })
    }
}

/// One historical quoting attempt for a single strategy.
///
/// The baseline assumes one instrument, a buy quote, a known quantity, and no
/// partial fill. P&L uses the execution price. The submitted bid is retained
/// because it is the quote that was actually sent.
#[derive(Debug, Clone, PartialEq)]
pub struct HistoricalObservation {
    strategy_id: StrategyId,
    submitted_bid: f64,
    quantity: f64,
    outcome: FillOutcome,
    fair_value_at_horizon: f64,
    fees: f64,
}

impl HistoricalObservation {
    /// Builds an observation after rejecting non-finite prices and invalid size or fees.
    pub fn new(
        strategy_id: StrategyId,
        submitted_bid: f64,
        quantity: f64,
        outcome: FillOutcome,
        fair_value_at_horizon: f64,
        fees: f64,
    ) -> Result<Self, InvalidObservation> {
        if !submitted_bid.is_finite() {
            return Err(InvalidObservation::NonFiniteSubmittedBid);
        }
        if !quantity.is_finite() || quantity <= 0.0 {
            return Err(InvalidObservation::InvalidQuantity);
        }
        if !fair_value_at_horizon.is_finite() {
            return Err(InvalidObservation::NonFiniteFairValue);
        }
        if !fees.is_finite() || fees < 0.0 {
            return Err(InvalidObservation::InvalidFees);
        }
        if let Some(execution_price) = outcome.execution_price() {
            if !execution_price.is_finite() {
                return Err(InvalidObservation::NonFiniteExecutionPrice);
            }
        }

        Ok(Self {
            strategy_id,
            submitted_bid,
            quantity,
            outcome,
            fair_value_at_horizon,
            fees,
        })
    }

    /// Strategy that submitted this quote.
    pub fn strategy_id(&self) -> &StrategyId {
        &self.strategy_id
    }

    /// Bid that was sent.
    pub fn submitted_bid(&self) -> f64 {
        self.submitted_bid
    }

    /// Order quantity. Always finite and positive.
    pub fn quantity(&self) -> f64 {
        self.quantity
    }

    /// Fill result. Unfilled values have no execution price.
    pub fn outcome(&self) -> FillOutcome {
        self.outcome
    }

    /// Fair value used to mark a fill at the evaluation horizon.
    pub fn fair_value_at_horizon(&self) -> f64 {
        self.fair_value_at_horizon
    }

    /// Fee charged if this attempt filled. Ignored when the attempt did not fill.
    pub fn fees(&self) -> f64 {
        self.fees
    }

    /// Mark-to-fair-value P&L of this attempt.
    ///
    /// Filled buys use [`mark_to_fair_pnl`]. Unfilled attempts return 0.
    pub fn pnl(&self) -> f64 {
        match self.outcome {
            FillOutcome::Unfilled => 0.0,
            FillOutcome::Filled { execution_price } => mark_to_fair_pnl(
                self.fair_value_at_horizon,
                execution_price,
                self.quantity,
                self.fees,
            ),
        }
    }
}

/// Mark-to-fair-value P&L of one filled buy.
///
/// `(fair_value_at_horizon - execution_price) * quantity - fees`.
#[must_use]
pub fn mark_to_fair_pnl(
    fair_value_at_horizon: f64,
    execution_price: f64,
    quantity: f64,
    fees: f64,
) -> f64 {
    (fair_value_at_horizon - execution_price) * quantity - fees
}

/// Input that cannot represent a quoting observation under the baseline rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InvalidObservation {
    /// Submitted bid was NaN or infinite.
    NonFiniteSubmittedBid,
    /// Quantity was non-positive or not finite.
    InvalidQuantity,
    /// Horizon fair value was NaN or infinite.
    NonFiniteFairValue,
    /// Fees were negative or not finite.
    InvalidFees,
    /// Execution price was NaN or infinite.
    NonFiniteExecutionPrice,
}

impl fmt::Display for InvalidObservation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::NonFiniteSubmittedBid => "submitted bid must be finite",
            Self::InvalidQuantity => "quantity must be finite and positive",
            Self::NonFiniteFairValue => "fair value at horizon must be finite",
            Self::InvalidFees => "fees must be finite and non-negative",
            Self::NonFiniteExecutionPrice => "execution price must be finite",
        };
        f.write_str(message)
    }
}

impl std::error::Error for InvalidObservation {}

#[cfg(test)]
mod tests {
    use super::*;

    fn strategy() -> StrategyId {
        StrategyId::new("A").unwrap()
    }

    fn observation(
        outcome: FillOutcome,
        fair_value: f64,
        quantity: f64,
        fees: f64,
    ) -> HistoricalObservation {
        HistoricalObservation::new(strategy(), 100.0, quantity, outcome, fair_value, fees).unwrap()
    }

    #[test]
    fn pnl_profitable_fill() {
        let filled = observation(
            FillOutcome::Filled {
                execution_price: 100.0,
            },
            101.0,
            2.0,
            0.5,
        );
        assert!((filled.pnl() - 1.5).abs() < 1e-12);
    }

    #[test]
    fn pnl_losing_fill() {
        let filled = observation(
            FillOutcome::Filled {
                execution_price: 100.0,
            },
            99.0,
            2.0,
            0.5,
        );
        assert!((filled.pnl() - -2.5).abs() < 1e-12);
    }

    #[test]
    fn pnl_unfilled_order() {
        let missed = observation(FillOutcome::Unfilled, 101.0, 2.0, 0.5);
        assert_eq!(missed.outcome().execution_price(), None);
        assert_eq!(missed.pnl(), 0.0);
    }

    #[test]
    fn rejects_invalid_quantity_and_fees() {
        let err =
            HistoricalObservation::new(strategy(), 100.0, 0.0, FillOutcome::Unfilled, 100.0, 0.0);
        assert_eq!(err, Err(InvalidObservation::InvalidQuantity));

        let err =
            HistoricalObservation::new(strategy(), 100.0, 1.0, FillOutcome::Unfilled, 100.0, -0.1);
        assert_eq!(err, Err(InvalidObservation::InvalidFees));
        assert_eq!(StrategyId::new(""), Err(EmptyStrategyId));
    }
}
