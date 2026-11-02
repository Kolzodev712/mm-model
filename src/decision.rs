//! Information available before a buy-quote decision.
//!
//! These types are the decision-time view. They do not carry fill labels,
//! execution prices, or the reference price at a future horizon.
//!
//! [`MarketState`] stores the top of book and one reference price already
//! estimated by [`crate::fair_value::FairValueMethod`]. Callers do not pass
//! that price in. It is computed once when the state is built and then reused
//! for every candidate quote on that snapshot.

use std::fmt;

use crate::fair_value::FairValueMethod;

/// Top of book plus the reference price estimated from that touch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MarketState {
    method: FairValueMethod,
    fair_value: f64,
    best_bid_price: f64,
    best_bid_quantity: f64,
    best_ask_price: f64,
    best_ask_quantity: f64,
}

impl MarketState {
    pub(crate) fn from_estimate(
        method: FairValueMethod,
        fair_value: f64,
        best_bid_price: f64,
        best_bid_quantity: f64,
        best_ask_price: f64,
        best_ask_quantity: f64,
    ) -> Self {
        Self {
            method,
            fair_value,
            best_bid_price,
            best_bid_quantity,
            best_ask_price,
            best_ask_quantity,
        }
    }

    /// Estimator that produced [`Self::fair_value`].
    #[must_use]
    pub fn fair_value_method(self) -> FairValueMethod {
        self.method
    }

    /// Book-derived reference price for this snapshot. Not a future price.
    #[must_use]
    pub fn fair_value(self) -> f64 {
        self.fair_value
    }

    #[must_use]
    pub fn best_bid_price(self) -> f64 {
        self.best_bid_price
    }

    #[must_use]
    pub fn best_bid_quantity(self) -> f64 {
        self.best_bid_quantity
    }

    #[must_use]
    pub fn best_ask_price(self) -> f64 {
        self.best_ask_price
    }

    #[must_use]
    pub fn best_ask_quantity(self) -> f64 {
        self.best_ask_quantity
    }
}

/// A buy quote that might be sent. Price and size are known before the decision.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CandidateQuote {
    price: f64,
    quantity: f64,
}

impl CandidateQuote {
    /// Builds a buy quote with a finite positive price and quantity.
    pub fn new(price: f64, quantity: f64) -> Result<Self, MarketInputError> {
        if !price.is_finite() || price <= 0.0 {
            return Err(MarketInputError::InvalidQuotePrice);
        }
        if !quantity.is_finite() || quantity <= 0.0 {
            return Err(MarketInputError::InvalidQuoteQuantity);
        }
        Ok(Self { price, quantity })
    }

    #[must_use]
    pub fn price(self) -> f64 {
        self.price
    }

    #[must_use]
    pub fn quantity(self) -> f64 {
        self.quantity
    }
}

/// Rejected decision-time market or quote input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarketInputError {
    /// Best bid price was non-positive or not finite.
    InvalidBidPrice,
    /// Best ask price was non-positive or not finite.
    InvalidAskPrice,
    /// Best bid quantity was negative or not finite.
    InvalidBidQuantity,
    /// Best ask quantity was negative or not finite.
    InvalidAskQuantity,
    /// Candidate price was non-positive or not finite.
    InvalidQuotePrice,
    /// Candidate quantity was non-positive or not finite.
    InvalidQuoteQuantity,
}

impl fmt::Display for MarketInputError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::InvalidBidPrice => "best bid price must be finite and positive",
            Self::InvalidAskPrice => "best ask price must be finite and positive",
            Self::InvalidBidQuantity => "best bid quantity must be finite and non-negative",
            Self::InvalidAskQuantity => "best ask quantity must be finite and non-negative",
            Self::InvalidQuotePrice => "candidate quote price must be finite and positive",
            Self::InvalidQuoteQuantity => "candidate quantity must be finite and positive",
        };
        f.write_str(message)
    }
}

impl std::error::Error for MarketInputError {}
