//! Historical baseline for evaluating buy-quote strategies.
//!
//! This crate calculates fill frequency, mark-to-fair-value P&L, and expected
//! value from recorded quoting attempts. It then selects the candidate with the
//! highest expected value, or `NoQuote` when not quoting is better.
//!
//! It is a deterministic baseline for later model-based quoting, not an optimal
//! market-making architecture. Historical aggregation is the current source of
//! fill probability and conditional P&L. A later model can supply those same
//! quantities from market state and a candidate quote; selection can stay a
//! comparison of expected values against not quoting.
//!
//! # Scope
//!
//! One instrument, buy quotes only, full fills, and no starting inventory.
//! There is no inventory or risk penalty, no adverse-selection model, no
//! learned predictor, and no exchange connectivity.
//!
//! # Counterfactuals
//!
//! Each observation is treated as an independent attempt at the quote that was
//! actually submitted. A historical fill at one quote price does not prove
//! whether a different quote would have filled at the same moment.
//!
//! # Layout
//!
//! - [`domain`] — observations and mark-to-fair-value P&L
//! - [`evaluation`] — per-strategy historical estimates
//! - [`selection`] — choose the highest expected value, or `NoQuote`

pub mod domain;
pub mod evaluation;
pub mod selection;

pub use domain::{
    EmptyStrategyId, FillOutcome, HistoricalObservation, InvalidObservation, StrategyId,
};
pub use evaluation::{evaluate_strategies, StrategyEvaluation};
pub use selection::{select_quote, QuoteDecision};

#[cfg(test)]
pub(crate) mod test_fixture;
