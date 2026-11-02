//! Fair value, features, reference models, prediction, and selection.
//!
//! This layer depends on [`crate::domain`] only. It does not know about
//! Binance, MMBK, or Parquet.

pub mod candidate;
pub mod fair_value;
pub mod feature;
pub mod fill;
pub mod pnl;
pub mod prediction;
pub mod selection;

#[cfg(feature = "artifact")]
pub mod artifact;
