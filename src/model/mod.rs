//! Fair value, features, probability models, and artifacts.
//!
//! This layer depends on [`crate::domain`] only. It does not know about
//! Binance, MMBK, or Parquet. It does not choose a trading action.

pub mod calibration;
pub mod fair_value;
pub mod feature;
pub mod fill;
pub mod market;
pub mod prediction;
pub mod probe;

#[cfg(feature = "artifact")]
pub mod artifact;
