//! Quote-evaluation baselines and a read-only market-data pipeline.
//!
//! This is not an optimal quoting system and it does not trade.
//!
//! - [`domain`] — prices, books, quotes, and the legacy observation type
//! - [`model`] — reference price, features, fill and P&L models, selection
//! - [`data`] — exchange intake, training rows, historical baseline
//! - [`storage`] — raw MMBK and, with `parquet-storage`, derived Parquet
//! - [`runtime`] — collection, conversion, training, and shadow workflows
//!
//! A historical fill at one quote price does not prove whether a different
//! quote would have filled at the same moment.

#![deny(unsafe_code)]

pub mod data;
pub mod domain;
pub mod model;
pub mod runtime;
pub mod storage;
