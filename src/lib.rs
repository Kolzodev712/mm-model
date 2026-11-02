//! Market-microstructure machine-learning research system.
//!
//! Inputs are public market data and optional historical quote outcomes.
//! Outputs are probabilities, validation metrics, and research logs.
//! This crate does not generate a trading signal, choose a quote, or submit an order.
//!
//! - [`domain`] — prices, books, and quote context
//! - [`model`] — reference price, features, and probability models
//! - [`data`] — public book intake, training rows, descriptive history
//! - [`storage`] — raw MMBK and, with `parquet-storage`, derived Parquet
//! - [`runtime`] — collection, conversion, training, and live inference

#![deny(unsafe_code)]

pub mod data;
pub mod domain;
pub mod model;
pub mod runtime;
pub mod storage;
