//! Binance spot partial-depth adapter.
//!
//! JSON parsing stays here. The rest of the crate sees [`crate::domain::BookRecord`].

pub mod parser;

#[cfg(feature = "live")]
pub mod client;

pub use parser::{
    parse_partial_depth, BinanceBookConfig, ParseError, BINANCE_EXCHANGE, BINANCE_SPOT_WS, BTCUSDT,
    PARTIAL_DEPTH_URL,
};

#[cfg(feature = "live")]
pub use client::{BinanceBookClient, ClientError};
