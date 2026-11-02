//! Raw public market data, separate from model training.
//!
//! The recorder writes immutable book prints. It does not estimate a reference
//! price, update model weights, or invent fill labels. Reference prices are
//! derived later by [`crate::fair_value`] from these prints. A historical fill
//! of one quote is not stored here and is not evidence about any other quote.

mod binance;
mod book;
mod format;

#[cfg(feature = "live")]
mod binance_client;
#[cfg(feature = "live")]
pub use binance_client::{BinanceBookClient, ClientError};

pub use binance::{
    parse_partial_depth, BinanceBookConfig, ParseError, BINANCE_EXCHANGE, BINANCE_SPOT_WS, BTCUSDT,
    PARTIAL_DEPTH_URL,
};
pub use book::{
    BookError, BookRecord, BookStamp, Instrument, Level, NameError, ParsedDepth, BOOK_DEPTH,
};
pub use format::{
    aligned_book_file_len, decode_book_header, encode_book_record, read_book_log,
    write_book_header, BookLogWriter, LogError, BOOK_HEADER_LEN, BOOK_RECORD_LEN,
};
