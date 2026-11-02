//! Raw public market data, separate from model training.
//!
//! The recorder writes immutable book prints. It does not estimate a reference
//! price, update model weights, or invent fill labels. Reference prices are
//! derived later by [`crate::fair_value`] from these prints. A historical fill
//! of one quote is not stored here and is not evidence about any other quote.

mod binance;
mod book;
mod format;

pub use binance::{parse_partial_depth, ParseError, BINANCE_EXCHANGE, BTCUSDT, PARTIAL_DEPTH_URL};
pub use book::{
    BookError, BookRecord, BookStamp, FairValueRecord, Instrument, Level, NameError, ParsedDepth,
    BOOK_DEPTH,
};
pub use format::{
    aligned_book_file_len, decode_book_header, encode_book_record, encode_fair_value,
    read_book_log, read_fair_value_log, write_book_header, write_fair_value_header, LogError,
    BOOK_HEADER_LEN, BOOK_RECORD_LEN, FAIR_RECORD_LEN,
};
