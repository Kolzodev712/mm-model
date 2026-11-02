//! Append-oriented MMBK book logs.
//!
//! MMBK is the raw source of truth. Parquet is rebuilt from it.

pub mod format;

pub use format::{
    aligned_book_file_len, decode_book_header, encode_book_record, read_book_log,
    write_book_header, BookLogWriter, LogError, BOOK_HEADER_LEN, BOOK_RECORD_LEN,
};
