//! Historical book scan that does not know about Parquet or MMBK.
//!
//! Implementations should visit one partition at a time. The reference
//! trainers still materialize a sample slice after this scan.

use crate::domain::book::BookRecord;

/// Inclusive receive-time window for one symbol.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BookRange {
    pub exchange: String,
    pub symbol: String,
    pub from_ns: i64,
    pub to_ns: i64,
}

/// In-memory books used by tests and by callers that already hold one batch.
pub struct MemoryBooks<'a> {
    pub exchange: &'a str,
    pub symbol: &'a str,
    pub books: &'a [BookRecord],
}

impl HistoricalBooks for MemoryBooks<'_> {
    type Error = std::convert::Infallible;

    fn for_each<F>(&self, range: &BookRange, mut visit: F) -> Result<u64, Self::Error>
    where
        F: FnMut(&BookRecord),
    {
        if self.exchange != range.exchange || self.symbol != range.symbol {
            return Ok(0);
        }
        let mut count = 0u64;
        for book in self.books {
            let ts = book.stamp().local_receive_ns;
            if ts >= range.from_ns && ts <= range.to_ns {
                visit(book);
                count += 1;
            }
        }
        Ok(count)
    }
}

/// Reads historical books without loading every partition at once.
pub trait HistoricalBooks {
    type Error;

    /// Calls `visit` for each book in range, in partition then file order.
    fn for_each<F>(&self, range: &BookRange, visit: F) -> Result<u64, Self::Error>
    where
        F: FnMut(&BookRecord);
}
