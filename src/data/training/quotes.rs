//! Historical submitted quotes. Book history does not create these labels.

use crate::data::training::dataset::QuoteEvent;

/// Inclusive decision-time window for one symbol.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuoteRange {
    pub exchange: String,
    pub symbol: String,
    pub from_ns: i64,
    pub to_ns: i64,
}

/// Reads quote outcomes for one window. Callers must not keep a process-lifetime copy.
pub trait QuoteHistory {
    type Error;

    fn for_each<F>(&self, range: &QuoteRange, visit: F) -> Result<u64, Self::Error>
    where
        F: FnMut(&QuoteEvent);
}

/// In-memory outcomes for tests.
pub struct MemoryQuoteHistory<'a> {
    pub exchange: &'a str,
    pub symbol: &'a str,
    pub quotes: &'a [QuoteEvent],
}

impl QuoteHistory for MemoryQuoteHistory<'_> {
    type Error = std::convert::Infallible;

    fn for_each<F>(&self, range: &QuoteRange, mut visit: F) -> Result<u64, Self::Error>
    where
        F: FnMut(&QuoteEvent),
    {
        if self.exchange != range.exchange || self.symbol != range.symbol {
            return Ok(0);
        }
        let mut count = 0u64;
        for quote in self.quotes {
            let ts = quote.decision_ns();
            if ts >= range.from_ns && ts <= range.to_ns {
                visit(quote);
                count += 1;
            }
        }
        Ok(count)
    }
}
