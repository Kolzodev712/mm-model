//! Read-only book source. Runners depend on this, not on a websocket crate.

use crate::marketdata::BookRecord;

/// Live or test source of book snapshots.
///
/// `Ok(None)` means the source ended. The book itself is returned by value so
/// the source trait does not wrap it in a large enum.
pub trait BookSource {
    /// Error for a failed read that the source did not recover from.
    type Error;

    /// Next book, or `None` when the source is finished.
    fn next_book(&mut self) -> Result<Option<BookRecord>, Self::Error>;
}

/// In-memory source for tests. No sockets and no sleeps.
#[derive(Debug, Clone)]
pub struct SliceSource {
    books: Vec<BookRecord>,
    index: usize,
    failures_before_close: u32,
    failures_emitted: u32,
}

impl SliceSource {
    #[must_use]
    pub fn new(books: Vec<BookRecord>) -> Self {
        Self {
            books,
            index: 0,
            failures_before_close: 0,
            failures_emitted: 0,
        }
    }

    /// Emit `count` errors after the books, then close.
    #[must_use]
    pub fn fail_after(mut self, count: u32) -> Self {
        self.failures_before_close = count;
        self
    }
}

impl BookSource for SliceSource {
    type Error = &'static str;

    fn next_book(&mut self) -> Result<Option<BookRecord>, Self::Error> {
        if let Some(book) = self.books.get(self.index).copied() {
            self.index += 1;
            return Ok(Some(book));
        }
        if self.failures_emitted < self.failures_before_close {
            self.failures_emitted += 1;
            return Err("mock stream failure");
        }
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_failure_is_visible_then_the_source_closes() {
        let mut source = SliceSource::new(Vec::new()).fail_after(1);
        assert_eq!(source.next_book(), Err("mock stream failure"));
        assert_eq!(source.next_book(), Ok(None));
    }
}
