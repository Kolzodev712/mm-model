//! Pure market types used by the research models.
//!
//! This module does not read sockets, files, or Parquet.

pub mod book;
pub mod market;
pub mod outcome;
pub mod units;

pub use book::{
    BookError, BookRecord, BookStamp, Instrument, Level, NameError, ParsedDepth, BOOK_DEPTH,
};
pub use market::{FairValueMethod, MarketInputError, MarketState, QuoteContext};
pub use outcome::{
    mark_to_fair_pnl, EmptyStrategyId, FillOutcome, HistoricalObservation, InvalidObservation,
    StrategyId,
};
pub use units::{DurationNanos, Fee, Pnl, Price, Probability, Quantity, UnitError, UnixNanos};
