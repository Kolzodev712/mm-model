//! Fixed-depth book prints.
//!
//! Depth is a fixed array of [`BOOK_DEPTH`] levels per side. Unused tail slots
//! are zero. This is not a map of prices to sizes. Prices and quantities are
//! `f64` for this educational recorder. The learned pipeline does not read an
//! external fair-value series from these records.

use std::fmt;

/// Levels kept per side.
pub const BOOK_DEPTH: usize = 10;

const NAME_LEN: usize = 16;

/// One price level. Quantity is the displayed size at that price.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Level {
    /// Price, finite and strictly positive when the level is in use.
    pub price: f64,
    /// Displayed quantity, finite and strictly positive when the level is in use.
    pub quantity: f64,
}

impl Level {
    /// Zero price and quantity, used for unused depth slots.
    pub const EMPTY: Self = Self {
        price: 0.0,
        quantity: 0.0,
    };

    fn is_tradable(self) -> bool {
        self.price.is_finite()
            && self.price > 0.0
            && self.quantity.is_finite()
            && self.quantity > 0.0
    }
}

/// Exchange and symbol packed into fixed buffers. No heap allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Instrument {
    exchange: [u8; NAME_LEN],
    symbol: [u8; NAME_LEN],
}

impl Instrument {
    /// Copies ASCII names of 1 to 16 bytes into fixed buffers.
    pub fn new(exchange: &str, symbol: &str) -> Result<Self, NameError> {
        Ok(Self {
            exchange: pack_name(exchange)?,
            symbol: pack_name(symbol)?,
        })
    }

    #[must_use]
    pub fn exchange(&self) -> &str {
        unpack_name(&self.exchange)
    }

    #[must_use]
    pub fn symbol(&self) -> &str {
        unpack_name(&self.symbol)
    }

    pub(crate) fn exchange_bytes(&self) -> &[u8; NAME_LEN] {
        &self.exchange
    }

    pub(crate) fn symbol_bytes(&self) -> &[u8; NAME_LEN] {
        &self.symbol
    }

    pub(crate) fn from_bytes(
        exchange: [u8; NAME_LEN],
        symbol: [u8; NAME_LEN],
    ) -> Result<Self, NameError> {
        unpack_checked(&exchange)?;
        unpack_checked(&symbol)?;
        Ok(Self { exchange, symbol })
    }
}

/// Clock and sequence fields for one book print.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BookStamp {
    /// Local receive time, unix nanoseconds.
    pub local_receive_ns: i64,
    /// Exchange event time, unix nanoseconds, when the feed provided one.
    pub exchange_event_ns: Option<i64>,
    /// Exchange update id, when the feed provided one.
    pub update_id: Option<u64>,
}

/// One top-N book print.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BookRecord {
    instrument: Instrument,
    stamp: BookStamp,
    bid_count: u8,
    ask_count: u8,
    bids: [Level; BOOK_DEPTH],
    asks: [Level; BOOK_DEPTH],
}

impl BookRecord {
    /// Validates counts, level values, and best-first ordering.
    ///
    /// Bids must be strictly decreasing in price. Asks must be strictly
    /// increasing. Slots at or past `bid_count` / `ask_count` are cleared.
    pub fn try_new(
        instrument: Instrument,
        stamp: BookStamp,
        mut bids: [Level; BOOK_DEPTH],
        bid_count: u8,
        mut asks: [Level; BOOK_DEPTH],
        ask_count: u8,
    ) -> Result<Self, BookError> {
        validate_side(&mut bids, bid_count, true)?;
        validate_side(&mut asks, ask_count, false)?;
        Ok(Self {
            instrument,
            stamp,
            bid_count,
            ask_count,
            bids,
            asks,
        })
    }

    /// Copies a parsed top-N snapshot into a record.
    ///
    /// `local_receive_ns` is the host clock. Exchange event time is set only
    /// when the payload contained one. Partial Binance depth snapshots usually
    /// do not.
    pub fn from_depth(
        instrument: Instrument,
        local_receive_ns: i64,
        depth: &ParsedDepth,
    ) -> Result<Self, BookError> {
        let exchange_event_ns = match depth.event_time_ms {
            Some(milliseconds) => Some(event_ms_to_ns(milliseconds)?),
            None => None,
        };
        Self::try_new(
            instrument,
            BookStamp {
                local_receive_ns,
                exchange_event_ns,
                update_id: Some(depth.last_update_id),
            },
            depth.bids,
            depth.bid_count,
            depth.asks,
            depth.ask_count,
        )
    }

    #[must_use]
    pub fn instrument(&self) -> Instrument {
        self.instrument
    }

    #[must_use]
    pub fn stamp(&self) -> BookStamp {
        self.stamp
    }

    #[must_use]
    pub fn bid_count(&self) -> u8 {
        self.bid_count
    }

    #[must_use]
    pub fn ask_count(&self) -> u8 {
        self.ask_count
    }

    #[must_use]
    pub fn bids(&self) -> &[Level; BOOK_DEPTH] {
        &self.bids
    }

    #[must_use]
    pub fn asks(&self) -> &[Level; BOOK_DEPTH] {
        &self.asks
    }

    /// Best bid, which is index 0 when at least one bid is present.
    #[must_use]
    pub fn best_bid(&self) -> Option<Level> {
        if self.bid_count == 0 {
            None
        } else {
            Some(self.bids[0])
        }
    }

    /// Best ask, which is index 0 when at least one ask is present.
    #[must_use]
    pub fn best_ask(&self) -> Option<Level> {
        if self.ask_count == 0 {
            None
        } else {
            Some(self.asks[0])
        }
    }
}

/// Top-N snapshot parsed from an exchange payload.
///
/// The recorder reuses one value and parses each message into it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParsedDepth {
    /// Exchange update id.
    pub last_update_id: u64,
    /// Exchange event time in unix milliseconds, when the payload had one.
    pub event_time_ms: Option<u64>,
    /// Best-first bids. Slots at or past `bid_count` are unused.
    pub bids: [Level; BOOK_DEPTH],
    /// Best-first asks. Slots at or past `ask_count` are unused.
    pub asks: [Level; BOOK_DEPTH],
    /// Number of valid bid levels, at most [`BOOK_DEPTH`].
    pub bid_count: u8,
    /// Number of valid ask levels, at most [`BOOK_DEPTH`].
    pub ask_count: u8,
}

impl ParsedDepth {
    /// Empty snapshot used as a reusable parse buffer.
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            last_update_id: 0,
            event_time_ms: None,
            bids: [Level::EMPTY; BOOK_DEPTH],
            asks: [Level::EMPTY; BOOK_DEPTH],
            bid_count: 0,
            ask_count: 0,
        }
    }
}

/// Legacy on-disk fair-value sample.
///
/// The learned dataset builder does not read this record. Reference prices for
/// training are estimated from [`BookRecord`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FairValueRecord {
    local_receive_ns: i64,
    fair_value: f64,
}

impl FairValueRecord {
    /// Stores a finite, strictly positive fair value.
    pub fn new(local_receive_ns: i64, fair_value: f64) -> Result<Self, BookError> {
        if !fair_value.is_finite() || fair_value <= 0.0 {
            return Err(BookError::InvalidFairValue);
        }
        Ok(Self {
            local_receive_ns,
            fair_value,
        })
    }

    #[must_use]
    pub fn local_receive_ns(self) -> i64 {
        self.local_receive_ns
    }

    #[must_use]
    pub fn fair_value(self) -> f64 {
        self.fair_value
    }
}

/// A name did not fit the fixed 16-byte buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NameError;

impl fmt::Display for NameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("instrument name must be 1 to 16 ASCII bytes")
    }
}

impl std::error::Error for NameError {}

/// A book or fair-value record violated its layout rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BookError {
    /// A side had more levels than [`BOOK_DEPTH`], or a counted level was not tradable.
    InvalidLevel,
    /// Bid or ask prices were not strictly best-first.
    Unordered,
    /// Exchange event milliseconds did not fit in unix nanoseconds.
    Timestamp,
    /// Fair value was non-positive or not finite.
    InvalidFairValue,
    /// A stored name was not valid ASCII.
    Name,
}

impl fmt::Display for BookError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::InvalidLevel => "book level is missing, non-positive, or not finite",
            Self::Unordered => "book levels must be strictly best-first",
            Self::Timestamp => "exchange event time does not fit in unix nanoseconds",
            Self::InvalidFairValue => "fair value must be finite and positive",
            Self::Name => "stored instrument name is not ASCII",
        };
        f.write_str(message)
    }
}

impl std::error::Error for BookError {}

impl From<NameError> for BookError {
    fn from(_: NameError) -> Self {
        Self::Name
    }
}

fn validate_side(levels: &mut [Level; BOOK_DEPTH], count: u8, bids: bool) -> Result<(), BookError> {
    if count as usize > BOOK_DEPTH {
        return Err(BookError::InvalidLevel);
    }
    let count = count as usize;
    for index in 0..count {
        if !levels[index].is_tradable() {
            return Err(BookError::InvalidLevel);
        }
        if index > 0 {
            let previous = levels[index - 1].price;
            let price = levels[index].price;
            let ordered = if bids {
                price < previous
            } else {
                price > previous
            };
            if !ordered {
                return Err(BookError::Unordered);
            }
        }
    }
    for level in levels.iter_mut().skip(count) {
        *level = Level::EMPTY;
    }
    Ok(())
}

pub(crate) fn event_ms_to_ns(milliseconds: u64) -> Result<i64, BookError> {
    let nanos = milliseconds
        .checked_mul(1_000_000)
        .ok_or(BookError::Timestamp)?;
    i64::try_from(nanos).map_err(|_| BookError::Timestamp)
}

fn pack_name(name: &str) -> Result<[u8; NAME_LEN], NameError> {
    if name.is_empty() || name.len() > NAME_LEN || !name.is_ascii() {
        return Err(NameError);
    }
    let mut packed = [0u8; NAME_LEN];
    packed[..name.len()].copy_from_slice(name.as_bytes());
    Ok(packed)
}

fn unpack_name(bytes: &[u8; NAME_LEN]) -> &str {
    unpack_checked(bytes).unwrap_or("")
}

fn unpack_checked(bytes: &[u8; NAME_LEN]) -> Result<&str, NameError> {
    let end = bytes.iter().position(|byte| *byte == 0).unwrap_or(NAME_LEN);
    if end == 0 {
        return Err(NameError);
    }
    if bytes[end..].iter().any(|byte| *byte != 0) {
        return Err(NameError);
    }
    let text = std::str::from_utf8(&bytes[..end]).map_err(|_| NameError)?;
    if !text.is_ascii() {
        return Err(NameError);
    }
    Ok(text)
}
