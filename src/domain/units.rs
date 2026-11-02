//! Zero-cost validated scalars for the learned and market-data paths.
//!
//! Prices and quantities in this build are still `f64`. These wrappers reject
//! non-finite values at construction so later arithmetic does not have to
//! rediscover that. They are `#[repr(transparent)]` and are not heap allocated.

use std::fmt;

/// Finite price strictly greater than zero.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Price(f64);

impl Price {
    pub fn new(value: f64) -> Result<Self, UnitError> {
        if value.is_finite() && value > 0.0 {
            Ok(Self(value))
        } else {
            Err(UnitError::Price)
        }
    }

    #[must_use]
    pub const fn get(self) -> f64 {
        self.0
    }
}

/// Finite quantity strictly greater than zero.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Quantity(f64);

impl Quantity {
    pub fn new(value: f64) -> Result<Self, UnitError> {
        if value.is_finite() && value > 0.0 {
            Ok(Self(value))
        } else {
            Err(UnitError::Quantity)
        }
    }

    #[must_use]
    pub const fn get(self) -> f64 {
        self.0
    }
}

/// Finite fee, zero or positive.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Fee(f64);

impl Fee {
    pub fn new(value: f64) -> Result<Self, UnitError> {
        if value.is_finite() && value >= 0.0 {
            Ok(Self(value))
        } else {
            Err(UnitError::Fee)
        }
    }

    #[must_use]
    pub const fn get(self) -> f64 {
        self.0
    }
}

/// Finite probability in `[0, 1]`.
///
/// Values outside that interval, including NaN, are rejected. Nothing here
/// clamps an invalid number into a plausible probability.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Probability(f64);

impl Probability {
    pub fn new(value: f64) -> Result<Self, UnitError> {
        if value.is_finite() && (0.0..=1.0).contains(&value) {
            Ok(Self(value))
        } else {
            Err(UnitError::Probability)
        }
    }

    #[must_use]
    pub const fn get(self) -> f64 {
        self.0
    }
}

/// Finite mark-to-reference P&L. Losses are negative. NaN is rejected.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Pnl(f64);

impl Pnl {
    pub fn new(value: f64) -> Result<Self, UnitError> {
        if value.is_finite() {
            Ok(Self(value))
        } else {
            Err(UnitError::Pnl)
        }
    }

    #[must_use]
    pub const fn get(self) -> f64 {
        self.0
    }
}

/// Unix time in nanoseconds. Negative values are rejected.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct UnixNanos(i64);

impl UnixNanos {
    pub fn new(nanos: i64) -> Result<Self, UnitError> {
        if nanos >= 0 {
            Ok(Self(nanos))
        } else {
            Err(UnitError::Time)
        }
    }

    #[must_use]
    pub const fn get(self) -> i64 {
        self.0
    }

    /// Elapsed time from `earlier` to `self`.
    ///
    /// Fails if `self` is before `earlier` or the subtraction overflows.
    /// It does not saturate into a plausible duration.
    pub fn saturating_gap_rejected(self, earlier: Self) -> Result<DurationNanos, UnitError> {
        self.0
            .checked_sub(earlier.0)
            .filter(|gap| *gap >= 0)
            .map(DurationNanos)
            .ok_or(UnitError::Time)
    }
}

/// Non-negative duration in nanoseconds.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DurationNanos(i64);

impl DurationNanos {
    pub fn new(nanos: i64) -> Result<Self, UnitError> {
        if nanos >= 0 {
            Ok(Self(nanos))
        } else {
            Err(UnitError::Time)
        }
    }

    #[must_use]
    pub const fn get(self) -> i64 {
        self.0
    }
}

/// Stable index of a learned candidate. Copyable. Not a heap string.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CandidateId(u16);

impl CandidateId {
    #[must_use]
    pub const fn new(id: u16) -> Self {
        Self(id)
    }

    #[must_use]
    pub const fn get(self) -> u16 {
        self.0
    }
}

/// A scalar failed its constructor check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnitError {
    Price,
    Quantity,
    Fee,
    Probability,
    Pnl,
    Time,
}

impl fmt::Display for UnitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::Price => "price must be finite and positive",
            Self::Quantity => "quantity must be finite and positive",
            Self::Fee => "fee must be finite and non-negative",
            Self::Probability => "probability must be finite and in [0, 1]",
            Self::Pnl => "pnl must be finite",
            Self::Time => "timestamp or duration is invalid",
        };
        f.write_str(message)
    }
}

impl std::error::Error for UnitError {}
