//! Local receive-time source.
//!
//! Application code that stamps a book takes a [`Clock`]. Tests use
//! [`ManualClock`]. Wall time is only read inside [`SystemClock`].

use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};

/// Unix nanoseconds from a clock.
pub trait Clock {
    fn unix_nanos(&self) -> Result<i64, ClockError>;
}

/// Host wall clock.
#[derive(Debug, Clone, Copy, Default)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn unix_nanos(&self) -> Result<i64, ClockError> {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| ClockError)?
            .as_nanos();
        i64::try_from(nanos).map_err(|_| ClockError)
    }
}

/// Clock fixed by the test or caller.
#[derive(Debug, Clone, Copy)]
pub struct ManualClock {
    nanos: i64,
}

impl ManualClock {
    pub fn new(nanos: i64) -> Result<Self, ClockError> {
        if nanos < 0 {
            return Err(ClockError);
        }
        Ok(Self { nanos })
    }

    pub fn set(&mut self, nanos: i64) -> Result<(), ClockError> {
        if nanos < 0 {
            return Err(ClockError);
        }
        self.nanos = nanos;
        Ok(())
    }
}

impl Clock for ManualClock {
    fn unix_nanos(&self) -> Result<i64, ClockError> {
        Ok(self.nanos)
    }
}

/// The clock could not produce a unix-nanosecond timestamp.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClockError;

impl fmt::Display for ClockError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("clock time is unavailable or out of range")
    }
}

impl std::error::Error for ClockError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manual_clock_returns_the_configured_time() {
        let mut clock = ManualClock::new(1_700_000_000_000_000_000).unwrap();
        assert_eq!(clock.unix_nanos().unwrap(), 1_700_000_000_000_000_000);
        clock.set(42).unwrap();
        assert_eq!(clock.unix_nanos().unwrap(), 42);
        assert!(ManualClock::new(-1).is_err());
    }
}
