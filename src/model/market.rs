//! Model family B: a binary outcome of the public book.
//!
//! Version 1 has one target. The label uses the reference price at `t` and the
//! reference price at `t + H`. Features must be built before that future price
//! is read.

use std::fmt;

/// What a market-outcome model is estimating.
///
/// Only [`Self::UpMove`] is implemented. Further targets would be additional
/// variants, not a plugin system.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(
    feature = "artifact",
    derive(serde::Serialize, serde::Deserialize),
    serde(rename_all = "snake_case")
)]
pub enum MarketOutcomeDefinition {
    /// `FV_(t+H)` is at least `threshold_bps` above `FV_t`.
    UpMove {
        /// Requested label delay, nanoseconds.
        horizon_ns: i64,
        /// Minimum upward return, in basis points.
        threshold_bps: f64,
    },
}

/// The target definition cannot be used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutcomeError {
    /// Horizon was not strictly positive.
    Horizon,
    /// Threshold was negative or not finite.
    Threshold,
    /// A reference price was not a positive finite number.
    ReferencePrice,
}

impl fmt::Display for OutcomeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::Horizon => "market-outcome horizon must be positive",
            Self::Threshold => "up-move threshold must be finite and non-negative",
            Self::ReferencePrice => "reference price must be finite and positive",
        };
        f.write_str(message)
    }
}

impl std::error::Error for OutcomeError {}

impl MarketOutcomeDefinition {
    /// Checks the horizon and threshold.
    pub fn new_up_move(horizon_ns: i64, threshold_bps: f64) -> Result<Self, OutcomeError> {
        if horizon_ns <= 0 {
            return Err(OutcomeError::Horizon);
        }
        if !threshold_bps.is_finite() || threshold_bps < 0.0 {
            return Err(OutcomeError::Threshold);
        }
        Ok(Self::UpMove {
            horizon_ns,
            threshold_bps,
        })
    }

    /// Requested label delay.
    #[must_use]
    pub const fn horizon_ns(self) -> i64 {
        match self {
            Self::UpMove { horizon_ns, .. } => horizon_ns,
        }
    }

    /// Minimum upward return, in basis points.
    #[must_use]
    pub const fn threshold_bps(self) -> f64 {
        match self {
            Self::UpMove { threshold_bps, .. } => threshold_bps,
        }
    }

    /// `(future - now) / now * 10_000`.
    pub fn return_bps(now: f64, future: f64) -> Result<f64, OutcomeError> {
        if !(now.is_finite() && now > 0.0 && future.is_finite() && future > 0.0) {
            return Err(OutcomeError::ReferencePrice);
        }
        Ok((future - now) / now * 10_000.0)
    }

    /// Binary label for this definition.
    ///
    /// An up move is true when the return is greater than or equal to the
    /// threshold, including an exact tie with the threshold.
    pub fn label(self, now: f64, future: f64) -> Result<bool, OutcomeError> {
        let return_bps = Self::return_bps(now, future)?;
        Ok(return_bps >= self.threshold_bps())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_exact_threshold_is_a_positive_label() {
        let target = MarketOutcomeDefinition::new_up_move(1_000, 1.0).unwrap();
        assert!(target.label(100.0, 100.01).unwrap());
        assert!(!target.label(100.0, 100.009).unwrap());
        assert!((MarketOutcomeDefinition::return_bps(100.0, 100.01).unwrap() - 1.0).abs() < 1e-9);
    }
}
