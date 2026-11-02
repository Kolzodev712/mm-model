//! Bounded reconnect backoff. The wait itself is performed by the client.

use std::time::Duration;

/// How many times a client may reconnect, and how long to wait.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReconnectPolicy {
    max_attempts: u32,
    base: Duration,
    cap: Duration,
}

impl ReconnectPolicy {
    pub fn new(max_attempts: u32, base: Duration, cap: Duration) -> Result<Self, &'static str> {
        if base.is_zero() || cap < base {
            return Err("reconnect backoff must be positive and cap must cover the base");
        }
        Ok(Self {
            max_attempts,
            base,
            cap,
        })
    }

    /// Backoff before attempt `attempt` (1-based). `None` means stop.
    #[must_use]
    pub fn backoff(self, attempt: u32) -> Option<Duration> {
        if attempt == 0 || attempt > self.max_attempts {
            return None;
        }
        let shift = attempt.saturating_sub(1).min(16);
        let factor = 1u32 << shift;
        let millis = self.base.as_millis().saturating_mul(u128::from(factor));
        let capped = millis.min(self.cap.as_millis());
        Some(Duration::from_millis(
            u64::try_from(capped).unwrap_or(u64::MAX),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backoff_grows_then_stops() {
        let policy =
            ReconnectPolicy::new(3, Duration::from_millis(100), Duration::from_millis(250))
                .unwrap();
        assert_eq!(policy.backoff(1), Some(Duration::from_millis(100)));
        assert_eq!(policy.backoff(2), Some(Duration::from_millis(200)));
        assert_eq!(policy.backoff(3), Some(Duration::from_millis(250)));
        assert_eq!(policy.backoff(4), None);
        assert_eq!(policy.backoff(0), None);
    }
}
