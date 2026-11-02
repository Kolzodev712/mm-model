//! Online aggregates for one live-inference run.
//!
//! Counts and sums only. Individual books are not retained. Nothing here
//! records a chosen quote.

/// What one evaluated book contributed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BookOutcome {
    pub probe_count: usize,
    pub market_probability: Option<f64>,
    pub best_bid: f64,
    pub best_ask: f64,
    pub fair_value: f64,
    pub latency_ns: u128,
}

/// Running totals. These are model probabilities, not realized outcomes.
#[derive(Debug, Clone, PartialEq)]
pub struct RunStats {
    pub books_received: u64,
    pub books_processed: u64,
    pub books_rejected: u64,
    pub client_errors: u64,
    pub model_errors: u64,
    pub reconnects: u64,
    pub probe_evaluations: u64,
    market_count: u64,
    sum_market_probability: f64,
    latency_count: u64,
    latency_sum_ns: u128,
    latency_max_ns: u128,
}

impl Default for RunStats {
    fn default() -> Self {
        Self {
            books_received: 0,
            books_processed: 0,
            books_rejected: 0,
            client_errors: 0,
            model_errors: 0,
            reconnects: 0,
            probe_evaluations: 0,
            market_count: 0,
            sum_market_probability: 0.0,
            latency_count: 0,
            latency_sum_ns: 0,
            latency_max_ns: 0,
        }
    }
}

impl RunStats {
    pub fn record_book(&mut self, outcome: BookOutcome) {
        self.books_received += 1;
        self.books_processed += 1;
        self.probe_evaluations += outcome.probe_count as u64;
        self.observe_latency(outcome.latency_ns);
        if let Some(probability) = outcome.market_probability {
            self.market_count += 1;
            self.sum_market_probability += probability;
        }
    }

    pub fn record_rejection(&mut self) {
        self.books_received += 1;
        self.books_rejected += 1;
    }

    pub fn record_model_error(&mut self) {
        self.books_received += 1;
        self.model_errors += 1;
    }

    pub fn record_client_error(&mut self) {
        self.client_errors += 1;
    }

    fn observe_latency(&mut self, nanos: u128) {
        self.latency_count += 1;
        self.latency_sum_ns = self.latency_sum_ns.saturating_add(nanos);
        self.latency_max_ns = self.latency_max_ns.max(nanos);
    }

    #[must_use]
    pub fn mean_market_probability(&self) -> Option<f64> {
        (self.market_count > 0).then_some(self.sum_market_probability / self.market_count as f64)
    }

    #[must_use]
    pub fn mean_latency_ns(&self) -> Option<u128> {
        (self.latency_count > 0).then_some(self.latency_sum_ns / u128::from(self.latency_count))
    }

    #[must_use]
    pub fn max_latency_ns(&self) -> Option<u128> {
        (self.latency_count > 0).then_some(self.latency_max_ns)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stats_accumulate_probabilities_without_choosing_a_probe() {
        let mut stats = RunStats::default();
        stats.record_book(BookOutcome {
            probe_count: 4,
            market_probability: Some(0.25),
            best_bid: 99.0,
            best_ask: 101.0,
            fair_value: 100.0,
            latency_ns: 10,
        });
        stats.record_book(BookOutcome {
            probe_count: 4,
            market_probability: Some(0.75),
            best_bid: 99.0,
            best_ask: 101.0,
            fair_value: 100.0,
            latency_ns: 30,
        });
        assert_eq!(stats.probe_evaluations, 8);
        assert!((stats.mean_market_probability().unwrap() - 0.5).abs() < 1e-12);
        assert_eq!(stats.mean_latency_ns(), Some(20));
        assert_eq!(stats.max_latency_ns(), Some(30));
    }
}
