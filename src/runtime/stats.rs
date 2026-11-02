//! Online aggregates for one shadow run.
//!
//! Counts, sums, min, and max only. Per-tick samples are not retained.
//! Selection counts are a fixed array indexed by [`CandidateId`].

use crate::domain::units::CandidateId;
use crate::model::candidate::MAX_CANDIDATES;
use crate::model::prediction::QuotePrediction;
use crate::model::selection::LearnedChoice;

/// What one evaluated book contributed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BookOutcome {
    pub choice: LearnedChoice,
    pub selected: Option<QuotePrediction>,
    pub candidate_count: usize,
    pub best_bid: f64,
    pub best_ask: f64,
    pub fair_value: f64,
    pub latency_ns: u128,
}

/// Running totals. These are model predictions, not realized trading results.
#[derive(Debug, Clone, PartialEq)]
pub struct RunStats {
    pub books_received: u64,
    pub books_processed: u64,
    pub books_rejected: u64,
    pub client_errors: u64,
    pub model_errors: u64,
    pub reconnects: u64,
    pub evaluations: u64,
    pub quote_decisions: u64,
    pub noquote_decisions: u64,
    selection_counts: [u64; MAX_CANDIDATES],
    selected_count: u64,
    sum_probability: f64,
    sum_conditional_pnl: f64,
    sum_ev: f64,
    min_ev: f64,
    max_ev: f64,
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
            evaluations: 0,
            quote_decisions: 0,
            noquote_decisions: 0,
            selection_counts: [0; MAX_CANDIDATES],
            selected_count: 0,
            sum_probability: 0.0,
            sum_conditional_pnl: 0.0,
            sum_ev: 0.0,
            min_ev: f64::INFINITY,
            max_ev: f64::NEG_INFINITY,
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
        self.evaluations += outcome.candidate_count as u64;
        self.observe_latency(outcome.latency_ns);
        match outcome.choice {
            LearnedChoice::NoQuote => self.noquote_decisions += 1,
            LearnedChoice::Quote { id, expected_value } => {
                self.quote_decisions += 1;
                if let Some(slot) = self.selection_counts.get_mut(id.get() as usize) {
                    *slot += 1;
                }
                if let Some(prediction) = outcome.selected {
                    self.selected_count += 1;
                    self.sum_probability += prediction.fill_probability.get();
                    self.sum_conditional_pnl += prediction.expected_pnl_if_filled.get();
                    self.sum_ev += expected_value;
                    self.min_ev = self.min_ev.min(expected_value);
                    self.max_ev = self.max_ev.max(expected_value);
                }
            }
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
    pub fn selection_count(&self, id: CandidateId) -> u64 {
        self.selection_counts
            .get(id.get() as usize)
            .copied()
            .unwrap_or(0)
    }

    #[must_use]
    pub fn mean_selected_probability(&self) -> Option<f64> {
        mean(self.sum_probability, self.selected_count)
    }

    #[must_use]
    pub fn mean_selected_conditional_pnl(&self) -> Option<f64> {
        mean(self.sum_conditional_pnl, self.selected_count)
    }

    #[must_use]
    pub fn mean_selected_ev(&self) -> Option<f64> {
        mean(self.sum_ev, self.selected_count)
    }

    #[must_use]
    pub fn min_selected_ev(&self) -> Option<f64> {
        (self.selected_count > 0).then_some(self.min_ev)
    }

    #[must_use]
    pub fn max_selected_ev(&self) -> Option<f64> {
        (self.selected_count > 0).then_some(self.max_ev)
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

fn mean(sum: f64, count: u64) -> Option<f64> {
    (count > 0).then_some(sum / count as f64)
}

/// Whether the outer run should stop or emit a progress line.
///
/// `elapsed` comes from [`std::time::Instant`], not from wall-clock subtraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RunPolicy {
    pub duration: std::time::Duration,
    pub report_every: std::time::Duration,
}

impl RunPolicy {
    #[must_use]
    pub fn should_stop(self, elapsed: std::time::Duration) -> bool {
        elapsed >= self.duration
    }

    #[must_use]
    pub fn should_report(
        self,
        elapsed: std::time::Duration,
        last_report: std::time::Duration,
    ) -> bool {
        elapsed.saturating_sub(last_report) >= self.report_every
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::units::{CandidateId, Pnl, Probability};
    use std::time::Duration;

    fn prediction(ev: f64) -> QuotePrediction {
        QuotePrediction {
            fill_probability: Probability::new(0.25).unwrap(),
            expected_pnl_if_filled: Pnl::new(ev / 0.25).unwrap(),
            expected_value: ev,
        }
    }

    #[test]
    fn stats_accumulate_without_storing_ticks() {
        let mut stats = RunStats::default();
        stats.record_book(BookOutcome {
            choice: LearnedChoice::Quote {
                id: CandidateId::new(1),
                expected_value: 0.2,
            },
            selected: Some(prediction(0.2)),
            candidate_count: 2,
            best_bid: 1.0,
            best_ask: 2.0,
            fair_value: 1.5,
            latency_ns: 100,
        });
        stats.record_book(BookOutcome {
            choice: LearnedChoice::NoQuote,
            selected: None,
            candidate_count: 2,
            best_bid: 1.0,
            best_ask: 2.0,
            fair_value: 1.5,
            latency_ns: 300,
        });
        assert_eq!(stats.books_processed, 2);
        assert_eq!(stats.quote_decisions, 1);
        assert_eq!(stats.noquote_decisions, 1);
        assert_eq!(stats.evaluations, 4);
        assert_eq!(stats.selection_count(CandidateId::new(1)), 1);
        assert_eq!(stats.mean_selected_ev(), Some(0.2));
        assert_eq!(stats.max_latency_ns(), Some(300));
        assert_eq!(stats.mean_latency_ns(), Some(200));
    }

    #[test]
    fn run_policy_uses_elapsed_duration_without_sleeping() {
        let policy = RunPolicy {
            duration: Duration::from_secs(60),
            report_every: Duration::from_secs(5),
        };
        assert!(!policy.should_stop(Duration::from_secs(59)));
        assert!(policy.should_stop(Duration::from_secs(60)));
        assert!(policy.should_report(Duration::from_secs(5), Duration::ZERO));
        assert!(!policy.should_report(Duration::from_secs(9), Duration::from_secs(5)));
    }
}
