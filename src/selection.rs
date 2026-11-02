//! Deterministic choice between evaluated strategies and not quoting.
//!
//! `NoQuote` has expected value 0. A strategy is chosen only when its expected
//! value is strictly greater than that. Nothing here applies a risk penalty,
//! confidence interval, or inventory adjustment.
//!
//! Expected value is [`StrategyEvaluation::expected_value`], the fill
//! probability times average P&L if filled.

use std::cmp::Ordering;

use crate::evaluation::StrategyEvaluation;
use crate::StrategyId;

/// Action with the highest expected value.
#[derive(Debug, Clone, PartialEq)]
pub enum QuoteDecision {
    /// Do not send a quote. Expected value is 0.
    NoQuote,
    /// Quote with the selected strategy.
    Quote {
        /// Strategy whose expected value won.
        strategy_id: StrategyId,
        /// Expected value per quoting attempt for that strategy.
        expected_value: f64,
    },
}

impl QuoteDecision {
    /// Expected value of the chosen action.
    pub fn expected_value(&self) -> f64 {
        match self {
            Self::NoQuote => 0.0,
            Self::Quote { expected_value, .. } => *expected_value,
        }
    }

    /// Strategy to quote, if the decision is to quote.
    pub fn strategy_id(&self) -> Option<&StrategyId> {
        match self {
            Self::NoQuote => None,
            Self::Quote { strategy_id, .. } => Some(strategy_id),
        }
    }
}

/// Selects the action with the highest expected value.
///
/// When no strategy has expected value strictly above 0, the result is
/// [`QuoteDecision::NoQuote`]. That includes an empty list, strategies with
/// no fills, and strategies whose expected value is only equal to not quoting.
///
/// If several strategies share the best expected value above 0, the
/// lexicographically smallest [`StrategyId`] wins so the choice is stable.
/// Non-finite expected values are ignored.
pub fn select_quote(evaluations: &[StrategyEvaluation]) -> QuoteDecision {
    let best = evaluations
        .iter()
        .filter(|evaluation| evaluation.expected_value.is_finite())
        .reduce(
            |best, candidate| match candidate.expected_value.total_cmp(&best.expected_value) {
                Ordering::Greater => candidate,
                Ordering::Equal if candidate.strategy_id < best.strategy_id => candidate,
                _ => best,
            },
        );

    match best {
        Some(evaluation) if evaluation.expected_value > 0.0 => QuoteDecision::Quote {
            strategy_id: evaluation.strategy_id.clone(),
            expected_value: evaluation.expected_value,
        },
        _ => QuoteDecision::NoQuote,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{FillOutcome, HistoricalObservation, StrategyId};
    use crate::evaluation::evaluate_strategies;
    use crate::test_fixture;

    fn filled(strategy: &str, fair_value: f64) -> HistoricalObservation {
        HistoricalObservation::new(
            StrategyId::new(strategy).unwrap(),
            100.0,
            1.0,
            FillOutcome::Filled {
                execution_price: 100.0,
            },
            fair_value,
            0.0,
        )
        .unwrap()
    }

    #[test]
    fn selects_strategy_with_highest_ev() {
        let decision = select_quote(&evaluate_strategies(&test_fixture::example_dataset()));
        match decision {
            QuoteDecision::Quote {
                strategy_id,
                expected_value,
            } => {
                assert_eq!(strategy_id.as_str(), "B");
                assert!((expected_value - 0.06).abs() < 1e-9);
            }
            QuoteDecision::NoQuote => panic!("strategy B has positive expected value"),
        }
    }

    #[test]
    fn selects_no_quote_when_all_ev_negative() {
        let observations = vec![filled("A", 99.0), filled("B", 99.5)];
        let decision = select_quote(&evaluate_strategies(&observations));
        assert_eq!(decision, QuoteDecision::NoQuote);
        assert_eq!(decision.expected_value(), 0.0);
    }

    #[test]
    fn empty_input_selects_no_quote() {
        assert_eq!(select_quote(&[]), QuoteDecision::NoQuote);
    }

    #[test]
    fn zero_fill_strategy_ties_no_quote() {
        let missed = HistoricalObservation::new(
            StrategyId::new("idle").unwrap(),
            99.0,
            1.0,
            FillOutcome::Unfilled,
            100.0,
            0.0,
        )
        .unwrap();
        let decision = select_quote(&evaluate_strategies(&[missed]));
        assert_eq!(decision, QuoteDecision::NoQuote);
    }

    #[test]
    fn equal_positive_ev_prefers_smaller_strategy_id() {
        let observations = vec![filled("C", 100.25), filled("A", 100.25)];
        assert_eq!(
            select_quote(&evaluate_strategies(&observations))
                .strategy_id()
                .unwrap()
                .as_str(),
            "A"
        );
    }
}
