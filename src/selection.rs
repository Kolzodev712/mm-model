//! Deterministic choice between evaluated strategies and not quoting.
//!
//! `NoQuote` has expected value 0. A strategy is chosen only when its expected
//! value is strictly greater than that. Nothing here applies a risk penalty,
//! confidence interval, or inventory adjustment.
//!
//! Expected value is fill probability times conditional P&L. For historical
//! strategies that value is [`StrategyEvaluation::expected_value`]. Learned
//! scores use the same comparison through [`select_action`].
//!
//! The result is an offline or shadow choice. It is not an instruction to
//! send an order.

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

/// One candidate action and its expected value per attempt.
///
/// `strategy_id` is borrowed. Selection clones only the winning id.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ActionScore<'a> {
    pub strategy_id: &'a StrategyId,
    pub expected_value: f64,
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
#[must_use]
pub fn select_quote(evaluations: &[StrategyEvaluation]) -> QuoteDecision {
    select_ranked(
        evaluations
            .iter()
            .map(|evaluation| (&evaluation.strategy_id, evaluation.expected_value)),
    )
}

/// Applies [`select_quote`]'s rule to expected values from any source.
///
/// Historical aggregation and learned models both end here. `NoQuote` has
/// expected value 0. This does not apply a risk check or send an order.
#[must_use]
pub fn select_action(actions: &[ActionScore<'_>]) -> QuoteDecision {
    select_ranked(
        actions
            .iter()
            .map(|action| (action.strategy_id, action.expected_value)),
    )
}

fn select_ranked<'a>(candidates: impl Iterator<Item = (&'a StrategyId, f64)>) -> QuoteDecision {
    let best = candidates
        .filter(|(_, expected_value)| expected_value.is_finite())
        .reduce(|best, candidate| match candidate.1.total_cmp(&best.1) {
            Ordering::Greater => candidate,
            Ordering::Equal if candidate.0 < best.0 => candidate,
            _ => best,
        });

    match best {
        Some((strategy_id, expected_value)) if expected_value > 0.0 => QuoteDecision::Quote {
            strategy_id: strategy_id.clone(),
            expected_value,
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
    fn learned_scores_use_the_same_rule() {
        let evaluations = evaluate_strategies(&test_fixture::example_dataset());
        let scores: Vec<ActionScore<'_>> = evaluations
            .iter()
            .map(|evaluation| ActionScore {
                strategy_id: &evaluation.strategy_id,
                expected_value: evaluation.expected_value,
            })
            .collect();
        assert_eq!(select_action(&scores), select_quote(&evaluations));
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
