//! Per-strategy estimates from historical quoting attempts.
//!
//! Observations that share a [`StrategyId`] are pooled.
//! Each attempt is treated as independent. There is no clock, inventory path,
//! or alignment between strategies: a fill on one strategy is not evidence
//! about whether another quote would have filled at that moment.
//!
//! The estimates below are the seam a later model can replace:
//!
//! - `P(fill | market state, candidate quote)`
//! - expected P&L conditional on execution
//!
//! [`crate::model::selection`] consumes the resulting expected value and does not
//! care whether it came from these counts or from a model.
//!
//! Learned fill and conditional-P&L models live beside this module. They do
//! not replace these historical counts.

use std::collections::BTreeMap;

use crate::domain::{HistoricalObservation, StrategyId};
use crate::model::selection::QuoteDecision;

/// Aggregated historical performance of one strategy.
#[derive(Debug, Clone, PartialEq)]
pub struct StrategyEvaluation {
    /// Strategy these attempts belong to.
    pub strategy_id: StrategyId,
    /// Quoting attempts observed for this strategy.
    pub attempts: usize,
    /// Attempts that filled in full.
    pub fills: usize,
    /// `fills / attempts`.
    pub fill_probability: f64,
    /// Sum of attempt P&L. Unfilled attempts contribute 0.
    pub total_pnl: f64,
    /// Mean P&L over filled attempts.
    ///
    /// `None` when `fills` is zero. The conditional average is undefined in
    /// that case; it is not reported as zero.
    pub average_pnl_per_fill: Option<f64>,
    /// `fill_probability * average_pnl_per_fill`.
    ///
    /// Zero when there are no fills, because fill probability is then zero
    /// and the missing conditional average contributes no P&L.
    pub expected_value: f64,
    /// `total_pnl / attempts`.
    ///
    /// Under the baseline assumptions this matches [`Self::expected_value`]
    /// up to floating-point error. Unfilled P&L is zero, so total P&L is the
    /// sum of filled P&L.
    pub expected_value_from_total: f64,
}

impl StrategyEvaluation {
    /// Reports whether the absolute difference of the two expected values is at most `tolerance`.
    pub fn expected_values_agree(&self, tolerance: f64) -> bool {
        (self.expected_value - self.expected_value_from_total).abs() <= tolerance
    }
}

/// Evaluates every strategy that appears in `observations`.
///
/// The result is ordered by [`StrategyId`]. An empty input returns an empty
/// vector. Strategies with attempts but no fills are included: fill
/// probability and both expected values are zero, and
/// [`StrategyEvaluation::average_pnl_per_fill`] is `None`.
pub fn evaluate_strategies(observations: &[HistoricalObservation]) -> Vec<StrategyEvaluation> {
    let mut grouped: BTreeMap<&StrategyId, Vec<&HistoricalObservation>> = BTreeMap::new();
    for observation in observations {
        grouped
            .entry(observation.strategy_id())
            .or_default()
            .push(observation);
    }

    grouped
        .into_iter()
        .map(|(strategy_id, observations)| evaluate_group((*strategy_id).clone(), &observations))
        .collect()
}

/// Selects the historical strategy with the highest expected value.
///
/// When no strategy has expected value strictly above 0, the result is
/// [`QuoteDecision::NoQuote`]. Ties keep the lexicographically smallest
/// [`StrategyId`]. Non-finite expected values are ignored.
#[must_use]
pub fn select_quote(evaluations: &[StrategyEvaluation]) -> QuoteDecision {
    crate::model::selection::select_ranked(
        evaluations
            .iter()
            .map(|evaluation| (&evaluation.strategy_id, evaluation.expected_value)),
    )
}

fn evaluate_group(
    strategy_id: StrategyId,
    observations: &[&HistoricalObservation],
) -> StrategyEvaluation {
    debug_assert!(!observations.is_empty());

    let attempt_count = observations.len();
    let mut fills = 0usize;
    let mut total_pnl = 0.0;
    let mut filled_pnl = 0.0;

    for observation in observations {
        let pnl = observation.pnl();
        total_pnl += pnl;
        if observation.outcome().is_filled() {
            fills += 1;
            filled_pnl += pnl;
        }
    }

    let fill_probability = fills as f64 / attempt_count as f64;
    let average_pnl_per_fill = if fills == 0 {
        None
    } else {
        Some(filled_pnl / fills as f64)
    };
    let expected_value = match average_pnl_per_fill {
        Some(average) => fill_probability * average,
        None => 0.0,
    };
    let expected_value_from_total = total_pnl / attempt_count as f64;

    StrategyEvaluation {
        strategy_id,
        attempts: attempt_count,
        fills,
        fill_probability,
        total_pnl,
        average_pnl_per_fill,
        expected_value,
        expected_value_from_total,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::baseline::fixture;
    use crate::domain::{FillOutcome, HistoricalObservation, StrategyId};

    const TOLERANCE: f64 = 1e-9;

    fn approx(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= TOLERANCE,
            "actual {actual} != expected {expected}"
        );
    }

    fn observation(
        strategy: &str,
        filled: bool,
        execution_or_bid: f64,
        fair_value: f64,
    ) -> HistoricalObservation {
        let outcome = if filled {
            FillOutcome::Filled {
                execution_price: execution_or_bid,
            }
        } else {
            FillOutcome::Unfilled
        };
        HistoricalObservation::new(
            StrategyId::new(strategy).unwrap(),
            execution_or_bid,
            1.0,
            outcome,
            fair_value,
            0.0,
        )
        .unwrap()
    }

    /// Two fills (P&L +2 and -0.5) and one miss.
    fn mixed_strategy() -> Vec<HistoricalObservation> {
        vec![
            observation("mixed", true, 100.0, 102.0),
            observation("mixed", false, 100.0, 101.0),
            observation("mixed", true, 100.0, 99.5),
        ]
    }

    #[test]
    fn fill_probability() {
        let evaluation = &evaluate_strategies(&mixed_strategy())[0];
        assert_eq!(evaluation.attempts, 3);
        assert_eq!(evaluation.fills, 2);
        approx(evaluation.fill_probability, 2.0 / 3.0);
    }

    #[test]
    fn average_pnl_conditional_on_fill() {
        let evaluation = &evaluate_strategies(&mixed_strategy())[0];
        let average = evaluation.average_pnl_per_fill.expect("two fills");
        approx(average, 1.5 / 2.0);
        approx(evaluation.total_pnl, 1.5);
    }

    #[test]
    fn expected_value_from_probability_and_conditional_pnl() {
        let evaluation = &evaluate_strategies(&mixed_strategy())[0];
        let from_parts = evaluation.fill_probability * evaluation.average_pnl_per_fill.unwrap();
        approx(evaluation.expected_value, from_parts);
        approx(evaluation.expected_value, 0.5);
    }

    #[test]
    fn expected_value_from_total_pnl() {
        let evaluation = &evaluate_strategies(&mixed_strategy())[0];
        approx(
            evaluation.expected_value_from_total,
            evaluation.total_pnl / evaluation.attempts as f64,
        );
        approx(evaluation.expected_value_from_total, 0.5);
    }

    #[test]
    fn expected_value_definitions_agree() {
        let evaluation = &evaluate_strategies(&mixed_strategy())[0];
        assert!(evaluation.expected_values_agree(TOLERANCE));

        let dataset = evaluate_strategies(&fixture::example_dataset());
        assert!(dataset
            .iter()
            .all(|eval| eval.expected_values_agree(TOLERANCE)));
    }

    #[test]
    fn strategy_with_zero_fills() {
        let observations = vec![
            observation("idle", false, 99.0, 100.0),
            observation("idle", false, 99.0, 98.0),
        ];
        let evaluation = &evaluate_strategies(&observations)[0];

        assert_eq!(evaluation.attempts, 2);
        assert_eq!(evaluation.fills, 0);
        assert_eq!(evaluation.fill_probability, 0.0);
        assert_eq!(evaluation.total_pnl, 0.0);
        assert_eq!(evaluation.average_pnl_per_fill, None);
        assert_eq!(evaluation.expected_value, 0.0);
        assert_eq!(evaluation.expected_value_from_total, 0.0);
        assert!(evaluation.expected_values_agree(TOLERANCE));
    }

    #[test]
    fn empty_input() {
        assert!(evaluate_strategies(&[]).is_empty());
    }

    #[test]
    fn example_dataset_statistics() {
        let evaluations = evaluate_strategies(&fixture::example_dataset());
        assert_eq!(evaluations.len(), 2);

        let strategy_a = &evaluations[0];
        assert_eq!(strategy_a.strategy_id.as_str(), "A");
        assert_eq!(strategy_a.attempts, 5);
        assert_eq!(strategy_a.fills, 3);
        approx(strategy_a.fill_probability, 0.6);
        approx(strategy_a.total_pnl, -0.10);
        approx(strategy_a.average_pnl_per_fill.unwrap(), -0.10 / 3.0);
        approx(strategy_a.expected_value, -0.02);
        approx(strategy_a.expected_value_from_total, -0.02);
        assert!(strategy_a.expected_values_agree(TOLERANCE));

        let strategy_b = &evaluations[1];
        assert_eq!(strategy_b.strategy_id.as_str(), "B");
        assert_eq!(strategy_b.attempts, 5);
        assert_eq!(strategy_b.fills, 1);
        approx(strategy_b.fill_probability, 0.2);
        approx(strategy_b.total_pnl, 0.30);
        approx(strategy_b.average_pnl_per_fill.unwrap(), 0.30);
        approx(strategy_b.expected_value, 0.06);
        approx(strategy_b.expected_value_from_total, 0.06);
        assert!(strategy_b.expected_values_agree(TOLERANCE));
    }

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
        let evaluations = evaluate_strategies(&fixture::example_dataset());
        let scores: Vec<crate::model::selection::ActionScore<'_>> = evaluations
            .iter()
            .map(|evaluation| crate::model::selection::ActionScore {
                strategy_id: &evaluation.strategy_id,
                expected_value: evaluation.expected_value,
            })
            .collect();
        assert_eq!(
            crate::model::selection::select_action(&scores),
            select_quote(&evaluations)
        );
    }

    #[test]
    fn selects_strategy_with_highest_ev() {
        let decision = select_quote(&evaluate_strategies(&fixture::example_dataset()));
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
