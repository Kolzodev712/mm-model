//! Deterministic choice between evaluated strategies and not quoting.
//!
//! `NoQuote` has expected value 0. A strategy is chosen only when its expected
//! value is strictly greater than that. Nothing here applies a risk penalty,
//! confidence interval, or inventory adjustment.
//!
//! Expected value is fill probability times conditional P&L. Historical
//! strategy rows and learned scores both use [`select_action`].
//!
//! The result is an offline or shadow choice. It is not an instruction to
//! send an order.

use std::cmp::Ordering;

use crate::domain::outcome::StrategyId;

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

/// Applies the historical rule to expected values from any source.
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

/// Copyable learned-candidate choice. No strategy-name string is cloned.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LearnedChoice {
    /// Do not quote. Expected value is 0.
    NoQuote,
    /// The candidate with the highest expected value.
    Quote {
        id: crate::domain::units::CandidateId,
        expected_value: f64,
    },
}

/// One learned candidate score.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScoredCandidate {
    pub id: crate::domain::units::CandidateId,
    pub expected_value: f64,
}

/// Selects among learned candidates with the same rule as [`select_action`].
///
/// Ties keep the smaller [`crate::domain::units::CandidateId`]. This function does not allocate.
#[must_use]
pub fn select_learned(scores: &[ScoredCandidate]) -> LearnedChoice {
    let best = scores
        .iter()
        .filter(|score| score.expected_value.is_finite())
        .reduce(
            |best, candidate| match candidate.expected_value.total_cmp(&best.expected_value) {
                Ordering::Greater => candidate,
                Ordering::Equal if candidate.id < best.id => candidate,
                _ => best,
            },
        );
    match best {
        Some(score) if score.expected_value > 0.0 => LearnedChoice::Quote {
            id: score.id,
            expected_value: score.expected_value,
        },
        _ => LearnedChoice::NoQuote,
    }
}

pub(crate) fn select_ranked<'a>(
    candidates: impl Iterator<Item = (&'a StrategyId, f64)>,
) -> QuoteDecision {
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
    use crate::domain::units::CandidateId;

    #[test]
    fn learned_tie_keeps_the_smaller_id() {
        let scores = [
            ScoredCandidate {
                id: CandidateId::new(2),
                expected_value: 1.0,
            },
            ScoredCandidate {
                id: CandidateId::new(1),
                expected_value: 1.0,
            },
        ];
        assert_eq!(
            select_learned(&scores),
            LearnedChoice::Quote {
                id: CandidateId::new(1),
                expected_value: 1.0,
            }
        );
    }

    #[test]
    fn non_positive_learned_scores_are_no_quote() {
        let scores = [ScoredCandidate {
            id: CandidateId::new(0),
            expected_value: 0.0,
        }];
        assert_eq!(select_learned(&scores), LearnedChoice::NoQuote);
    }
}
