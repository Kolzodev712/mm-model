//! Prediction boundary for learned quote scores.
//!
//! Models return a fill probability and a conditional P&L. They do not compute
//! expected value and they do not select a quote. [`predict_quote`] extracts
//! features once, calls both models, and multiplies:
//!
//! ```text
//! EV = P(fill | X) * E[P&L | fill, X]
//! ```
//!
//! `NoQuote` is not represented here. Its expected value stays 0 in selection.

use std::fmt;

use crate::decision::{CandidateQuote, MarketState};
use crate::features::{extract_features, FeatureVector, FEATURE_COUNT};

/// Predicts `P(fill | features)` in `[0, 1]`.
pub trait FillProbabilityModel<const N: usize> {
    /// Fill probability for one decision-time feature row.
    fn predict_fill_probability(&self, features: &FeatureVector<N>) -> f64;
}

/// Predicts `E[P&L | fill, features]`.
///
/// Implementations are trained on filled outcomes only.
pub trait ConditionalPnlModel<const N: usize> {
    /// Conditional mark-to-fair P&L for one decision-time feature row.
    fn predict_pnl_if_filled(&self, features: &FeatureVector<N>) -> f64;
}

/// Learned estimates for one candidate buy, plus the deterministic expected value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuotePrediction {
    /// Fill probability after clamping into `[0, 1]`.
    pub fill_probability: f64,
    /// Model output for P&L given a fill. This is not multiplied by probability.
    pub expected_pnl_if_filled: f64,
    /// `fill_probability * expected_pnl_if_filled`.
    pub expected_value: f64,
}

/// Full-batch gradient descent settings shared by the reference trainers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrainConfig {
    /// Step size. Must be finite and strictly positive.
    pub learning_rate: f64,
    /// Number of passes over the sample slice. Zero leaves weights unchanged.
    pub epochs: u32,
}

/// Trainer input that cannot be run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainError {
    /// No samples were supplied.
    EmptyDataset,
    /// Learning rate was non-positive or not finite.
    InvalidLearningRate,
}

impl fmt::Display for TrainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::EmptyDataset => "training requires at least one sample",
            Self::InvalidLearningRate => "learning rate must be finite and positive",
        };
        f.write_str(message)
    }
}

impl std::error::Error for TrainError {}

/// A model weight was NaN or infinite.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidWeights;

impl fmt::Display for InvalidWeights {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("model weights must be finite")
    }
}

impl std::error::Error for InvalidWeights {}

/// A model returned a non-finite score.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredictError {
    /// Fill probability was NaN or infinite.
    NonFiniteFillProbability,
    /// Conditional P&L was NaN or infinite.
    NonFiniteConditionalPnl,
    /// The output slice length did not match the candidate slice.
    CandidateCountMismatch,
}

impl fmt::Display for PredictError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::NonFiniteFillProbability => "fill probability must be finite",
            Self::NonFiniteConditionalPnl => "conditional pnl must be finite",
            Self::CandidateCountMismatch => "candidate and output slices have different lengths",
        };
        f.write_str(message)
    }
}

impl std::error::Error for PredictError {}

/// Scores one candidate from a market state whose reference price is already set.
///
/// Feature extraction and both predictions allocate nothing. The reference
/// price is read from `market` and is not estimated again. Fill probability
/// is clamped into `[0, 1]` before the product. The result is a shadow score,
/// not an order.
pub fn predict_quote<F, P>(
    market: &MarketState,
    quote: &CandidateQuote,
    fill_model: &F,
    pnl_model: &P,
) -> Result<QuotePrediction, PredictError>
where
    F: FillProbabilityModel<FEATURE_COUNT>,
    P: ConditionalPnlModel<FEATURE_COUNT>,
{
    let features = extract_features(market, quote);
    let probability = fill_model.predict_fill_probability(&features);
    if !probability.is_finite() {
        return Err(PredictError::NonFiniteFillProbability);
    }
    let conditional = pnl_model.predict_pnl_if_filled(&features);
    if !conditional.is_finite() {
        return Err(PredictError::NonFiniteConditionalPnl);
    }
    let fill_probability = probability.clamp(0.0, 1.0);
    Ok(QuotePrediction {
        fill_probability,
        expected_pnl_if_filled: conditional,
        expected_value: fill_probability * conditional,
    })
}

/// Scores every candidate against one already-estimated market state.
///
/// `out` must have one slot per quote. No reference price is recomputed and
/// no heap memory is allocated.
pub fn score_candidates<F, P>(
    market: &MarketState,
    quotes: &[CandidateQuote],
    fill_model: &F,
    pnl_model: &P,
    out: &mut [QuotePrediction],
) -> Result<(), PredictError>
where
    F: FillProbabilityModel<FEATURE_COUNT>,
    P: ConditionalPnlModel<FEATURE_COUNT>,
{
    if quotes.len() != out.len() {
        return Err(PredictError::CandidateCountMismatch);
    }
    for (quote, slot) in quotes.iter().zip(out.iter_mut()) {
        *slot = predict_quote(market, quote, fill_model, pnl_model)?;
    }
    Ok(())
}

pub(crate) fn validate_train_config(config: TrainConfig) -> Result<(), TrainError> {
    if !config.learning_rate.is_finite() || config.learning_rate <= 0.0 {
        return Err(TrainError::InvalidLearningRate);
    }
    Ok(())
}

pub(crate) fn finite_weights<const N: usize>(
    bias: f64,
    weights: &[f64; N],
) -> Result<(), InvalidWeights> {
    if !bias.is_finite() || weights.iter().any(|weight| !weight.is_finite()) {
        return Err(InvalidWeights);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decision::CandidateQuote;
    use crate::fair_value::FairValueMethod;
    use crate::features::FEATURE_COUNT;
    use crate::linear::LinearRegression;
    use crate::logistic::LogisticRegression;
    use crate::selection::{select_action, ActionScore, QuoteDecision};
    use crate::StrategyId;

    #[test]
    fn expected_value_is_the_product_of_the_two_predictions() {
        let market = FairValueMethod::Microprice
            .derive_top(99.9, 2.0, 100.1, 1.0)
            .unwrap();
        let quote = CandidateQuote::new(99.8, 1.0).unwrap();
        let other = CandidateQuote::new(99.7, 1.0).unwrap();
        let mut scored = [QuotePrediction {
            fill_probability: 0.0,
            expected_pnl_if_filled: 0.0,
            expected_value: 0.0,
        }; 2];
        score_candidates(
            &market,
            &[quote, other],
            &LogisticRegression::<FEATURE_COUNT>::zeros(),
            &LinearRegression::from_weights(0.4, [0.0; FEATURE_COUNT]).unwrap(),
            &mut scored,
        )
        .unwrap();
        assert_eq!(scored[0].fill_probability, scored[1].fill_probability);
        assert_eq!(
            market.fair_value(),
            FairValueMethod::Microprice
                .estimate(99.9, 2.0, 100.1, 1.0)
                .unwrap()
        );
        let fill_model = LogisticRegression::<FEATURE_COUNT>::zeros();
        let pnl_model = LinearRegression::from_weights(0.4, [0.0; FEATURE_COUNT]).unwrap();

        let prediction = predict_quote(&market, &quote, &fill_model, &pnl_model).unwrap();
        assert_eq!(prediction.fill_probability, 0.5);
        assert_eq!(prediction.expected_pnl_if_filled, 0.4);
        assert_eq!(
            prediction.expected_value,
            prediction.fill_probability * prediction.expected_pnl_if_filled
        );

        let better = StrategyId::new("B").unwrap();
        let worse = StrategyId::new("A").unwrap();
        let chosen = select_action(&[
            ActionScore {
                strategy_id: &worse,
                expected_value: -1.0,
            },
            ActionScore {
                strategy_id: &better,
                expected_value: prediction.expected_value,
            },
        ]);
        assert_eq!(
            chosen,
            QuoteDecision::Quote {
                strategy_id: better,
                expected_value: prediction.expected_value,
            }
        );

        let negative = predict_quote(
            &market,
            &quote,
            &fill_model,
            &LinearRegression::from_weights(-1.0, [0.0; FEATURE_COUNT]).unwrap(),
        )
        .unwrap();
        assert!(negative.expected_value < 0.0);
        let only = StrategyId::new("A").unwrap();
        assert_eq!(
            select_action(&[ActionScore {
                strategy_id: &only,
                expected_value: negative.expected_value,
            }]),
            QuoteDecision::NoQuote
        );
    }
}
