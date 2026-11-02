//! Probability outputs. Nothing here chooses a quote or emits an action.

use std::fmt;

use crate::domain::market::{MarketState, QuoteContext};
use crate::domain::units::Probability;
use crate::model::feature::market::{extract_market_features, MARKET_FEATURE_COUNT};
use crate::model::feature::v1::{extract_features, FeatureError, FeatureVector, FEATURE_COUNT};
use crate::model::market::MarketOutcomeDefinition;

/// Predicts `P(event | features)` in `[0, 1]`, or an error if the arithmetic is invalid.
pub trait FillProbabilityModel<const N: usize> {
    /// Probability for one decision-time feature row.
    fn predict_fill_probability(
        &self,
        features: &FeatureVector<N>,
    ) -> Result<Probability, PredictError>;
}

/// Model A output: the probability a supplied quote context fills within H.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuoteOutcomePrediction {
    pub fill_probability: Probability,
}

/// Model B output: the probability of one defined market outcome.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MarketOutcomePrediction {
    pub definition: MarketOutcomeDefinition,
    pub probability: Probability,
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
    /// A training prediction was not finite.
    NonFinitePrediction,
}

impl fmt::Display for TrainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::EmptyDataset => "training requires at least one sample",
            Self::InvalidLearningRate => "learning rate must be finite and positive",
            Self::NonFinitePrediction => "training prediction was not finite",
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
    /// Probability was NaN, infinite, or outside `[0, 1]`.
    InvalidProbability,
    /// Feature extraction failed.
    InvalidFeatures,
    /// The output slice length did not match the probe slice.
    ProbeCountMismatch,
}

impl fmt::Display for PredictError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::InvalidProbability => "probability must be finite and inside [0, 1]",
            Self::InvalidFeatures => "features could not be extracted",
            Self::ProbeCountMismatch => "probe and output slices have different lengths",
        };
        f.write_str(message)
    }
}

impl std::error::Error for PredictError {}

/// `P(fill within H | market state, quote context)`.
///
/// Feature extraction allocates nothing. Invalid model output is returned as
/// an error and is not replaced with `0.5`.
pub fn predict_quote_outcome<F>(
    market: &MarketState,
    quote: &QuoteContext,
    fill_model: &F,
) -> Result<QuoteOutcomePrediction, PredictError>
where
    F: FillProbabilityModel<FEATURE_COUNT>,
{
    let features = extract_features(market, quote).map_err(feature_error)?;
    let fill_probability = fill_model.predict_fill_probability(&features)?;
    Ok(QuoteOutcomePrediction { fill_probability })
}

/// Writes one fill probability per probe, in the same order as `quotes`.
pub fn predict_quote_probes<F>(
    market: &MarketState,
    quotes: &[QuoteContext],
    fill_model: &F,
    out: &mut [QuoteOutcomePrediction],
) -> Result<(), PredictError>
where
    F: FillProbabilityModel<FEATURE_COUNT>,
{
    if quotes.len() != out.len() {
        return Err(PredictError::ProbeCountMismatch);
    }
    for (quote, slot) in quotes.iter().zip(out.iter_mut()) {
        *slot = predict_quote_outcome(market, quote, fill_model)?;
    }
    Ok(())
}

/// `P(market outcome | current market features)`.
pub fn predict_market_outcome<M>(
    market: &MarketState,
    definition: MarketOutcomeDefinition,
    model: &M,
) -> Result<MarketOutcomePrediction, PredictError>
where
    M: FillProbabilityModel<MARKET_FEATURE_COUNT>,
{
    let features = extract_market_features(market).map_err(feature_error)?;
    let probability = model.predict_fill_probability(&features)?;
    Ok(MarketOutcomePrediction {
        definition,
        probability,
    })
}

fn feature_error(_: FeatureError) -> PredictError {
    PredictError::InvalidFeatures
}

pub(crate) fn finite_weights(bias: f64, weights: &[f64]) -> Result<(), InvalidWeights> {
    if bias.is_finite() && weights.iter().all(|weight| weight.is_finite()) {
        Ok(())
    } else {
        Err(InvalidWeights)
    }
}

pub(crate) fn validate_train_config(config: TrainConfig) -> Result<(), TrainError> {
    if config.learning_rate.is_finite() && config.learning_rate > 0.0 {
        Ok(())
    } else {
        Err(TrainError::InvalidLearningRate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::fair_value::FairValueMethod;
    use crate::model::fill::LogisticRegression;

    #[test]
    fn quote_probability_depends_on_the_quote_context() {
        let market = FairValueMethod::Microprice
            .derive_top(99.9, 1.0, 100.1, 1.0)
            .unwrap();
        let mut weights = [0.0; FEATURE_COUNT];
        weights[0] = -0.5;
        let model = LogisticRegression::from_weights(0.0, weights).unwrap();
        let near = QuoteContext::new(market.fair_value() * 0.999, 1.0).unwrap();
        let far = QuoteContext::new(market.fair_value() * 0.99, 1.0).unwrap();
        let near_p = predict_quote_outcome(&market, &near, &model)
            .unwrap()
            .fill_probability
            .get();
        let far_p = predict_quote_outcome(&market, &far, &model)
            .unwrap()
            .fill_probability
            .get();
        assert!(near_p > far_p);
    }

    #[test]
    fn probes_stay_in_the_supplied_order() {
        let market = FairValueMethod::Microprice
            .derive_top(99.9, 1.0, 100.1, 1.0)
            .unwrap();
        let model = LogisticRegression::<FEATURE_COUNT>::zeros();
        let quotes = [
            QuoteContext::new(99.0, 1.0).unwrap(),
            QuoteContext::new(99.5, 1.0).unwrap(),
        ];
        let mut out = [
            QuoteOutcomePrediction {
                fill_probability: Probability::new(0.0).unwrap(),
            },
            QuoteOutcomePrediction {
                fill_probability: Probability::new(0.0).unwrap(),
            },
        ];
        predict_quote_probes(&market, &quotes, &model, &mut out).unwrap();
        assert_eq!(out[0].fill_probability.get(), 0.5);
        assert_eq!(out[1].fill_probability.get(), out[0].fill_probability.get());
        assert!(predict_quote_probes(&market, &quotes, &model, &mut out[..1]).is_err());
    }
}
