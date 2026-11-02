//! Deterministic demo weights used only to exercise live inference.
//!
//! The quote rows are [`LabelProvenance::Simulated`]. They are not exchange fills.
//! Output from these models is not a validated artifact.

use crate::data::training::example::{
    logistic_samples, DecisionTimeInputs, LabelProvenance, LabelTiming, ObservedOutcome,
    TrainingExample,
};
use crate::domain::market::QuoteContext;
use crate::model::fair_value::FairValueMethod;
use crate::model::feature::market::MARKET_FEATURE_COUNT;
use crate::model::feature::v1::{FeatureVector, FEATURE_COUNT};
use crate::model::fill::{FillSample, LogisticRegression};
use crate::model::prediction::{TrainConfig, TrainError};

pub const DEMO_WARNING: &str = "DEMO MODEL — probabilities are not from a validated artifact";

/// Trains the demo quote-fill model on a fixed simulated sample.
pub fn train_demo_quote_model() -> Result<LogisticRegression<FEATURE_COUNT>, TrainError> {
    let rows = demo_rows();
    debug_assert!(rows
        .iter()
        .all(|row| row.provenance() == LabelProvenance::Simulated));
    let samples = logistic_samples(
        &rows,
        LabelProvenance::Simulated,
        FairValueMethod::Microprice,
    )
    .map_err(|_| TrainError::EmptyDataset)?;
    let mut model = LogisticRegression::<FEATURE_COUNT>::zeros();
    model.train(
        &samples,
        TrainConfig {
            learning_rate: 0.2,
            epochs: 80,
        },
    )?;
    Ok(model)
}

/// Trains the demo market model on a fixed feature sample.
pub fn train_demo_market_model() -> Result<LogisticRegression<MARKET_FEATURE_COUNT>, TrainError> {
    let samples = [
        FillSample {
            features: FeatureVector::new([1.0, 0.2, 0.1]).unwrap(),
            positive: true,
        },
        FillSample {
            features: FeatureVector::new([1.0, -0.4, -0.2]).unwrap(),
            positive: false,
        },
        FillSample {
            features: FeatureVector::new([2.0, 0.5, 0.3]).unwrap(),
            positive: true,
        },
        FillSample {
            features: FeatureVector::new([4.0, -0.8, -0.4]).unwrap(),
            positive: false,
        },
    ];
    let mut model = LogisticRegression::<MARKET_FEATURE_COUNT>::zeros();
    model.train(
        &samples,
        TrainConfig {
            learning_rate: 0.2,
            epochs: 80,
        },
    )?;
    Ok(model)
}

fn demo_rows() -> Vec<TrainingExample> {
    let method = FairValueMethod::Microprice;
    let market = method.derive_top(99.9, 2.0, 100.1, 1.0).unwrap();
    let timing = LabelTiming {
        decision_ns: 1_000,
        decision_snapshot_ns: 1_000,
        requested_horizon_ns: 500,
        horizon_snapshot_ns: 1_500,
    };
    let specs = [
        (99.5, true, 0.4),
        (99.7, true, 0.2),
        (99.95, false, 0.0),
        (99.4, true, 0.5),
        (99.98, false, 0.0),
    ];
    specs
        .into_iter()
        .map(|(price, filled, pnl)| {
            let outcome = if filled {
                ObservedOutcome::Filled {
                    pnl,
                    horizon_fair_value: 100.2,
                }
            } else {
                ObservedOutcome::Unfilled
            };
            TrainingExample::new(
                DecisionTimeInputs::new(market, QuoteContext::new(price, 1.0).unwrap()),
                outcome,
                LabelProvenance::Simulated,
                method,
                timing,
            )
            .unwrap()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn demo_training_is_simulated_and_deterministic() {
        let rows = demo_rows();
        assert!(rows
            .iter()
            .all(|row| row.provenance() == LabelProvenance::Simulated));
        let first = train_demo_quote_model().unwrap();
        let second = train_demo_quote_model().unwrap();
        assert_eq!(first.bias(), second.bias());
        assert_eq!(first.weights(), second.weights());
        let market = train_demo_market_model().unwrap();
        assert!(market.bias().is_finite());
    }
}
