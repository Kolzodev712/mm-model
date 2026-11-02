//! Deterministic synthetic model used only to exercise the shadow path.
//!
//! The rows are [`LabelProvenance::Simulated`]. They are not exchange fills.
//! Output from this model is not trading-valid.

use crate::decision::CandidateQuote;
use crate::fair_value::FairValueMethod;
use crate::features::FEATURE_COUNT;
use crate::linear::LinearRegression;
use crate::logistic::LogisticRegression;
use crate::model::{TrainConfig, TrainError};
use crate::trainset::{
    logistic_samples, pnl_samples, DecisionTimeInputs, LabelProvenance, LabelTiming,
    ObservedOutcome, TrainingExample,
};

pub const DEMO_WARNING: &str = "DEMO MODEL — predictions are not trading-valid";

/// Trains the reference models on a fixed simulated sample.
pub fn train_demo_models() -> Result<
    (
        LogisticRegression<FEATURE_COUNT>,
        LinearRegression<FEATURE_COUNT>,
    ),
    TrainError,
> {
    let rows = demo_rows();
    debug_assert!(rows
        .iter()
        .all(|row| row.provenance() == LabelProvenance::Simulated));
    let fill_rows = logistic_samples(
        &rows,
        LabelProvenance::Simulated,
        FairValueMethod::Microprice,
    )
    .map_err(|_| TrainError::EmptyDataset)?;
    let pnl_rows = pnl_samples(
        &rows,
        LabelProvenance::Simulated,
        FairValueMethod::Microprice,
    )
    .map_err(|_| TrainError::EmptyDataset)?;
    let mut fill = LogisticRegression::<FEATURE_COUNT>::zeros();
    let mut pnl = LinearRegression::<FEATURE_COUNT>::zeros();
    let config = TrainConfig {
        learning_rate: 0.2,
        epochs: 80,
    };
    fill.train(&fill_rows, config)?;
    pnl.train(&pnl_rows, config)?;
    Ok((fill, pnl))
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
                DecisionTimeInputs::new(market, CandidateQuote::new(price, 1.0).unwrap()),
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
        let (first_fill, first_pnl) = train_demo_models().unwrap();
        let (second_fill, second_pnl) = train_demo_models().unwrap();
        assert_eq!(first_fill.bias(), second_fill.bias());
        assert_eq!(first_fill.weights(), second_fill.weights());
        assert_eq!(first_pnl.bias(), second_pnl.bias());
        assert_eq!(first_pnl.weights(), second_pnl.weights());
    }
}
