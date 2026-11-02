//! Labeled examples for offline training.
//!
//! Decision-time inputs and observed outcomes are different types. A fill label
//! is attached only when the caller supplies one. Nothing in this module looks
//! at a later price and declares that a quote which was never sent would have
//! filled.
//!
//! [`LabelProvenance::Observed`] means an order was submitted and the fill or
//! miss was recorded. [`LabelProvenance::Simulated`] means a simulator wrote
//! the label and marked it as simulated. Converting a slice to model samples
//! requires one provenance and rejects a mixture.

use std::fmt;

use crate::domain::market::{CandidateQuote, MarketState};
use crate::model::fair_value::FairValueMethod;
use crate::model::feature::v1::{extract_features, FEATURE_COUNT};
use crate::model::fill::FillSample;
use crate::model::pnl::PnlSample;

/// Where a fill label came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(
    feature = "artifact",
    derive(serde::Serialize, serde::Deserialize),
    serde(rename_all = "snake_case")
)]
pub enum LabelProvenance {
    /// The quote was submitted and the outcome was observed.
    Observed,
    /// A simulator produced the outcome and marked it as simulated.
    Simulated,
}

/// Market and candidate quote visible before the decision.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecisionTimeInputs {
    market: MarketState,
    quote: CandidateQuote,
}

impl DecisionTimeInputs {
    /// Stores a decision-time market and quote.
    #[must_use]
    pub fn new(market: MarketState, quote: CandidateQuote) -> Self {
        Self { market, quote }
    }

    #[must_use]
    pub fn market(&self) -> MarketState {
        self.market
    }

    #[must_use]
    pub fn quote(&self) -> CandidateQuote {
        self.quote
    }
}

/// Clock fields for one derived label.
///
/// `horizon_offset_ns` is `horizon_snapshot_ns - (decision_ns + requested_horizon_ns)`.
/// A non-negative offset means the horizon book is at or after the requested time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LabelTiming {
    /// Quote decision time `t`, unix nanoseconds.
    pub decision_ns: i64,
    /// Local receive time of the book that produced `FV_t`.
    pub decision_snapshot_ns: i64,
    /// Requested evaluation delay `H`, nanoseconds.
    pub requested_horizon_ns: i64,
    /// Local receive time of the book that produced `FV_(t+H)`.
    pub horizon_snapshot_ns: i64,
}

impl LabelTiming {
    /// How much later the horizon snapshot is than `t + H`.
    ///
    /// Overflow or a horizon before the target is an error, not a saturated duration.
    pub fn horizon_offset_ns(self) -> Result<i64, ExampleError> {
        let target = self
            .decision_ns
            .checked_add(self.requested_horizon_ns)
            .ok_or(ExampleError::InvalidTiming)?;
        self.horizon_snapshot_ns
            .checked_sub(target)
            .filter(|offset| *offset >= 0)
            .ok_or(ExampleError::InvalidTiming)
    }
}

/// What happened after a quote was actually submitted or explicitly simulated.
///
/// A filled label's reference price is the horizon estimate. It is not a
/// decision-time feature and it is not supplied on [`FillFacts`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ObservedOutcome {
    /// The quote did not fill. There is no fill P&L.
    Unfilled,
    /// The quote filled.
    Filled {
        /// Mark-to-reference P&L using the horizon estimate.
        pnl: f64,
        /// `FV_(t+H)` from the same estimator as the decision-time state.
        horizon_fair_value: f64,
    },
}

/// One training row: decision-time inputs, a later outcome, provenance, and
/// the fair-value definition used to build both reference prices.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrainingExample {
    inputs: DecisionTimeInputs,
    outcome: ObservedOutcome,
    provenance: LabelProvenance,
    method: FairValueMethod,
    timing: LabelTiming,
}

impl TrainingExample {
    /// Builds a row.
    ///
    /// `method` must be the estimator stored on the market state. A filled
    /// outcome must have a finite P&L and a finite positive horizon reference price.
    pub fn new(
        inputs: DecisionTimeInputs,
        outcome: ObservedOutcome,
        provenance: LabelProvenance,
        method: FairValueMethod,
        timing: LabelTiming,
    ) -> Result<Self, ExampleError> {
        if inputs.market().fair_value_method() != method {
            return Err(ExampleError::FairValueMethodMismatch);
        }
        if let ObservedOutcome::Filled {
            pnl,
            horizon_fair_value,
        } = outcome
        {
            if !pnl.is_finite() || !horizon_fair_value.is_finite() || horizon_fair_value <= 0.0 {
                return Err(ExampleError::NonFinitePnl);
            }
        }
        Ok(Self {
            inputs,
            outcome,
            provenance,
            method,
            timing,
        })
    }

    #[must_use]
    pub fn inputs(&self) -> DecisionTimeInputs {
        self.inputs
    }

    #[must_use]
    pub fn outcome(&self) -> ObservedOutcome {
        self.outcome
    }

    #[must_use]
    pub fn provenance(&self) -> LabelProvenance {
        self.provenance
    }

    /// Estimator that produced this row's decision-time and horizon prices.
    #[must_use]
    pub fn fair_value_method(&self) -> FairValueMethod {
        self.method
    }

    #[must_use]
    pub fn timing(&self) -> LabelTiming {
        self.timing
    }
}

/// Execution facts for a submitted or simulated buy.
///
/// There is no horizon reference price here. The dataset builder derives that
/// from a later book snapshot with the configured estimator.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FillFacts {
    /// The quote did not fill.
    Unfilled,
    /// The quote filled in full at `execution_price`.
    Filled {
        /// Price of the fill.
        execution_price: f64,
        /// Non-negative fee charged on the fill.
        fees: f64,
    },
}

/// Turns examples of one provenance into logistic-regression samples.
///
/// A slice that mixes [`LabelProvenance::Observed`] and
/// [`LabelProvenance::Simulated`] is rejected. Features are taken from
/// [`DecisionTimeInputs`] only.
pub fn logistic_samples(
    examples: &[TrainingExample],
    provenance: LabelProvenance,
    method: FairValueMethod,
) -> Result<Vec<FillSample<FEATURE_COUNT>>, ExampleError> {
    let mut samples = Vec::with_capacity(examples.len());
    for example in examples {
        if example.provenance != provenance {
            return Err(ExampleError::MixedProvenance);
        }
        if example.method != method {
            return Err(ExampleError::MixedFairValueMethod);
        }
        samples.push(FillSample {
            features: extract_features(&example.inputs.market, &example.inputs.quote)
                .map_err(|_| ExampleError::InvalidFeatures)?,
            filled: matches!(example.outcome, ObservedOutcome::Filled { .. }),
        });
    }
    Ok(samples)
}

/// Turns filled examples of one provenance into linear-regression samples.
///
/// Unfilled rows are omitted. They are not inserted with P&L of zero.
/// Mixed provenance is rejected before any sample is returned.
pub fn pnl_samples(
    examples: &[TrainingExample],
    provenance: LabelProvenance,
    method: FairValueMethod,
) -> Result<Vec<PnlSample<FEATURE_COUNT>>, ExampleError> {
    let mut samples = Vec::new();
    for example in examples {
        if example.provenance != provenance {
            return Err(ExampleError::MixedProvenance);
        }
        if example.method != method {
            return Err(ExampleError::MixedFairValueMethod);
        }
        if let ObservedOutcome::Filled { pnl, .. } = example.outcome {
            samples.push(PnlSample {
                features: extract_features(&example.inputs.market, &example.inputs.quote)
                    .map_err(|_| ExampleError::InvalidFeatures)?,
                pnl,
            });
        }
    }
    Ok(samples)
}

/// A training row that cannot be used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExampleError {
    /// Filled P&L or the horizon reference price was not usable.
    NonFinitePnl,
    /// Fill facts were not finite, or fees were negative.
    InvalidFill,
    /// The slice contained more than the requested provenance.
    MixedProvenance,
    /// The slice contained more than the requested fair-value method.
    MixedFairValueMethod,
    /// The declared method does not match the market state's estimator.
    FairValueMethodMismatch,
    /// Decision-time inputs did not produce a feature row.
    InvalidFeatures,
    /// A timestamp combination overflowed or went backwards.
    InvalidTiming,
}

impl fmt::Display for ExampleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::NonFinitePnl => {
                "filled pnl and horizon reference price must be finite and positive"
            }
            Self::InvalidFill => "fill facts must be finite, with non-negative fees",
            Self::MixedProvenance => "observed and simulated labels cannot be trained together",
            Self::MixedFairValueMethod => {
                "rows built with different fair-value estimators cannot be trained together"
            }
            Self::FairValueMethodMismatch => {
                "training row estimator does not match its market state"
            }
            Self::InvalidFeatures => "training row did not produce finite features",
            Self::InvalidTiming => "label timing overflowed or is out of order",
        };
        f.write_str(message)
    }
}

impl std::error::Error for ExampleError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::outcome::mark_to_fair_pnl;
    use crate::model::fair_value::FairValueMethod;

    fn inputs(method: FairValueMethod) -> DecisionTimeInputs {
        DecisionTimeInputs::new(
            method.derive_top(99.9, 2.0, 100.1, 1.0).unwrap(),
            CandidateQuote::new(99.8, 2.0).unwrap(),
        )
    }

    fn timing() -> LabelTiming {
        LabelTiming {
            decision_ns: 1_000,
            decision_snapshot_ns: 1_000,
            requested_horizon_ns: 500,
            horizon_snapshot_ns: 1_600,
        }
    }

    fn example(
        outcome: ObservedOutcome,
        provenance: LabelProvenance,
        method: FairValueMethod,
    ) -> TrainingExample {
        TrainingExample::new(inputs(method), outcome, provenance, method, timing()).unwrap()
    }

    #[test]
    fn outcome_does_not_change_decision_time_features() {
        let method = FairValueMethod::Microprice;
        let missed = example(ObservedOutcome::Unfilled, LabelProvenance::Observed, method);
        let filled = example(
            ObservedOutcome::Filled {
                pnl: 3.5,
                horizon_fair_value: 101.0,
            },
            LabelProvenance::Observed,
            method,
        );
        let missed_rows = logistic_samples(&[missed], LabelProvenance::Observed, method).unwrap();
        let filled_rows = logistic_samples(&[filled], LabelProvenance::Observed, method).unwrap();
        assert_eq!(missed_rows[0].features, filled_rows[0].features);
        assert!(!missed_rows[0].filled);
        assert!(filled_rows[0].filled);
    }

    #[test]
    fn mixed_provenance_is_rejected() {
        let method = FairValueMethod::Microprice;
        let rows = [
            example(ObservedOutcome::Unfilled, LabelProvenance::Observed, method),
            example(
                ObservedOutcome::Filled {
                    pnl: 1.0,
                    horizon_fair_value: 101.0,
                },
                LabelProvenance::Simulated,
                method,
            ),
        ];
        assert_eq!(
            logistic_samples(&rows, LabelProvenance::Observed, method),
            Err(ExampleError::MixedProvenance)
        );
        assert_eq!(
            pnl_samples(&rows, LabelProvenance::Simulated, method),
            Err(ExampleError::MixedProvenance)
        );
    }

    #[test]
    fn mixed_fair_value_methods_are_rejected() {
        let rows = [
            example(
                ObservedOutcome::Filled {
                    pnl: 1.0,
                    horizon_fair_value: 101.0,
                },
                LabelProvenance::Observed,
                FairValueMethod::Microprice,
            ),
            example(
                ObservedOutcome::Filled {
                    pnl: 1.0,
                    horizon_fair_value: 101.0,
                },
                LabelProvenance::Observed,
                FairValueMethod::Midpoint,
            ),
        ];
        assert_eq!(
            pnl_samples(
                &rows,
                LabelProvenance::Observed,
                FairValueMethod::Microprice
            ),
            Err(ExampleError::MixedFairValueMethod)
        );
    }

    #[test]
    fn conditional_pnl_samples_skip_unfilled_rows() {
        let method = FairValueMethod::Microprice;
        let rows = [
            example(
                ObservedOutcome::Unfilled,
                LabelProvenance::Simulated,
                method,
            ),
            example(
                ObservedOutcome::Filled {
                    pnl: 1.25,
                    horizon_fair_value: 101.0,
                },
                LabelProvenance::Simulated,
                method,
            ),
            example(
                ObservedOutcome::Filled {
                    pnl: -0.5,
                    horizon_fair_value: 99.0,
                },
                LabelProvenance::Simulated,
                method,
            ),
        ];
        let samples = pnl_samples(&rows, LabelProvenance::Simulated, method).unwrap();
        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0].pnl, 1.25);
        assert_eq!(samples[1].pnl, -0.5);
    }

    #[test]
    fn legacy_pnl_formula_stays_available_for_the_historical_baseline() {
        assert_eq!(mark_to_fair_pnl(101.0, 99.0, 2.0, 0.5), 3.5);
    }
}
