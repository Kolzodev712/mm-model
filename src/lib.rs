//! Quote-evaluation baselines for one instrument and buy quotes.
//!
//! This is not an optimal quoting system. Two estimators can feed the same
//! deterministic selector:
//!
//! - historical aggregation in [`evaluation`]: fills / attempts and average
//!   filled P&L;
//! - a learned baseline in [`model`]: logistic fill probability and linear
//!   conditional P&L from decision-time features.
//!
//! Expected value is always `P(fill) * E[P&L | fill]`. `NoQuote` has expected
//! value 0. Neither estimator submits orders or applies a risk limit.
//!
//! # Counterfactuals
//!
//! A historical fill at one quote price does not prove whether a different
//! quote would have filled at the same moment. The dataset builder does not
//! invent that label from later prices. Simulated labels stay marked as
//! simulated and are not mixed into an observed training set.
//!
//! # Layout
//!
//! - [`domain`] — recorded attempts and mark-to-fair P&L
//! - [`evaluation`] — historical per-strategy estimates
//! - [`selection`] — highest expected value, or `NoQuote`
//! - [`decision`] — market state and candidate quote at decision time
//! - [`features`] — fixed feature row
//! - [`logistic`] / [`linear`] — reference models and offline trainers
//! - [`model`] — combine the two predictions into expected value
//! - [`trainset`] — decision-time inputs, outcomes, and provenance
//! - [`fair_value`] — midpoint and microprice reference prices from a book
//! - [`dataset`] — align books and quote outcomes; both reference prices are estimated
//! - [`marketdata`] — fixed-depth book log and the Binance payload parser

#![deny(unsafe_code)]

#[cfg(feature = "artifact")]
pub mod artifact;
pub mod candidates;
pub mod clock;
pub mod dataset;
pub mod decision;
pub mod demo;
pub mod domain;
pub mod evaluate;
pub mod evaluation;
pub mod fair_value;
pub mod features;
pub mod linear;
pub mod logistic;
pub mod marketdata;
pub mod model;
pub mod reconnect;
pub mod selection;
pub mod source;
pub mod stats;
pub mod trainset;
pub mod units;

pub use dataset::{build_training_examples, BuildError, DatasetConfig, QuoteEvent, TrainingSet};
pub use decision::{CandidateQuote, MarketInputError, MarketState};
pub use domain::{
    mark_to_fair_pnl, EmptyStrategyId, FillOutcome, HistoricalObservation, InvalidObservation,
    StrategyId,
};
pub use evaluation::{evaluate_strategies, StrategyEvaluation};
pub use fair_value::{FairValueError, FairValueMethod};
pub use features::{
    extract_features, FeatureVector, FEATURE_COUNT, FEATURE_NAMES, FEATURE_SCHEMA_V1,
};
pub use linear::{LinearRegression, PnlSample};
pub use logistic::{sigmoid, FillSample, LogisticRegression};
pub use model::{
    predict_quote, score_candidates, ConditionalPnlModel, FillProbabilityModel, PredictError,
    QuotePrediction, TrainConfig, TrainError,
};
pub use selection::{
    select_action, select_learned, select_quote, ActionScore, LearnedChoice, QuoteDecision,
    ScoredCandidate,
};
pub use trainset::{
    logistic_samples, pnl_samples, DecisionTimeInputs, ExampleError, FillFacts, LabelProvenance,
    LabelTiming, ObservedOutcome, TrainingExample,
};
pub use units::{CandidateId, DurationNanos, Fee, Pnl, Price, Probability, Quantity, UnixNanos};

#[cfg(test)]
pub(crate) mod test_fixture;
