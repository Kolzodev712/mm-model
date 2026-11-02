//! Logistic regression for `P(fill | features)`.
//!
//! This is a reference baseline, not a claim about the best fill model.
//!
//! ```text
//! z = bias + w · x
//! P(fill) = sigmoid(z)
//! ```
//!
//! Training minimizes mean binary cross-entropy with full-batch gradient
//! descent. There is no shuffling and no per-sample allocation. The gradient
//! buffer is a stack array reused each epoch.
//!
//! For label `y` in `{0, 1}` and prediction `p = sigmoid(z)`:
//!
//! ```text
//! loss = mean( -(y ln p + (1 - y) ln (1 - p)) )
//! d(loss) / d(bias) = mean(p - y)
//! d(loss) / d(w_j)  = mean((p - y) x_j)
//! ```
//!
//! Logarithms in the reported loss clamp `p` to `[1e-12, 1 - 1e-12]`.
//! The gradient uses the unclamped sigmoid.

use crate::domain::units::Probability;
use crate::model::feature::v1::FeatureVector;
use crate::model::prediction::{
    finite_weights, validate_train_config, FillProbabilityModel, InvalidWeights, PredictError,
    TrainConfig, TrainError,
};

/// One fill observation already turned into features.
///
/// `filled` must come from a submitted order or from a record explicitly marked
/// as simulated. This type does not invent a label for a quote that was never sent.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FillSample<const N: usize> {
    /// Decision-time features.
    pub features: FeatureVector<N>,
    /// `true` when that submitted quote filled.
    pub filled: bool,
}

/// Bias plus a fixed weight array.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogisticRegression<const N: usize> {
    bias: f64,
    weights: [f64; N],
}

impl<const N: usize> LogisticRegression<N> {
    /// Zero bias and zero weights. `sigmoid(0) = 0.5` for every row.
    #[must_use]
    pub fn zeros() -> Self {
        Self {
            bias: 0.0,
            weights: [0.0; N],
        }
    }

    /// Builds a model from finite parameters.
    pub fn from_weights(bias: f64, weights: [f64; N]) -> Result<Self, InvalidWeights> {
        finite_weights(bias, &weights)?;
        Ok(Self { bias, weights })
    }

    /// Bias term.
    #[must_use]
    pub fn bias(&self) -> f64 {
        self.bias
    }

    /// Weight array.
    #[must_use]
    pub fn weights(&self) -> &[f64; N] {
        &self.weights
    }

    /// Mean binary cross-entropy of the current parameters.
    pub fn mean_loss(&self, samples: &[FillSample<N>]) -> Result<f64, TrainError> {
        if samples.is_empty() {
            return Err(TrainError::EmptyDataset);
        }
        let mut total = 0.0;
        for sample in samples {
            let probability = self
                .predict_probability(&sample.features)
                .map_err(|_| TrainError::NonFinitePrediction)?
                .get();
            total += binary_cross_entropy(sample_label(sample.filled), probability);
        }
        Ok(total / samples.len() as f64)
    }

    /// Runs full-batch gradient descent and returns the mean loss afterward.
    ///
    /// Sample order is the update order. Identical samples and [`TrainConfig`]
    /// produce identical weights.
    pub fn train(
        &mut self,
        samples: &[FillSample<N>],
        config: TrainConfig,
    ) -> Result<f64, TrainError> {
        validate_train_config(config)?;
        if samples.is_empty() {
            return Err(TrainError::EmptyDataset);
        }

        let scale = 1.0 / samples.len() as f64;
        for _epoch in 0..config.epochs {
            let mut bias_grad = 0.0;
            let mut weight_grad = [0.0; N];
            for sample in samples {
                let probability = self
                    .predict_probability(&sample.features)
                    .map_err(|_| TrainError::NonFinitePrediction)?
                    .get();
                let residual = probability - sample_label(sample.filled);
                bias_grad += residual;
                for (gradient, value) in weight_grad.iter_mut().zip(sample.features.as_array()) {
                    *gradient += residual * value;
                }
            }
            self.bias -= config.learning_rate * bias_grad * scale;
            for (weight, gradient) in self.weights.iter_mut().zip(weight_grad) {
                *weight -= config.learning_rate * gradient * scale;
            }
        }
        self.mean_loss(samples)
    }
}

impl<const N: usize> LogisticRegression<N> {
    fn predict_probability(
        &self,
        features: &FeatureVector<N>,
    ) -> Result<Probability, PredictError> {
        sigmoid(self.bias + features.dot(&self.weights))
    }
}

impl<const N: usize> FillProbabilityModel<N> for LogisticRegression<N> {
    fn predict_fill_probability(
        &self,
        features: &FeatureVector<N>,
    ) -> Result<Probability, PredictError> {
        self.predict_probability(features)
    }
}

fn sample_label(filled: bool) -> f64 {
    if filled {
        1.0
    } else {
        0.0
    }
}

/// Numerically stable logistic sigmoid.
///
/// A non-finite linear score is an error. It is not replaced with `0.5`.
pub fn sigmoid(z: f64) -> Result<Probability, PredictError> {
    if !z.is_finite() {
        return Err(PredictError::InvalidProbability);
    }
    let probability = if z >= 0.0 {
        let exp_neg = (-z).exp();
        1.0 / (1.0 + exp_neg)
    } else {
        let exp_pos = z.exp();
        exp_pos / (1.0 + exp_pos)
    };
    Probability::new(probability).map_err(|_| PredictError::InvalidProbability)
}

fn binary_cross_entropy(label: f64, probability: f64) -> f64 {
    const EPS: f64 = 1e-12;
    let clipped = probability.clamp(EPS, 1.0 - EPS);
    -(label * clipped.ln() + (1.0 - label) * (1.0 - clipped).ln())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::feature::v1::FeatureVector;

    fn row(value: f64) -> FeatureVector<1> {
        FeatureVector::new([value]).unwrap()
    }

    fn separable() -> Vec<FillSample<1>> {
        vec![
            FillSample {
                features: row(-2.0),
                filled: false,
            },
            FillSample {
                features: row(-1.0),
                filled: false,
            },
            FillSample {
                features: row(1.0),
                filled: true,
            },
            FillSample {
                features: row(2.0),
                filled: true,
            },
        ]
    }

    #[test]
    fn zero_weights_predict_one_half() {
        let model = LogisticRegression::<4>::zeros();
        let features = FeatureVector::new([3.0, -1.0, 0.5, 8.0]).unwrap();
        assert_eq!(
            model.predict_fill_probability(&features).unwrap().get(),
            0.5
        );
    }

    #[test]
    fn probabilities_stay_inside_the_unit_interval() {
        let model = LogisticRegression::from_weights(0.0, [50.0]).unwrap();
        for value in [-1.0e6, -10.0, 0.0, 10.0, 1.0e6] {
            let probability = model.predict_fill_probability(&row(value)).unwrap().get();
            assert!((0.0..=1.0).contains(&probability));
        }
        assert!(sigmoid(f64::INFINITY).is_err());
        assert!(sigmoid(f64::NEG_INFINITY).is_err());
        assert!(sigmoid(f64::NAN).is_err());
    }

    #[test]
    fn separable_training_reduces_loss() {
        let samples = separable();
        let mut model = LogisticRegression::<1>::zeros();
        let before = model.mean_loss(&samples).unwrap();
        let config = TrainConfig {
            learning_rate: 0.2,
            epochs: 400,
        };
        let after = model.train(&samples, config).unwrap();
        assert!(after < before);
        assert!(model.weights()[0] > 0.0);
    }

    #[test]
    fn training_is_deterministic() {
        let samples = separable();
        let config = TrainConfig {
            learning_rate: 0.2,
            epochs: 50,
        };
        let mut first = LogisticRegression::<1>::zeros();
        let mut second = LogisticRegression::<1>::zeros();
        first.train(&samples, config).unwrap();
        second.train(&samples, config).unwrap();
        assert_eq!(first.bias(), second.bias());
        assert_eq!(first.weights(), second.weights());
    }
}
