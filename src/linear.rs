//! Linear regression for `E[P&L | fill, features]`.
//!
//! This is a reference baseline, not a claim about the best P&L model.
//!
//! ```text
//! predicted_pnl = bias + w · x
//! ```
//!
//! Training minimizes mean squared error with full-batch gradient descent:
//!
//! ```text
//! loss = mean( (prediction - y)^2 )
//! d(loss) / d(bias) = mean( 2 (prediction - y) )
//! d(loss) / d(w_j)  = mean( 2 (prediction - y) x_j )
//! ```
//!
//! Every sample is a filled outcome. Unfilled attempts are not given a P&L of
//! zero and passed in here. The trainer allocates no heap memory: the gradient
//! is a stack array reused for every epoch.

use crate::features::FeatureVector;
use crate::model::{
    finite_weights, validate_train_config, ConditionalPnlModel, InvalidWeights, PredictError,
    TrainConfig, TrainError,
};
use crate::units::Pnl;

/// P&L of one fill, already paired with decision-time features.
///
/// `pnl` uses the mark-to-fair definition of a filled buy. Do not construct
/// this from an unfilled attempt.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PnlSample<const N: usize> {
    /// Decision-time features.
    pub features: FeatureVector<N>,
    /// Realized mark-to-fair P&L of that fill.
    pub pnl: f64,
}

/// Bias plus a fixed weight array.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearRegression<const N: usize> {
    bias: f64,
    weights: [f64; N],
}

impl<const N: usize> LinearRegression<N> {
    /// Zero bias and zero weights.
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

    /// Prediction `bias + w · x`. Non-finite results are errors.
    pub fn predict(&self, features: &FeatureVector<N>) -> Result<Pnl, PredictError> {
        let value = self.bias + features.dot(&self.weights);
        Pnl::new(value).map_err(|_| PredictError::NonFiniteConditionalPnl)
    }

    /// Mean squared error of the current parameters.
    pub fn mean_loss(&self, samples: &[PnlSample<N>]) -> Result<f64, TrainError> {
        if samples.is_empty() {
            return Err(TrainError::EmptyDataset);
        }
        let mut total = 0.0;
        for sample in samples {
            let error = self
                .predict(&sample.features)
                .map_err(|_| TrainError::NonFinitePrediction)?
                .get()
                - sample.pnl;
            total += error * error;
        }
        Ok(total / samples.len() as f64)
    }

    /// Runs full-batch gradient descent and returns the mean squared error afterward.
    ///
    /// Sample order is the update order. Identical samples and [`TrainConfig`]
    /// produce identical weights.
    pub fn train(
        &mut self,
        samples: &[PnlSample<N>],
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
                let predicted = self
                    .predict(&sample.features)
                    .map_err(|_| TrainError::NonFinitePrediction)?
                    .get();
                let residual = 2.0 * (predicted - sample.pnl);
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

impl<const N: usize> ConditionalPnlModel<N> for LinearRegression<N> {
    fn predict_pnl_if_filled(&self, features: &FeatureVector<N>) -> Result<Pnl, PredictError> {
        self.predict(features)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::FeatureVector;

    fn row(value: f64) -> FeatureVector<1> {
        FeatureVector::new([value]).unwrap()
    }

    fn line() -> Vec<PnlSample<1>> {
        // y = 1 + 2x
        [-1.0, 0.0, 1.0, 2.0]
            .into_iter()
            .map(|x| PnlSample {
                features: row(x),
                pnl: 1.0 + 2.0 * x,
            })
            .collect()
    }

    #[test]
    fn explicit_weights_match_the_linear_formula() {
        let model = LinearRegression::from_weights(1.5, [2.0, 0.0, -1.0, 4.0]).unwrap();
        let features = FeatureVector::new([3.0, 10.0, 4.0, 0.5]).unwrap();
        let expected = 1.5 + 2.0 * 3.0 - 4.0 + 4.0 * 0.5;
        assert_eq!(model.predict(&features).unwrap().get(), expected);
    }

    #[test]
    fn fitting_a_line_reduces_squared_error() {
        let samples = line();
        let mut model = LinearRegression::<1>::zeros();
        let before = model.mean_loss(&samples).unwrap();
        let config = TrainConfig {
            learning_rate: 0.05,
            epochs: 800,
        };
        let after = model.train(&samples, config).unwrap();
        assert!(after < before);
    }

    #[test]
    fn training_is_deterministic() {
        let samples = line();
        let config = TrainConfig {
            learning_rate: 0.05,
            epochs: 40,
        };
        let mut first = LinearRegression::<1>::zeros();
        let mut second = LinearRegression::<1>::zeros();
        first.train(&samples, config).unwrap();
        second.train(&samples, config).unwrap();
        assert_eq!(first.bias(), second.bias());
        assert_eq!(first.weights(), second.weights());
    }
}
