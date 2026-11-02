//! Versioned model file. Loading is a cold path and may allocate.
//!
//! The live book parser does not use this module.

use std::fmt;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::model::fair_value::FairValueMethod;
use crate::model::feature::v1::{FEATURE_COUNT, FEATURE_SCHEMA_V1};
use crate::model::fill::LogisticRegression;
use crate::model::pnl::LinearRegression;
use crate::model::prediction::InvalidWeights;

pub const ARTIFACT_VERSION: u32 = 1;

/// Parameters and the metadata required to refuse an incompatible run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelArtifact {
    pub version: u32,
    pub feature_schema: u32,
    pub fair_value_method: String,
    pub training_horizon_ns: i64,
    pub fill_bias: f64,
    pub fill_weights: Vec<f64>,
    pub pnl_bias: f64,
    pub pnl_weights: Vec<f64>,
    #[serde(default)]
    pub example_count: u64,
    #[serde(default)]
    pub fill_count: u64,
    #[serde(default)]
    pub label_provenance: String,
    #[serde(default)]
    pub window_from_ns: i64,
    #[serde(default)]
    pub window_to_ns: i64,
    #[serde(default)]
    pub dataset_id: String,
}

impl ModelArtifact {
    pub fn from_models(
        method: FairValueMethod,
        training_horizon_ns: i64,
        fill: &LogisticRegression<FEATURE_COUNT>,
        pnl: &LinearRegression<FEATURE_COUNT>,
    ) -> Self {
        Self {
            version: ARTIFACT_VERSION,
            feature_schema: FEATURE_SCHEMA_V1,
            fair_value_method: method.name().to_string(),
            training_horizon_ns,
            fill_bias: fill.bias(),
            fill_weights: fill.weights().to_vec(),
            pnl_bias: pnl.bias(),
            pnl_weights: pnl.weights().to_vec(),
            example_count: 0,
            fill_count: 0,
            label_provenance: String::new(),
            window_from_ns: 0,
            window_to_ns: 0,
            dataset_id: String::new(),
        }
    }

    pub fn validate_for_run(&self, method: FairValueMethod) -> Result<(), ArtifactError> {
        if self.version != ARTIFACT_VERSION {
            return Err(ArtifactError::Version);
        }
        if self.feature_schema != FEATURE_SCHEMA_V1 {
            return Err(ArtifactError::FeatureSchema);
        }
        if self.fair_value_method != method.name() {
            return Err(ArtifactError::FairValueMethod);
        }
        if self.training_horizon_ns <= 0 {
            return Err(ArtifactError::Horizon);
        }
        if self.fill_weights.len() != FEATURE_COUNT || self.pnl_weights.len() != FEATURE_COUNT {
            return Err(ArtifactError::Weights);
        }
        Ok(())
    }

    pub fn validate_horizon(&self, horizon_ns: i64) -> Result<(), ArtifactError> {
        if self.training_horizon_ns != horizon_ns {
            return Err(ArtifactError::Horizon);
        }
        Ok(())
    }

    pub fn models(
        &self,
    ) -> Result<
        (
            LogisticRegression<FEATURE_COUNT>,
            LinearRegression<FEATURE_COUNT>,
        ),
        ArtifactError,
    > {
        self.validate_for_run(parse_method(&self.fair_value_method)?)?;
        let mut fill_weights = [0.0; FEATURE_COUNT];
        let mut pnl_weights = [0.0; FEATURE_COUNT];
        fill_weights.copy_from_slice(&self.fill_weights);
        pnl_weights.copy_from_slice(&self.pnl_weights);
        let fill = LogisticRegression::from_weights(self.fill_bias, fill_weights)
            .map_err(|_| ArtifactError::Weights)?;
        let pnl = LinearRegression::from_weights(self.pnl_bias, pnl_weights)
            .map_err(|_| ArtifactError::Weights)?;
        Ok((fill, pnl))
    }

    pub fn write_json(&self, path: &Path) -> Result<(), ArtifactError> {
        let text = serde_json::to_string_pretty(self).map_err(|_| ArtifactError::Json)?;
        fs::write(path, text).map_err(|_| ArtifactError::Io)
    }

    pub fn read_json(path: &Path) -> Result<Self, ArtifactError> {
        let text = fs::read_to_string(path).map_err(|_| ArtifactError::Io)?;
        serde_json::from_str(&text).map_err(|_| ArtifactError::Json)
    }
}

fn parse_method(name: &str) -> Result<FairValueMethod, ArtifactError> {
    match name {
        "microprice" => Ok(FairValueMethod::Microprice),
        "midpoint" => Ok(FairValueMethod::Midpoint),
        _ => Err(ArtifactError::FairValueMethod),
    }
}

impl From<InvalidWeights> for ArtifactError {
    fn from(_: InvalidWeights) -> Self {
        Self::Weights
    }
}

/// A model file cannot be used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArtifactError {
    Version,
    FeatureSchema,
    FairValueMethod,
    Horizon,
    Weights,
    Json,
    Io,
}

impl fmt::Display for ArtifactError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::Version => "unsupported model artifact version",
            Self::FeatureSchema => "model feature schema does not match this build",
            Self::FairValueMethod => "model fair-value method does not match the run",
            Self::Horizon => "model training horizon does not match",
            Self::Weights => "model weights are missing, the wrong length, or not finite",
            Self::Json => "model artifact is not valid JSON",
            Self::Io => "model artifact could not be read or written",
        };
        f.write_str(message)
    }
}

impl std::error::Error for ArtifactError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_and_method_mismatches_fail() {
        let (fill, pnl) = crate::data::synthetic::train_demo_models().unwrap();
        let mut artifact =
            ModelArtifact::from_models(FairValueMethod::Microprice, 1_000, &fill, &pnl);
        artifact
            .validate_for_run(FairValueMethod::Microprice)
            .unwrap();
        assert_eq!(
            artifact.validate_for_run(FairValueMethod::Midpoint),
            Err(ArtifactError::FairValueMethod)
        );
        artifact.feature_schema = 99;
        assert_eq!(
            artifact.validate_for_run(FairValueMethod::Microprice),
            Err(ArtifactError::FeatureSchema)
        );
        artifact.feature_schema = FEATURE_SCHEMA_V1;
        assert_eq!(
            artifact.validate_horizon(2_000),
            Err(ArtifactError::Horizon)
        );
        artifact.validate_horizon(1_000).unwrap();
    }
}
