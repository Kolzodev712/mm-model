//! Versioned model file. Loading is a cold path and may allocate.
//!
//! The live book parser does not use this module.

use std::fmt;
use std::fs;
use std::io::Write;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::data::training::example::LabelProvenance;
use crate::model::fair_value::FairValueMethod;
use crate::model::feature::v1::{FEATURE_COUNT, FEATURE_SCHEMA_VERSION};
use crate::model::fill::LogisticRegression;
use crate::model::pnl::LinearRegression;
use crate::model::prediction::InvalidWeights;

pub const ARTIFACT_VERSION: u32 = 3;

/// Holdout comparison that was good enough to release for shadow experiments.
///
/// This is not a claim of statistical significance or trading profit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactState {
    PublishedForShadow,
}

/// Structured holdout comparison. Baselines are computed from the training slice only.
///
/// `ev_*_error` is mean absolute error of predicted attempt P&L on quotes that
/// were actually submitted. It does not measure whether a different quote would
/// have been better.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ValidationReport {
    /// Rows whose decision time was before the holdout, before purging.
    pub rows_before_purge: u64,
    /// Training rows removed because `horizon_snapshot_ns` reached the holdout.
    pub purged_training_rows: u64,
    /// `rows_before_purge - purged_training_rows`. This is the fitted sample.
    pub train_examples: u64,
    pub holdout_examples: u64,
    /// Decision time of the first holdout row. Training labels must finish before this.
    pub holdout_start_ns: i64,
    pub train_fills: u64,
    pub train_misses: u64,
    pub holdout_fills: u64,
    pub holdout_misses: u64,
    pub fill_model_loss: f64,
    pub fill_baseline_loss: f64,
    pub pnl_model_loss: f64,
    pub pnl_baseline_loss: f64,
    pub ev_model_error: f64,
    pub ev_baseline_error: f64,
}

/// Parameters and the metadata required to refuse an incompatible run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelArtifact {
    pub version: u32,
    pub feature_schema: u32,
    pub fair_value_method: FairValueMethod,
    pub training_horizon_ns: i64,
    pub fill_bias: f64,
    pub fill_weights: Vec<f64>,
    pub pnl_bias: f64,
    pub pnl_weights: Vec<f64>,
    pub label_provenance: LabelProvenance,
    pub state: ArtifactState,
    pub window_from_ns: i64,
    pub window_to_ns: i64,
    pub dataset_id: String,
    pub report: ValidationReport,
}

/// Provenance stored with a released artifact.
pub struct ArtifactMetadata {
    pub provenance: LabelProvenance,
    pub window_from_ns: i64,
    pub window_to_ns: i64,
    pub dataset_id: String,
}

impl ModelArtifact {
    /// Builds a complete shadow release, or refuses incomplete evidence.
    pub fn published_for_shadow(
        method: FairValueMethod,
        training_horizon_ns: i64,
        fill: &LogisticRegression<FEATURE_COUNT>,
        pnl: &LinearRegression<FEATURE_COUNT>,
        metadata: ArtifactMetadata,
        report: ValidationReport,
    ) -> Result<Self, ArtifactError> {
        let ArtifactMetadata {
            provenance,
            window_from_ns,
            window_to_ns,
            dataset_id,
        } = metadata;
        if training_horizon_ns <= 0 || window_from_ns >= window_to_ns || dataset_id.is_empty() {
            return Err(ArtifactError::Evidence);
        }
        if provenance != LabelProvenance::Observed {
            return Err(ArtifactError::Provenance);
        }
        if !report_is_acceptable(&report) {
            return Err(ArtifactError::Evidence);
        }
        let artifact = Self {
            version: ARTIFACT_VERSION,
            feature_schema: FEATURE_SCHEMA_VERSION,
            fair_value_method: method,
            training_horizon_ns,
            fill_bias: fill.bias(),
            fill_weights: fill.weights().to_vec(),
            pnl_bias: pnl.bias(),
            pnl_weights: pnl.weights().to_vec(),
            label_provenance: provenance,
            state: ArtifactState::PublishedForShadow,
            window_from_ns,
            window_to_ns,
            dataset_id,
            report,
        };
        artifact.validate_for_run(method)?;
        Ok(artifact)
    }

    /// Refuses a file that is not an observed model released for shadow experiments.
    pub fn validate_for_shadow(
        &self,
        method: FairValueMethod,
        horizon_ns: i64,
    ) -> Result<(), ArtifactError> {
        self.validate_for_run(method)?;
        self.validate_horizon(horizon_ns)?;
        if self.state != ArtifactState::PublishedForShadow {
            return Err(ArtifactError::State);
        }
        if self.label_provenance != LabelProvenance::Observed {
            return Err(ArtifactError::Provenance);
        }
        if !report_is_acceptable(&self.report) {
            return Err(ArtifactError::Evidence);
        }
        Ok(())
    }

    pub fn validate_for_run(&self, method: FairValueMethod) -> Result<(), ArtifactError> {
        if self.version != ARTIFACT_VERSION {
            return Err(ArtifactError::Version);
        }
        if self.feature_schema != FEATURE_SCHEMA_VERSION {
            return Err(ArtifactError::FeatureSchema);
        }
        if self.fair_value_method != method {
            return Err(ArtifactError::FairValueMethod);
        }
        if self.training_horizon_ns <= 0 {
            return Err(ArtifactError::Horizon);
        }
        if self.fill_weights.len() != FEATURE_COUNT || self.pnl_weights.len() != FEATURE_COUNT {
            return Err(ArtifactError::Weights);
        }
        if !self.fill_bias.is_finite()
            || !self.pnl_bias.is_finite()
            || self.fill_weights.iter().any(|weight| !weight.is_finite())
            || self.pnl_weights.iter().any(|weight| !weight.is_finite())
        {
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
        self.validate_for_run(self.fair_value_method)?;
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
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).map_err(|_| ArtifactError::Io)?;
            }
        }
        let tmp = path.with_extension("json.tmp");
        let text = serde_json::to_string_pretty(self).map_err(|_| ArtifactError::Json)?;
        {
            let mut file = fs::File::create(&tmp).map_err(|_| ArtifactError::Io)?;
            file.write_all(text.as_bytes())
                .map_err(|_| ArtifactError::Io)?;
            file.sync_all().map_err(|_| ArtifactError::Io)?;
        }
        fs::rename(&tmp, path).map_err(|_| ArtifactError::Io)
    }

    pub fn read_json(path: &Path) -> Result<Self, ArtifactError> {
        let meta = fs::metadata(path).map_err(|_| ArtifactError::Io)?;
        if meta.len() > 1_048_576 {
            return Err(ArtifactError::Io);
        }
        let text = fs::read_to_string(path).map_err(|_| ArtifactError::Io)?;
        let artifact: Self = serde_json::from_str(&text).map_err(|_| ArtifactError::Json)?;
        if artifact.version != ARTIFACT_VERSION {
            return Err(ArtifactError::Version);
        }
        Ok(artifact)
    }
}

/// Fill and conditional P&L may tie their constant baselines. Combined EV must be strictly better.
pub fn losses_accept(
    fill_model: f64,
    fill_baseline: f64,
    pnl_model: f64,
    pnl_baseline: f64,
    ev_model: f64,
    ev_baseline: f64,
) -> bool {
    fill_model <= fill_baseline && pnl_model <= pnl_baseline && ev_model < ev_baseline
}

fn report_is_acceptable(report: &ValidationReport) -> bool {
    let losses = [
        report.fill_model_loss,
        report.fill_baseline_loss,
        report.pnl_model_loss,
        report.pnl_baseline_loss,
        report.ev_model_error,
        report.ev_baseline_error,
    ];
    report.rows_before_purge >= report.purged_training_rows
        && report.train_examples == report.rows_before_purge - report.purged_training_rows
        && report.train_examples > 0
        && report.holdout_examples > 0
        && report.train_fills > 0
        && report.train_misses > 0
        && report.holdout_fills > 0
        && report.holdout_misses > 0
        && losses.iter().all(|value| value.is_finite())
        && losses_accept(
            report.fill_model_loss,
            report.fill_baseline_loss,
            report.pnl_model_loss,
            report.pnl_baseline_loss,
            report.ev_model_error,
            report.ev_baseline_error,
        )
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
    State,
    Provenance,
    Evidence,
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
            Self::State => "model artifact is not released for shadow experiments",
            Self::Provenance => "shadow loading requires observed fill labels",
            Self::Evidence => "model artifact does not record enough training evidence",
        };
        f.write_str(message)
    }
}

impl std::error::Error for ArtifactError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn passing_report() -> ValidationReport {
        ValidationReport {
            rows_before_purge: 16,
            purged_training_rows: 0,
            train_examples: 16,
            holdout_examples: 4,
            holdout_start_ns: 1_000,
            train_fills: 4,
            train_misses: 4,
            holdout_fills: 2,
            holdout_misses: 2,
            fill_model_loss: 0.4,
            fill_baseline_loss: 0.6,
            pnl_model_loss: 1.0,
            pnl_baseline_loss: 1.5,
            ev_model_error: 0.2,
            ev_baseline_error: 0.3,
        }
    }

    #[test]
    fn schema_method_and_old_version_fail() {
        let (fill, pnl) = crate::data::synthetic::train_demo_models().unwrap();
        let artifact = ModelArtifact::published_for_shadow(
            FairValueMethod::Microprice,
            1_000,
            &fill,
            &pnl,
            ArtifactMetadata {
                provenance: LabelProvenance::Observed,
                window_from_ns: 0,
                window_to_ns: 10,
                dataset_id: "dataset".into(),
            },
            passing_report(),
        )
        .unwrap();
        artifact
            .validate_for_shadow(FairValueMethod::Microprice, 1_000)
            .unwrap();
        assert_eq!(
            artifact.validate_for_run(FairValueMethod::Midpoint),
            Err(ArtifactError::FairValueMethod)
        );
        assert_eq!(
            artifact.validate_horizon(2_000),
            Err(ArtifactError::Horizon)
        );
        assert!(ModelArtifact::published_for_shadow(
            FairValueMethod::Microprice,
            1_000,
            &fill,
            &pnl,
            ArtifactMetadata {
                provenance: LabelProvenance::Simulated,
                window_from_ns: 0,
                window_to_ns: 10,
                dataset_id: "dataset".into(),
            },
            passing_report(),
        )
        .is_err());
        let path = std::env::temp_dir().join(format!(
            "mm-artifact-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&path, r#"{"version":1}"#).unwrap();
        assert_eq!(
            ModelArtifact::read_json(&path).unwrap_err(),
            ArtifactError::Json
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_tie_with_the_constant_ev_baseline_is_not_published() {
        let (fill, pnl) = crate::data::synthetic::train_demo_models().unwrap();
        let mut report = passing_report();
        report.ev_model_error = report.ev_baseline_error;
        assert!(!losses_accept(
            report.fill_model_loss,
            report.fill_baseline_loss,
            report.pnl_model_loss,
            report.pnl_baseline_loss,
            report.ev_model_error,
            report.ev_baseline_error,
        ));
        assert_eq!(
            ModelArtifact::published_for_shadow(
                FairValueMethod::Microprice,
                1_000,
                &fill,
                &pnl,
                ArtifactMetadata {
                    provenance: LabelProvenance::Observed,
                    window_from_ns: 0,
                    window_to_ns: 10,
                    dataset_id: "dataset".into(),
                },
                report,
            )
            .unwrap_err(),
            ArtifactError::Evidence
        );
    }
}
