//! Versioned model file. Loading is a cold path and may allocate.
//!
//! A file exists because the fit passed the research validation rule:
//! holdout log loss strictly below the constant training-rate baseline.
//! It is not a deployment to a trading system.

use std::fmt;
use std::fs;
use std::io::Write;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::model::calibration::{CalibrationBin, CALIBRATION_BINS};
use crate::model::fair_value::FairValueMethod;
use crate::model::feature::market::{MARKET_FEATURE_COUNT, MARKET_FEATURE_SCHEMA_VERSION};
use crate::model::feature::v1::{FEATURE_COUNT, FEATURE_SCHEMA_VERSION};
use crate::model::fill::LogisticRegression;
use crate::model::prediction::InvalidWeights;

pub const ARTIFACT_VERSION: u32 = 5;

/// Which predictive problem a file answers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelFamily {
    /// `P(fill within H | market state, quote context)`.
    QuoteFill,
    /// `P(reference price rises by at least a threshold within H | market state)`.
    MarketUpMove,
}

/// Where the binary labels came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LabelSource {
    /// Quote outcomes exported from a system that submitted the quotes.
    ObservedQuote,
    /// Quote outcomes marked as simulated.
    SimulatedQuote,
    /// Labels computed from later public book snapshots.
    PublicBook,
}

/// Structured holdout comparison. The constant baseline uses the training slice only.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ValidationReport {
    pub rows_before_purge: u64,
    pub purged_training_rows: u64,
    /// Fitted sample. Equals `rows_before_purge - purged_training_rows`.
    pub train_examples: u64,
    pub holdout_examples: u64,
    pub holdout_start_ns: i64,
    pub train_positives: u64,
    pub train_negatives: u64,
    pub holdout_positives: u64,
    pub holdout_negatives: u64,
    pub log_loss: f64,
    pub baseline_log_loss: f64,
    pub brier: f64,
    pub baseline_brier: f64,
    pub calibration: [CalibrationBin; CALIBRATION_BINS],
}

/// Parameters and the metadata required to refuse an incompatible run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelArtifact {
    pub version: u32,
    pub family: ModelFamily,
    pub feature_schema: u32,
    pub fair_value_method: FairValueMethod,
    pub training_horizon_ns: i64,
    /// Present for [`ModelFamily::MarketUpMove`]. Absent for quote fill.
    pub threshold_bps: Option<f64>,
    pub bias: f64,
    pub weights: Vec<f64>,
    pub label_source: LabelSource,
    pub window_from_ns: i64,
    pub window_to_ns: i64,
    pub dataset_id: String,
    pub report: ValidationReport,
}

/// Identity stored with a validated artifact.
pub struct ArtifactMetadata {
    pub label_source: LabelSource,
    pub window_from_ns: i64,
    pub window_to_ns: i64,
    pub dataset_id: String,
}

impl ModelArtifact {
    /// Builds a validated research artifact, or refuses a report that did not beat the baseline.
    pub fn validated<const N: usize>(
        family: ModelFamily,
        method: FairValueMethod,
        training_horizon_ns: i64,
        threshold_bps: Option<f64>,
        model: &LogisticRegression<N>,
        metadata: ArtifactMetadata,
        report: ValidationReport,
    ) -> Result<Self, ArtifactError> {
        let ArtifactMetadata {
            label_source,
            window_from_ns,
            window_to_ns,
            dataset_id,
        } = metadata;
        if training_horizon_ns <= 0 || window_from_ns >= window_to_ns || dataset_id.is_empty() {
            return Err(ArtifactError::Evidence);
        }
        if !family_matches(family, label_source, threshold_bps) {
            return Err(ArtifactError::Target);
        }
        if !report_is_acceptable(&report) {
            return Err(ArtifactError::Evidence);
        }
        let artifact = Self {
            version: ARTIFACT_VERSION,
            family,
            feature_schema: schema_version(family),
            fair_value_method: method,
            training_horizon_ns,
            threshold_bps,
            bias: model.bias(),
            weights: model.weights().to_vec(),
            label_source,
            window_from_ns,
            window_to_ns,
            dataset_id,
            report,
        };
        artifact.validate_family(family, method)?;
        Ok(artifact)
    }

    /// Refuses a file whose family, schema, method, horizon, or threshold does not match.
    pub fn validate_for_inference(
        &self,
        family: ModelFamily,
        method: FairValueMethod,
        horizon_ns: i64,
        threshold_bps: Option<f64>,
    ) -> Result<(), ArtifactError> {
        self.validate_family(family, method)?;
        if self.training_horizon_ns != horizon_ns {
            return Err(ArtifactError::Horizon);
        }
        match (self.threshold_bps, threshold_bps) {
            (None, None) => {}
            (Some(stored), Some(expected)) if (stored - expected).abs() <= 1e-9 => {}
            _ => return Err(ArtifactError::Target),
        }
        if !report_is_acceptable(&self.report) {
            return Err(ArtifactError::Evidence);
        }
        Ok(())
    }

    pub fn validate_family(
        &self,
        family: ModelFamily,
        method: FairValueMethod,
    ) -> Result<(), ArtifactError> {
        if self.version != ARTIFACT_VERSION {
            return Err(ArtifactError::Version);
        }
        if self.family != family {
            return Err(ArtifactError::Target);
        }
        if self.feature_schema != schema_version(family) {
            return Err(ArtifactError::FeatureSchema);
        }
        if self.fair_value_method != method {
            return Err(ArtifactError::FairValueMethod);
        }
        if self.training_horizon_ns <= 0 {
            return Err(ArtifactError::Horizon);
        }
        if self.weights.len() != feature_count(family) {
            return Err(ArtifactError::Weights);
        }
        if !self.bias.is_finite() || self.weights.iter().any(|weight| !weight.is_finite()) {
            return Err(ArtifactError::Weights);
        }
        if !family_matches(family, self.label_source, self.threshold_bps) {
            return Err(ArtifactError::Target);
        }
        Ok(())
    }

    /// Quote-fill weights. Refuses a market-outcome file.
    pub fn quote_model(&self) -> Result<LogisticRegression<FEATURE_COUNT>, ArtifactError> {
        self.validate_family(ModelFamily::QuoteFill, self.fair_value_method)?;
        let mut weights = [0.0; FEATURE_COUNT];
        weights.copy_from_slice(&self.weights);
        LogisticRegression::from_weights(self.bias, weights).map_err(|_| ArtifactError::Weights)
    }

    /// Market-outcome weights. Refuses a quote-fill file.
    pub fn market_model(&self) -> Result<LogisticRegression<MARKET_FEATURE_COUNT>, ArtifactError> {
        self.validate_family(ModelFamily::MarketUpMove, self.fair_value_method)?;
        let mut weights = [0.0; MARKET_FEATURE_COUNT];
        weights.copy_from_slice(&self.weights);
        LogisticRegression::from_weights(self.bias, weights).map_err(|_| ArtifactError::Weights)
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

fn schema_version(family: ModelFamily) -> u32 {
    match family {
        ModelFamily::QuoteFill => FEATURE_SCHEMA_VERSION,
        ModelFamily::MarketUpMove => MARKET_FEATURE_SCHEMA_VERSION,
    }
}

fn feature_count(family: ModelFamily) -> usize {
    match family {
        ModelFamily::QuoteFill => FEATURE_COUNT,
        ModelFamily::MarketUpMove => MARKET_FEATURE_COUNT,
    }
}

fn family_matches(
    family: ModelFamily,
    label_source: LabelSource,
    threshold_bps: Option<f64>,
) -> bool {
    match family {
        ModelFamily::QuoteFill => {
            threshold_bps.is_none()
                && matches!(
                    label_source,
                    LabelSource::ObservedQuote | LabelSource::SimulatedQuote
                )
        }
        ModelFamily::MarketUpMove => {
            matches!(threshold_bps, Some(bps) if bps.is_finite() && bps >= 0.0)
                && label_source == LabelSource::PublicBook
        }
    }
}

/// Holdout log loss must be strictly better than the constant training baseline.
pub fn log_loss_accepts(model_loss: f64, baseline_loss: f64) -> bool {
    model_loss < baseline_loss
}

fn report_is_acceptable(report: &ValidationReport) -> bool {
    let numbers = [
        report.log_loss,
        report.baseline_log_loss,
        report.brier,
        report.baseline_brier,
    ];
    report.rows_before_purge >= report.purged_training_rows
        && report.train_examples == report.rows_before_purge - report.purged_training_rows
        && report.train_examples > 0
        && report.holdout_examples > 0
        && report.train_positives > 0
        && report.train_negatives > 0
        && report.holdout_positives > 0
        && report.holdout_negatives > 0
        && numbers.iter().all(|value| value.is_finite())
        && report.calibration.iter().all(bin_is_finite)
        && log_loss_accepts(report.log_loss, report.baseline_log_loss)
}

fn bin_is_finite(bin: &CalibrationBin) -> bool {
    bin.lower.is_finite()
        && bin.upper.is_finite()
        && match (bin.count, bin.mean_predicted, bin.observed_frequency) {
            (0, None, None) => true,
            (count, Some(mean), Some(frequency)) if count > 0 => {
                mean.is_finite() && frequency.is_finite()
            }
            _ => false,
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
    Target,
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
            Self::Target => "model family, label source, or threshold does not match",
            Self::Evidence => "model artifact did not pass the configured validation rule",
        };
        f.write_str(message)
    }
}

impl std::error::Error for ArtifactError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::calibration::{score_probabilities, CalibrationBin, CALIBRATION_BINS};

    fn empty_bins() -> [CalibrationBin; CALIBRATION_BINS] {
        score_probabilities(&[(0.2, false), (0.8, true)])
            .unwrap()
            .bins
    }

    fn passing_report() -> ValidationReport {
        ValidationReport {
            rows_before_purge: 16,
            purged_training_rows: 0,
            train_examples: 16,
            holdout_examples: 4,
            holdout_start_ns: 1_000,
            train_positives: 4,
            train_negatives: 12,
            holdout_positives: 2,
            holdout_negatives: 2,
            log_loss: 0.4,
            baseline_log_loss: 0.6,
            brier: 0.2,
            baseline_brier: 0.25,
            calibration: empty_bins(),
        }
    }

    fn quote_artifact(
        source: LabelSource,
        report: ValidationReport,
    ) -> Result<ModelArtifact, ArtifactError> {
        let model = LogisticRegression::<FEATURE_COUNT>::zeros();
        ModelArtifact::validated(
            ModelFamily::QuoteFill,
            FairValueMethod::Microprice,
            1_000,
            None,
            &model,
            ArtifactMetadata {
                label_source: source,
                window_from_ns: 0,
                window_to_ns: 10,
                dataset_id: "dataset".into(),
            },
            report,
        )
    }

    #[test]
    fn a_quote_artifact_is_refused_as_a_market_model() {
        let artifact = quote_artifact(LabelSource::ObservedQuote, passing_report()).unwrap();
        artifact
            .validate_for_inference(
                ModelFamily::QuoteFill,
                FairValueMethod::Microprice,
                1_000,
                None,
            )
            .unwrap();
        assert_eq!(
            artifact.validate_for_inference(
                ModelFamily::MarketUpMove,
                FairValueMethod::Microprice,
                1_000,
                Some(1.0),
            ),
            Err(ArtifactError::Target)
        );
        assert!(artifact.market_model().is_err());
        assert!(artifact.quote_model().is_ok());
    }

    #[test]
    fn a_tie_with_the_constant_baseline_is_not_validated() {
        let mut report = passing_report();
        report.log_loss = report.baseline_log_loss;
        assert!(!log_loss_accepts(report.log_loss, report.baseline_log_loss));
        assert_eq!(
            quote_artifact(LabelSource::ObservedQuote, report).unwrap_err(),
            ArtifactError::Evidence
        );
    }

    #[test]
    fn an_older_artifact_version_is_rejected() {
        let path = std::env::temp_dir().join(format!(
            "mm-artifact-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&path, r#"{"version":3}"#).unwrap();
        assert_eq!(
            ModelArtifact::read_json(&path).unwrap_err(),
            ArtifactError::Json
        );
        let artifact = quote_artifact(LabelSource::ObservedQuote, passing_report()).unwrap();
        let mut value = serde_json::to_value(&artifact).unwrap();
        value["version"] = serde_json::json!(4);
        std::fs::write(&path, value.to_string()).unwrap();
        assert_eq!(
            ModelArtifact::read_json(&path).unwrap_err(),
            ArtifactError::Version
        );
        let _ = std::fs::remove_file(&path);
    }
}
