//! Holdout probability scores. No plots.

/// Number of fixed probability bins from 0 to 1.
pub const CALIBRATION_BINS: usize = 10;

/// One bin of predicted probability versus the observed frequency.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(
    feature = "artifact",
    derive(serde::Serialize, serde::Deserialize),
    serde(deny_unknown_fields)
)]
pub struct CalibrationBin {
    pub lower: f64,
    pub upper: f64,
    pub count: u64,
    /// Mean predicted probability. `None` when the bin is empty.
    pub mean_predicted: Option<f64>,
    /// Observed positive frequency. `None` when the bin is empty.
    pub observed_frequency: Option<f64>,
}

/// Log loss, Brier score, and the ten-bin summary for one set of predictions.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(
    feature = "artifact",
    derive(serde::Serialize, serde::Deserialize),
    serde(deny_unknown_fields)
)]
pub struct ProbabilityScores {
    pub log_loss: f64,
    pub brier: f64,
    pub bins: [CalibrationBin; CALIBRATION_BINS],
}

/// Scores `(probability, label)` pairs.
///
/// An empty slice, or a probability outside `[0, 1]`, is an error.
pub fn score_probabilities(pairs: &[(f64, bool)]) -> Result<ProbabilityScores, ScoreError> {
    if pairs.is_empty() {
        return Err(ScoreError::Empty);
    }
    let mut counts = [0u64; CALIBRATION_BINS];
    let mut predicted_sum = [0.0; CALIBRATION_BINS];
    let mut positive_sum = [0u64; CALIBRATION_BINS];
    let mut log_loss = 0.0;
    let mut brier = 0.0;
    for &(probability, label) in pairs {
        if !(probability.is_finite() && (0.0..=1.0).contains(&probability)) {
            return Err(ScoreError::Probability);
        }
        let bin = bin_index(probability);
        counts[bin] += 1;
        predicted_sum[bin] += probability;
        if label {
            positive_sum[bin] += 1;
        }
        let y = if label { 1.0 } else { 0.0 };
        log_loss += binary_cross_entropy(y, probability);
        let gap = probability - y;
        brier += gap * gap;
    }
    let scale = pairs.len() as f64;
    let mut bins = [CalibrationBin {
        lower: 0.0,
        upper: 0.0,
        count: 0,
        mean_predicted: None,
        observed_frequency: None,
    }; CALIBRATION_BINS];
    for (index, bin) in bins.iter_mut().enumerate() {
        let lower = index as f64 / CALIBRATION_BINS as f64;
        let upper = (index + 1) as f64 / CALIBRATION_BINS as f64;
        let count = counts[index];
        *bin = CalibrationBin {
            lower,
            upper,
            count,
            mean_predicted: (count > 0).then(|| predicted_sum[index] / count as f64),
            observed_frequency: (count > 0).then(|| positive_sum[index] as f64 / count as f64),
        };
    }
    Ok(ProbabilityScores {
        log_loss: log_loss / scale,
        brier: brier / scale,
        bins,
    })
}

fn bin_index(probability: f64) -> usize {
    if probability >= 1.0 {
        CALIBRATION_BINS - 1
    } else {
        let index = (probability * CALIBRATION_BINS as f64).floor() as usize;
        index.min(CALIBRATION_BINS - 1)
    }
}

fn binary_cross_entropy(label: f64, probability: f64) -> f64 {
    const EPS: f64 = 1e-12;
    let clipped = probability.clamp(EPS, 1.0 - EPS);
    -(label * clipped.ln() + (1.0 - label) * (1.0 - clipped).ln())
}

/// Probability scoring failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScoreError {
    Empty,
    Probability,
}

impl std::fmt::Display for ScoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => f.write_str("probability scoring requires at least one row"),
            Self::Probability => f.write_str("predicted probability must be finite and in [0, 1]"),
        }
    }
}

impl std::error::Error for ScoreError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn brier_and_bins_match_a_fixed_sample() {
        let scores = score_probabilities(&[(0.2, false), (0.2, true), (1.0, true)]).unwrap();
        let expected_brier = ((0.2_f64 - 0.0).powi(2) + (0.2_f64 - 1.0).powi(2) + 0.0) / 3.0;
        assert!((scores.brier - expected_brier).abs() < 1e-12);
        assert_eq!(scores.bins[2].count, 2);
        assert!((scores.bins[2].mean_predicted.unwrap() - 0.2).abs() < 1e-12);
        assert!((scores.bins[2].observed_frequency.unwrap() - 0.5).abs() < 1e-12);
        assert_eq!(scores.bins[9].count, 1);
        assert!(scores.bins[0].mean_predicted.is_none());
        assert!(scores.log_loss.is_finite());
    }
}
