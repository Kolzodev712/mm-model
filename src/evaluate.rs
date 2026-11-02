//! One-book shadow evaluation. Buffers are supplied by the caller.
//!
//! `BookRecord -> fair value -> candidates -> features -> models -> EV -> selection`
//! does not allocate when the caller reuses `candidates` and `predictions`.

use std::fmt;

use crate::candidates::{BpsGrid, CandidateSlot, MAX_CANDIDATES};
use crate::fair_value::{FairValueError, FairValueMethod};
use crate::features::FEATURE_COUNT;
use crate::marketdata::BookRecord;
use crate::model::{
    predict_quote, ConditionalPnlModel, FillProbabilityModel, PredictError, QuotePrediction,
};
use crate::selection::{select_learned, LearnedChoice, ScoredCandidate};
use crate::units::CandidateId;

/// Result of scoring one book against the configured grid.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SnapshotEval {
    pub fair_value: f64,
    pub best_bid: f64,
    pub best_ask: f64,
    pub candidate_count: usize,
    pub choice: LearnedChoice,
    pub selected: Option<QuotePrediction>,
    pub selected_price: Option<f64>,
}

/// Why one book could not be scored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvalError {
    FairValue(FairValueError),
    Candidates,
    Predict(PredictError),
}

impl fmt::Display for EvalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FairValue(err) => write!(f, "fair value failed: {err}"),
            Self::Candidates => f.write_str("candidate generation failed"),
            Self::Predict(err) => write!(f, "prediction failed: {err}"),
        }
    }
}

impl std::error::Error for EvalError {}

/// Scores one book into reused buffers and selects the best candidate.
pub fn evaluate_book<F, P>(
    book: &BookRecord,
    method: FairValueMethod,
    grid: &BpsGrid,
    fill_model: &F,
    pnl_model: &P,
    candidates: &mut [CandidateSlot],
    predictions: &mut [QuotePrediction],
) -> Result<SnapshotEval, EvalError>
where
    F: FillProbabilityModel<FEATURE_COUNT>,
    P: ConditionalPnlModel<FEATURE_COUNT>,
{
    let market = method.derive_book(book).map_err(EvalError::FairValue)?;
    let count = grid
        .generate(&market, candidates)
        .map_err(|_| EvalError::Candidates)?;
    if predictions.len() < count {
        return Err(EvalError::Candidates);
    }
    let mut scores = [ScoredCandidate {
        id: CandidateId::new(0),
        expected_value: 0.0,
    }; MAX_CANDIDATES];
    for index in 0..count {
        predictions[index] =
            predict_quote(&market, &candidates[index].quote, fill_model, pnl_model)
                .map_err(EvalError::Predict)?;
        scores[index] = ScoredCandidate {
            id: candidates[index].id,
            expected_value: predictions[index].expected_value,
        };
    }
    let choice = select_learned(&scores[..count]);
    let (selected, selected_price) = match choice {
        LearnedChoice::NoQuote => (None, None),
        LearnedChoice::Quote { id, .. } => {
            let found = candidates[..count]
                .iter()
                .zip(predictions[..count].iter())
                .find(|(candidate, _)| candidate.id == id);
            match found {
                Some((candidate, prediction)) => (Some(*prediction), Some(candidate.quote.price())),
                None => (None, None),
            }
        }
    };
    let best_bid = book.best_bid().map(|level| level.price).unwrap_or(0.0);
    let best_ask = book.best_ask().map(|level| level.price).unwrap_or(0.0);
    Ok(SnapshotEval {
        fair_value: market.fair_value(),
        best_bid,
        best_ask,
        candidate_count: count,
        choice,
        selected,
        selected_price,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::candidates::BpsGrid;
    use crate::features::FEATURE_COUNT;
    use crate::linear::LinearRegression;
    use crate::logistic::LogisticRegression;
    use crate::marketdata::{BookRecord, BookStamp, Instrument, Level, BOOK_DEPTH};
    use crate::source::{BookSource, SliceSource};
    use crate::stats::{BookOutcome, RunStats};
    use crate::units::CandidateId;

    fn book() -> BookRecord {
        let mut bids = [Level::EMPTY; BOOK_DEPTH];
        let mut asks = [Level::EMPTY; BOOK_DEPTH];
        bids[0] = Level {
            price: 99.9,
            quantity: 1.0,
        };
        asks[0] = Level {
            price: 100.1,
            quantity: 1.0,
        };
        BookRecord::try_new(
            Instrument::new("binance", "BTCUSDT").unwrap(),
            BookStamp {
                local_receive_ns: 1_000,
                exchange_event_ns: None,
                update_id: Some(1),
            },
            bids,
            1,
            asks,
            1,
        )
        .unwrap()
    }

    #[test]
    fn known_weights_select_the_deeper_discount() {
        let grid = BpsGrid::parse("1,5", 1.0).unwrap();
        let fill = LogisticRegression::<FEATURE_COUNT>::zeros();
        let mut pnl_weights = [0.0; FEATURE_COUNT];
        pnl_weights[0] = 1_000.0;
        let pnl = LinearRegression::from_weights(0.0, pnl_weights).unwrap();
        let mut candidates = [empty_slot(); MAX_CANDIDATES];
        let mut predictions = [empty_prediction(); MAX_CANDIDATES];
        let snapshot = book();
        let evaluated = evaluate_book(
            &snapshot,
            FairValueMethod::Microprice,
            &grid,
            &fill,
            &pnl,
            &mut candidates,
            &mut predictions,
        )
        .unwrap();
        match evaluated.choice {
            LearnedChoice::Quote { id, expected_value } => {
                assert_eq!(id, CandidateId::new(1));
                assert!(expected_value > 0.0);
                assert!(evaluated.selected_price.unwrap() < candidates[0].quote.price());
            }
            LearnedChoice::NoQuote => panic!("both candidates have positive expected value"),
        }

        let mut stats = RunStats::default();
        stats.record_book(BookOutcome {
            choice: evaluated.choice,
            selected: evaluated.selected,
            candidate_count: evaluated.candidate_count,
            best_bid: evaluated.best_bid,
            best_ask: evaluated.best_ask,
            fair_value: evaluated.fair_value,
            latency_ns: 10,
        });
        assert_eq!(stats.quote_decisions, 1);
        assert_eq!(stats.evaluations, 2);
        assert_eq!(stats.selection_count(CandidateId::new(1)), 1);
        assert!(stats.mean_selected_probability().unwrap() > 0.0);

        let mut source = SliceSource::new(vec![snapshot]);
        assert!(source.next_book().unwrap().is_some());
        assert!(source.next_book().unwrap().is_none());
    }

    fn empty_slot() -> CandidateSlot {
        use crate::decision::CandidateQuote;
        CandidateSlot {
            id: CandidateId::new(0),
            quote: CandidateQuote::new(1.0, 1.0).unwrap(),
        }
    }

    fn empty_prediction() -> QuotePrediction {
        use crate::units::{Pnl, Probability};
        QuotePrediction {
            fill_probability: Probability::new(0.0).unwrap(),
            expected_pnl_if_filled: Pnl::new(0.0).unwrap(),
            expected_value: 0.0,
        }
    }
}
