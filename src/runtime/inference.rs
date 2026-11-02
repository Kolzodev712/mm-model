//! Live probability estimates from a public book.
//!
//! `BookRecord -> fair value -> features -> probabilities`.
//! Probe contexts are scored independently. The result has no chosen probe.

use std::fmt;

use crate::domain::book::BookRecord;
use crate::domain::units::Probability;
use crate::model::fair_value::{FairValueError, FairValueMethod};
use crate::model::feature::market::MARKET_FEATURE_COUNT;
use crate::model::feature::v1::FEATURE_COUNT;
use crate::model::fill::LogisticRegression;
use crate::model::market::MarketOutcomeDefinition;
use crate::model::prediction::{
    predict_market_outcome, predict_quote_outcome, PredictError, QuoteOutcomePrediction,
};
use crate::model::probe::{Probe, ProbeError, ProbeGrid, MAX_PROBES};

/// One probe and its fill probability. Order follows the configured grid.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProbeProbability {
    pub offset_bps: f64,
    pub quantity: f64,
    pub fill_probability: Probability,
}

/// Probabilities for one book. Probe values are written into the caller buffer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BookProbabilities {
    pub fair_value: f64,
    pub best_bid: f64,
    pub best_ask: f64,
    pub market: Option<crate::model::prediction::MarketOutcomePrediction>,
    pub probe_count: usize,
}

/// Why one book could not be scored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvalError {
    FairValue(FairValueError),
    Probes(ProbeError),
    Predict(PredictError),
}

impl fmt::Display for EvalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FairValue(err) => write!(f, "fair value failed: {err}"),
            Self::Probes(err) => write!(f, "probe grid failed: {err}"),
            Self::Predict(err) => write!(f, "prediction failed: {err}"),
        }
    }
}

impl std::error::Error for EvalError {}

/// Models available for one live book. Either family may be absent.
pub struct InferenceInputs<'a> {
    pub quote_model: Option<&'a LogisticRegression<FEATURE_COUNT>>,
    pub market_model: Option<&'a LogisticRegression<MARKET_FEATURE_COUNT>>,
    pub market_definition: Option<MarketOutcomeDefinition>,
}

/// Caller-owned buffers so scoring does not allocate.
pub struct InferenceBuffers<'a> {
    pub probes: &'a mut [Probe],
    pub probabilities: &'a mut [ProbeProbability],
}

/// Scores one book into reused buffers.
///
/// Quote and market probabilities are independent. No probe is marked as better
/// than another.
pub fn evaluate_book(
    book: &BookRecord,
    method: FairValueMethod,
    grid: &ProbeGrid,
    inputs: &InferenceInputs<'_>,
    buffers: &mut InferenceBuffers<'_>,
) -> Result<BookProbabilities, EvalError> {
    let fair_value = method.estimate_book(book).map_err(EvalError::FairValue)?;
    let bid = book
        .best_bid()
        .ok_or(EvalError::FairValue(FairValueError::NoBid))?;
    let ask = book
        .best_ask()
        .ok_or(EvalError::FairValue(FairValueError::NoAsk))?;
    let market = crate::domain::market::MarketState::from_estimate(
        method,
        fair_value,
        bid.price,
        bid.quantity,
        ask.price,
        ask.quantity,
    );
    let market_prediction = match (inputs.market_model, inputs.market_definition) {
        (Some(model), Some(definition)) => {
            Some(predict_market_outcome(&market, definition, model).map_err(EvalError::Predict)?)
        }
        _ => None,
    };
    let count = if let Some(model) = inputs.quote_model {
        let n = grid
            .generate(&market, buffers.probes)
            .map_err(EvalError::Probes)?;
        if buffers.probabilities.len() < n {
            return Err(EvalError::Probes(ProbeError::Buffer));
        }
        for (index, probe) in buffers.probes[..n].iter().enumerate() {
            let prediction: QuoteOutcomePrediction =
                predict_quote_outcome(&market, &probe.quote, model).map_err(EvalError::Predict)?;
            buffers.probabilities[index] = ProbeProbability {
                offset_bps: probe.offset_bps,
                quantity: probe.quote.quantity(),
                fill_probability: prediction.fill_probability,
            };
        }
        n
    } else {
        0
    };
    Ok(BookProbabilities {
        fair_value,
        best_bid: bid.price,
        best_ask: ask.price,
        market: market_prediction,
        probe_count: count,
    })
}

/// Live inference loop. Logged numbers are probabilities, not fills or P&L.
#[cfg(all(feature = "live", feature = "parquet-storage"))]
pub fn run_live<C: crate::runtime::clock::Clock>(
    config: &crate::runtime::config::AppConfig,
    client: &mut crate::data::market::binance::client::BinanceBookClient<C>,
    quote_model: Option<&LogisticRegression<FEATURE_COUNT>>,
    market_model: Option<&LogisticRegression<MARKET_FEATURE_COUNT>>,
    market_definition: Option<MarketOutcomeDefinition>,
    should_stop: &impl Fn() -> bool,
) -> Result<crate::runtime::stats::RunStats, InferenceRunError> {
    use crate::data::market::source::BookSource;
    let crate::runtime::config::TaskConfig::LiveInference(task) = &config.task else {
        return Err(InferenceRunError::Config);
    };
    let pairs: Vec<(f64, f64)> = task
        .models
        .iter()
        .flat_map(|model| {
            model
                .probes
                .iter()
                .map(|probe| (probe.distance_bps, probe.quantity))
        })
        .collect();
    let grid = ProbeGrid::from_pairs(&pairs).map_err(InferenceRunError::Grid)?;
    let mut probes = [empty_probe(); MAX_PROBES];
    let mut probabilities = [empty_probability(); MAX_PROBES];
    let mut stats = crate::runtime::stats::RunStats::default();
    let started = std::time::Instant::now();
    let mut last_report = std::time::Duration::ZERO;
    while !should_stop() && task.run_for.is_none_or(|limit| started.elapsed() < limit) {
        match client.next_book() {
            Err(err) if client.is_idle(&err) => continue,
            Err(err) => return Err(InferenceRunError::Client(err)),
            Ok(None) => break,
            Ok(Some(book)) => {
                let tick = std::time::Instant::now();
                match evaluate_book(
                    &book,
                    config.market.fair_value,
                    &grid,
                    &InferenceInputs {
                        quote_model,
                        market_model,
                        market_definition,
                    },
                    &mut InferenceBuffers {
                        probes: &mut probes,
                        probabilities: &mut probabilities,
                    },
                ) {
                    Ok(result) => {
                        let latency_ns = tick.elapsed().as_nanos();
                        log_book(&result, &probabilities);
                        stats.record_book(crate::runtime::stats::BookOutcome {
                            probe_count: result.probe_count,
                            market_probability: result
                                .market
                                .map(|prediction| prediction.probability.get()),
                            best_bid: result.best_bid,
                            best_ask: result.best_ask,
                            fair_value: result.fair_value,
                            latency_ns,
                        });
                    }
                    Err(_) => stats.record_model_error(),
                }
            }
        }
        stats.reconnects = client.reconnects();
        if started.elapsed().saturating_sub(last_report) >= task.report_every {
            log_progress(&stats);
            last_report = started.elapsed();
        }
    }
    stats.reconnects = client.reconnects();
    log_progress(&stats);
    Ok(stats)
}

#[cfg(all(feature = "live", feature = "parquet-storage"))]
fn log_book(result: &BookProbabilities, probabilities: &[ProbeProbability]) {
    #[cfg(feature = "observe")]
    {
        let market = result.market.map(|prediction| prediction.probability.get());
        let probes: Vec<String> = probabilities[..result.probe_count]
            .iter()
            .map(|probe| {
                format!(
                    "{:.4}bp={:.4}",
                    probe.offset_bps,
                    probe.fill_probability.get()
                )
            })
            .collect();
        tracing::debug!(
            fair = result.fair_value,
            bid = result.best_bid,
            ask = result.best_ask,
            market_probability = ?market,
            probes = ?probes,
            "live inference probabilities"
        );
    }
}

#[cfg(all(feature = "live", feature = "parquet-storage"))]
fn log_progress(stats: &crate::runtime::stats::RunStats) {
    #[cfg(feature = "observe")]
    tracing::info!(
        books = stats.books_processed,
        rejected = stats.books_rejected,
        model_errors = stats.model_errors,
        reconnects = stats.reconnects,
        probe_evaluations = stats.probe_evaluations,
        mean_market_probability = ?stats.mean_market_probability(),
        mean_latency_ns = ?stats.mean_latency_ns(),
        "live inference progress; these are probabilities, not realized outcomes"
    );
}

fn empty_probe() -> Probe {
    Probe {
        offset_bps: 0.0,
        quote: crate::domain::market::QuoteContext::new(1.0, 1.0).expect("placeholder quote"),
    }
}

fn empty_probability() -> ProbeProbability {
    ProbeProbability {
        offset_bps: 0.0,
        quantity: 1.0,
        fill_probability: Probability::new(0.0).expect("zero"),
    }
}

#[cfg(all(feature = "live", feature = "parquet-storage"))]
#[derive(Debug)]
pub enum InferenceRunError {
    Client(crate::data::market::binance::client::ClientError),
    Grid(ProbeError),
    Config,
}

#[cfg(all(feature = "live", feature = "parquet-storage"))]
impl fmt::Display for InferenceRunError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Client(err) => write!(f, "{err}"),
            Self::Grid(err) => write!(f, "{err}"),
            Self::Config => f.write_str("live inference was started without a live-inference task"),
        }
    }
}

#[cfg(all(feature = "live", feature = "parquet-storage"))]
impl std::error::Error for InferenceRunError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::book::{BookStamp, Instrument, Level, BOOK_DEPTH};
    use crate::model::fair_value::FairValueMethod;

    fn book() -> BookRecord {
        let mut bids = [Level::EMPTY; BOOK_DEPTH];
        let mut asks = [Level::EMPTY; BOOK_DEPTH];
        bids[0] = Level {
            price: 99.9,
            quantity: 2.0,
        };
        asks[0] = Level {
            price: 100.1,
            quantity: 1.0,
        };
        BookRecord::try_new(
            Instrument::new("binance", "BTCUSDT").unwrap(),
            BookStamp {
                local_receive_ns: 1,
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
    fn probes_are_independent_and_keep_grid_order() {
        let mut weights = [0.0; FEATURE_COUNT];
        weights[0] = -1.0;
        let quote = LogisticRegression::from_weights(-1.0, weights).unwrap();
        let market = LogisticRegression::<MARKET_FEATURE_COUNT>::from_weights(
            0.4,
            [0.0; MARKET_FEATURE_COUNT],
        )
        .unwrap();
        let definition = MarketOutcomeDefinition::new_up_move(1_000_000_000, 1.0).unwrap();
        let grid = ProbeGrid::parse("0.5,5", 0.001).unwrap();
        let mut probes = [empty_probe(); MAX_PROBES];
        let mut probabilities = [empty_probability(); MAX_PROBES];
        let result = evaluate_book(
            &book(),
            FairValueMethod::Microprice,
            &grid,
            &InferenceInputs {
                quote_model: Some(&quote),
                market_model: Some(&market),
                market_definition: Some(definition),
            },
            &mut InferenceBuffers {
                probes: &mut probes,
                probabilities: &mut probabilities,
            },
        )
        .unwrap();
        assert_eq!(result.probe_count, 2);
        assert_eq!(probabilities[0].offset_bps, 0.5);
        assert_eq!(probabilities[1].offset_bps, 5.0);
        assert!(probabilities[0].fill_probability.get() > probabilities[1].fill_probability.get());
        let market_probability = result.market.unwrap().probability.get();
        assert!((0.0..=1.0).contains(&market_probability));
        assert_eq!(result.market.unwrap().definition, definition);
    }
}
