//! Live shadow inference. No orders are sent and no fills are observed.

#![deny(unsafe_code)]

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use clap::{Parser, ValueEnum};
use mm_model::artifact::ModelArtifact;
use mm_model::candidates::{BpsGrid, CandidateSlot, MAX_CANDIDATES};
use mm_model::clock::SystemClock;
use mm_model::demo::{train_demo_models, DEMO_WARNING};
use mm_model::evaluate::evaluate_book;
use mm_model::fair_value::FairValueMethod;
use mm_model::features::{FEATURE_COUNT, FEATURE_SCHEMA_V1};
use mm_model::marketdata::{BinanceBookClient, BinanceBookConfig, ClientError, BINANCE_SPOT_WS};
use mm_model::model::QuotePrediction;
use mm_model::reconnect::ReconnectPolicy;
use mm_model::selection::LearnedChoice;
use mm_model::source::BookSource;
use mm_model::stats::{BookOutcome, RunPolicy, RunStats};
use mm_model::units::{Pnl, Probability};
use tracing::info;

#[derive(Debug, Clone, Copy, ValueEnum)]
enum FairArg {
    Microprice,
    Midpoint,
}

impl FairArg {
    fn method(self) -> FairValueMethod {
        match self {
            Self::Microprice => FairValueMethod::Microprice,
            Self::Midpoint => FairValueMethod::Midpoint,
        }
    }
}

#[derive(Debug, Parser)]
#[command(name = "mm-shadow")]
struct Args {
    /// How long to run, for example 60s, 5m, or 500ms.
    #[arg(long)]
    duration: String,
    /// How often to emit a progress line, for example 5s.
    #[arg(long)]
    report_every: String,
    /// Spot symbol. BTCUSDT is only the example default.
    #[arg(long, default_value = "BTCUSDT")]
    symbol: String,
    /// Book-derived reference price.
    #[arg(long, value_enum, default_value = "microprice")]
    fair_value: FairArg,
    /// Candidate order quantity.
    #[arg(long, default_value_t = 0.001)]
    quantity: f64,
    /// Comma-separated basis-point discounts from the reference price.
    #[arg(long, default_value = "0.5,1,2,5")]
    offsets_bps: String,
    /// Previously written model artifact.
    #[arg(long, group = "source")]
    model: Option<PathBuf>,
    /// Train a deterministic simulated model at startup. Not trading-valid.
    #[arg(long, group = "source")]
    demo_model: bool,
    /// tracing filter, for example info or debug.
    #[arg(long, default_value = "info")]
    log_level: String,
}

fn main() -> ExitCode {
    let args = Args::parse();
    if let Err(err) = run(args) {
        eprintln!("mm-shadow: {err}");
        return ExitCode::from(2);
    }
    ExitCode::SUCCESS
}

fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    if args.model.is_some() == args.demo_model {
        return Err("specify exactly one of --model or --demo-model".into());
    }
    tracing_subscriber::fmt()
        .with_env_filter(args.log_level.clone())
        .with_target(false)
        .init();

    let duration = parse_duration(&args.duration)?;
    let report_every = parse_duration(&args.report_every)?;
    let method = args.fair_value.method();
    let grid = BpsGrid::parse(&args.offsets_bps, args.quantity)?;
    let (fill, pnl, horizon_ns, demo) = load_models(&args, method)?;
    if demo {
        eprintln!("{DEMO_WARNING}");
        tracing::warn!("{DEMO_WARNING}");
    }

    info!(
        run_mode = if demo { "demo" } else { "artifact" },
        symbol = args.symbol.as_str(),
        duration_ms = duration.as_millis() as u64,
        report_every_ms = report_every.as_millis() as u64,
        fair_value = method.name(),
        feature_schema = FEATURE_SCHEMA_V1,
        feature_count = FEATURE_COUNT,
        offsets_bps = args.offsets_bps.as_str(),
        quantity = args.quantity,
        training_horizon_ns = horizon_ns,
        "shadow run starting; no orders will be sent"
    );

    let config = BinanceBookConfig::new(BINANCE_SPOT_WS, &args.symbol, 10, 100)?;
    let policy = ReconnectPolicy::new(8, Duration::from_millis(250), Duration::from_secs(5))
        .map_err(|err| err.to_string())?;
    let mut client = BinanceBookClient::connect(&config, SystemClock, policy)?;
    let shutdown = install_shutdown()?;
    let mut stats = RunStats::default();
    let mut candidates = [empty_slot(); MAX_CANDIDATES];
    let mut predictions = [empty_prediction(); MAX_CANDIDATES];
    let started = Instant::now();
    let run_policy = RunPolicy {
        duration,
        report_every,
    };
    let mut last_report = Duration::ZERO;

    while !shutdown.load(Ordering::Relaxed) {
        let elapsed = started.elapsed();
        if run_policy.should_stop(elapsed) {
            break;
        }
        if run_policy.should_report(elapsed, last_report) {
            log_progress(&stats, elapsed, None);
            last_report = elapsed;
        }
        match client.next_book() {
            Ok(Some(book)) => {
                let tick = Instant::now();
                match evaluate_book(
                    &book,
                    method,
                    &grid,
                    &fill,
                    &pnl,
                    &mut candidates,
                    &mut predictions,
                ) {
                    Ok(evaluated) => {
                        let latency_ns = tick.elapsed().as_nanos();
                        stats.record_book(BookOutcome {
                            choice: evaluated.choice,
                            selected: evaluated.selected,
                            candidate_count: evaluated.candidate_count,
                            best_bid: evaluated.best_bid,
                            best_ask: evaluated.best_ask,
                            fair_value: evaluated.fair_value,
                            latency_ns,
                        });
                        if run_policy.should_report(started.elapsed(), last_report) {
                            log_progress(
                                &stats,
                                started.elapsed(),
                                Some((&evaluated, &candidates)),
                            );
                            last_report = started.elapsed();
                        }
                    }
                    Err(err) => {
                        stats.record_model_error();
                        tracing::warn!(error = %err, "skipped book");
                    }
                }
            }
            Ok(None) => break,
            Err(ClientError::Malformed) => {
                stats.record_rejection();
                tracing::warn!("malformed book message");
            }
            Err(err) => {
                stats.record_client_error();
                tracing::error!(error = %err, "binance client stopped");
                break;
            }
        }
    }
    stats.reconnects = client.reconnects();
    log_summary(&stats, duration, started.elapsed());
    Ok(())
}

fn load_models(
    args: &Args,
    method: FairValueMethod,
) -> Result<
    (
        mm_model::logistic::LogisticRegression<FEATURE_COUNT>,
        mm_model::linear::LinearRegression<FEATURE_COUNT>,
        i64,
        bool,
    ),
    Box<dyn std::error::Error>,
> {
    if args.demo_model {
        let (fill, pnl) = train_demo_models()?;
        return Ok((fill, pnl, 500, true));
    }
    let path = args.model.as_ref().ok_or("model path is missing")?;
    let artifact = ModelArtifact::read_json(path)?;
    artifact.validate_for_run(method)?;
    let (fill, pnl) = artifact.models()?;
    Ok((fill, pnl, artifact.training_horizon_ns, false))
}

fn log_progress(
    stats: &RunStats,
    elapsed: Duration,
    latest: Option<(&mm_model::evaluate::SnapshotEval, &[CandidateSlot])>,
) {
    if let Some((evaluated, slots)) = latest {
        let (decision, id, price, probability, pnl, ev) = describe(evaluated, slots);
        info!(
            elapsed_ms = elapsed.as_millis() as u64,
            books_received = stats.books_received,
            books_processed = stats.books_processed,
            books_rejected = stats.books_rejected,
            client_errors = stats.client_errors,
            model_errors = stats.model_errors,
            best_bid = evaluated.best_bid,
            best_ask = evaluated.best_ask,
            fair_value = evaluated.fair_value,
            candidate_count = evaluated.candidate_count,
            selected_id = id,
            selected_price = price,
            predicted_fill_probability = probability,
            predicted_conditional_pnl = pnl,
            predicted_ev = ev,
            decision,
            "shadow progress"
        );
    } else {
        info!(
            elapsed_ms = elapsed.as_millis() as u64,
            books_received = stats.books_received,
            books_processed = stats.books_processed,
            books_rejected = stats.books_rejected,
            "shadow progress"
        );
    }
}

fn describe(
    evaluated: &mm_model::evaluate::SnapshotEval,
    slots: &[CandidateSlot],
) -> (&'static str, u16, f64, f64, f64, f64) {
    match evaluated.choice {
        LearnedChoice::NoQuote => ("noquote", 0, 0.0, 0.0, 0.0, 0.0),
        LearnedChoice::Quote { id, expected_value } => {
            let price = slots
                .iter()
                .find(|slot| slot.id == id)
                .map(|slot| slot.quote.price())
                .unwrap_or(0.0);
            let prediction = evaluated.selected;
            (
                "quote",
                id.get(),
                price,
                prediction
                    .map(|item| item.fill_probability.get())
                    .unwrap_or(0.0),
                prediction
                    .map(|item| item.expected_pnl_if_filled.get())
                    .unwrap_or(0.0),
                expected_value,
            )
        }
    }
}

fn log_summary(stats: &RunStats, requested: Duration, elapsed: Duration) {
    info!(
        requested_duration_ms = requested.as_millis() as u64,
        elapsed_ms = elapsed.as_millis() as u64,
        books_received = stats.books_received,
        books_processed = stats.books_processed,
        books_rejected = stats.books_rejected,
        client_errors = stats.client_errors,
        model_errors = stats.model_errors,
        reconnects = stats.reconnects,
        evaluations = stats.evaluations,
        quote_decisions = stats.quote_decisions,
        noquote_decisions = stats.noquote_decisions,
        mean_selected_fill_probability = stats.mean_selected_probability(),
        mean_selected_conditional_pnl = stats.mean_selected_conditional_pnl(),
        mean_selected_ev = stats.mean_selected_ev(),
        min_selected_ev = stats.min_selected_ev(),
        max_selected_ev = stats.max_selected_ev(),
        latency_count = stats.books_processed,
        mean_latency_ns = stats.mean_latency_ns(),
        max_latency_ns = stats.max_latency_ns(),
        "shadow summary; these are model predictions, not realized fills or P&L"
    );
}

fn install_shutdown() -> Result<Arc<AtomicBool>, Box<dyn std::error::Error>> {
    let flag = Arc::new(AtomicBool::new(false));
    signal_hook::flag::register(signal_hook::consts::SIGINT, Arc::clone(&flag))?;
    signal_hook::flag::register(signal_hook::consts::SIGTERM, Arc::clone(&flag))?;
    Ok(flag)
}

fn parse_duration(text: &str) -> Result<Duration, String> {
    let (number, scale) = if let Some(rest) = text.strip_suffix("ms") {
        (rest, 1u64)
    } else if let Some(rest) = text.strip_suffix('s') {
        (rest, 1_000)
    } else if let Some(rest) = text.strip_suffix('m') {
        (rest, 60_000)
    } else if let Some(rest) = text.strip_suffix('h') {
        (rest, 3_600_000)
    } else {
        return Err(format!("duration must end with ms, s, m, or h: {text}"));
    };
    let value: u64 = number
        .parse()
        .map_err(|_| format!("duration is not a number: {text}"))?;
    if value == 0 {
        return Err("duration must be positive".into());
    }
    let millis = value
        .checked_mul(scale)
        .ok_or_else(|| format!("duration is too large: {text}"))?;
    Ok(Duration::from_millis(millis))
}

fn empty_slot() -> CandidateSlot {
    use mm_model::decision::CandidateQuote;
    use mm_model::units::CandidateId;
    CandidateSlot {
        id: CandidateId::new(0),
        quote: CandidateQuote::new(1.0, 1.0).expect("unit placeholder"),
    }
}

fn empty_prediction() -> QuotePrediction {
    QuotePrediction {
        fill_probability: Probability::new(0.0).expect("zero probability"),
        expected_pnl_if_filled: Pnl::new(0.0).expect("zero pnl"),
        expected_value: 0.0,
    }
}
