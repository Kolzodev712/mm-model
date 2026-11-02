//! Read-only book recorder.
//!
//! The process owns configuration and the run loop. The websocket client,
//! clock, and log writer live in the library.

#![deny(unsafe_code)]

use std::process::ExitCode;
use std::time::Duration;

use clap::Parser;
use mm_model::clock::SystemClock;
use mm_model::marketdata::{BinanceBookClient, BinanceBookConfig, BookLogWriter, BINANCE_SPOT_WS};
use mm_model::reconnect::ReconnectPolicy;
use mm_model::source::BookSource;

#[derive(Debug, Parser)]
#[command(name = "record_book")]
struct Args {
    /// Output path for the MMBK book log.
    output: String,
    /// Spot symbol. BTCUSDT is only the example default.
    #[arg(long, default_value = "BTCUSDT")]
    symbol: String,
    /// Append this many records before flushing. There is no per-record fsync.
    #[arg(long, default_value_t = 10)]
    flush_every: u32,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("record_book: {err}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let config = BinanceBookConfig::new(BINANCE_SPOT_WS, &args.symbol, 10, 100)?;
    let policy = ReconnectPolicy::new(5, Duration::from_millis(200), Duration::from_secs(2))
        .map_err(|err| err.to_string())?;
    let mut client = BinanceBookClient::connect(&config, SystemClock, policy)?;
    let mut writer = BookLogWriter::open(std::path::Path::new(&args.output), args.flush_every)?;
    while let Some(book) = client.next_book()? {
        writer.write(&book)?;
    }
    Ok(())
}
