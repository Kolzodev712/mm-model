//! Read-only recorder for the Binance spot BTCUSDT partial book.
//!
//! ```text
//! cargo run --features recorder --bin record_book -- books.mmbk
//! ```
//!
//! The process appends top-10 snapshots from the public 100 ms stream until
//! it is disconnected or interrupted. It does not train a model, submit
//! orders, or synthesize fair values. A torn tail already in the file is
//! truncated before new records are appended. Each record is flushed to the
//! operating system; a crash can still drop the last buffered write.
//!
//! WebSocket reads allocate inside tungstenite. This process does not allocate
//! a book, a feature vector, or a JSON tree per update.

use std::env;
use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom, Write};
use std::process::ExitCode;
use std::time::{SystemTime, UNIX_EPOCH};

use mm_model::marketdata::{
    aligned_book_file_len, decode_book_header, encode_book_record, parse_partial_depth,
    write_book_header, BookRecord, Instrument, ParsedDepth, BINANCE_EXCHANGE, BOOK_HEADER_LEN,
    BOOK_RECORD_LEN, BTCUSDT, PARTIAL_DEPTH_URL,
};
use tungstenite::{connect, Message};

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
    let path = env::args()
        .nth(1)
        .ok_or("usage: record_book <output-file>")?;
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)?;
    prepare_log(&mut file)?;

    let instrument = Instrument::new(BINANCE_EXCHANGE, BTCUSDT)?;
    let (mut socket, _response) = connect(PARTIAL_DEPTH_URL)?;
    let mut depth = ParsedDepth::empty();
    let mut encoded = [0u8; BOOK_RECORD_LEN];

    loop {
        match socket.read()? {
            Message::Text(payload) => {
                let local_ns = unix_time_ns()?;
                if parse_partial_depth(payload.as_str(), &mut depth).is_err() {
                    eprintln!("record_book: skipped malformed book message");
                    continue;
                }
                let record = BookRecord::from_depth(instrument, local_ns, &depth)?;
                encode_book_record(&record, &mut encoded)?;
                file.write_all(&encoded)?;
                file.flush()?;
            }
            Message::Ping(payload) => {
                socket.send(Message::Pong(payload))?;
            }
            Message::Close(_) => break,
            Message::Binary(_) | Message::Frame(_) | Message::Pong(_) => {}
        }
    }
    Ok(())
}

fn prepare_log(file: &mut std::fs::File) -> Result<(), Box<dyn std::error::Error>> {
    let len = file.metadata()?.len();
    if len == 0 {
        write_book_header(file)?;
        file.flush()?;
        return Ok(());
    }
    file.seek(SeekFrom::Start(0))?;
    let mut header = [0u8; BOOK_HEADER_LEN];
    file.read_exact(&mut header)?;
    decode_book_header(&header)?;
    let aligned = aligned_book_file_len(len)?;
    if aligned != len {
        file.set_len(aligned)?;
    }
    file.seek(SeekFrom::Start(aligned))?;
    Ok(())
}

fn unix_time_ns() -> Result<i64, Box<dyn std::error::Error>> {
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    i64::try_from(nanos).map_err(|_| "local clock does not fit in unix nanoseconds".into())
}
