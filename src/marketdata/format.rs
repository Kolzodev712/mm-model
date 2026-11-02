//! Versioned little-endian logs for book prints and external fair values.
//!
//! Book file, version 1:
//!
//! ```text
//! header, 12 bytes:
//!   0..4   magic b"MMBK"
//!   4..6   version u16 = 1
//!   6..8   depth u16 = 10
//!   8..12  record length u32 = 384
//!
//! record, 384 bytes:
//!   0..8     local receive time, unix ns, i64
//!   8..16    exchange event time, unix ns, i64
//!   16..24   update id, u64
//!   24       flags: bit 0 event time present, bit 1 update id present
//!   25       bid count
//!   26       ask count
//!   27       reserved 0
//!   28..44   exchange name, ASCII, NUL padded
//!   44..60   symbol, ASCII, NUL padded
//!   60..220  10 bids, each f64 price then f64 quantity
//!   220..380 10 asks, same layout
//!   380..384 reserved 0
//! ```
//!
//! A trailing partial record is ignored on read. Encode writes one stack
//! buffer and does not build a JSON document.
//!
//! [`BookLogWriter`] flushes after a configured number of records. It does not
//! `fsync` each update. A crash can lose the records still in the process or
//! operating-system buffer.

use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;

use super::book::{BookError, BookRecord, BookStamp, Instrument, Level, BOOK_DEPTH};

/// Bytes in a book-log header.
pub const BOOK_HEADER_LEN: usize = 12;
/// Bytes in one book record.
pub const BOOK_RECORD_LEN: usize = 384;

const BOOK_MAGIC: &[u8; 4] = b"MMBK";
const VERSION: u16 = 1;
const FLAG_EVENT: u8 = 0b0000_0001;
const FLAG_UPDATE: u8 = 0b0000_0010;

/// Failure while reading a log.
#[derive(Debug)]
pub enum LogError {
    /// The reader returned an I/O error.
    Io(io::Error),
    /// The file had no header.
    Empty,
    /// The header was shorter than 12 bytes.
    TruncatedHeader,
    /// The magic bytes were not recognized.
    BadMagic,
    /// The version is not supported.
    UnsupportedVersion(u16),
    /// The header depth does not match this build.
    UnexpectedDepth(u16),
    /// The header record length does not match this build.
    UnexpectedRecordLen(u32),
    /// A full record could not be decoded.
    CorruptRecord,
    /// Flush interval was zero.
    InvalidFlush,
}

impl fmt::Display for LogError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "log read failed: {err}"),
            Self::Empty => f.write_str("log is empty"),
            Self::TruncatedHeader => f.write_str("log header is incomplete"),
            Self::BadMagic => f.write_str("log magic does not match"),
            Self::UnsupportedVersion(version) => write!(f, "unsupported log version {version}"),
            Self::UnexpectedDepth(depth) => write!(f, "unexpected book depth {depth}"),
            Self::UnexpectedRecordLen(len) => write!(f, "unexpected record length {len}"),
            Self::CorruptRecord => f.write_str("log record is corrupt"),
            Self::InvalidFlush => f.write_str("flush interval must be at least one record"),
        }
    }
}

impl std::error::Error for LogError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl From<io::Error> for LogError {
    fn from(err: io::Error) -> Self {
        Self::Io(err)
    }
}

/// Length of a book file after dropping a torn final record.
///
/// An empty path is `0`. A non-empty file shorter than the header is an error.
pub fn aligned_book_file_len(file_len: u64) -> Result<u64, LogError> {
    if file_len == 0 {
        return Ok(0);
    }
    let header = BOOK_HEADER_LEN as u64;
    if file_len < header {
        return Err(LogError::TruncatedHeader);
    }
    let record = BOOK_RECORD_LEN as u64;
    let payload = file_len - header;
    Ok(header + payload - payload % record)
}

/// Writes a version-1 book header.
pub fn write_book_header<W: Write>(writer: &mut W) -> io::Result<()> {
    let mut header = [0u8; BOOK_HEADER_LEN];
    encode_header(
        &mut header,
        BOOK_MAGIC,
        BOOK_DEPTH as u16,
        BOOK_RECORD_LEN as u32,
    );
    writer.write_all(&header)
}

/// Checks a book header.
pub fn decode_book_header(bytes: &[u8]) -> Result<(), LogError> {
    decode_header(bytes, BOOK_MAGIC, BOOK_DEPTH as u16, BOOK_RECORD_LEN as u32)
}

/// Encodes one book record into `dest`. `dest` must be [`BOOK_RECORD_LEN`] bytes.
pub fn encode_book_record(record: &BookRecord, dest: &mut [u8]) -> Result<(), LogError> {
    if dest.len() != BOOK_RECORD_LEN {
        return Err(LogError::UnexpectedRecordLen(dest.len() as u32));
    }
    dest.fill(0);
    let stamp = record.stamp();
    write_i64(dest, 0, stamp.local_receive_ns);
    let mut flags = 0u8;
    if let Some(event_ns) = stamp.exchange_event_ns {
        write_i64(dest, 8, event_ns);
        flags |= FLAG_EVENT;
    }
    if let Some(update_id) = stamp.update_id {
        write_u64(dest, 16, update_id);
        flags |= FLAG_UPDATE;
    }
    dest[24] = flags;
    dest[25] = record.bid_count();
    dest[26] = record.ask_count();
    dest[28..44].copy_from_slice(record.instrument().exchange_bytes());
    dest[44..60].copy_from_slice(record.instrument().symbol_bytes());
    write_levels(dest, 60, record.bids());
    write_levels(dest, 220, record.asks());
    Ok(())
}

/// Reads a book log. A short tail after the last full record is discarded.
pub fn read_book_log<R: Read>(reader: &mut R) -> Result<Vec<BookRecord>, LogError> {
    let mut header = [0u8; BOOK_HEADER_LEN];
    match read_full(reader, &mut header)? {
        ReadFull::Eof => return Err(LogError::Empty),
        ReadFull::Partial => return Err(LogError::TruncatedHeader),
        ReadFull::Full => decode_book_header(&header)?,
    }
    let mut records = Vec::new();
    let mut raw = [0u8; BOOK_RECORD_LEN];
    while let ReadFull::Full = read_full(reader, &mut raw)? {
        records.push(decode_book_record(&raw)?);
    }
    Ok(records)
}

/// Appends fixed-size book records and flushes in batches.
pub struct BookLogWriter {
    file: File,
    encoded: [u8; BOOK_RECORD_LEN],
    pending: u32,
    flush_every: u32,
}

impl BookLogWriter {
    /// Opens or creates a book log. `flush_every` must be at least 1.
    pub fn open(path: &Path, flush_every: u32) -> Result<Self, LogError> {
        if flush_every == 0 {
            return Err(LogError::InvalidFlush);
        }
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)?;
        prepare_book_file(&mut file)?;
        Ok(Self {
            file,
            encoded: [0; BOOK_RECORD_LEN],
            pending: 0,
            flush_every,
        })
    }

    /// Encodes `record` into the reused buffer and appends it.
    pub fn write(&mut self, record: &BookRecord) -> Result<(), LogError> {
        encode_book_record(record, &mut self.encoded)?;
        self.file.write_all(&self.encoded)?;
        self.pending += 1;
        if self.pending >= self.flush_every {
            self.file.flush()?;
            self.pending = 0;
        }
        Ok(())
    }
}

impl Drop for BookLogWriter {
    fn drop(&mut self) {
        let _ = self.file.flush();
    }
}

fn prepare_book_file(file: &mut File) -> Result<(), LogError> {
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

fn encode_header(dest: &mut [u8], magic: &[u8; 4], depth: u16, record_len: u32) {
    dest[0..4].copy_from_slice(magic);
    dest[4..6].copy_from_slice(&VERSION.to_le_bytes());
    dest[6..8].copy_from_slice(&depth.to_le_bytes());
    dest[8..12].copy_from_slice(&record_len.to_le_bytes());
}

fn decode_header(
    bytes: &[u8],
    magic: &[u8; 4],
    depth: u16,
    record_len: u32,
) -> Result<(), LogError> {
    if bytes.len() < BOOK_HEADER_LEN {
        return Err(LogError::TruncatedHeader);
    }
    if &bytes[0..4] != magic {
        return Err(LogError::BadMagic);
    }
    let version = u16::from_le_bytes(copy_array(bytes, 4)?);
    if version != VERSION {
        return Err(LogError::UnsupportedVersion(version));
    }
    let stored_depth = u16::from_le_bytes(copy_array(bytes, 6)?);
    if stored_depth != depth {
        return Err(LogError::UnexpectedDepth(stored_depth));
    }
    let stored_len = u32::from_le_bytes(copy_array(bytes, 8)?);
    if stored_len != record_len {
        return Err(LogError::UnexpectedRecordLen(stored_len));
    }
    Ok(())
}

fn decode_book_record(raw: &[u8]) -> Result<BookRecord, LogError> {
    let flags = *raw.get(24).ok_or(LogError::CorruptRecord)?;
    let exchange_event_ns = if flags & FLAG_EVENT != 0 {
        Some(read_i64(raw, 8)?)
    } else {
        None
    };
    let update_id = if flags & FLAG_UPDATE != 0 {
        Some(read_u64(raw, 16)?)
    } else {
        None
    };
    let bid_count = *raw.get(25).ok_or(LogError::CorruptRecord)?;
    let ask_count = *raw.get(26).ok_or(LogError::CorruptRecord)?;
    let instrument = Instrument::from_bytes(copy_array(raw, 28)?, copy_array(raw, 44)?)
        .map_err(|_| LogError::CorruptRecord)?;
    let record = BookRecord::try_new(
        instrument,
        BookStamp {
            local_receive_ns: read_i64(raw, 0)?,
            exchange_event_ns,
            update_id,
        },
        read_levels(raw, 60)?,
        bid_count,
        read_levels(raw, 220)?,
        ask_count,
    );
    record.map_err(corrupt)
}

fn write_levels(dest: &mut [u8], offset: usize, levels: &[Level; BOOK_DEPTH]) {
    for (index, level) in levels.iter().enumerate() {
        let at = offset + index * 16;
        write_f64(dest, at, level.price);
        write_f64(dest, at + 8, level.quantity);
    }
}

fn read_levels(raw: &[u8], offset: usize) -> Result<[Level; BOOK_DEPTH], LogError> {
    let mut levels = [Level::EMPTY; BOOK_DEPTH];
    for (index, level) in levels.iter_mut().enumerate() {
        let at = offset + index * 16;
        *level = Level {
            price: read_f64(raw, at)?,
            quantity: read_f64(raw, at + 8)?,
        };
    }
    Ok(levels)
}

fn corrupt(_: BookError) -> LogError {
    LogError::CorruptRecord
}

fn write_u64(dest: &mut [u8], offset: usize, value: u64) {
    dest[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn write_i64(dest: &mut [u8], offset: usize, value: i64) {
    dest[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn write_f64(dest: &mut [u8], offset: usize, value: f64) {
    write_u64(dest, offset, value.to_bits());
}

fn read_u64(raw: &[u8], offset: usize) -> Result<u64, LogError> {
    Ok(u64::from_le_bytes(copy_array(raw, offset)?))
}

fn read_i64(raw: &[u8], offset: usize) -> Result<i64, LogError> {
    Ok(i64::from_le_bytes(copy_array(raw, offset)?))
}

fn read_f64(raw: &[u8], offset: usize) -> Result<f64, LogError> {
    Ok(f64::from_bits(read_u64(raw, offset)?))
}

fn copy_array<const N: usize>(bytes: &[u8], offset: usize) -> Result<[u8; N], LogError> {
    let slice = bytes
        .get(offset..offset + N)
        .ok_or(LogError::CorruptRecord)?;
    let mut raw = [0u8; N];
    raw.copy_from_slice(slice);
    Ok(raw)
}

enum ReadFull {
    Eof,
    Partial,
    Full,
}

fn read_full<R: Read>(reader: &mut R, buf: &mut [u8]) -> Result<ReadFull, LogError> {
    let mut filled = 0;
    while filled < buf.len() {
        match reader.read(&mut buf[filled..]) {
            Ok(0) if filled == 0 => return Ok(ReadFull::Eof),
            Ok(0) => return Ok(ReadFull::Partial),
            Ok(count) => filled += count,
            Err(err) if err.kind() == io::ErrorKind::Interrupted => continue,
            Err(err) => return Err(LogError::Io(err)),
        }
    }
    Ok(ReadFull::Full)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::marketdata::{Instrument, Level, BOOK_DEPTH};

    fn sample(update_id: u64, event_ns: Option<i64>) -> BookRecord {
        let mut bids = [Level::EMPTY; BOOK_DEPTH];
        let mut asks = [Level::EMPTY; BOOK_DEPTH];
        bids[0] = Level {
            price: 100.5,
            quantity: 1.25,
        };
        bids[1] = Level {
            price: 100.4,
            quantity: 2.0,
        };
        asks[0] = Level {
            price: 100.7,
            quantity: 0.5,
        };
        asks[1] = Level {
            price: 100.8,
            quantity: 3.0,
        };
        BookRecord::try_new(
            Instrument::new("binance", "BTCUSDT").unwrap(),
            BookStamp {
                local_receive_ns: 1_700_000_000_000_000_000,
                exchange_event_ns: event_ns,
                update_id: Some(update_id),
            },
            bids,
            2,
            asks,
            2,
        )
        .unwrap()
    }

    #[test]
    fn book_log_roundtrip_preserves_timestamps_and_drops_a_torn_tail() {
        let first = sample(160, Some(1_672_515_782_136_000_000));
        let second = sample(161, None);
        let mut bytes = Vec::new();
        write_book_header(&mut bytes).unwrap();
        for record in [&first, &second] {
            let mut raw = [0u8; BOOK_RECORD_LEN];
            encode_book_record(record, &mut raw).unwrap();
            bytes.extend_from_slice(&raw);
        }
        bytes.extend_from_slice(&[1, 2, 3, 4, 5]);

        let decoded = read_book_log(&mut bytes.as_slice()).unwrap();
        assert_eq!(decoded, vec![first, second]);
        assert_eq!(
            decoded[0].stamp().exchange_event_ns,
            Some(1_672_515_782_136_000_000)
        );
        assert_eq!(decoded[1].stamp().exchange_event_ns, None);
        assert_eq!(decoded[0].stamp().update_id, Some(160));
        assert_eq!(
            decoded[0].stamp().local_receive_ns,
            first.stamp().local_receive_ns
        );
    }

    #[test]
    fn torn_file_length_drops_only_the_partial_tail() {
        let len = BOOK_HEADER_LEN as u64 + BOOK_RECORD_LEN as u64 + 5;
        assert_eq!(
            aligned_book_file_len(len).unwrap(),
            (BOOK_HEADER_LEN + BOOK_RECORD_LEN) as u64
        );
        assert_eq!(aligned_book_file_len(0).unwrap(), 0);
    }

    #[test]
    fn bad_magic_is_rejected() {
        let mut header = [0u8; BOOK_HEADER_LEN];
        header.copy_from_slice(b"XXXX........");
        assert!(matches!(
            decode_book_header(&header),
            Err(LogError::BadMagic)
        ));
    }
}
