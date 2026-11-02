//! Derived Parquet books. Rebuild them from MMBK if this file is wrong.
//!
//! Conversion is a batch over a closed partition. It is not on the websocket path.
//! Unused price levels are null, never a fake zero price.

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow_array::builder::{Float64Builder, Int64Builder, UInt64Builder, UInt8Builder};
use arrow_array::{
    Array, Float64Array, Int64Array, RecordBatch, StringArray, UInt64Array, UInt8Array,
};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;

use crate::data::training::history::{BookRange, HistoricalBooks};
use crate::domain::book::{BookRecord, BookStamp, Instrument, Level, BOOK_DEPTH};
use crate::storage::manifest::Catalog;

pub const BOOK_PARQUET_SCHEMA: u32 = 1;

/// Typed Parquet settings. Not scattered magic numbers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParquetConfig {
    pub zstd_level: i32,
    pub schema_version: u32,
}

impl Default for ParquetConfig {
    fn default() -> Self {
        Self {
            zstd_level: 3,
            schema_version: BOOK_PARQUET_SCHEMA,
        }
    }
}

pub fn write_books(
    path: &Path,
    config: ParquetConfig,
    exchange: &str,
    symbol: &str,
    partition_start_ns: i64,
    partition_end_ns: i64,
    books: &[BookRecord],
) -> Result<(), ParquetError> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|_| ParquetError::Io)?;
    }
    let schema = book_schema(
        config.schema_version,
        exchange,
        symbol,
        partition_start_ns,
        partition_end_ns,
        books.len() as u64,
    );
    let batch = batch_from_books(&schema, exchange, symbol, books)?;
    let level = ZstdLevel::try_new(config.zstd_level).map_err(|_| ParquetError::Config)?;
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(level))
        .build();
    let file = File::create(path).map_err(|_| ParquetError::Io)?;
    let mut writer =
        ArrowWriter::try_new(file, schema, Some(props)).map_err(|_| ParquetError::Write)?;
    writer.write(&batch).map_err(|_| ParquetError::Write)?;
    writer.close().map_err(|_| ParquetError::Write)?;
    Ok(())
}

pub fn read_books(
    path: &Path,
    expect_exchange: &str,
    expect_symbol: &str,
) -> Result<Vec<BookRecord>, ParquetError> {
    let file = File::open(path).map_err(|_| ParquetError::Io)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(|_| ParquetError::Read)?;
    let meta = builder.schema().metadata();
    let version = meta.get("schema_version").ok_or(ParquetError::Schema)?;
    if version != &BOOK_PARQUET_SCHEMA.to_string() {
        return Err(ParquetError::Schema);
    }
    let depth = meta.get("book_depth").ok_or(ParquetError::Schema)?;
    if depth != &BOOK_DEPTH.to_string() {
        return Err(ParquetError::Schema);
    }
    let source_format = meta
        .get("source_format_version")
        .ok_or(ParquetError::Schema)?;
    if source_format != "1" {
        return Err(ParquetError::Schema);
    }
    let expected_rows = meta
        .get("row_count")
        .ok_or(ParquetError::Schema)?
        .parse::<u64>()
        .map_err(|_| ParquetError::Schema)?;
    let exchange = meta.get("exchange").ok_or(ParquetError::Schema)?;
    let symbol = meta.get("symbol").ok_or(ParquetError::Schema)?;
    if exchange != expect_exchange || symbol != expect_symbol {
        return Err(ParquetError::Identity);
    }
    let reader = builder.build().map_err(|_| ParquetError::Read)?;
    let mut books = Vec::new();
    for batch in reader {
        let batch = batch.map_err(|_| ParquetError::Read)?;
        books.extend(books_from_batch(&batch)?);
    }
    if books.len() as u64 != expected_rows {
        return Err(ParquetError::Schema);
    }
    Ok(books)
}

/// Published Parquet partitions known to the catalog, scanned one file at a time.
pub struct ParquetHistory<'a> {
    catalog: &'a Catalog,
}

impl<'a> ParquetHistory<'a> {
    #[must_use]
    pub fn new(catalog: &'a Catalog) -> Self {
        Self { catalog }
    }
}

impl HistoricalBooks for ParquetHistory<'_> {
    type Error = ParquetError;

    fn for_each<F>(&self, range: &BookRange, mut visit: F) -> Result<u64, Self::Error>
    where
        F: FnMut(&BookRecord),
    {
        let mut count = 0u64;
        let partitions = self.catalog.published_overlapping(
            &range.exchange,
            &range.symbol,
            range.from_ns,
            range.to_ns,
        );
        for partition in partitions {
            let Some(path) = partition.parquet_path.as_deref() else {
                return Err(ParquetError::Missing);
            };
            let books = read_books(Path::new(path), &range.exchange, &range.symbol)?;
            for book in &books {
                let ts = book.stamp().local_receive_ns;
                if ts >= range.from_ns && ts <= range.to_ns {
                    visit(book);
                    count += 1;
                }
            }
        }
        Ok(count)
    }
}

fn book_schema(
    version: u32,
    exchange: &str,
    symbol: &str,
    start_ns: i64,
    end_ns: i64,
    rows: u64,
) -> Arc<Schema> {
    let mut fields = vec![
        field("exchange", DataType::Utf8, false),
        field("symbol", DataType::Utf8, false),
        field("local_receive_ns", DataType::Int64, false),
        field("exchange_event_ns", DataType::Int64, true),
        field("update_id", DataType::UInt64, true),
        field("bid_count", DataType::UInt8, false),
        field("ask_count", DataType::UInt8, false),
    ];
    for side in ["bid", "ask"] {
        for index in 1..=BOOK_DEPTH {
            fields.push(field(
                &format!("{side}_{index:02}_price"),
                DataType::Float64,
                true,
            ));
            fields.push(field(
                &format!("{side}_{index:02}_qty"),
                DataType::Float64,
                true,
            ));
        }
    }
    let mut metadata = std::collections::HashMap::new();
    metadata.insert("schema_version".into(), version.to_string());
    metadata.insert("exchange".into(), exchange.to_string());
    metadata.insert("symbol".into(), symbol.to_string());
    metadata.insert("book_depth".into(), BOOK_DEPTH.to_string());
    metadata.insert("partition_start_ns".into(), start_ns.to_string());
    metadata.insert("partition_end_ns".into(), end_ns.to_string());
    metadata.insert("source_format_version".into(), "1".into());
    metadata.insert("row_count".into(), rows.to_string());
    Arc::new(Schema::new_with_metadata(fields, metadata))
}

fn field(name: &str, ty: DataType, nullable: bool) -> Field {
    Field::new(name, ty, nullable)
}

fn batch_from_books(
    schema: &Schema,
    exchange: &str,
    symbol: &str,
    books: &[BookRecord],
) -> Result<RecordBatch, ParquetError> {
    let n = books.len();
    let exchanges = StringArray::from(vec![exchange.to_string(); n]);
    let symbols = StringArray::from(vec![symbol.to_string(); n]);
    let mut local = Int64Builder::with_capacity(n);
    let mut event = Int64Builder::with_capacity(n);
    let mut update = UInt64Builder::with_capacity(n);
    let mut bid_count = UInt8Builder::with_capacity(n);
    let mut ask_count = UInt8Builder::with_capacity(n);
    let mut bid_px = level_builders(n);
    let mut bid_qty = level_builders(n);
    let mut ask_px = level_builders(n);
    let mut ask_qty = level_builders(n);
    for book in books {
        local.append_value(book.stamp().local_receive_ns);
        match book.stamp().exchange_event_ns {
            Some(value) => event.append_value(value),
            None => event.append_null(),
        }
        match book.stamp().update_id {
            Some(value) => update.append_value(value),
            None => update.append_null(),
        }
        bid_count.append_value(book.bid_count());
        ask_count.append_value(book.ask_count());
        push_levels(&mut bid_px, &mut bid_qty, book.bids(), book.bid_count());
        push_levels(&mut ask_px, &mut ask_qty, book.asks(), book.ask_count());
    }
    let mut columns: Vec<arrow_array::ArrayRef> = vec![
        Arc::new(exchanges),
        Arc::new(symbols),
        Arc::new(local.finish()),
        Arc::new(event.finish()),
        Arc::new(update.finish()),
        Arc::new(bid_count.finish()),
        Arc::new(ask_count.finish()),
    ];
    for index in 0..BOOK_DEPTH {
        columns.push(Arc::new(bid_px[index].finish()));
        columns.push(Arc::new(bid_qty[index].finish()));
    }
    for index in 0..BOOK_DEPTH {
        columns.push(Arc::new(ask_px[index].finish()));
        columns.push(Arc::new(ask_qty[index].finish()));
    }
    RecordBatch::try_new(Arc::new(schema.clone()), columns).map_err(|_| ParquetError::Write)
}

fn level_builders(n: usize) -> Vec<Float64Builder> {
    (0..BOOK_DEPTH)
        .map(|_| Float64Builder::with_capacity(n))
        .collect()
}

fn push_levels(
    prices: &mut [Float64Builder],
    qtys: &mut [Float64Builder],
    levels: &[Level; BOOK_DEPTH],
    count: u8,
) {
    for index in 0..BOOK_DEPTH {
        if index < count as usize {
            prices[index].append_value(levels[index].price);
            qtys[index].append_value(levels[index].quantity);
        } else {
            prices[index].append_null();
            qtys[index].append_null();
        }
    }
}

fn books_from_batch(batch: &RecordBatch) -> Result<Vec<BookRecord>, ParquetError> {
    let exchange = string_col(batch, 0)?;
    let symbol = string_col(batch, 1)?;
    let local = int_col(batch, 2)?;
    let event = int_col(batch, 3)?;
    let update = uint_col(batch, 4)?;
    let bid_count = byte_col(batch, 5)?;
    let ask_count = byte_col(batch, 6)?;
    let mut books = Vec::with_capacity(batch.num_rows());
    for row in 0..batch.num_rows() {
        let instrument = Instrument::new(exchange.value(row), symbol.value(row))
            .map_err(|_| ParquetError::Book)?;
        let bids = read_side(batch, 7, row, bid_count.value(row))?;
        let asks = read_side(batch, 7 + BOOK_DEPTH * 2, row, ask_count.value(row))?;
        let stamp = BookStamp {
            local_receive_ns: local.value(row),
            exchange_event_ns: if event.is_null(row) {
                None
            } else {
                Some(event.value(row))
            },
            update_id: if update.is_null(row) {
                None
            } else {
                Some(update.value(row))
            },
        };
        books.push(
            BookRecord::try_new(
                instrument,
                stamp,
                bids,
                bid_count.value(row),
                asks,
                ask_count.value(row),
            )
            .map_err(|_| ParquetError::Book)?,
        );
    }
    Ok(books)
}

fn read_side(
    batch: &RecordBatch,
    start: usize,
    row: usize,
    count: u8,
) -> Result<[Level; BOOK_DEPTH], ParquetError> {
    let mut levels = [Level::EMPTY; BOOK_DEPTH];
    for (index, level) in levels.iter_mut().enumerate() {
        let price = float_col(batch, start + index * 2)?;
        let qty = float_col(batch, start + index * 2 + 1)?;
        if index < count as usize {
            if price.is_null(row) || qty.is_null(row) {
                return Err(ParquetError::Book);
            }
            *level = Level {
                price: price.value(row),
                quantity: qty.value(row),
            };
        }
    }
    Ok(levels)
}

fn string_col(batch: &RecordBatch, index: usize) -> Result<&StringArray, ParquetError> {
    batch
        .column(index)
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or(ParquetError::Read)
}

fn int_col(batch: &RecordBatch, index: usize) -> Result<&Int64Array, ParquetError> {
    batch
        .column(index)
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or(ParquetError::Read)
}

fn uint_col(batch: &RecordBatch, index: usize) -> Result<&UInt64Array, ParquetError> {
    batch
        .column(index)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .ok_or(ParquetError::Read)
}

fn byte_col(batch: &RecordBatch, index: usize) -> Result<&UInt8Array, ParquetError> {
    batch
        .column(index)
        .as_any()
        .downcast_ref::<UInt8Array>()
        .ok_or(ParquetError::Read)
}

fn float_col(batch: &RecordBatch, index: usize) -> Result<&Float64Array, ParquetError> {
    batch
        .column(index)
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or(ParquetError::Read)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParquetError {
    Io,
    Config,
    Write,
    Read,
    Schema,
    Identity,
    Missing,
    Book,
}

impl std::fmt::Display for ParquetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::Io => "parquet file could not be opened",
            Self::Config => "parquet compression settings were rejected",
            Self::Write => "parquet batch could not be written",
            Self::Read => "parquet batch could not be read",
            Self::Schema => "parquet schema version is not supported",
            Self::Identity => "parquet exchange or symbol does not match the query",
            Self::Missing => "published partition has no parquet path",
            Self::Book => "parquet row is not a valid book",
        };
        f.write_str(message)
    }
}

impl std::error::Error for ParquetError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::book::{BookRecord, BookStamp, Instrument, Level, BOOK_DEPTH};
    use crate::storage::manifest::Catalog;
    use crate::storage::partition::{PartitionId, PartitionRecord, PartitionState};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp() -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("mm-parquet-{nanos}"));
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    fn book(ns: i64, event: Option<i64>, update: Option<u64>, levels: u8) -> BookRecord {
        let mut bids = [Level::EMPTY; BOOK_DEPTH];
        let mut asks = [Level::EMPTY; BOOK_DEPTH];
        for index in 0..levels as usize {
            bids[index] = Level {
                price: 100.0 - index as f64,
                quantity: 1.0 + index as f64,
            };
            asks[index] = Level {
                price: 101.0 + index as f64,
                quantity: 2.0 + index as f64,
            };
        }
        BookRecord::try_new(
            Instrument::new("binance", "BTCUSDT").unwrap(),
            BookStamp {
                local_receive_ns: ns,
                exchange_event_ns: event,
                update_id: update,
            },
            bids,
            levels,
            asks,
            levels,
        )
        .unwrap()
    }

    #[test]
    fn one_book_roundtrips_with_null_event_and_update() {
        let root = temp();
        let path = root.join("book.parquet");
        let original = book(10, None, None, 1);
        write_books(
            &path,
            ParquetConfig::default(),
            "binance",
            "BTCUSDT",
            0,
            10,
            &[original],
        )
        .unwrap();
        let read = read_books(&path, "binance", "BTCUSDT").unwrap();
        assert_eq!(read, vec![original]);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn many_books_keep_order_and_full_depth() {
        let root = temp();
        let path = root.join("books.parquet");
        let rows = vec![
            book(1, Some(1), Some(7), 10),
            book(2, None, Some(8), 3),
            book(3, Some(3), None, 10),
        ];
        write_books(
            &path,
            ParquetConfig::default(),
            "binance",
            "BTCUSDT",
            1,
            3,
            &rows,
        )
        .unwrap();
        assert_eq!(read_books(&path, "binance", "BTCUSDT").unwrap(), rows);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn schema_version_and_identity_are_rejected() {
        let root = temp();
        let path = root.join("bad.parquet");
        let row = book(1, None, None, 1);
        let config = ParquetConfig {
            zstd_level: 3,
            schema_version: 99,
        };
        write_books(&path, config, "binance", "BTCUSDT", 1, 1, &[row]).unwrap();
        assert_eq!(
            read_books(&path, "binance", "BTCUSDT").unwrap_err(),
            ParquetError::Schema
        );
        write_books(
            &path,
            ParquetConfig::default(),
            "binance",
            "BTCUSDT",
            1,
            1,
            &[row],
        )
        .unwrap();
        assert_eq!(
            read_books(&path, "binance", "ETHUSDT").unwrap_err(),
            ParquetError::Identity
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn range_query_uses_published_partitions_only() {
        let root = temp();
        let mut catalog = Catalog::open(&root).unwrap();
        let early_id =
            PartitionId::from_receive_ns("binance", "BTCUSDT", 1_793_624_400_000_000_000).unwrap();
        let late_id = PartitionId::from_receive_ns(
            "binance",
            "BTCUSDT",
            1_793_624_400_000_000_000 + 3_600_000_000_000,
        )
        .unwrap();
        let early = book(1_793_624_400_000_000_000 + 5, None, Some(1), 1);
        let late = book(
            1_793_624_400_000_000_000 + 3_600_000_000_000 + 5,
            None,
            Some(2),
            1,
        );
        let open_book = book(
            1_793_624_400_000_000_000 + 7_200_000_000_000,
            None,
            Some(3),
            1,
        );
        write_books(
            &early_id.parquet_path(&root),
            ParquetConfig::default(),
            "binance",
            "BTCUSDT",
            early.stamp().local_receive_ns,
            early.stamp().local_receive_ns,
            &[early],
        )
        .unwrap();
        write_books(
            &late_id.parquet_path(&root),
            ParquetConfig::default(),
            "binance",
            "BTCUSDT",
            late.stamp().local_receive_ns,
            late.stamp().local_receive_ns,
            &[late],
        )
        .unwrap();
        let mut early_row =
            PartitionRecord::open(early_id.clone(), &root, early.stamp().local_receive_ns);
        early_row.state = PartitionState::Published;
        early_row.parquet_path = Some(early_id.parquet_path(&root).display().to_string());
        early_row.end_ns = early.stamp().local_receive_ns;
        early_row.raw_count = 1;
        early_row.parquet_count = Some(1);
        let mut late_row =
            PartitionRecord::open(late_id.clone(), &root, late.stamp().local_receive_ns);
        late_row.state = PartitionState::Published;
        late_row.parquet_path = Some(late_id.parquet_path(&root).display().to_string());
        late_row.end_ns = late.stamp().local_receive_ns;
        let mut open_row = PartitionRecord::open(
            PartitionId::from_receive_ns("binance", "BTCUSDT", open_book.stamp().local_receive_ns)
                .unwrap(),
            &root,
            open_book.stamp().local_receive_ns,
        );
        open_row.state = PartitionState::Open;
        open_row.parquet_path = Some(root.join("ignored.parquet").display().to_string());
        catalog.upsert(early_row);
        catalog.upsert(late_row);
        catalog.upsert(open_row);
        catalog.save().unwrap();
        let history = ParquetHistory::new(&catalog);
        let mut seen = Vec::new();
        let count = history
            .for_each(
                &crate::data::training::history::BookRange {
                    exchange: "binance".into(),
                    symbol: "BTCUSDT".into(),
                    from_ns: late.stamp().local_receive_ns,
                    to_ns: late.stamp().local_receive_ns,
                },
                |book| seen.push(*book),
            )
            .unwrap();
        assert_eq!(count, 1);
        assert_eq!(seen, vec![late]);
        let _ = std::fs::remove_dir_all(&root);
    }
}
