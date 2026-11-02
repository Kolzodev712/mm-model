//! Appends books to the open UTC-hour MMBK partition.
//!
//! Rotation closes that partition in the catalog. Conversion is a separate step.
//! The collector does not know the Parquet column layout.

use std::fs;

use crate::domain::book::BookRecord;
use crate::storage::manifest::{Catalog, ManifestError};
use crate::storage::partition::{partition_for, PartitionId, PartitionRecord, PartitionState};
use crate::storage::raw::{read_book_log, BookLogWriter};

macro_rules! runtime_info {
    ($($arg:tt)*) => {{
        #[cfg(feature = "observe")]
        tracing::info!($($arg)*);
    }};
}

pub struct Collector {
    catalog: Catalog,
    exchange: String,
    symbol: String,
    flush_every: u32,
    open: Option<OpenPartition>,
}

struct OpenPartition {
    id: PartitionId,
    writer: BookLogWriter,
    count: u64,
    start_ns: i64,
    end_ns: i64,
}

impl Collector {
    pub fn new(
        catalog: Catalog,
        exchange: impl Into<String>,
        symbol: impl Into<String>,
        flush_every: u32,
    ) -> Self {
        Self {
            catalog,
            exchange: exchange.into(),
            symbol: symbol.into(),
            flush_every: flush_every.max(1),
            open: None,
        }
    }

    pub fn catalog_mut(&mut self) -> &mut Catalog {
        &mut self.catalog
    }

    /// Writes one book. Returns the id of a partition that just closed, if any.
    pub fn push(&mut self, book: &BookRecord) -> Result<Option<PartitionId>, CollectError> {
        let id = partition_for(&self.exchange, &self.symbol, book)?;
        let rotated = if self.open.as_ref().is_some_and(|open| open.id != id) {
            self.close_current()?
        } else {
            None
        };
        if self.open.is_none() {
            self.open_partition(id, book.stamp().local_receive_ns)?;
        }
        let open = self.open.as_mut().ok_or(CollectError::Closed)?;
        open.writer.write(book)?;
        open.count += 1;
        open.end_ns = book.stamp().local_receive_ns;
        if book.stamp().local_receive_ns < open.start_ns {
            open.start_ns = book.stamp().local_receive_ns;
        }
        Ok(rotated)
    }

    /// Closes the active partition so it can be converted. It is not published here.
    pub fn finish(&mut self) -> Result<Option<PartitionId>, CollectError> {
        self.close_current()
    }

    fn open_partition(&mut self, id: PartitionId, start_ns: i64) -> Result<(), CollectError> {
        if let Some(existing) = self.catalog.get(&id) {
            if existing.state != PartitionState::Open {
                return Err(CollectError::Immutable);
            }
        }
        let path = id.raw_path(self.catalog.root());
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|_| CollectError::Io)?;
        }
        let writer = BookLogWriter::open(&path, self.flush_every)?;
        let (start_ns, end_ns) = if let Some(existing) = self.catalog.get(&id) {
            (
                existing.start_ns.min(start_ns),
                existing.end_ns.max(start_ns),
            )
        } else {
            let mut record = PartitionRecord::open(id.clone(), self.catalog.root(), start_ns);
            record.raw_path = path.display().to_string();
            self.catalog.upsert(record);
            self.catalog.save()?;
            (start_ns, start_ns)
        };
        self.open = Some(OpenPartition {
            id,
            writer,
            count: 0,
            start_ns,
            end_ns,
        });
        runtime_info!("raw partition opened");
        Ok(())
    }

    fn close_current(&mut self) -> Result<Option<PartitionId>, CollectError> {
        let Some(open) = self.open.take() else {
            return Ok(None);
        };
        let path = open.id.raw_path(self.catalog.root());
        drop(open.writer);
        let id = open.id.clone();
        let mut file = fs::File::open(&path).map_err(|_| CollectError::Io)?;
        let raw_count = read_book_log(&mut file)?.len() as u64;
        if let Some(row) = self.catalog.get(&id).cloned() {
            let mut row = row;
            row.state = PartitionState::Closed;
            row.raw_count = raw_count;
            row.start_ns = open.start_ns;
            row.end_ns = open.end_ns;
            row.raw_path = path.display().to_string();
            self.catalog.upsert(row);
            self.catalog.save()?;
        }
        runtime_info!("raw partition closed");
        Ok(Some(id))
    }
}

#[derive(Debug)]
pub enum CollectError {
    Partition(crate::storage::partition::PartitionError),
    Manifest(ManifestError),
    Log(crate::storage::raw::LogError),
    Io,
    Closed,
    /// A completed partition for this hour already exists and must not be appended.
    Immutable,
}

impl From<crate::storage::partition::PartitionError> for CollectError {
    fn from(value: crate::storage::partition::PartitionError) -> Self {
        Self::Partition(value)
    }
}

impl From<ManifestError> for CollectError {
    fn from(value: ManifestError) -> Self {
        Self::Manifest(value)
    }
}

impl From<crate::storage::raw::LogError> for CollectError {
    fn from(value: crate::storage::raw::LogError) -> Self {
        Self::Log(value)
    }
}

impl std::fmt::Display for CollectError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Partition(err) => write!(f, "{err}"),
            Self::Manifest(err) => write!(f, "{err}"),
            Self::Log(err) => write!(f, "{err}"),
            Self::Io => f.write_str("could not create a partition directory"),
            Self::Closed => f.write_str("collector has no open partition"),
            Self::Immutable => f.write_str("completed raw partition cannot be reopened"),
        }
    }
}

impl std::error::Error for CollectError {}
