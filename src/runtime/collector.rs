//! Appends books to the open UTC-hour MMBK partition.
//!
//! Rotation closes that partition in the catalog. Conversion is a separate step.
//! The collector does not know the Parquet column layout.

use std::fs;
use std::sync::{Arc, Mutex, MutexGuard};

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
    catalog: Arc<Mutex<Catalog>>,
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
            catalog: Arc::new(Mutex::new(catalog)),
            exchange: exchange.into(),
            symbol: symbol.into(),
            flush_every: flush_every.max(1),
            open: None,
        }
    }

    pub fn catalog_mutex(&self) -> &Mutex<Catalog> {
        &self.catalog
    }

    pub fn shared_catalog(&self) -> Arc<Mutex<Catalog>> {
        Arc::clone(&self.catalog)
    }

    fn guard(&self) -> MutexGuard<'_, Catalog> {
        self.catalog.lock().unwrap_or_else(|err| err.into_inner())
    }

    /// Writes one book. Returns the id of a partition that just closed, if any.
    pub fn push(&mut self, book: &BookRecord) -> Result<Option<PartitionId>, CollectError> {
        if book.instrument().exchange() != self.exchange
            || book.instrument().symbol() != self.symbol
        {
            return Err(CollectError::Instrument);
        }
        let id = partition_for(&self.exchange, &self.symbol, book)?;
        if let Some(open) = self.open.as_ref() {
            if (id.day.as_str(), id.hour) < (open.id.day.as_str(), open.id.hour) {
                return Err(CollectError::Rewound);
            }
        }
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

    /// Flushes the open hour and leaves it `Open`. Shutdown must not publish a partial hour.
    pub fn finish(&mut self) -> Result<(), CollectError> {
        if let Some(open) = self.open.as_mut() {
            open.writer.flush()?;
        }
        Ok(())
    }

    /// Closes the active hour so a test can convert it. Production shutdown uses [`Self::finish`].
    pub fn seal(&mut self) -> Result<Option<PartitionId>, CollectError> {
        self.close_current()
    }

    fn open_partition(&mut self, id: PartitionId, start_ns: i64) -> Result<(), CollectError> {
        let mut catalog = self.guard();
        if let Some(existing) = catalog.get(&id) {
            if existing.state != PartitionState::Open {
                return Err(CollectError::Immutable);
            }
        }
        let path = id.raw_path(catalog.root());
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|_| CollectError::Io)?;
        }
        let writer = BookLogWriter::open(&path, self.flush_every)?;
        let prior = if path.is_file() {
            let mut file = fs::File::open(&path).map_err(|_| CollectError::Io)?;
            read_book_log(&mut file)?.len() as u64
        } else {
            0
        };
        let (start_ns, end_ns) = if let Some(existing) = catalog.get(&id) {
            (
                existing.start_ns.min(start_ns),
                existing.end_ns.max(start_ns),
            )
        } else {
            let mut record = PartitionRecord::open(id.clone(), catalog.root(), start_ns);
            record.raw_path = path.display().to_string();
            catalog.upsert(record);
            catalog.save()?;
            (start_ns, start_ns)
        };
        drop(catalog);
        self.open = Some(OpenPartition {
            id,
            writer,
            count: prior,
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
        let id = open.id.clone();
        let raw_count = open.count;
        let start_ns = open.start_ns;
        let end_ns = open.end_ns;
        let path = {
            let catalog = self.guard();
            id.raw_path(catalog.root())
        };
        open.writer.close()?;
        let mut catalog = self.guard();
        if let Some(row) = catalog.get(&id).cloned() {
            let mut row = row;
            row.state = PartitionState::Closed;
            row.raw_count = raw_count;
            row.start_ns = start_ns;
            row.end_ns = end_ns;
            row.raw_path = path.display().to_string();
            catalog.upsert(row);
            catalog.save()?;
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
    /// The book is not the instrument this collector was configured to write.
    Instrument,
    /// A timestamp moved into an earlier hour. The open hour was left untouched.
    Rewound,
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
            Self::Instrument => f.write_str("book instrument does not match the collector"),
            Self::Rewound => f.write_str("book timestamp moved into an earlier partition"),
        }
    }
}

impl std::error::Error for CollectError {}
