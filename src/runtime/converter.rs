//! Turns one closed MMBK partition into a published Parquet file.
//!
//! The collector does not call this. A crash after the Parquet file exists but
//! before the catalog update is repaired by running conversion again.

use std::fs;
use std::sync::{Mutex, MutexGuard};

use crate::storage::manifest::{Catalog, ManifestError};
use crate::storage::parquet::{read_books, write_books, ParquetConfig, ParquetError};
use crate::storage::partition::{PartitionId, PartitionState};
use crate::storage::raw::read_book_log;

macro_rules! runtime_info {
    ($($arg:tt)*) => {{
        #[cfg(feature = "observe")]
        tracing::info!($($arg)*);
    }};
}

pub fn publish_partition(
    catalog: &Mutex<Catalog>,
    id: &PartitionId,
    config: ParquetConfig,
) -> Result<(), ConvertError> {
    let catalog_guard = lock_catalog(catalog);
    let record = catalog_guard
        .get(id)
        .cloned()
        .ok_or(ConvertError::Missing)?;
    if record.state == PartitionState::Open {
        return Err(ConvertError::StillOpen);
    }
    if record.state == PartitionState::Published {
        let expected = id.parquet_path(catalog_guard.root());
        if expected.is_file() {
            return Ok(());
        }
        if !id.raw_path(catalog_guard.root()).is_file() {
            return Err(ConvertError::Ambiguous);
        }
    }
    let raw_path = id.raw_path(catalog_guard.root());
    drop(catalog_guard);
    let books = match fs::File::open(raw_path) {
        Ok(mut file) => match read_book_log(&mut file) {
            Ok(books) if books.len() as u64 == record.raw_count => books,
            _ => {
                lock_catalog(catalog).mark(
                    id,
                    PartitionState::Failed,
                    Some("raw log could not be read".into()),
                )?;
                return Err(ConvertError::Raw);
            }
        },
        Err(_) => {
            lock_catalog(catalog).mark(
                id,
                PartitionState::Failed,
                Some("raw log could not be opened".into()),
            )?;
            return Err(ConvertError::Raw);
        }
    };
    if books.iter().any(|book| {
        book.instrument().exchange() != id.exchange
            || book.instrument().symbol() != id.symbol
            || PartitionId::from_receive_ns(&id.exchange, &id.symbol, book.stamp().local_receive_ns)
                .ok()
                .is_none_or(|book_id| book_id.day != id.day || book_id.hour != id.hour)
    }) {
        lock_catalog(catalog).mark(
            id,
            PartitionState::Failed,
            Some("raw records do not match the partition".into()),
        )?;
        return Err(ConvertError::Raw);
    }
    lock_catalog(catalog).mark(id, PartitionState::Validated, None)?;
    let root = lock_catalog(catalog).root().to_path_buf();
    let final_path = id.parquet_path(&root);
    let tmp_path = final_path.with_extension("parquet.tmp");
    if let Err(err) = write_books(
        &tmp_path,
        config,
        &id.exchange,
        &id.symbol,
        record.start_ns,
        record.end_ns,
        &books,
    ) {
        let _ = fs::remove_file(&tmp_path);
        lock_catalog(catalog).mark(id, PartitionState::Failed, Some(err.to_string()))?;
        return Err(ConvertError::Parquet(err));
    }
    let written = match read_books(&tmp_path, &id.exchange, &id.symbol) {
        Ok(rows) => rows,
        Err(err) => {
            let _ = fs::remove_file(&tmp_path);
            lock_catalog(catalog).mark(id, PartitionState::Failed, Some(err.to_string()))?;
            return Err(ConvertError::Parquet(err));
        }
    };
    if written.len() != books.len() {
        let _ = fs::remove_file(&tmp_path);
        lock_catalog(catalog).mark(
            id,
            PartitionState::Failed,
            Some("parquet row count mismatch".into()),
        )?;
        return Err(ConvertError::Count);
    }
    if let Some(parent) = final_path.parent() {
        if let Err(err) = fs::create_dir_all(parent) {
            lock_catalog(catalog).mark(id, PartitionState::Failed, Some(err.to_string()))?;
            return Err(ConvertError::Io);
        }
    }
    if let Err(err) = fs::rename(&tmp_path, &final_path) {
        lock_catalog(catalog).mark(id, PartitionState::Failed, Some(err.to_string()))?;
        return Err(ConvertError::Io);
    }
    let mut catalog_guard = lock_catalog(catalog);
    let mut published = catalog_guard
        .get(id)
        .cloned()
        .ok_or(ConvertError::Missing)?;
    published.state = PartitionState::Converted;
    published.parquet_count = Some(written.len() as u64);
    published.raw_count = books.len() as u64;
    published.error = None;
    catalog_guard.upsert(published);
    catalog_guard.save()?;
    catalog_guard.mark(id, PartitionState::Published, None)?;
    runtime_info!("parquet partition published");
    Ok(())
}

fn lock_catalog(catalog: &Mutex<Catalog>) -> MutexGuard<'_, Catalog> {
    catalog.lock().unwrap_or_else(|err| err.into_inner())
}

#[derive(Debug)]
pub enum ConvertError {
    Missing,
    StillOpen,
    Ambiguous,
    Raw,
    Count,
    Io,
    Manifest(ManifestError),
    Parquet(ParquetError),
}

impl From<ManifestError> for ConvertError {
    fn from(value: ManifestError) -> Self {
        Self::Manifest(value)
    }
}

impl std::fmt::Display for ConvertError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Missing => f.write_str("partition is not in the catalog"),
            Self::StillOpen => f.write_str("open partitions cannot be converted"),
            Self::Ambiguous => {
                f.write_str("catalog says published but the parquet file is missing")
            }
            Self::Raw => f.write_str("raw partition could not be validated"),
            Self::Count => f.write_str("parquet row count does not match the raw log"),
            Self::Io => f.write_str("parquet file could not be published"),
            Self::Manifest(err) => write!(f, "{err}"),
            Self::Parquet(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for ConvertError {}
