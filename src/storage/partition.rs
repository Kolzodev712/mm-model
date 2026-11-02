//! UTC time partitions for one symbol stream.
//!
//! Identity is the UTC hour of `local_receive_ns`. Completed partitions are
//! immutable. The collector writes only the open hour.

use std::path::{Path, PathBuf};

use crate::domain::book::BookRecord;

pub const PARTITION_SCHEMA_VERSION: u32 = 2;

/// Lifecycle of one hour of captured books.
///
/// Training may read only [`PartitionState::Published`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum PartitionState {
    Open,
    Closed,
    Validated,
    Converted,
    Published,
    Failed,
}

impl PartitionState {
    #[must_use]
    pub const fn is_published(self) -> bool {
        matches!(self, Self::Published)
    }
}

/// Stable id for one UTC hour of one symbol.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PartitionId {
    pub exchange: String,
    pub symbol: String,
    pub day: String,
    pub hour: u8,
}

impl PartitionId {
    /// Hour bucket in UTC. `ns` is unix nanoseconds.
    pub fn from_receive_ns(exchange: &str, symbol: &str, ns: i64) -> Result<Self, PartitionError> {
        if ns < 0 {
            return Err(PartitionError::Time);
        }
        let secs = ns / 1_000_000_000;
        let days = secs.div_euclid(86_400);
        let hour = ((secs.div_euclid(3_600)).rem_euclid(24)) as u8;
        let (year, month, day) = civil_from_unix_days(days).ok_or(PartitionError::Time)?;
        Ok(Self {
            exchange: exchange.to_string(),
            symbol: symbol.to_string(),
            day: format!("{year:04}-{month:02}-{day:02}"),
            hour,
        })
    }

    #[must_use]
    pub fn raw_relative(&self) -> PathBuf {
        PathBuf::from("raw")
            .join(&self.exchange)
            .join(&self.symbol)
            .join(&self.day)
            .join(format!("{:02}.mmbk", self.hour))
    }

    #[must_use]
    pub fn parquet_relative(&self) -> PathBuf {
        PathBuf::from("books")
            .join(&self.exchange)
            .join(&self.symbol)
            .join(&self.day)
            .join(format!("{:02}.parquet", self.hour))
    }

    pub fn raw_path(&self, root: &Path) -> PathBuf {
        root.join(self.raw_relative())
    }

    pub fn parquet_path(&self, root: &Path) -> PathBuf {
        root.join(self.parquet_relative())
    }
}

/// Catalog row for one partition.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PartitionRecord {
    pub id: PartitionId,
    pub state: PartitionState,
    pub raw_count: u64,
    pub parquet_count: Option<u64>,
    pub schema_version: u32,
    pub start_ns: i64,
    pub end_ns: i64,
    pub error: Option<String>,
}

impl PartitionRecord {
    pub fn open(id: PartitionId, start_ns: i64) -> Self {
        Self {
            id,
            state: PartitionState::Open,
            raw_count: 0,
            parquet_count: None,
            schema_version: PARTITION_SCHEMA_VERSION,
            start_ns,
            end_ns: start_ns,
            error: None,
        }
    }

    /// Inclusive range overlap with `[from_ns, to_ns]`.
    #[must_use]
    pub fn overlaps(&self, from_ns: i64, to_ns: i64) -> bool {
        self.start_ns <= to_ns && self.end_ns >= from_ns
    }
}

/// A book belongs in the partition that contains its receive time.
pub fn partition_for(
    exchange: &str,
    symbol: &str,
    book: &BookRecord,
) -> Result<PartitionId, PartitionError> {
    PartitionId::from_receive_ns(exchange, symbol, book.stamp().local_receive_ns)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartitionError {
    Time,
}

impl std::fmt::Display for PartitionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("book timestamp is not a valid UTC partition key")
    }
}

impl std::error::Error for PartitionError {}

/// Howard Hinnant's civil-from-days, days since 1970-01-01.
fn civil_from_unix_days(days: i64) -> Option<(i32, u32, u32)> {
    let z = days.checked_add(719_468)?;
    let era = if z >= 0 { z } else { z.checked_sub(146_096)? } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = i64::try_from(yoe).ok()? + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if month <= 2 { y + 1 } else { y };
    Some((
        i32::try_from(year).ok()?,
        u32::try_from(month).ok()?,
        u32::try_from(day).ok()?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn utc_hour_is_stable() {
        // 2026-11-02 13:00:00 UTC
        let ns = 1_793_624_400_000_000_000;
        let id = PartitionId::from_receive_ns("binance", "BTCUSDT", ns).unwrap();
        assert_eq!(id.day, "2026-11-02");
        assert_eq!(id.hour, 13);
        assert_eq!(
            id.raw_relative(),
            PathBuf::from("raw/binance/BTCUSDT/2026-11-02/13.mmbk")
        );
    }
}
