//! Durable partition catalog.
//!
//! The file is rewritten through a temporary name and `rename`, so a crash
//! does not leave a half-written catalog that looks complete.

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::partition::{PartitionId, PartitionRecord, PartitionState};

const MANIFEST_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ManifestFile {
    version: u32,
    partitions: Vec<PartitionRecord>,
}

/// Catalog rooted at the data directory.
#[derive(Debug, Clone)]
pub struct Catalog {
    root: PathBuf,
    file: ManifestFile,
}

impl Catalog {
    pub fn open(root: impl Into<PathBuf>) -> Result<Self, ManifestError> {
        let root = root.into();
        fs::create_dir_all(&root).map_err(|_| ManifestError::Io)?;
        let path = manifest_path(&root);
        let file = if path.exists() {
            let text = fs::read_to_string(&path).map_err(|_| ManifestError::Io)?;
            serde_json::from_str(&text).map_err(|_| ManifestError::Corrupt)?
        } else {
            ManifestFile {
                version: MANIFEST_VERSION,
                partitions: Vec::new(),
            }
        };
        if file.version != MANIFEST_VERSION {
            return Err(ManifestError::Version);
        }
        if file.partitions.len() > 100_000 {
            return Err(ManifestError::Corrupt);
        }
        for row in &file.partitions {
            if !component_ok(&row.id.exchange)
                || !component_ok(&row.id.symbol)
                || !day_ok(&row.id.day)
                || row.id.hour > 23
                || stored_path_rejected(&root, &row.raw_path)
                || row
                    .parquet_path
                    .as_ref()
                    .is_some_and(|path| stored_path_rejected(&root, path))
            {
                return Err(ManifestError::Corrupt);
            }
        }
        Ok(Self { root, file })
    }

    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    #[must_use]
    pub fn records(&self) -> &[PartitionRecord] {
        &self.file.partitions
    }

    pub fn upsert(&mut self, record: PartitionRecord) {
        if let Some(existing) = self
            .file
            .partitions
            .iter_mut()
            .find(|row| row.id == record.id)
        {
            *existing = record;
        } else {
            self.file.partitions.push(record);
        }
    }

    #[must_use]
    pub fn get(&self, id: &PartitionId) -> Option<&PartitionRecord> {
        self.file.partitions.iter().find(|row| &row.id == id)
    }

    pub fn published_overlapping(
        &self,
        exchange: &str,
        symbol: &str,
        from_ns: i64,
        to_ns: i64,
    ) -> Vec<PartitionRecord> {
        self.file
            .partitions
            .iter()
            .filter(|row| {
                row.state.is_published()
                    && row.id.exchange == exchange
                    && row.id.symbol == symbol
                    && row.overlaps(from_ns, to_ns)
            })
            .cloned()
            .collect()
    }

    pub fn save(&self) -> Result<(), ManifestError> {
        let path = manifest_path(&self.root);
        let tmp = self.root.join("manifest.json.tmp");
        let text = serde_json::to_string_pretty(&self.file).map_err(|_| ManifestError::Corrupt)?;
        {
            let mut file = fs::File::create(&tmp).map_err(|_| ManifestError::Io)?;
            use std::io::Write;
            file.write_all(text.as_bytes())
                .map_err(|_| ManifestError::Io)?;
            file.sync_all().map_err(|_| ManifestError::Io)?;
        }
        fs::rename(&tmp, &path).map_err(|_| ManifestError::Io)?;
        Ok(())
    }

    pub fn mark(
        &mut self,
        id: &PartitionId,
        state: PartitionState,
        error: Option<String>,
    ) -> Result<(), ManifestError> {
        let row = self
            .file
            .partitions
            .iter_mut()
            .find(|row| &row.id == id)
            .ok_or(ManifestError::Missing)?;
        row.state = state;
        row.error = error;
        self.save()
    }
}

fn manifest_path(root: &Path) -> PathBuf {
    root.join("manifest.json")
}

fn component_ok(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 32
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_')
}

fn day_ok(value: &str) -> bool {
    let bytes = value.as_bytes();
    bytes.len() == 10
        && bytes[4] == b'-'
        && bytes[7] == b'-'
        && bytes
            .iter()
            .enumerate()
            .all(|(index, byte)| index == 4 || index == 7 || byte.is_ascii_digit())
}

fn stored_path_rejected(root: &Path, path: &str) -> bool {
    if path.is_empty() || path.contains('\0') {
        return true;
    }
    if path.split(['/', '\\']).any(|part| part == "..") {
        return true;
    }
    let candidate = Path::new(path);
    candidate.is_absolute() && !candidate.starts_with(root)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ManifestError {
    Io,
    Corrupt,
    Version,
    Missing,
}

impl std::fmt::Display for ManifestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::Io => "manifest could not be read or written",
            Self::Corrupt => "manifest JSON is invalid",
            Self::Version => "manifest version is not supported",
            Self::Missing => "partition is not in the manifest",
        };
        f.write_str(message)
    }
}

impl std::error::Error for ManifestError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absolute_path_outside_the_data_root_is_rejected() {
        let root = std::env::temp_dir().join(format!(
            "mm-manifest-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        let body = r#"{
            "version": 1,
            "partitions": [{
                "id": {"exchange":"binance","symbol":"BTCUSDT","day":"2026-11-02","hour":13},
                "state": "published",
                "raw_path": "/etc/passwd",
                "parquet_path": "/tmp/not-in-this-root.parquet",
                "raw_count": 0,
                "parquet_count": null,
                "schema_version": 1,
                "start_ns": 0,
                "end_ns": 1,
                "error": null
            }]
        }"#;
        fs::write(root.join("manifest.json"), body).unwrap();
        assert!(matches!(Catalog::open(&root), Err(ManifestError::Corrupt)));
        let _ = fs::remove_dir_all(&root);
    }
}
