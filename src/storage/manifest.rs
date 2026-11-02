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
        Ok(Self { root, file })
    }

    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
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
        fs::write(&tmp, text).map_err(|_| ManifestError::Io)?;
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
