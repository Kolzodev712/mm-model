//! Persistence. Raw MMBK and, when enabled, derived Parquet books.

pub mod raw;

#[cfg(feature = "parquet-storage")]
pub mod manifest;
#[cfg(feature = "parquet-storage")]
pub mod parquet;
#[cfg(feature = "parquet-storage")]
pub mod partition;
