//! Decision-time feature schemas.
//!
//! Version 1 is the only schema. A model artifact must name the same version.

pub mod v1;

pub use v1::{
    extract_features, FeatureError, FeatureVector, EXCHANGE_SPREAD, FEATURE_COUNT, FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION, QUOTE_DISTANCE_FROM_BEST_BID, QUOTE_DISTANCE_FROM_FAIR,
    RELATIVE_CANDIDATE_QUANTITY, TOP_QUANTITY_IMBALANCE,
};
