//! Decision-time feature schemas.
//!
//! Quote-conditioned rows use [`v1`]. Market-outcome rows use [`market`].
//! A model artifact must name the schema for its own family.

pub mod market;
pub mod v1;

pub use v1::{
    extract_features, FeatureError, FeatureVector, EXCHANGE_SPREAD, FEATURE_COUNT, FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION, QUOTE_DISTANCE_FROM_BEST_BID, QUOTE_DISTANCE_FROM_FAIR,
    RELATIVE_CANDIDATE_QUANTITY, TOP_QUANTITY_IMBALANCE,
};
