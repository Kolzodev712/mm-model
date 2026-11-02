//! Analytical quote contexts around the current reference price.
//!
//! An offset is a hypothetical distance, in basis points, below that price:
//! `price = fair * (1 - bps / 10_000)`.
//! Each probe is scored on its own. The grid does not rank or choose one.

use std::fmt;

use crate::domain::market::{MarketState, QuoteContext};
use crate::domain::units::Quantity;

/// Maximum probes retained from configuration.
pub const MAX_PROBES: usize = 8;

/// One hypothetical quote context.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Probe {
    pub offset_bps: f64,
    pub quote: QuoteContext,
}

/// Basis-point offsets parsed at startup.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProbeGrid {
    offsets_bps: [f64; MAX_PROBES],
    quantities: [f64; MAX_PROBES],
    len: usize,
}

impl ProbeGrid {
    /// Parses `0.5,1,2,5` style text. At least one offset, at most [`MAX_PROBES`].
    pub fn parse(text: &str, quantity: f64) -> Result<Self, ProbeError> {
        let quantity = Quantity::new(quantity).map_err(|_| ProbeError::Quantity)?;
        let mut offsets_bps = [0.0; MAX_PROBES];
        let mut len = 0usize;
        for part in text.split(',') {
            let part = part.trim();
            if part.is_empty() {
                return Err(ProbeError::Offset);
            }
            if len == MAX_PROBES {
                return Err(ProbeError::TooMany);
            }
            let bps: f64 = part.parse().map_err(|_| ProbeError::Offset)?;
            if !bps.is_finite() || bps < 0.0 {
                return Err(ProbeError::Offset);
            }
            offsets_bps[len] = bps;
            len += 1;
        }
        if len == 0 {
            return Err(ProbeError::Offset);
        }
        let mut quantities = [0.0; MAX_PROBES];
        quantities[..len].fill(quantity.get());
        Ok(Self {
            offsets_bps,
            quantities,
            len,
        })
    }

    /// Builds a grid from `(distance_bps, quantity)` pairs. An empty list is a grid with no probes.
    pub fn from_pairs(pairs: &[(f64, f64)]) -> Result<Self, ProbeError> {
        if pairs.len() > MAX_PROBES {
            return Err(ProbeError::TooMany);
        }
        let mut offsets_bps = [0.0; MAX_PROBES];
        let mut quantities = [0.0; MAX_PROBES];
        for (index, (bps, quantity)) in pairs.iter().copied().enumerate() {
            if !bps.is_finite() || bps < 0.0 {
                return Err(ProbeError::Offset);
            }
            Quantity::new(quantity).map_err(|_| ProbeError::Quantity)?;
            offsets_bps[index] = bps;
            quantities[index] = quantity;
        }
        Ok(Self {
            offsets_bps,
            quantities,
            len: pairs.len(),
        })
    }

    #[must_use]
    pub fn len(self) -> usize {
        self.len
    }

    #[must_use]
    pub fn is_empty(self) -> bool {
        self.len == 0
    }

    /// Offset at `index`, in basis points.
    #[must_use]
    pub fn offset_bps(self, index: usize) -> Option<f64> {
        self.offsets_bps
            .get(index)
            .copied()
            .filter(|_| index < self.len)
    }

    /// Writes probes into `out` in configuration order. Returns how many were written.
    ///
    /// No heap allocation.
    pub fn generate(&self, market: &MarketState, out: &mut [Probe]) -> Result<usize, ProbeError> {
        if out.len() < self.len {
            return Err(ProbeError::Buffer);
        }
        let fair = market.fair_value();
        for (index, bps) in self.offsets_bps[..self.len].iter().copied().enumerate() {
            let price = fair * (1.0 - bps / 10_000.0);
            let quote =
                QuoteContext::new(price, self.quantities[index]).map_err(|_| ProbeError::Price)?;
            out[index] = Probe {
                offset_bps: bps,
                quote,
            };
        }
        Ok(self.len)
    }
}

/// Probe configuration or generation failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeError {
    Offset,
    TooMany,
    Quantity,
    Buffer,
    Price,
}

impl fmt::Display for ProbeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::Offset => "probe offsets must be finite, non-negative basis points",
            Self::TooMany => "too many probe offsets",
            Self::Quantity => "probe quantity must be finite and positive",
            Self::Buffer => "probe buffer is smaller than the configured grid",
            Self::Price => "probe price is not a positive finite number",
        };
        f.write_str(message)
    }
}

impl std::error::Error for ProbeError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::fair_value::FairValueMethod;

    #[test]
    fn probes_keep_configuration_order_and_sit_at_the_requested_distance() {
        let grid = ProbeGrid::parse("0.5,1,2,5", 0.01).unwrap();
        let market = FairValueMethod::Microprice
            .derive_top(99.9, 1.0, 100.1, 1.0)
            .unwrap();
        let mut slots = [Probe {
            offset_bps: 0.0,
            quote: QuoteContext::new(1.0, 0.01).unwrap(),
        }; MAX_PROBES];
        let n = grid.generate(&market, &mut slots).unwrap();
        assert_eq!(n, 4);
        let fair = market.fair_value();
        for (index, probe) in slots[..n].iter().enumerate() {
            let bps = grid.offset_bps(index).unwrap();
            assert_eq!(probe.offset_bps, bps);
            let expected = fair * (1.0 - bps / 10_000.0);
            assert!((probe.quote.price() - expected).abs() < 1e-9);
        }
    }
}
