//! Fixed buy-candidate grid around the book-derived reference price.
//!
//! Offsets are basis points below that price:
//! `quote = fair * (1 - bps / 10_000)`.
//! These are shadow prices, not tick-rounded live orders.
//! The grid is parsed once. Generation writes into a caller buffer.

use std::fmt;

use crate::domain::market::{CandidateQuote, MarketState};
use crate::domain::units::{CandidateId, Quantity};

/// Maximum candidates retained from configuration.
pub const MAX_CANDIDATES: usize = 8;

/// One generated candidate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CandidateSlot {
    pub id: CandidateId,
    pub quote: CandidateQuote,
}

/// Basis-point offsets parsed at startup.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BpsGrid {
    offsets_bps: [f64; MAX_CANDIDATES],
    len: usize,
    quantity: Quantity,
}

impl BpsGrid {
    /// Parses `0.5,1,2,5` style text. At least one offset, at most [`MAX_CANDIDATES`].
    pub fn parse(text: &str, quantity: f64) -> Result<Self, CandidateError> {
        let quantity = Quantity::new(quantity).map_err(|_| CandidateError::Quantity)?;
        let mut offsets_bps = [0.0; MAX_CANDIDATES];
        let mut len = 0usize;
        for part in text.split(',') {
            let part = part.trim();
            if part.is_empty() {
                return Err(CandidateError::Offset);
            }
            if len == MAX_CANDIDATES {
                return Err(CandidateError::TooMany);
            }
            let bps: f64 = part.parse().map_err(|_| CandidateError::Offset)?;
            if !bps.is_finite() || bps < 0.0 {
                return Err(CandidateError::Offset);
            }
            offsets_bps[len] = bps;
            len += 1;
        }
        if len == 0 {
            return Err(CandidateError::Offset);
        }
        Ok(Self {
            offsets_bps,
            len,
            quantity,
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

    /// Writes candidates into `out`. Returns how many were written.
    ///
    /// No heap allocation. Ids are the grid index and stay stable.
    pub fn generate(
        &self,
        market: &MarketState,
        out: &mut [CandidateSlot],
    ) -> Result<usize, CandidateError> {
        if out.len() < self.len {
            return Err(CandidateError::Buffer);
        }
        let fair = market.fair_value();
        for (index, bps) in self.offsets_bps[..self.len].iter().copied().enumerate() {
            let price = fair * (1.0 - bps / 10_000.0);
            let quote = CandidateQuote::new(price, self.quantity.get())
                .map_err(|_| CandidateError::Price)?;
            let id = u16::try_from(index).map_err(|_| CandidateError::TooMany)?;
            out[index] = CandidateSlot {
                id: CandidateId::new(id),
                quote,
            };
        }
        Ok(self.len)
    }
}

/// Candidate configuration or generation failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CandidateError {
    Offset,
    TooMany,
    Quantity,
    Buffer,
    Price,
}

impl fmt::Display for CandidateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::Offset => "candidate offsets must be finite, non-negative basis points",
            Self::TooMany => "too many candidate offsets",
            Self::Quantity => "candidate quantity must be finite and positive",
            Self::Buffer => "candidate buffer is smaller than the configured grid",
            Self::Price => "candidate price is not a positive finite number",
        };
        f.write_str(message)
    }
}

impl std::error::Error for CandidateError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::fair_value::FairValueMethod;

    #[test]
    fn grid_ids_are_stable_and_prices_sit_below_fair() {
        let grid = BpsGrid::parse("0.5,1,2,5", 0.01).unwrap();
        assert_eq!(grid.len(), 4);
        let market = FairValueMethod::Microprice
            .derive_top(99.9, 1.0, 100.1, 1.0)
            .unwrap();
        let mut slots = [CandidateSlot {
            id: CandidateId::new(0),
            quote: CandidateQuote::new(1.0, 0.01).unwrap(),
        }; MAX_CANDIDATES];
        let n = grid.generate(&market, &mut slots).unwrap();
        assert_eq!(n, 4);
        let fair = market.fair_value();
        for (index, slot) in slots[..n].iter().enumerate() {
            assert_eq!(slot.id.get(), index as u16);
            assert!(slot.quote.price() < fair);
            assert_eq!(slot.quote.quantity(), 0.01);
        }
        let expected = fair * (1.0 - 5.0 / 10_000.0);
        assert!((slots[3].quote.price() - expected).abs() < 1e-9);
    }
}
