//! Hypothetical quoting history shared by unit tests.
//!
//! Quantities are 1 and fees are 0. Statistics are not stored here; tests
//! derive them through the library.

use crate::domain::{FillOutcome, HistoricalObservation, StrategyId};

/// Ten buy-quote attempts: five for strategy A and five for strategy B.
pub fn example_dataset() -> Vec<HistoricalObservation> {
    let rows = [
        ("A", 99.95, true, 100.05),
        ("A", 99.95, false, 100.10),
        ("A", 99.95, true, 99.70),
        ("A", 99.95, true, 100.00),
        ("A", 99.95, false, 99.95),
        ("B", 99.80, false, 99.90),
        ("B", 99.80, true, 100.10),
        ("B", 99.80, false, 100.05),
        ("B", 99.80, false, 99.80),
        ("B", 99.80, false, 100.00),
    ];

    rows.into_iter()
        .map(|(strategy, bid, filled, fair_value)| {
            let outcome = if filled {
                FillOutcome::Filled {
                    execution_price: bid,
                }
            } else {
                FillOutcome::Unfilled
            };
            HistoricalObservation::new(
                StrategyId::new(strategy).unwrap(),
                bid,
                1.0,
                outcome,
                fair_value,
                0.0,
            )
            .unwrap()
        })
        .collect()
}
