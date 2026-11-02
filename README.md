# Optimal market-making quote model

Historical baseline for choosing among buy-quote strategies. This is a
deterministic calculation library, not a trained model and not a final
quoting architecture.

Given recorded quoting attempts, it estimates:

- fill probability: fills / attempts
- total mark-to-fair-value P&L
- average P&L conditional on a fill
- expected value per quoting attempt

For a filled buy:

```text
pnl = (fair_value_at_horizon - execution_price) * quantity - fees
```

An unfilled order has P&L of 0. Expected value is computed two ways:

```text
ev = p_fill * average_pnl_if_filled
ev = total_pnl / attempts
```

With unfilled P&L defined as 0, those are the same quantity up to floating-point
error. The selector takes the strategy with the highest expected value.
`NoQuote` has EV of 0 and is chosen when no strategy is strictly above that.

## What this deliberately does not model

- More than one instrument, sell quotes, or partial fills
- Inventory, risk limits, or adverse selection
- Confidence intervals or shrinkage of the historical frequencies
- Learned fill or P&L models
- Networking, exchange sessions, async runtimes, or storage

The calculation is split so a later model can replace the historical counts
without changing the decision rule:

`market state + candidate quote` → predicted fill probability and conditional P&L → expected value → quote or `NoQuote`

Today, historical aggregation supplies those predictions.

## Counterfactuals

These statistics describe quotes that were actually submitted. They are not
evidence about quotes that were not sent.

A historical fill at one quote price does not prove whether a different quote
would have filled at the same moment.

## Layout

- `src/domain.rs` — observations and P&L
- `src/evaluation.rs` — per-strategy historical estimates
- `src/selection.rs` — comparison against `NoQuote`
