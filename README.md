# Market-making quote baselines

This repository is a study bench for buy-quote scoring. It is not an optimal
quoting system and it does not trade.

```text
                  Binance
                     |
                     v
             BinanceBookClient
                     |
                     v
                BookRecord
                     |
                     v
            FairValueMethod
                     |
                     v
                MarketState
                     |
              CandidateGenerator
                     |
           +---------+---------+
           |                   |
           v                   v
     Logistic model       Linear model
       P(fill|X)         E[P&L|fill,X]
           |                   |
           +---------+---------+
                     |
                     v
              Expected Value
                     |
                     v
              select / NoQuote
                     |
                     v
             SHADOW LOG ONLY
```

No orders are sent. No realized P&L is measured. Public order-book data does
not provide fill labels for hypothetical quotes. Demo-model output is not
trading-valid.

## What is deterministic

The historical baseline in `domain`, `evaluation`, and `selection` still
aggregates quotes that were actually submitted:

```text
P(fill) = fills / attempts
pnl = (fair_value_at_horizon - execution_price) * quantity - fees
EV = P(fill) * average filled P&L
```

`HistoricalObservation` still stores a horizon price the caller already has.
That path is the legacy benchmark. The learned dataset builder does not use it.

`NoQuote` has EV 0. The highest finite EV above 0 wins. Learned candidates use
the same rule through `select_learned`, with a copyable `CandidateId` instead
of a heap string.

Reference price is also deterministic. `FairValueMethod::Microprice` is the
default:

```text
(P_ask * Q_bid + P_bid * Q_ask) / (Q_bid + Q_ask)
```

Midpoint is `(P_bid + P_ask) / 2`. Both reject empty, non-finite, non-positive,
and locked or crossed books. They do not return 0 and microprice does not fall
back to midpoint. This price is an exchange-implied reference from the same
Binance touch used by the features. It is not an independent fair value.

## What is learned

Feature schema v1 has five decision-time ratios:

| Index | Name | Formula |
| --- | --- | --- |
| 0 | `quote_distance_from_fair` | `(fair - quote_price) / fair` |
| 1 | `exchange_spread` | `(best_ask - best_bid) / fair` |
| 2 | `quote_distance_from_best_bid` | `(quote_price - best_bid) / fair` |
| 3 | `top_quantity_imbalance` | `(bid_qty - ask_qty) / (bid_qty + ask_qty)` |
| 4 | `relative_candidate_quantity` | `candidate_quantity / (bid_qty + ask_qty)` |

Logistic regression predicts fill probability. Linear regression predicts
conditional P&L. Expected value is the product, outside either model. A
non-finite score is an error. NaN is not turned into 0.5 and an invalid
probability is not clamped.

Training is offline. A model artifact records the schema version, fair-value
method, horizon, and weights. A run refuses a file that does not match.
Observed and simulated labels cannot be trained together.

Dataset alignment uses the latest valid book at or before the decision and the
first valid book at or after `t + H`. Rows are rejected when the decision book
is older than `max_decision_staleness` or the horizon book is later than
`max_horizon_slippage`. Timestamp gaps use checked arithmetic.

## What is live market data

`record_book` appends Binance spot partial depth (top 10, 100 ms by default)
to a fixed 384-byte `MMBK` log. The symbol and endpoint are client
configuration. BTCUSDT is only the example default. The writer flushes every
N records and does not fsync each update, so a crash can drop the last batch.

`mm-shadow` reads the same public stream, derives the reference price,
scores a fixed basis-point candidate grid, and logs predictions for a bounded
duration. It does not submit orders, update weights, or claim fills.

```text
just shadow --duration 60s --report-every 5s --symbol BTCUSDT --demo-model
```

`--demo-model` trains a deterministic simulated sample at startup and prints
`DEMO MODEL — predictions are not trading-valid`. `--model ./models/btc.json`
loads a previously written artifact. One of those two flags is required.
There is no silent fallback.

A model file can be mounted read-only:

```text
docker run --rm -v ./models:/models:ro mm-model-shadow \
  --duration 5m --report-every 10s --symbol BTCUSDT --model /models/btc.json
```

## Development

```text
./scripts/install-just.sh
just fmt
just clippy
just tests
just check
just docker-build
just docker-shadow --duration 60s --report-every 5s --symbol BTCUSDT --demo-model
```

Library features:

| Feature | Dependencies | Used by |
| --- | --- | --- |
| `live` | `tungstenite`, `rustls` (ring) | Binance websocket client, `record_book`, `mm-shadow` |
| `cli` | `clap` | both binaries |
| `observe` | `tracing`, `tracing-subscriber`, `signal-hook` | `mm-shadow` logs and SIGINT/SIGTERM |
| `artifact` | `serde`, `serde_json` | model file load/save |

The default library build has no websocket, CLI, or JSON dependency.
Inference after startup does not allocate: fair value, candidate fill,
feature extraction, both predictions, the EV product, and learned selection
use stack buffers supplied by the caller. Networking, logging, training, and
artifact load may allocate.

## Not in this repository

Order entry, private exchange APIs, inventory, online learning, neural nets,
multi-venue fair value, and a risk engine. A fill at one price still does not
prove that a different price would have filled.
