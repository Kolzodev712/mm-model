# Market-making quote baselines

Reference code for scoring buy quotes on one instrument. This is not an
optimal quoting model. It is a place to study two estimators that feed the
same deterministic decision rule.

```text
Binance BTCUSDT book
        |
        v
FairValueMethod (microprice by default)
        |
       FV_t
        |
        +----------------------+
        |                      |
        v                      v
 feature generation      CandidateQuote
        |                      |
        +----------+-----------+
                   |
                   v
        logistic P(fill), linear E[P&L | fill]
                   |
                   v
        EV = P(fill) * E[P&L | fill]
                   |
                   v
   deterministic selection against NoQuote (EV = 0)
```

The same method applied to the book at `t + H` produces the horizon
reference price used in training labels.

Nothing here submits, cancels, or amends orders. A learned score is for
offline evaluation, replay, or shadow comparison.

## Deterministic historical baseline

The historical formulas are the same benchmark as before. A legacy
`HistoricalObservation` still stores a horizon price the caller already has.
The learned pipeline does not. Given quotes that were actually submitted:

```text
P(fill) = fills / attempts

pnl = (fair_value_at_horizon - execution_price) * quantity - fees
      (0 when the quote did not fill)

E[P&L | fill] = total filled P&L / fills

EV = P(fill) * E[P&L | fill]
   = total P&L / attempts
```

`NoQuote` has EV 0. The highest finite EV above 0 is selected. Equal positive
EVs use the lexicographically smallest strategy id.

These counts are a benchmark. They are not the learned models.

A historical fill at one quote price does not prove whether a different quote
would have filled at the same moment.

## Learned baseline

Implemented, and intentionally small:

| Piece | Role | Status |
| --- | --- | --- |
| `fair_value` | Midpoint and microprice from the best bid and ask | Implemented |
| `decision` | Top of book plus one estimated reference price | Implemented |
| `features` | Four relative features, fixed array | Implemented |
| `logistic` | Fill probability, binary cross-entropy, batch gradient descent | Reference model |
| `linear` | Conditional P&L, mean squared error, batch gradient descent | Reference model |
| `model::predict_quote` | One feature row, then `EV = p * conditional P&L` | Implemented |
| `selection::select_action` | Same rule as the historical selector | Implemented |
| `trainset` | Decision-time inputs, outcome, observed vs simulated | Implemented |
| `dataset` | Books plus quote outcomes; both reference prices are estimated | Implemented |
| `marketdata` | Fixed top-10 book log and Binance payload parser | Implemented |
| `record_book` | Read-only public websocket recorder | Implemented, experimental |
| Risk limits, inventory, live orders | | Not implemented |

Logistic regression and linear regression are educational baselines. Fitting
them does not make the quote optimal.

### Features

All four are known before the order is sent. None use the fill, the execution
price, or the horizon reference price. Distances are divided by the
book-derived reference price on that snapshot. The estimator rejects a touch
with a non-positive quantity, so feature extraction does not divide by zero.

Microprice is a function of the same best-level quantities as
`top_quantity_imbalance`. The features are correlated. They are both kept so
that overlap can be measured later.

| Index | Name | Definition | Unit |
| --- | --- | --- | --- |
| 0 | `quote_distance_from_fair` | `(fair - quote_price) / fair` | fraction of fair |
| 1 | `exchange_spread` | `(best_ask - best_bid) / fair` | fraction of fair |
| 2 | `quote_distance_from_best_bid` | `(quote_price - best_bid) / fair` | fraction of fair |
| 3 | `top_quantity_imbalance` | `(bid_qty - ask_qty) / (bid_qty + ask_qty)` | dimensionless |

Ask distance is omitted. For a buy it is determined by feature 2 and the
spread. Candidate quantity is also omitted, so a conditional-P&L label is the
total mark-to-fair P&L of that fill, not an explicit per-unit model.

Inference is a stack array in and a scalar out. It does not allocate.

### Training

Offline only. There is no path from the websocket into weight updates.

```text
exchange -> recorder -> book log
                              \
quote/fill records ----------> dataset builder
                              /   one FairValueMethod
                              \
                    FV_t from the book at t
                    FV_(t+H) from the first valid book at or after t+H
                              \
                               offline trainer -> weights
```

The training set records which method produced it. Rows from midpoint and
microprice are not trained together.

Fill labels are `1` or `0` from a submitted order, or from a row marked
`Simulated`. The conditional-P&L model is fit only on fills. An unfilled row
is not given P&L of 0 and passed to that model. Observed and simulated rows
are rejected if they are handed to one training call together.

Losses:

- fill model: mean binary cross-entropy, full-batch gradient descent
- P&L model: mean squared error, full-batch gradient descent

Learning rate and epoch count are the only training settings.

## Recorder

`record_book` subscribes to the public Binance spot stream

`wss://stream.binance.com:9443/ws/btcusdt@depth10@100ms`

and appends top-10 bid and ask levels. Each record can store the exchange
event time and update id when the payload has them. The partial-depth payload
documented by Binance has `lastUpdateId`, bids, and asks, and no event time,
so that field is usually empty. Local receive time is always stored.

```text
cargo run --features recorder --bin record_book -- books.mmbk
```

The file format is little-endian, version 1, magic `MMBK`, fixed 384-byte
records. Layout is in `src/marketdata/format.rs`. A short tail from a torn
write is ignored on read and truncated before the next append. The recorder
does not write a reference price. An old `MMFV` log of caller-supplied prices
is still readable and is not an input to training.

The websocket library allocates its own read buffers. This crate does not
claim those allocations are free. The parser writes into one reused
fixed-depth struct and does not build a JSON tree.

Raw book data does not label whether an unsubmitted quote would have filled.
The dataset builder returns an error when quote outcomes are absent. It will
not invent them from later prices. No live quote/fill recorder exists yet, so
those labels are not produced by this repository.

## Reference price

`FairValueMethod` is deterministic and reads only the best bid and ask.

Midpoint is `(P_bid + P_ask) / 2`. It ignores displayed size, except that
both sizes must be strictly positive and the book must be uncrossed.
Microprice, the default, is

```text
(P_ask * Q_bid + P_bid * Q_ask) / (Q_bid + Q_ask)
```

A larger bid quantity pulls the price toward the ask. A larger ask quantity
pulls it toward the bid. When the quantities are equal, microprice matches
the midpoint. Invalid books return an error. They do not become zero and
microprice does not fall back to midpoint.

This estimate is an exchange-implied reference price from the same Binance
book the quote features use. It is not an independent fair value, it can
repeat information already in the features, and it is not a fundamental or
cross-market price. A multi-venue composite, a predictive fair-value model,
or a production engine can replace the enum later. None of those are
implemented.

For a training row, `FV_t` comes from the latest valid book at or before the
decision. `FV_(t+H)` comes from the first valid book at or after `t + H`.
The row stores the decision time, the requested horizon, the horizon
snapshot time, and the offset. A filled buy is marked with

```text
(FV_(t+H) - execution_price) * quantity - fees
```

## Deferred

Neural nets, learned fair-value models, multi-exchange or trade-flow fair
value, futures basis, momentum, Kalman filters, depth-weighted fair value,
boosting, reinforcement learning, online learning, inventory, sell quotes,
partial fills, queue position, model serving, order entry, and a risk engine.

## What to study next

Join recorded books with real submitted-quote outcomes and compare microprice
labels with midpoint labels and with the historical baseline. Measure how
much of the microprice signal is already in the imbalance feature, and how
large the horizon offset is on the recorded stream. A fill at one price still
says nothing by itself about a price that was not sent.
