# Market-making quote baselines

This repository is an experimental study bench for buy-quote scoring. It is
not an optimal quoting system, it does not send orders, and passing tests
does not make it production-ready.

The executable has one composition root, `src/bin/mm_model.rs`. Change the
experiment by editing `configuration()`. There is no CLI parser.

```text
Collect
Train
CollectAndTrain
Shadow
```

Those are typed `RunMode` values, not commands.

## Pipeline

```text
                   Binance
                      |
                      v
                 Collector
                      |
                      v
                 Raw MMBK
                      |
               rotate / close
                      |
                      v
                  Converter
                      |
                      v
              Published Parquet
                      |
                 +----+----+
                 |         |
                 v         v
              Trainer   Analysis
                 |
                 v
             Model Artifact
                 |
                 v
               Shadow
```

```text
X
 |
 +-------------------+
 |                   |
 v                   v
Fill Model         P&L Model
P(fill | X)        E[P&L | fill, X]
 |                   |
 +---------+---------+
           |
           v
EV = probability * conditional P&L
           |
           v
candidate selection
           |
           v
Quote / NoQuote
```

## Source of truth

```text
MMBK = raw source of truth

Parquet books = derived historical analytical representation

Training examples = derived labels and features

Model artifact = derived from a specific training dataset and configuration
```

Raw capture is append-only and partitioned by UTC hour:

```text
data/raw/binance/BTCUSDT/2026-11-02/13.mmbk
```

Closed hours are converted in a batch to ZSTD Parquet under `data/books/`.
A corrupt or outdated Parquet file is rebuilt from MMBK. Parquet is not the
only copy of captured books.

The catalog is `data/manifest.json`, rewritten through a temporary file and
`rename`. A partition moves through:

```text
Open -> Closed -> Validated -> Converted -> Published
```

`Failed` is terminal for that attempt. Training reads `Published` partitions
only. The open hour is never trained.

Public Binance books do not contain fill labels for hypothetical quotes.
Training requires observed quote/fill records or explicitly simulated labels.
`LabelProvenance` keeps those apart. This pipeline does not invent fills from
price paths.

## Layout

```text
domain    prices, books, quotes, outcomes
model     fair value, features, logistic fill, linear P&L, selection, artifacts
data      exchange intake, training rows, historical baseline, synthetic demo rows
storage   MMBK, manifest, Parquet
runtime   collect, convert, train, shadow, orchestration
bin       configuration and startup only
```

Dependencies point inward: domain knows nothing about files or sockets. The
model does not know about Binance or Parquet. Training reads books through
`HistoricalBooks`.

## Modes

`Collect` writes raw partitions and publishes Parquet. It does not fit a model.

`Train` reads published Parquet inside the configured window and writes an
artifact when the reference fit reduces loss and sees at least one fill. It
does not open a websocket.

`CollectAndTrain` collects, converts each closed hour, and fits after every
newly published partition (`TrainingTrigger::EveryPublishedPartitions`).
Collection and training are separate functions. In this build they run on one
thread, so collection pauses during conversion and training. A failed
conversion marks that partition `Failed` and collection continues. A failed
fit keeps the previous artifact. A manifest that cannot be saved stops the
state change.

`Shadow` scores the live book and logs a quote or `NoQuote`. It does not write
partitions, send orders, or change weights. The default configuration uses the
synthetic demo model and prints `DEMO MODEL — predictions are not trading-valid`.

A newly trained artifact is available for later shadow experiments only when
validation passes. That is not a trading approval.

## Math that stayed

Historical EV, for quotes that were actually submitted:

```text
P(fill) = fills / attempts
pnl = (fair_value_at_horizon - execution_price) * quantity - fees
EV = P(fill) * average filled P&L
```

Unfilled attempts contribute 0. The conditional average is absent when there
are no fills. `NoQuote` has EV 0. The highest finite EV above 0 wins. Ties
keep the smaller id.

`FairValueMethod::Microprice` is the default:

```text
(P_ask * Q_bid + P_bid * Q_ask) / (Q_bid + Q_ask)
```

Midpoint is `(P_bid + P_ask) / 2`. Both reject empty, non-finite, non-positive,
and locked or crossed books. They do not return 0, and microprice does not
fall back to midpoint.

Feature schema v1 has five decision-time ratios, including candidate quantity.
Logistic regression predicts fill probability. Linear regression predicts
conditional P&L. The product is outside either model. A non-finite score is an
error. NaN is not turned into 0.5.

Dataset alignment uses the latest valid book at or before the decision and the
first valid book at or after `t + H`, with staleness and slippage limits.
Timestamp gaps use checked arithmetic.

The reference trainers are full-batch gradient descent on one materialized
sample slice. Storage can scan one partition at a time. The trainer is not yet
a streaming or minibatch algorithm, and it is not suitable for an arbitrarily
large history. The binary configuration chooses the training window.

A fit is not released for shadow because training loss fell. Rows are split
chronologically into train, validation, and test. The test slice is not used
to accept the model. Release requires observed labels, at least 20 examples
and 5 fills, and validation loss no worse than a constant baseline. Anything
smaller, or any simulated label set, stays unreleased. That is not evidence
the model is economically useful.

Price-ratio features are stored in basis points so the reference gradient step
can move. Under microprice, distance from the best bid is determined by the
other features and its weight is not separately identified.

Collection appends the open MMBK hour on one thread. Closed hours go to one
worker on a two-slot channel. A full channel stops the run. It does not drop
books. Shutdown flushes the open hour and leaves it open. The next process
can close an hour that is already in the past.

## Run

```text
just build
just run
just docker-build
just docker-run
just docker-up
```

`just run` uses `./data` unless `MM_DATA_ROOT` is set. The container image
sets `MM_DATA_ROOT=/data` and expects the host directory mounted there:

```text
docker run --rm -v ./data:/data mm-model
```

Compose runs that same image. It publishes no ports, drops all capabilities,
and mounts `./data` at `/data`:

```text
docker compose up --build
just docker-up
```

The process exits when the configured run ends, and Compose does not restart
it. The container root filesystem is read-only. `./data` is the writable
mount. `just docker-down` removes the container and leaves `./data` on the host.

The image is non-root, has no compiler, and opens no inbound port. Historical
data is not copied into the image. The mounted directory must be writable by
uid 10001 when the configured mode records or trains.

```text
just fmt
just clippy
just tests
just check
```

Tests do not open a network connection.

Library features:

| Feature | Dependencies | Role |
| --- | --- | --- |
| `live` | `tungstenite`, `rustls` | Binance websocket client |
| `observe` | `tracing`, `tracing-subscriber`, `signal-hook` | runtime logs and SIGINT/SIGTERM |
| `artifact` | `serde`, `serde_json` | model files and the manifest |
| `parquet-storage` | Arrow Parquet, ZSTD | derived book files; implies `artifact` |

The default library build does not compile the websocket client or Parquet.
Inference after startup does not allocate: fair value, candidates, features,
both predictions, the EV product, and learned selection use stack buffers.
Capture, conversion, training, and artifact IO may allocate.

## Not in this repository

Order entry, private exchange APIs, inventory, online learning, neural nets,
multi-venue fair value, a risk engine, and a database. A fill at one price
still does not prove that a different price would have filled.
