set positional-arguments

# Check formatting.
fmt:
    cargo fmt --all --check

# Lint every target and feature.
clippy:
    cargo clippy --all-targets --all-features -- -D warnings

# Run the full test suite. Tests do not open a live exchange connection.
tests:
    cargo test --all-features

# Format, lint, and test.
check: fmt clippy tests

# Run the read-only book recorder. Example: just record books.mmbk --symbol BTCUSDT
record *args:
    cargo run --bin record_book --features "live,cli" -- {{args}}

# Run the shadow inference harness. No orders are sent.
shadow *args:
    cargo run --bin mm-shadow --features "live,cli,observe,artifact" -- {{args}}

# Build the shadow-runner image.
docker-build:
    docker build -t mm-model-shadow .

# Run the shadow image. Example: just docker-shadow --duration 60s --report-every 5s --demo-model
docker-shadow *args:
    docker run --rm mm-model-shadow {{args}}
