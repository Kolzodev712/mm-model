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

# Build the experimental binary. The mode is selected in src/bin/mm_model.rs.
build:
    cargo build --bin mm_model --features "live,observe,artifact,parquet-storage"

# Run that binary. Without MM_CONFIG it loads configs/default.json.
run:
    cargo run --bin mm_model --features "live,observe,artifact,parquet-storage"

# Run that binary with one JSON config. Example: just run-config configs/collect-btcusdt-2h.json
run-config path:
    MM_CONFIG={{path}} cargo run --bin mm_model --features "live,observe,artifact,parquet-storage"

# Build the runtime image. It does not contain historical data.
docker-build:
    docker build -t mm-model .

# Run the image with a host data directory. No CLI arguments.
docker-run:
    mkdir -p data
    docker run --rm -v "{{justfile_directory()}}/data:/data" mm-model

# Build and run the same image through Compose. The container exits when the configured run ends.
docker-up:
    mkdir -p data
    docker compose up --build

# Remove the Compose container. The host data directory is left in place.
docker-down:
    docker compose down
