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

# Run that binary. Edit configuration() to change the experiment.
run:
    cargo run --bin mm_model --features "live,observe,artifact,parquet-storage"

# Build the runtime image. It does not contain historical data.
docker-build:
    docker build -t mm-model .

# Run the image with a host data directory. No CLI arguments.
docker-run:
    mkdir -p data
    docker run --rm -v "{{justfile_directory()}}/data:/data" mm-model
