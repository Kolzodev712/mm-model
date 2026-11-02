# Pinned to the toolchain used to test this repository.
FROM rust:1.97.1-bookworm AS builder
WORKDIR /src
COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY src ./src
RUN cargo build --release --bin mm_model --features "live,observe,artifact,parquet-storage"

FROM debian:bookworm-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 --user-group mmrunner
COPY --from=builder /src/target/release/mm_model /usr/local/bin/mm_model
COPY configs /app/configs
USER mmrunner
ENV MM_DATA_ROOT=/data
ENV MM_CONFIG=/app/configs/default.json
WORKDIR /home/mmrunner
ENTRYPOINT ["mm_model"]
