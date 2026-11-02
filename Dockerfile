# Pinned to the toolchain used to test this repository.
FROM rust:1.97.1-bookworm AS builder
WORKDIR /src
COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY src ./src
RUN cargo build --release --bin mm-shadow --features "live,cli,observe,artifact"

FROM debian:bookworm-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 --user-group mmrunner
COPY --from=builder /src/target/release/mm-shadow /usr/local/bin/mm-shadow
USER mmrunner
ENTRYPOINT ["mm-shadow"]
