#!/usr/bin/env bash
# Install the `just` command runner for this user if it is not already on PATH.
# Does not use root. An existing `just` is left unchanged.
set -euo pipefail

if command -v just >/dev/null 2>&1; then
    exit 0
fi

cargo install just --locked
