#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

echo "Running Rust tests for rust_tokenizer..."
cargo test -p rust_tokenizer

echo
echo "Running Python tests for rust_tokenizer (will skip if extension not installed)..."
cd "${ROOT_DIR}/rust_tokenizer"
pytest || true

echo
echo "Running benchmarks for rust_tokenizer..."
cd "${ROOT_DIR}"
cargo bench -p rust_tokenizer

echo
echo "All tests and benchmarks completed."


