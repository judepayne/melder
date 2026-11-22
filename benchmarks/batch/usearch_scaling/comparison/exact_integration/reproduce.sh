#!/bin/bash
set -euo pipefail
# Invoke from production repository root. Source remains pristine; edits are private.
artifact="$PWD/benchmarks/batch/usearch_scaling/comparison/exact_integration"
work="${1:-/tmp/melder-exact-reproduction}"
mkdir "$work"
mkdir "$work/src"
git archive 446d45e | tar -x -C "$work/src"
(cd "$work/src" && patch -p1 < "$artifact/prepared-normalized.patch")
CARGO_TARGET_DIR="$work/target" cargo test --release --manifest-path "$work/src/Cargo.toml" --lib prepared
CARGO_TARGET_DIR="$work/target" cargo build --release --manifest-path "$work/src/Cargo.toml"
# Use ../run.py with this target/release/meld as the candidate; see REPORT.md.
