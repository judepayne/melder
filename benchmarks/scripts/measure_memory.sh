#!/usr/bin/env bash
# measure_memory.sh — measure peak RSS of a meld batch run on macOS
#
# Usage: ./benchmarks/scripts/measure_memory.sh <config.yaml>
#
# Samples the process RSS every 100ms while meld runs, then reports:
#   - peak RSS (MB)
#   - final RSS just before exit (MB)
#   - a time-series CSV for inspection

set -euo pipefail

CONFIG="${1:?Usage: $0 <config.yaml>}"
MELD="./target/release/meld"
SAMPLE_INTERVAL=0.05  # seconds
OUTFILE="/tmp/meld_rss_samples.csv"

if [[ ! -x "$MELD" ]]; then
    echo "ERROR: $MELD not found. Run: cargo build --release --features usearch"
    exit 1
fi

echo "timestamp_s,rss_mb" > "$OUTFILE"

# Launch meld in background
"$MELD" run -c "$CONFIG" &
MELD_PID=$!

echo "Started meld (PID $MELD_PID), sampling RSS every ${SAMPLE_INTERVAL}s..."

START=$(python3 -c "import time; print(time.time())")
PEAK_RSS=0

# Sample loop — runs until meld exits
while kill -0 "$MELD_PID" 2>/dev/null; do
    # macOS ps reports RSS in kilobytes
    RSS_KB=$(ps -o rss= -p "$MELD_PID" 2>/dev/null || echo "0")
    RSS_KB=$(echo "$RSS_KB" | tr -d ' ')
    if [[ -n "$RSS_KB" && "$RSS_KB" -gt 0 ]]; then
        NOW=$(python3 -c "import time; print(f'{time.time():.3f}')")
        ELAPSED=$(python3 -c "print(f'{$NOW - $START:.3f}')")
        RSS_MB=$(python3 -c "print(f'{$RSS_KB / 1024:.1f}')")
        echo "${ELAPSED},${RSS_MB}" >> "$OUTFILE"
        # Track peak
        if (( RSS_KB > PEAK_RSS )); then
            PEAK_RSS=$RSS_KB
        fi
    fi
    sleep "$SAMPLE_INTERVAL"
done

# Wait for exit code
wait "$MELD_PID" || true

PEAK_MB=$(python3 -c "print(f'{$PEAK_RSS / 1024:.1f}')")
SAMPLES=$(wc -l < "$OUTFILE")
SAMPLES=$((SAMPLES - 1))  # subtract header

echo ""
echo "=== Memory Measurement Results ==="
echo "Peak RSS:       ${PEAK_MB} MB"
echo "Samples taken:  ${SAMPLES}"
echo "Sample data:    ${OUTFILE}"
echo ""
echo "--- First and last 5 samples ---"
head -6 "$OUTFILE"
echo "..."
tail -5 "$OUTFILE"
