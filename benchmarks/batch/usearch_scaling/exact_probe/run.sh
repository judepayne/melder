#!/bin/bash
# Portable standalone runner. Scheduling/grants, if needed, belong to the caller.
set -euo pipefail
cd "$(dirname "$0")"
TARGET="${EXACT_TARGET:-$(mktemp -d "${TMPDIR:-/tmp}/melder-exact.XXXXXX")}"
mkdir -p "$TARGET"
rustc --edition=2021 -C opt-level=3 -C debuginfo=0 main.rs -o "$TARGET/exact-probe"
"$TARGET/exact-probe" verify
OUT="${EXACT_OUT:-$TARGET/results.csv}"
mkdir -p "$(dirname "$OUT")"
python3 - "$TARGET/exact-probe" "$OUT" <<'PY'
import csv, os, platform, re, subprocess, sys
binary, output = sys.argv[1:]
n = os.environ.get('EXACT_N', '100000')
rounds = os.environ.get('EXACT_ROUNDS', '4')
repeats = int(os.environ.get('EXACT_REPEATS', '3'))
modes = os.environ.get('EXACT_MODES', 'original_446d45e ascii_fast normalized hash intern').split()
compiler = subprocess.check_output(['rustc', '--version'], text=True).strip()
system = platform.platform()
mac = sys.platform == 'darwin'
with open(output, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['repeat','order','mode','corpus','records','shortlist','rounds','prep_s','score_s','pairs_per_s','checksum','process_peak_rss_bytes','rustc','platform'])
    for repeat in range(1, repeats + 1):
        for corpus in ['codes', 'lei', 'unicode']:
            # Reverse order on alternating repeats to reduce systematic order bias.
            for order, mode in enumerate(modes if repeat % 2 else modes[::-1], 1):
                result = subprocess.run(['/usr/bin/time', '-l' if mac else '-v', binary, mode, corpus, n, rounds], text=True, capture_output=True, check=True)
                pattern = r'(\d+)\s+maximum resident set size' if mac else r'Maximum resident set size \(kbytes\):\s*(\d+)'
                match = re.search(pattern, result.stderr)
                if not match:
                    raise RuntimeError('RSS unavailable: ' + result.stderr)
                rss = int(match[1]) * (1 if mac else 1024)
                writer.writerow([repeat, order] + result.stdout.strip().split(',') + [rss, compiler, system])
                f.flush()
print('Results: ' + output)
PY
