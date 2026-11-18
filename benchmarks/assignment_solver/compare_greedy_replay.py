#!/usr/bin/env python3
r"""Compare output-only replay of frozen phase-2B greedy logs (not matching).

Example (binaries must already be built):
  python3 benchmarks/assignment_solver/compare_greedy_replay.py \
    --before /tmp/meld-before --after /tmp/meld-after --out /tmp/replay-comparison

Uses the same original WAL and scoring log for every fresh subprocess. No graph
runs, encoder work, builds, or end-to-end timings are included. CSVs must match
byte-for-byte across binaries and repetitions. --sqlite additionally writes and
compares SQLite schema and sorted table contents, not physical database bytes.
Requires only Python's standard library. Run on an otherwise idle machine.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import sqlite3
import statistics
import subprocess
import time

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CAPTURE = Path('/tmp/melder-b2-final.z9pZQC/warm')
CSV_NAMES = ('relationships.csv', 'unmatched.csv', 'candidates.csv')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def replace_section(text, name, body):
    """Replace a top-level YAML mapping without a PyYAML dependency."""
    pattern = rf'^{re.escape(name)}:[^\n]*\n(?:(?!^[^\s#])[^\n]*(?:\n|$))*'
    replacement = f'{name}:\n{body}\n'
    text, count = re.subn(pattern, lambda _: replacement, text, flags=re.MULTILINE)
    if count > 1:
        raise ValueError(f'duplicate config section: {name}')
    return text if count else text.rstrip() + '\n\n' + replacement


def one_file(directory, pattern):
    paths = sorted(directory.glob(pattern))
    if len(paths) != 1 or not paths[0].is_file():
        raise ValueError(f'expected one {pattern} in {directory}, found {paths}')
    return paths[0].resolve()


def sqlite_contents(path):
    """Fingerprint all logical rows in a read-only DB, preserving duplicates."""
    def quote(name):
        return '"' + name.replace('"', '""') + '"'

    with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True) as conn:
        schema = conn.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master "
            "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
        ).fetchall()
        tables = {}
        for (name,) in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ):
            columns = list(conn.execute(f'PRAGMA table_info({quote(name)})'))
            order = ', '.join(str(i + 1) for i in range(len(columns)))
            digest = hashlib.sha256()
            count = 0
            for row in conn.execute(f'SELECT * FROM {quote(name)} ORDER BY {order}'):
                # repr retains NULL, numeric types and binary values unambiguously.
                digest.update((repr(row) + '\n').encode('utf-8'))
                count += 1
            tables[name] = dict(rows=count, sha256=digest.hexdigest())
    return dict(schema=schema, tables=tables)


def save_report(out, report):
    temporary = out / 'report.json.tmp'
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(out / 'report.json')


def run_suite(args):
    # Validate all sources before creating output or launching a timed process.
    sources = {}
    for size in args.sizes:
        source = args.capture / f'{size}-greedy-0'
        config = (source / 'config.yaml').resolve(strict=True)
        text = config.read_text()
        if re.search(r'^\s*strategy:\s*graph\b', text, re.MULTILINE):
            raise ValueError(f'not a greedy config: {config}')
        wal = one_file(source / 'output', '*.match_log*.ndjson')
        scoring = one_file(source / 'output', '*.scoring_log.ndjson*')
        if not (scoring.name.endswith('.ndjson') or scoring.name.endswith('.ndjson.zst')):
            raise ValueError(f'unsupported scoring log suffix: {scoring}')
        sources[size] = dict(config=config, text=text, wal=wal, scoring=scoring)
    for binary in (args.before, args.after):
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise ValueError(f'not an executable file: {binary}')

    args.out.mkdir(parents=True, exist_ok=False)
    binaries = {}
    for label in ('before', 'after'):
        pinned = args.out / f'meld-{label}'
        shutil.copy2(getattr(args, label), pinned)
        binaries[label] = pinned
    fingerprints = {
        str(path): sha256(path)
        for source in sources.values()
        for path in (source['config'], source['wal'], source['scoring'])
    }
    report = dict(
        kind='greedy-output-only-replay', status='running',
        platform=platform.platform(), python=platform.python_version(), cwd=str(REPO),
        driver_sha256=sha256(Path(__file__).resolve()),
        capture=str(args.capture), repeats=args.repeats, sqlite=args.sqlite,
        timing='fresh subprocess wall seconds; configs, hashes and comparisons excluded; no cache eviction',
        rayon_num_threads='4',
        binaries={label: dict(source=str(getattr(args, label)), pinned=str(path), sha256=sha256(path))
                  for label, path in binaries.items()},
        inputs=fingerprints, runs=[], summaries={},
    )
    save_report(args.out, report)
    try:
        for size, source in sources.items():
            reference = None
            for rep in range(args.repeats):
                # Alternate the first binary to reduce systematic warm-page-cache bias.
                order = ('before', 'after') if rep % 2 == 0 else ('after', 'before')
                for label in order:
                    directory = args.out / f'{size}-{label}-{rep}'
                    directory.mkdir()
                    csv_dir = directory / 'csv'
                    config = directory / 'config.yaml'
                    # Export locates the fixed scoring log through output.csv_dir_path,
                    # while --out-dir directs new CSVs elsewhere. Never copy/rewrite logs.
                    text = replace_section(source['text'], 'output',
                        f'  csv_dir_path: {json.dumps(str(source["scoring"].parent))}\n' +
                        (f'  db_path: {json.dumps(str(directory / "output.sqlite"))}\n' if args.sqlite else ''))
                    text = replace_section(text, 'live',
                        f'  match_log_path: {json.dumps(str(source["wal"]))}\n')
                    compression = 'zstd' if source['scoring'].suffix == '.zst' else 'none'
                    text = replace_section(text, 'scoring_log',
                        f'  enabled: true\n  compression: {compression}\n')
                    config.write_text(text)
                    command = [str(binaries[label]), 'export', '--config', str(config), '--out-dir', str(csv_dir)]
                    with (directory / 'stdout.log').open('w') as stdout, (directory / 'stderr.log').open('w') as stderr:
                        start = time.perf_counter()
                        result = subprocess.run(command, cwd=REPO,
                            env={**os.environ, 'RAYON_NUM_THREADS': '4'}, stdout=stdout, stderr=stderr,
                            timeout=args.timeout, check=False)
                        seconds = time.perf_counter() - start
                    run = dict(size=size, binary=label, rep=rep, seconds=seconds,
                               command=command, returncode=result.returncode)
                    report['runs'].append(run)
                    save_report(args.out, report)
                    if result.returncode:
                        raise RuntimeError(f'export failed: see {directory}/stderr.log')
                    actual_names = sorted(p.name for p in csv_dir.glob('*.csv'))
                    if actual_names != sorted(CSV_NAMES):
                        raise ValueError(f'unexpected CSV outputs: {directory}: {actual_names}')
                    signature = dict(csv={name: dict(bytes=(csv_dir / name).stat().st_size,
                                                    sha256=sha256(csv_dir / name)) for name in CSV_NAMES})
                    if args.sqlite:
                        signature['sqlite'] = sqlite_contents(directory / 'output.sqlite')
                    run['outputs'] = signature
                    if reference is None:
                        reference = signature
                    if signature != reference:
                        raise ValueError(f'output mismatch: {directory} (versus first {size} replay)')
                    run['equivalent'] = True
                    save_report(args.out, report)
                    print(f'{size} {label} {rep}: {seconds:.3f}s; outputs identical', flush=True)
            medians = {
                label: statistics.median(r['seconds'] for r in report['runs']
                                         if r['size'] == size and r['binary'] == label)
                for label in binaries
            }
            report['summaries'][size] = dict(median_seconds=medians,
                speedup=medians['before'] / medians['after'])
            print(f'{size}: before {medians["before"]:.3f}s; after {medians["after"]:.3f}s; '
                  f'{medians["before"] / medians["after"]:.2f}x', flush=True)
        report['status'] = 'passed'
    except Exception as error:
        report['status'] = 'failed'
        report['error'] = str(error)
        raise
    finally:
        changed = [path for path, digest in fingerprints.items() if sha256(Path(path)) != digest]
        report['inputs_unchanged'] = not changed
        report['binaries_unchanged'] = all(
            sha256(path) == report['binaries'][label]['sha256'] for label, path in binaries.items())
        if changed or not report['binaries_unchanged']:
            report['status'] = 'failed'
            report['integrity_error'] = dict(changed_inputs=changed, binaries_unchanged=report['binaries_unchanged'])
        save_report(args.out, report)
    if report['status'] != 'passed':
        raise RuntimeError(f'integrity check failed: {args.out}/report.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--before', required=True, type=Path, help='existing baseline executable (copied/pinned)')
    parser.add_argument('--after', required=True, type=Path, help='existing optimized executable (copied/pinned)')
    parser.add_argument('--capture', type=Path, default=DEFAULT_CAPTURE, help='warm phase-2B run root')
    parser.add_argument('--out', required=True, type=Path, help='new directory; must not exist')
    parser.add_argument('--sizes', nargs='+', choices=('10k', '100k'), default=['10k', '100k'])
    parser.add_argument('--repeats', type=int, choices=range(1, 10), default=3)
    parser.add_argument('--timeout', type=float, default=600, help='seconds per export')
    parser.add_argument('--sqlite', action='store_true', help='also generate and compare logical SQLite output')
    args = parser.parse_args()
    for name in ('before', 'after', 'capture', 'out'):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if len(set(args.sizes)) != len(args.sizes) or args.timeout <= 0:
        parser.error('sizes must be unique and timeout must be positive')
    run_suite(args)


if __name__ == '__main__':
    main()
