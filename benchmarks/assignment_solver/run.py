#!/usr/bin/env python3
"""Bounded sequential release benchmarks. No ONNX, third-party modules, or edge CSVs.

Example (build is separate from every measured child):
  python3 benchmarks/assignment_solver/run.py --timeout 60 --budget 1800
  python3 benchmarks/assignment_solver/run.py --timeout 60 --matrix --workers 4 --repeats 4
  python3 benchmarks/assignment_solver/run.py --timeout 120 --sizes 100000 \
      --seeds 0x5eedb1,0x5eedb2 --include-chain --output /tmp/solver-b1
N means vertices PER SIDE. Timeouts are evidence, not a performance gate decision.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parent
PROTOTYPE = ROOT / "prototype"


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def capture(command, cwd=ROOT):
    try:
        result = subprocess.run(command, cwd=cwd, capture_output=True, timeout=15)
        return {"command": command, "returncode": result.returncode,
                "stdout": result.stdout.decode(errors="replace").strip(),
                "stderr": result.stderr.decode(errors="replace").strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": command, "error": str(error)}


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def rss_metadata(usage):
    # getrusage(2): Darwin ru_maxrss is bytes; Linux is KiB. Do not apply
    # /usr/bin/time's differently formatted platform-dependent output rules.
    raw = usage.ru_maxrss
    if sys.platform == "darwin":
        return {"raw": raw, "unit": "bytes", "bytes": raw,
                "source": "wait4 rusage.ru_maxrss; Darwin getrusage(2)"}
    if sys.platform.startswith("linux"):
        return {"raw": raw, "unit": "KiB", "bytes": raw * 1024,
                "source": "wait4 rusage.ru_maxrss; Linux getrusage(2)"}
    return {"raw": raw, "unit": "unknown", "bytes": None,
            "source": "wait4 rusage.ru_maxrss (units not inferred)"}


def measured(command, timeout, stdout_path, stderr_path):
    """Reap exactly this child with wait4; never use cumulative RUSAGE_CHILDREN."""
    start = time.monotonic()
    timed_out = False
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(command, cwd=PROTOTYPE, stdout=stdout,
                                   stderr=stderr, start_new_session=True)
        try:
            while True:
                pid, status, usage = os.wait4(process.pid, os.WNOHANG)
                if pid:
                    break
                if time.monotonic() - start >= timeout:
                    timed_out = True
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass  # Child exited between the nonblocking reap and kill.
                    _, status, usage = os.wait4(process.pid, 0)
                    break
                time.sleep(0.01)
        except BaseException:
            # No orphan solver when the runner is interrupted.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            _, status, _ = os.wait4(process.pid, 0)
            process.returncode = os.waitstatus_to_exitcode(status)
            raise
        process.returncode = os.waitstatus_to_exitcode(status)
    return {"wall_s": time.monotonic() - start,
            "timeout_s": timeout, "timed_out": timed_out,
            "returncode": process.returncode, "peak_rss": rss_metadata(usage),
            "user_cpu_s": usage.ru_utime, "system_cpu_s": usage.ru_stime}


def source_identity():
    paths = sorted((PROTOTYPE / "src").rglob("*.rs")) + sorted((PROTOTYPE / "tests").rglob("*.rs"))
    paths += [PROTOTYPE / "Cargo.toml", PROTOTYPE / "Cargo.lock", Path(__file__)]
    return {str(path.relative_to(ROOT)): sha256(path) for path in paths if path.is_file()}


def csv_ints(text):
    try:
        values = [int(part, 0) for part in text.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not values or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("expected nonnegative integers")
    return values


def configurations(matrix, workers):
    """Legacy default stays sequential SSP; matrix is an explicit opt-in."""
    baseline = {"name": "sequential_ssp", "workers": 1, "batch_admissible": False}
    if not matrix:
        return [baseline]
    return [baseline,
            {"name": "parallel_components", "workers": workers, "batch_admissible": False},
            {"name": "admissible_batching", "workers": 1, "batch_admissible": True},
            {"name": "both", "workers": workers, "batch_admissible": True}]


def rotated_configs(repeats, configs, offset=0):
    """Rotate the first treatment per fixture/repeat; never overlap children."""
    for repeat in range(repeats):
        rotation = (offset + repeat) % len(configs)
        for config in configs[rotation:] + configs[:rotation]:
            yield repeat, config


def check_recipe(graph, expected, fixture):
    """Compare every observed graph, even when its solve subsequently times out."""
    fields = ("recipe", "shape", "n_b", "n_a", "requested_degree", "seed",
              "edges", "scale", "digest_algorithm", "graph_digest")
    signature = {field: graph[field] for field in fields}
    if signature["recipe"] != "sparse-v1":
        raise ValueError("unexpected graph recipe")
    if fixture in expected and expected[fixture] != signature:
        raise ValueError("recipe checksum/metadata mismatch across configurations or repeats")
    expected.setdefault(fixture, signature)
    return signature


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--timeout", type=float, required=True, help="hard per-process seconds, including generation")
    parser.add_argument("--budget", type=float, default=1800, help="total measured campaign seconds, excluding build (default 1800)")
    parser.add_argument("--build-timeout", type=float, default=600)
    parser.add_argument("--sizes", type=csv_ints, default=[1000, 10000, 100000])
    parser.add_argument("--degrees", type=csv_ints, default=[5, 20])
    parser.add_argument("--seeds", type=csv_ints, default=[0x5EEDB1])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--matrix", action="store_true", help="run the rotated 2x2 workers x admissible-batching controls")
    parser.add_argument("--workers", type=int, default=2, help="parallel treatment worker count (>= 2; only with --matrix)")
    parser.add_argument("--include-chain", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if any(not math.isfinite(value) or value <= 0 for value in
           (args.timeout, args.budget, args.build_timeout)) or args.repeats < 1:
        parser.error("timeouts/budget must be finite and positive; repeats >= 1")
    if args.workers < 2:
        parser.error("--workers must be >= 2 for distinct parallel controls")
    configs = configurations(args.matrix, args.workers)
    if any(n < 1 for n in args.sizes) or any(d < 2 for d in args.degrees):
        parser.error("sizes >= 1 and degrees >= 2 required")
    if any(seed > 2**64 - 1 for seed in args.seeds):
        parser.error("seed must fit u64")
    if not hasattr(os, "wait4"):
        parser.error("wait4 required for per-process peak RSS (macOS/Linux supported)")
    output = (args.output or ROOT / "results" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")).resolve()
    output.mkdir(parents=True, exist_ok=False)
    identity = {
        "started_utc": now(), "runner_argv": sys.argv,
        "parameters": {key: str(value) if isinstance(value, Path) else value
                       for key, value in vars(args).items()},
        "host": {"platform": platform.platform(), "machine": platform.machine(),
                 "processor": platform.processor(), "logical_cpus": os.cpu_count(),
                 "python": sys.version},
        "rustc": capture(["rustc", "-Vv"]), "cargo": capture(["cargo", "--version"]),
        "git_head": capture(["git", "rev-parse", "HEAD"]),
        "git_status": capture(["git", "status", "--short"]),
        "source_sha256": source_identity(),
        "environment": {key: os.environ.get(key) for key in
                        ("RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "CARGO_BUILD_TARGET", "CARGO_PROFILE_RELEASE_LTO", "CARGO_PROFILE_RELEASE_OPT_LEVEL")},
        "configurations": configs,
        "protocol": "sequential fresh processes from one pinned binary; no warmups; rotate treatment start per fixture/repeat; runtime excludes build; N per side; peak RSS includes graph generation and validation; raw graph regenerated from sparse-v1 recipe and seed; verify identical recipe checksums across controls/repeats; solve_wall_s is elapsed solve wall time, build_s/search_s/extract_s are component phase sums plus serial overhead, not parallel wall time; no comparison to unmatched historical graphs",
    }
    if sys.platform == "darwin":
        identity["host"]["hardware"] = capture(["sysctl", "-n", "machdep.cpu.brand_string", "hw.memsize"])
    elif sys.platform.startswith("linux"):
        for name in ("cpuinfo", "meminfo"):
            path = Path("/proc") / name
            if path.exists():
                identity["host"][name] = path.read_text()
    save(output / "identity.json", identity)
    build_command = ["cargo", "build", "--release", "--manifest-path", str(PROTOTYPE / "Cargo.toml"), "--bin", "measure", "--message-format=json"]
    build = {"command": build_command, **measured(build_command, args.build_timeout,
                                                  output / "build.stdout", output / "build.stderr")}
    save(output / "build.json", build)
    if build.get("timed_out") or build.get("returncode", 0) != 0:
        print(f"Build failed/timed out; evidence: {output}", file=sys.stderr)
        return 1
    built_binary = None
    for line in (output / "build.stdout").read_text().splitlines():
        artifact = json.loads(line)
        if (artifact.get("reason") == "compiler-artifact"
                and artifact.get("target", {}).get("name") == "measure"
                and artifact.get("executable")):
            built_binary = Path(artifact["executable"])
    sources = source_identity()
    if built_binary is None or sources != identity["source_sha256"]:
        print(f"No pinned executable or sources changed during build; evidence: {output}", file=sys.stderr)
        return 1
    binary = output / "measure"
    shutil.copy2(built_binary, binary)
    for name in sources:
        destination = output / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    identity.update(binary_path=str(binary), binary_sha256=sha256(binary),
                    source_sha256=sources, source_snapshot="source")
    save(output / "identity.json", identity)
    started = time.monotonic()
    records = []
    recipes = {}
    failed = False
    for n in sorted(args.sizes):
        cases = [(shape, degree) for shape in ("skewed", "giant") for degree in args.degrees]
        if args.include_chain:
            cases.append(("chain", 2))
        for shape, degree in cases:
            for seed in args.seeds:
                for repeat, config in rotated_configs(args.repeats, configs, len(records) // len(configs)):
                    name = f"{len(records):03d}-{shape}-n{n}-d{degree}-s{seed}-r{repeat}"
                    if args.matrix:
                        name += "-" + config["name"]
                    command = [str(binary), str(n), str(degree), shape, str(seed)]
                    if args.matrix:
                        command += ["--workers", str(config["workers"])]
                        if config["batch_admissible"]:
                            command.append("--batch-admissible")
                    record = {"id": name, "started_utc": now(), "command": command,
                              "recipe": "sparse-v1", "n_b": n, "n_a": n,
                              "shape": shape, "degree": degree, "seed": seed, "repeat": repeat,
                              "configuration": config, "sequence": len(records),
                              "identity_file": "identity.json", "binary_sha256": identity["binary_sha256"]}
                    remaining = args.budget - (time.monotonic() - started)
                    if remaining <= 0:
                        record["status"] = "skipped_budget"
                    else:
                        stdout = output / (name + ".stdout.jsonl")
                        stderr = output / (name + ".stderr")
                        record.update(stdout=stdout.name, stderr=stderr.name)
                        try:
                            record.update(measured(command, min(args.timeout, remaining), stdout, stderr))
                            events = []
                            parse_errors = []
                            for line in stdout.read_text().splitlines():
                                try:
                                    events.append(json.loads(line))
                                except ValueError as error:
                                    parse_errors.append(str(error))
                            record["events"] = events
                            if parse_errors:
                                record["stdout_parse_errors"] = parse_errors
                            if events and events[0].get("event") == "graph":
                                record["verified_recipe"] = check_recipe(
                                    events[0], recipes, (n, degree, shape, seed))
                            success = (not parse_errors and len(events) == 2
                                       and events[0].get("event") == "graph"
                                       and events[1].get("status") == "ok"
                                       and events[1].get("workers") == config["workers"]
                                       and events[1].get("batch_admissible") == config["batch_admissible"])
                            record["status"] = ("timeout" if record["timed_out"] else
                                                "ok" if record["returncode"] == 0 and success else "failed")
                        except (OSError, ValueError, KeyError) as error:
                            record.update(status="runner_error", error=str(error))
                    failed |= record["status"] != "ok"
                    save(output / (name + ".json"), record)
                    records.append({"id": name, "status": record["status"]})
                    save(output / "index.json", {"runs": records, "elapsed_s": time.monotonic() - started})
                    print(f"{name}: {record['status']}", flush=True)
    print(f"Evidence: {output}")
    return int(failed)


class RunnerTests(unittest.TestCase):
    """Run with: python3 -m unittest discover -s benchmarks/assignment_solver -p run.py."""

    def test_matrix_is_factorial_and_rotates(self):
        configs = configurations(True, 4)
        self.assertEqual({(item["workers"], item["batch_admissible"]) for item in configs},
                         {(1, False), (4, False), (1, True), (4, True)})
        runs = list(rotated_configs(4, configs))
        for repeat in range(4):
            group = [config for index, config in runs if index == repeat]
            self.assertEqual(group[0], configs[repeat])
            self.assertCountEqual(group, configs)
        self.assertEqual(list(rotated_configs(1, configs, 1))[0][1], configs[1])
        self.assertEqual(configurations(False, 4), [configs[0]])

    def test_recipe_checks_reject_different_fixture_graphs(self):
        graph = {"recipe": "sparse-v1", "shape": "skewed", "n_b": 10, "n_a": 10,
                 "requested_degree": 5, "seed": 123, "edges": 40, "scale": 10**9,
                 "digest_algorithm": "fnv1a64", "graph_digest": "abc"}
        expected = {}
        check_recipe(graph, expected, (10, 5, "skewed", 123))
        check_recipe(dict(graph, generation_s=99), expected, (10, 5, "skewed", 123))
        with self.assertRaisesRegex(ValueError, "checksum"):
            check_recipe(dict(graph, graph_digest="def"), expected, (10, 5, "skewed", 123))
        with self.assertRaises(KeyError):
            check_recipe({"recipe": "sparse-v1"}, expected, "incomplete")

    def test_child_success_and_peak_rss(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = measured([sys.executable, "-c", "print(123)"], 5,
                              root / "stdout", root / "stderr")
            self.assertEqual(result["returncode"], 0, result)
            self.assertFalse(result["timed_out"], result)
            self.assertEqual((root / "stdout").read_text().strip(), "123")
            if sys.platform == "darwin" or sys.platform.startswith("linux"):
                self.assertGreater(result["peak_rss"]["bytes"], 0, result)

    def test_child_timeout_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = measured([sys.executable, "-c", "import time; time.sleep(10)"],
                              0.05, root / "stdout", root / "stderr")
            self.assertTrue(result["timed_out"], result)
            self.assertEqual(result["returncode"], -signal.SIGKILL, result)
            self.assertLess(result["wall_s"], 5, result)


if __name__ == "__main__":
    raise SystemExit(main())
