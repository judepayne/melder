#!/usr/bin/env python3
"""Stdlib-only ordered HTTP competition benchmark; never builds melder."""
import argparse
import json
import math
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

HERE = Path(__file__).resolve().parent


def percentile(values, percent):
    """Nearest-rank percentile (milliseconds)."""
    return sorted(values)[max(0, math.ceil(len(values) * percent / 100) - 1)] if values else None


def workload(groups):
    """Return mirrored gadgets: one receiver and two competing proposers."""
    result = []
    for side in ("a", "b"):
        other = "b" if side == "a" else "a"
        for index in range(groups):
            group = f"{side}{index:06d}"
            def record(role, competitive):
                return {"id": f"{group}-{role}", "group": group,
                        "name": f"company {group} trading", "stable": "1",
                        "competitive": str(competitive)}
            result.append({"side": side, "receiver_side": other,
                           "receiver": record("receiver", 2),
                           "incumbent": record("incumbent", 1),
                           "arrival": record("arrival", 2)})
    return result


def config(root, strategy, cap):
    command = shlex.join([sys.executable, str(HERE / "unused_encoder.py")])
    return {
        "job": {"name": "assignment_live"},
        "datasets": {s: {"path": str(root / f"{s}.csv"), "id_field": "id"}
                     for s in ("a", "b")},
        "cross_map": {"path": str(root / "crossmap.csv"),
                      "a_id_field": "a_id", "b_id_field": "b_id"},
        "embeddings": {"remote_encoder_cmd": command,
                       "a_cache_dir": str(root / "cache_a"),
                       "b_cache_dir": str(root / "cache_b")},
        "performance": {"encoder_pool_size": 1},
        "vector_backend": "flat", "top_n": 10,
        "ann_candidates": 20, "bm25_candidates": 20,
        "blocking": {"enabled": True, "fields": [
            {"field_a": "group", "field_b": "group"}]},
        "match_fields": [
            {"method": "bm25", "weight": 0.1,
             "fields": [{"field_a": "name", "field_b": "name"}]},
            {"field_a": "stable", "field_b": "stable", "method": "numeric", "weight": 0.6},
            {"field_a": "competitive", "field_b": "competitive", "method": "numeric", "weight": 0.3}],
        "thresholds": {"auto_match": 0.55, "review_floor": 0.4},
        "output": {"csv_dir_path": str(root / "output")},
        "live": {"skip_initial_match": True, "match_log_path": str(root / "wal.ndjson"),
                 "assignment": {"strategy": strategy, "max_displacements": cap}},
    }


class Client:
    def __init__(self, port, timeout):
        self.origin = f"http://127.0.0.1:{port}"
        self.base = self.origin + "/api/v1/"
        self.timeout = timeout
        # Ignore proxy environment variables for isolated loopback traffic.
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def call(self, path, payload=None):
        data = None if payload is None else json.dumps(payload).encode("ascii")
        url = self.origin + path if path.startswith("/") else self.base + path
        req = urllib.request.Request(url, data=data,
                                     headers={"Content-Type": "application/json"})
        with self.opener.open(req, timeout=self.timeout) as response:
            result = json.load(response)
        if isinstance(result, dict) and (result.get("status") == "error" or "error" in result):
            raise RuntimeError(f"{path}: {result}")
        return result

    def add(self, side, record):
        return self.call(f"{side}/add", {"record": record})


def pairs(client):
    result = set()
    offset = 0
    while True:
        page = client.call(f"crossmap/pairs?offset={offset}&limit=500")
        rows = page["pairs"]
        result.update((r["a_id"], r["b_id"]) for r in rows)
        offset += len(rows)
        if offset >= page["total"]:
            return result
        if not rows:
            raise RuntimeError("pair pagination made no progress")


def pair(gadget, role):
    ids = (gadget[role]["id"], gadget["receiver"]["id"])
    return ids if gadget["side"] == "a" else ids[::-1]


def run_one(binary, root, strategy, args):
    root.mkdir()
    for side in ("a", "b"):
        (root / f"{side}.csv").write_text("id,group,name,stable,competitive\n", encoding="ascii")
    config_path = root / "config.yaml"
    # JSON is a YAML subset, avoiding a PyYAML dependency.
    config_path.write_text(json.dumps(config(root, strategy, args.cap), indent=2), encoding="ascii")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    client = Client(port, args.timeout)
    report = {"strategy": strategy, "errors": [], "sides": {}}
    log_path = root / "server.log"
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen([str(binary), "serve", "--config", str(config_path),
                                 "--port", str(port), "--bind", "127.0.0.1"],
                                cwd=root, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.perf_counter() + args.startup_timeout
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f"server exited {proc.returncode}; see {log_path}")
                try:
                    client.call("health")
                    break
                except (OSError, ValueError):
                    if time.perf_counter() >= deadline:
                        raise RuntimeError(f"startup timeout; see {log_path}")
                    time.sleep(0.05)
            gadgets = workload(args.groups)
            # All setup is ordered, unmeasured HTTP traffic. No manual confirms.
            for g in gadgets:
                client.add(g["receiver_side"], g["receiver"])
                client.add(g["side"], g["incumbent"])
            expected_setup = {pair(g, "incumbent") for g in gadgets}
            if pairs(client) != expected_setup:
                raise RuntimeError("setup did not produce exactly the automatic incumbent pairs")
            latencies = []
            displacements = caps = changes = orphans = successful = 0
            measured_start = time.perf_counter()
            for side in ("a", "b"):
                side_latencies = []
                side_displacements = side_caps = side_errors = 0
                side_start = time.perf_counter()
                for g in (g for g in gadgets if g["side"] == side):
                    start = time.perf_counter()
                    try:
                        response = client.add(side, g["arrival"])
                        successful += 1
                        assignment = response.get("assignment")
                        if strategy == "deferred_acceptance" and not isinstance(assignment, dict):
                            raise RuntimeError("enabled add response missing assignment summary")
                        assignment = assignment or {}
                        d = assignment.get("displacements", 0)
                        c = int(assignment.get("limit_reached", False))
                        displacements += d
                        side_displacements += d
                        caps += c
                        side_caps += c
                        changes += len(assignment.get("changes", []))
                        orphans += len(assignment.get("unmatched", []))
                    except (OSError, ValueError, RuntimeError) as exc:
                        side_errors += 1
                        report["errors"].append(f"{side}/{g['arrival']['id']}: {exc}")
                    finally:
                        latency = (time.perf_counter() - start) * 1000
                        side_latencies.append(latency)
                        latencies.append(latency)
                elapsed = time.perf_counter() - side_start
                report["sides"][side] = metrics(side_latencies, elapsed)
                report["sides"][side].update(displacements=side_displacements,
                                             cap_hits=side_caps, errors=side_errors)
            elapsed = time.perf_counter() - measured_start
            report.update(metrics(latencies, elapsed))
            report.update(successful_http_requests=successful, displacements=displacements,
                          assignment_changes=changes, cap_hits=caps,
                          response_orphan_events=orphans)
            final_pairs = pairs(client)
            preferred = {pair(g, "arrival") for g in gadgets}
            report.update(final_pairs=len(final_pairs), preferred_pairs=len(final_pairs & preferred),
                          incumbent_pairs_retained=len(final_pairs & expected_setup),
                          stats=client.call("crossmap/stats"))
            report["final_unmatched"] = {
                s: client.call(f"{s}/unmatched?limit=1")["total"] for s in ("a", "b")}
            if strategy == "deferred_acceptance":
                if displacements <= 0:
                    report["errors"].append("required observed assignment.displacements > 0")
                for side in ("a", "b"):
                    if report["sides"][side]["displacements"] <= 0:
                        report["errors"].append(f"no observed displacement on mirror side {side}")
                expected_final = preferred
            else:
                expected_final = expected_setup
            if final_pairs != expected_final:
                report["errors"].append("final pairs differ from expected competition outcome")
            if len({a for a, _ in final_pairs}) != len(final_pairs) or len({b for _, b in final_pairs}) != len(final_pairs):
                report["errors"].append("crossmap is not bijective")
            if report["final_unmatched"] != {"a": args.groups, "b": args.groups}:
                report["errors"].append("unexpected orphan counts (one loser per gadget expected)")
        except (OSError, ValueError, RuntimeError, KeyError) as exc:
            report["errors"].append(str(exc))
        finally:
            if proc.poll() is None:
                try:
                    client.call("/admin/shutdown", {})
                    proc.wait(timeout=15)
                except (OSError, ValueError, subprocess.TimeoutExpired):
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
            if proc.returncode != 0:
                report["errors"].append(f"server exit code {proc.returncode}")
    if "UNEXPECTED_ENCODE" in log_path.read_text(encoding="utf-8", errors="replace"):
        report["errors"].append("unexpected embedding call; no-model workload violated")
    report["passed"] = not report["errors"]
    return report


def metrics(latencies, elapsed):
    return {"requests": len(latencies), "wall_seconds": elapsed,
            "requests_per_second": len(latencies) / elapsed if elapsed else 0,
            "latency_ms": {f"p{p}": percentile(latencies, p) for p in (50, 95, 99)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True, help="existing meld binary (never built)")
    parser.add_argument("--groups", type=int, default=100, help="competition groups per mirror side")
    parser.add_argument("--cap", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=30, help="HTTP timeout seconds")
    parser.add_argument("--startup-timeout", type=float, default=60)
    parser.add_argument("--output", type=Path, help="optional persistent JSON report")
    parser.add_argument("--keep-workdir", action="store_true", help="retain temp configs, WALs and logs")
    args = parser.parse_args()
    binary = args.binary.resolve()
    if not binary.is_file() or args.groups < 1 or args.cap < 0 or min(args.timeout, args.startup_timeout) <= 0:
        parser.error("need an existing binary, groups >= 1, cap >= 0 and positive timeouts")
    temp = None
    if args.keep_workdir:
        root = Path(tempfile.mkdtemp(prefix="assignment-live-"))
    else:
        temp = tempfile.TemporaryDirectory(prefix="assignment-live-")
        root = Path(temp.name)
    try:
        report = {"binary": str(binary), "groups_per_side": args.groups,
                  "cap": args.cap, "workdir": str(root), "workdir_retained": args.keep_workdir,
                  "runs": [run_one(binary, root / strategy, strategy, args)
                           for strategy in ("greedy", "deferred_acceptance")]}
        report["passed"] = all(run["passed"] for run in report["runs"])
        text = json.dumps(report, indent=2) + "\n"
        if args.output:
            args.output.write_text(text, encoding="ascii")
        print(text, end="")
        return 0 if report["passed"] else 1
    finally:
        if temp is not None:
            temp.cleanup()


if __name__ == "__main__":
    sys.exit(main())
