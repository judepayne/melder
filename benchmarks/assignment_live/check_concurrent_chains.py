#!/usr/bin/env python3
"""Local HTTP contention and paired chain benchmark (stdlib only; never builds).

Examples:
  python3 benchmarks/assignment_live/check_concurrent_chains.py --binary /path/meld --mode contention
  python3 benchmarks/assignment_live/check_concurrent_chains.py --binary /path/meld --mode performance --groups 100 --repeats 3

Performance is opt-in: fresh servers receive identical ordered setup and roots
with 1 then 10 clients. Synthetic two-dimensional embeddings go through the real
scoring pipeline, not a mocked assignment graph. No inference/network model is
involved. These are targeted chain costs, not production throughput predictions.
Contention pauses the first root's encoder *under the assignment mutation gate*
(upsert_record takes assignment_guard before encoding). Other HTTP requests are
started while that pause is observed. This proves an internal gate holder and
client overlap, not that every waiting request has reached a server-side barrier.
Admin flush is deliberately allowed to observe an atomic chain prefix.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import shlex
import socket
import sqlite3
import struct
import subprocess
import sys
import tempfile
import threading
import time

from run import Client, config, metrics, pairs

SCRIPT = Path(__file__).resolve()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def record(group, role, angle):
    return {"record_id": f"{group}-{role}", "group": group,
            "vector": str(angle)}


def gadget(group, side="a", root_angle=0):
    # cos(15) > cos(20) > cos(25) > .7. r0 prefers root over p0;
    # p0 prefers r0 over r1, but r1 prefers p0 over p1. Thus root
    # displaces p0, then p0 displaces p1: exactly TWO steps, no free sink.
    return {"side": side, "other": "b" if side == "a" else "a",
            "r0": record(group, "r0", 0), "r1": record(group, "r1", 35),
            "p0": record(group, "p0", 15), "p1": record(group, "p1", 60),
            "root": record(group, "root", root_angle)}


def edge(g, proposer, receiver):
    ids = (g[proposer]["record_id"], g[receiver]["record_id"])
    return ids if g["side"] == "a" else ids[::-1]


def setup(client, g, records):
    # p1-r0 is below threshold; when r1 arrives p0 reciprocally rejects it
    # in favour of r0, so r1 can accept p1. No protected/manual links.
    for role in ("r0", "p0", "p1", "r1"):
        side = g["other"] if role.startswith("r") else g["side"]
        response = client.add(side, g[role])
        require(response.get("assignment", {}).get("displacements") == 0,
                f"unexpected setup displacement: {response}")
        records[side][g[role]["record_id"]] = g[role]


def summary(response, expected=None):
    value = response.get("assignment")
    require(isinstance(value, dict), f"missing assignment summary: {response}")
    require(not value["limit_reached"], f"unexpected displacement cap: {value}")
    if expected is not None:
        require(value["displacements"] == expected,
                f"expected {expected} actual displacements, got {value}")
    displaced = [c for c in value["changes"] if c["expected"]]
    require(len(displaced) == value["displacements"], "summary/audit displacement mismatch")
    for change in displaced:
        require(len(change["expected"]) == len(change["replacements"]) == 1,
                f"unexpected chain step shape: {change}")
        replacement = change["replacements"][0]
        receiver_score = replacement["score_b_to_a" if change["proposer"]["side"] == "a"
                                     else "score_a_to_b"]
        require(receiver_score > change["previous_receiver_score"],
                f"non-strict receiver improvement: {change}")
    if expected == 2:
        require(len(value["changes"]) == 2
                and value["changes"][1]["proposer"] in value["changes"][0]["newly_unmatched"],
                "two displacements were not one connected orphan chain")
    return value


def encoder(root):
    """Wire-compatible deterministic encoder, with one optional file latch."""
    print(json.dumps({"type": "handshake", "protocol_version": 1,
                      "vector_dim": 2, "model_id": "assignment-chain-angles-v1",
                      "max_batch_size": 1000}), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        texts = request["texts"]
        # Only contention arms the latch. Performance never waits/sleeps.
        arm = root / "hold.json"
        if arm.exists() and json.loads(arm.read_text())["text"] in texts:
            (root / "encoder-entered").write_text(str(time.monotonic()))
            deadline = time.monotonic() + 20
            while not (root / "encoder-release").exists():
                if time.monotonic() >= deadline:
                    raise RuntimeError("encoder latch deadline exceeded")
                time.sleep(.005)
        vectors = []
        for text in texts:
            angle = math.radians(float(text))
            vectors.extend((math.cos(angle), math.sin(angle)))
        payload = struct.pack(f"<{len(vectors)}f", *vectors)
        envelope = {"type": "encode_response", "results": [{"ok": True} for _ in texts],
                    "vector_count": len(texts), "vector_dim": 2}
        sys.stdout.buffer.write(json.dumps(envelope).encode() + b"\n")
        sys.stdout.buffer.write(struct.pack("<I", len(payload)) + payload)
        sys.stdout.buffer.flush()


def prepare(root, backend):
    root.mkdir(parents=True)
    for side in ("a", "b"):
        (root / f"{side}.csv").write_text("record_id,group,vector\n")
    cfg = config(root, "deferred_acceptance", 32)
    for dataset in cfg["datasets"].values():
        dataset["id_field"] = "record_id"  # Never use SQLite's reserved internal id.
    cfg["embeddings"]["remote_encoder_cmd"] = shlex.join(
        [sys.executable, str(SCRIPT), "--encoder-root", str(root)])
    cfg["match_fields"] = [{"field_a": "vector", "field_b": "vector",
                            "method": "embedding", "weight": 1.0}]
    cfg["thresholds"] = {"auto_match": .7, "review_floor": .69}
    if backend == "sqlite":
        cfg["live"]["db_path"] = str(root / "state.sqlite")
    (root / "config.yaml").write_text(json.dumps(cfg, indent=2))


@contextmanager
def server(binary, root, args):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    client = Client(port, args.timeout)
    with (root / "server.log").open("w") as log:
        process = subprocess.Popen(
            [str(binary), "serve", "--config", str(root / "config.yaml"),
             "--port", str(port), "--bind", "127.0.0.1"],
            cwd=root, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + args.startup_timeout
            while True:
                require(process.poll() is None, f"server exited; see {root / 'server.log'}")
                try:
                    client.call("health")
                    break
                except OSError:
                    require(time.monotonic() < deadline, "startup deadline exceeded")
                    time.sleep(.05)
            yield client
        finally:
            (root / "encoder-release").touch()
            if process.poll() is None:
                try:
                    client.call("/admin/shutdown", {})
                    process.wait(timeout=args.timeout)
                except (OSError, ValueError, subprocess.TimeoutExpired):
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)


def all_rows(client, path, key):
    result = []
    while True:
        page = client.call(f"{path}?offset={len(result)}&limit=500")
        rows = page[key]
        result.extend(rows)
        if len(result) >= page["total"]:
            return result
        require(bool(rows), f"pagination stalled: {path}")


def verify(client, root, backend, records, expected):
    actual = pairs(client)
    require(actual == expected, f"unexpected membership: {actual ^ expected}")
    require(len({a for a, _ in actual}) == len(actual)
            and len({b for _, b in actual}) == len(actual), "non-bijective CrossMap")
    stats = client.call("crossmap/stats")
    require(stats["crossmap_pairs"] == len(actual), f"wrong pair stats: {stats}")
    for side, position in (("a", 0), ("b", 1)):
        matched = {p[position]: p[1 - position] for p in actual}
        unmatched = {r["id"] for r in all_rows(client, f"{side}/unmatched", "records")}
        require(unmatched == records[side].keys() - matched.keys(),
                f"{side}: stale unmatched set: {unmatched}")
        require(stats[f"records_{side}"] == len(records[side])
                and stats[f"unmatched_{side}"] == len(unmatched), f"wrong stats: {stats}")
        for rid, record_value in records[side].items():
            query = client.call(f"{side}/query?id={rid}")
            require(query["record"] == record_value, f"stale query record: {query}")
            state = query["crossmap"]
            require(state["status"] == ("matched" if rid in matched else "unmatched")
                    and state.get("paired_id") == matched.get(rid), f"stale query flag: {query}")
    reviews = all_rows(client, "review/list", "reviews")
    # Rejected automatic proposals are intentionally demoted to review, even
    # above auto_match. Only unmatched losers may own these reviews; their
    # candidates may legitimately already be matched to someone stronger.
    expected_reviews = set()
    for side, position in (("a", 0), ("b", 1)):
        matched_ids = {p[position] for p in actual}
        for rid in records[side].keys() - matched_ids:
            group, role = rid.rsplit("-", 1)
            if role in ("p1", "root"):
                expected_reviews.add((side, rid, f"{group}-{'r1' if role == 'p1' else 'r0'}"))
    review_keys = {(r["side"], r["id"], r["candidate_id"]) for r in reviews}
    require(len(review_keys) == len(reviews) and review_keys == expected_reviews,
            f"unexpected review ownership: {reviews}; expected {expected_reviews}")
    for review in reviews:
        other = "b" if review["side"] == "a" else "a"
        left = records[review["side"]][review["id"]]
        right = records[other][review["candidate_id"]]
        score = math.cos(math.radians(float(left["vector"]) - float(right["vector"])))
        require(abs(review["score"] - score) < 1e-5, f"stale review score: {review}")
    client.call("/admin/flush", {})
    with (root / "output" / "relationships.csv").open(newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        # The writer uses configured dataset ID names, B then A. Both are
        # record_id here, so DictReader would silently lose one ID column.
        require(header[:2] == ["record_id", "record_id"], f"unexpected CSV IDs: {header}")
        relationships = [dict(zip(["b_id", "a_id"] + header[2:], row)) for row in reader]
    exported = [(r["a_id"], r["b_id"]) for r in relationships if r["relationship_type"] == "match"]
    require(len(exported) == len(set(exported)) and set(exported) == actual,
            f"flushed relationships disagree with CrossMap: {exported}")
    review_pairs = {(rid, candidate) if side == "a" else (candidate, rid)
                    for side, rid, candidate in review_keys}
    exported_reviews = [(r["a_id"], r["b_id"]) for r in relationships
                        if r["relationship_type"] == "review"]
    require(len(exported_reviews) == len(review_pairs) and set(exported_reviews) == review_pairs,
            f"flushed reviews disagree with API: {exported_reviews}; expected {review_pairs}")
    if backend == "sqlite":
        with sqlite3.connect(f"file:{root / 'state.sqlite'}?mode=ro", uri=True) as db:
            require(set(db.execute("SELECT a_id,b_id FROM crossmap")) == actual,
                    "SQL membership disagrees with API")
            sql_reviews = set(db.execute("SELECT side, id, candidate_id FROM reviews"))
            require(sql_reviews == review_keys, f"SQL reviews disagree: {sql_reviews}")
            for side in ("a", "b"):
                ids = {r[0] for r in db.execute(f"SELECT id FROM {side}_records")}
                require(ids == records[side].keys(), f"{side}: SQL records disagree")
                sql_unmatched = {r[0] for r in db.execute(f"SELECT id FROM {side}_unmatched")}
                matched = {p[0 if side == 'a' else 1] for p in actual}
                require(sql_unmatched == ids - matched, f"{side}: SQL orphan set disagrees")
    return {"pairs": len(actual), "reviews": len(reviews),
            "orphans": {s: len(records[s]) - len(actual) for s in records},
            "sql_checked": backend == "sqlite", "relationships_checked": True}


def contention(binary, root, backend, side, args):
    prepare(root, backend)
    records = {"a": {}, "b": {}}
    with server(binary, root, args) as client:
        expected = set()
        # Prove BOTH competing requests would individually cause the same real
        # two-displacement chain, not merely an attempted match on disjoint data.
        preflight = []
        for index, angle in enumerate((-2, 0)):
            g = gadget(f"proof{index}", side, angle)
            setup(client, g, records)
            expected |= {edge(g, "p0", "r0"), edge(g, "p1", "r1")}
            require(pairs(client) == expected, "strict automatic setup precondition failed")
            result = summary(client.add(side, g["root"]), 2)
            records[side][g["root"]["record_id"]] = g["root"]
            expected -= {edge(g, "p0", "r0"), edge(g, "p1", "r1")}
            expected |= {edge(g, "root", "r0"), edge(g, "p0", "r1")}
            preflight.append(result)
        g = gadget("contended", side, -2)
        g["strong"] = record("contended", "strong", 0)
        setup(client, g, records)
        expected |= {edge(g, "p0", "r0"), edge(g, "p1", "r1")}
        require(pairs(client) == expected, "strict contested setup precondition failed")
        (root / "hold.json").write_text(json.dumps({"text": "-2"}))
        pool = ThreadPoolExecutor(max_workers=5)
        timeline = {}
        lock = threading.Lock()
        barrier = threading.Barrier(5)  # Four competitors plus main thread.

        def request(name, path, payload, wait=True):
            if wait:
                barrier.wait(timeout=args.timeout)
            with lock:
                timeline[name] = {"start": time.monotonic()}
            try:
                return client.call(path, payload)
            finally:
                with lock:
                    timeline[name]["end"] = time.monotonic()

        futures = {}
        try:
            futures["first_root"] = pool.submit(request, "first_root", f"{side}/add",
                                                {"record": g["root"]}, False)
            deadline = time.monotonic() + args.timeout
            while not (root / "encoder-entered").exists():
                require(not futures["first_root"].done(), "first root escaped encoder latch")
                require(time.monotonic() < deadline, "encoder entry deadline exceeded")
                time.sleep(.005)
            hold_observed = time.monotonic()
            jobs = [("second_root", f"{side}/add", {"record": g["strong"]}),
                    ("bulk_upsert", f"{side}/add-batch",
                     {"records": [g["p0"], records[side]["proof0-p0"]]}),
                    ("delete", f"{side}/remove", {"id": g["p1"]["record_id"]}),
                    ("flush", "/admin/flush", {})]
            for name, path, payload in jobs:
                futures[name] = pool.submit(request, name, path, payload)
            barrier.wait(timeout=args.timeout)
            deadline = time.monotonic() + args.timeout
            while len(timeline) < 5:
                require(time.monotonic() < deadline, "client launch deadline exceeded")
                time.sleep(.005)
            # Give waiting HTTP calls time to enter the server. This is not an
            # assertion that their internal gate acquisition has been observed.
            time.sleep(.1)
            require(not futures["first_root"].done(), "gate holder completed before release")
            for name in ("second_root", "bulk_upsert", "delete"):
                require(not futures[name].done(), f"{name} escaped held mutation gate")
            release = time.monotonic()
            (root / "encoder-release").touch()
            responses = {name: future.result(timeout=max(.001, deadline - time.monotonic()))
                         for name, future in futures.items()}
        finally:
            (root / "encoder-release").touch()
            pool.shutdown(wait=False, cancel_futures=True)
        first = summary(responses["first_root"], 2)
        second = summary(responses["second_root"], 1)
        for change, old_pair in zip(first["changes"],
                                    (edge(g, "p0", "r0"), edge(g, "p1", "r1"))):
            require(change["expected"] == [{"a_id": old_pair[0], "b_id": old_pair[1]}],
                    f"first root did not displace the intended existing records: {change}")
        old_pair = edge(g, "root", "r0")
        require(second["changes"][0]["expected"] == [{"a_id": old_pair[0], "b_id": old_pair[1]}],
                "second root did not contend for the same receiver as the first root")
        require(len(responses["bulk_upsert"]["results"]) == 2, "missing bulk results")
        for item in responses["bulk_upsert"]["results"]:
            require(item.get("status") == "updated" and item.get("classification") == "auto",
                    f"incoherent bulk item: {item}")
            assignment = summary(item)
            require(assignment["partner"] == item["id"].replace("-p0", "-r1"),
                    f"bulk partner differs from final membership: {item}")
        require(responses["delete"].get("status") == "removed", "delete did not complete")
        for name in ("first_root", "second_root"):
            require(responses[name].get("classification") == "auto"
                    and responses[name]["assignment"]["partner"] == g["r0"]["record_id"],
                    f"root response inconsistent at its commit: {responses[name]}")
        records[side][g["root"]["record_id"]] = g["root"]
        records[side][g["strong"]["record_id"]] = g["strong"]
        del records[side][g["p1"]["record_id"]]
        expected -= {edge(g, "p0", "r0"), edge(g, "p1", "r1")}
        expected |= {edge(g, "strong", "r0"), edge(g, "p0", "r1")}
        require(all(v["start"] < release for v in timeline.values()), "clients did not overlap gate holder")
        check = verify(client, root, backend, records, expected)
        return {"mode": "contention", "backend": backend, "side": side,
                "passed": True, "preflight_displacements": [r["displacements"] for r in preflight],
                "gate_evidence": "encoder latch under upsert assignment_guard; waiting clients not internally instrumented",
                "hold_observed": hold_observed, "release": release, "timeline": timeline,
                "responses": responses, "final": check}


def performance(binary, root, backend, clients, args):
    prepare(root, backend)
    records = {"a": {}, "b": {}}
    gadgets = [gadget(f"group{i:06d}", "a" if i % 2 == 0 else "b")
               for i in range(args.groups)]
    inputs = [{"side": g["side"], "record": g["root"]} for g in gadgets]
    fingerprint = hashlib.sha256(json.dumps({"setup": gadgets, "roots": inputs},
                                           sort_keys=True).encode()).hexdigest()
    with server(binary, root, args) as client:
        initial = set()
        for g in gadgets:
            setup(client, g, records)
            initial |= {edge(g, "p0", "r0"), edge(g, "p1", "r1")}
        require(pairs(client) == initial, "strict performance setup precondition failed")
        barrier = threading.Barrier(clients + 1)
        stop = threading.Event()

        def worker(index):
            barrier.wait(timeout=args.timeout)
            results = []
            for g in gadgets[index::clients]:
                require(not stop.is_set(), "workload cancelled after deadline/failure")
                start = time.perf_counter()
                response = client.add(g["side"], g["root"])
                latency = (time.perf_counter() - start) * 1000
                assignment = summary(response, 2)  # EVERY root must displace >1.
                results.append({"id": g["root"]["record_id"], "latency_ms": latency,
                                "displacements": assignment["displacements"],
                                # Each root is one seed with a linear orphan chain;
                                # depth is displacement count, not len(changes).
                                "chain_depth": assignment["displacements"],
                                "changes": len(assignment["changes"]),
                                "orphans": len(assignment["unmatched"]),
                                "cap_hit": assignment["limit_reached"]})
            return results

        pool = ThreadPoolExecutor(max_workers=clients)
        try:
            futures = [pool.submit(worker, i) for i in range(clients)]
            start = time.perf_counter()
            barrier.wait(timeout=args.timeout)
            deadline = time.monotonic() + args.run_timeout
            samples = [row for f in futures
                       for row in f.result(timeout=max(.001, deadline - time.monotonic()))]
            elapsed = time.perf_counter() - start
        finally:
            stop.set()
            pool.shutdown(wait=False, cancel_futures=True)
        expected = set()
        for g in gadgets:
            records[g["side"]][g["root"]["record_id"]] = g["root"]
            expected |= {edge(g, "root", "r0"), edge(g, "p0", "r1")}
        check = verify(client, root, backend, records, expected)
        result = metrics([r["latency_ms"] for r in samples], elapsed)
        depths = [r["chain_depth"] for r in samples]
        result.update(mode="performance", backend=backend, clients=clients, passed=True,
                      input_sha256=fingerprint, samples=sorted(samples, key=lambda r: r["id"]),
                      displacements=sum(depths), displacements_per_second=sum(depths) / elapsed,
                      chain_depth={"min": min(depths), "max": max(depths),
                                   "mean": sum(depths) / len(depths)},
                      cap_hits=sum(r["cap_hit"] for r in samples),
                      response_orphan_events=sum(r["orphans"] for r in samples), final=check)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--mode", choices=("contention", "performance", "all"), default="contention")
    parser.add_argument("--backend", choices=("memory", "sqlite", "both"), default="both")
    parser.add_argument("--groups", type=int, default=100, help="independent chains, alternating A/B roots")
    parser.add_argument("--repeats", type=int, default=3, help="fresh-server paired timing repetitions")
    parser.add_argument("--timeout", type=float, default=10, help="per-HTTP and contention deadline seconds")
    parser.add_argument("--startup-timeout", type=float, default=60)
    parser.add_argument("--run-timeout", type=float, default=300, help="whole measured workload deadline seconds")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--keep-workdir", action="store_true")
    parser.add_argument("--encoder-root", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.encoder_root:
        encoder(args.encoder_root)
        return 0
    if not args.binary or not args.binary.is_file():
        parser.error("--binary must name an existing meld binary")
    if args.groups < 10 or args.repeats < 1 or min(args.timeout, args.startup_timeout, args.run_timeout) <= 0:
        parser.error("require groups >= 10, repeats >= 1, and positive deadlines")
    # Latch has its own emergency timeout; keep HTTP waits below that bound.
    if args.mode != "performance" and args.timeout > 15:
        parser.error("contention --timeout must be <= 15 seconds (encoder latch safety bound)")
    binary = args.binary.resolve()
    temp = None if args.keep_workdir else tempfile.TemporaryDirectory(prefix="assignment-chains-")
    root = Path(tempfile.mkdtemp(prefix="assignment-chains-")) if temp is None else Path(temp.name)
    report = {"binary": str(binary), "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
              "platform": platform.platform(), "python": sys.version, "workdir": str(root),
              "workdir_retained": args.keep_workdir, "arguments": vars(args).copy(), "runs": [],
              "scope": "targeted synthetic real-scoring chains; no full G3 or production performance claim"}
    report["arguments"] = {k: str(v) if isinstance(v, Path) else v for k, v in report["arguments"].items()}
    backends = ("memory", "sqlite") if args.backend == "both" else (args.backend,)
    try:
        for backend in backends:
            if args.mode in ("contention", "all"):
                for side in ("a", "b"):
                    name = f"contention-{backend}-{side}"
                    try:
                        report["runs"].append(contention(binary, root / name, backend, side, args))
                    except Exception as exc:
                        report["runs"].append({"name": name, "passed": False, "error": repr(exc)})
            if args.mode in ("performance", "all"):
                for repeat in range(args.repeats):
                    paired = []
                    for clients in (1, 10):
                        name = f"performance-{backend}-{repeat}-{clients}"
                        try:
                            result = performance(binary, root / name, backend, clients, args)
                            result["repeat"] = repeat
                            paired.append(result)
                            report["runs"].append(result)
                        except Exception as exc:
                            report["runs"].append({"name": name, "passed": False, "error": repr(exc)})
                    if len(paired) == 2:
                        require(paired[0]["input_sha256"] == paired[1]["input_sha256"], "unpaired inputs")
        report["passed"] = all(r["passed"] for r in report["runs"])
        text = json.dumps(report, indent=2) + "\n"
        if args.output:
            args.output.write_text(text)
        print(text, end="")
        return 0 if report["passed"] else 1
    finally:
        if temp is not None:
            temp.cleanup()


if __name__ == "__main__":
    sys.exit(main())
