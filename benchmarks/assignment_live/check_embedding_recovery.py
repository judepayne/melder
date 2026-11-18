#!/usr/bin/env python3
"""Regression: recover changed embeddings from WAL over a warm OLD vector cache.

Uses the real in-memory live matching/assignment pipeline and a deterministic
2D remote encoder (this script's private --encoder mode). Never builds melder.
"""
import argparse
import json
from pathlib import Path
import shlex
import struct
import sys
import tempfile
import time

from check_recovery import server
from run import config, pairs


OLD = {"id": "existing-a", "name": "old"}
NEW = {"id": "existing-a", "name": "new"}


def encoder():
    """Serve protocol v1 with orthogonal OLD/NEW vectors, without model IO."""
    output = sys.stdout.buffer
    output.write((json.dumps({"type": "handshake", "protocol_version": 1,
                              "vector_dim": 2, "model_id": "recovery-orthogonal-v1",
                              "max_batch_size": 256}) + "\n").encode())
    output.flush()
    for line in sys.stdin:
        request = json.loads(line)
        vectors = [{"old": (1., 0.), "new": (0., 1.)}[text.strip().lower()]
                   for text in request["texts"]]
        payload = b"".join(struct.pack("<2f", *vector) for vector in vectors)
        output.write((json.dumps({"type": "encode_response",
                                  "results": [{"ok": True} for _ in vectors],
                                  "vector_count": len(vectors), "vector_dim": 2}) + "\n").encode())
        output.write(struct.pack("<I", len(payload)))
        output.write(payload)
        output.flush()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def shutdown(process, client):
    client.call("/admin/shutdown", {})
    require(process.wait(timeout=15) == 0, "graceful cache-warming shutdown failed")


def cache_snapshot(root):
    return {str(path.relative_to(root)): path.read_bytes()
            for path in (root / "cache_a").rglob("*") if path.is_file()}


def check(binary, backend):
    with tempfile.TemporaryDirectory(prefix="assignment-embedding-recovery-") as directory:
        root = Path(directory)
        # An existing dataset ID is essential: WAL-only IDs take a different
        # startup/cache path and can falsely make this regression pass.
        (root / "a.csv").write_text("id,name\nexisting-a,old\n", encoding="ascii")
        (root / "b.csv").write_text("id,name\n", encoding="ascii")
        settings = config(root, "deferred_acceptance", 32)
        settings["embeddings"]["remote_encoder_cmd"] = shlex.join(
            [sys.executable, str(Path(__file__).resolve()), "--encoder"])
        settings["vector_backend"] = backend
        settings["match_fields"] = [{"field_a": "name", "field_b": "name",
                                     "method": "embedding", "weight": 1.0}]
        settings["blocking"] = {"enabled": False}
        settings["thresholds"] = {"auto_match": 0.9, "review_floor": 0.8}
        (root / "config.yaml").write_text(json.dumps(settings), encoding="ascii")

        with server(binary, root, 1) as (process, client):
            # Confirm the initial OLD vector really participates in scoring.
            old_probe = client.call("b/match", {"record": {"id": "old-probe", "name": "old"}})
            require(any(row["id"] == OLD["id"] and row["score"] > .99
                        for row in old_probe["matches"]), f"OLD fixture did not score: {old_probe}")
            require(not pairs(client), "warmup unexpectedly created a protected pair")
            shutdown(process, client)
        old_cache = cache_snapshot(root)
        require(bool(old_cache), "graceful warmup produced no on-disk A vector cache")
        print("Warmup: base existing-a=old scores 1.0; graceful shutdown saved OLD cache.", flush=True)

        # Avoid unrelated legacy second-resolution WAL filename collisions.
        time.sleep(1.1)
        with server(binary, root, 2) as (process, client):
            updated = client.add("a", NEW)
            require(updated["status"] == "updated", f"not an existing-ID update: {updated}")
            # Prove this exact NEW-only proposal is accepted before the crash.
            control = client.add("b", {"id": "control-b", "name": "new"})
            require(pairs(client) == {(NEW["id"], "control-b")},
                    f"NEW control proposal did not match before crash: {control}")
            client.call("b/remove", {"id": "control-b"})
            require(not pairs(client), "control removal left a protected incumbent")
            client.call("/admin/flush", {})
            require(cache_snapshot(root) == old_cache,
                    "fixture invalid: update/flush saved vectors before SIGKILL")
            events = [json.loads(line) for path in root.glob("wal*.ndjson")
                      for line in path.read_text().splitlines() if line.strip()]
            require(any(event.get("record") == NEW for event in events),
                    f"fixture invalid: flushed WAL lacks NEW record: {events}")
            process.kill()
            process.wait(timeout=8)
        require(cache_snapshot(root) == old_cache, "SIGKILL unexpectedly replaced OLD cache")
        print("Crash: NEW control accepted; removed control pair; flushed NEW WAL; SIGKILL left OLD cache unchanged.", flush=True)

        time.sleep(1.1)
        with server(binary, root, 3) as (process, client):
            recovered = client.call("a/query?id=existing-a")
            require(recovered.get("record") == NEW,
                    f"fixture/recovery failure: NEW record itself was not restored: {recovered}")
            require(not pairs(client), "fixture invalid: restored protected pair hides proposal")
            health = client.call("health")
            require((health["records_a"], health["records_b"]) == (1, 0),
                    f"fixture invalid: control record removal was not recovered: {health}")
            # Read-only OLD probe makes a failure diagnostic distinguish stale
            # vectors from generic candidate/proposal fixture problems.
            old_after_restart = client.call(
                "b/match", {"record": {"id": "old-probe", "name": "old"}})
            proposal = client.add("b", {"id": "proposal-b", "name": "new"})
            actual = pairs(client)
            require(actual == {(NEW["id"], "proposal-b")},
                    "CHANGED-EMBEDDING RECOVERY FAILED: record restored as NEW, "
                    "no protected pairs, but NEW-only proposal was not accepted; "
                    f"pairs={sorted(actual)}, response={json.dumps(proposal, sort_keys=True)}, "
                    f"OLD probe={json.dumps(old_after_restart, sort_keys=True)}")
            require(proposal.get("assignment", {}).get("displacements") == 0,
                    f"unexpected displacement in free-endpoint proposal: {proposal}")
            shutdown(process, client)
        print(f"PASS ({backend}): WAL restored changed embedding over OLD cache; real reciprocal proposal accepted existing-a <-> proposal-b.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--backend", choices=("flat", "usearch"), default="flat")
    parser.add_argument("--encoder", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.encoder:
        encoder()
    elif args.binary is None:
        parser.error("--binary is required")
    else:
        try:
            check(args.binary.resolve(strict=True), args.backend)
        except (OSError, ValueError, RuntimeError, KeyError) as error:
            parser.exit(1, f"FAIL: {error}\n")


if __name__ == "__main__":
    main()
