#!/usr/bin/env python3
"""Check live WAL-only state after crash, quiet shutdown, and another restart."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import socket
import subprocess
import tempfile
import time

from run import Client, config, pair, pairs, workload


@contextmanager
def server(binary, root, run_number):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    client = Client(port, 5)
    path = root / f"server-{run_number}.log"
    with path.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            [str(binary), "serve", "--config", str(root / "config.yaml"), "--port", str(port), "--bind", "127.0.0.1"],
            cwd=root, stdout=stream, stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 15
            while True:
                if process.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError(f"server failed to start: {path.read_text()}")
                try:
                    client.call("health")
                    break
                except OSError:
                    time.sleep(.05)
            yield process, client
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=8)


def check(binary):
    with tempfile.TemporaryDirectory(prefix="assignment-recovery-") as directory:
        root = Path(directory)
        for side in ("a", "b"):
            (root / f"{side}.csv").write_text("id,group,name,stable,competitive\n", encoding="ascii")
        (root / "config.yaml").write_text(json.dumps(config(root, "deferred_acceptance", 32)), encoding="ascii")
        gadgets = workload(1)
        expected_pairs = {pair(g, "arrival") for g in gadgets}

        def verify(client):
            if pairs(client) != expected_pairs:
                raise RuntimeError("recovered bijection differs from the original")
            stats = client.call("health")
            if (stats["records_a"], stats["records_b"]) != (3, 3):
                raise RuntimeError(f"WAL-only records were lost: {stats}")

        with server(binary, root, 1) as (process, client):
            for g in gadgets:
                client.add(g["receiver_side"], g["receiver"])
                client.add(g["side"], g["incumbent"])
                outcome = client.add(g["side"], g["arrival"])
                if outcome["assignment"]["displacements"] != 1:
                    raise RuntimeError(f"fixture did not cause displacement: {outcome}")
            verify(client)
            client.call("/admin/flush", {})
            process.kill()
            process.wait(timeout=8)

        # Separate legacy second-granularity filenames, so this also reproduces
        # loss on a binary that reused a filename during a same-second restart.
        time.sleep(1.1)
        with server(binary, root, 2) as (process, client):
            verify(client)
            client.call("/admin/shutdown", {})
            if process.wait(timeout=15) != 0:
                raise RuntimeError("quiet second server run did not shut down cleanly")

        with server(binary, root, 3) as (process, client):
            verify(client)
            client.call("/admin/shutdown", {})
            if process.wait(timeout=15) != 0:
                raise RuntimeError("third server run did not shut down cleanly")

        events = [json.loads(line) for path in root.glob("wal*.ndjson")
                  for line in path.read_text().splitlines() if line.strip()]
        changes = [event for event in events if event.get("type") == "assignment_delta"]
        if len(changes) != 4 or sum(bool(event["expected"]) for event in changes) != 2:
            raise RuntimeError("audited assignment steps were lost or duplicated")
        print("Crash, quiet shutdown, and second restart preserve six records, two final pairs, and four audit steps.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    args = parser.parse_args()
    check(args.binary.resolve(strict=True))


if __name__ == "__main__":
    main()
