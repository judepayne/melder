#!/usr/bin/env python3
"""Check cross-segment delete/recreate recovery and offline export on three starts."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import tempfile

from check_recovery import server
from run import config, pairs, workload


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def check(binary):
    for side in ("a", "b"):
        with tempfile.TemporaryDirectory(prefix="assignment-segment-recreate-") as directory:
            root = Path(directory)
            for dataset in ("a", "b"):
                (root / f"{dataset}.csv").write_text(
                    "id,group,name,stable,competitive\n", encoding="ascii")
            (root / "config.yaml").write_text(
                json.dumps(config(root, "deferred_acceptance", 4)), encoding="ascii")
            gadget = next(g for g in workload(1) if g["side"] == side)
            record = gadget["incumbent"]

            def clean_stop(process, client):
                client.call("/admin/shutdown", {})
                require(process.wait(timeout=15) == 0, "unclean shutdown")

            def export_empty(phase):
                output = root / f"export-{phase}"
                subprocess.run(
                    [str(binary), "export", "--config", str(root / "config.yaml"),
                     "--out-dir", str(output)],
                    cwd=root, check=True, capture_output=True, text=True, timeout=20)
                with (output / "results.csv").open(newline="", encoding="utf-8") as stream:
                    rows = list(csv.DictReader(stream))
                require(not rows, f"offline export resurrected pair at phase {phase}: {rows}")

            with server(binary, root, 1) as (process, client):
                client.add(gadget["receiver_side"], gadget["receiver"])
                response = client.add(side, record)
                require(len(response["assignment"]["changes"]) == 1,
                        "fixture did not install an assignment delta")
                require(len(pairs(client)) == 1, "fixture did not match")
                clean_stop(process, client)

            with server(binary, root, 2) as (process, client):
                require(len(pairs(client)) == 1, "first pair did not restore")
                client.call(f"{side}/remove", {"id": record["id"]})
                response = client.add(side, {**record, "group": "different"})
                require(not response["assignment"]["changes"],
                        "replacement unexpectedly produced a new delta")
                require(not pairs(client), "deleted pair survived before shutdown")
                clean_stop(process, client)
            # Crucially, export before the warm loader can reconcile anything.
            export_empty(2)

            with server(binary, root, 3) as (process, client):
                require(not pairs(client), "deleted assignment resurrected after compaction")
                clean_stop(process, client)
            export_empty(3)
    print("Both sides: delete/recreate in a delta-free segment stays unmatched through offline export and third start.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    args = parser.parse_args()
    check(args.binary.resolve(strict=True))


if __name__ == "__main__":
    main()
