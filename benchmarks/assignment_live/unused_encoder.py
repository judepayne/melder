#!/usr/bin/env python3
"""Handshake-only guard for live startup; this workload has no embeddings.

Current live startup initializes an encoder even for BM25/numeric configs.
Never fabricate vectors: any unexpected encode request fails the benchmark.
"""
import json
import sys


def main():
    print(json.dumps({"type": "handshake", "protocol_version": 1,
                      "vector_dim": 1, "model_id": "unused-no-embeddings",
                      "max_batch_size": 1}), flush=True)
    for _line in sys.stdin:
        print(json.dumps({"type": "encode_response",
                          "error": "assignment_live forbids embedding calls"}), flush=True)
        print("UNEXPECTED_ENCODE", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
