#!/usr/bin/env python3
# ruff: noqa: E501
"""
Reference remote encoder stub for melder's subprocess-backed RemoteEncoder.

============================================================================
  READ ME FIRST — what this file is for
============================================================================

This file serves three audiences:

  1. **Script authors** building a real remote encoder for their org.
     Use this as a labelled skeleton. The sections below show where each
     concern (auth, transport, retries, etc.) plugs in. The STUB
     implementation below each section header is a no-op — replace it
     with your real code.

  2. **Integration tests** (tests/remote_encoder.rs) that need every
     failure-injection mode the spec describes: handshake timeouts,
     mid-call crashes, hangs, whole-batch errors, per-record errors,
     slot respawn, slot-unhealthy promotion, pool exhaustion, stderr
     drain.

  3. **The batch benchmark** at benchmarks/batch/10kx10k_remote_encoder/,
     where the stub provides deterministic vectors + optional sleep so
     we can measure the end-to-end overhead of the subprocess encoder
     path without paying for a real remote service.

============================================================================
  Division of concerns: what the protocol fixes vs. what you write
============================================================================

**Fixed by the melder protocol (do not deviate):**

  * Read NDJSON envelopes from stdin, one per line, UTF-8.
  * For envelopes with a trailer (encode responses with ok=true entries),
    write a 4-byte little-endian u32 length prefix followed by exactly
    that many bytes of raw little-endian f32 vectors to stdout.
  * Emit a `{"type":"handshake",...}` envelope within 30 seconds of
    process start, before anything else.
  * Use `protocol_version: 1` in Phase 1.
  * Declare `vector_dim` in the handshake; every encode response must
    use the same dim. Mismatch → melder tears down the slot.
  * Per-record errors → `{"error":"..."}` entries inside `results`.
    Whole-batch errors → top-level `{"error":"..."}` with no `results`
    field. Both are fail-fast under Phase 1.
  * Read EOF on stdin → finish any in-flight work, exit 0.

**What you write (this is the per-organisation part):**

  * Section 1 — Startup & auth: load credentials, construct HTTP client,
    warm the remote connection.
  * Section 4 — Remote service call: everything about calling the
    central embedding service. Auth refresh on 401, rate-limit / 5xx
    backoff, per-request timeout, remote batch-size splitting, response
    shape mapping, L2-normalisation if the remote doesn't, error
    classification (content policy → per-record error, rate limit after
    exhausted retries → whole-batch error).

============================================================================
  ★ The timeout footgun ★
============================================================================

Your subprocess's own remote-service timeout MUST be strictly less than
melder's `performance.encoder_call_timeout_ms`. If your remote call is
still outstanding when melder's timeout fires, melder will SIGKILL this
subprocess mid-flight — you do not get to emit a clean error, and any
in-flight remote calls will leak. Be conservative: if melder's timeout
is 60s, make your remote call timeout at most ~50s so you have room to
write a clean response.

============================================================================
  CLI flags (for integration tests and benchmarks)
============================================================================

  --vector-dim N            Declared vector dimension (default: 16).
  --model-id STRING         handshake model_id (default: "stub").
  --max-batch-size N        handshake max_batch_size (default: 256).
  --sleep-ms N              Sleep N milliseconds per encode call (default: 0).
                            The benchmark uses this to simulate remote latency.

  --fail-handshake          Sleep past 30s so the handshake never arrives.
  --skip-handshake          Don't emit the handshake at all (same effect).
  --bad-protocol-version    Emit handshake with protocol_version=999.
  --zero-dim                Emit handshake with vector_dim=0.

  --fail-on-call N          sys.exit(1) on the Nth encode call (1-indexed).
  --hang-on-call N          Sleep 999s on the Nth encode call (triggers melder's
                            per-call timeout).
  --batch-error-every N     Return a whole-batch error every Nth call.
  --record-error-rate P     Probability (0.0-1.0) of emitting a per-record
                            error for each text in a batch. Deterministic
                            (hash-based), so tests are reproducible.
  --garbage-response        Emit garbage on stdout instead of a valid response
                            (tests the ProtocolViolation path).

  --log-stderr              Emit diagnostic lines to stderr (tests the drain
                            thread forwarding).
"""

import argparse
import hashlib
import json
import math
import struct
import sys
import time


# ============================================================================
# 1. Startup & auth
# ============================================================================
# Real script: load credentials from env/file/keychain/mTLS, construct your
# HTTP client, warm the remote connection (e.g. make a dry-run call to the
# central embedding service). If this takes longer than a few seconds,
# remember the 30s handshake deadline.
#
# Stub: no-op.

def startup_and_auth(args: argparse.Namespace) -> None:
    if args.log_stderr:
        print("[stub] startup complete", file=sys.stderr, flush=True)


# ============================================================================
# 2. Handshake
# ============================================================================
# Emit {"type":"handshake", ...} on stdout within 30s.
#
# Fields:
#   protocol_version: MUST be 1 in Phase 1.
#   vector_dim:       dimensionality of the vectors you'll return. Must be
#                     consistent across the whole run.
#   model_id:         non-empty string; logged + used for billing attribution
#                     on metered services.
#   max_batch_size:   informational only (melder logs it but won't enforce).
#
# Stub: emit from CLI-controlled values, respecting the failure-injection flags.

def emit_handshake(args: argparse.Namespace) -> None:
    if args.fail_handshake:
        # Never emit the handshake; melder's 30s timer will fire.
        time.sleep(60)
        return
    if args.skip_handshake:
        # Exit silently so melder sees stdout EOF before handshake.
        return
    version = 999 if args.bad_protocol_version else 1
    dim = 0 if args.zero_dim else args.vector_dim
    msg = {
        "type": "handshake",
        "protocol_version": version,
        "vector_dim": dim,
        "model_id": args.model_id,
        "max_batch_size": args.max_batch_size,
    }
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


# ============================================================================
# 3. Request loop
# ============================================================================
# Read one envelope from stdin, call the remote service, write the response.
# Sequential — do NOT try to service multiple requests from one process.
# Melder runs N subprocesses for N-way concurrency via its pool.
#
# Clean exit on EOF: read_line() returning empty string → flush → return 0.

def request_loop(args: argparse.Namespace) -> None:
    call_index = 0
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        call_index += 1

        try:
            msg = json.loads(line)
        except json.JSONDecodeError as e:
            # Protocol violation on the input side — a real script would
            # still try to respond with a batch error so melder logs it.
            write_batch_error(f"malformed request JSON: {e}")
            continue

        if msg.get("type") != "encode":
            write_batch_error(f"unexpected message type: {msg.get('type')!r}")
            continue

        handle_encode(msg, call_index, args)


def handle_encode(
    msg: dict, call_index: int, args: argparse.Namespace
) -> None:
    # --- Failure injection: crashes ---
    if args.fail_on_call and call_index == args.fail_on_call:
        if args.log_stderr:
            print(
                f"[stub] crashing on call {call_index} as requested",
                file=sys.stderr,
                flush=True,
            )
        sys.exit(1)

    # --- Failure injection: hang ---
    if args.hang_on_call and call_index == args.hang_on_call:
        if args.log_stderr:
            print(
                f"[stub] hanging on call {call_index} as requested",
                file=sys.stderr,
                flush=True,
            )
        time.sleep(999)
        return

    # --- Failure injection: garbage response ---
    if args.garbage_response:
        sys.stdout.write("NOT JSON NOT VALID NOT ANYTHING\n")
        sys.stdout.flush()
        return

    # --- Failure injection: whole-batch error ---
    if args.batch_error_every and call_index % args.batch_error_every == 0:
        write_batch_error("stub: injected batch-level error")
        return

    # Sleep to simulate remote latency (benchmark use).
    if args.sleep_ms:
        time.sleep(args.sleep_ms / 1000.0)

    texts = msg.get("texts", [])
    handle_normal_encode(texts, args)


def handle_normal_encode(texts: list, args: argparse.Namespace) -> None:
    # ========================================================================
    # 4. Remote service call  ← the load-bearing part that every author rewrites
    # ========================================================================
    # Real script:
    #   - POST to the central embedding service (HTTPS / gRPC).
    #   - Handle 429 with `Retry-After` backoff.
    #   - Handle 5xx with exponential backoff + jitter.
    #   - Refresh auth token on 401 and retry.
    #   - Enforce the remote service's own batch-size limit; split internally
    #     if melder sends a batch larger than the remote accepts.
    #   - Extract vectors in request order.
    #   - L2-normalise if the remote returns un-normalised vectors (melder
    #     expects unit vectors for cosine-similarity scoring to work).
    #   - Classify errors:
    #       * content policy rejection → per-record {"error": "..."}
    #       * rate limit after all retries exhausted → whole-batch error
    #       * auth failure after refresh → whole-batch error
    #       * transport error → whole-batch error
    #
    # Stub: deterministic hash→vector, optional per-record error injection.
    results: list = []
    vectors: list = []
    for i, text in enumerate(texts):
        if _should_error_for(text, i, args.record_error_rate):
            results.append({"error": "stub: injected per-record error"})
        else:
            results.append({"ok": True})
            vectors.append(deterministic_vector(text, args.vector_dim))

    write_encode_response(results, vectors, args.vector_dim)


def _should_error_for(text: str, index: int, rate: float) -> bool:
    if rate <= 0.0:
        return False
    # Deterministic: hash the text + index, bucket into [0,1).
    h = hashlib.sha256(f"{index}:{text}".encode()).digest()
    bucket = int.from_bytes(h[:4], "little") / 2**32
    return bucket < rate


def deterministic_vector(text: str, dim: int) -> list:
    """Deterministic L2-normalised vector derived from SHA-256(text)."""
    out: list = []
    seed = text.encode()
    i = 0
    while len(out) < dim:
        h = hashlib.sha256(seed + b":" + str(i).encode()).digest()
        for j in range(0, len(h), 2):
            if len(out) >= dim:
                break
            word = int.from_bytes(h[j : j + 2], "little")
            out.append(word / 65535.0 - 0.5)
        i += 1
    norm = math.sqrt(sum(x * x for x in out)) or 1.0
    return [x / norm for x in out]


# ============================================================================
# Response writers (protocol serialisation — do not modify)
# ============================================================================

def write_encode_response(results: list, vectors: list, dim: int) -> None:
    env = {
        "type": "encode_response",
        "results": results,
        "vector_count": len(vectors),
        "vector_dim": dim,
    }
    sys.stdout.write(json.dumps(env) + "\n")
    sys.stdout.flush()
    if vectors:
        body = bytearray()
        for v in vectors:
            body.extend(struct.pack(f"<{dim}f", *v))
        sys.stdout.buffer.write(struct.pack("<I", len(body)))
        sys.stdout.buffer.write(bytes(body))
        sys.stdout.buffer.flush()


def write_batch_error(message: str) -> None:
    env = {"type": "encode_response", "error": message}
    sys.stdout.write(json.dumps(env) + "\n")
    sys.stdout.flush()


# ============================================================================
# 5. Shutdown
# ============================================================================
# melder closes our stdin → we see EOF in the `for line in sys.stdin:` loop
# → the loop ends → we return cleanly from main() → exit 0.
#
# A real script would also close any pooled HTTP connections, flush metrics,
# revoke short-lived tokens. Stub does nothing.

def shutdown(args: argparse.Namespace) -> None:
    if args.log_stderr:
        print("[stub] clean shutdown", file=sys.stderr, flush=True)


# ============================================================================
# Entry point
# ============================================================================

def parse_args(argv: list) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--vector-dim", type=int, default=16)
    p.add_argument("--model-id", default="stub")
    p.add_argument("--max-batch-size", type=int, default=256)
    p.add_argument("--sleep-ms", type=int, default=0)
    p.add_argument("--fail-handshake", action="store_true")
    p.add_argument("--skip-handshake", action="store_true")
    p.add_argument("--bad-protocol-version", action="store_true")
    p.add_argument("--zero-dim", action="store_true")
    p.add_argument("--fail-on-call", type=int, default=0)
    p.add_argument("--hang-on-call", type=int, default=0)
    p.add_argument("--batch-error-every", type=int, default=0)
    p.add_argument("--record-error-rate", type=float, default=0.0)
    p.add_argument("--garbage-response", action="store_true")
    p.add_argument("--log-stderr", action="store_true")
    return p.parse_args(argv)


def main(argv: list) -> int:
    args = parse_args(argv)
    startup_and_auth(args)
    emit_handshake(args)
    if args.fail_handshake or args.skip_handshake:
        return 0
    try:
        request_loop(args)
    finally:
        shutdown(args)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
