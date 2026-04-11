← [Back to Index](./) | [Configuration](configuration.md) | [Performance](performance.md)

# Remote Encoder

Melder can delegate embedding inference to a user-supplied subprocess so
that you can plug in your organisation's central embedding service
instead of running a model locally. Records, the vector index, BM25,
synonym matching, crossmap, and the match log all stay in-process as
usual — only the `text → vector` step is externalised.

This is the right fit when your organisation wraps AI components behind
internal APIs (banks, pharma, regulated tech), and you need melder to
call those services instead of loading an ONNX model. Everything else
about melder is unchanged.

## Is this for you?

**Yes, if:**

- Your organisation mandates that embedding models run behind a central
  internal service (security, governance, cost control).
- You want to reuse a model that only exists inside your org's
  infrastructure.
- You're willing to accept lower throughput than local inference in
  exchange for compliance with organisational constraints.

**No, if:**

- You want to speed up melder — local ONNX is faster (~1,300 rec/s
  local vs. ~50–200 rec/s typical remote).
- You need high-throughput live mode — remote encoders work in live
  mode but are bounded by round-trip latency.
- You need centralised BM25, lexical search, or federated records —
  those are explicit non-goals. Embeddings only.

## Configuration

Set `embeddings.remote_encoder_cmd` to a shell command that launches your
script. Do **not** also set `embeddings.model` — exactly one of the two
must be set.

```yaml
embeddings:
  remote_encoder_cmd: "python scripts/my_encoder.py --env prod"
  a_cache_dir: cache/a
  b_cache_dir: cache/b

performance:
  # REQUIRED when remote_encoder_cmd is set. No default.
  # Each slot is an independent subprocess; N slots = N concurrent calls.
  encoder_pool_size: 8
  # Optional. Defaults to 60000ms. Your script's own remote-service
  # timeout MUST be strictly less than this — see "The timeout footgun".
  encoder_call_timeout_ms: 60000
  # Optional. Number of texts per encode request. Defaults vary; 128–256
  # is typical. Passed to your script verbatim; your script decides
  # whether to split further against the remote service's own limit.
  encoder_batch_size: 256
```

Validation rules:

- **XOR**: exactly one of `embeddings.model` / `embeddings.remote_encoder_cmd`.
  Both set → error. Neither set → error.
- **`encoder_pool_size` is required** when `remote_encoder_cmd` is set.
  No default (startup cost scales with pool size, so we make you think
  about it).
- **`encoder_device` is silently ignored** when remote is set (an
  info-level log line is emitted). GPU encoding doesn't apply — your
  script owns the compute.

## How it works

When melder starts, it spawns `encoder_pool_size` copies of your command
as long-lived subprocesses — one per pool slot. Each slot handles one
encode call at a time: melder writes a request envelope to stdin, your
script calls the central service, and writes a response envelope (plus
binary vector payload) to stdout. This continues for the lifetime of the
melder run.

Startup cost (interpreter initialisation, model warming, auth handshake)
is paid **once per slot at startup** — not per call. All slots start in
parallel, so effective startup delay is the slowest slot's delay, not
the sum. With 8 slots and 3-second per-slot startup, total startup is
~3 seconds.

Throughput is bounded by `slot_count × (1 / remote_call_latency)`. For
typical corporate embedding services this is ~50–200 requests/second;
compare to ~1,300 req/s for local ONNX.

## The subprocess contract (wire protocol)

This is the contract your script must conform to. The framing and
message shapes are fixed — do not deviate.

### Framing

Messages flow over stdin (melder → script) and stdout (script → melder).
Each message is:

1. A **JSON envelope**: a single line of UTF-8 JSON terminated by `\n`.
2. An optional **binary trailer**: a 4-byte little-endian unsigned
   length prefix followed by exactly that many bytes of raw payload.
   Used only for encode responses containing vectors.

The envelope's `vector_count` and `vector_dim` fields tell the receiver
whether a trailer follows. Envelopes without trailers omit both the
length prefix and the payload.

**stderr** is captured by melder and forwarded to the log with a `slot=N`
field. Your script may emit diagnostic messages to stderr freely — it
does not interfere with the IPC channel.

### Startup handshake

Within **30 seconds** of spawn, your script must emit exactly one
handshake envelope on stdout:

```json
{"type": "handshake", "protocol_version": 1, "vector_dim": 768, "model_id": "acme-embedding-v2", "max_batch_size": 256}
```

Fields:

- `protocol_version`: must be `1`. Phase 1 only supports version 1.
- `vector_dim`: dimensionality of the vectors your script returns.
  Must be the same for every response.
- `model_id`: non-empty string identifying the remote model. Logged and
  used for cost attribution on metered services.
- `max_batch_size`: informational. Melder logs it but does not enforce it.
  If your script receives a larger batch than it can handle, it must
  either split internally or return a batch-level error.

If the handshake doesn't arrive within 30 seconds, melder kills the
subprocess and enters the respawn cycle. If none of the slots handshake
successfully, melder exits with `RemoteSpawnFailed` — fail-loud at
startup is intentional.

### Encode request (melder → script)

```json
{"type": "encode", "texts": ["Goldman Sachs Group Inc", "JP Morgan Chase & Co"]}
```

No trailer. `texts` is a JSON array of UTF-8 strings.

### Encode response (script → melder)

Three shapes.

**All records succeeded:**

```json
{"type": "encode_response", "results": [{"ok": true}, {"ok": true}], "vector_count": 2, "vector_dim": 768}
```

Followed by a binary trailer: 4-byte LE u32 length prefix (`2 * 768 * 4 = 6144`),
then 6144 bytes of raw little-endian `f32` data. The two vectors are
packed contiguously, no internal framing.

**Some records failed per-record:**

```json
{"type": "encode_response", "results": [{"ok": true}, {"error": "content policy rejection"}, {"ok": true}], "vector_count": 2, "vector_dim": 768}
```

Followed by a binary trailer containing **only vectors for the successful
results, in the original request order** (here: 2 vectors, 6144 bytes).
Melder walks the `results` array and consumes one vector from the
trailer for each `{"ok": true}` entry, skipping `{"error": ...}` entries.

**Whole-batch failure:**

```json
{"type": "encode_response", "error": "rate limit exceeded after internal retries"}
```

No trailer. No `results` field. Melder treats this as final — it will
not retry the batch. Use this for errors that affect the entire call
(auth failure, rate limit exhausted, remote service down).

### Shutdown

Melder closes your script's stdin. Your script should see EOF in its
read loop, flush any in-flight work, and exit 0. Melder waits 5 seconds
for the process to exit cleanly, then sends SIGTERM (Unix), waits
another 5 seconds, then SIGKILL.

There is no explicit shutdown message in the protocol. Close-stdin is
idiomatic on Unix and simpler than a protocol message that would itself
need error handling.

## Example: writing a real script

The reference implementation is at [`tests/fixtures/stub_encoder.py`](../tests/fixtures/stub_encoder.py).
Copy it as a starting point; it's a labelled skeleton with every
functional section clearly marked and stubbed, ready for you to replace
with your org-specific code.

### What the protocol fixes (do not deviate)

- Read NDJSON envelopes from stdin, one per line, UTF-8.
- Emit the handshake on stdout within 30 seconds.
- For each encode request, write a response envelope + binary trailer
  to stdout.
- Return vectors in the **same order** as the input texts, packed
  little-endian, with only successful records in the trailer.
- Return per-record errors as `{"error": "..."}` entries inside
  `results`. Return whole-batch errors as top-level `{"error": "..."}`
  with no `results` field.
- Exit cleanly on stdin EOF.

### What you write (the org-specific part)

Everything on the "remote service" side of the pipe. Melder owns
nothing here:

**Authentication and transport**

- Load credentials (env var, credential file, keychain, mTLS client cert).
- Construct your HTTP/gRPC client with the right CA bundle, proxy
  settings, and TLS options.
- Refresh short-lived tokens (OAuth, STS, etc.) mid-run. Your script is
  long-lived; tokens aren't.

**Retries against the remote service**

- Rate-limit backoff (429 `Retry-After`).
- 5xx transient errors with exponential backoff and jitter.
- Auth refresh on 401 and retry.
- Cold-start tolerance (first call after the remote service has idled).
- Circuit-break if the remote is hard-down, so you fail fast to melder
  rather than eating the per-call timeout.

**Request and response shaping**

- Map melder's neutral `{texts: [...]}` envelope to whatever schema the
  remote API expects (OpenAI `input`, Vertex `instances`, Bedrock
  `inputText`, etc.).
- Enforce the remote service's own batch-size limit — if melder sends a
  batch larger than your remote accepts, split internally.
- Normalise vectors if the remote returns unnormalised output (melder's
  scoring expects unit vectors).
- Classify errors:
  - **Content-policy rejection** → per-record `{"error": "..."}`.
  - **Rate limit after exhausted retries** → whole-batch error.
  - **Auth failure after refresh** → whole-batch error.
  - **Transport error** → whole-batch error.

**Observability (your side)**

- Log to stderr freely — melder forwards it to the main log with a
  `slot` field.
- Track per-call latency and retry counts in whatever metrics system
  your org uses.

### ★ The timeout footgun ★

**Your script's own remote-service timeout MUST be strictly less than
`performance.encoder_call_timeout_ms`.**

If melder's per-call timeout fires while your remote call is still
outstanding, melder sends SIGKILL — you do not get to write a clean
error response, and any in-flight remote calls leak. Be conservative:
if melder's timeout is 60 seconds, set your remote call timeout to
around 50 seconds. That gives you room to catch the error and emit a
clean whole-batch error envelope before melder gives up.

## Lifecycle guarantees

Melder promises:

- Each slot is spawned exactly once at startup. You do not need to
  handle "what if two copies of me are running concurrently" — each
  slot is its own subprocess.
- Handshakes must complete within 30 seconds. If yours doesn't, the
  slot is killed and enters the respawn cycle.
- On subprocess crash, hang (per-call timeout exceeded), broken pipe,
  or protocol violation, melder kills the slot and respawns it with
  exponential backoff (1 second → 2s → 4s → ... → capped at 60s).
- After 5 consecutive failures within a 60-second rolling window, a
  slot is marked **unhealthy** and no more calls are dispatched to it.
- When all slots become unhealthy, encode calls fail with
  `EncoderError::PoolExhausted` and the current batch run aborts.
- On clean shutdown, stdin is closed, then 5s grace, then SIGTERM, then
  5s grace, then SIGKILL.
- Melder never retries failed requests against the remote service —
  your script owns that retry loop. Melder only retries subprocess
  lifecycle failures (crash/hang/violation).

You do **not** need to handle:

- The wire format beyond what's documented above — melder owns framing
  and binary marshalling.
- Per-call timeouts — melder enforces them. Your job is to have an
  internal timeout that's strictly less than melder's.
- Pool sizing or concurrency — melder runs N subprocesses for N-way
  concurrency; each of your subprocesses is sequential.
- IPC errors — melder handles them and respawns you.

## Failure semantics

Melder's Phase 1 policy is **fail-fast-loud**:

- Any per-record error, or any whole-batch error, fails the current
  melder run. Before exiting, melder emits a structured tracing event
  for every error seen with full context: record text (or prefix if
  longer than 200 chars), error message, slot index, latency, model_id,
  and the configured command.
- No silent skipping, no partial-success writing, no threshold-based
  leniency. Silent data loss is worse than loud failure.
- Re-running a failed batch is cheap. Your script's retry logic and
  melder's cold-run caching work together: on the second attempt, your
  script can fast-path records it already successfully encoded if you
  build that in.

A more permissive "leniency mode" (errors streamed to `encode_errors.csv`,
batch continues until a threshold) is additive on top of this behaviour
and may be added if customers demonstrate real need. Phase 1 does not
ship it.

## Observability

All observability hooks into melder's existing `tracing` infrastructure —
no new metrics backends, no new log files.

Tracing spans emitted by the remote encoder:

- `remote_encoder_call` — wraps each encode call. Fields: `slot`,
  `batch_size`, `latency_ms`, `outcome` (success | per_record_errors |
  batch_error | timeout | subprocess_died).
- `remote_encoder_handshake` — wraps slot startup. Fields: `slot`,
  `latency_ms`, `vector_dim`, `model_id`.
- `remote_encoder_respawn` — wraps respawn attempts. Fields: `slot`,
  `attempt`, `backoff_ms`, `last_error`.

Structured events fire on every encoder error (per-record or
whole-batch), carrying the full context from the failure-semantics
section above.

**Cost attribution.** On metered remote services, the key counter is
**total texts encoded** — this is what your vendor will charge you for,
and it's emitted via tracing on every encode span as `batch_size`. Sum
the `batch_size` fields across the `remote_encoder_call` spans in your
log aggregator for per-run cost reconciliation.

## Testing your script

The stub at [`tests/fixtures/stub_encoder.py`](../tests/fixtures/stub_encoder.py)
doubles as a working reference implementation and a suite of
failure-injection modes. Point melder at it via `remote_encoder_cmd`
and you can verify your pipeline works end-to-end before swapping in
a real script.

```yaml
embeddings:
  remote_encoder_cmd: "python3 tests/fixtures/stub_encoder.py --vector-dim 384 --model-id smoke-test"
```

The stub has CLI flags for every failure mode the integration tests
cover (`--fail-handshake`, `--fail-on-call N`, `--hang-on-call N`,
`--batch-error-every N`, `--record-error-rate P`, `--garbage-response`,
etc.). See the top of the file for the complete list.

A worked example configuration is at
[`benchmarks/batch/10kx10k_remote_encoder/cold/`](../benchmarks/batch/10kx10k_remote_encoder/cold/) —
a cold batch run against the stub, exercising every new config knob.
Run it with `python3 benchmarks/batch/10kx10k_remote_encoder/cold/run_test.py`.

## Troubleshooting

**Handshake timeout / slot 0 failed to spawn.** Your script didn't emit
a valid handshake envelope within 30 seconds of spawn. Common causes:
interpreter startup is slow (warm a venv), auth is blocking (move it
behind a timeout), you forgot to `flush()` after writing the handshake.

**`vector_dim mismatch` error.** Slot 0's handshake declared one dim,
slot N declared a different one. All slots must return the same dim.
Check whether your script's model selection is deterministic.

**`RemoteSpawnFailed` at startup.** At least one slot failed to
handshake after the initial respawn cycle. Check stderr captured in the
melder log (with `slot=N` field) for the script's actual error output.

**`Timeout` errors during encode.** Your script is blocked on the remote
service. Most likely your script's internal timeout is longer than
`encoder_call_timeout_ms`, so melder is killing you. Shorten your
internal timeout.

**`ProtocolViolation` errors.** Your script wrote something melder
couldn't parse — malformed JSON, truncated trailer, wrong byte order.
Validate your envelope against a JSON linter and confirm you're using
`struct.pack("<f", ...)` (little-endian) for vectors.

**Pool exhausted.** All slots have crossed the unhealthy threshold
(5 failures within a 60s window). Your script is failing too often for
melder to make progress. Check the log for the per-slot respawn events
and look at the underlying error.

## Known limitations

Phase 1 does not include:

- **Vector cache.** A persistent `(text_hash, model_id) → vector` cache
  for skipping repeat encode calls across runs. Planned as additive on
  top of Phase 1 — no protocol changes.
- **Leniency mode.** Per-record errors collapse into whole-batch
  failures. A threshold-based mode with an `encode_errors.csv` stream
  is planned but not shipped.
- **Tunable lifecycle timings.** Respawn backoff, slot-unhealthy
  thresholds, handshake timeout, and shutdown grace periods are all
  hardcoded.
- **Orphan cleanup on melder crash.** If melder panics or is SIGKILLed,
  your subprocesses may become orphans. On Linux this is fixable with
  `PR_SET_PDEATHSIG`; macOS has no direct equivalent. Both are deferred.
- **Rust-native HTTP transport.** The trait is ready for one but we
  don't ship one — a subprocess is the only Phase 1 transport.
- **Remote indexing / searching.** Phase 1 is encoder-only; the vector
  index stays local. Phase 2 (not implemented) would add optional
  remote vector-DB backends.
