//! Wire protocol for the remote encoder: NDJSON envelope + optional
//! binary trailer.
//!
//! This is the contract between melder and the user-supplied subprocess
//! script. See `docs/remote-encoder.md` for the full prose description —
//! this module is strictly the bytes-on-the-wire codec.
//!
//! ## Framing
//!
//! Every message is:
//!
//! 1. A JSON envelope on one UTF-8 line, terminated by `\n`.
//! 2. An optional binary trailer: 4-byte little-endian `u32` length prefix
//!    followed by exactly that many bytes of raw payload (little-endian
//!    `f32` vectors, contiguous, no internal framing).
//!
//! The envelope's `vector_count` / `vector_dim` fields tell the reader
//! whether a trailer follows and how to parse it. Messages without
//! trailers omit both the length prefix and the payload bytes.

use std::io::{self, BufRead, Read, Write};

use serde_json::Value;

/// A parsed response frame from the subprocess.
///
/// Both variants preserve enough context for fail-fast tracing: the per-record
/// `results` array retains the `ok`/`error` shape so each record's outcome is
/// individually inspectable before the batch fails loudly.
#[derive(Debug, Clone)]
pub enum Frame {
    /// Successful encode response (possibly with some per-record errors).
    /// `vectors` contains exactly one entry per `{"ok":true}` in `results`,
    /// in the original request order; `ok_count == vectors.len()`.
    EncodeResponse {
        /// Per-record outcomes in the order of the original request.
        /// `None` → per-record error (with message); `Some(idx)` → index
        /// into `vectors` for the successful result.
        results: Vec<PerRecord>,
        vectors: Vec<Vec<f32>>,
    },
    /// Whole-batch error reported by the subprocess. Final — melder does
    /// not retry. Under Phase 1's fail-fast policy this collapses into a
    /// `BatchError` EncoderError at the call site.
    BatchError(String),
}

#[derive(Debug, Clone)]
pub enum PerRecord {
    /// Index of the vector for this record inside the trailer.
    Ok(usize),
    /// Per-record error message as reported by the subprocess.
    Err(String),
}

/// Codec error — all of these are translated to `EncoderError::ProtocolViolation`
/// or `SubprocessDied` at the caller, depending on whether the IO source
/// is still alive.
#[derive(Debug, thiserror::Error)]
pub enum ProtocolError {
    #[error("i/o error: {0}")]
    Io(#[from] io::Error),

    #[error("stream closed (EOF)")]
    Eof,

    #[error("malformed envelope: {0}")]
    MalformedEnvelope(String),

    #[error("unexpected message type: expected {expected}, got {got}")]
    UnexpectedType { expected: &'static str, got: String },

    #[error("trailer length mismatch: expected {expected} bytes, got {got}")]
    TrailerLengthMismatch { expected: usize, got: usize },

    #[error("vector_count {count} inconsistent with ok entries {ok}")]
    VectorCountMismatch { count: usize, ok: usize },
}

// ---------------------------------------------------------------------------
// Writers
// ---------------------------------------------------------------------------

/// Write an envelope (single JSON line, `\n`-terminated) to `w`.
pub fn write_envelope<W: Write>(w: &mut W, envelope: &Value) -> io::Result<()> {
    let mut line = serde_json::to_vec(envelope).map_err(io::Error::other)?;
    line.push(b'\n');
    w.write_all(&line)?;
    w.flush()
}

/// Write an `encode` request envelope. Produces:
/// `{"type":"encode","texts":[...]}\n`
pub fn write_encode_request<W: Write>(w: &mut W, texts: &[&str]) -> io::Result<()> {
    let envelope = serde_json::json!({
        "type": "encode",
        "texts": texts,
    });
    write_envelope(w, &envelope)
}

// ---------------------------------------------------------------------------
// Readers
// ---------------------------------------------------------------------------

/// Read a single `\n`-terminated JSON envelope from `r`.
///
/// Returns `Eof` when the stream closes cleanly between messages.
pub fn read_envelope<R: BufRead>(r: &mut R) -> Result<Value, ProtocolError> {
    let mut line = String::new();
    let n = r.read_line(&mut line)?;
    if n == 0 {
        return Err(ProtocolError::Eof);
    }
    // Strip trailing newline (read_line keeps it).
    let trimmed = line.trim_end_matches(['\n', '\r']);
    if trimmed.is_empty() {
        return Err(ProtocolError::MalformedEnvelope("empty line".into()));
    }
    serde_json::from_str(trimmed)
        .map_err(|e| ProtocolError::MalformedEnvelope(format!("{e} (line: {trimmed:?})")))
}

/// Read `n` raw `f32` vectors of dimension `dim` from `r`, one at a time.
///
/// Reads exactly `n * dim * 4` bytes, interpreting each contiguous
/// `dim * 4`-byte block as a little-endian `f32` vector. Returns EOF-style
/// errors if the stream ends early.
pub fn read_vectors<R: Read>(
    r: &mut R,
    n: usize,
    dim: usize,
) -> Result<Vec<Vec<f32>>, ProtocolError> {
    let mut out = Vec::with_capacity(n);
    let mut buf = vec![0u8; dim * 4];
    for _ in 0..n {
        r.read_exact(&mut buf)?;
        let vec: Vec<f32> = buf
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        out.push(vec);
    }
    Ok(out)
}

/// Read the 4-byte LE `u32` length prefix of a trailer.
pub fn read_trailer_len<R: Read>(r: &mut R) -> Result<u32, ProtocolError> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

// ---------------------------------------------------------------------------
// High-level parsers
// ---------------------------------------------------------------------------

/// Parse a handshake envelope into its individual fields. Does NOT validate
/// content — that's the slot's responsibility (it needs slot-specific context
/// like "which dim do we expect?").
#[derive(Debug, Clone)]
pub struct HandshakeInfo {
    pub protocol_version: u32,
    pub vector_dim: usize,
    pub model_id: String,
    pub max_batch_size: Option<usize>,
}

pub fn parse_handshake(env: &Value) -> Result<HandshakeInfo, ProtocolError> {
    let ty = env.get("type").and_then(|v| v.as_str()).unwrap_or_default();
    if ty != "handshake" {
        return Err(ProtocolError::UnexpectedType {
            expected: "handshake",
            got: ty.to_string(),
        });
    }
    let protocol_version = env
        .get("protocol_version")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| {
            ProtocolError::MalformedEnvelope("handshake missing protocol_version".into())
        })? as u32;
    let vector_dim = env
        .get("vector_dim")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| ProtocolError::MalformedEnvelope("handshake missing vector_dim".into()))?
        as usize;
    let model_id = env
        .get("model_id")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ProtocolError::MalformedEnvelope("handshake missing model_id".into()))?
        .to_string();
    let max_batch_size = env
        .get("max_batch_size")
        .and_then(|v| v.as_u64())
        .map(|n| n as usize);
    Ok(HandshakeInfo {
        protocol_version,
        vector_dim,
        model_id,
        max_batch_size,
    })
}

/// Parse an encode-response envelope + trailer.
///
/// Steps:
/// 1. Confirm `type == "encode_response"`.
/// 2. If the envelope carries a top-level `error` field, the whole batch
///    failed — return `Frame::BatchError`, no trailer is expected.
/// 3. Otherwise the envelope has a `results` array: walk it, and for each
///    `{"ok":true}` entry expect one vector in the trailer (in order),
///    skipping entries with `{"error":"..."}`.
/// 4. Read the 4-byte length prefix and the exact expected byte count.
///    The envelope's `vector_count` / `vector_dim` MUST match the `ok`
///    count and slot dim respectively.
pub fn read_encode_response<R: BufRead>(
    r: &mut R,
    expected_dim: usize,
) -> Result<Frame, ProtocolError> {
    let env = read_envelope(r)?;
    let ty = env.get("type").and_then(|v| v.as_str()).unwrap_or_default();
    if ty != "encode_response" {
        return Err(ProtocolError::UnexpectedType {
            expected: "encode_response",
            got: ty.to_string(),
        });
    }

    // Batch-level error (no trailer).
    if let Some(err) = env.get("error").and_then(|v| v.as_str()) {
        return Ok(Frame::BatchError(err.to_string()));
    }

    // Parse results array.
    let results_arr = env
        .get("results")
        .and_then(|v| v.as_array())
        .ok_or_else(|| {
            ProtocolError::MalformedEnvelope("encode_response missing results".into())
        })?;
    let mut per_records = Vec::with_capacity(results_arr.len());
    let mut ok_count = 0usize;
    for (i, entry) in results_arr.iter().enumerate() {
        if entry.get("ok").and_then(|v| v.as_bool()) == Some(true) {
            per_records.push(PerRecord::Ok(ok_count));
            ok_count += 1;
        } else if let Some(msg) = entry.get("error").and_then(|v| v.as_str()) {
            per_records.push(PerRecord::Err(msg.to_string()));
        } else {
            return Err(ProtocolError::MalformedEnvelope(format!(
                "results[{i}]: must have either ok=true or error=\"...\""
            )));
        }
    }

    // Verify vector_count / vector_dim match.
    let declared_count = env
        .get("vector_count")
        .and_then(|v| v.as_u64())
        .map(|n| n as usize);
    let declared_dim = env
        .get("vector_dim")
        .and_then(|v| v.as_u64())
        .map(|n| n as usize);
    if let Some(c) = declared_count
        && c != ok_count
    {
        return Err(ProtocolError::VectorCountMismatch {
            count: c,
            ok: ok_count,
        });
    }
    if let Some(d) = declared_dim
        && d != expected_dim
    {
        return Err(ProtocolError::MalformedEnvelope(format!(
            "vector_dim mismatch: handshake declared {expected_dim}, response declared {d}"
        )));
    }

    // Read trailer (skipped when ok_count == 0 and no trailer expected).
    let vectors = if ok_count == 0 {
        Vec::new()
    } else {
        let expected_bytes = ok_count * expected_dim * 4;
        let declared_len = read_trailer_len(r)? as usize;
        if declared_len != expected_bytes {
            return Err(ProtocolError::TrailerLengthMismatch {
                expected: expected_bytes,
                got: declared_len,
            });
        }
        read_vectors(r, ok_count, expected_dim)?
    };

    Ok(Frame::EncodeResponse {
        results: per_records,
        vectors,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn encode_f32s(vs: &[Vec<f32>]) -> Vec<u8> {
        let mut out = Vec::new();
        for v in vs {
            for &x in v {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
        out
    }

    fn response_bytes(envelope: &Value, vectors: &[Vec<f32>]) -> Vec<u8> {
        let mut out = serde_json::to_vec(envelope).unwrap();
        out.push(b'\n');
        if !vectors.is_empty() {
            let body = encode_f32s(vectors);
            out.extend_from_slice(&(body.len() as u32).to_le_bytes());
            out.extend_from_slice(&body);
        }
        out
    }

    #[test]
    fn encode_request_roundtrip() {
        let mut buf = Vec::new();
        write_encode_request(&mut buf, &["hello", "world"]).unwrap();
        let s = std::str::from_utf8(&buf).unwrap();
        assert!(s.ends_with('\n'));
        let v: Value = serde_json::from_str(s.trim()).unwrap();
        assert_eq!(v["type"], "encode");
        assert_eq!(v["texts"][0], "hello");
        assert_eq!(v["texts"][1], "world");
    }

    #[test]
    fn parse_valid_handshake() {
        let env = serde_json::json!({
            "type": "handshake",
            "protocol_version": 1,
            "vector_dim": 384,
            "model_id": "stub-v1",
            "max_batch_size": 256,
        });
        let h = parse_handshake(&env).unwrap();
        assert_eq!(h.protocol_version, 1);
        assert_eq!(h.vector_dim, 384);
        assert_eq!(h.model_id, "stub-v1");
        assert_eq!(h.max_batch_size, Some(256));
    }

    #[test]
    fn parse_handshake_wrong_type() {
        let env = serde_json::json!({ "type": "encode_response" });
        let err = parse_handshake(&env).unwrap_err();
        assert!(matches!(err, ProtocolError::UnexpectedType { .. }));
    }

    #[test]
    fn parse_handshake_missing_dim() {
        let env = serde_json::json!({
            "type": "handshake",
            "protocol_version": 1,
            "model_id": "stub",
        });
        let err = parse_handshake(&env).unwrap_err();
        assert!(matches!(err, ProtocolError::MalformedEnvelope(_)));
    }

    #[test]
    fn read_encode_response_all_ok() {
        let vectors = vec![vec![1.0_f32, 2.0, 3.0, 4.0], vec![5.0_f32, 6.0, 7.0, 8.0]];
        let env = serde_json::json!({
            "type": "encode_response",
            "results": [{"ok": true}, {"ok": true}],
            "vector_count": 2,
            "vector_dim": 4,
        });
        let bytes = response_bytes(&env, &vectors);
        let mut cur = Cursor::new(bytes);
        let frame = read_encode_response(&mut cur, 4).unwrap();
        match frame {
            Frame::EncodeResponse {
                results,
                vectors: vs,
            } => {
                assert_eq!(results.len(), 2);
                assert!(matches!(results[0], PerRecord::Ok(0)));
                assert!(matches!(results[1], PerRecord::Ok(1)));
                assert_eq!(vs, vectors);
            }
            _ => panic!("expected EncodeResponse"),
        }
    }

    #[test]
    fn read_encode_response_partial_errors() {
        // Two records succeed, one fails in the middle.
        let vectors = vec![vec![1.0_f32, 2.0], vec![3.0_f32, 4.0]];
        let env = serde_json::json!({
            "type": "encode_response",
            "results": [
                {"ok": true},
                {"error": "content policy"},
                {"ok": true}
            ],
            "vector_count": 2,
            "vector_dim": 2,
        });
        let bytes = response_bytes(&env, &vectors);
        let mut cur = Cursor::new(bytes);
        let frame = read_encode_response(&mut cur, 2).unwrap();
        match frame {
            Frame::EncodeResponse {
                results,
                vectors: vs,
            } => {
                assert!(matches!(results[0], PerRecord::Ok(0)));
                assert!(matches!(results[1], PerRecord::Err(ref m) if m == "content policy"));
                assert!(matches!(results[2], PerRecord::Ok(1)));
                assert_eq!(vs.len(), 2);
                assert_eq!(vs[0], vec![1.0, 2.0]);
                assert_eq!(vs[1], vec![3.0, 4.0]);
            }
            _ => panic!("expected EncodeResponse"),
        }
    }

    #[test]
    fn read_encode_response_whole_batch_error() {
        let env = serde_json::json!({
            "type": "encode_response",
            "error": "rate limit exceeded"
        });
        let mut out = serde_json::to_vec(&env).unwrap();
        out.push(b'\n');
        let mut cur = Cursor::new(out);
        let frame = read_encode_response(&mut cur, 4).unwrap();
        match frame {
            Frame::BatchError(msg) => assert_eq!(msg, "rate limit exceeded"),
            _ => panic!("expected BatchError"),
        }
    }

    #[test]
    fn read_encode_response_trailer_length_mismatch() {
        // Declare 2 vectors of dim 4 (32 bytes) but send only 16 bytes.
        let env = serde_json::json!({
            "type": "encode_response",
            "results": [{"ok": true}, {"ok": true}],
            "vector_count": 2,
            "vector_dim": 4,
        });
        let mut bytes = serde_json::to_vec(&env).unwrap();
        bytes.push(b'\n');
        // Length prefix declares 16, expected is 32.
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&[0u8; 16]);
        let mut cur = Cursor::new(bytes);
        let err = read_encode_response(&mut cur, 4).unwrap_err();
        assert!(matches!(err, ProtocolError::TrailerLengthMismatch { .. }));
    }

    #[test]
    fn read_encode_response_vector_count_mismatch() {
        // Results say 2 ok, but vector_count claims 3.
        let env = serde_json::json!({
            "type": "encode_response",
            "results": [{"ok": true}, {"ok": true}],
            "vector_count": 3,
            "vector_dim": 2,
        });
        let mut bytes = serde_json::to_vec(&env).unwrap();
        bytes.push(b'\n');
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&[0u8; 16]);
        let mut cur = Cursor::new(bytes);
        let err = read_encode_response(&mut cur, 2).unwrap_err();
        assert!(matches!(err, ProtocolError::VectorCountMismatch { .. }));
    }

    #[test]
    fn read_envelope_empty_line_errors() {
        let mut cur = Cursor::new(b"\n".to_vec());
        let err = read_envelope(&mut cur).unwrap_err();
        assert!(matches!(err, ProtocolError::MalformedEnvelope(_)));
    }

    #[test]
    fn read_envelope_eof() {
        let mut cur = Cursor::new(Vec::<u8>::new());
        let err = read_envelope(&mut cur).unwrap_err();
        assert!(matches!(err, ProtocolError::Eof));
    }

    #[test]
    fn read_envelope_malformed_json() {
        let mut cur = Cursor::new(b"not json\n".to_vec());
        let err = read_envelope(&mut cur).unwrap_err();
        assert!(matches!(err, ProtocolError::MalformedEnvelope(_)));
    }

    #[test]
    fn read_vectors_exact_bytes() {
        let vs = vec![vec![1.0_f32, 2.0], vec![3.0_f32, 4.0], vec![5.0_f32, 6.0]];
        let bytes = encode_f32s(&vs);
        let mut cur = Cursor::new(bytes);
        let read = read_vectors(&mut cur, 3, 2).unwrap();
        assert_eq!(read, vs);
    }

    #[test]
    fn read_vectors_truncated_errors() {
        // 2 vectors of dim 4 = 32 bytes, supply 16.
        let bytes = vec![0u8; 16];
        let mut cur = Cursor::new(bytes);
        let err = read_vectors(&mut cur, 2, 4).unwrap_err();
        assert!(matches!(err, ProtocolError::Io(_)));
    }

    #[test]
    fn read_encode_response_zero_ok_no_trailer() {
        // All records errored per-record; no trailer expected.
        let env = serde_json::json!({
            "type": "encode_response",
            "results": [{"error": "a"}, {"error": "b"}],
            "vector_count": 0,
            "vector_dim": 2,
        });
        let mut bytes = serde_json::to_vec(&env).unwrap();
        bytes.push(b'\n');
        let mut cur = Cursor::new(bytes);
        let frame = read_encode_response(&mut cur, 2).unwrap();
        match frame {
            Frame::EncodeResponse { results, vectors } => {
                assert_eq!(results.len(), 2);
                assert!(vectors.is_empty());
            }
            _ => panic!("expected EncodeResponse"),
        }
    }
}
