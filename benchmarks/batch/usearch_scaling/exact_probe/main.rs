//! Isolated exact scoring probe; includes the production scorer without modifying it.
#[path = "../../../../src/scoring/exact.rs"]
mod exact;

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::hint::black_box;
use std::time::Instant;

// Benchmark-only equivalent of src/scoring/exact.rs at 446d45e.
fn original(a: &str, b: &str) -> f64 {
    let a = a.trim();
    let b = b.trim();
    if a.is_empty() && b.is_empty() { return 0.0; }
    if a.eq_ignore_ascii_case(b) || a.to_lowercase() == b.to_lowercase() { 1.0 } else { 0.0 }
}
fn normalize(s: &str) -> String { s.trim().to_lowercase() }
fn hash(s: &str) -> u64 { let mut h = DefaultHasher::new(); s.hash(&mut h); h.finish() }
fn data(kind: &str, n: usize) -> Vec<String> {
    let special = ["", "  ", "\u{2003}", " GB ", "gb", "Café", "CAFÉ", "İ", "i\u{307}", "Σ", "σ", "ς", "ΟΣ", "οσ", "ß", "SS"];
    (0..n).map(|i| {
        if i < special.len() { return special[i].to_owned(); }
        // Repeated entities allow both duplicate IDs and unsuccessful comparisons.
        let k = i / 3;
        let s = match kind {
            "codes" => format!("{:02}{}", k % 97, (b'A' + (k % 26) as u8) as char),
            "lei" => format!("{:018}{}{}", k, (b'A' + (k % 26) as u8) as char, k % 10),
            "unicode" => format!("Société Café 東京 Αθήνα İSTANBUL entity {k:09} — international holding"),
            _ => panic!("unknown corpus"),
        };
        match i % 3 { 0 => format!("  {s}  "), 1 => s.to_lowercase(), _ => s }
    }).collect()
}
struct Prepared { normalized: Vec<String>, hashes: Vec<u64>, ids: Vec<u32> }
impl Prepared {
    fn new(raw: &[String], mode: &str) -> Self {
        let normalized: Vec<_> = if matches!(mode, "original_446d45e" | "ascii_fast") { vec![] } else { raw.iter().map(|s| normalize(s)).collect() };
        let hashes = if mode == "hash" { normalized.iter().map(|s| hash(s)).collect() } else { vec![] };
        let ids = if mode == "intern" {
            let mut dict = HashMap::new();
            normalized.iter().map(|s| { let next = dict.len() as u32 + 1; if s.is_empty() { 0 } else { *dict.entry(s.as_str()).or_insert(next) } }).collect()
        } else { vec![] };
        Self { normalized, hashes, ids }
    }
    fn score(&self, raw: &[String], mode: &str, a: usize, b: usize) -> f64 {
        let equal = match mode {
            "original_446d45e" => return original(&raw[a], &raw[b]),
            "ascii_fast" => return exact::score(&raw[a], &raw[b]),
            "normalized" => !self.normalized[a].is_empty() && self.normalized[a] == self.normalized[b],
            "hash" => !self.normalized[a].is_empty() && self.hashes[a] == self.hashes[b] && self.normalized[a] == self.normalized[b],
            "intern" => self.ids[a] != 0 && self.ids[a] == self.ids[b],
            _ => panic!("unknown mode"),
        };
        if equal { 1.0 } else { 0.0 }
    }
}
// At most 32 candidates/query: two nearby duplicate opportunities plus deterministic
// pseudo-random distractors. No all-pairs or truth-only shortcut.
fn candidate(q: usize, slot: usize, n: usize) -> usize {
    match slot { 0 => (q + 1) % n, 1 => (q + 2) % n, _ => (q.wrapping_mul(104729).wrapping_add(slot * 7919)) % n }
}
fn verify() {
    verify_adversarial();
    for kind in ["codes", "lei", "unicode"] {
        let raw = data(kind, 256);
        for mode in ["ascii_fast", "normalized", "hash", "intern"] {
            let p = Prepared::new(&raw, mode);
            for a in 0..raw.len() { for b in 0..raw.len() {
                assert_eq!(original(&raw[a], &raw[b]).to_bits(), p.score(&raw, mode, a, b).to_bits(), "{kind}/{mode} pair {a},{b}");
            }}
            // Force all hashes to collide: equality verification must remain correct.
            if mode == "hash" { let mut p = p; p.hashes.fill(0); for a in 0..raw.len() { for b in 0..raw.len() { assert_eq!(exact::score(&raw[a], &raw[b]), p.score(&raw, mode, a, b)); } } }
        }
    }
    println!("verification: every pair identical, including forced hash collisions");
}
// Final_Sigma is contextual: test ASCII-case bypass and lowercase fallback
// together, not merely isolated Unicode characters or aggregate checksums.
fn verify_adversarial() {
    let mut raw: Vec<String> = [
        "", " \u{2003}", "AΣ", "aΣ", "aς", "aσ", "ΣA", "Σa",
        "ΟΣ", "οΣ", "ος", "οσ", "İ", "i\u{307}", "I", "i", "ı",
        "K", "K", "k", "ß", "SS", "ẞ", "Σ\u{301}",
    ].iter().map(|s| (*s).to_owned()).collect();
    // Include cased/case-ignorable marks, punctuation, and expanding lowercase.
    for context in ["", "'", "\u{301}", "\u{345}", "İ", "Σ", "ς", "1", " "] {
        for c in b'A'..=b'Z' {
            for letter in [c as char, (c + 32) as char] {
                raw.push(format!(" {letter}{context}Σ "));
                raw.push(format!(" Σ{context}{letter} "));
            }
        }
    }
    assert_ne!(normalize("AΣ"), "AΣ".chars().flat_map(char::to_lowercase).collect::<String>(),
        "character-wise lowercase is NOT an equivalent cache key");
    assert_ne!(normalize("AΣ"), normalize("aσ"), "final sigma must be contextual");
    assert_eq!(original("AΣ", "aΣ"), 1.0, "ASCII bypass with non-ASCII sigma");
    assert_eq!(normalize("AΣ"), normalize("aΣ"), "bypass must agree with normalization");
    assert_eq!(original("AΣ", "aς"), 1.0, "contextual lowercase fallback");
    assert_eq!(original("AΣ", "aσ"), 0.0, "normal sigma is not final sigma");
    for mode in ["ascii_fast", "normalized", "hash", "intern"] {
        let mut p = Prepared::new(&raw, mode);
        if mode == "hash" { p.hashes.fill(0); }
        for a in 0..raw.len() { for b in 0..raw.len() {
            assert_eq!(original(&raw[a], &raw[b]).to_bits(), p.score(&raw, mode, a, b).to_bits(),
                "adversarial/{mode}: {:?} vs {:?}", raw[a], raw[b]);
        }}
    }
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("verify") { verify(); return; }
    let mode = &args[1]; let kind = &args[2];
    let n: usize = args.get(3).map(|s| s.parse().expect("n")).unwrap_or(100_000);
    let rounds: usize = args.get(4).map(|s| s.parse().expect("rounds")).unwrap_or(4);
    let raw = data(kind, n);
    let start = Instant::now();
    let p = Prepared::new(&raw, mode);
    let prep = start.elapsed().as_secs_f64();
    // Full shortlist validation, outside timing; compare each pair, not aggregate.
    for q in 0..n { for slot in 0..32 { let b = candidate(q, slot, n); assert_eq!(original(&raw[q], &raw[b]).to_bits(), p.score(&raw, mode, q, b).to_bits(), "pair {q},{b}"); } }
    let start = Instant::now(); let mut sum = 0.0;
    for _ in 0..rounds { for q in 0..n { for slot in 0..32 { sum += black_box(p.score(black_box(&raw), black_box(mode), black_box(q), black_box(candidate(q, slot, n)))); } } }
    let elapsed = start.elapsed().as_secs_f64();
    println!("{mode},{kind},{n},32,{rounds},{prep:.9},{elapsed:.9},{:.0},{sum}", n as f64 * 32.0 * rounds as f64 / elapsed);
}
