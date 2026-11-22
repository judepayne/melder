#!/usr/bin/env python3
"""Instrument a DISPOSABLE source tree at 446d45e, not the measured release binary.
Timers sum elapsed time across concurrent query workers; never read them as wall time.
"""
import argparse
from pathlib import Path


def patch(root):
    engine = root/'src/batch/engine.rs'
    bm25 = root/'src/bm25/simple.rs'
    a, b = engine.read_text(), bm25.read_text()
    def sub(s, old, new):
        if s.count(old) != 1:
            raise ValueError('wrong source revision or already patched: '+old)
        return s.replace(old,new)
    a = sub(a, '    let scoring_start = Instant::now();', '''    let clocks = [const { std::sync::atomic::AtomicU64::new(0) }; 4];
    let mark = |slot:usize, since: Instant| { clocks[slot].fetch_add(since.elapsed().as_nanos() as u64,std::sync::atomic::Ordering::Relaxed); };
    let scoring_start = Instant::now();''')
    for i,(start,end) in enumerate([('// Pre-query the blocking index','// Fetch the query combined vec'),('// BM25 candidate generation (pre-computed','// Synonym candidate generation.'),('// Synonym candidate generation.','let scoring_query = pipeline::PreparedQuery'),('let results = pipeline::search(','// Send to scoring log if enabled.')]):
        a=sub(a,'            '+start,'            let tick = Instant::now();\n            '+start)
        a=sub(a,'            '+end,f'            mark({i},tick);\n            '+end)
    a=sub(a,'    let scoring_elapsed = scoring_start.elapsed().as_secs_f64();','''    let scoring_elapsed = scoring_start.elapsed().as_secs_f64();
    eprintln!("WORKER_NS block={} bm25_inclusive={} synonym={} search={} membership_within_bm25={}", clocks[0].load(Ordering::Relaxed),clocks[1].load(Ordering::Relaxed),clocks[2].load(Ordering::Relaxed),clocks[3].load(Ordering::Relaxed),crate::bm25::simple::MEMBERSHIP.load(Ordering::Relaxed));''')
    b=sub(b,'const DEFAULT_EXHAUSTIVE_THRESHOLD:', 'pub(crate) static MEMBERSHIP: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);\nconst DEFAULT_EXHAUSTIVE_THRESHOLD:')
    b=sub(b,'        let allowed: Option<std::collections::HashSet<u32>>', '        let tick = std::time::Instant::now();\n        let allowed: Option<std::collections::HashSet<u32>>')
    b=sub(b,'        if allowed.as_ref().is_some_and(|ids| ids.is_empty()) {','        MEMBERSHIP.fetch_add(tick.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);\n        if allowed.as_ref().is_some_and(|ids| ids.is_empty()) {')
    engine.write_text(a); bm25.write_text(b)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('disposable_tree',type=Path);patch(ap.parse_args().disposable_tree)
