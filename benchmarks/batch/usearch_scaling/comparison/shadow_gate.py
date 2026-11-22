#!/usr/bin/env python3
"""Instrument a disposable candidate source tree. No timing claims for this build.
For each blocked query compare cached IDs/retrieval with another backend lookup
and uncached BM25 on the SAME index. Stabilize query term order in both calls
so randomized HashMap iteration does not confound a tie at the top-k cutoff.
"""
import argparse
from pathlib import Path


def once(text, needle, repl):
    if text.count(needle) != 1:
        raise ValueError(f'expected one occurrence of {needle!r}')
    return text.replace(needle, repl)


def patch(root):
    engine = root / 'src/batch/engine.rs'
    bm25 = root / 'src/bm25/simple.rs'
    a, b = engine.read_text(), bm25.read_text()
    b = once(b,'        let term_idfs: Vec<', '        let mut term_idfs: Vec<')
    b = once(b,'        if term_idfs.is_empty()', '        term_idfs.sort_unstable_by_key(|term| term.0);\n        if term_idfs.is_empty()')
    b = once(b,'        for (term, &tf) in &tf_map {', '''        let mut terms: Vec<_> = tf_map.iter().collect();
        terms.sort_unstable_by_key(|(term, _)| *term);
        for (term, &tf) in terms {''')
    a = once(a,'    let scoring_start = Instant::now();','''    let checks = std::sync::atomic::AtomicUsize::new(0);
    let scoring_start = Instant::now();''')
    a = once(a,'            // Synonym candidate generation.', '''            if config.blocking.enabled && let Some(idx) = &bm25_index_a {
                let fresh = reference.blocking_query_from(&b_record, Side::B)?;
                assert_eq!(fresh, blocked_ids, "backend eligibility or order changed for {b_id}");
                let (old_ids, old_scores) = pipeline::bm25_candidate_stage(idx, &b_record, Side::B, Some(&fresh), bm25_candidates_n);
                assert_eq!(old_ids, bm25_cand_ids, "BM25 shortlist/order changed for {b_id}");
                assert_eq!(old_scores, bm25_scores_map, "BM25 score changed for {b_id}");
                checks.fetch_add(1,Ordering::Relaxed);
            }
            // Synonym candidate generation.''')
    a = once(a,'    let scoring_elapsed = scoring_start.elapsed().as_secs_f64();','''    let scoring_elapsed = scoring_start.elapsed().as_secs_f64();
    let snapshot = block_cache.lock().unwrap_or_else(|e| e.into_inner());
    eprintln!("SHADOW_QUERIES_OK={} retained_blocks={} estimated_bytes={}",checks.load(Ordering::Relaxed), snapshot.entries.len(), snapshot.bytes);
    drop(snapshot);''')
    bm25.write_text(b); engine.write_text(a)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('disposable_tree', type=Path); patch(ap.parse_args().disposable_tree)
