//! Generic test suite for VectorDB implementations.
//!
//! Each test runs via a macro that generates a module per backend.
//! Currently runs against FlatVectorDB.

use crate::models::{Record, Side};

use super::VectorDB;

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

/// Empty record for flat backend tests (ignored by FlatVectorDB).
fn dummy_record() -> Record {
    Record::new()
}

/// Default side for flat backend tests (ignored by FlatVectorDB).
const SIDE: Side = Side::A;

/// Deterministic pseudo-random unit vector (same LCG as VecIndex tests).
fn random_unit_vec(dim: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_add(1);
    let mut v: Vec<f32> = (0..dim)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as i32 as f32) / (i32::MAX as f32)
        })
        .collect();
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

// ---------------------------------------------------------------------------
// Macro to generate tests for a given backend
// ---------------------------------------------------------------------------

macro_rules! vectordb_tests {
    ($mod_name:ident, $factory:expr) => {
        mod $mod_name {
            use super::*;
            use std::collections::HashSet;

            const DIM: usize = 384;

            fn make_db() -> Box<dyn VectorDB> {
                Box::new($factory())
            }

            #[test]
            fn upsert_and_len() {
                let db = make_db();
                assert_eq!(db.len(), 0);
                assert!(db.is_empty());

                db.upsert("a", &random_unit_vec(DIM, 0), &dummy_record(), SIDE)
                    .unwrap();
                assert_eq!(db.len(), 1);
                assert!(!db.is_empty());

                db.upsert("b", &random_unit_vec(DIM, 1), &dummy_record(), SIDE)
                    .unwrap();
                assert_eq!(db.len(), 2);
            }

            #[test]
            fn upsert_replaces() {
                let db = make_db();
                let v1 = random_unit_vec(DIM, 0);
                let v2 = random_unit_vec(DIM, 1);

                db.upsert("a", &v1, &dummy_record(), SIDE).unwrap();
                db.upsert("a", &v2, &dummy_record(), SIDE).unwrap();
                assert_eq!(db.len(), 1);

                let got = db.get("a").unwrap().unwrap();
                let dot: f32 = got.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
                assert!(dot > 0.99, "expected vector to match v2 (dot={})", dot);
            }

            #[test]
            fn contains_and_get() {
                let db = make_db();
                let v = random_unit_vec(DIM, 42);

                assert!(!db.contains("x"));
                assert!(db.get("x").unwrap().is_none());

                db.upsert("x", &v, &dummy_record(), SIDE).unwrap();
                assert!(db.contains("x"));

                let got = db.get("x").unwrap().unwrap();
                assert_eq!(got.len(), DIM);
                let dot: f32 = got.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
                assert!(dot > 0.99, "retrieved vector should match (dot={})", dot);
            }

            #[test]
            fn remove_basic() {
                let db = make_db();
                db.upsert("a", &random_unit_vec(DIM, 0), &dummy_record(), SIDE)
                    .unwrap();
                db.upsert("b", &random_unit_vec(DIM, 1), &dummy_record(), SIDE)
                    .unwrap();
                db.upsert("c", &random_unit_vec(DIM, 2), &dummy_record(), SIDE)
                    .unwrap();

                assert_eq!(db.len(), 3);
                assert!(db.remove("b").unwrap());
                assert_eq!(db.len(), 2);
                assert!(!db.contains("b"));
                assert!(db.contains("a"));
                assert!(db.contains("c"));
            }

            #[test]
            fn remove_nonexistent() {
                let db = make_db();
                assert!(!db.remove("xyz").unwrap());
            }

            #[test]
            fn remove_then_reinsert() {
                let db = make_db();
                let v = random_unit_vec(DIM, 0);

                db.upsert("a", &v, &dummy_record(), SIDE).unwrap();
                assert!(db.remove("a").unwrap());
                assert!(!db.contains("a"));
                assert_eq!(db.len(), 0);

                db.upsert("a", &v, &dummy_record(), SIDE).unwrap();
                assert!(db.contains("a"));
                assert_eq!(db.len(), 1);
            }

            #[test]
            fn search_self_match() {
                let db = make_db();
                let n = 100;
                for i in 0..n {
                    db.upsert(
                        &format!("id_{}", i),
                        &random_unit_vec(DIM, i as u64),
                        &dummy_record(),
                        SIDE,
                    )
                    .unwrap();
                }

                let query = random_unit_vec(DIM, 0);
                let results = db.search(&query, 5, &dummy_record(), SIDE).unwrap();

                assert_eq!(results.len(), 5);
                assert_eq!(results[0].id, "id_0");
                assert!(
                    (results[0].score - 1.0).abs() < 0.02,
                    "self-similarity = {}, expected ~1.0",
                    results[0].score
                );
            }

            #[test]
            fn search_sorted_descending() {
                let db = make_db();
                for i in 0..50 {
                    db.upsert(
                        &format!("id_{}", i),
                        &random_unit_vec(DIM, i as u64),
                        &dummy_record(),
                        SIDE,
                    )
                    .unwrap();
                }

                let query = random_unit_vec(DIM, 999);
                let results = db.search(&query, 10, &dummy_record(), SIDE).unwrap();

                for w in results.windows(2) {
                    assert!(
                        w[0].score >= w[1].score - 0.001,
                        "not sorted: {} < {}",
                        w[0].score,
                        w[1].score
                    );
                }
            }

            #[test]
            fn search_k_larger_than_n() {
                let db = make_db();
                db.upsert("a", &random_unit_vec(DIM, 0), &dummy_record(), SIDE)
                    .unwrap();
                db.upsert("b", &random_unit_vec(DIM, 1), &dummy_record(), SIDE)
                    .unwrap();

                let results = db
                    .search(&random_unit_vec(DIM, 99), 100, &dummy_record(), SIDE)
                    .unwrap();
                assert_eq!(results.len(), 2);
            }

            #[test]
            fn search_empty() {
                let db = make_db();
                let results = db
                    .search(&random_unit_vec(DIM, 0), 5, &dummy_record(), SIDE)
                    .unwrap();
                assert!(results.is_empty());
            }

            #[test]
            fn search_filtered_basic() {
                let db = make_db();
                db.upsert("a", &random_unit_vec(DIM, 0), &dummy_record(), SIDE)
                    .unwrap();
                db.upsert("b", &random_unit_vec(DIM, 1), &dummy_record(), SIDE)
                    .unwrap();
                db.upsert("c", &random_unit_vec(DIM, 2), &dummy_record(), SIDE)
                    .unwrap();
                db.upsert("d", &random_unit_vec(DIM, 3), &dummy_record(), SIDE)
                    .unwrap();

                let allowed: HashSet<String> = ["c", "d"].iter().map(|s| s.to_string()).collect();

                let query = random_unit_vec(DIM, 0);
                let results = db
                    .search_filtered(&query, 10, &allowed, &dummy_record(), SIDE)
                    .unwrap();

                assert!(results.len() <= 2);
                for r in &results {
                    assert!(
                        allowed.contains(&r.id),
                        "unexpected id '{}' not in allowed set",
                        r.id
                    );
                }
            }

            #[test]
            fn search_filtered_empty_allowed() {
                let db = make_db();
                db.upsert("a", &random_unit_vec(DIM, 0), &dummy_record(), SIDE)
                    .unwrap();

                let allowed: HashSet<String> = HashSet::new();
                let results = db
                    .search_filtered(
                        &random_unit_vec(DIM, 0),
                        10,
                        &allowed,
                        &dummy_record(),
                        SIDE,
                    )
                    .unwrap();
                assert!(results.is_empty());
            }

            #[test]
            fn search_after_remove() {
                let db = make_db();
                db.upsert("a", &random_unit_vec(DIM, 0), &dummy_record(), SIDE)
                    .unwrap();
                db.upsert("b", &random_unit_vec(DIM, 1), &dummy_record(), SIDE)
                    .unwrap();

                db.remove("a").unwrap();

                let results = db
                    .search(&random_unit_vec(DIM, 0), 10, &dummy_record(), SIDE)
                    .unwrap();
                for r in &results {
                    assert_ne!(r.id, "a", "removed id should not appear in search results");
                }
            }

            #[test]
            fn dimension_mismatch_upsert() {
                let db = make_db();
                let wrong_dim = vec![1.0_f32; DIM + 1];
                let result = db.upsert("a", &wrong_dim, &dummy_record(), SIDE);
                assert!(result.is_err());
            }

            #[test]
            fn dimension_mismatch_search() {
                let db = make_db();
                db.upsert("a", &random_unit_vec(DIM, 0), &dummy_record(), SIDE)
                    .unwrap();

                let wrong_dim = vec![1.0_f32; DIM + 1];
                let result = db.search(&wrong_dim, 5, &dummy_record(), SIDE);
                assert!(result.is_err());
            }

            #[test]
            fn dim_accessor() {
                let db = make_db();
                assert_eq!(db.dim(), DIM);
            }

            #[test]
            fn save_creates_file() {
                let db = make_db();
                for i in 0..10 {
                    db.upsert(
                        &format!("id_{}", i),
                        &random_unit_vec(DIM, i as u64),
                        &dummy_record(),
                        SIDE,
                    )
                    .unwrap();
                }

                let dir = tempfile::tempdir().unwrap();
                let path = dir.path().join("test.vectordb");
                db.save(&path).unwrap();
                // Flat backend creates the file at `path`; usearch backend
                // creates a `.usearchdb` directory alongside it.
                let saved = path.exists() || path.with_extension("usearchdb").exists();
                assert!(saved, "neither {:?} nor the .usearchdb dir exists", path);
            }

            #[test]
            fn many_upserts_and_removes() {
                let db = make_db();

                for i in 0..200 {
                    db.upsert(
                        &format!("id_{}", i),
                        &random_unit_vec(DIM, i as u64),
                        &dummy_record(),
                        SIDE,
                    )
                    .unwrap();
                }
                assert_eq!(db.len(), 200);

                for i in 0..100 {
                    db.remove(&format!("id_{}", i)).unwrap();
                }
                assert_eq!(db.len(), 100);

                for i in 0..100 {
                    assert!(
                        !db.contains(&format!("id_{}", i)),
                        "id_{} should have been removed",
                        i
                    );
                }
                for i in 100..200 {
                    assert!(
                        db.contains(&format!("id_{}", i)),
                        "id_{} should still exist",
                        i
                    );
                }

                let results = db
                    .search(&random_unit_vec(DIM, 150), 5, &dummy_record(), SIDE)
                    .unwrap();
                for r in &results {
                    let num: usize = r.id.strip_prefix("id_").unwrap().parse().unwrap();
                    assert!(num >= 100, "search returned removed id: {}", r.id);
                }
            }
        }
    };
}

// ---------------------------------------------------------------------------
// Run the generic suite against FlatVectorDB
// ---------------------------------------------------------------------------

vectordb_tests!(flat_tests, || {
    crate::vectordb::flat::FlatVectorDB::new(DIM)
});

// ---------------------------------------------------------------------------
// Run the generic suite against UsearchVectorDB (no blocking → single block)
// ---------------------------------------------------------------------------

#[cfg(feature = "usearch")]
vectordb_tests!(usearch_tests, || {
    crate::vectordb::usearch_backend::UsearchVectorDB::new(DIM, None)
});

// ---------------------------------------------------------------------------
// FlatVectorDB-specific tests (save/load round-trip, staleness)
// ---------------------------------------------------------------------------

mod flat_persistence_tests {
    use super::*;

    const DIM: usize = 384;

    #[test]
    fn save_and_load_roundtrip() {
        let db = crate::vectordb::flat::FlatVectorDB::new(DIM);
        let n = 50;
        let vecs: Vec<Vec<f32>> = (0..n).map(|i| random_unit_vec(DIM, i as u64)).collect();

        for (i, v) in vecs.iter().enumerate() {
            db.upsert(&format!("id_{}", i), v, &dummy_record(), SIDE)
                .unwrap();
        }

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.vectordb");

        db.save(&path).unwrap();

        let loaded = crate::vectordb::flat::FlatVectorDB::load(&path).unwrap();

        assert_eq!(loaded.len(), n);
        assert_eq!(loaded.dim(), DIM);

        for (i, v) in vecs.iter().enumerate() {
            let id = format!("id_{}", i);
            assert!(loaded.contains(&id));
            let got = loaded.get(&id).unwrap().unwrap();
            let dot: f32 = got.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
            assert!(
                dot > 0.999,
                "vector {} corrupted after round-trip (dot={})",
                id,
                dot
            );
        }

        let query = random_unit_vec(DIM, 0);
        let results = loaded.search(&query, 5, &dummy_record(), SIDE).unwrap();
        assert_eq!(results[0].id, "id_0");
    }

    #[test]
    fn staleness_check() {
        let db = crate::vectordb::flat::FlatVectorDB::new(DIM);
        for i in 0..10 {
            db.upsert(
                &format!("id_{}", i),
                &random_unit_vec(DIM, i as u64),
                &dummy_record(),
                SIDE,
            )
            .unwrap();
        }

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.vectordb");
        db.save(&path).unwrap();

        assert!(!crate::vectordb::flat::FlatVectorDB::is_stale(&path, 10).unwrap());
        assert!(crate::vectordb::flat::FlatVectorDB::is_stale(&path, 11).unwrap());
        assert!(
            crate::vectordb::flat::FlatVectorDB::is_stale(&dir.path().join("nonexistent"), 10)
                .unwrap()
        );
    }
}

// ---------------------------------------------------------------------------
// UsearchVectorDB-specific tests (blocking, persistence, cross-block)
// ---------------------------------------------------------------------------

#[cfg(feature = "usearch")]
mod usearch_block_tests {
    use super::*;
    use crate::config::schema::{BlockingConfig, BlockingFieldPair};
    use crate::vectordb::usearch_backend::UsearchVectorDB;
    use std::collections::HashSet;

    const DIM: usize = 16; // smaller dim for faster block tests

    fn make_record(fields: &[(&str, &str)]) -> Record {
        fields
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    fn blocking_config(field_a: &str, field_b: &str) -> BlockingConfig {
        BlockingConfig {
            enabled: true,
            operator: "and".to_string(),
            fields: vec![BlockingFieldPair {
                field_a: field_a.to_string(),
                field_b: field_b.to_string(),
            }],
            field_a: None,
            field_b: None,
        }
    }

    #[test]
    fn records_in_same_block_find_each_other() {
        let cfg = blocking_config("country", "country");
        let db = UsearchVectorDB::new(DIM, Some(&cfg));

        let v0 = random_unit_vec(DIM, 0);
        let v1 = random_unit_vec(DIM, 1);

        let rec_us = make_record(&[("country", "US")]);

        db.upsert("a", &v0, &rec_us, Side::A).unwrap();
        db.upsert("b", &v1, &rec_us, Side::B).unwrap();

        let results = db.search(&v0, 5, &rec_us, Side::A).unwrap();
        assert!(!results.is_empty());
        // Should find at least the self-match.
        assert!(results.iter().any(|r| r.id == "a"));
    }

    #[test]
    fn records_in_different_blocks_are_isolated() {
        let cfg = blocking_config("country", "country");
        let db = UsearchVectorDB::new(DIM, Some(&cfg));

        let v0 = random_unit_vec(DIM, 0);
        let v1 = random_unit_vec(DIM, 1);

        let rec_us = make_record(&[("country", "US")]);
        let rec_gb = make_record(&[("country", "GB")]);

        db.upsert("us_1", &v0, &rec_us, Side::A).unwrap();
        db.upsert("gb_1", &v1, &rec_gb, Side::A).unwrap();

        // Searching from US block should not find GB records.
        let results = db.search(&v1, 10, &rec_us, Side::A).unwrap();
        for r in &results {
            assert_ne!(
                r.id, "gb_1",
                "cross-block leak: found GB record from US search"
            );
        }

        // And vice versa.
        let results = db.search(&v0, 10, &rec_gb, Side::A).unwrap();
        for r in &results {
            assert_ne!(
                r.id, "us_1",
                "cross-block leak: found US record from GB search"
            );
        }
    }

    #[test]
    fn missing_blocking_field_uses_default_block() {
        let cfg = blocking_config("country", "country");
        let db = UsearchVectorDB::new(DIM, Some(&cfg));

        let v0 = random_unit_vec(DIM, 0);
        let v1 = random_unit_vec(DIM, 1);

        // Record with country field.
        let rec_us = make_record(&[("country", "US")]);
        // Record without country field.
        let rec_empty = make_record(&[("name", "foo")]);

        db.upsert("us_1", &v0, &rec_us, Side::A).unwrap();
        db.upsert("no_country", &v1, &rec_empty, Side::A).unwrap();

        // no_country record should be in the default block.
        assert!(db.contains("no_country"));
        // Searching from US block should NOT find the no_country record.
        let results = db.search(&v1, 10, &rec_us, Side::A).unwrap();
        for r in &results {
            assert_ne!(r.id, "no_country");
        }
    }

    #[test]
    fn blocking_key_is_case_insensitive() {
        let cfg = blocking_config("country", "country");
        let db = UsearchVectorDB::new(DIM, Some(&cfg));

        let v0 = random_unit_vec(DIM, 0);
        let v1 = random_unit_vec(DIM, 1);

        let rec_upper = make_record(&[("country", "US")]);
        let rec_lower = make_record(&[("country", "us")]);

        db.upsert("a", &v0, &rec_upper, Side::A).unwrap();
        db.upsert("b", &v1, &rec_lower, Side::B).unwrap();

        // Both should be in the same block.
        let results = db.search(&v0, 10, &rec_lower, Side::A).unwrap();
        assert!(results.iter().any(|r| r.id == "a"));
        assert!(results.iter().any(|r| r.id == "b"));
    }

    #[test]
    fn upsert_moves_record_between_blocks() {
        let cfg = blocking_config("country", "country");
        let db = UsearchVectorDB::new(DIM, Some(&cfg));

        let v = random_unit_vec(DIM, 0);

        let rec_us = make_record(&[("country", "US")]);
        let rec_gb = make_record(&[("country", "GB")]);

        // Insert into US block.
        db.upsert("a", &v, &rec_us, Side::A).unwrap();
        let results = db.search(&v, 10, &rec_us, Side::A).unwrap();
        assert!(results.iter().any(|r| r.id == "a"));

        // Move to GB block by re-upserting with different record.
        db.upsert("a", &v, &rec_gb, Side::A).unwrap();

        // Should no longer be in US block.
        let results = db.search(&v, 10, &rec_us, Side::A).unwrap();
        assert!(!results.iter().any(|r| r.id == "a"));

        // Should be in GB block.
        let results = db.search(&v, 10, &rec_gb, Side::A).unwrap();
        assert!(results.iter().any(|r| r.id == "a"));

        assert_eq!(db.len(), 1);
    }

    #[test]
    fn search_filtered_within_block() {
        let cfg = blocking_config("country", "country");
        let db = UsearchVectorDB::new(DIM, Some(&cfg));

        let rec_us = make_record(&[("country", "US")]);

        for i in 0..20 {
            db.upsert(
                &format!("id_{}", i),
                &random_unit_vec(DIM, i as u64),
                &rec_us,
                Side::A,
            )
            .unwrap();
        }

        let allowed: HashSet<String> = ["id_5", "id_10", "id_15"]
            .iter()
            .map(|s| s.to_string())
            .collect();

        let query = random_unit_vec(DIM, 5);
        let results = db
            .search_filtered(&query, 10, &allowed, &rec_us, Side::A)
            .unwrap();

        for r in &results {
            assert!(
                allowed.contains(&r.id),
                "unexpected id '{}' not in allowed set",
                r.id
            );
        }
        assert!(results.len() <= 3);
    }

    #[test]
    fn save_and_load_roundtrip() {
        let cfg = blocking_config("country", "country");
        let db = UsearchVectorDB::new(DIM, Some(&cfg));

        let rec_us = make_record(&[("country", "US")]);
        let rec_gb = make_record(&[("country", "GB")]);

        let n = 10;
        let vecs: Vec<Vec<f32>> = (0..n).map(|i| random_unit_vec(DIM, i as u64)).collect();

        for (i, v) in vecs.iter().enumerate() {
            let rec = if i % 2 == 0 { &rec_us } else { &rec_gb };
            db.upsert(&format!("id_{}", i), v, rec, Side::A).unwrap();
        }

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.usearch");

        db.save(&path).unwrap();

        let loaded = UsearchVectorDB::load(&path, "f32", "load", 0).unwrap();

        assert_eq!(loaded.len(), n);
        assert_eq!(loaded.dim(), DIM);

        // Verify all vectors are retrievable.
        for (i, v) in vecs.iter().enumerate() {
            let id = format!("id_{}", i);
            assert!(loaded.contains(&id), "missing: {}", id);
            let got = loaded.get(&id).unwrap().unwrap();
            let dot: f32 = got.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
            assert!(
                dot > 0.99,
                "vector {} corrupted after round-trip (dot={})",
                id,
                dot
            );
        }

        // Verify block isolation survived round-trip.
        let query = random_unit_vec(DIM, 0);
        let results = loaded.search(&query, 10, &rec_us, Side::A).unwrap();
        for r in &results {
            let num: usize = r.id.strip_prefix("id_").unwrap().parse().unwrap();
            assert!(
                num.is_multiple_of(2),
                "US block search returned GB record: {}",
                r.id
            );
        }
    }

    #[test]
    fn staleness_check() {
        let db = UsearchVectorDB::new(DIM, None);
        for i in 0..5 {
            db.upsert(
                &format!("id_{}", i),
                &random_unit_vec(DIM, i as u64),
                &dummy_record(),
                SIDE,
            )
            .unwrap();
        }

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.usearch");
        db.save(&path).unwrap();

        assert!(!UsearchVectorDB::is_stale(&path, 5).unwrap());
        assert!(UsearchVectorDB::is_stale(&path, 6).unwrap());
        assert!(UsearchVectorDB::is_stale(&dir.path().join("nonexistent"), 5).unwrap());
    }

    #[test]
    fn no_blocking_all_in_one_block() {
        let db = UsearchVectorDB::new(DIM, None);

        for i in 0..50 {
            db.upsert(
                &format!("id_{}", i),
                &random_unit_vec(DIM, i as u64),
                &dummy_record(),
                Side::A,
            )
            .unwrap();
        }

        assert_eq!(db.len(), 50);

        // All records should be searchable from any dummy query.
        let results = db
            .search(&random_unit_vec(DIM, 0), 10, &dummy_record(), Side::A)
            .unwrap();
        assert_eq!(results.len(), 10);
        assert_eq!(results[0].id, "id_0");
    }

    #[test]
    fn f16_quantization_insert_and_search() {
        // F16 quantization: vectors are stored as half-precision but the API
        // still accepts/returns f32. Search should still find self-matches.
        let db = UsearchVectorDB::new_with_emb_specs(DIM, None, Vec::new(), "f16", 0);

        let n = 50;
        for i in 0..n {
            db.upsert(
                &format!("id_{}", i),
                &random_unit_vec(DIM, i as u64),
                &dummy_record(),
                Side::A,
            )
            .unwrap();
        }

        assert_eq!(db.len(), n);

        // Self-match: query for id_0's vector, expect id_0 as top result.
        let query = random_unit_vec(DIM, 0);
        let results = db.search(&query, 5, &dummy_record(), Side::A).unwrap();
        assert_eq!(results[0].id, "id_0");
        // F16 has ~0.1% relative error, so self-similarity should still be
        // very close to 1.0 (within ~0.01).
        assert!(
            results[0].score > 0.98,
            "F16 self-similarity too low: {}",
            results[0].score
        );
    }

    #[test]
    fn bf16_quantization_insert_and_search() {
        let db = UsearchVectorDB::new_with_emb_specs(DIM, None, Vec::new(), "bf16", 0);

        let n = 50;
        for i in 0..n {
            db.upsert(
                &format!("id_{}", i),
                &random_unit_vec(DIM, i as u64),
                &dummy_record(),
                Side::A,
            )
            .unwrap();
        }

        assert_eq!(db.len(), n);

        let query = random_unit_vec(DIM, 0);
        let results = db.search(&query, 5, &dummy_record(), Side::A).unwrap();
        assert_eq!(results[0].id, "id_0");
        assert!(
            results[0].score > 0.97,
            "BF16 self-similarity too low: {}",
            results[0].score
        );
    }

    #[test]
    fn f16_save_and_load_roundtrip() {
        let db = UsearchVectorDB::new_with_emb_specs(DIM, None, Vec::new(), "f16", 0);

        for i in 0..10 {
            db.upsert(
                &format!("id_{}", i),
                &random_unit_vec(DIM, i as u64),
                &dummy_record(),
                Side::A,
            )
            .unwrap();
        }

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_f16.usearch");
        db.save(&path).unwrap();

        let loaded = UsearchVectorDB::load(&path, "f16", "load", 0).unwrap();
        assert_eq!(loaded.len(), 10);

        // Verify search still works after load.
        let query = random_unit_vec(DIM, 0);
        let results = loaded.search(&query, 5, &dummy_record(), Side::A).unwrap();
        assert_eq!(results[0].id, "id_0");
        assert!(
            results[0].score > 0.98,
            "F16 round-trip self-similarity too low: {}",
            results[0].score
        );
    }

    #[test]
    fn mmap_load_same_results_as_in_memory_load() {
        // Build a small index, save it, then load once with "load" and once
        // with "mmap". Both must return identical search results.
        let db = UsearchVectorDB::new(DIM, None);
        let n = 20;
        let vecs: Vec<Vec<f32>> = (0..n).map(|i| random_unit_vec(DIM, i as u64)).collect();
        for (i, v) in vecs.iter().enumerate() {
            db.upsert(&format!("id_{}", i), v, &dummy_record(), Side::A)
                .unwrap();
        }

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_mmap.usearch");
        db.save(&path).unwrap();

        let loaded = UsearchVectorDB::load(&path, "f32", "load", 0).unwrap();
        let mmaped = UsearchVectorDB::load(&path, "f32", "mmap", 0).unwrap();

        assert_eq!(mmaped.len(), n, "mmap load: wrong record count");
        assert_eq!(mmaped.dim(), DIM, "mmap load: wrong dimension");

        // Both backends must return the same top-5 IDs in the same order.
        let query = random_unit_vec(DIM, 99);
        let results_load = loaded.search(&query, 5, &dummy_record(), Side::A).unwrap();
        let results_mmap = mmaped.search(&query, 5, &dummy_record(), Side::A).unwrap();

        assert_eq!(
            results_load.len(),
            results_mmap.len(),
            "mmap and load returned different result counts"
        );
        for (rl, rm) in results_load.iter().zip(results_mmap.iter()) {
            assert_eq!(
                rl.id, rm.id,
                "mmap search result mismatch: load={:?} mmap={:?}",
                rl.id, rm.id
            );
        }

        // contains() must work on the mmap'd index.
        for i in 0..n {
            let id = format!("id_{}", i);
            assert!(mmaped.contains(&id), "mmap: missing id {}", id);
        }
    }

    #[test]
    fn mmap_staleness_check_unaffected_by_mode() {
        // is_stale() reads only the manifest JSON (before any load/view call)
        // so vector_index_mode must have no effect on it.
        let db = UsearchVectorDB::new(DIM, None);
        for i in 0..5 {
            db.upsert(
                &format!("id_{}", i),
                &random_unit_vec(DIM, i as u64),
                &dummy_record(),
                SIDE,
            )
            .unwrap();
        }

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_stale_mmap.usearch");
        db.save(&path).unwrap();

        // Staleness is purely count-based — mode is irrelevant.
        assert!(
            !UsearchVectorDB::is_stale(&path, 5).unwrap(),
            "fresh index should not be stale"
        );
        assert!(
            UsearchVectorDB::is_stale(&path, 6).unwrap(),
            "wrong count should be stale"
        );
    }
}

// ---------------------------------------------------------------------------
// Combined-index encoding scheduling and content regressions
// ---------------------------------------------------------------------------

mod encode_and_upsert_tests {
    use std::collections::HashMap;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use rayon::prelude::*;

    use crate::encoder::Encoder;
    use crate::error::{EncoderError, MelderError};
    use crate::vectordb::flat::FlatVectorDB;

    use super::*;

    #[derive(Debug)]
    struct NestedEncoder {
        slots: Vec<Mutex<()>>,
        batch_size: usize,
        active: AtomicUsize,
        peak: AtomicUsize,
        completed: AtomicUsize,
        batches: Mutex<Vec<Vec<String>>>,
        fail: bool,
        panic_on_encode: bool,
    }

    impl NestedEncoder {
        fn new(slots: usize, batch_size: usize, fail: bool) -> Self {
            Self {
                slots: (0..slots).map(|_| Mutex::new(())).collect(),
                batch_size,
                active: AtomicUsize::new(0),
                peak: AtomicUsize::new(0),
                completed: AtomicUsize::new(0),
                batches: Mutex::new(Vec::new()),
                fail,
                panic_on_encode: false,
            }
        }
    }

    fn text_vector(text: &str) -> Vec<f32> {
        let x = text.bytes().map(|b| b as f32).sum::<f32>();
        let y = text.len() as f32 + 1.0;
        let norm = (x * x + y * y).sqrt();
        vec![x / norm, y / norm]
    }

    impl Encoder for NestedEncoder {
        fn encode(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EncoderError> {
            // Assert before taking a session lock or starting nested work:
            // the old outer Rayon scheduler fails deterministically, not by hanging.
            assert!(
                rayon::current_thread_index().is_none(),
                "session-acquiring encoding must never enter on a Rayon worker"
            );
            let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
            self.peak.fetch_max(active, Ordering::SeqCst);
            assert!(
                active <= self.slots.len(),
                "encoding concurrency {active} exceeded advertised slots {}",
                self.slots.len()
            );
            self.batches
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(texts.iter().map(|text| text.to_string()).collect());

            // Mirror local encoding: hold a session mutex throughout nested
            // tokenizer parallelism, including the blocking session fallback.
            let session = self
                .slots
                .iter()
                .find_map(|slot| slot.try_lock().ok())
                .unwrap_or_else(|| self.slots[0].lock().unwrap_or_else(|e| e.into_inner()));
            let vectors = texts.par_iter().map(|text| text_vector(text)).collect();
            drop(session);
            self.completed.fetch_add(1, Ordering::SeqCst);
            self.active.fetch_sub(1, Ordering::SeqCst);
            assert!(!self.panic_on_encode, "intentional encoder panic");
            if self.fail {
                Err(EncoderError::Inference(
                    "intentional nested encoder failure".into(),
                ))
            } else {
                Ok(vectors)
            }
        }

        fn dim(&self) -> usize {
            2
        }

        fn pool_size(&self) -> usize {
            self.slots.len()
        }

        fn encode_batch_size(&self) -> usize {
            self.batch_size
        }
    }

    fn make_records(count: usize) -> (Vec<String>, HashMap<String, Record>) {
        let ids: Vec<String> = (0..count).map(|i| format!("row_{i}")).collect();
        let records = ids
            .iter()
            .enumerate()
            .map(|(i, id)| {
                let mut record = Record::from([
                    ("a_name".into(), format!("  A name {i}  ")),
                    ("b_name".into(), format!("\tB different name {i}\n")),
                    ("a_city".into(), format!(" A city {i} ")),
                    ("b_city".into(), format!(" B different city {i} ")),
                ]);
                if i % 11 == 0 {
                    record.remove("a_city");
                    record.remove("b_city");
                }
                if i % 13 == 0 {
                    record.insert("a_name".into(), " \t ".into());
                    record.insert("b_name".into(), "\n ".into());
                }
                (id.clone(), record)
            })
            .collect();
        (ids, records)
    }

    fn specs() -> Vec<(String, String, f64)> {
        vec![
            ("a_name".into(), "b_name".into(), 0.25),
            ("a_city".into(), "b_city".into(), 0.75),
        ]
    }

    #[test]
    fn encode_and_upsert_nested_rayon_preserves_batches_and_vectors()
    -> Result<(), Box<dyn std::error::Error>> {
        let (ids, records) = make_records(5000);
        let id_refs: Vec<&String> = ids.iter().collect();
        let specs = specs();
        let caller_pool = rayon::ThreadPoolBuilder::new().num_threads(2).build()?;

        // Default/minimum-sized batches, both session counts, both sides,
        // and a custom batch larger than the entire dataset.
        for (slots, batch_size, side, inside_rayon) in [
            (1, 1, Side::A, false),
            (4, 64, Side::B, true),
            (4, 6000, Side::A, true),
        ] {
            let encoder = NestedEncoder::new(slots, batch_size, false);
            let db = FlatVectorDB::new(4);
            let encode = || {
                super::super::encode_and_upsert(
                    &db,
                    &id_refs,
                    &records,
                    &specs,
                    &encoder,
                    4,
                    matches!(side, Side::A),
                    side,
                    "test",
                )
            };
            if inside_rayon {
                caller_pool.install(|| {
                    assert!(
                        rayon::current_thread_index().is_some(),
                        "caller must be on Rayon"
                    );
                    encode()
                })?;
            } else {
                encode()?;
            }

            assert_eq!(db.len(), ids.len(), "every requested ID must be indexed");
            assert_eq!(
                encoder.active.load(Ordering::SeqCst),
                0,
                "workers must be joined"
            );
            let peak = encoder.peak.load(Ordering::SeqCst);
            let chunks = ids.len().div_ceil(batch_size.max(64));
            assert!(
                peak > 0 && peak <= slots.min(chunks),
                "worker bound: peak={peak}"
            );
            let batches = encoder.batches.lock().unwrap_or_else(|e| e.into_inner());
            assert_eq!(batches.len(), chunks * 2, "one encode per field per chunk");
            assert_eq!(
                encoder.completed.load(Ordering::SeqCst),
                batches.len(),
                "all encodes completed before return"
            );
            let mut sizes: Vec<usize> = batches.iter().map(Vec::len).collect();
            sizes.sort_unstable();
            let expected_sizes = if batch_size > ids.len() {
                vec![5000, 5000]
            } else {
                let mut sizes = vec![64; 156];
                sizes.extend([8, 8]);
                sizes.sort_unstable();
                sizes
            };
            assert_eq!(
                sizes, expected_sizes,
                "both fields include the final partial batch"
            );

            let mut expected_texts = Vec::new();
            for id in &ids {
                let record = &records[id];
                let mut expected_vector = Vec::new();
                for (field_a, field_b, weight) in &specs {
                    let field = if matches!(side, Side::A) {
                        field_a
                    } else {
                        field_b
                    };
                    let text = record.get(field).map(|s| s.trim()).unwrap_or_default();
                    expected_texts.push(text.to_string());
                    expected_vector
                        .extend(text_vector(text).iter().map(|v| v * weight.sqrt() as f32));
                }
                assert_eq!(
                    db.get(id)?,
                    Some(expected_vector),
                    "weighted field concatenation for {id} on {side:?}"
                );
            }
            let mut actual_texts: Vec<String> = batches.iter().flatten().cloned().collect();
            expected_texts.sort_unstable();
            actual_texts.sort_unstable();
            assert_eq!(
                actual_texts, expected_texts,
                "each selected, trimmed field encoded exactly once (missing fields become empty)"
            );
        }
        Ok(())
    }

    #[test]
    fn encode_and_upsert_empty_input_never_encodes() -> Result<(), MelderError> {
        let encoder = NestedEncoder::new(1, 64, true);
        let db = FlatVectorDB::new(4);
        super::super::encode_and_upsert(
            &db,
            &[],
            &HashMap::new(),
            &specs(),
            &encoder,
            4,
            true,
            Side::A,
            "empty",
        )?;
        assert!(db.is_empty(), "empty input must not insert vectors");
        assert!(
            encoder
                .batches
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .is_empty(),
            "empty input must not call encode"
        );
        Ok(())
    }

    #[test]
    fn encode_and_upsert_encoder_error_cancels_and_joins_workers() {
        let (ids, records) = make_records(5000);
        let id_refs: Vec<&String> = ids.iter().collect();
        for slots in [1, 4] {
            let encoder = NestedEncoder::new(slots, 64, true);
            let db = FlatVectorDB::new(4);
            let result = super::super::encode_and_upsert(
                &db,
                &id_refs,
                &records,
                &specs(),
                &encoder,
                4,
                true,
                Side::A,
                "failure",
            );
            assert!(
                matches!(result, Err(MelderError::Encoder(EncoderError::Inference(ref message))) if message == "intentional nested encoder failure"),
                "original typed encoder error must survive worker joins: {result:?}"
            );
            let calls = encoder
                .batches
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .len();
            assert!(
                calls > 0 && calls <= slots,
                "only initially in-flight failing workers may encode: {calls} calls for {slots} slots"
            );
            assert_eq!(
                encoder.active.load(Ordering::SeqCst),
                0,
                "no encoding may remain active after error return"
            );
            assert_eq!(
                encoder.completed.load(Ordering::SeqCst),
                calls,
                "every started encode must finish before error return"
            );
            assert!(
                db.is_empty(),
                "failed first field must never upsert a partial vector"
            );
        }
    }

    #[test]
    fn encode_and_upsert_encoder_panic_cancels_and_joins_workers() {
        let (ids, records) = make_records(5000);
        let id_refs: Vec<&String> = ids.iter().collect();
        for slots in [1, 4] {
            let mut encoder = NestedEncoder::new(slots, 64, false);
            encoder.panic_on_encode = true;
            let db = FlatVectorDB::new(4);
            let result = super::super::encode_and_upsert(
                &db,
                &id_refs,
                &records,
                &specs(),
                &encoder,
                4,
                true,
                Side::A,
                "panic",
            );
            assert!(
                matches!(result, Err(MelderError::Other(ref error)) if error.to_string() == "encoding worker panicked"),
                "worker panic must become a startup error: {result:?}"
            );
            let calls = encoder.completed.load(Ordering::SeqCst);
            assert!(
                calls > 0 && calls <= slots,
                "only initially in-flight workers may encode after panic: {calls} calls for {slots} slots"
            );
            assert_eq!(
                encoder.active.load(Ordering::SeqCst),
                0,
                "panicked workers must be joined before error return"
            );
            assert!(db.is_empty(), "panic must not upsert a partial vector");
        }
    }

    #[test]
    fn encode_and_upsert_upsert_error_stops_later_batches() {
        let (ids, records) = make_records(5000);
        let id_refs: Vec<&String> = ids.iter().collect();
        let encoder = NestedEncoder::new(1, 64, false);
        // Deliberately wrong index dimension causes the first upsert to fail.
        let db = FlatVectorDB::new(3);
        let result = super::super::encode_and_upsert(
            &db,
            &id_refs,
            &records,
            &specs(),
            &encoder,
            4,
            true,
            Side::A,
            "upsert failure",
        );
        assert!(
            matches!(result, Err(MelderError::Other(_))),
            "upsert error must propagate: {result:?}"
        );
        assert_eq!(
            encoder
                .batches
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .len(),
            2,
            "only the first chunk's two fields may encode before upsert failure"
        );
        assert_eq!(
            encoder.active.load(Ordering::SeqCst),
            0,
            "upsert error must join encoding workers"
        );
        assert_eq!(
            encoder.completed.load(Ordering::SeqCst),
            2,
            "no incomplete encode after upsert error"
        );
        assert!(db.is_empty(), "failed upsert must not insert a vector");
    }
}
