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

        let loaded = UsearchVectorDB::load(&path).unwrap();

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
            assert!(num % 2 == 0, "US block search returned GB record: {}", r.id);
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
}
