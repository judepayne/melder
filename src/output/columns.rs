//! Shared collision-free column naming for CSV and Parquet outputs.

use std::collections::{HashMap, HashSet};

// B is the source/query side; A is the target/reference side. Metadata keeps its
// public names. Qualify only colliding input names, reserving every original
// name first so generated aliases cannot rename otherwise noncolliding fields.
fn names(columns: &[(&str, Option<&str>)]) -> Vec<String> {
    let mut counts = HashMap::new();
    let mut used = HashSet::new();
    for (name, _) in columns {
        *counts.entry(*name).or_insert(0) += 1;
        used.insert(name.to_string());
    }
    columns
        .iter()
        .map(|(name, side)| {
            if side.is_none() || counts[name] == 1 {
                return name.to_string();
            }
            let base = format!("{}_{name}", side.expect("input column has a side"));
            let mut candidate = base.clone();
            let mut suffix = 2;
            while !used.insert(candidate.clone()) {
                candidate = format!("{base}_{suffix}");
                suffix += 1;
            }
            candidate
        })
        .collect()
}

/// Resolve relationship ID, metadata, and target payload column names.
pub(super) fn relationships(a_fields: &[String], a_id: &str, b_id: &str) -> Vec<String> {
    let mut columns = vec![
        (b_id, Some("source")),
        (a_id, Some("target")),
        ("score", None),
        ("relationship_type", None),
        ("reason", None),
        ("rank", None),
    ];
    columns.extend(
        a_fields
            .iter()
            .filter(|f| f.as_str() != a_id)
            .map(|f| (f.as_str(), Some("target"))),
    );
    names(&columns)
}

/// Resolve unmatched source ID, payload, and best-candidate column names.
pub(super) fn unmatched(b_fields: &[String], b_id: &str) -> Vec<String> {
    let mut columns = vec![(b_id, Some("source"))];
    columns.extend(
        b_fields
            .iter()
            .filter(|f| f.as_str() != b_id)
            .map(|f| (f.as_str(), Some("source"))),
    );
    columns.extend([("best_score", None), ("best_a_id", None)]);
    names(&columns)
}

/// Resolve candidate IDs without colliding with rank or score metadata.
pub(super) fn candidates(b_id: &str, a_id: &str) -> Vec<String> {
    names(&[
        (b_id, Some("source")),
        ("rank", None),
        (a_id, Some("target")),
        ("score", None),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_ids_are_qualified_and_distinct_ids_preserved() {
        for id in ["id", "entity_id"] {
            assert_eq!(
                candidates(id, id),
                [
                    format!("source_{id}"),
                    "rank".into(),
                    format!("target_{id}"),
                    "score".into()
                ],
                "shared IDs must identify their sides",
            );
        }
        assert_eq!(
            candidates("b_id", "a_id"),
            ["b_id", "rank", "a_id", "score"]
        );
    }

    #[test]
    fn metadata_and_existing_aliases_are_preserved() {
        let fields = [
            "id",
            "score",
            "target_score",
            "target_score_2",
            "source_id",
            "id_copy",
        ]
        .map(String::from);
        assert_eq!(
            relationships(&fields, "id", "id"),
            [
                "source_id_2",
                "target_id",
                "score",
                "relationship_type",
                "reason",
                "rank",
                "target_score_3",
                "target_score",
                "target_score_2",
                "source_id",
                "id_copy"
            ],
            "aliases must not steal existing payload names",
        );
        assert_eq!(
            unmatched(&["best_score".into(), "best_a_id".into()], "best_score"),
            [
                "source_best_score",
                "source_best_a_id",
                "best_score",
                "best_a_id"
            ],
            "metadata must retain its names even when the ID collides",
        );
        assert_eq!(
            candidates("rank", "score"),
            ["source_rank", "rank", "target_score", "score"]
        );
    }
}
