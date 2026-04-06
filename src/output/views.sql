-- Output DB views. Loaded at build time via include_str!().

-- All confirmed 1:1 pairs (scored + asserted).
CREATE VIEW IF NOT EXISTS confirmed_matches AS
SELECT * FROM relationships WHERE relationship_type = 'match';

-- Confirmed pairs that were scored normally (no pre-score shortcut).
CREATE VIEW IF NOT EXISTS scored_matches AS
SELECT * FROM relationships WHERE relationship_type = 'match' AND reason = 'top_scoring';

-- Confirmed pairs from pre-score paths (canonical, exact, crossmap).
CREATE VIEW IF NOT EXISTS asserted_matches AS
SELECT * FROM relationships WHERE relationship_type = 'match' AND reason NOT IN ('top_scoring', 'downgraded');

-- Pairs awaiting human review.
CREATE VIEW IF NOT EXISTS review_queue AS
SELECT * FROM relationships WHERE relationship_type = 'review';

-- Best unscored candidates (rank 1, below review floor). Scoring log only.
CREATE VIEW IF NOT EXISTS near_misses AS
SELECT * FROM relationships WHERE relationship_type = 'candidate' AND rank = 1;

-- Candidates at ranks 2+. Scoring log only.
CREATE VIEW IF NOT EXISTS runner_ups AS
SELECT * FROM relationships WHERE rank > 1;

-- A-side records with no match or review relationship.
CREATE VIEW IF NOT EXISTS unmatched_a AS
SELECT a.* FROM a_records a
WHERE NOT EXISTS (
    SELECT 1 FROM relationships r
    WHERE r.a_id = a.id
    AND r.relationship_type IN ('match', 'review')
);

-- B-side records with no match or review relationship.
CREATE VIEW IF NOT EXISTS unmatched_b AS
SELECT b.* FROM b_records b
WHERE NOT EXISTS (
    SELECT 1 FROM relationships r
    WHERE r.b_id = b.id
    AND r.relationship_type IN ('match', 'review')
);

-- Previously confirmed pairs that were explicitly broken.
CREATE VIEW IF NOT EXISTS broken_matches AS
SELECT * FROM relationships WHERE relationship_type = 'broken';

-- Aggregate counts by type and reason.
CREATE VIEW IF NOT EXISTS summary AS
SELECT
    relationship_type,
    reason,
    COUNT(*) AS count,
    AVG(score) AS avg_score,
    MIN(score) AS min_score,
    MAX(score) AS max_score
FROM relationships
GROUP BY relationship_type, reason;

-- Join of relationships x field_scores for explainability queries.
-- Only populated when scoring log is enabled.
CREATE VIEW IF NOT EXISTS relationship_detail AS
SELECT
    r.a_id, r.b_id, r.score AS composite_score,
    r.relationship_type, r.reason, r.rank,
    fs.field_a, fs.field_b, fs.method,
    fs.score AS field_score, fs.weight
FROM relationships r
LEFT JOIN field_scores fs ON r.a_id = fs.a_id AND r.b_id = fs.b_id;
